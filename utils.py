from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from transformer_lens import utils
import gc


# ----------------

from transformers import AutoTokenizer
from typing import Dict, List
import einops
import numpy as np

def keep_single_column(dataset, col_name: str):
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful when we want to tokenize and mix together different strings
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset

def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
):
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with padding, then remove padding at the end.

    This tokenization is useful for training language models, as it allows us to efficiently train on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding). Further, for models with absolute positional encodings, this avoids privileging early tokens (eg, news articles often begin with CNN, and models may learn to use early positional encodings to predict these)

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"

    Note: There is a bug when inputting very small datasets (eg, <1 batch per process) where it just outputs nothing. I'm not super sure why
    """
    dataset = keep_single_column(dataset, column_name)
    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)
        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)
        num_batches = num_tokens // (seq_len)
        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        # num_proc=(num_proc if not streaming else None),
        remove_columns=[column_name],
    )
    # tokenized_dataset.set_format(type="torch", columns=["tokens"])
    return tokenized_dataset


# -----

class TokenDataset(Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data = data_tensor
        self.labels = label_tensor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def get_splits(n_validation=32):
    training = load_dataset("c4", split="train", streaming=True).with_format("torch")

    validation = list(training.take(n_validation))
    validation = torch.stack([row["tokens"] for row in validation])
    
    return training, validation

def get_untokenized_splits(tokenizer, n_validation=32):
    training = load_dataset("c4", 'en', split="train", streaming=True).with_format("torch")
    new_training = tokenize_and_concatenate(training, tokenizer, streaming=True, max_length=256)
    
    validation = list(new_training.take(n_validation))
    validation = torch.stack([row["tokens"] for row in validation])
    
    return new_training, validation


class ConstrainedAdam(torch.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)
    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=-1, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=-1, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=-1, keepdim=True)


# @torch.inference_mode()
@torch.no_grad()
def create_unigram_lookup_table(model, config):
    hook_pt = config.hook_pt
    vocab_size = len(model.tokenizer.vocab) # obviously this isn't he same as tokenizer.vocab_size, duh
    batch_size = 1024
    
    tokens = torch.arange(vocab_size, device=config.device, dtype=torch.long)
    bosses = torch.ones_like(tokens) * model.tokenizer.bos_token_id
    
    full = torch.stack([bosses, tokens], dim=-1)
    result = torch.empty(vocab_size, model.cfg.d_model, device=config.device)
    
    for batch in full.split(batch_size, dim=0):
        _, cache = model.run_with_cache(batch, names_filter=[hook_pt], stop_at_layer=config.layer + 1)
        result[batch[:, 1]] = cache[hook_pt][:, -1]
    
    return result.detach()


@torch.inference_mode()
def create_unigram_lookup_table_pos(model, config, bos_pos=0, unigram_pos=1, prepend_bos=True, upto=None):
    hook_pt = config.hook_pt
    vocab_size = model.tokenizer.vocab_size
    batch_size = 1024
    bos_embed = model.W_E[model.tokenizer.bos_token_id]
    
    upto = vocab_size if upto is None else min(upto, vocab_size)
    embeds = model.W_E[:upto] + (0.0 if unigram_pos is None else model.W_pos[unigram_pos])
    
    if prepend_bos is True:
        bosses = bos_embed + (0.0 if bos_pos is None else model.W_pos[bos_pos])
        full = torch.stack([torch.tile(bosses[None], (embeds.size(0),1)), embeds], dim=1)
    else:
        full = embeds[:,None]
    
    result = torch.empty(upto, model.cfg.d_model, device=config.device)
    
    # Assumes vocab in original order
    for batch_num,batch in enumerate(full.split(batch_size, dim=0)):
        _, cache = model.run_with_cache(batch, names_filter=[hook_pt],
                                        start_at_layer=0, stop_at_layer=config.layer + 1)
        result[batch_num*batch_size:(batch_num+1)*batch_size] = cache[hook_pt][:, -1]
    
    return result.detach()


class Sampler:
    """
    A class for sampling activations from a model at a certain point in the network.
    It stores the activations in a large buffer and returns them in a single tensor.
    """
    def __init__(self, config, dataset, model):
        self.config = config
        self.model = model
        
        self.d_model = model.cfg.d_model
        self.n_ctx = config.n_ctx

        assert config.buffer_size % (config.in_batch * self.n_ctx) == 0, "samples must be a multiple of loader batch size"
        self.n_inputs = config.buffer_size // (config.in_batch * self.n_ctx)

        self.loader = DataLoader(dataset, batch_size=config.in_batch)
        self.batches = []

    def collect(self):
        acts, toks = zip(*self.batches)
        acts_res = rearrange(torch.cat(acts, dim=0), "... d_model -> (...) d_model")
        toks_res = rearrange(torch.cat(toks, dim=0), "... -> (...)")
        
        self.batches = []
        return TokenDataset(acts_res, toks_res)

    @torch.inference_mode()
    def extract(self, batch):
        hook_pt = self.config.hook_pt
        _, cache = self.model.run_with_cache(batch, names_filter=[hook_pt], stop_at_layer=self.config.layer + 1)
        return cache[hook_pt]

    def __iter__(self):
        self.batches = []

        for batch in self.loader:
            # tokens = batch["tokens"][..., :self.n_ctx]
            tokens = batch["tokens"]
            acts = (self.extract(tokens), tokens)
            self.batches.append(acts)
            
            if len(self.batches) == self.n_inputs:
                yield self.collect()
                
                # there is some memory leak it seems, no time to debug
                gc.collect()
                torch.cuda.empty_cache()