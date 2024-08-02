from torch import nn
import torch
from transformer_lens import utils
from dataclasses import dataclass
from einops import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import wandb
from tqdm import tqdm
from collections import namedtuple
from utils import create_unigram_lookup_table
from torch.optim import Adam


Loss = namedtuple('Loss', ['reconstruction', 'sparsity', 'auxiliary'])

@dataclass
class Config:
    buffer_size: int = 2**18  # ~250k tokens
    n_buffers: int = 100      # ~25M tokens
    
    # transformer_lens specific
    point: str = "resid_mid"
    layer: int = 0

    in_batch: int = 32
    out_batch: int = 4096
    n_ctx: int = 256

    expansion: int = 4
    lr: float = 1e-4
    lookup_lr: float = 0.01

    d_in: int = 512                      # For sae_vis
    d_hidden: int = 2048                 # For sae_vis
    apply_b_dec_to_input: bool = True    # For sae_vis

    validation_interval: int = 500
    not_active_thresh: int = 2

    sparsity: float = 1.0
    device: str = "cuda"
    lookup: str | None = 'learned'
    lookup_scale: float = 0.5
    
    identity_scale: float = 0.5
    magnitude_relu: bool = True
    k: int = 15
    
    @property
    def hook_pt(self):
        return utils.get_act_name(self.point, self.layer)

class Lookup(nn.Module):
    def __init__(self, config, model) -> None:
        super().__init__()
        self.lookup = config.lookup
        
        lookup_grad = config.lookup == 'learned'
        scale_grad = config.lookup == 'scaled'
        
        self.W_original = create_unigram_lookup_table(model, config).detach()
        
        if self.lookup is not None:
            # self.W_lookup = nn.Parameter(config.lookup_scale*self.W_original.clone(), requires_grad=lookup_grad)
            
            self.W_lookup = nn.Embedding.from_pretrained(config.lookup_scale*self.W_original.clone(), freeze=not lookup_grad)
            #self.W_lookup = nn.Embedding(self.W_original.size(0), self.W_original.size(1))
            # self.W_lookup = nn.Parameter(torch.zeros(model.tokenizer.vocab_size, model.cfg.d_model, device=config.device), requires_grad=lookup_grad)
            # self.W_scale = nn.Parameter(torch.ones(self.W_lookup.size(0), device=config.device), requires_grad=scale_grad)
    
    def forward(self, x, y):
        if self.lookup is None:
            return x
        else:
            return x + self.W_lookup(y.cuda())

class BaseSAE(nn.Module):
    """
    Base class for all Sparse Auto Encoders.
    Provides a common interface for training and evaluation.
    """
    def __init__(self, config, model) -> None:
        super().__init__()
        self.config = config
        self.cfg = config     # For sae_vis
        
        self.d_model = model.cfg.d_model
        self.n_ctx = config.n_ctx
        self.d_hidden = self.config.expansion * self.d_model
        
        self.sparsity = config.sparsity
        self.steps_not_active = torch.zeros(self.d_hidden)
        self.step = 0
        
        self.lookup = Lookup(config, model)
    
    def decode(self, x):
        return x
    
    def encode(self, x):
        return torch.zeros_like(x),
    
    def preprocess(self, x):
        return x
    
    def postprocess(self, x):
        return x
    
    def forward(self, x, y=None):
        x_hid, *_ = self.encode(x)
        return self.lookup(self.decode(x_hid), y)
    
    def loss(self, x, x_hid, x_hat, *args):
        reconstruction = (x_hat - x).pow(2).mean(0).sum(dim=-1)
        return Loss(reconstruction, torch.tensor(0, device=x.device), torch.tensor(0, device=x.device))
    
    @classmethod
    def from_pretrained(cls, path, *args, device='cuda', **kwargs):
        state = torch.load(path, map_location=torch.device(device))
        new = cls(*args, **kwargs)
        new.load_state_dict(state)
        
        if 'lookup.W_lookup.weight' in state:   # hack for now (load_state_dict doesn't overwrite existing?)
            if 'lookup.W_lookup.weight' in new.__dict__:
                new.lookup.W_lookup.weight = nn.Parameter(state['lookup.W_lookup.weight'])
        
        return new
    
    def calculate_metrics(self, x, x_hid, x_hat, losses, *args):
        self.steps_not_active[x_hid.sum(0) > 0] = 0
        
        metrics = dict(step=self.step)
        
        metrics["dead_fraction"] = (self.steps_not_active > 2).float().mean().item()
            
        metrics["reconstruction_loss"] = losses.reconstruction.item()
        metrics["sparsity_loss"] = losses.sparsity.item()
        metrics["auxiliary_loss"] = losses.auxiliary.item()
        
        metrics["nmse"] = ((x_hat - x).pow(2).sum(-1) / (x.pow(2).sum(-1) + 1e-4)).mean(0).item()
            
        metrics["l1"] = x_hid.sum(-1).mean().item()
        metrics["l0"] = (x_hid > 0).float().sum(-1).mean().item()
        
        if hasattr(self.lookup, "W_lookup"):
            sims = einsum(self.lookup.W_lookup.weight.detach(), self.lookup.W_original.detach(), "t d, t d -> t") / (self.lookup.W_original.norm(dim=-1)**2)
            metrics["lookup_sim"] = sims.mean().item()
            metrics["lookup_max"] = sims.max().item()
        
        self.steps_not_active += 1
        return metrics
    
    def train(self, sampler, model, validation, log=True, name=None):
        if log: wandb.init(project="sae", name=name)
        
        self.step = 0
        self.steps = self.config.n_buffers * (self.config.buffer_size // self.config.out_batch)

        # scheduler = LambdaLR(self.optimizer, lr_lambda=lambda t: min(5*(1 - t/self.steps), 1.0))
        scheduler = CosineAnnealingLR(self.optimizer, self.steps, eta_min=3e-5)

        for buffer, _ in tqdm(zip(sampler, range(self.config.n_buffers)), total=self.config.n_buffers):
            loader = DataLoader(buffer, batch_size=self.config.out_batch, shuffle=True, drop_last=True)
            for x, y in loader:
                x = self.preprocess(x)
                
                x_hid, *rest = self.encode(x)
                x_hat = self.decode(x_hid)
                x_hat = self.lookup(x_hat, y)
                
                x = self.postprocess(x)
                
                losses = self.loss(x, x_hid, x_hat, *rest)
                metrics = self.calculate_metrics(x, x_hid, x_hat, losses, *rest)

                loss = (losses.reconstruction + self.sparsity * losses.sparsity + losses.auxiliary).sum()

                self.optimizer.zero_grad()
                loss.backward()
                
                # Dirty hack to increase update of lookup table
                # Due to the structure of the lookup table, we actually need to sum, not mean over the batch size
                # This is a remedy for that
                # self.lookup.W_lookup.grad = self.config.out_batch * self.lookup.W_lookup.grad
                
                self.optimizer.step()
                scheduler.step()

                if (self.step % self.config.validation_interval == 0):
                    clean, corrupt, loss = self.patch_loss(model, validation)
                    metrics["ce_added"] = (loss - clean) / (clean + 0.001)
                    metrics["ce_recovered"] = 1 - ((loss - clean) / (corrupt - clean))
                    metrics["ce_clean"] = clean
                    metrics["ce_corrupt"] = corrupt
                    metrics["ce_patched"] = loss
                    metrics["lr"] = scheduler.get_last_lr()[0]

                if log: wandb.log(metrics)
                self.step += 1
        
        if log: wandb.finish()
    
    @torch.inference_mode()
    def patch_loss(self, model, validation):
        hook_pt = utils.get_act_name(self.config.point, self.config.layer)

        # validation = validation[..., :self.config.n_ctx]
        baseline, cache = model.run_with_cache(validation, return_type="loss", names_filter=[hook_pt])
        
        x = self.preprocess(cache[hook_pt])
        x_hat = self.forward(x, validation)

        # run model with recons patched in per instance
        patch_hook = lambda act, hook: x_hat
        loss = model.run_with_hooks(validation, return_type="loss", fwd_hooks = [(hook_pt, patch_hook)])
        
        zero_hook = lambda act, hook: torch.zeros_like(act)
        corrupt = model.run_with_hooks(validation, return_type="loss", fwd_hooks = [(hook_pt, zero_hook)])

        return baseline.item(), corrupt.item(), loss.item()