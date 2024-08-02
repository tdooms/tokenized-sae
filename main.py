# %%
%load_ext autoreload
%autoreload 2

from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from einops import *
import torch

from base import Config
from utils import *
from vanilla import VanillaSAE
from gated import GatedSAE
from rainbow import RainbowSAE
from anthropic import AnthropicSAE
from open import OpenSAE
from base import BaseSAE

# %%
model = HookedTransformer.from_pretrained("gpt2").cuda()
train, validation = get_untokenized_splits(model.tokenizer, n_validation=16)
# %%

config = Config(
    n_buffers=3_000,
    expansion=16,
    buffer_size=2**17,
    sparsity=16.0,
    lookup='learned',
    layer=8,
    lr=1e-4,
    k=20,
    lookup_scale=0.0,
    identity_scale=1.0,
    n_ctx=256,
    point="resid_pre",
    device="cuda",
)

sampler = Sampler(config, train, model)
sae = OpenSAE(config, model)

torch.backends.cudnn.benchmark = True
sae.train(sampler, model, validation, log=True)

# %%
sims = einsum(sae.lookup.W_lookup.detach(), sae.lookup.W_original.detach(), "t d, t d -> t") / sae.lookup.W_original.norm(dim=-1)**2
print(sims.max().item())
import plotly.express as px
px.histogram(sims.cpu().detach())
# %%
# loop over init scales
model = HookedTransformer.from_pretrained("gpt2").cuda()
    
for k in [5]:
    train, validation = get_untokenized_splits(model.tokenizer, n_validation=16)

    config = Config(
        n_buffers=3_000,
        expansion=16,
        buffer_size=2**17,
        lookup='learned',
        layer=8,
        lr=1e-4,
        k=k,
        lookup_scale=0.5,
        identity_scale=0.5,
        lookup_lr=0.01, 
        n_ctx=256,
        point="resid_pre",
        device="cuda",
    )

    sampler = Sampler(config, train, model)
    sae = OpenSAE(config, model)

    torch.backends.cudnn.benchmark = True
    sae.train(sampler, model, validation, log=True)
    
    torch.save(sae.state_dict(), f"saes/k_sweep2/gpt2_resid_pre_8_ot{k}.pt")
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
# %%
# loop over sparsities again
model = HookedTransformer.from_pretrained("gpt2").cuda()
    
for k in [5]:
    train, validation = get_untokenized_splits(model.tokenizer, n_validation=16)

    config = Config(
        n_buffers=3_000,
        expansion=16,
        buffer_size=2**17,
        # lookup='learned',
        lookup=None,
        layer=8,
        lr=1e-4,
        k=k,
        # lookup_scale=0.5,
        identity_scale=1.0,
        n_ctx=256,
        point="resid_pre",
        device="cuda",
    )

    sampler = Sampler(config, train, model)
    sae = OpenSAE(config, model)

    torch.backends.cudnn.benchmark = True
    sae.train(sampler, model, validation, log=True)
    
    torch.save(sae.state_dict(), f"saes/k_sweep2/gpt2_resid_pre_8_o{k}.pt")
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
# %%

# loop over layers
model = HookedTransformer.from_pretrained("gpt2").cuda()
    
for lr in [0.01, 0.005, 0.001]:
    train, validation = get_untokenized_splits(model.tokenizer, n_validation=16)

    config = Config(
        n_buffers=2_000,
        expansion=16,
        buffer_size=2**16,
        # sparsity=16.0,
        lookup='learned',
        # lookup=None,
        layer=8,
        lr=1e-4,
        lookup_lr=lr,
        k=30,
        lookup_scale=0.5,
        identity_scale=0.5,
        n_ctx=256,
        point="resid_pre",
        device="cuda",
    )

    sampler = Sampler(config, train, model)
    sae = OpenSAE(config, model)

    torch.backends.cudnn.benchmark = True
    sae.train(sampler, model, validation, log=True)
    
    torch.save(sae.state_dict(), f"saes/lookup_lr_sweep/gpt2_resid_pre_8_o30_lr{lr}.pt")
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
# %%

model = HookedTransformer.from_pretrained("pythia-1.4b").cuda()
train, validation = get_untokenized_splits(model.tokenizer, n_validation=16)
# %%

config = Config(
    n_buffers=2_000,
    expansion=8,
    buffer_size=2**16,
    lookup='learned',
    # lookup=None,
    layer=16,
    lr=1e-4,
    k=50,
    lookup_scale=0.5,
    identity_scale=0.5,
    # identity_scale=1.0,
    n_ctx=256,
    point="resid_pre",
    device="cuda",
)

sampler = Sampler(config, train, model)
sae = OpenSAE(config, model)

torch.backends.cudnn.benchmark = True
sae.train(sampler, model, validation, log=True)
# %%
torch.save(sae.state_dict(), "saes/pythia_resid_pre_16_aot50.pt")

import gc
gc.collect()
torch.cuda.empty_cache()
# %%
for k in [50]:
    train, validation = get_untokenized_splits(model.tokenizer, n_validation=16)

    config = Config(
        n_buffers=2_000,
        expansion=8,
        buffer_size=2**16,
        lookup='learned',
        layer=16,
        lr=1e-4,
        k=k,
        lookup_scale=0.5,
        identity_scale=0.5,
        n_ctx=256,
        point="resid_pre",
        device="cuda",
    )

    sampler = Sampler(config, train, model)
    sae = OpenSAE(config, model)

    torch.backends.cudnn.benchmark = True
    sae.train(sampler, model, validation, log=True)
    torch.save(sae.state_dict(), f"saes/pythia_resid_pre_16_ot{k}_long.pt")
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
# %%
for k in [50, 30]:
    train, validation = get_untokenized_splits(model.tokenizer, n_validation=16)

    config = Config(
        n_buffers=2_000,
        expansion=8,
        buffer_size=2**16,
        lookup=None,
        layer=16,
        lr=1e-4,
        k=k,
        identity_scale=1.0,
        n_ctx=256,
        point="resid_pre",
        device="cuda",
    )

    sampler = Sampler(config, train, model)
    sae = OpenSAE(config, model)

    torch.backends.cudnn.benchmark = True
    sae.train(sampler, model, validation, log=True)
    torch.save(sae.state_dict(), f"saes/pythia_resid_pre_16_o{k}_long.pt")
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
# %%

model = HookedTransformer.from_pretrained("gpt2").cuda()
    
for i, j in zip([16, 17, 18], [1_000, 500, 250]):
    train, validation = get_untokenized_splits(model.tokenizer, n_validation=16)

    config = Config(
        n_buffers=j//2,
        expansion=16,
        buffer_size=2**i,
        # sparsity=16.0,
        lookup='learned',
        layer=8,
        lr=1e-4,
        k=30,
        lookup_scale=0.5,
        identity_scale=0.5,
        n_ctx=256,
        point="resid_pre",
        device="cuda",
    )

    sampler = Sampler(config, train, model)
    sae = OpenSAE(config, model)

    torch.backends.cudnn.benchmark = True
    sae.train(sampler, model, validation, log=True)
    
    torch.save(sae.state_dict(), f"saes/buf_sweep/gpt2_resid_pre_8_ot30_e{i}.pt")
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
# %%

model = HookedTransformer.from_pretrained("pythia-1.4b").cuda()

for i in [0.0, 0.1, 0.2, 0.3, 0.4]:
    train, validation = get_untokenized_splits(model.tokenizer, n_validation=16)

    config = Config(
        n_buffers=1_000,
        expansion=8,
        buffer_size=2**16,
        lookup='learned',
        layer=16,
        lr=1e-4,
        k=50,
        lookup_scale=i,
        identity_scale=1-i,
        n_ctx=256,
        point="resid_pre",
        device="cuda",
    )

    sampler = Sampler(config, train, model)
    sae = OpenSAE(config, model)

    torch.backends.cudnn.benchmark = True
    sae.train(sampler, model, validation, log=True)
    torch.save(sae.state_dict(), f"saes/init_pythia_sweep/pythia_resid_pre_16_ot30_i{i}.pt")
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()