import torch
from torch import nn
from einops import *
from torch.optim import Adam, SGD

from base import BaseSAE, Loss


class OpenSAE(BaseSAE):
    """
    The main difference between this is the loss function.
    Specifically, it uses the activation * the output norm as the sparsity term.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        device = config.device

        W_dec = torch.randn(self.d_model, self.d_hidden, device=device)
        W_dec /= torch.norm(W_dec, dim=-2, keepdim=True)
        
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(config.identity_scale * W_dec.mT.clone())

        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model, device=device))

        parameters = [
            dict(params=[self.W_enc, self.W_dec, self.b_enc, self.b_dec], lr=self.config.lr),
            dict(params=self.lookup.parameters(), lr=self.config.lookup_lr)
        ]
        
        self.optimizer = Adam(parameters, betas=(0.9, 0.999))

    def encode(self, x):
        x_flat = rearrange(x, "... d -> (...) d")
        
        full = einsum(x_flat - self.b_dec, self.W_enc, "... d, h d -> ... h") + self.b_enc
        _, indices = full.topk(self.config.k, dim=-1)
        rows = torch.arange(full.size(0)).unsqueeze(1)

        mask = torch.zeros_like(full)
        mask[rows, indices] = 1
        
        # This isn't hacky at all
        return (full * mask).view(*x.shape[:-1], full.shape[-1]),

    def decode(self, x):
        return einsum(x, self.W_dec, "... h, d h -> ... d") + self.b_dec

    def loss(self, x, x_hid, x_hat):
        reconstruction = (x_hat - x).pow(2).mean(0).sum(dim=-1)
        return Loss(reconstruction, torch.tensor(0, device=x.device), torch.tensor(0, device=x.device))
