import torch
from torch import nn
from einops import *
from torch.optim import Adam

from sae.base import BaseSAE, Loss


class AnthropicSAE(BaseSAE):
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

        # Contrary to Anthropic, we actually still use a decoder norm because it seems more logical.
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model, device=device))

        self.optimizer = Adam(self.parameters(), lr=self.config.lr, betas=(0.9, 0.999))
        
    def encode(self, x):
        return torch.relu(einsum(x - self.b_dec, self.W_enc, "... d, h d -> ... h") + self.b_enc),
        
    def decode(self, x):
        return einsum(x, self.W_dec, "... h, d h -> ... d") + self.b_dec

    def loss(self, x, x_hid, x_hat):
        reconstruction = (x_hat - x).pow(2).mean(0).sum(dim=-1)

        norm = self.W_dec.norm(dim=-2)
        
        sparsity = einsum(x_hid, norm, "batch h, h -> batch").mean(dim=0)
        return Loss(reconstruction, sparsity, torch.tensor(0, device=x.device))