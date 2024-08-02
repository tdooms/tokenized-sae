import torch
from torch import nn
from einops import *
from torch.optim import Adam
from base import BaseSAE, Loss

class RainbowSAE(BaseSAE):
    """
    This is a combination of gated SAE with the Anthropic loss.
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        device = config.device

        W_dec = torch.randn(self.d_model, self.d_hidden, device=device)
        W_dec /= torch.norm(W_dec, dim=-2, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)

        if config.identity_init:
            self.W_gate = nn.Parameter(W_dec.mT.clone())
        else:
            W_gate = torch.empty(self.d_hidden, self.d_model, device=device)
            torch.nn.init.xavier_normal_(W_gate)
            self.W_gate = nn.Parameter(W_gate)
        
        self.r_mag = nn.Parameter(torch.zeros(self.d_hidden, device=device))

        self.b_gate = nn.Parameter(torch.zeros(self.d_hidden, device=device))
        self.b_mag = nn.Parameter(torch.zeros(self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model, device=device))

        self.optimizer = Adam(self.parameters(), lr=config.lr, betas=(0.9, 0.999))
        self.relu = nn.ReLU() if config.magnitude_relu else nn.Identity()

    def encode(self, x):
        preact = einsum(x - self.b_dec, self.W_gate, "... d, h d -> ... h")
        magnitude = preact * torch.exp(self.r_mag) + self.b_mag
        
        hidden_act = self.relu(magnitude) * (preact + self.b_gate > 0).float()
        gated_act = torch.relu(preact + self.b_gate)

        return hidden_act, gated_act

    def decode(self, x):
        return einsum(x, self.W_dec, "... h, d h -> ... d") + self.b_dec

    def loss(self, x, _, x_hat, gated_act):
        recons_losses = (x - x_hat).pow(2).mean(dim=0).sum(dim=-1)

        W_dec_clone = self.W_dec.detach()
        b_dec_clone = self.b_dec.detach()
        
        norm = W_dec_clone.norm(dim=-2)
        lambda_ = min(1, self.step/self.steps * 20)
        sparsity_losses = lambda_ * einsum(gated_act, norm, "batch h, h -> batch").mean(dim=-1)

        gated_recons = einsum(gated_act, W_dec_clone, "batch h, d h -> batch d") + b_dec_clone
        aux_losses = (x - gated_recons).pow(2).mean(dim=0).sum(dim=-1)

        return Loss(recons_losses, sparsity_losses, aux_losses)
    
    def calculate_metrics(self, x, x_hid, x_hat, losses, *args):
        metrics = super().calculate_metrics(x, x_hid, x_hat, losses, *args)
        metrics["auxiliary_loss"] = losses.auxiliary.item()
        return metrics