import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

def kron(a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-1:]) * torch.tensor(b.shape[-1:]))
    res = a.unsqueeze(-1) * b.unsqueeze(-2)
    siz0 = res.shape[:-2]
    return res.reshape(siz0 + siz1)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data) 
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Encoder(nn.Module):
    def __init__(self, obs_dim, action_dim, device, state_dim=3, tau=1, softmax="vanilla"):
        super().__init__()

        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.tau = tau
        self.softmax = softmax
        self.encoder = nn.Linear(obs_dim, state_dim, bias=False)
        self.apply(weight_init)


    def forward(self, obs):
        obs= obs.clone().detach().requires_grad_(True).to(self.device)
        #obs = torch.tensor(obs, device=self.device)
        state_encoding = self.encoder(obs)
        if self.softmax == "gumble":
            state_encoding = F.gumbel_softmax(state_encoding, tau=self.tau, hard=False)
        elif self.softmax == 'vanilla': 
            state_encoding = F.softmax(state_encoding / self.tau, dim=-1)
        
        return state_encoding


    def copy_encoder(self, target):
        self.encoder.load_state_dict(target.encoder.state_dict())


    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)


    def load_encoder(self, path):
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict)
        












