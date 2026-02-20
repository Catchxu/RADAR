import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from .layer import Encoder


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=[1024, 512, 256], latent_dim=256, **kwargs):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, **kwargs)
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 16),
        )
        # Additional initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return self.critic(self.encoder(x))  # shape: (batch_size, 16)
