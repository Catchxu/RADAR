import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from .layer import Encoder


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=[1024, 512, 256], latent_dim=256, **kwargs):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, **kwargs)
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
        )
        # Additional initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return self.critic(self.encoder(x))  # shape: (batch_size, 1)


class DiscriminatorResBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden_dim = dim * expansion

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.act = nn.LeakyReLU(0.2, inplace=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, dim)
        """
        z = self.norm(h)
        z = self.fc1(z)
        z = self.act(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return h + z


class Discriminator_phase2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_batches: int,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        expansion: int = 2,
        dropout: float = 0.0,
        num_states: int = 2,   # normal / disease
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            DiscriminatorResBlock(
                dim=hidden_dim,
                expansion=expansion,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)

        # three heads
        self.adv_head = nn.Linear(hidden_dim, 1)
        self.batch_head = nn.Linear(hidden_dim, num_batches)
        self.state_head = nn.Linear(hidden_dim, num_states)

    def forward(self, x: torch.Tensor):
        """
        x: (B, L)
        Returns:
            adv_logits:   (B, 1)
            batch_logits: (B, K_batch)
            state_logits: (B, K_state)
        """
        h = self.input_proj(x)

        for block in self.blocks:
            h = block(h)

        h = self.out_norm(h)

        adv_logits = self.adv_head(h)
        batch_logits = self.batch_head(h)
        state_logits = self.state_head(h)

        return adv_logits, batch_logits, state_logits