import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class Alignment_Generator(nn.Module):
    """
    G(x_tgt) = x_tgt + Delta
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.0,
        delta_scale: float = 1.0,
    ):
        super().__init__()
        self.delta_scale = delta_scale

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.in_proj(x)
        h = self.blocks(h)
        delta = self.out(h)
        return x + self.delta_scale * delta


class Alignment_Discriminator(nn.Module):
    """
    D_adv(x) -> scalar
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.in_proj(x)
        h = self.blocks(h)
        return self.out(h).squeeze(-1)


class FiLMResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = dim * expansion

        self.norm = nn.LayerNorm(dim)

        self.gamma = nn.Linear(cond_dim, dim)
        self.beta = nn.Linear(cond_dim, dim)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        u = self.norm(h)

        gamma = self.gamma(cond)
        beta = self.beta(cond)

        u = (1.0 + gamma) * u + beta

        z = self.fc1(u)
        z = self.act(z)
        z = self.dropout(z)
        z = self.fc2(z)

        return h + z


class Condition_Generator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_batches: int,
        hidden_dim: int = 512,
        cond_dim: int = 128,
        num_blocks: int = 6,
        expansion: int = 4,
        dropout: float = 0.1,
        delta_scale: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_batches = num_batches
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.delta_scale = delta_scale

        self.domain_embed = nn.Embedding(num_batches, cond_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            FiLMResBlock(
                dim=hidden_dim,
                cond_dim=cond_dim,
                expansion=expansion,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, input_dim)

        # 让模型初始时接近 identity: x_hat ≈ x
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, x: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, input_dim)
        domain: (B,) source batch/domain id
        """
        cond = self.domain_embed(domain)

        h = self.input_proj(x)

        for block in self.blocks:
            h = block(h, cond)

        h = self.out_norm(h)

        delta = self.delta_scale * self.delta_head(h)
        x_hat = x + delta

        return x_hat
    
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


class Discriminator_twohead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_batches: int,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        expansion: int = 2,
        dropout: float = 0.0,
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

    def forward(self, x: torch.Tensor):
        """
        x: (B, L)
        Returns:
            adv_logits:   (B, 1)
            batch_logits: (B, K_batch)
        """
        h = self.input_proj(x)

        for block in self.blocks:
            h = block(h)

        h = self.out_norm(h)

        adv_logits = self.adv_head(h)
        batch_logits = self.batch_head(h)

        return {
            "adv_logits": adv_logits,
            "batch_logits": batch_logits,
        }
    