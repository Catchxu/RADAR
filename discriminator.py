import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Discriminator(nn.Module):
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

        # Shared trunk -> two heads
        self.adv_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, num_batches)

    def forward(self, x: torch.Tensor):
        """
        x: (B, L)
        Returns:
            adv_logits: (B, 1)
            cls_logits: (B, K)
        """
        h = self.input_proj(x)

        for block in self.blocks:
            h = block(h)

        h = self.out_norm(h)

        adv_logits = self.adv_head(h)
        cls_logits = self.cls_head(h)
        return adv_logits, cls_logits