import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = dim * expansion

        self.norm = nn.LayerNorm(dim)

        # FiLM parameters generated from condition embedding
        self.gamma = nn.Linear(cond_dim, dim)
        self.beta = nn.Linear(cond_dim, dim)

        # Feed-forward network
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        h:    (B, dim)
        cond: (B, cond_dim)
        """
        u = self.norm(h)

        gamma = self.gamma(cond)  # (B, dim)
        beta = self.beta(cond)    # (B, dim)

        # FiLM modulation
        u = gamma * u + beta

        z = self.fc1(u)
        z = self.act(z)
        z = self.dropout(z)
        z = self.fc2(z)

        return h + z


class Generator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_batches: int,
        hidden_dim: int = 512,
        cond_dim: int = 128,
        num_blocks: int = 6,
        expansion: int = 4,
        dropout: float = 0.1,
        use_change_gate: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_batches = num_batches
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.use_change_gate = use_change_gate

        # Domain embedding
        self.domain_embed = nn.Embedding(num_batches, cond_dim)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual FiLM blocks
        self.blocks = nn.ModuleList([
            FiLMResBlock(
                dim=hidden_dim,
                cond_dim=cond_dim,
                expansion=expansion,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        # Final normalization before output head
        self.out_norm = nn.LayerNorm(hidden_dim)

        # Residual delta prediction
        self.delta_head = nn.Linear(hidden_dim, input_dim)

        # Optional gate for sparse / controlled editing
        if self.use_change_gate:
            self.gate_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, L)
        domain: (B,)
        return: (B, L)
        """
        cond = self.domain_embed(domain)   # (B, cond_dim)

        h = self.input_proj(x)             # (B, hidden_dim)

        for block in self.blocks:
            h = block(h, cond)

        h = self.out_norm(h)

        delta = self.delta_head(h)         # (B, L)

        if self.use_change_gate:
            gate = torch.sigmoid(self.gate_head(h))   # (B, L)
            x_hat = x + gate * delta
        else:
            x_hat = x + delta

        return x_hat