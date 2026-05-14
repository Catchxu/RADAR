import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from .layer import Encoder, Decoder, MemoryBlock, StyleBlock


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=[1024, 512, 256], latent_dim=256, **kwargs,):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim,  **kwargs)
        self.decoder = Decoder(input_dim, hidden_dim,  **kwargs)
        self.latent_dim = latent_dim
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


class GeneratorWithMemory(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        memory_size: int = 512,
        num_heads: int = 8,
        temperature: float = 1.0,
        use_memory_bank: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.use_memory_bank = use_memory_bank

        self.extractor = AutoEncoder(input_dim, hidden_dim, latent_dim, **kwargs)

        if self.use_memory_bank:
            self.memory = MemoryBlock(latent_dim, memory_size, num_heads, temperature)
        else:
            self.memory = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def encode(self, x):
        return self.extractor.encoder(x)  # Z

    def decode(self, z):
        return self.extractor.decoder(z)  # Xhat

    def forward(self, x, update_mem: bool = True):
        z = self.encode(x)  # Z = EI(X)

        if self.use_memory_bank:
            z_mem = self.memory(z, update_mem=update_mem)  # Zmem
        else:
            z_mem = z 

        x_hat = self.decode(z_mem)  # Xhat
        return x_hat, z, z_mem


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


class Generator_phase2(nn.Module):
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
    
class GeneratorWithStyle(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_batches, **kwargs,):
        super().__init__()
        self.extractor = AutoEncoder(input_dim, hidden_dim, latent_dim, **kwargs)
        self.style = StyleBlock(num_batches, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def init_from_phase1(self, phase1_G: GeneratorWithMemory, strict: bool = True):
        # Init EIII/GIII from EI/GI
        self.extractor.encoder.load_state_dict(phase1_G.extractor.encoder.state_dict(), strict=strict)
        self.extractor.decoder.load_state_dict(phase1_G.extractor.decoder.state_dict(), strict=strict)

    def forward(self, x_t, batch_onehot):
        # batch_onehot: [B, Nb] (paper Bt)
        z_t = self.extractor.encoder(x_t)          # Zt
        z_tilde = self.style(z_t, batch_onehot)    # Ztilde = Zt - Bt S
        x_hat_r = self.extractor.decoder(z_tilde)  # Xhat_r
        return x_hat_r, z_t, z_tilde