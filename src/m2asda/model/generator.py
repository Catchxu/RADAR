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
    def __init__(self, input_dim, hidden_dim, latent_dim,
                 memory_size: int = 512,
                 num_heads: int = 8,
                 temperature: float = 1.0,
                 **kwargs):
        super().__init__()
        self.extractor = AutoEncoder(input_dim, hidden_dim, latent_dim, **kwargs)

        # MemoryBlock(latent_dim, memory_size, num_heads, temperature)
        self.memory = MemoryBlock(latent_dim, memory_size, num_heads, temperature)

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
        z = self.encode(x)                             # Z = EI(X)
        z_mem = self.memory(z, update_mem=update_mem)  # Zmem
        x_hat = self.decode(z_mem)                     # Xhat = GI(Zmem)
        return x_hat, z, z_mem


class GeneratorWithPairs(nn.Module):
    def __init__(self, n_ref: int, n_tgt: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_tgt, n_ref))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, Zr: torch.Tensor):
        # Zr: [n_ref, latent_dim]
        P = F.relu(self.W)      # [n_tgt, n_ref]
        Zhat_t = P @ Zr         # [n_tgt, latent_dim]
        return Zhat_t


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
