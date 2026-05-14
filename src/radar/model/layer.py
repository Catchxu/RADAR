import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        normalization: bool = True,
        activation: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]

        if normalization:
            layers.append(nn.BatchNorm1d(out_dim))
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        normalization: bool = True,
        activation: bool = True,
        dropout: float = 0.1,
        last_linear: bool = True,
    ):
        super().__init__()
        assert len(dims) >= 2, f"dims must have at least 2 elements, got {dims}"

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            is_last = i == len(dims) - 2

            if is_last and last_linear:
                layers.append(nn.Linear(in_dim, out_dim))
            else:
                layers.append(
                    LinearBlock(
                        in_dim,
                        out_dim,
                        normalization=normalization,
                        activation=activation,
                        dropout=dropout,
                    )
                )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        normalization: bool = True,
        activation: bool = True,
        dropout: float = 0.1,
        last_linear: bool = True,
    ):
        super().__init__()

        dims = [input_dim] + list(hidden_dims)
        self.mlp = MLP(
            dims=dims,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
            last_linear=last_linear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        normalization: bool = True,
        activation: bool = True,
        dropout: float = 0.1,
        last_linear: bool = True,
    ):
        super().__init__()
        enc_dims = [input_dim] + list(hidden_dims)
        dec_dims = list(reversed(enc_dims))
        self.mlp = MLP(
            dims=dec_dims,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
            last_linear=last_linear,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class MemoryBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        memory_size: int = 512,
        num_heads: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert latent_dim % num_heads == 0

        self.mem_dim = memory_size
        self.z_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.temperature = float(temperature)

        # Trainable W^Q, W^K, W^V (p -> p)
        self.Wq = nn.Linear(latent_dim, latent_dim, bias=False)
        self.Wk = nn.Linear(latent_dim, latent_dim, bias=False)
        self.Wv = nn.Linear(latent_dim, latent_dim, bias=False)

        # FIFO memory queue (no gradients)
        self.register_buffer("mem", torch.empty(memory_size, latent_dim))
        self.register_buffer("mem_ptr", torch.zeros(1, dtype=torch.long))
        self._init_mem()

    def _init_mem(self):
        stdv = 1.0 / math.sqrt(self.mem.size(1))
        self.mem.uniform_(-stdv, stdv)

    @torch.no_grad()
    def enqueue(self, z: torch.Tensor):
        z = z.detach()
        B = z.size(0)
        ptr = int(self.mem_ptr.item())

      
        if B >= self.mem_dim:
            self.mem.copy_(z[-self.mem_dim:])
            self.mem_ptr[0] = 0
            return

        end = ptr + B


        if end <= self.mem_dim:
            self.mem[ptr:end].copy_(z)

        else:
            first = self.mem_dim - ptr
            self.mem[ptr:].copy_(z[:first])

            rem = B - first
            if rem > 0:
                self.mem[:rem].copy_(z[first:first + rem])

        self.mem_ptr[0] = (ptr + B) % self.mem_dim

    def forward(self, z: torch.Tensor, update_mem: bool = True) -> torch.Tensor:
        if update_mem:
            self.enqueue(z)

        # Q=ZWq, K=MWk, V=MWv
        Q = self.Wq(z)         # [B,p]
        K = self.Wk(self.mem)  # [M,p]
        V = self.Wv(self.mem)  # [M,p]

        # Split heads
        B = Q.size(0)
        Qh = Q.view(B, self.num_heads, self.head_dim)  # [B,h,d]
        Kh = K.view(self.mem_dim, self.num_heads, self.head_dim).permute(1, 0, 2)  # [h,M,d]
        Vh = V.view(self.mem_dim, self.num_heads, self.head_dim).permute(1, 0, 2)  # [h,M,d]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        logits = torch.einsum("bhd,hmd->bhm", Qh, Kh) / scale  # [B,h,M]
        if self.temperature != 1.0:
            logits = logits / self.temperature
        attn = F.softmax(logits, dim=-1)  # [B,h,M]

        Oh = torch.einsum("bhm,hmd->bhd", attn, Vh)  # [B,h,d]
        z_mem = Oh.reshape(B, self.z_dim)  # [B,p]
        return z_mem




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        assert d_model % nheads == 0

        self.d_k = d_model // nheads
        self.h = nheads
        self.dropout = nn.Dropout(dropout)

        # W_Q, W_K, W_V, W_O
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None  # optional: store last attention map

    def attention(self, query, key, value):
        """
        query: (B, h, Lq, d_k)
        key:   (B, h, Lk, d_k)
        value: (B, h, Lk, d_k)
        return:
          out:  (B, h, Lq, d_k)
          attn: (B, h, Lq, Lk)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (B,h,Lq,Lk)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, value), attn

    def forward(self, q, k, v):
        """
        Expected:
          q,k,v: (B, L, d_model)

        We still support rank-2 input:
          (L, d_model) -> treated as a single sample (batch=1)
        """
        squeeze_batch = False
        if q.dim() == 2:
            q = q.unsqueeze(0)  # (1, L, d_model)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            squeeze_batch = True

        B, Lq, _ = q.shape
        _, Lk, _ = k.shape

        # Linear projections
        q, k, v = [lin(x) for lin, x in zip(self.linears[:3], (q, k, v))]  # (B, L, d_model)

        # Split heads -> (B, h, L, d_k)
        q = q.view(B, Lq, self.h, self.d_k).transpose(1, 2)
        k = k.view(B, Lk, self.h, self.d_k).transpose(1, 2)
        v = v.view(B, Lk, self.h, self.d_k).transpose(1, 2)

        # Attention
        x, self.attn = self.attention(q, k, v)  # (B, h, Lq, d_k)

        # Concat heads -> (B, Lq, d_model)
        x = x.transpose(1, 2).contiguous().view(B, Lq, self.h * self.d_k)

        # Output projection
        x = self.linears[3](x)  # (B, Lq, d_model)

        if squeeze_batch:
            return x.squeeze(0)  # (Lq, d_model)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nheads, hidden_dim=512, dropout=0.3) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(d_model, nheads, dropout=dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # self-attn / cross-attn on token dimension only
        attn_out = self.attention(q, k, v)  # (B, L, d_model) or (L, d_model)
        x = self.norm[0](q + self.dropout(attn_out))
        f = self.fc(x)
        x = self.norm[1](x + self.dropout(f))
        return x


class TFBlock(nn.Module):
    """
    Interface unchanged:
      forward(z, res_z) -> z_star

    Supported input shapes:
      z, res_z: (N, d) or (B, N, d)

    Behavior:
      For each cell, build a 2-token sequence [z_i, res_z_i]
      and do attention ONLY within these 2 tokens.
      No cross-cell mixing.
    """
    def __init__(self, latent_dim, num_layers=3, nheads=4, hidden_dim=512, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerLayer(latent_dim, nheads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, res_z):
        if z.shape != res_z.shape:
            raise ValueError(f"z.shape {z.shape} must match res_z.shape {res_z.shape}")

        input_dim = z.dim()

        if input_dim == 2:
            # z, res_z: (N, d)
            # Treat each cell as one sample, with 2 tokens: [z_i, res_z_i]
            # tokens: (N, 2, d)
            tokens = torch.stack([z, res_z], dim=1)
            restore_shape = z.shape  # (N, d)

        elif input_dim == 3:
            # z, res_z: (B, N, d)
            # Build per-cell 2-token sequences, then flatten cells into batch
            B, N, D = z.shape
            # (B, N, 2, d) -> (B*N, 2, d)
            tokens = torch.stack([z, res_z], dim=2).reshape(B * N, 2, D)
            restore_shape = (B, N, D)

        else:
            raise ValueError(f"Expected z/res_z rank 2 or 3, got rank {input_dim}")

        tokens = self.dropout(tokens)

        # Self-attention within each cell's 2-token sequence only
        for layer in self.layers:
            tokens = layer(tokens, tokens, tokens)

        # Take the updated first token as fused z_star
        # token 0 corresponds to original z
        z_star = tokens[:, 0, :]

        if input_dim == 2:
            # (N, d)
            return z_star.view(*restore_shape)

        # (B, N, d)
        return z_star.view(*restore_shape)

class StyleBlock(nn.Module):
    def __init__(self, num_batches: int, latent_dim: int):
        super().__init__()
        self.n = num_batches
        self.style = nn.Parameter(torch.Tensor(num_batches, latent_dim))
        self._init_parameters()

    def _init_parameters(self):
        stdv = 1.0 / math.sqrt(self.style.size(1))
        self.style.data.uniform_(-stdv, stdv)

    def forward(self, z, batchid):
        if self.n == 1:
            return z - self.style[0]
        else:
            s = torch.mm(batchid, self.style)
            return z - s