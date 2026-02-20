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

        end = ptr + B
        if end <= self.mem_dim:
            self.mem[ptr:end] = z
        else:
            first = self.mem_dim - ptr
            self.mem[ptr:] = z[:first]
            self.mem[: end % self.mem_dim] = z[first:]
        self.mem_ptr[0] = end % self.mem_dim

    def forward(self, z: torch.Tensor, update_mem: bool = True) -> torch.Tensor:
        if update_mem:
            self.enqueue(z)

        # Q=ZWq, K=MWk, V=MWv
        Q = self.Wq(z)  # [B,p]
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        assert d_model % nheads == 0

        self.d_k = d_model // nheads
        self.h = nheads
        self.dropout = nn.Dropout(dropout)

        # 4 linear layers: W_Q, W_K, W_V, W_O
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])

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
        Accepts:
          q,k,v: (N, d_model)  or (B, N, d_model)

        Returns:
          (N, d_model) or (B, N, d_model) matching input rank.
        """
        squeeze_batch = False
        if q.dim() == 2:
            # Treat N as sequence length (tokens), add batch dim = 1
            q = q.unsqueeze(0)  # (1, N, d_model)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            squeeze_batch = True

        B, Nq, _ = q.shape
        _, Nk, _ = k.shape

        # Linear projections
        q, k, v = [lin(x) for lin, x in zip(self.linears[:3], (q, k, v))]  # (B, N, d_model)

        # Split heads: (B, h, N, d_k)
        q = q.view(B, Nq, self.h, self.d_k).transpose(1, 2)
        k = k.view(B, Nk, self.h, self.d_k).transpose(1, 2)
        v = v.view(B, Nk, self.h, self.d_k).transpose(1, 2)

        # Attention
        x, self.attn = self.attention(q, k, v)  # x: (B, h, Nq, d_k)

        # Concat heads: (B, Nq, d_model)
        x = x.transpose(1, 2).contiguous().view(B, Nq, self.h * self.d_k)

        # Output projection
        x = self.linears[3](x)  # (B, Nq, d_model)

        if squeeze_batch:
            return x.squeeze(0)  # (Nq, d_model)
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
        # cross-attn: Q from q, K/V from k,v
        attn_out = self.attention(q, k, v)  # (N, d_model) if input is (N,d_model)
        x = self.norm[0](q + self.dropout(attn_out))
        f = self.fc(x)
        x = self.norm[1](x + self.dropout(f))
        return x


class TFBlock(nn.Module):
    def __init__(self, latent_dim, num_layers=3, nheads=4, hidden_dim=512, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([TransformerLayer(latent_dim, nheads, hidden_dim, dropout) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, z, res_z):
        z = self.dropout(z)
        res_z = self.dropout(res_z)

        for layer in self.layers:
            z = layer(z, res_z, res_z)  # cross-attention: Q=z, K/V=res_z

        return z
