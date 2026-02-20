import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from sklearn.cluster import KMeans

from .generator import GeneratorWithMemory
from .layer import TFBlock, Encoder, LinearBlock


class Subtyper(nn.Module):
    def __init__(
        self,
        input_dim,
        generator: GeneratorWithMemory,
        num_types: int,
        alpha: float = 1.0,
        kmeans_n_init: int = 20,
        num_layers: int = 3,
        nheads: int = 4,
        hidden_dim=[1024, 512, 256],
        ff_hidden_dim: int = 512,
        dropout: float = 0.1,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()

        self.G = generator
        self.z_dim = int(self.G.extractor.latent_dim)

        self.num_types = int(num_types)
        self.alpha = float(alpha)
        self.kmeans_n_init = int(kmeans_n_init)
        self.eps = float(eps)

        self.E_delta = Encoder(input_dim, hidden_dim, **kwargs)
        # Trainable fusion block
        self.fusion = TFBlock(self.z_dim, num_layers, nheads, ff_hidden_dim, dropout)

        # Cluster centers (DEC-style)
        self.register_buffer("mu", torch.empty(self.num_types, self.z_dim))
        nn.init.normal_(self.mu, mean=0.0, std=0.02)

        # Optional classifier head (kept for compatibility)
        self.classifer = nn.Linear(self.z_dim, self.num_types)

        # Initialize only trainable components (DO NOT touch Phase-I generator weights)
        self._init_trainable_weights()

    def _init_trainable_weights(self):
        """Xavier init for trainable Linear layers (E_delta, fusion, classifier)."""

        def init_linear(mod: nn.Module):
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)

        init_linear(self.E_delta)
        init_linear(self.fusion)
        init_linear(self.classifer)

    def forward(self, z: torch.Tensor, res: torch.Tensor):
        res_z = self.E_delta(res)  # [N, z_dim]
        z_star = self.fusion(z, res_z)  # [N, z_dim]

        # Student-t / Cauchy kernel
        dist2 = torch.sum((z_star.unsqueeze(1) - self.mu) ** 2, dim=2)  # [N, K]
        q = 1.0 / (1.0 + dist2 / (self.alpha + self.eps))  # [N, K]
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q / (q.sum(dim=1, keepdim=True) + self.eps)
        return z_star, q

    def pretrain(self, z: torch.Tensor, res: torch.Tensor):
        res_z = self.E_delta(res)
        z_star = self.fusion(z, res_z)
        return self.classifer(z_star)

    @staticmethod
    def target_distribution(q: torch.Tensor, eps: float = 1e-8):
        p = (q**2) / (q.sum(dim=0, keepdim=True) + eps)
        p = p / (p.sum(dim=1, keepdim=True) + eps)
        return p

    @torch.no_grad()
    def mu_init(self, feat: torch.Tensor, seed: int = 0):
        feat_np = feat.detach().cpu().numpy()
        km = KMeans(n_clusters=self.num_types, n_init=self.kmeans_n_init, random_state=seed)
        km.fit(feat_np)
        centers = torch.tensor(km.cluster_centers_, device=self.mu.device, dtype=self.mu.dtype)
        self.mu.copy_(centers)

    @torch.no_grad()
    def mu_update(self, feat: torch.Tensor, q: torch.Tensor):
        labels = torch.argmax(q, dim=1)  # [N]
        w = F.one_hot(labels, num_classes=self.num_types).to(dtype=feat.dtype)  # [N, K]

        sum_feat = w.transpose(0, 1) @ feat  # [K, z_dim]
        count = w.sum(dim=0).unsqueeze(1)  # [K, 1]

        new_mu = sum_feat / (count + self.eps)
        mask = count.squeeze(1) > 0
        self.mu[mask].copy_(new_mu[mask])
