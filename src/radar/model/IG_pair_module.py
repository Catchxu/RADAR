# pair_module.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from captum.attr import IntegratedGradients
from tqdm import tqdm


def _to_float_tensor(x):
    if torch.is_tensor(x):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


class SimpleRefAttentionReconstructor(nn.Module):
    """
    Simple attention-based reconstruction module.

    Input:
        z_tgt: [B, d]
        z_ref: [N_ref, d] or [B, N_ref, d]

    Output:
        z_hat: [B, d]
        alpha: [B, N_ref]

    The value vectors are directly taken from the original reference latent
    embeddings. We intentionally avoid using W_V and W_O here, so that
    attribution scores remain easier to interpret at the reference-cell level.
    """

    def __init__(self, latent_dim, attn_dim=None, temperature=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.attn_dim = attn_dim or latent_dim

        self.W_q = nn.Linear(latent_dim, self.attn_dim, bias=False)
        self.W_k = nn.Linear(latent_dim, self.attn_dim, bias=False)

  
        self.temperature = temperature or math.sqrt(self.attn_dim)
        # Initialize the attention projections close to identity when possible.
        if self.attn_dim == latent_dim:
            nn.init.eye_(self.W_q.weight)
            nn.init.eye_(self.W_k.weight)

    def forward(self, z_tgt, z_ref):
        q = self.W_q(z_tgt)

        if z_ref.dim() == 2:
            # Shared reference bank: [N_ref, d]
            k = self.W_k(z_ref)
            scores = torch.matmul(q, k.T) / self.temperature
            alpha = torch.softmax(scores, dim=-1)
            z_hat = torch.matmul(alpha, z_ref)

        elif z_ref.dim() == 3:
            # Batched reference bank: [B, N_ref, d]
            k = self.W_k(z_ref)
            scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / self.temperature
            alpha = torch.softmax(scores, dim=-1)
            z_hat = torch.bmm(alpha.unsqueeze(1), z_ref).squeeze(1)

        else:
            raise ValueError(f"z_ref must be [N_ref, d] or [B, N_ref, d], got {z_ref.shape}")

        return z_hat, alpha


class ReconstructionScoreWrapper(nn.Module):
    """
    Wrapper used by Captum Integrated Gradients.

    The input is a reference bank:
        z_ref_input: [1, N_ref, d] or [B_ig, N_ref, d]

    The output is the reconstruction score:
        F = -||z_tgt - z_hat||_2^2

    Integrated Gradients attributes this scalar score back to the reference bank.
    """

    def __init__(self, reconstructor, z_tgt_single):
        super().__init__()
        self.reconstructor = reconstructor
        self.register_buffer("z_tgt_single", z_tgt_single.view(1, -1))

    def forward(self, z_ref_input):
        if z_ref_input.dim() == 2:
            z_ref_input = z_ref_input.unsqueeze(0)

        batch_size = z_ref_input.shape[0]
        z_tgt = self.z_tgt_single.expand(batch_size, -1)

        z_hat, _ = self.reconstructor(z_tgt, z_ref_input)

        score = -((z_tgt - z_hat) ** 2).sum(dim=1)
        return score


class IGReferencePairModule:
    """
    Target-reference pair module based on reconstruction attribution.

    Main workflow:
        1. Use the frozen Phase-I encoder to extract Z_ref and Z_tgt.
        2. Train a simple attention reconstructor to reconstruct each target
           embedding from the full reference latent bank.
        3. For each target cell, use Integrated Gradients to attribute the
           reconstruction score back to the reference bank.
        4. Aggregate positive IG contributions across latent channels for each
           reference cell.
        5. Select the reference cell with the largest contribution as the
           matched reference anchor.
    """

    def __init__(
        self,
        phase1_encoder,
        device="cuda",
        attn_dim=None,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=512,
        epochs=100,
        normalize_latent=True,
    ):
        self.encoder = phase1_encoder
        self.device = device
        self.attn_dim = attn_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.normalize_latent = normalize_latent

        self.reconstructor = None
        self.z_ref = None
        self.z_tgt = None
        self.history = []

    def freeze_encoder(self):
        self.encoder.to(self.device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, X, encode_batch_size=4096):
        """
        Encode an expression matrix using the frozen Phase-I encoder.

        Args:
            X: Tensor or numpy array with shape [N, n_gene].
            encode_batch_size: Batch size used during encoding.

        Returns:
            z: Tensor with shape [N, d].
        """
        self.freeze_encoder()

        X = _to_float_tensor(X)
        loader = DataLoader(TensorDataset(X), batch_size=encode_batch_size, shuffle=False)

        zs = []
        for (xb,) in tqdm(loader, desc="Encoding"):
            xb = xb.to(self.device)
            z = self.encoder(xb)

            # Support encoders that return a tuple or list.
            if isinstance(z, (tuple, list)):
                z = z[0]

            zs.append(z.detach().cpu())

        z = torch.cat(zs, dim=0).float()

        if self.normalize_latent:
            z = F.normalize(z, p=2, dim=1)

        return z

    def fit(self, X_ref, X_tgt_normal):
        """
        Train the attention reconstruction module.

        Args:
            X_ref: Reference expression matrix with shape [N_ref, n_gene].
            X_tgt_normal: Phase-I predicted normal target expression matrix
                with shape [N_tgt, n_gene].
        """
        print("Extracting reference latent...")
        z_ref = self.encode(X_ref)

        print("Extracting target latent...")
        z_tgt = self.encode(X_tgt_normal)

        latent_dim = z_ref.shape[1]

        self.z_ref = z_ref.to(self.device)
        self.z_tgt = z_tgt.cpu()

        self.reconstructor = SimpleRefAttentionReconstructor(
            latent_dim=latent_dim,
            attn_dim=self.attn_dim,
        ).to(self.device)

        dataset = TensorDataset(self.z_tgt)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.reconstructor.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.reconstructor.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_n = 0

            for (zt,) in loader:
                zt = zt.to(self.device)

                z_hat, alpha = self.reconstructor(zt, self.z_ref)

                loss = ((zt - z_hat) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * zt.shape[0]
                total_n += zt.shape[0]

            avg_loss = total_loss / total_n
            self.history.append(avg_loss)

            print(f"[PairModule] Epoch {epoch + 1}/{self.epochs} | recon_loss={avg_loss:.6f}")

        return self

    @torch.no_grad()
    def attention_pair(self, target_index):
        """
        Return the reference cell with the largest attention weight.

        This is mainly used as a debugging or sanity-check function.
        """
        self.reconstructor.eval()

        zt = self.z_tgt[target_index].to(self.device).view(1, -1)
        z_hat, alpha = self.reconstructor(zt, self.z_ref)

        pair_idx = torch.argmax(alpha[0]).item()
        pair_weight = alpha[0, pair_idx].item()
        recon_mse = ((zt - z_hat) ** 2).mean().item()

        return pair_idx, pair_weight, recon_mse

    def ig_pair_one(
        self,
        target_index,
        ig_steps=32,
        internal_batch_size=4,
        score_mode="positive",
    ):
        """
        Run IG-based matching for one target cell.

        Args:
            target_index: Index of the target cell.
            ig_steps: Number of interpolation steps for Integrated Gradients.
            internal_batch_size: Internal batch size used by Captum.
            score_mode:
                "positive": sum(ReLU(IG)) across latent channels.
                "abs": sum(abs(IG)) across latent channels.

        Returns:
            A dictionary containing the IG-based pair and debugging information.
        """
        assert self.reconstructor is not None, "Call fit() first."

        self.reconstructor.eval()

        zt = self.z_tgt[target_index].to(self.device)

        wrapper = ReconstructionScoreWrapper(
            reconstructor=self.reconstructor,
            z_tgt_single=zt,
        ).to(self.device)

        ig = IntegratedGradients(wrapper)

        ref_input = self.z_ref.detach().unsqueeze(0)
        ref_input.requires_grad_(True)

        ref_mean = self.z_ref.detach().mean(dim=0, keepdim=True)
        baseline = ref_mean.repeat(self.z_ref.shape[0], 1).unsqueeze(0)

        attr = ig.attribute(
            ref_input,
            baselines=baseline,
            n_steps=ig_steps,
            internal_batch_size=internal_batch_size,
        )

        attr = attr.squeeze(0).detach()

        if score_mode == "positive":
            ref_scores = torch.clamp(attr, min=0.0).sum(dim=1)
        elif score_mode == "abs":
            ref_scores = attr.abs().sum(dim=1)
        else:
            raise ValueError("score_mode must be 'positive' or 'abs'.")

        pair_idx = torch.argmax(ref_scores).item()
        pair_score = ref_scores[pair_idx].item()

        attn_idx, attn_weight, recon_mse = self.attention_pair(target_index)

        result = {
            "target_index": target_index,
            "pair_index": pair_idx,
            "pair_score": pair_score,
            "attention_pair_index": attn_idx,
            "attention_pair_weight": attn_weight,
            "recon_mse": recon_mse,
        }

        del attr, ref_scores, ref_input, baseline, wrapper, ig
        torch.cuda.empty_cache()

        return result

    def match_all(
        self,
        ig_steps=32,
        internal_batch_size=4,
        score_mode="positive",
        max_targets=None,
    ):
        """
        Run IG-based matching for all target cells.

        Args:
            ig_steps: Number of interpolation steps for Integrated Gradients.
            internal_batch_size: Internal batch size used by Captum.
            score_mode: Attribution aggregation mode.
            max_targets: Optional maximum number of target cells to match.

        Returns:
            A list of dictionaries. Each dictionary stores the matching result
            for one target cell.
        """
        n_tgt = self.z_tgt.shape[0]
        if max_targets is not None:
            n_tgt = min(n_tgt, max_targets)

        results = []

        for i in tqdm(range(n_tgt), desc="IG matching"):
            res = self.ig_pair_one(
                target_index=i,
                ig_steps=ig_steps,
                internal_batch_size=internal_batch_size,
                score_mode=score_mode,
            )
            results.append(res)

        return results