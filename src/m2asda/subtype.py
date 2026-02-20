import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from typing import Dict, Any
from tqdm import tqdm

from .utils import seed_everything, update_configs_with_args, PairDataset
from .model import GeneratorWithMemory, Subtyper
from .configs import SubtypeConfigs
from .loss import KLLoss


class SubtypeModel:
    n_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: torch.device
    random_state: int
    n_genes: int
    s_configs: Dict[str, Any]

    def __init__(self, generator, num_types, configs: SubtypeConfigs):
        for k, v in configs.__dict__.items():
            setattr(self, k, v)
        seed_everything(self.random_state)
        self._init_model(generator, num_types)

    def _init_model(self, generator: GeneratorWithMemory, num_types: int):
        # Phase III: freeze Phase-I reconstruction module (E^I, M^I, G^I)
        self.G = generator.to(self.device)
        self.G.eval()
        for p in self.G.parameters():
            p.requires_grad_(False)

        # Trainable subtyping module (E^Δ + TFBlock + clustering)
        self.S = Subtyper(input_dim=self.n_genes, generator=self.G, num_types=num_types, **self.s_configs).to(self.device)
        self.opt_S = optim.Adam(
            self.S.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=self.weight_decay,
        )
        self.sch_S = CosineAnnealingLR(self.opt_S, T_max=self.n_epochs)

        self.loss = KLLoss().to(self.device)

    @torch.no_grad()
    def _full_assign_and_update_mu(self, z: torch.Tensor, res: torch.Tensor, eval_bs: int = 4096):

        self.S.eval()

        # 用固定顺序的 loader，保证 y 的顺序稳定，便于和 y_prev 对比
        ds = PairDataset(z.detach().cpu(), res.detach().cpu())
        ld = DataLoader(ds, batch_size=eval_bs, shuffle=False, num_workers=4, pin_memory=True)

        K = self.S.num_types
        D = self.S.z_dim
        sum_feat = torch.zeros(K, D, device=self.device)
        count = torch.zeros(K, 1, device=self.device)

        ys = []

        for z_b, res_b in ld:
            z_b = z_b.to(self.device, non_blocking=True)
            res_b = res_b.to(self.device, non_blocking=True)

            z_star, q = self.S(z_b, res_b)  # forward 不更新 mu
            y = torch.argmax(q, dim=1)  # [B]
            ys.append(y.detach().cpu())

            # 累计每个簇的特征和/计数，用于硬更新 mu
            w = F.one_hot(y, num_classes=K).to(dtype=z_star.dtype)  # [B,K]
            sum_feat += w.transpose(0, 1) @ z_star  # [K,D]
            count += w.sum(dim=0).unsqueeze(1)  # [K,1]

        # 硬更新 mu（只更新非空簇）
        new_mu = sum_feat / (count + self.S.eps)
        mask = count.squeeze(1) > 0
        self.S.mu[mask].copy_(new_mu[mask])

        return torch.cat(ys, dim=0)  # [N] on CPU

    def train(self, adata: ad.AnnData):
        X = adata.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        data = torch.from_numpy(X).float().to(self.device)

        # z = E^I(H) (frozen), res = Δ = H - G^I(M^I(E^I(H)))
        z, res = self.generate_z_res(data)

        # Initialize cluster centers by k-means on the initial fused embeddings
        with torch.no_grad():
            self.S.eval()
            res_z = self.S.E_delta(res)  # E^Δ(Δ)
            z_star0 = self.S.fusion(z, res_z)
            self.S.mu_init(z_star0, seed=self.random_state)

        dataset = PairDataset(z.detach().cpu(), res.detach().cpu())
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        self.S.train()

        y_prev = None

        with tqdm(total=self.n_epochs) as t:
            for it in range(self.n_epochs):
                t.set_description("Training Epochs")

                self.S.train()
                for z_b, res_b in loader:
                    z_b = z_b.to(self.device, non_blocking=True)
                    res_b = res_b.to(self.device, non_blocking=True)

                    z_star, q = self.S(z_b, res_b)

                    p = self.S.target_distribution(q, eps=self.S.eps).detach()

                    self.opt_S.zero_grad(set_to_none=True)
                    loss = self.loss(p, q)
                    loss.backward()
                    self.opt_S.step()

                self.sch_S.step()

                # 2) epoch 末：全量计算 hard labels，并硬更新一次 mu
                y = self._full_assign_and_update_mu(z, res, eval_bs=self.batch_size)

                # 3) 论文早停：hard assignment change < 0.001
                if y_prev is not None:
                    delta = (y != y_prev).float().mean().item()
                    t.set_postfix(Loss=float(loss.item()), delta=delta)
                    if delta < 0.001:
                        break
                else:
                    t.set_postfix(Loss=float(loss.item()), delta=float("nan"))

                y_prev = y
                t.update(1)

        with torch.no_grad():
            self.S.eval()
            return self.predict_labels(z, res)

    @torch.no_grad()
    def generate_z_res(self, data: torch.Tensor):
        self.G.eval()
        x_hat, z, _z_mem = self.G(data, update_mem=False)
        res = data - x_hat.detach()
        return z.detach(), res.detach()

    @torch.no_grad()
    def predict_labels(self, z: torch.Tensor, res: torch.Tensor):
        """Recompute q without calling Subtyper.forward() (avoid mu_update during eval)."""
        res_z = self.S.E_delta(res)
        z_star = self.S.fusion(z, res_z)

        dist2 = torch.sum((z_star.unsqueeze(1) - self.S.mu) ** 2, dim=2)  # [N, K]
        q = 1.0 / (1.0 + dist2 / (self.S.alpha + self.S.eps))
        q = q.pow((self.S.alpha + 1.0) / 2.0)
        q = q / (q.sum(dim=1, keepdim=True) + self.S.eps)

        return torch.argmax(q, dim=1).detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2ASDA for anomaly subtyping (Phase III).")
    configs = SubtypeConfigs()

    # Data path arguments
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--read_path", type=str, required=True, help="Path to read the h5ad file (already aligned).")
    data_group.add_argument("--save_path", type=str, default="result.csv", help="Path to save output csv file")
    data_group.add_argument("--pth_path", type=str, required=True, help="Path to read the trained Phase-I generator")

    # SubtypeModel arguments with defaults from SubtypeConfigs
    s_group = parser.add_argument_group("SubtypeModel Parameters")
    s_group.add_argument("--n_epochs", type=int, default=configs.n_epochs, help="Number of epochs")
    s_group.add_argument("--batch_size", type=int, default=configs.batch_size, help="Batch size")
    s_group.add_argument("--learning_rate", type=float, default=configs.learning_rate, help="Learning rate")
    s_group.add_argument("--weight_decay", type=float, default=configs.weight_decay, help="Weight decay rate")
    s_group.add_argument("--GPU", type=str, default=configs.GPU, help="GPU ID for training, e.g., cuda:0")
    s_group.add_argument("--random_state", type=int, default=configs.random_state, help="Random seed")
    s_group.add_argument("--n_genes", type=int, default=configs.n_genes, help="Number of genes")
    s_group.add_argument("--num_types", type=int, required=True, help="Number of anomaly subtypes")

    args = parser.parse_args()
    args_dict = vars(args)

    update_configs_with_args(configs, args_dict, None)
    configs.build()
    configs.clear()

    print("=============== SubtypeModel Parameters ===============")
    for key, value in configs.__dict__.items():
        print(f"{key} = {value}")

    adata = sc.read_h5ad(args.read_path)
    generator = torch.load(args.pth_path, map_location="cpu", weights_only=False)

    model = SubtypeModel(generator, args.num_types, configs)
    sublabel = model.train(adata)

    df = pd.DataFrame({"subtype": sublabel}, index=adata.obs_names)
    df.to_csv(args.save_path)
