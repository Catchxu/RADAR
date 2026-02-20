import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .utils import seed_everything, update_configs_with_args, set_requires_grad, PairDatasetWithBatch, to_dense
from .model import GeneratorWithMemory, GeneratorWithPairs, Discriminator, GeneratorWithStyle
from .configs import PairConfigs, CorrectConfigs
from .loss import rpgan_G_loss, rpgan_D_loss


class PairModel:
    # expected attributes from PairConfigs
    n_epochs: int
    learning_rate: float
    n_critic: int
    loss_weight: Dict[str, float]
    device: torch.device
    random_state: int
    n_genes: int
    d_configs: Dict[str, Any]
    # optional (if you add to configs)
    pair_batch_size: Optional[int]

    def __init__(self, phase1_G, n_ref, n_tgt, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._init_model(phase1_G, n_ref, n_tgt)
        seed_everything(self.random_state)

    def _init_model(self, phase1_G: GeneratorWithMemory, n_ref: int, n_tgt: int):
        # Freeze Phase-I encoder EI
        self.EI = phase1_G.extractor.encoder.to(self.device)
        self.EI.eval()
        set_requires_grad(self.EI, False)

        # Module II generator: W in R^{n_tgt x n_ref}, forward(Zr)->Zhat_t
        self.G = GeneratorWithPairs(n_ref=n_ref, n_tgt=n_tgt).to(self.device)

        # IMPORTANT: DII runs on latent space (p-dim), so d_configs.input_dim must be latent_dim
        self.D = Discriminator(**self.d_configs).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)
        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)

        self.n_ref = n_ref
        self.n_tgt = n_tgt

    def check(self, ref: ad.AnnData, tgt: ad.AnnData):
        if self.n_ref != ref.n_obs:
            raise RuntimeError("Number of cells in ref is different with n_ref")
        if self.n_tgt != tgt.n_obs:
            raise RuntimeError("Number of cells in tgt is different with n_tgt")
        if not (ref.var_names == tgt.var_names).all():
            raise RuntimeError("ref and tgt have different genes")

    def _forward_Zhat(self, Zr: torch.Tensor, tgt_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Full-batch: Zhat_t = relu(W) @ Zr
        Mini-batch over targets (optional): pick rows of W, compute Zhat for those targets
        """
        if tgt_idx is None:
            return self.G(Zr)  # [Nt, p]
        # update only rows corresponding to tgt_idx
        W_rows = self.G.W.index_select(0, tgt_idx)  # [B, Nr]
        P_rows = F.relu(W_rows)  # [B, Nr]
        return P_rows @ Zr  # [B, p]

    def train(self, ref: ad.AnnData, tgt: ad.AnnData) -> ad.AnnData:
        self.check(ref, tgt)

        ref_x = torch.tensor(to_dense(ref.X), dtype=torch.float32, device=self.device)
        tgt_x = torch.tensor(to_dense(tgt.X), dtype=torch.float32, device=self.device)

        tqdm.write("Begin to find Kin Pairs between datasets (Module II)...")

        # Encode once: Zr, Zt
        with torch.no_grad():
            Zr = self.EI(ref_x)  # [Nr, p]
            Zt = self.EI(tgt_x)  # [Nt, p]

        self.G.train()
        self.D.train()

        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description("Training Epochs (PairModel)")

                for _ in range(self.n_critic):
                    self.UpdateD(Zr, Zt)

                self.UpdateG(Zr, Zt)

                t.set_postfix(G_Loss=float(self.G_loss.item()), D_Loss=float(self.D_loss.item()))
                t.update(1)

                self.sch_G.step()
                self.sch_D.step()

        # Kin pairing: argmax over each target row of P = relu(W)
        P = torch.relu(self.G.W).detach().cpu().numpy()  # [Nt, Nr]
        idx = list(ref.obs_names[P.argmax(axis=1)])
        ref_pair = ref[idx]

        tqdm.write("Kin Pairs between datasets have been found.\n")
        return ref_pair

    def UpdateG(self, Zr: torch.Tensor, Zt: torch.Tensor):
        # freeze D params during G step
        set_requires_grad(self.D, False)

        if self.pair_batch_size is None:
            Zhat = self._forward_Zhat(Zr, None)
            real = Zt
            fake = Zhat
        else:
            B = min(int(self.pair_batch_size), Zt.size(0))
            idx = torch.randint(0, Zt.size(0), (B,), device=Zt.device)
            real = Zt.index_select(0, idx)
            fake = self._forward_Zhat(Zr, idx)

        self.G_loss = rpgan_G_loss(self.D, x_real=real, x_fake=fake)

        self.opt_G.zero_grad()
        self.G_loss.backward()
        self.opt_G.step()

        set_requires_grad(self.D, True)

    def UpdateD(self, Zr: torch.Tensor, Zt: torch.Tensor):
        if self.pair_batch_size is None:
            with torch.no_grad():
                Zhat = self._forward_Zhat(Zr, None)
            real = Zt
            fake = Zhat
        else:
            B = min(int(self.pair_batch_size), Zt.size(0))
            idx = torch.randint(0, Zt.size(0), (B,), device=Zt.device)
            real = Zt.index_select(0, idx)
            with torch.no_grad():
                fake = self._forward_Zhat(Zr, idx)

        gamma = float(self.loss_weight["gamma"])
        self.D_loss = rpgan_D_loss(self.D, x_real=real, x_fake=fake, gamma=gamma)

        self.opt_D.zero_grad()
        self.D_loss.backward()
        self.opt_D.step()


class CorrectModel:
    # expected attributes from CorrectConfigs
    n_epochs: int
    batch_size: int
    learning_rate: float
    n_critic: int
    loss_weight: Dict[str, float]
    device: torch.device
    random_state: int
    n_genes: int
    g_configs: Dict[str, Any]
    d_configs: Dict[str, Any]

    def __init__(self, num_batches: int, phase1_G: GeneratorWithMemory, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._init_model(num_batches, phase1_G)
        seed_everything(self.random_state)

    def _init_model(self, num_batches: int, phase1_G: GeneratorWithMemory):
        self.G = GeneratorWithStyle(num_batches=num_batches, **self.g_configs).to(self.device)
        # paper: init EIII/GIII from phase-I EI/GI
        self.G.init_from_phase1(phase1_G, strict=True)

        self.D = Discriminator(**self.d_configs).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)
        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)

    def train(self, ref: ad.AnnData, tgt: ad.AnnData, ref_pair: ad.AnnData, batch_key: str = "batch") -> ad.AnnData:
        Nb = int(self.G.style.n)
        dataset = PairDatasetWithBatch(ref_pair, tgt, batch_key=batch_key, num_batches=Nb)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        tqdm.write("Begin to correct batch effects between datasets (Module III)...")

        self.G.train()
        self.D.train()

        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description("Training Epochs (CorrectModel)")

                for x_r, x_t, b in loader:
                    x_r = x_r.to(self.device)
                    x_t = x_t.to(self.device)
                    b = b.to(self.device)

                    for _ in range(self.n_critic):
                        self.UpdateD(x_r, x_t, b)

                    self.UpdateG(x_r, x_t, b)

                t.set_postfix(G_Loss=float(self.G_loss.item()), D_Loss=float(self.D_loss.item()))
                t.update(1)

                self.sch_G.step()
                self.sch_D.step()

        # paper semantics: correct targets into reference space (keep ref unchanged)
        corrected_tgt = self.correct_targets(tgt, batch_key=batch_key)
        out = ad.concat([ref, corrected_tgt], join="inner")
        out.raw = out.copy()
        tqdm.write("Batch effects have been corrected.\n")
        return out

    def UpdateG(self, x_r: torch.Tensor, x_t: torch.Tensor, b_onehot: torch.Tensor):
        set_requires_grad(self.D, False)

        x_hat_r, _, _ = self.G(x_t, b_onehot)
        self.G_loss = rpgan_G_loss(self.D, x_real=x_r, x_fake=x_hat_r)

        self.opt_G.zero_grad()
        self.G_loss.backward()
        self.opt_G.step()

        set_requires_grad(self.D, True)

    def UpdateD(self, x_r: torch.Tensor, x_t: torch.Tensor, b_onehot: torch.Tensor):
        with torch.no_grad():
            x_hat_r, _, _ = self.G(x_t, b_onehot)

        gamma = float(self.loss_weight["gamma"])
        self.D_loss = rpgan_D_loss(self.D, x_real=x_r, x_fake=x_hat_r, gamma=gamma)

        self.opt_D.zero_grad()
        self.D_loss.backward()
        self.opt_D.step()

    @torch.no_grad()
    def correct_targets(self, tgt: ad.AnnData, batch_key: str = "batch") -> ad.AnnData:
        X = torch.tensor(to_dense(tgt.X), dtype=torch.float32)
        codes = tgt.obs[batch_key]
        if hasattr(codes, "cat"):
            codes = codes.cat.codes.to_numpy()
        else:
            codes = np.asarray(codes)

        Nb = int(self.G.style.n)

        self.G.eval()
        corrected = []

        for i in range(0, X.size(0), self.batch_size):
            x = X[i : i + self.batch_size].to(self.device)
            c = torch.tensor(codes[i : i + self.batch_size], dtype=torch.long, device=self.device)

            b = torch.zeros(x.size(0), Nb, device=self.device)
            if Nb == 1:
                b[:, 0] = 1.0
            else:
                b.scatter_(1, c.view(-1, 1), 1.0)

            x_hat_r, _, _ = self.G(x, b)
            corrected.append(x_hat_r.cpu())

        corrected = torch.cat(corrected, dim=0).numpy()
        out = tgt.copy()
        out.X = corrected
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2ASDA Phase II (pairing + transferring) batch correction.")
    p_configs = PairConfigs()
    c_configs = CorrectConfigs()

    # Data arguments
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--read_path", nargs="+", required=True, help="Paths to multiple h5ad files (first is ref)")
    data_group.add_argument("--save_path", type=str, default="correct.h5ad", help="Output corrected h5ad path")
    data_group.add_argument("--pth_path", type=str, required=True, help="Path to load trained Phase-I GeneratorWithMemory")
    data_group.add_argument("--batch_key", type=str, default="batch", help="obs key for target dataset one-hot")
    data_group.add_argument("--phase1_csv_paths", nargs="+", default=None, help="Phase-I prediction csvs for each target (same order as read_path[1:]). Each csv has index=cell_id and a 'label' column.")
    data_group.add_argument("--phase1_label_col", type=str, default="label", help="Column name in Phase-I csv used as ASC label (1=ASC, 0=normal).")

    # PairModel args
    p_group = parser.add_argument_group("PairModel Parameters")
    p_group.add_argument("--n_epochs_p", type=int, default=p_configs.n_epochs)
    p_group.add_argument("--learning_rate_p", type=float, default=p_configs.learning_rate)
    p_group.add_argument("--n_critic_p", type=int, default=p_configs.n_critic)
    p_group.add_argument("--gamma_p", type=float, default=p_configs.gamma)  # gamma for R1/R2
    p_group.add_argument("--GPU_p", type=str, default=p_configs.GPU)
    p_group.add_argument("--random_state_p", type=int, default=p_configs.random_state)
    p_group.add_argument("--n_genes_p", type=int, default=p_configs.n_genes)
    # optional (only works if PairConfigs defines pair_batch_size)
    p_group.add_argument("--pair_batch_size_p", type=int, default=getattr(p_configs, "pair_batch_size", 0))

    # CorrectModel args
    c_group = parser.add_argument_group("CorrectModel Parameters")
    c_group.add_argument("--n_epochs_c", type=int, default=c_configs.n_epochs)
    c_group.add_argument("--batch_size_c", type=int, default=c_configs.batch_size)
    c_group.add_argument("--learning_rate_c", type=float, default=c_configs.learning_rate)
    c_group.add_argument("--n_critic_c", type=int, default=c_configs.n_critic)
    c_group.add_argument("--gamma_c", type=float, default=c_configs.gamma)  # gamma for R1/R2
    c_group.add_argument("--GPU_c", type=str, default=c_configs.GPU)
    c_group.add_argument("--random_state_c", type=int, default=c_configs.random_state)
    c_group.add_argument("--n_genes_c", type=int, default=c_configs.n_genes)

    args = parser.parse_args()
    args_dict = vars(args)

    update_configs_with_args(p_configs, args_dict, "_p")
    update_configs_with_args(c_configs, args_dict, "_c")
    p_configs.build()
    p_configs.clear()
    c_configs.build()
    c_configs.clear()

    for k, v in p_configs.__dict__.items():
        print(f"{k} = {v}")

    for k, v in c_configs.__dict__.items():
        print(f"{k} = {v}")

    # Read h5ad
    adata_list: List[ad.AnnData] = []
    for path in args_dict["read_path"]:
        adata_list.append(sc.read_h5ad(path))

    ref = adata_list[0]
    phase1_csv_paths = args_dict.get("phase1_csv_paths", None)
    label_col = args_dict.get("phase1_label_col", "label")

    if phase1_csv_paths is not None:
        if len(phase1_csv_paths) != (len(adata_list) - 1):
            raise ValueError(f"--phase1_csv_paths must have {len(adata_list)-1} files (one per target), " f"but got {len(phase1_csv_paths)}.")

        new_targets = []
        for i, (adata_t, csv_path) in enumerate(zip(adata_list[1:], phase1_csv_paths), start=1):
            df = pd.read_csv(csv_path, index_col=0)
            if label_col not in df.columns:
                raise ValueError(f"{csv_path} missing column '{label_col}'. Columns: {list(df.columns)}")

            # align by cell ids (obs_names). Missing -> treat as normal (0)
            labels = df[label_col].reindex(adata_t.obs_names).fillna(0).astype(int).to_numpy()
            keep = labels == 0

            before = adata_t.n_obs
            adata_t = adata_t[keep].copy()
            after = adata_t.n_obs
            print(f"[PhaseII filter] tgt{i}: kept {after}/{before} (removed {before-after} ASCs) from {csv_path}")

            if after == 0:
                raise RuntimeError(f"After filtering, tgt{i} has 0 cells left. Check {csv_path} / cell ids match.")

            new_targets.append(adata_t)

        # replace targets with filtered ones
        adata_list = [ref] + new_targets

    # Build tgt with dataset labels -> tgt.obs[batch_key] used to form B_t one-hot
    batch_key = args_dict["batch_key"]
    keys = [f"tgt{i}" for i in range(1, len(adata_list))]  # keys = ["tgt1", "tgt2", ...]
    tgt = ad.concat(
        adata_list[1:],
        label=batch_key,
        keys=keys,
        index_unique=None,
        join="inner",
    )

    # Load Phase-I generator
    device_p = p_configs.device
    phase1_G = torch.load(args_dict["pth_path"], map_location=device_p, weights_only=False)
    if isinstance(phase1_G, torch.nn.Module):
        phase1_G = phase1_G.to(device_p)
        phase1_G.eval()

    # Train Module II
    # ref.n_obs = number of reference cells
    pair_model = PairModel(phase1_G, ref.n_obs, tgt.n_obs, **p_configs.__dict__)
    ref_pair = pair_model.train(ref, tgt)

    # Train Module III
    num_batches = len(adata_list) - 1
    correct_model = CorrectModel(num_batches, phase1_G, **c_configs.__dict__)
    adata_corrected = correct_model.train(ref, tgt, ref_pair, batch_key=batch_key)

    adata_corrected.write_h5ad(args_dict["save_path"])
