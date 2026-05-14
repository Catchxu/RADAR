import argparse
from typing import Dict, Any, List

import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .utils import seed_everything, update_configs_with_args, set_requires_grad, PairDatasetWithBatch, to_dense
from .model import GeneratorWithMemory, Discriminator, GeneratorWithStyle
from .configs import CorrectConfigs
from .loss import rpgan_G_loss, rpgan_D_loss


class CorrectModel:
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
        self.G = GeneratorWithStyle(
            num_batches=num_batches,
            **self.g_configs,
        ).to(self.device)

        self.G.init_from_phase1(phase1_G, strict=True)

        self.D = Discriminator(**self.d_configs).to(self.device)

        self.opt_G = optim.Adam(
            self.G.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )
        self.opt_D = optim.Adam(
            self.D.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999),
        )

        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)
        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)

    def train(
        self,
        ref: ad.AnnData,
        tgt: ad.AnnData,
        ref_pair: ad.AnnData,
        batch_key: str = "batch",
    ) -> ad.AnnData:
        Nb = int(self.G.style.n)

        dataset = PairDatasetWithBatch(
            ref_pair,
            tgt,
            batch_key=batch_key,
            num_batches=Nb,
        )
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

                t.set_postfix(
                    G_Loss=float(self.G_loss.item()),
                    D_Loss=float(self.D_loss.item()),
                )
                t.update(1)

                self.sch_G.step()
                self.sch_D.step()

        corrected_tgt = self.correct_targets(tgt, batch_key=batch_key)

        out = ad.concat(
            [ref, corrected_tgt],
            join="inner",
            label="domain",
            keys=["ref", "corrected_tgt"],
            index_unique=None,
        )
        out.raw = out.copy()

        tqdm.write("Batch effects have been corrected.\n")
        return out

    def UpdateG(
        self,
        x_r: torch.Tensor,
        x_t: torch.Tensor,
        b_onehot: torch.Tensor,
    ):
        set_requires_grad(self.D, False)

        x_hat_r, _, _ = self.G(x_t, b_onehot)
        self.G_loss = rpgan_G_loss(self.D, x_real=x_r, x_fake=x_hat_r)

        self.opt_G.zero_grad()
        self.G_loss.backward()
        self.opt_G.step()

        set_requires_grad(self.D, True)

    def UpdateD(
        self,
        x_r: torch.Tensor,
        x_t: torch.Tensor,
        b_onehot: torch.Tensor,
    ):
        with torch.no_grad():
            x_hat_r, _, _ = self.G(x_t, b_onehot)

        gamma = float(self.loss_weight["gamma"])
        self.D_loss = rpgan_D_loss(
            self.D,
            x_real=x_r,
            x_fake=x_hat_r,
            gamma=gamma,
        )

        self.opt_D.zero_grad()
        self.D_loss.backward()
        self.opt_D.step()

    @torch.no_grad()
    def correct_targets(
        self,
        tgt: ad.AnnData,
        batch_key: str = "batch",
    ) -> ad.AnnData:
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
            c = torch.tensor(
                codes[i : i + self.batch_size],
                dtype=torch.long,
                device=self.device,
            )

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


def align_ref_tgt_genes(ref: ad.AnnData, tgt: ad.AnnData):
    ref_genes = ref.var_names.astype(str)
    tgt_gene_set = set(tgt.var_names.astype(str))

    common_genes = [g for g in ref_genes if g in tgt_gene_set]

    if len(common_genes) == 0:
        raise ValueError("No shared genes between reference and target.")

    ref = ref[:, common_genes].copy()
    tgt = tgt[:, common_genes].copy()

    print(f"Shared genes used for correction: {len(common_genes)}")

    return ref, tgt


def build_ref_pair_from_ig_csv(
    ref: ad.AnnData,
    tgt: ad.AnnData,
    pair_csv_path: str,
    target_col: str = "target_cell_id",
    ref_col: str = "paired_ref_cell_id",
):
    pair_df = pd.read_csv(pair_csv_path)

    required_cols = {target_col, ref_col}
    missing = required_cols - set(pair_df.columns)
    if missing:
        raise ValueError(
            f"Pair CSV is missing required columns: {missing}. "
            f"Available columns: {list(pair_df.columns)}"
        )

    if not ref.obs_names.is_unique:
        raise ValueError("Reference obs_names are not unique.")

    if not tgt.obs_names.is_unique:
        raise ValueError(
            "Target obs_names are not unique. "
            "Please make target cell ids unique before using pair CSV."
        )

    pair_df[target_col] = pair_df[target_col].astype(str)
    pair_df[ref_col] = pair_df[ref_col].astype(str)

    valid_mask = (
        pair_df[target_col].isin(tgt.obs_names)
        & pair_df[ref_col].isin(ref.obs_names)
    )
    pair_df = pair_df.loc[valid_mask].copy()

    if pair_df.empty:
        raise ValueError(
            "No valid target-reference pairs found. "
            "Check whether target_cell_id and paired_ref_cell_id match obs_names."
        )

    print(f"Valid IG pairs used for CorrectModel: {len(pair_df)}")

    tgt_ordered = tgt[pair_df[target_col].values].copy()
    ref_pair = ref[pair_df[ref_col].values].copy()

    if tgt_ordered.n_obs != ref_pair.n_obs:
        raise RuntimeError("ref_pair and tgt have different numbers of cells.")

    return ref_pair, tgt_ordered


def load_phase1_generator(pth_path: str, device):
    try:
        phase1_G = torch.load(
            pth_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        phase1_G = torch.load(
            pth_path,
            map_location=device,
        )

    if not isinstance(phase1_G, torch.nn.Module):
        raise TypeError(
            f"Expected Phase-I checkpoint to be a full torch module, "
            f"but got {type(phase1_G)}."
        )

    phase1_G = phase1_G.to(device)
    phase1_G.eval()

    for p in phase1_G.parameters():
        p.requires_grad_(False)

    return phase1_G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RADAR Phase II: CorrectModel only."
    )

    c_configs = CorrectConfigs()

    parser.add_argument(
        "--ref_path",
        type=str,
        default="data/ref_clean_colorectum.h5ad",
        help="Reference h5ad path.",
    )
    parser.add_argument(
        "--tgt_paths",
        nargs="+",
        required=True,
        help="Target h5ad path(s).",
    )
    parser.add_argument(
        "--pair_csv_path",
        type=str,
        default="output/pair_module/target_ref_pairs_ig.csv",
        help="IG pairing result CSV.",
    )
    parser.add_argument(
        "--pth_path",
        type=str,
        default="ckpt/phase1_G.pth",
        help="Path to trained Phase-I GeneratorWithMemory.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/correct.h5ad",
        help="Output corrected h5ad path.",
    )
    parser.add_argument(
        "--batch_key",
        type=str,
        default="batch",
        help="obs key for target batch labels.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="target_cell_id",
        help="Target cell id column in pair CSV.",
    )
    parser.add_argument(
        "--ref_col",
        type=str,
        default="paired_ref_cell_id",
        help="Paired reference cell id column in pair CSV.",
    )

    parser.add_argument("--n_epochs_c", type=int, default=c_configs.n_epochs)
    parser.add_argument("--batch_size_c", type=int, default=c_configs.batch_size)
    parser.add_argument("--learning_rate_c", type=float, default=c_configs.learning_rate)
    parser.add_argument("--n_critic_c", type=int, default=c_configs.n_critic)
    parser.add_argument("--gamma_c", type=float, default=c_configs.gamma)
    parser.add_argument("--GPU_c", type=str, default=c_configs.GPU)
    parser.add_argument("--random_state_c", type=int, default=c_configs.random_state)
    parser.add_argument("--n_genes_c", type=int, default=c_configs.n_genes)

    args = parser.parse_args()
    args_dict = vars(args)

    update_configs_with_args(c_configs, args_dict, "_c")
    c_configs.build()
    c_configs.clear()

    for k, v in c_configs.__dict__.items():
        print(f"{k} = {v}")

    seed_everything(c_configs.random_state)

    print(f"Reading reference: {args.ref_path}")
    ref = sc.read_h5ad(args.ref_path)

    print("Reading target dataset(s)...")
    tgt_list: List[ad.AnnData] = [sc.read_h5ad(p) for p in args.tgt_paths]

    batch_key = args.batch_key
    keys = [f"tgt{i}" for i in range(1, len(tgt_list) + 1)]

    tgt = ad.concat(
        tgt_list,
        label=batch_key,
        keys=keys,
        index_unique=None,
        join="inner",
    )

    if not hasattr(tgt.obs[batch_key], "cat"):
        tgt.obs[batch_key] = tgt.obs[batch_key].astype("category")

    ref, tgt = align_ref_tgt_genes(ref, tgt)

    print(f"Reference cells: {ref.n_obs}")
    print(f"Target cells: {tgt.n_obs}")
    print(f"Reference genes: {ref.n_vars}")
    print(f"Target genes: {tgt.n_vars}")

    print(f"Building paired reference cells from: {args.pair_csv_path}")
    ref_pair, tgt = build_ref_pair_from_ig_csv(
        ref=ref,
        tgt=tgt,
        pair_csv_path=args.pair_csv_path,
        target_col=args.target_col,
        ref_col=args.ref_col,
    )

    print(f"Paired reference cells: {ref_pair.n_obs}")
    print(f"Ordered target cells: {tgt.n_obs}")

    print(f"Loading Phase-I generator: {args.pth_path}")
    phase1_G = load_phase1_generator(args.pth_path, c_configs.device)

    num_batches = len(tgt.obs[batch_key].cat.categories)
    print(f"Number of target batches: {num_batches}")

    correct_model = CorrectModel(
        num_batches=num_batches,
        phase1_G=phase1_G,
        **c_configs.__dict__,
    )

    adata_corrected = correct_model.train(
        ref=ref,
        tgt=tgt,
        ref_pair=ref_pair,
        batch_key=batch_key,
    )

    print(f"Saving corrected AnnData to: {args.save_path}")
    adata_corrected.write_h5ad(args.save_path)