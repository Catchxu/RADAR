import argparse
import re
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
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
        self.G = generator.to(self.device)
        self.G.eval()
        for p in self.G.parameters():
            p.requires_grad_(False)

        self.S = Subtyper(
            input_dim=self.n_genes,
            generator=self.G,
            num_types=num_types,
            **self.s_configs
        ).to(self.device)

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

            z_star, q = self.S(z_b, res_b)
            y = torch.argmax(q, dim=1)
            ys.append(y.detach().cpu())

            w = F.one_hot(y, num_classes=K).to(dtype=z_star.dtype)
            sum_feat += w.transpose(0, 1) @ z_star
            count += w.sum(dim=0).unsqueeze(1)

        new_mu = sum_feat / (count + self.S.eps)
        mask = count.squeeze(1) > 0
        self.S.mu[mask].copy_(new_mu[mask])

        return torch.cat(ys, dim=0)

    @torch.no_grad()
    def _compute_full_q_p(self, z: torch.Tensor, res: torch.Tensor, eval_bs: int = 4096):
        self.S.eval()

        ds = PairDataset(z.detach().cpu(), res.detach().cpu())
        ld = DataLoader(
            ds,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        q_list = []

        for z_b, res_b in ld:
            z_b = z_b.to(self.device, non_blocking=True)
            res_b = res_b.to(self.device, non_blocking=True)

            _, q_b = self.S(z_b, res_b)
            q_list.append(q_b.detach().cpu())

        q_all = torch.cat(q_list, dim=0)
        p_all = self.S.target_distribution(q_all, eps=self.S.eps).detach()
        return q_all, p_all

    def train(self, adata: ad.AnnData):
        X = adata.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        data = torch.from_numpy(X).float().to(self.device)

        z, res = self.generate_z_res(data)

        with torch.no_grad():
            self.S.eval()

            ds = PairDataset(z.detach().cpu(), res.detach().cpu())
            ld = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            z_star_list = []
            for z_b, res_b in ld:
                z_b = z_b.to(self.device, non_blocking=True)
                res_b = res_b.to(self.device, non_blocking=True)

                res_z_b = self.S.E_delta(res_b)
                z_star_b = self.S.fusion(z_b, res_z_b)
                z_star_list.append(z_star_b.detach().cpu())

            z_star0 = torch.cat(z_star_list, dim=0)
            self.S.mu_init(z_star0, seed=self.random_state)

        z_cpu = z.detach().cpu()
        res_cpu = res.detach().cpu()
        idx = torch.arange(z_cpu.size(0))

        dataset = TensorDataset(z_cpu, res_cpu, idx)
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
            for _ in range(self.n_epochs):
                t.set_description("Training Epochs")

                _, p_all = self._compute_full_q_p(z, res, eval_bs=self.batch_size)

                self.S.train()
                epoch_loss = 0.0
                n_batches = 0

                for z_b, res_b, idx_b in loader:
                    z_b = z_b.to(self.device, non_blocking=True)
                    res_b = res_b.to(self.device, non_blocking=True)
                    p_b = p_all[idx_b].to(self.device, non_blocking=True)

                    z_star, q = self.S(z_b, res_b)

                    self.opt_S.zero_grad(set_to_none=True)
                    loss = self.loss(p_b, q)
                    loss.backward()
                    self.opt_S.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                self.sch_S.step()
                mean_loss = epoch_loss / max(n_batches, 1)

                y = self._full_assign_and_update_mu(z, res, eval_bs=self.batch_size)

                if y_prev is not None:
                    delta = (y != y_prev).float().mean().item()
                    t.set_postfix(mean_loss=f"{mean_loss:.4f}", delta=f"{delta:.4f}")
                    if delta < 0.0001:
                        break
                else:
                    t.set_postfix(mean_loss=f"{mean_loss:.4f}", delta="nan")

                y_prev = y
                t.update(1)

        with torch.no_grad():
            self.S.eval()
            return self.predict_labels(z, res, eval_bs=self.batch_size)

    @torch.no_grad()
    def generate_z_res(self, data: torch.Tensor):
        self.G.eval()
        x_hat, z, _z_mem = self.G(data, update_mem=False)
        res = data - x_hat.detach()
        return z.detach(), res.detach()

    @torch.no_grad()
    def predict_labels(self, z: torch.Tensor, res: torch.Tensor, eval_bs: int = 1024):
        self.S.eval()

        ds = PairDataset(z.detach().cpu(), res.detach().cpu())
        ld = DataLoader(
            ds,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        preds = []

        for z_b, res_b in ld:
            z_b = z_b.to(self.device, non_blocking=True)
            res_b = res_b.to(self.device, non_blocking=True)

            res_z = self.S.E_delta(res_b)
            z_star = self.S.fusion(z_b, res_z)

            dist2 = torch.sum((z_star.unsqueeze(1) - self.S.mu) ** 2, dim=2)
            q = 1.0 / (1.0 + dist2 / (self.S.alpha + self.S.eps))
            q = q.pow((self.S.alpha + 1.0) / 2.0)
            q = q / (q.sum(dim=1, keepdim=True) + self.S.eps)

            y_b = torch.argmax(q, dim=1)
            preds.append(y_b.cpu())

        return torch.cat(preds, dim=0).numpy()
    
    def save(self, save_path: str | Path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "subtyper_state_dict": self.S.state_dict(),
                "num_types": self.S.num_types,
                "n_genes": self.n_genes,
                "s_configs": self.s_configs,
            },
            save_path,
        )


def sanitize_filename(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[\\/:*?\"<>| ]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def read_prediction_csv(csv_path: str, cell_col: str | None, assay_col: str | None) -> Tuple[pd.DataFrame, str]:
    pred_df = pd.read_csv(csv_path)

    if cell_col is None:
        cell_col = pred_df.columns[0]

    if cell_col not in pred_df.columns:
        raise KeyError(f"`{cell_col}` not found in pred csv. Available columns: {list(pred_df.columns)}")

    pred_df = pred_df.copy()
    pred_df[cell_col] = pred_df[cell_col].astype(str).str.strip()

    if assay_col is not None:
        if assay_col not in pred_df.columns:
            raise KeyError(f"`{assay_col}` not found in pred csv. Available columns: {list(pred_df.columns)}")
        pred_df[assay_col] = pred_df[assay_col].astype(str).str.strip()

    return pred_df, cell_col


def build_configs_for_adata(args, n_genes: int) -> SubtypeConfigs:
    configs = SubtypeConfigs()
    args_dict = vars(args).copy()
    args_dict["n_genes"] = n_genes
    update_configs_with_args(configs, args_dict, None)
    configs.build()
    configs.clear()
    return configs


def run_subtyping_on_adata(
    adata_sub: ad.AnnData,
    generator,
    args,
) -> np.ndarray:
    if adata_sub.n_obs == 0:
        raise ValueError("No cells available for subtyping.")
    if adata_sub.n_obs < args.num_types:
        raise ValueError(
            f"Number of anomaly cells ({adata_sub.n_obs}) is smaller than num_types ({args.num_types})."
        )

    configs = build_configs_for_adata(args, adata_sub.n_vars)
    model = SubtypeModel(generator, args.num_types, configs)
    sublabel = model.train(adata_sub)
    if args.out_model:
        model.save(args.out_model)
    return sublabel


def collect_anomaly_cells(
    adata: ad.AnnData,
    pred_df: pd.DataFrame,
    cell_col: str,
    label_col: str,
    anomaly_label: str,
    target_assay: str = "",
    adata_assay_col: str = "assay",
    csv_assay_col: str = "assay",
) -> Tuple[pd.Index, ad.AnnData]:
    if label_col not in pred_df.columns:
        raise KeyError(f"`{label_col}` not found in pred csv. Available columns: {list(pred_df.columns)}")

    pred_df = pred_df.copy()
    pred_df[label_col] = pred_df[label_col].astype(str).str.strip()

    adata_use = adata
    pred_use = pred_df

    if str(target_assay).strip():
        target_assay = str(target_assay).strip()

        if adata_assay_col not in adata.obs.columns:
            raise KeyError(f"`{adata_assay_col}` not found in adata.obs")
        if csv_assay_col not in pred_df.columns:
            raise KeyError(f"`{csv_assay_col}` not found in pred csv. Available columns: {list(pred_df.columns)}")

        adata_assay = adata.obs[adata_assay_col].astype(str).str.strip()
        pred_use[csv_assay_col] = pred_use[csv_assay_col].astype(str).str.strip()

        adata_use = adata[adata_assay == target_assay].copy()
        pred_use = pred_use[pred_use[csv_assay_col] == target_assay].copy()

        if adata_use.n_obs == 0:
            raise ValueError(f"No cells found in adata for target_assay='{target_assay}'.")
        if pred_use.shape[0] == 0:
            raise ValueError(f"No rows found in pred csv for target_assay='{target_assay}'.")

    pred_mask = pred_use[label_col] == str(anomaly_label).strip()
    anomaly_cells = pd.Index(pred_use.loc[pred_mask, cell_col].astype(str))

    matched_cells = adata_use.obs_names.intersection(anomaly_cells)
    return matched_cells, adata_use


def save_subtype_csv(
    save_path: Path,
    adata_sub: ad.AnnData,
    subtype: np.ndarray,
    assay_col: str = "assay",
):
    out_df = pd.DataFrame({
        "cell_id": adata_sub.obs_names.astype(str),
        "subtype": subtype.astype(int),
    })

    if assay_col in adata_sub.obs.columns:
        out_df.insert(1, "assay", adata_sub.obs[assay_col].astype(str).values)

    out_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RADAR Phase III subtype on predicted anomaly cells."
    )

    configs = SubtypeConfigs()

    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--read_path", type=str, required=True, help="Path to tgt h5ad file")
    data_group.add_argument("--pred_csv", type=str, required=True, help="Prediction csv path")
    data_group.add_argument("--pth_path", type=str, required=True, help="Trained Phase-I generator path")

    data_group.add_argument("--save_dir", type=str, default="subtype_outputs", help="Directory to save outputs")
    data_group.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Output csv path; if empty, auto-save to subtype_{target_assay}.csv or subtype_all.csv"
    )

    data_group.add_argument(
        "--out_model",
        type=str,
        default="",
        help="Path to save trained Subtyper weights, e.g. subtype_model.pth"
    )

    data_group.add_argument(
        "--target_assay",
        type=str,
        default="",
        help="If set, only run subtype on this assay (treat this assay as one tgt dataset, like Fig.7). If empty, use the whole tgt dataset (like Fig.8)."
    )
    data_group.add_argument(
        "--adata_assay_col",
        type=str,
        default="assay",
        help="Assay column name in adata.obs"
    )
    data_group.add_argument(
        "--csv_assay_col",
        type=str,
        default="assay",
        help="Assay column name in prediction csv"
    )

    data_group.add_argument(
        "--cell_col",
        type=str,
        default=None,
        help="Cell ID column in csv; default is the first column"
    )
    data_group.add_argument(
        "--label_col",
        type=str,
        default="pred",
        help="Prediction label column in csv"
    )
    data_group.add_argument(
        "--anomaly_label",
        type=str,
        default="abnormal",
        help="Label value in csv that means anomaly"
    )

    

    s_group = parser.add_argument_group("SubtypeModel Parameters")
    s_group.add_argument("--n_epochs", type=int, default=configs.n_epochs, help="Number of epochs")
    s_group.add_argument("--batch_size", type=int, default=configs.batch_size, help="Batch size")
    s_group.add_argument("--learning_rate", type=float, default=configs.learning_rate, help="Learning rate")
    s_group.add_argument("--weight_decay", type=float, default=configs.weight_decay, help="Weight decay rate")
    s_group.add_argument("--GPU", type=str, default=configs.GPU, help="GPU ID, e.g. cuda:0")
    s_group.add_argument("--n_genes", type=int, default=configs.n_genes, help="Number of genes")
    s_group.add_argument("--num_types", type=int, required=True, help="Number of anomaly subtypes")

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.read_path)
    adata.obs_names = adata.obs_names.astype(str)

    need_assay_col = bool(str(args.target_assay).strip())
    pred_df, resolved_cell_col = read_prediction_csv(
        csv_path=args.pred_csv,
        cell_col=args.cell_col,
        assay_col=args.csv_assay_col if need_assay_col else None,
    )

    print("=============== Runtime Arguments ===============")
    for k, v in vars(args).items():
        print(f"{k} = {v}")
    print(f"resolved_cell_col = {resolved_cell_col}")

    generator = torch.load(args.pth_path, map_location="cpu", weights_only=False)

    anomaly_cells, adata_use = collect_anomaly_cells(
        adata=adata,
        pred_df=pred_df,
        cell_col=resolved_cell_col,
        label_col=args.label_col,
        anomaly_label=args.anomaly_label,
        target_assay=args.target_assay,
        adata_assay_col=args.adata_assay_col,
        csv_assay_col=args.csv_assay_col,
    )

    mode = "single-assay / Fig.7 style" if str(args.target_assay).strip() else "whole-tgt / Fig.8 style"
    print(f"mode = {mode}")
    if str(args.target_assay).strip():
        print(f"target_assay = {args.target_assay}")
        print(f"Cells in selected assay = {adata_use.n_obs}")

    print(f"Number of matched predicted anomaly cells = {len(anomaly_cells)}")

    if len(anomaly_cells) == 0:
        raise ValueError("No anomaly cells were found after matching csv and h5ad.")
    if len(anomaly_cells) < args.num_types:
        raise ValueError(
            f"Number of anomaly cells ({len(anomaly_cells)}) is smaller than num_types ({args.num_types})."
        )

    adata_sub = adata_use[anomaly_cells].copy()
    subtype = run_subtyping_on_adata(
        adata_sub=adata_sub,
        generator=generator,
        args=args,
    )

    if args.save_path:
        out_path = Path(args.save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if str(args.target_assay).strip():
            assay_safe = sanitize_filename(args.target_assay)
            out_path = save_dir / f"subtype_{assay_safe}.csv"
        else:
            out_path = save_dir / "subtype_all.csv"

    save_subtype_csv(
        save_path=out_path,
        adata_sub=adata_sub,
        subtype=subtype,
        assay_col=args.adata_assay_col,
    )
    print(f"Saved: {out_path}")