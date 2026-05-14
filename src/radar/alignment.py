from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from .model import Generator_phase2, Discriminator_phase2
from .loss import phase2RpLoss
from .utils import seed_everything, update_configs_with_args, set_requires_grad


class CorrectModel:
    def __init__(
        self,
        batch_key: str = "assay",
        disease_key: str = "disease",
        normal_value: str = "normal",
        ref_domain_prefix: str = "REF__",
        epochs: int = 100,
        batch_size: int = 128,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        hidden_dim: int = 512,
        cond_dim: int = 128,
        g_num_blocks: int = 6,
        d_num_blocks: int = 3,
        g_dropout: float = 0.1,
        d_dropout: float = 0.0,
        use_change_gate: bool = False,
        lambda_batch: float = 1.0,
        lambda_state: float = 1.0,
        lambda_rec: float = 3.0,
        lambda_id: float = 1.0,
        d_steps: int = 1,
        g_steps: int = 2,
        domain_balance_power: float = 0.5,
        device: str | torch.device | None = None,
        seed: int = 42,
        tgt_normal_csv: str | Path | None = None,
        tgt_normal_id_col: str = "Unnamed: 0",
        tgt_normal_label_col: str = "label",
        tgt_normal_value: str = "0",
    ):
        self.batch_key = batch_key
        self.disease_key = disease_key
        self.normal_value = normal_value
        self.ref_domain_prefix = ref_domain_prefix

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.g_num_blocks = g_num_blocks
        self.d_num_blocks = d_num_blocks
        self.g_dropout = g_dropout
        self.d_dropout = d_dropout
        self.use_change_gate = use_change_gate

        self.lambda_batch = lambda_batch
        self.lambda_state = lambda_state
        self.lambda_rec = lambda_rec
        self.lambda_id = lambda_id

        self.d_steps = d_steps
        self.g_steps = g_steps
        self.domain_balance_power = domain_balance_power

        self.seed = seed

        self.tgt_normal_csv = None if tgt_normal_csv in (None, "") else Path(tgt_normal_csv)
        self.tgt_normal_id_col = tgt_normal_id_col
        self.tgt_normal_label_col = tgt_normal_label_col
        self.tgt_normal_value = str(tgt_normal_value)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.G = None
        self.D = None
        self.opt_G = None
        self.opt_D = None
        self.criterion = None

        self.history = []
        self.input_dim = None
        self.domain_names_ = None
        self.ref_domain_name_ = None
        self.gene_names_ = None
        self.state_names_ = None
        self.train_batch_key_ = None
        self.train_state_key_ = None
        self.tgt_state_source_ = "phase1_csv" if self.tgt_normal_csv is not None else "obs_disease_key"

    @staticmethod
    def _load_adata(adata_or_path: str | Path | ad.AnnData) -> ad.AnnData:
        if isinstance(adata_or_path, ad.AnnData):
            return adata_or_path
        return ad.read_h5ad(str(adata_or_path))


    @staticmethod
    def _matches_value(series: pd.Series, target_value: str) -> np.ndarray:
        return series.astype(str).str.strip().eq(str(target_value).strip()).to_numpy()

    def _infer_tgt_train_state(self, tgt: ad.AnnData) -> tuple[np.ndarray, pd.Series | None]:
        # Case 1: no Phase-I prediction CSV; use tgt.obs[disease_key]
        if self.tgt_normal_csv is None:
            if self.disease_key not in tgt.obs:
                raise KeyError(f"`{self.disease_key}` not found in tgt.obs")

            state = np.where(
                tgt.obs[self.disease_key].astype(str) == str(self.normal_value),
                "normal",
                "disease",
            )
            return state, None

        # Case 2: use Phase-I prediction CSV
        df = pd.read_csv(self.tgt_normal_csv)

        id_col = self.tgt_normal_id_col
        label_col = self.tgt_normal_label_col

        for col in [id_col, label_col]:
            if col not in df.columns:
                raise KeyError(f"`{col}` not found in {self.tgt_normal_csv}. Available columns: {list(df.columns)}")

        df = df[[id_col, label_col]].copy()
        df[id_col] = df[id_col].astype(str)

        pred_labels = df.set_index(id_col)[label_col]

        tgt_ids = pd.Index(tgt.obs_names.astype(str))

        if not tgt_ids.isin(pred_labels.index).all():
            missing = tgt_ids.difference(pred_labels.index)
            raise ValueError(
                f"Some target cells are missing in tgt_normal_csv. "
                f"Missing count={len(missing)}. Examples: {missing[:10].tolist()}"
            )

        pred_labels = pred_labels.reindex(tgt_ids)

        normal_mask = self._matches_value(pred_labels, self.tgt_normal_value)
        state = np.where(normal_mask, "normal", "disease")

        return state, pred_labels

    def _annotate_tgt_state(self, tgt: ad.AnnData) -> ad.AnnData:
        state, pred_labels = self._infer_tgt_train_state(tgt)
        tgt.obs["phase1_train_state"] = state
        return tgt

    def _prepare_train_adata(
        self,
        ref_adata_or_path: str | Path | ad.AnnData,
        tgt_adata_or_path: str | Path | ad.AnnData,
    ) -> ad.AnnData:
        ref = self._load_adata(ref_adata_or_path).copy()
        tgt = self._load_adata(tgt_adata_or_path).copy()

        if self.batch_key not in tgt.obs:
            raise KeyError(f"`{self.batch_key}` not found in tgt.obs")

        self.gene_names_ = ref.var_names.copy()

        ref_domain_name = f"{self.ref_domain_prefix}ref"
        tgt_domains = set(tgt.obs[self.batch_key].astype(str).tolist())
        while ref_domain_name in tgt_domains:
            ref_domain_name = f"{self.ref_domain_prefix}{ref_domain_name}"

        train_batch_key = "__train_domain__"
        train_state_key = "__train_state__"

        ref.obs[train_batch_key] = ref_domain_name
        tgt.obs[train_batch_key] = tgt.obs[self.batch_key].astype(str).values

        ref.obs[train_state_key] = "normal"
        tgt = self._annotate_tgt_state(tgt)
        tgt.obs[train_state_key] = tgt.obs["phase1_train_state"].astype(str).values

        train_adata = ad.concat([ref, tgt], axis=0, join="outer", merge="same")
        train_adata = train_adata[:, ref.var_names].copy()

        self.ref_domain_name_ = ref_domain_name
        self.train_batch_key_ = train_batch_key
        self.train_state_key_ = train_state_key
        return train_adata

    def _init_model(self, num_domains: int, input_dim: int):
        self.input_dim = input_dim

        self.G = Generator_phase2(
            input_dim=input_dim,
            num_batches=num_domains,
            hidden_dim=self.hidden_dim,
            cond_dim=self.cond_dim,
            num_blocks=self.g_num_blocks,
            dropout=self.g_dropout,
            use_change_gate=self.use_change_gate,
        ).to(self.device)

        self.D = Discriminator_phase2(
            input_dim=input_dim,
            num_batches=num_domains,
            hidden_dim=self.hidden_dim,
            num_blocks=self.d_num_blocks,
            dropout=self.d_dropout,
            num_states=2,
        ).to(self.device)

        self.criterion = phase2RpLoss(
            lambda_batch=self.lambda_batch,
            lambda_state=self.lambda_state,
            lambda_rec=self.lambda_rec,
            lambda_id=self.lambda_id,
        )

        self.opt_G = torch.optim.AdamW(self.G.parameters(), lr=self.lr_g)
        self.opt_D = torch.optim.AdamW(self.D.parameters(), lr=self.lr_d)

    def _sample_target_domains(self, c_src: torch.Tensor) -> torch.Tensor:
        c_tgt = torch.randint(
            low=0,
            high=len(self.domain_names_),
            size=c_src.shape,
            device=self.device,
        )
        same = c_tgt == c_src
        if same.any():
            c_tgt[same] = (c_tgt[same] + 1) % len(self.domain_names_)
        return c_tgt

    def _build_train_loader(
        self,
        x: np.ndarray,
        c: np.ndarray,
        s: np.ndarray,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(c),
            torch.from_numpy(s),
        )

        data_loader_generator = torch.Generator().manual_seed(self.seed)

        if self.domain_balance_power <= 0:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                generator=data_loader_generator,
            )

        domain_counts = np.bincount(c, minlength=len(self.domain_names_)).astype(np.float64)
        domain_weights = 1.0 / np.power(np.maximum(domain_counts, 1.0), self.domain_balance_power)
        sample_weights = domain_weights[c]
        sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=False,
        )

    def _train_step(
        self,
        x_real: torch.Tensor,
        c_src: torch.Tensor,
        s_src: torch.Tensor,
    ) -> dict:
        self.G.train()
        self.D.train()

        logs_d_all = []
        logs_g_all = []

        set_requires_grad(self.D, True)
        set_requires_grad(self.G, False)

        for _ in range(self.d_steps):
            c_tgt = self._sample_target_domains(c_src)

            self.opt_D.zero_grad(set_to_none=True)

            d_loss, d_log = self.criterion.discriminator_loss(
                generator=self.G,
                discriminator=self.D,
                x_real=x_real,
                c_src=c_src,
                c_tgt=c_tgt,
                s_src=s_src,
            )

            d_loss.backward()
            self.opt_D.step()

            logs_d_all.append({k: float(v) for k, v in d_log.items()})

        set_requires_grad(self.D, False)
        set_requires_grad(self.G, True)

        for _ in range(self.g_steps):
            c_tgt = self._sample_target_domains(c_src)

            self.opt_G.zero_grad(set_to_none=True)

            g_loss, g_log = self.criterion.generator_loss(
                generator=self.G,
                discriminator=self.D,
                x_real=x_real,
                c_src=c_src,
                c_tgt=c_tgt,
                s_src=s_src,
            )

            g_loss.backward()
            self.opt_G.step()

            logs_g_all.append({k: float(v) for k, v in g_log.items()})
        set_requires_grad(self.D, True)
        set_requires_grad(self.G, True)

        logs = {}

        if logs_d_all:
            for k in logs_d_all[0].keys():
                logs[k] = float(np.mean([x[k] for x in logs_d_all]))

        if logs_g_all:
            for k in logs_g_all[0].keys():
                logs[k] = float(np.mean([x[k] for x in logs_g_all]))

        return logs

    def fit(
        self,
        ref_adata_or_path: str | Path | ad.AnnData,
        tgt_adata_or_path: str | Path | ad.AnnData,
        show_progress: bool = True,
    ):
        seed_everything(self.seed)

        train_adata = self._prepare_train_adata(
            ref_adata_or_path=ref_adata_or_path,
            tgt_adata_or_path=tgt_adata_or_path,
        )
    
        domains = train_adata.obs[self.train_batch_key_].astype("category")
        states = train_adata.obs[self.train_state_key_].astype("category")

        c = domains.cat.codes.to_numpy(np.int64)
        s = states.cat.codes.to_numpy(np.int64)
        x = train_adata.X
        self.domain_names_ = [str(x) for x in domains.cat.categories]
        self.state_names_ = [str(x) for x in states.cat.categories]


        self._init_model(num_domains=len(self.domain_names_), input_dim=train_adata.n_vars)
        loader = self._build_train_loader(x=x, c=c, s=s)

        self.history = []
        epoch_iterator = tqdm(
            range(1, self.epochs + 1),
            desc="Training",
            disable=not show_progress,
        )

        for epoch in epoch_iterator:
            running = {}
            n_steps = 0

            for x_real, c_src, s_src in loader:
                x_real = x_real.to(self.device, non_blocking=True)
                c_src = c_src.to(self.device, non_blocking=True)
                s_src = s_src.to(self.device, non_blocking=True)

                logs = self._train_step(
                    x_real=x_real,
                    c_src=c_src,
                    s_src=s_src,
                )

                for k, v in logs.items():
                    running[k] = running.get(k, 0.0) + v
                n_steps += 1

            epoch_log = {"epoch": epoch}
            for k, v in running.items():
                epoch_log[k] = v / max(n_steps, 1)
            self.history.append(epoch_log)

        return self

    def build_joint_adata(
        self,
        ref_adata_or_path: str | Path | ad.AnnData,
        tgt_adata_or_path: str | Path | ad.AnnData,
        ref_bio_key: str,
        tgt_bio_key: str,
    ) -> ad.AnnData:
        if self.ref_domain_name_ is None or self.gene_names_ is None:
            raise RuntimeError("Please fit the model first.")

        ref = self._load_adata(ref_adata_or_path).copy()
        tgt = self._load_adata(tgt_adata_or_path).copy()

        if ref_bio_key not in ref.obs:
            raise KeyError(f"`{ref_bio_key}` not found in ref.obs")
        if tgt_bio_key not in tgt.obs:
            raise KeyError(f"`{tgt_bio_key}` not found in tgt.obs")
        if self.batch_key not in tgt.obs:
            raise KeyError(f"`{self.batch_key}` not found in tgt.obs")

        ref.obs[self.batch_key] = self.ref_domain_name_
        ref.obs["assay_eval"] = self.ref_domain_name_
        tgt.obs["assay_eval"] = tgt.obs[self.batch_key].astype(str).values

        ref.obs["bio_label"] = ref.obs[ref_bio_key].astype(str).values
        tgt.obs["bio_label"] = tgt.obs[tgt_bio_key].astype(str).values

        ref.obs["dataset"] = "ref"
        tgt.obs["dataset"] = "tgt"
        tgt = self._annotate_tgt_state(tgt)

        joint = ad.concat([ref, tgt], axis=0, join="outer", merge="same")
        joint = joint[:, self.gene_names_].copy()
        return joint

    def _resolve_target_domain(self, target_domain: str) -> str:
        if self.domain_names_ is None or self.ref_domain_name_ is None:
            raise RuntimeError("Please fit the model first.")
        if target_domain == "ref":
            return self.ref_domain_name_
        if target_domain not in self.domain_names_:
            raise KeyError(
                f"Unknown target_domain: {target_domain}. "
                f"Available: {self.domain_names_} (or use 'ref')"
            )
        return target_domain

    @torch.no_grad()
    def align_adata_to_domain(
        self,
        adata_or_path: str | Path | ad.AnnData,
        target_domain: str,
        batch_size: int = 1024,
    ) -> ad.AnnData:
        adata = self._load_adata(adata_or_path)

        target_domain = self._resolve_target_domain(target_domain)
        target_idx = {name: i for i, name in enumerate(self.domain_names_)}[target_domain]

        x = adata.X

        loader = DataLoader(
            TensorDataset(torch.from_numpy(x)),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.G.eval()

        aligned_chunks = []
        for (x_batch,) in loader:
            x_batch = x_batch.to(self.device)

            c_tgt = torch.full(
                (x_batch.shape[0],),
                target_idx,
                dtype=torch.long,
                device=self.device,
            )

            x_aligned = self.G(x_batch, c_tgt)
            aligned_chunks.append(x_aligned.cpu())

        adata.X = torch.cat(aligned_chunks, dim=0).numpy()
        return adata

    @torch.no_grad()
    def translate_joint(
        self,
        ref_adata_or_path: str | Path | ad.AnnData,
        tgt_adata_or_path: str | Path | ad.AnnData,
        target_domain: str,
        ref_bio_key: str,
        tgt_bio_key: str,
        batch_size: int = 1024,
    ) -> ad.AnnData:
        joint = self.build_joint_adata(
            ref_adata_or_path=ref_adata_or_path,
            tgt_adata_or_path=tgt_adata_or_path,
            ref_bio_key=ref_bio_key,
            tgt_bio_key=tgt_bio_key,
        )
        corrected = self.align_adata_to_domain(
            adata_or_path=joint,
            target_domain=target_domain,
            batch_size=batch_size,
        )
        return corrected

    def save(self, save_path: str | Path):
        if self.G is None:
            raise RuntimeError("Please fit the model first.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
        {
            "generator_state_dict": self.G.state_dict(),

            "input_dim": self.input_dim,
            "num_domains": len(self.domain_names_),
            "hidden_dim": self.hidden_dim,
            "cond_dim": self.cond_dim,
            "g_num_blocks": self.g_num_blocks,
            "use_change_gate": self.use_change_gate,

            "domain_names": self.domain_names_,
            "ref_domain_name": self.ref_domain_name_,
        },
            save_path,
        )


def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--ref_h5ad", type=str, required=True)
    p.add_argument("--tgt_h5ad", type=str, required=True)
    p.add_argument("--out_h5ad", type=str, default="")
    p.add_argument("--target_domain", type=str, required=True)

    p.add_argument("--out_model", type=str, default="")

    p.add_argument("--batch_key", type=str, default="assay")
    p.add_argument("--disease_key", type=str, default="disease")
    p.add_argument("--normal_value", type=str, default="normal")
    p.add_argument("--ref_bio_key", type=str, default="cell_type")
    p.add_argument("--tgt_bio_key", type=str, default="cell_state_label")
    p.add_argument("--ref_domain_prefix", type=str, default="REF__")

    p.add_argument("--tgt_normal_csv", type=str, default="")
    p.add_argument("--tgt_normal_id_col", type=str, default="Unnamed: 0")
    p.add_argument("--tgt_normal_label_col", type=str, default="label")
    p.add_argument("--tgt_normal_value", type=str, default="0")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--align_batch_size", type=int, default=1024)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--cond_dim", type=int, default=128)
    p.add_argument("--g_num_blocks", type=int, default=6)
    p.add_argument("--d_num_blocks", type=int, default=3)
    p.add_argument("--g_dropout", type=float, default=0.1)
    p.add_argument("--d_dropout", type=float, default=0.0)
    p.add_argument("--use_change_gate", action="store_true")
    p.add_argument("--lambda_batch", type=float, default=1.0)
    p.add_argument("--lambda_state", type=float, default=1.0)
    p.add_argument("--lambda_rec", type=float, default=3.0)
    p.add_argument("--lambda_id", type=float, default=1.0)

    p.add_argument("--d_steps", type=int, default=1)
    p.add_argument("--g_steps", type=int, default=2)
    p.add_argument("--domain_balance_power", type=float, default=0.5)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_progress", action="store_true")

    return p


def main():
    args = build_argparser().parse_args()

    model = CorrectModel(
        batch_key=args.batch_key,
        disease_key=args.disease_key,
        normal_value=args.normal_value,
        ref_domain_prefix=args.ref_domain_prefix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        g_num_blocks=args.g_num_blocks,
        d_num_blocks=args.d_num_blocks,
        g_dropout=args.g_dropout,
        d_dropout=args.d_dropout,
        use_change_gate=args.use_change_gate,
        lambda_batch=args.lambda_batch,
        lambda_state=args.lambda_state,
        lambda_rec=args.lambda_rec,
        lambda_id=args.lambda_id,
        d_steps=args.d_steps,
        g_steps=args.g_steps,
        domain_balance_power=args.domain_balance_power,
        device=args.device,
        seed=args.seed,
        tgt_normal_csv=args.tgt_normal_csv,
        tgt_normal_id_col=args.tgt_normal_id_col,
        tgt_normal_label_col=args.tgt_normal_label_col,
        tgt_normal_value=args.tgt_normal_value,
    )

    model.fit(
        ref_adata_or_path=args.ref_h5ad,
        tgt_adata_or_path=args.tgt_h5ad,
        show_progress=not args.no_progress,
    )

    corrected = model.translate_joint(
        ref_adata_or_path=args.ref_h5ad,
        tgt_adata_or_path=args.tgt_h5ad,
        target_domain=args.target_domain,
        ref_bio_key=args.ref_bio_key,
        tgt_bio_key=args.tgt_bio_key,
        batch_size=args.align_batch_size,
    )

    if args.out_h5ad:
        out_h5ad = Path(args.out_h5ad)
        out_h5ad.parent.mkdir(parents=True, exist_ok=True)
        corrected.write_h5ad(out_h5ad)
        print("saved_h5ad:", out_h5ad)

    if args.out_model:
        model.save(args.out_model)

    print("available_domains:", model.domain_names_)
    print("ref_domain_name:", model.ref_domain_name_)
    print("target_domain_requested:", args.target_domain)
    print("d_steps:", model.d_steps)
    print("g_steps:", model.g_steps)
    print("domain_balance_power:", model.domain_balance_power)
    print("tgt_state_source:", model.tgt_state_source_)
    print("tgt_normal_csv:", str(model.tgt_normal_csv) if model.tgt_normal_csv is not None else "")
    print("tgt_normal_label_col:", model.tgt_normal_label_col)
    print("tgt_normal_value:", model.tgt_normal_value)


if __name__ == "__main__":
    main()
