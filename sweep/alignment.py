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


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False   
    



def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        num_workers: int = 0,
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

        self.num_workers = num_workers
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
    def _to_dense_float32(X) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    @staticmethod
    def _check_ref_tgt_compatible(ref: ad.AnnData, tgt: ad.AnnData):
        if ref.n_obs == 0 or ref.n_vars == 0:
            raise ValueError("ref has no cells or no genes.")
        if tgt.n_obs == 0 or tgt.n_vars == 0:
            raise ValueError("tgt has no cells or no genes.")
        if ref.n_vars != tgt.n_vars:
            raise ValueError(f"Feature mismatch: ref.n_vars={ref.n_vars}, tgt.n_vars={tgt.n_vars}")
        if not np.array_equal(np.asarray(ref.var_names), np.asarray(tgt.var_names)):
            raise ValueError("Feature name/order mismatch between ref and tgt.")

    @staticmethod
    def _matches_value(series: pd.Series, target_value: str) -> np.ndarray:
        series_str = series.astype(str).str.strip()
        target_str = str(target_value).strip()
        str_mask = (series_str == target_str).to_numpy()

        series_num = pd.to_numeric(series, errors="coerce")
        try:
            target_num = float(target_value)
        except (TypeError, ValueError):
            target_num = None

        if target_num is None:
            return str_mask

        numeric_mask = np.zeros(len(series_num), dtype=bool)
        valid = series_num.notna().to_numpy()
        if valid.any():
            numeric_mask[valid] = np.isclose(series_num.to_numpy(dtype=np.float64)[valid], target_num)
        return numeric_mask | str_mask

    def _infer_tgt_train_state(self, tgt: ad.AnnData) -> tuple[np.ndarray, pd.Series | None]:
        if self.tgt_normal_csv is None:
            if self.disease_key not in tgt.obs:
                raise KeyError(f"`{self.disease_key}` not found in tgt.obs")
            state = np.where(
                tgt.obs[self.disease_key].astype(str) == str(self.normal_value),
                "normal",
                "disease",
            )
            return state, None

        if not self.tgt_normal_csv.exists():
            raise FileNotFoundError(f"tgt_normal_csv not found: {self.tgt_normal_csv}")

        df = pd.read_csv(self.tgt_normal_csv)
        if self.tgt_normal_id_col not in df.columns:
            raise KeyError(
                f"`{self.tgt_normal_id_col}` not found in {self.tgt_normal_csv}. "
                f"Available columns: {list(df.columns)}"
            )
        if self.tgt_normal_label_col not in df.columns:
            raise KeyError(
                f"`{self.tgt_normal_label_col}` not found in {self.tgt_normal_csv}. "
                f"Available columns: {list(df.columns)}"
            )

        df = df[[self.tgt_normal_id_col, self.tgt_normal_label_col]].copy()
        df[self.tgt_normal_id_col] = df[self.tgt_normal_id_col].astype(str)

        duplicated = df[self.tgt_normal_id_col].duplicated(keep=False)
        if duplicated.any():
            dup_ids = df.loc[duplicated, self.tgt_normal_id_col].tolist()[:10]
            raise ValueError(
                "Duplicate cell ids found in tgt_normal_csv. "
                f"Examples: {dup_ids}"
            )

        pred_labels = df.set_index(self.tgt_normal_id_col)[self.tgt_normal_label_col]
        pred_labels.index = pred_labels.index.astype(str)

        tgt_ids = pd.Index(tgt.obs_names.astype(str))
        missing = tgt_ids.difference(pred_labels.index)
        if len(missing) > 0:
            raise ValueError(
                "Some target cells are missing in tgt_normal_csv. "
                f"Missing count={len(missing)}. Examples: {missing[:10].tolist()}"
            )

        pred_labels = pred_labels.reindex(tgt_ids)
        normal_mask = self._matches_value(pred_labels, self.tgt_normal_value)
        state = np.where(normal_mask, "normal", "disease")
        return state, pred_labels

    def _annotate_tgt_state(self, tgt: ad.AnnData) -> ad.AnnData:
        state, pred_labels = self._infer_tgt_train_state(tgt)
        tgt.obs["phase1_train_state"] = state
        tgt.obs["phase1_state_source"] = self.tgt_state_source_
        if pred_labels is not None:
            tgt.obs["phase1_pred_label"] = pred_labels.to_numpy()
            tgt.obs["phase1_pred_is_normal"] = (state == "normal")
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

        self._check_ref_tgt_compatible(ref, tgt)
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
                num_workers=self.num_workers,
                drop_last=False,
                generator=data_loader_generator,
                worker_init_fn=seed_worker,
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
            num_workers=self.num_workers,
            drop_last=False,
            worker_init_fn=seed_worker,
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

        for _ in range(self.d_steps):
            c_tgt = self._sample_target_domains(c_src)

            self.opt_D.zero_grad()
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

        for _ in range(self.g_steps):
            c_tgt = self._sample_target_domains(c_src)

            self.opt_G.zero_grad()
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

        self.domain_names_ = [str(x) for x in domains.cat.categories]
        self.state_names_ = [str(x) for x in states.cat.categories]

        x = self._to_dense_float32(train_adata.X)

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

            epoch_iterator.set_postfix(
                d_loss=f"{epoch_log.get('d_loss', 0.0):.4f}",
                d_adv=f"{epoch_log.get('d_adv', 0.0):.4f}",
                d_bat=f"{epoch_log.get('d_batch_real', 0.0):.4f}",
                d_sta=f"{epoch_log.get('d_state_real', 0.0):.4f}",
                g_loss=f"{epoch_log.get('g_loss', 0.0):.4f}",
                g_adv=f"{epoch_log.get('g_adv', 0.0):.4f}",
                g_bat=f"{epoch_log.get('g_batch_fake', 0.0):.4f}",
                g_sta=f"{epoch_log.get('g_state_fake', 0.0):.4f}",
                g_rec=f"{epoch_log.get('g_rec', 0.0):.4f}",
                g_id=f"{epoch_log.get('g_id', 0.0):.4f}",
            )

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

        self._check_ref_tgt_compatible(ref, tgt)

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
        copy: bool = True,
    ) -> ad.AnnData:
        if self.G is None:
            raise RuntimeError("Please fit the model first.")
        if self.domain_names_ is None or self.gene_names_ is None:
            raise RuntimeError("Model metadata is missing.")

        adata = self._load_adata(adata_or_path)
        if copy:
            adata = adata.copy()

        if self.batch_key not in adata.obs:
            raise KeyError(f"`{self.batch_key}` not found in adata.obs")
        if adata.n_vars != self.input_dim:
            raise ValueError(
                f"Feature mismatch: adata.n_vars={adata.n_vars}, model.input_dim={self.input_dim}"
            )
        if not np.array_equal(np.asarray(adata.var_names), np.asarray(self.gene_names_)):
            raise ValueError("Feature name/order mismatch between adata and training data.")

        target_domain = self._resolve_target_domain(target_domain)
        domain_to_idx = {name: i for i, name in enumerate(self.domain_names_)}
        target_idx = domain_to_idx[target_domain]

        self.G.eval()
        x = self._to_dense_float32(adata.X)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(x)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        aligned_chunks = []
        for (x_batch,) in loader:
            x_batch = x_batch.to(self.device, non_blocking=True)
            c_tgt = torch.full(
                (x_batch.shape[0],),
                fill_value=target_idx,
                dtype=torch.long,
                device=self.device,
            )
            x_aligned = self.G(x_batch, c_tgt)
            aligned_chunks.append(x_aligned.detach().cpu())

        x_aligned = torch.cat(aligned_chunks, dim=0).numpy()
        adata.X = x_aligned
        adata.obs[f"{self.batch_key}_aligned_target"] = target_domain
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
            copy=False,
        )
        corrected.uns["available_domains"] = list(self.domain_names_)
        corrected.uns["ref_domain_name"] = str(self.ref_domain_name_)
        corrected.uns["target_domain_requested"] = str(target_domain)
        corrected.uns["target_domain_actual"] = str(corrected.obs[f"{self.batch_key}_aligned_target"].iloc[0])
        corrected.uns["tgt_state_source"] = self.tgt_state_source_
        corrected.uns["tgt_normal_csv"] = str(self.tgt_normal_csv) if self.tgt_normal_csv is not None else ""
        corrected.uns["tgt_normal_label_col"] = self.tgt_normal_label_col
        corrected.uns["tgt_normal_value"] = self.tgt_normal_value
        corrected.uns["seed"] = int(self.seed)
        return corrected

    def save(self, save_path: str | Path):
        if self.G is None:
            raise RuntimeError("Please fit the model first.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "generator_state_dict": self.G.state_dict(),
                "discriminator_state_dict": self.D.state_dict(),
                "domain_names": self.domain_names_,
                "state_names": self.state_names_,
                "input_dim": self.input_dim,
                "ref_domain_name": self.ref_domain_name_,
                "gene_names": list(self.gene_names_) if self.gene_names_ is not None else None,
                "batch_key": self.batch_key,
                "disease_key": self.disease_key,
                "normal_value": self.normal_value,
                "d_steps": self.d_steps,
                "g_steps": self.g_steps,
                "domain_balance_power": self.domain_balance_power,
                "history": self.history,
                "seed": self.seed,
                "tgt_state_source": self.tgt_state_source_,
                "tgt_normal_csv": str(self.tgt_normal_csv) if self.tgt_normal_csv is not None else "",
                "tgt_normal_id_col": self.tgt_normal_id_col,
                "tgt_normal_label_col": self.tgt_normal_label_col,
                "tgt_normal_value": self.tgt_normal_value,
            },
            save_path,
        )

    def save_history_csv(self, csv_path: str | Path):
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.history).to_csv(csv_path, index=False)


def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--ref_h5ad", type=str, required=True)
    p.add_argument("--tgt_h5ad", type=str, required=True)
    p.add_argument("--out_h5ad", type=str, default="")
    p.add_argument("--target_domain", type=str, required=True)

    p.add_argument("--out_model", type=str, default="")
    p.add_argument("--out_history_csv", type=str, default="")

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

    p.add_argument("--num_workers", type=int, default=0)
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
        num_workers=args.num_workers,
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
    if args.out_history_csv:
        model.save_history_csv(args.out_history_csv)

    print("available_domains:", model.domain_names_)
    print("ref_domain_name:", model.ref_domain_name_)
    print("target_domain_requested:", args.target_domain)
    print("target_domain_actual:", corrected.uns["target_domain_actual"])
    print("d_steps:", model.d_steps)
    print("g_steps:", model.g_steps)
    print("domain_balance_power:", model.domain_balance_power)
    print("tgt_state_source:", model.tgt_state_source_)
    print("tgt_normal_csv:", str(model.tgt_normal_csv) if model.tgt_normal_csv is not None else "")
    print("tgt_normal_label_col:", model.tgt_normal_label_col)
    print("tgt_normal_value:", model.tgt_normal_value)
    print("seed:", model.seed)


if __name__ == "__main__":
    main()
