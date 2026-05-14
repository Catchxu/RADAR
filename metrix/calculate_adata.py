from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score


def _to_dense_float32(X):
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _build_umap_space(
    adata: ad.AnnData,
    X,
    seed: int = 42,
) -> ad.AnnData:
    X = _to_dense_float32(X)

    a = ad.AnnData(
        X=X.copy(),
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )

    sc.pp.pca(a, random_state=seed)
    sc.pp.neighbors(a,random_state=seed)
    sc.tl.umap(a, random_state=seed)
    return a


def _asw_score(X, labels):
    labels = np.asarray(labels).astype(str)
    uniq, cnt = np.unique(labels, return_counts=True)
    if len(uniq) < 2 or np.min(cnt) < 2:
        return np.nan
    return float(silhouette_score(X, labels, metric="euclidean"))


def _ilisi_score(X, batch_labels, k=30, beta=1.0):
    X = np.asarray(X, dtype=np.float64)
    batch_labels = np.asarray(batch_labels).astype(str)

    batches, batch_idx = np.unique(batch_labels, return_inverse=True)
    n_batches = len(batches)
    if n_batches < 2:
        return 1.0

    nn = NearestNeighbors(
        n_neighbors=min(k + 1, X.shape[0]),
        metric="euclidean",
    )
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)

    dists = dists[:, 1:]
    idx = idx[:, 1:]

    vals = []
    for di, neigh in zip(dists, idx):
        w = np.exp(-beta * (di ** 2) / 2.0)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            continue

        p = w / s
        batch_mass = np.bincount(
            batch_idx[neigh],
            weights=p,
            minlength=n_batches,
        ).astype(np.float64)

        denom = np.sum(batch_mass ** 2)
        if denom > 0 and np.isfinite(denom):
            vals.append(np.clip(1.0 / denom, 1.0, float(n_batches)))

    return float(np.mean(vals)) if vals else np.nan


def _batchkl_score(X, batch_labels, k=30, sample_n=100, seed=42):
    X = np.asarray(X, dtype=np.float64)
    batch_labels = np.asarray(batch_labels).astype(str)

    batches, batch_idx = np.unique(batch_labels, return_inverse=True)
    n_batches = len(batches)
    if n_batches < 2:
        return 0.0

    q = np.bincount(batch_idx, minlength=n_batches).astype(np.float64)
    q = q / q.sum()

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(
        X.shape[0],
        size=min(sample_n, X.shape[0]),
        replace=False,
    )

    nn = NearestNeighbors(
        n_neighbors=min(k + 1, X.shape[0]),
        metric="euclidean",
    )
    nn.fit(X)
    neigh = nn.kneighbors(X[sample_idx], return_distance=False)[:, 1:]

    vals = []
    for row in neigh:
        p = np.bincount(batch_idx[row], minlength=n_batches).astype(np.float64)
        p = p / p.sum()
        mask = p > 0
        vals.append(np.sum(p[mask] * np.log((p[mask] + 1e-12) / (q[mask] + 1e-12))))

    return float(np.mean(vals)) if vals else np.nan


def _ari_leiden_score(X, labels, n_neighbors=30, resolution=1.0, seed=42):
    tmp = ad.AnnData(X=np.asarray(X, dtype=np.float32).copy())
    tmp.obs["label"] = pd.Categorical(np.asarray(labels).astype(str))

    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=n_neighbors)
    sc.tl.leiden(
        tmp,
        key_added="leiden",
        resolution=resolution,
        random_state=seed,
    )
    return float(adjusted_rand_score(tmp.obs["label"], tmp.obs["leiden"]))


def evaluate_corrected_adata(
    adata: ad.AnnData,
    batch_key: str = "assay",
    bio_key: str = "cell_type",
    seed: int = 42,
    eval_rep: str = "X_umap",
    k_metric: int = 30,
    ilisi_beta: float = 1.0,
    batchkl_sample_n: int = 100,
    leiden_res: float = 1.0,
) -> dict:

    np.random.seed(seed)

    if batch_key not in adata.obs:
        raise KeyError(f"missing obs['{batch_key}']")
    if bio_key not in adata.obs:
        raise KeyError(f"missing obs['{bio_key}']")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Input AnnData has no cells or no genes.")

    a = _build_umap_space(adata, adata.X, seed=seed)
    X = np.asarray(a.obsm[eval_rep], dtype=np.float64)

    asw_batch = _asw_score(X, a.obs[batch_key])
    asw_bio = _asw_score(X, a.obs[bio_key])

    return {
        f"1-ASW_{batch_key}": (1.0 - asw_batch) if pd.notna(asw_batch) else np.nan,
        f"ASW_{bio_key}": asw_bio,
        f"iLISI({batch_key})": _ilisi_score(
            X,
            a.obs[batch_key],
            k=k_metric,
            beta=ilisi_beta,
        ),
        f"BatchKL({batch_key})": _batchkl_score(
            X,
            a.obs[batch_key],
            k=k_metric,
            sample_n=batchkl_sample_n,
            seed=seed,
        ),
        f"ARI(Leiden vs {bio_key})": _ari_leiden_score(
            X,
            a.obs[bio_key],
            n_neighbors=k_metric,
            resolution=leiden_res,
            seed=seed,
        ),
        "n_cells": int(a.n_obs),
    }

def evaluate_adata_multi_seed(
    adata: ad.AnnData,
    n_runs: int = 100,
    seeds: list[int] | None = None,
    batch_key: str = "assay",
    bio_key: str = "cell_type",
    eval_rep: str = "X_umap",
    k_metric: int = 30,
    ilisi_beta: float = 1.0,
    batchkl_sample_n: int = 100,
    leiden_res: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Repeat evaluation on the same AnnData with different random seeds.

    Returns
    -------
    runs_df
        One row per seed.
    summary_df
        Summary statistics across seeds for the 5 metrics.
    """
    if seeds is None:
        seeds = list(range(n_runs))
    else:
        seeds = [int(s) for s in seeds]
        if len(seeds) == 0:
            raise ValueError("seeds is empty.")
        n_runs = len(seeds)

    rows = []
    for seed in seeds:
        res = evaluate_corrected_adata(
            adata=adata,
            batch_key=batch_key,
            bio_key=bio_key,
            seed=seed,
            eval_rep=eval_rep,
            k_metric=k_metric,
            ilisi_beta=ilisi_beta,
            batchkl_sample_n=batchkl_sample_n,
            leiden_res=leiden_res,
        )
        row = {"seed": int(seed)}
        row.update(res)
        rows.append(row)

    runs_df = pd.DataFrame(rows)

    metric_cols = [
        f"1-ASW_{batch_key}",
        f"ASW_{bio_key}",
        f"iLISI({batch_key})",
        f"BatchKL({batch_key})",
        f"ARI(Leiden vs {bio_key})",
    ]

    summary_rows = []
    for col in metric_cols:
        x = pd.to_numeric(runs_df[col], errors="coerce")
        summary_rows.append(
            {
                "metric": col,
                "mean": float(x.mean()),
                "std": float(x.std(ddof=1)),
                "min": float(x.min()),
                "median": float(x.median()),
                "max": float(x.max()),
                "q05": float(x.quantile(0.05)),
                "q95": float(x.quantile(0.95)),
                "n_runs": int(x.notna().sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return runs_df, summary_df


def save_multi_seed_evaluation(
    adata: ad.AnnData,
    out_runs_csv: str,
    out_summary_csv: str,
    n_runs: int = 100,
    seeds: list[int] | None = None,
    batch_key: str = "assay",
    bio_key: str = "cell_type",
    eval_rep: str = "X_umap",
    k_metric: int = 30,
    ilisi_beta: float = 1.0,
    batchkl_sample_n: int = 100,
    leiden_res: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs_df, summary_df = evaluate_adata_multi_seed(
        adata=adata,
        n_runs=n_runs,
        seeds=seeds,
        batch_key=batch_key,
        bio_key=bio_key,
        eval_rep=eval_rep,
        k_metric=k_metric,
        ilisi_beta=ilisi_beta,
        batchkl_sample_n=batchkl_sample_n,
        leiden_res=leiden_res,
    )

    runs_df.to_csv(out_runs_csv, index=False)
    summary_df.to_csv(out_summary_csv, index=False)
    return runs_df, summary_df