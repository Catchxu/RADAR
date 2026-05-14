import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, silhouette_score


proj_root = Path(__file__).resolve().parent.parent
CORR_PATH = proj_root / "data" / "corrected_tgt_ref.h5ad"
OUTDIR = proj_root / "metrix" / "phase2_metrics"
os.makedirs(OUTDIR, exist_ok=True)

ASSAY_KEY = "assay_eval"
CELLTYPE_KEY = "bio_label"


SEED = 42
N_PCS = 50
N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.3
EVAL_REP = "X_umap_eval"


K_METRIC = 30
ILISI_BETA = 1.0
BATCHKL_SAMPLE_N = 100
LEIDEN_RES = 1.0



def to_dense_float32(X):
    if sp.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    return np.asarray(X, dtype=np.float32)



def build_independent_eval_space(adata, X, n_pcs=50, n_neighbors=15, min_dist=0.3, seed=42):
    adata_eval = adata.copy()
    adata_eval.X = X.copy()

    sc.pp.pca(adata_eval, n_comps=n_pcs, random_state=seed)
    sc.pp.neighbors(
        adata_eval,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        random_state=seed,
    )
    sc.tl.umap(adata_eval, min_dist=min_dist, random_state=seed)

    adata_eval.obsm["X_pca_eval"] = adata_eval.obsm["X_pca"].copy()
    adata_eval.obsm["X_umap_eval"] = adata_eval.obsm["X_umap"].copy()
    return adata_eval


def asw_on_embedding(X, labels):
    labels = np.asarray(labels).astype(str)
    uniq, cnt = np.unique(labels, return_counts=True)
    if len(uniq) < 2 or np.min(cnt) < 2:
        return float("nan")
    return float(silhouette_score(X, labels, metric="euclidean"))


def ilisi_on_embedding_paper(X, batch_labels, k=30, beta=1.0):
    X = np.asarray(X, dtype=np.float64)
    batch_labels = np.asarray(batch_labels).astype(str)

    n = X.shape[0]
    if n < 2:
        return float("nan")

    batches, batch_idx = np.unique(batch_labels, return_inverse=True)
    n_batches = len(batches)
    if n_batches < 2:
        return 1.0

    n_neighbors = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)

    dists = dists[:, 1:]
    idx = idx[:, 1:]

    lisi_vals = []
    for i in range(n):
        di = dists[i]
        neigh = idx[i]

        w = np.exp(-beta * (di ** 2) / 2.0)
        w_sum = w.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            continue
        p = w / w_sum

        s = np.bincount(batch_idx[neigh], weights=p, minlength=n_batches).astype(np.float64)

        denom = np.sum(s ** 2)
        if denom <= 0 or not np.isfinite(denom):
            continue

        lisi_i = 1.0 / denom
        lisi_i = np.clip(lisi_i, 1.0, float(n_batches))
        lisi_vals.append(lisi_i)

    if len(lisi_vals) == 0:
        return float("nan")

    return float(np.mean(lisi_vals))


def batchkl_on_embedding_paper(X, batch_labels, k=30, sample_n=100, seed=42):
    X = np.asarray(X, dtype=np.float64)
    batch_labels = np.asarray(batch_labels).astype(str)

    n = X.shape[0]
    if n < 2:
        return float("nan")

    batches, batch_idx = np.unique(batch_labels, return_inverse=True)
    n_batches = len(batches)
    if n_batches < 2:
        return 0.0

    q = np.bincount(batch_idx, minlength=n_batches).astype(np.float64)
    q = q / q.sum()

    rng = np.random.default_rng(seed)
    m = min(sample_n, n)
    sample_idx = rng.choice(n, size=m, replace=False)

    n_neighbors = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X)
    neigh = nn.kneighbors(X[sample_idx], return_distance=False)
    neigh = neigh[:, 1:]

    kl_vals = []
    for row in neigh:
        local = np.bincount(batch_idx[row], minlength=n_batches).astype(np.float64)
        p = local / local.sum()

        mask = p > 0
        kl = np.sum(p[mask] * np.log((p[mask] + 1e-12) / (q[mask] + 1e-12)))
        kl_vals.append(kl)

    if len(kl_vals) == 0:
        return float("nan")

    return float(np.mean(kl_vals))


def compute_ari_leiden_on_embedding(X, y, n_neighbors=30, resolution=1.0, seed=42):

    tmp = ad.AnnData(X=X.copy())
    tmp.obs["label"] = pd.Categorical(np.asarray(y).astype(str))

    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=n_neighbors, random_state=seed)

    try:
        sc.tl.leiden(
            tmp,
            key_added="leiden",
            resolution=resolution,
            random_state=seed,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    except Exception:
        sc.tl.leiden(
            tmp,
            key_added="leiden",
            resolution=resolution,
            random_state=seed,
        )

    return float(adjusted_rand_score(tmp.obs["label"], tmp.obs["leiden"]))


def compute_five_metrics(adata, batch_key, bio_key, rep_key="X_umap_eval", seed=42):

    X_eval = np.asarray(adata.obsm[rep_key], dtype=np.float64)

    asw_batch = asw_on_embedding(
        X_eval,
        adata.obs[batch_key].to_numpy(),
    )

    asw_bio = asw_on_embedding(
        X_eval,
        adata.obs[bio_key].to_numpy(),
    )

    ilisi = ilisi_on_embedding_paper(
        X_eval,
        adata.obs[batch_key].to_numpy(),
        k=K_METRIC,
        beta=ILISI_BETA,
    )

    batchkl = batchkl_on_embedding_paper(
        X_eval,
        adata.obs[batch_key].to_numpy(),
        k=K_METRIC,
        sample_n=BATCHKL_SAMPLE_N,
        seed=seed,
    )

    ari = compute_ari_leiden_on_embedding(
        X_eval,
        adata.obs[bio_key].to_numpy(),
        n_neighbors=K_METRIC,
        resolution=LEIDEN_RES,
        seed=seed,
    )

    return {
        f"1-ASW_{batch_key}": 1.0 - asw_batch if pd.notna(asw_batch) else float('nan'),
        f"ASW_{bio_key}": asw_bio,
        f"iLISI({batch_key})": ilisi,
        f"BatchKL({batch_key})": batchkl,
        f"ARI(Leiden vs {bio_key})": ari,
        "n_cells": int(adata.n_obs),
        "seed": int(seed),
        "eval_rep": rep_key,
    }

adata_corr = sc.read_h5ad(CORR_PATH)

for k in [ASSAY_KEY, CELLTYPE_KEY]:
    if k not in adata_corr.obs.columns:
        raise KeyError(f"[corrected] missing obs['{k}']")

adata_corr.obs[ASSAY_KEY] = adata_corr.obs[ASSAY_KEY].astype("category")
adata_corr.obs[CELLTYPE_KEY] = adata_corr.obs[CELLTYPE_KEY].astype("category")

X_corr = to_dense_float32(adata_corr.X)
corr_source = "adata_corr.X"

print("Evaluation source:")
print(f"  CORRECTED -> {corr_source}")
print("X_corr shape:", X_corr.shape)
print(f"ASSAY_KEY   = {ASSAY_KEY}")
print(f"CELLTYPE_KEY= {CELLTYPE_KEY}")
print(f"SEED        = {SEED}")

print("\nBuilding CORRECTED independent PCA/UMAP...")
adata_corr_eval = build_independent_eval_space(
    adata_corr,
    X_corr,
    n_pcs=N_PCS,
    n_neighbors=N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    seed=SEED,
)


print("\nPlotting UMAP with scanpy...")

sc.pl.umap(
    adata_corr_eval,
    color=[ASSAY_KEY, CELLTYPE_KEY],
    title=[f"Corrected | {ASSAY_KEY}", f"Corrected | {CELLTYPE_KEY}"],
    wspace=0.4,      
    frameon=False,  
    show=False 
)

out_png = OUTDIR / "corrected_umap_assay_eval_bio_label.png"
plt.savefig(out_png, dpi=400, bbox_inches="tight")
plt.close()
print("Saved figure:", out_png)

print(f"\nComputing 5 metrics on {EVAL_REP} with seed={SEED} ...")

corr_metrics = compute_five_metrics(
    adata_corr_eval,
    batch_key=ASSAY_KEY,
    bio_key=CELLTYPE_KEY,
    rep_key=EVAL_REP,
    seed=SEED,
)

df = pd.DataFrame([corr_metrics], index=["Corrected"])

out_csv = OUTDIR / "metrics_corrected_paper5.csv"
df.to_csv(out_csv)

print("Saved metrics CSV:", out_csv)
print(df)

print("\nDone.")