import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


def dirichlet_profiles(K: int, G: int, alpha_scale: float, seed: int):
    rng = np.random.default_rng(seed)
    prof = []
    for _ in range(K):
        alpha = rng.lognormal(mean=np.log(alpha_scale), sigma=0.6, size=G)
        p = rng.dirichlet(alpha)
        prof.append(p)
    return np.stack(prof, axis=0)  # [K, G]


def simulate_counts(n_cells: int, gene_probs: np.ndarray, mean_umi: int, umi_sigma: float, seed: int):
    rng = np.random.default_rng(seed)
    G = gene_probs.shape[-1]

    lib = rng.lognormal(mean=np.log(mean_umi), sigma=umi_sigma, size=n_cells).astype(int)
    lib = np.clip(lib, 800, None)

    X = np.zeros((n_cells, G), dtype=np.float32)
    for i in range(n_cells):
        X[i] = rng.multinomial(lib[i], gene_probs).astype(np.float32)
    return X


def apply_batch_effect_logspace(X: np.ndarray, seed: int):
    """
    Apply batch effect in log1p space then invert back.
    """
    rng = np.random.default_rng(seed)
    G = X.shape[1]

    tmp = ad.AnnData(X.copy())
    sc.pp.normalize_total(tmp, target_sum=1e4)
    Y = np.log1p(tmp.X)

    gene_scale = rng.lognormal(mean=0.0, sigma=0.06, size=G).astype(np.float32)  # mult in log-space
    gene_shift = rng.normal(0.0, 0.12, size=G).astype(np.float32)  # add in log-space
    Y = Y * gene_scale + gene_shift + rng.normal(0.0, 0.03, size=Y.shape)

    X2 = np.expm1(Y)
    X2[X2 < 0] = 0.0
    return X2.astype(np.float32)


def preprocess_like_m2asda(adata: ad.AnnData):
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata, base=np.e)
    # ensure dense
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    return adata


def build_target(
    name: str,
    n_cells: int,
    asc_frac: float,
    profiles: np.ndarray,
    normal_names,
    asc_up_genes: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    K, G = profiles.shape

    n_asc = int(round(n_cells * asc_frac))
    n_norm = n_cells - n_asc

    # normal cells
    norm_type = rng.integers(0, K, size=n_norm)
    X_norm = np.zeros((n_norm, G), dtype=np.float32)
    for i in range(n_norm):
        p = profiles[norm_type[i]]
        X_norm[i : i + 1] = simulate_counts(1, p, mean_umi=6500, umi_sigma=0.35, seed=seed + i)

    # anomaly subtype
    base_type = int(rng.integers(0, K))
    p_asc = profiles[base_type].copy()
    spike_idx = rng.choice(G, size=min(asc_up_genes, G), replace=False)
    p_asc[spike_idx] *= 8.0
    p_asc = p_asc / p_asc.sum()
    X_asc = simulate_counts(n_asc, p_asc, mean_umi=7000, umi_sigma=0.35, seed=seed + 999)

    X = np.vstack([X_norm, X_asc]).astype(np.float32)
    X = apply_batch_effect_logspace(X, seed=seed + 12345)

    cell_type = np.array([normal_names[t] for t in norm_type] + [f"ASC_{name}"] * n_asc, dtype=object)
    is_asc = np.array([0] * n_norm + [1] * n_asc, dtype=int)

    adata = ad.AnnData(X=X)
    adata.obs["cell_type"] = cell_type
    adata.obs["is_asc_gt"] = is_asc
    adata.obs["batch"] = name  # optional, safe
    adata.obs_names = [f"{name}_cell{i:04d}" for i in range(n_cells)]
    return adata


def save_phase1_csv(adata: ad.AnnData, out_csv: Path, seed: int = 123):
    rng = np.random.default_rng(seed)
    y = adata.obs["is_asc_gt"].to_numpy().astype(int)

    score = rng.normal(0.2, 0.05, size=adata.n_obs)
    score[y == 1] = rng.normal(0.8, 0.05, size=(y == 1).sum())

    df = pd.DataFrame({"score": score, "label": y}, index=adata.obs_names)
    df.to_csv(out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="fake_m2asda_data")
    ap.add_argument("--n_genes", type=int, default=3000)
    ap.add_argument("--n_ref", type=int, default=512)
    ap.add_argument("--n_targets", type=int, default=2)
    ap.add_argument("--n_tgt", type=int, default=320)
    ap.add_argument("--asc_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = args.n_genes
    gene_names = [f"g{i:04d}" for i in range(G)]

    normal_names = ["T0", "T1", "T2"]
    profiles = dirichlet_profiles(K=len(normal_names), G=G, alpha_scale=0.15, seed=args.seed)

    # ref
    rng = np.random.default_rng(args.seed)
    ref_types = rng.integers(0, len(normal_names), size=args.n_ref)
    X_ref = np.zeros((args.n_ref, G), dtype=np.float32)
    for i in range(args.n_ref):
        X_ref[i : i + 1] = simulate_counts(1, profiles[ref_types[i]], mean_umi=7000, umi_sigma=0.35, seed=args.seed + i)

    ref = ad.AnnData(X=X_ref)
    ref.var_names = gene_names
    ref.obs["cell_type"] = [normal_names[t] for t in ref_types]
    ref.obs["is_asc_gt"] = 0
    ref.obs_names = [f"ref_cell{i:04d}" for i in range(args.n_ref)]
    ref = preprocess_like_m2asda(ref)

    ref_path = out_dir / "ref.h5ad"
    ref.write_h5ad(ref_path)

    # targets
    phase1_csvs = []
    asc_files = []

    for i in range(1, args.n_targets + 1):
        name = f"tgt{i}"
        tgt = build_target(
            name=name,
            n_cells=args.n_tgt,
            asc_frac=args.asc_frac,
            profiles=profiles,
            normal_names=normal_names,
            asc_up_genes=120 + 40 * i,
            seed=args.seed + 1000 * i,
        )
        tgt.var_names = gene_names
        tgt = preprocess_like_m2asda(tgt)

        tgt_path = out_dir / f"{name}.h5ad"
        tgt.write_h5ad(tgt_path)

        # phase1 label csv (ground-truth)
        csv_path = out_dir / f"phase1_{name}.csv"
        save_phase1_csv(tgt, csv_path)
        phase1_csvs.append(csv_path)

        # asc-only h5ad (for Phase III input)
        asc = tgt[tgt.obs["is_asc_gt"].astype(int) == 1].copy()
        asc_path = out_dir / f"asc_{name}.h5ad"
        asc.write_h5ad(asc_path)
        asc_files.append(asc_path)

    # a convenience merged ASC file
    asc_all = ad.concat([ad.read_h5ad(p) for p in asc_files], label="source", keys=[p.stem for p in asc_files])
    asc_all.write_h5ad(out_dir / "asc_all.h5ad")

    print(f"[OK] wrote: {ref_path}")
    print(f"[OK] wrote targets: {args.n_targets} files into {out_dir}")
    print(f"[OK] wrote phase1 csvs: {[p.name for p in phase1_csvs]}")
    print(f"[OK] wrote asc files: {[p.name for p in asc_files]} + asc_all.h5ad")


if __name__ == "__main__":
    main()
