#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

H5AD_PATH = str(PROJECT_ROOT / "data" / "tgt_clean_colorectum.h5ad")
OUTDIR = str(PROJECT_ROOT / "metrix" / "phase3_metrics_multi")

LABEL_COL = "cell_state_label"
ASSAY_COL = "assay"
SUBTYPE_COL = "subtype"
TRUE_SUBTYPES = None

EVALS = [
    {
        "name": "assay_10x_3_v2",
        "subtype_csv": str(PROJECT_ROOT / "output" / "subtype_by_assay" / "subtype_10x_3'_v2.csv"),
        "assay_value": "10x 3' v2",
    },
    {
        "name": "assay_10x_3_v3",
        "subtype_csv": str(PROJECT_ROOT / "output" / "subtype_by_assay" / "subtype_10x_3'_v3.csv"),
        "assay_value": "10x 3' v3",
    },
    {
        "name": "cross_assay_all",
        "subtype_csv": str(PROJECT_ROOT / "output" / "subtype_aligned" / "subtype_all.csv"),
        "assay_value": None,
    },
]

N_PCS = 50
N_NEIGHBORS = 15
MIN_DIST = 0.3
SEED = 42

LIGHT_GREY = "#D9D9D9"


def _auto_set_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df.columns) == 0:
        raise ValueError("CSV has no columns.")
    first_col = df.columns[0]
    df = df.set_index(first_col)
    df.index = df.index.astype(str)
    return df


def _read_indexed_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _auto_set_index(df)


def _infer_true_subtypes(series: pd.Series, explicit_values: list[str] | None = None) -> list[str]:
    s = series.astype(str)

    if explicit_values is not None:
        if len(explicit_values) == 0:
            raise ValueError("TRUE_SUBTYPES is empty.")
        return [str(x) for x in explicit_values]

    uniq = pd.Index(s.unique())
    tumor_like = [x for x in uniq if "tumor" in str(x).lower()]
    if len(tumor_like) == 0:
        raise ValueError(
            "Failed to auto-detect true ASC subtypes from LABEL_COL. "
            "Please set TRUE_SUBTYPES manually."
        )
    return sorted(map(str, tumor_like))


def _safe_ari(y_true: Sequence, y_pred: Sequence) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return np.nan
    return float(adjusted_rand_score(y_true, y_pred))


def _series_sorted_unique_str(s: pd.Series) -> list[str]:
    vals = s.dropna().astype(str).unique().tolist()
    try:
        return sorted(vals, key=lambda x: float(x))
    except Exception:
        return sorted(vals)


def _make_color_map(categories: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {cat: cmap(i % 10) for i, cat in enumerate(categories)}


def _build_umap(adata: sc.AnnData) -> np.ndarray:
    a = adata.copy()

    if a.n_obs < 3:
        raise ValueError(f"Too few cells to build UMAP: n_obs={a.n_obs}")
    if a.n_vars < 2:
        raise ValueError(f"Too few genes/features to build UMAP: n_vars={a.n_vars}")

    max_pcs = min(N_PCS, a.n_obs - 1, a.n_vars - 1)
    max_pcs = max(2, max_pcs)

    n_neighbors = min(N_NEIGHBORS, a.n_obs - 1)
    n_neighbors = max(2, n_neighbors)

    sc.pp.pca(a, n_comps=max_pcs, random_state=SEED)
    sc.pp.neighbors(
        a,
        n_neighbors=n_neighbors,
        n_pcs=max_pcs,
        random_state=SEED,
    )
    sc.tl.umap(a, min_dist=MIN_DIST, random_state=SEED)
    return a.obsm["X_umap"]


def remap_pred_subtypes_to_true_names(
    true_subtype: pd.Series,
    pred_subtype: pd.Series,
    true_is_asc: pd.Series,
    pred_is_asc: pd.Series,
) -> tuple[pd.Series, dict[str, str]]:
    mask = true_is_asc & pred_is_asc & pred_subtype.notna()

    pred_subtype = pred_subtype.copy()
    pred_subtype.index = pred_subtype.index.astype(str)

    if mask.sum() == 0:
        remapped = pred_subtype.copy()
        remapped[~pred_is_asc] = np.nan
        return remapped, {}

    tab = pd.crosstab(
        true_subtype[mask].astype(str),
        pred_subtype[mask].astype(str),
    )

    if tab.shape[0] == 0 or tab.shape[1] == 0:
        remapped = pred_subtype.copy()
        remapped[~pred_is_asc] = np.nan
        return remapped, {}

    cost = -tab.to_numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[str, str] = {}
    for i, j in zip(row_ind, col_ind):
        mapping[str(tab.columns[j])] = str(tab.index[i])

    remapped = pred_subtype.astype(str).map(mapping)

    all_pred_labels = pred_subtype.dropna().astype(str).unique().tolist()
    for lab in all_pred_labels:
        if lab not in mapping:
            remapped[pred_subtype.astype(str) == lab] = f"unmatched_{lab}"

    remapped[~pred_is_asc] = np.nan
    return remapped, mapping


def _plot_truth(
    ax,
    umap: np.ndarray,
    true_subtype: pd.Series,
    true_is_asc: pd.Series,
    title: str,
    color_map: dict[str, tuple[float, float, float, float]] | None = None,
):
    ax.scatter(umap[:, 0], umap[:, 1], s=6, c=LIGHT_GREY, alpha=0.5, linewidths=0, rasterized=True)

    cats = sorted(pd.Index(true_subtype[true_is_asc].astype(str).unique()).tolist())
    cmap = color_map if color_map is not None else _make_color_map(cats)

    for cat in cats:
        mask = true_is_asc & (true_subtype.astype(str) == cat)
        ax.scatter(
            umap[mask.values, 0],
            umap[mask.values, 1],
            s=10,
            c=[cmap[cat]],
            alpha=0.95,
            linewidths=0,
            label=cat,
            rasterized=True,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(loc="best", frameon=False, fontsize=9, markerscale=2)


def _plot_pred(
    ax,
    umap: np.ndarray,
    pred_is_asc: pd.Series,
    pred_subtype: pd.Series,
    title: str,
    color_map: dict[str, tuple[float, float, float, float]] | None = None,
):
    ax.scatter(umap[:, 0], umap[:, 1], s=6, c=LIGHT_GREY, alpha=0.5, linewidths=0, rasterized=True)

    valid_mask = pred_is_asc & pred_subtype.notna()
    cats = _series_sorted_unique_str(pred_subtype[valid_mask])
    cmap = color_map if color_map is not None else _make_color_map(cats)

    for cat in cats:
        mask = valid_mask & (pred_subtype.astype(str) == cat)
        ax.scatter(
            umap[mask.values, 0],
            umap[mask.values, 1],
            s=10,
            c=[cmap[cat]],
            alpha=0.95,
            linewidths=0,
            label=cat,
            rasterized=True,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(loc="best", frameon=False, fontsize=9, markerscale=2)


def _evaluate_one(
    adata_full: sc.AnnData,
    eval_name: str,
    subtype_csv: str,
    assay_value: str | None,
    outdir: Path,
) -> pd.DataFrame:
    if assay_value is None:
        adata = adata_full.copy()
    else:
        if ASSAY_COL not in adata_full.obs.columns:
            raise ValueError(f"{ASSAY_COL!r} not found in adata.obs.")
        mask = adata_full.obs[ASSAY_COL].astype(str).eq(str(assay_value))
        adata = adata_full[mask].copy()

    if adata.n_obs == 0:
        raise ValueError(f"No cells found for evaluation: {eval_name}")

    true_subtype = adata.obs[LABEL_COL].astype(str).copy()
    true_subtypes = _infer_true_subtypes(true_subtype, TRUE_SUBTYPES)
    true_is_asc = true_subtype.isin(true_subtypes)

    subtype_df = _read_indexed_csv(subtype_csv)
    if SUBTYPE_COL not in subtype_df.columns:
        raise ValueError(
            f"{SUBTYPE_COL!r} not found in {subtype_csv}. "
            f"Available columns: {list(subtype_df.columns)}"
        )

    pred_subtype_raw = subtype_df[SUBTYPE_COL].copy()
    pred_subtype_raw.index = pred_subtype_raw.index.astype(str)
    pred_subtype_raw = pred_subtype_raw.reindex(adata.obs_names)

    pred_is_asc = pred_subtype_raw.notna()

    pred_subtype, pred_to_true_mapping = remap_pred_subtypes_to_true_names(
        true_subtype=true_subtype,
        pred_subtype=pred_subtype_raw,
        true_is_asc=true_is_asc,
        pred_is_asc=pred_is_asc,
    )

    y_true_det = true_is_asc.astype(int)
    y_pred_det = pred_is_asc.astype(int)
    macro_f1 = float(f1_score(y_true_det, y_pred_det, average="macro", zero_division=0))

    tp_mask = true_is_asc & pred_is_asc & pred_subtype.notna()
    y_true_sub = true_subtype[tp_mask].astype(str)
    y_pred_sub = pred_subtype[tp_mask].astype(str)

    ari = _safe_ari(y_true_sub, y_pred_sub)
    product = float(macro_f1 * ari) if np.isfinite(ari) else np.nan

    tp = int((true_is_asc & pred_is_asc).sum())
    fp = int((~true_is_asc & pred_is_asc).sum())
    fn = int((true_is_asc & ~pred_is_asc).sum())
    tn = int((~true_is_asc & ~pred_is_asc).sum())

    mapping_text = "; ".join([f"{k}->{v}" for k, v in pred_to_true_mapping.items()])

    metrics = pd.DataFrame(
        {
            "eval_name": [eval_name],
            "assay_value": [assay_value if assay_value is not None else "all"],
            "macro_f1_detection": [macro_f1],
            "ari_on_tp_intersection": [ari],
            "macro_f1_x_ari": [product],
            "n_total": [adata.n_obs],
            "n_true_asc": [int(true_is_asc.sum())],
            "n_pred_asc": [int(pred_is_asc.sum())],
            "n_tp_for_ari": [int(tp_mask.sum())],
            "TP": [tp],
            "FP": [fp],
            "FN": [fn],
            "TN": [tn],
            "true_subtypes": ["|".join(map(str, true_subtypes))],
            "subtype_csv": [subtype_csv],
            "pred_to_true_mapping": [mapping_text],
        }
    )

    umap = _build_umap(adata)

    true_cats = sorted(pd.Index(true_subtype[true_is_asc].astype(str).unique()).tolist())
    pred_cats = sorted(pd.Index(pred_subtype[pred_is_asc & pred_subtype.notna()].astype(str).unique()).tolist())

    all_cats: list[str] = []
    for x in true_cats + pred_cats:
        if x not in all_cats:
            all_cats.append(x)

    shared_color_map = _make_color_map(all_cats)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    _plot_truth(
        axes[0],
        umap=umap,
        true_subtype=true_subtype,
        true_is_asc=true_is_asc,
        title=f"{eval_name} | True ASC subtypes (n={int(true_is_asc.sum())})",
        color_map=shared_color_map,
    )

    _plot_pred(
        axes[1],
        umap=umap,
        pred_is_asc=pred_is_asc,
        pred_subtype=pred_subtype,
        title=f"{eval_name} | Pred ASC + subtype (n={int((pred_is_asc & pred_subtype.notna()).sum())})",
        color_map=shared_color_map,
    )

    fig.suptitle(
        f"{eval_name} | Macro-F1={macro_f1:.4f} | ARI={ari:.4f} | Product={product:.4f}",
        fontsize=14,
    )

    pair_png = outdir / f"{eval_name}_umap_pair.png"
    fig.savefig(pair_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics_csv = outdir / f"{eval_name}_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    print("=" * 80)
    print(metrics.to_string(index=False))
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved figure : {pair_png}")

    return metrics


def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    adata_full = sc.read_h5ad(H5AD_PATH)
    if LABEL_COL not in adata_full.obs.columns:
        raise ValueError(
            f"{LABEL_COL!r} not found in adata.obs. "
            f"Available columns: {list(adata_full.obs.columns)}"
        )

    adata_full.obs_names = adata_full.obs_names.astype(str)

    all_metrics = []
    for cfg in EVALS:
        metrics = _evaluate_one(
            adata_full=adata_full,
            eval_name=cfg["name"],
            subtype_csv=cfg["subtype_csv"],
            assay_value=cfg["assay_value"],
            outdir=outdir,
        )
        all_metrics.append(metrics)

    summary = pd.concat(all_metrics, axis=0, ignore_index=True)
    summary_csv = outdir / "summary_metrics.csv"
    summary.to_csv(summary_csv, index=False)

    print("=" * 80)
    print("Done.")
    print(summary.to_string(index=False))
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()