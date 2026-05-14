from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

proj_root = Path(__file__).resolve().parent.parent
tgt_path = proj_root / "data" / "tgt_clean_colorectum.h5ad"
csv_path = proj_root / "output" / "phase1_pred_tgt.csv"

out_dir = proj_root / "metrix" / "phase1_metrix_by_assay"
out_dir.mkdir(parents=True, exist_ok=True)


def metrics_row(method, y_true, y_pred, score, threshold=np.nan):
    return {
        "Method": method,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_pos1": precision_score(y_true, y_pred, zero_division=0),
        "Recall_pos1": recall_score(y_true, y_pred, zero_division=0),
        "F1_pos1": f1_score(y_true, y_pred, zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, score),
        "PR_AUC": average_precision_score(y_true, score),
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def evaluate_assay(adata, df, assay):
    assay_dir = out_dir / str(assay).replace("/", "_")
    assay_dir.mkdir(parents=True, exist_ok=True)

    y_true = (adata.obs["disease"].astype(str).str.lower() != "normal").astype(int).to_numpy()
    score = df["score"].astype(float).to_numpy()
    gmm_pred = df["label"].astype(int).to_numpy()

    thresholds = np.unique(score)
    f1_macros = [
        f1_score(y_true, (score >= th).astype(int), average="macro", zero_division=0)
        for th in thresholds
    ]
    best_th = thresholds[np.argmax(f1_macros)]
    best_pred = (score >= best_th).astype(int)

    metrics_df = pd.DataFrame([
        metrics_row("GMM_label", y_true, gmm_pred, score),
        metrics_row("BestF1macro_cutoff", y_true, best_pred, score, best_th),
    ])
    metrics_df.to_csv(assay_dir / "evaluation_metric.csv", index=False)

    pred_df = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "assay": adata.obs["assay"].astype(str).to_numpy(),
        "score": score,
        "GMM_label": gmm_pred,
        "GMM_pred": np.where(gmm_pred == 1, "abnormal", "normal"),
        "BestF1macro_cutoff": best_th,
        "BestF1macro_pred": np.where(best_pred == 1, "abnormal", "normal"),
    })
    pred_df.to_csv(assay_dir / "pred_with_meta.csv", index=False)

    fpr, tpr, _ = roc_curve(y_true, score)
    precision, recall, _ = precision_recall_curve(y_true, score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(y_true, score):.4f}")
    axes[0].plot([0, 1], [0, 1], lw=1, linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve ({assay})")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve ({assay})")
    axes[0].legend()

    axes[1].plot(recall, precision, lw=2, label=f"AUC={average_precision_score(y_true, score):.4f}")
    axes[1].plot([0, 1], [y_true.mean(), y_true.mean()], lw=1, linestyle="--")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve ({assay})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(assay_dir / "roc_pr_curves.png", dpi=300)
    plt.close()

    return {
        "assay": assay,
        "n_cells": len(y_true),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        **metrics_df.loc[metrics_df["Method"] == "GMM_label"].iloc[0].to_dict(),
    }


tgt = sc.read_h5ad(tgt_path)
df = pd.read_csv(csv_path, index_col=0)

tgt.obs_names = tgt.obs_names.astype(str)
df.index = df.index.astype(str)

summary = []

for assay in tgt.obs["assay"].astype(str).unique():
    mask = tgt.obs["assay"].astype(str) == assay
    adata_sub = tgt[mask].copy()
    df_sub = df.loc[adata_sub.obs_names].copy()
    summary.append(evaluate_assay(adata_sub, df_sub, assay))

summary_df = pd.DataFrame(summary)
summary_df.to_csv(out_dir / "summary_by_assay.csv", index=False)

print(summary_df.to_string(index=False))
print(f"\nSaved results to: {out_dir}")