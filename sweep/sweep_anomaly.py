import argparse
import gc
import itertools
import warnings
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
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

from .configs import AnomalyConfigs
from .anomaly import AnomalyModel


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def format_float_for_name(x: float) -> str:
    s = f"{x:.8g}"
    s = s.replace(".", "p").replace("-", "m")
    return s


def build_run_name(cfg: dict) -> str:
    return (
        f"ep{cfg['n_epochs']}"
        f"_bs{cfg['batch_size']}"
        f"_lr{format_float_for_name(cfg['learning_rate'])}"
        f"_nc{cfg['n_critic']}"
        f"_gm{format_float_for_name(cfg['gamma'])}"
        f"_do{format_float_for_name(cfg['dropout'])}"
        f"_norm{int(cfg['normalization'])}"
        f"_mb{int(cfg['use_memory_bank'])}"
        f"_ms{cfg['memory_size']}"
        f"_sd{cfg['random_state']}"
    )


def compute_metrics(y_true_arr, y_pred_arr):
    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec = precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
    rec = recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true_arr, y_pred_arr)
    f1_pos1 = f1_score(y_true_arr, y_pred_arr, pos_label=1, average="binary", zero_division=0)
    return acc, prec, rec, f1_macro, mcc, f1_pos1


def evaluate_one_run(
    *,
    tgt: sc.AnnData,
    score: np.ndarray,
    label: np.ndarray | None,
    metrics_out: Path,
    plot_out: Path,
):
    if "disease" not in tgt.obs.columns:
        raise KeyError("tgt.obs missing 'disease' column.")

    disease = tgt.obs["disease"].astype(str).str.strip().str.lower()
    y_true = (disease != "normal").astype(int).to_numpy()

    score = np.asarray(score, dtype=float).reshape(-1)
    if np.isnan(score).any():
        raise ValueError("Found NaN in score.")

    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true has only one class; ROC/PR curves are undefined.")

    score_flipped = False
    roc_auc_raw = roc_auc_score(y_true, score)
    roc_auc_neg = roc_auc_score(y_true, -score)
    if roc_auc_neg > roc_auc_raw:
        score = -score
        score_flipped = True

    roc_auc = roc_auc_score(y_true, score)
    pr_auc = average_precision_score(y_true, score)

    rows = []

    if label is not None:
        y_pred_gmm = np.asarray(label).astype(int).reshape(-1)
        acc_gmm, prec_gmm, rec_gmm, f1m_gmm, mcc_gmm, f1p_gmm = compute_metrics(y_true, y_pred_gmm)
        rows.append(
            {
                "Method": "GMM_label",
                "Threshold": np.nan,
                "Accuracy": acc_gmm,
                "Precision_pos1": prec_gmm,
                "Recall_pos1": rec_gmm,
                "F1_macro": f1m_gmm,
                "F1_pos1": f1p_gmm,
                "MCC": mcc_gmm,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc,
                "ScoreFlipped": score_flipped,
            }
        )
    else:
        y_pred_gmm = None

    order = np.argsort(-score)
    s_sorted = score[order]
    y_sorted = y_true[order]

    cum_pos = np.cumsum(y_sorted).astype(np.int64)
    cum_cnt = np.arange(1, len(y_sorted) + 1, dtype=np.int64)

    change = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    idx = np.where(change)[0]

    tp = cum_pos[idx]
    pred_pos = cum_cnt[idx]
    fp = pred_pos - tp

    P = int(y_true.sum())
    N = int(len(y_true) - P)
    fn = P - tp
    tn = N - fp

    accs = (tp + tn) / (P + N)
    best_acc = float(accs.max())

    best_idx_all = idx[accs == best_acc]
    best_th = float(np.median(s_sorted[best_idx_all]))

    y_pred_best = (score >= best_th).astype(int)
    acc_best, prec_best, rec_best, f1m_best, mcc_best, f1p_best = compute_metrics(y_true, y_pred_best)

    rows.append(
        {
            "Method": "BestAcc_cutoff",
            "Threshold": best_th,
            "Accuracy": acc_best,
            "Precision_pos1": prec_best,
            "Recall_pos1": rec_best,
            "F1_macro": f1m_best,
            "F1_pos1": f1p_best,
            "MCC": mcc_best,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "ScoreFlipped": score_flipped,
        }
    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_out, index=False)

    fpr, tpr, _ = roc_curve(y_true, score)
    p, r, _ = precision_recall_curve(y_true, score)
    baseline = float(y_true.mean())

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(r, p, lw=2, label=f"PR (AUC={pr_auc:.4f})")
    plt.plot([0, 1], [baseline, baseline], lw=1, linestyle="--", label=f"Baseline={baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_out, dpi=300)
    plt.close()

    summary = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": best_th,
        "best_acc": acc_best,
        "best_prec_pos1": prec_best,
        "best_rec_pos1": rec_best,
        "best_f1_macro": f1m_best,
        "best_f1_pos1": f1p_best,
        "best_mcc": mcc_best,
        "score_flipped": score_flipped,
    }

    if y_pred_gmm is not None:
        summary["gmm_acc"] = rows[0]["Accuracy"]
        summary["gmm_prec_pos1"] = rows[0]["Precision_pos1"]
        summary["gmm_rec_pos1"] = rows[0]["Recall_pos1"]
        summary["gmm_f1_macro"] = rows[0]["F1_macro"]
        summary["gmm_f1_pos1"] = rows[0]["F1_pos1"]
        summary["gmm_mcc"] = rows[0]["MCC"]
        summary["gmm_positive_rate"] = float(y_pred_gmm.mean())
    else:
        summary["gmm_acc"] = np.nan
        summary["gmm_prec_pos1"] = np.nan
        summary["gmm_rec_pos1"] = np.nan
        summary["gmm_f1_macro"] = np.nan
        summary["gmm_f1_pos1"] = np.nan
        summary["gmm_mcc"] = np.nan
        summary["gmm_positive_rate"] = np.nan

    return metrics_df, summary


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="Sweep Phase I anomaly model and save metrics/plots for each hyperparameter combination."
    )

    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--tgt_path", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--n_epochs_list", type=str, default="100")
    parser.add_argument("--batch_size_list", type=str, default="256")
    parser.add_argument("--learning_rate_list", type=str, default="1e-4")
    parser.add_argument("--n_critic_list", type=str, default="2")
    parser.add_argument("--gamma_list", type=str, default="0.1")
    parser.add_argument("--dropout_list", type=str, default="0.1")
    parser.add_argument("--normalization_list", type=str, default="0,1")
    parser.add_argument("--random_state_list", type=str, default="2026")
    parser.add_argument("--use_memory_bank_list", type=str, default="1")
    parser.add_argument("--memory_size_list", type=str, default="512")

    parser.add_argument("--GPU", type=str, default="cuda:0")
    parser.add_argument("--no_gmm", action="store_true")

    parser.add_argument("--index_start", type=int, default=0)
    parser.add_argument("--index_end", type=int, default=None)
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--stop_on_error", action="store_true")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    metrics_dir = outdir / "metrics"
    plots_dir = outdir / "plots"
    logs_dir = outdir / "logs"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ref = sc.read_h5ad(args.ref_path)
    tgt = sc.read_h5ad(args.tgt_path)
    n_genes = ref.n_vars

    n_epochs_list = parse_int_list(args.n_epochs_list)
    batch_size_list = parse_int_list(args.batch_size_list)
    learning_rate_list = parse_float_list(args.learning_rate_list)
    n_critic_list = parse_int_list(args.n_critic_list)
    gamma_list = parse_float_list(args.gamma_list)
    dropout_list = parse_float_list(args.dropout_list)
    normalization_list = parse_int_list(args.normalization_list)
    random_state_list = parse_int_list(args.random_state_list)
    use_memory_bank_list = parse_int_list(args.use_memory_bank_list)
    memory_size_list = parse_int_list(args.memory_size_list)

    for x in normalization_list:
        if x not in (0, 1):
            raise ValueError(f"normalization_list must contain only 0 or 1, got {x}")

    for x in use_memory_bank_list:
        if x not in (0, 1):
            raise ValueError(f"use_memory_bank_list must contain only 0 or 1, got {x}")

    for x in memory_size_list:
        if x <= 0:
            raise ValueError(f"memory_size_list must contain positive integers, got {x}")

    grid = list(
        itertools.product(
            n_epochs_list,
            batch_size_list,
            learning_rate_list,
            n_critic_list,
            gamma_list,
            dropout_list,
            normalization_list,
            random_state_list,
            use_memory_bank_list,
            memory_size_list,
        )
    )

    grid = grid[args.index_start:args.index_end]
    if args.max_runs is not None:
        grid = grid[:args.max_runs]

    records = []

    for run_idx, (
        n_epochs,
        batch_size,
        learning_rate,
        n_critic,
        gamma,
        dropout,
        normalization,
        random_state,
        use_memory_bank,
        memory_size,
    ) in enumerate(grid, start=1):
        cfg = {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "n_critic": n_critic,
            "gamma": gamma,
            "dropout": dropout,
            "normalization": normalization,
            "random_state": random_state,
            "use_memory_bank": use_memory_bank,
            "memory_size": memory_size,
        }
        run_name = build_run_name(cfg)

        metrics_out = metrics_dir / f"{run_name}.csv"
        plot_out = plots_dir / f"{run_name}.png"
        log_out = logs_dir / f"{run_name}.txt"

        print(f"[{run_idx}/{len(grid)}] Running: {run_name}")

        try:
            start = time.time()

            configs = AnomalyConfigs()
            configs.n_epochs = n_epochs
            configs.batch_size = batch_size
            configs.learning_rate = learning_rate
            configs.n_critic = n_critic
            configs.gamma = gamma
            configs.dropout = dropout
            configs.normalization = bool(normalization)
            configs.random_state = random_state
            configs.use_memory_bank = bool(use_memory_bank)
            configs.memory_size = memory_size
            configs.GPU = args.GPU
            configs.n_genes = n_genes
            configs.build()
            configs.clear()

            model = AnomalyModel(configs)
            model.train(ref)

            if args.no_gmm:
                score = model.predict(tgt, run_gmm=False)
                label = None
            else:
                score, label = model.predict(tgt, run_gmm=True)

            metrics_df, summary = evaluate_one_run(
                tgt=tgt,
                score=score,
                label=label,
                metrics_out=metrics_out,
                plot_out=plot_out,
            )

            elapsed = time.time() - start

            with open(log_out, "w", encoding="utf-8") as f:
                f.write(f"run_name = {run_name}\n")
                for k, v in cfg.items():
                    f.write(f"{k} = {v}\n")
                f.write(f"elapsed_sec = {elapsed:.4f}\n")
                f.write("\n")
                f.write(metrics_df.to_string(index=False))
                f.write("\n")

            records.append(
                {
                    **cfg,
                    "run_name": run_name,
                    "status": "ok",
                    "elapsed_sec": elapsed,
                    "metrics_path": str(metrics_out),
                    "plot_path": str(plot_out),
                    **summary,
                }
            )

            pd.DataFrame(records).to_csv(outdir / "summary.csv", index=False)
            print(f"  -> done, elapsed={elapsed:.1f}s")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            records.append(
                {
                    **cfg,
                    "run_name": run_name,
                    "status": f"failed:{type(e).__name__}",
                    "elapsed_sec": np.nan,
                    "metrics_path": "",
                    "plot_path": "",
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "best_threshold": np.nan,
                    "best_acc": np.nan,
                    "best_prec_pos1": np.nan,
                    "best_rec_pos1": np.nan,
                    "best_f1_macro": np.nan,
                    "best_f1_pos1": np.nan,
                    "best_mcc": np.nan,
                    "score_flipped": np.nan,
                    "gmm_acc": np.nan,
                    "gmm_prec_pos1": np.nan,
                    "gmm_rec_pos1": np.nan,
                    "gmm_f1_macro": np.nan,
                    "gmm_f1_pos1": np.nan,
                    "gmm_mcc": np.nan,
                    "gmm_positive_rate": np.nan,
                }
            )
            pd.DataFrame(records).to_csv(outdir / "summary.csv", index=False)
            print(f"  -> failed: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.stop_on_error:
                raise

    print(f"Saved summary to: {outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()