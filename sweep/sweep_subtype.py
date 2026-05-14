import argparse
import copy
import gc
import itertools
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score

from .configs import SubtypeConfigs
from .utils import update_configs_with_args
from .subtype import SubtypeModel


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def build_seed_list(random_state_list: str, num_random_seeds: int | None, seed_start: int):
    if random_state_list.strip():
        return parse_int_list(random_state_list)
    if num_random_seeds is not None:
        return list(range(seed_start, seed_start + num_random_seeds))
    return [42]


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def collect_predicted_anomaly_subset(
    adata,
    pred_df: pd.DataFrame,
    cell_col: str,
    label_col: str,
    anomaly_label: str,
    target_assay: str = "",
    adata_assay_col: str = "assay",
    csv_assay_col: str = "assay",
):
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
    pred_anomaly_cells = pd.Index(pred_use.loc[pred_mask, cell_col].astype(str))
    pred_anomaly_cells = adata_use.obs_names.intersection(pred_anomaly_cells)

    if len(pred_anomaly_cells) == 0:
        raise ValueError("No predicted anomaly cells found after matching pred_csv and h5ad.")

    return adata_use[pred_anomaly_cells].copy()


def build_configs(args, adata_pred, num_types, n_epochs, batch_size, learning_rate, weight_decay, random_state):
    configs = SubtypeConfigs()
    args_dict = {
        "read_path": args.read_path,
        "pred_csv": args.pred_csv,
        "pth_path": args.pth_path,
        "cell_col": args.cell_col,
        "label_col": args.label_col,
        "anomaly_label": args.anomaly_label,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "GPU": args.GPU,
        "random_state": random_state,
        "n_genes": adata_pred.n_vars,
        "num_types": num_types,
    }
    update_configs_with_args(configs, args_dict, None)
    configs.build()
    configs.clear()
    return configs


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--read_path", type=str, required=True)
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--pth_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--cell_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default="pred")
    parser.add_argument("--anomaly_label", type=str, default="abnormal")

    # subtype semantics
    parser.add_argument("--target_assay", type=str, default="")
    parser.add_argument("--adata_assay_col", type=str, default="assay")
    parser.add_argument("--csv_assay_col", type=str, default="assay")

    # truth
    parser.add_argument("--truth_label_col", type=str, default="cell_state_label")
    parser.add_argument(
        "--truth_subtype_values",
        type=str,
        default="tumor02_MMRd,tumor02_MMRp",
        help="comma-separated true subtype labels"
    )

    # sweep params
    parser.add_argument("--num_types_list", type=str, default="2")
    parser.add_argument("--n_epochs_list", type=str, default="200")
    parser.add_argument("--batch_size_list", type=str, default="512")
    parser.add_argument("--learning_rate_list", type=str, default="1e-4")
    parser.add_argument("--weight_decay_list", type=str, default="0")

    parser.add_argument("--random_state_list", type=str, default="")
    parser.add_argument("--num_random_seeds", type=int, default=None)
    parser.add_argument("--seed_start", type=int, default=0)

    parser.add_argument("--GPU", type=str, default="cuda:0")
    parser.add_argument("--save_every", type=int, default=1)

    args = parser.parse_args()

    num_types_list = parse_int_list(args.num_types_list)
    n_epochs_list = parse_int_list(args.n_epochs_list)
    batch_size_list = parse_int_list(args.batch_size_list)
    learning_rate_list = parse_float_list(args.learning_rate_list)
    weight_decay_list = parse_float_list(args.weight_decay_list)
    truth_subtype_values = parse_str_list(args.truth_subtype_values)
    random_state_list = build_seed_list(
        args.random_state_list,
        args.num_random_seeds,
        args.seed_start,
    )

    mode = "single_assay_fig7" if str(args.target_assay).strip() else "whole_tgt_fig8"
    print("mode:", mode)
    if str(args.target_assay).strip():
        print("target_assay:", args.target_assay)
    print("n_random_seeds:", len(random_state_list))

    adata = sc.read_h5ad(args.read_path)
    adata.obs_names = adata.obs_names.astype(str)

    need_assay_col = bool(str(args.target_assay).strip())
    pred_df, resolved_cell_col = read_prediction_csv(
        csv_path=args.pred_csv,
        cell_col=args.cell_col,
        assay_col=args.csv_assay_col if need_assay_col else None,
    )

    adata_pred = collect_predicted_anomaly_subset(
        adata=adata,
        pred_df=pred_df,
        cell_col=resolved_cell_col,
        label_col=args.label_col,
        anomaly_label=args.anomaly_label,
        target_assay=args.target_assay,
        adata_assay_col=args.adata_assay_col,
        csv_assay_col=args.csv_assay_col,
    )

    if args.truth_label_col not in adata_pred.obs.columns:
        raise KeyError(f"{args.truth_label_col} not found in adata.obs")

    true_label_series = adata_pred.obs[args.truth_label_col].astype(str)
    eval_mask = true_label_series.isin(truth_subtype_values)

    if eval_mask.sum() == 0:
        raise ValueError("Intersection between predicted anomaly cells and true subtype cells is empty.")

    print("n_predicted_anomaly:", adata_pred.n_obs)
    print("n_eval_intersection:", int(eval_mask.sum()))

    try:
        base_generator = torch.load(args.pth_path, map_location="cpu", weights_only=False)
    except TypeError:
        base_generator = torch.load(args.pth_path, map_location="cpu")

    results = []

    combos = list(itertools.product(
        num_types_list,
        n_epochs_list,
        batch_size_list,
        learning_rate_list,
        weight_decay_list,
        random_state_list,
    ))

    print("total_runs:", len(combos))

    for run_idx, (num_types, n_epochs, batch_size, learning_rate, weight_decay, random_state) in enumerate(combos, start=1):
        print(
            f"[{run_idx}/{len(combos)}] "
            f"num_types={num_types}, epochs={n_epochs}, batch_size={batch_size}, "
            f"lr={learning_rate}, wd={weight_decay}, seed={random_state}"
        )

        if adata_pred.n_obs < num_types:
            results.append({
                "status": "skip",
                "run_idx": run_idx,
                "mode": mode,
                "target_assay": args.target_assay if str(args.target_assay).strip() else "",
                "num_types": num_types,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "random_state": random_state,
                "n_predicted_anomaly": int(adata_pred.n_obs),
                "n_eval_intersection": int(eval_mask.sum()),
                "ari": np.nan,
                "error": "n_obs < num_types",
            })
            continue

        seed_everything(random_state)

        try:
            configs = build_configs(
                args=args,
                adata_pred=adata_pred,
                num_types=num_types,
                n_epochs=n_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                random_state=random_state,
            )

            generator = copy.deepcopy(base_generator)
            model = SubtypeModel(generator, num_types, configs)
            pred_subtype = model.train(adata_pred)

            pred_eval = pred_subtype[eval_mask.to_numpy()]
            true_eval = true_label_series[eval_mask].to_numpy()

            true_codes = pd.Categorical(
                true_eval,
                categories=truth_subtype_values
            ).codes

            ari = adjusted_rand_score(true_codes, pred_eval)

            results.append({
                "status": "ok",
                "run_idx": run_idx,
                "mode": mode,
                "target_assay": args.target_assay if str(args.target_assay).strip() else "",
                "num_types": num_types,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "random_state": random_state,
                "n_predicted_anomaly": int(adata_pred.n_obs),
                "n_eval_intersection": int(eval_mask.sum()),
                "ari": float(ari),
            })
            print("ARI =", ari)

        except Exception as e:
            results.append({
                "status": "error",
                "run_idx": run_idx,
                "mode": mode,
                "target_assay": args.target_assay if str(args.target_assay).strip() else "",
                "num_types": num_types,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "random_state": random_state,
                "n_predicted_anomaly": int(adata_pred.n_obs),
                "n_eval_intersection": int(eval_mask.sum()),
                "ari": np.nan,
                "error": repr(e),
            })
            print("ERROR:", repr(e))

        if run_idx % args.save_every == 0:
            pd.DataFrame(results).to_csv(args.output_csv, index=False)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"Saved to: {args.output_csv}")


if __name__ == "__main__":
    main()