from __future__ import annotations

import argparse
import gc
import itertools
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from .correct_stargan import CorrectModel
from metrix.calculate_adata import evaluate_corrected_adata


LOSS_PRESETS = {

    "With_state_supervision": {
        "lr_g": 1.0e-4,
        "lr_d": 1.0e-4,
        "lambda_batch": 3.0,
        "lambda_state": 0.5,
        "lambda_rec": 3.0,
        "lambda_id": 0.4,
    },
    "without_state_supervision": {
        "lr_g": 1.0e-4,
        "lr_d": 1.0e-4,
        "lambda_batch": 3.0,
        "lambda_state": 0.0,
        "lambda_rec": 3.0,
        "lambda_id": 0.4,
    },
}


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_bool_list(s: str) -> list[bool]:
    vals = []
    for x in s.split(","):
        x = x.strip().lower()
        if not x:
            continue
        if x in {"1", "true", "t", "yes", "y", "on"}:
            vals.append(True)
        elif x in {"0", "false", "f", "no", "n", "off"}:
            vals.append(False)
        else:
            raise ValueError(f"Cannot parse boolean value: {x}")
    return vals


def fmt_float_tag(x: float) -> str:
    s = f"{x:g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def make_run_name(combo_idx: int, combo: dict, seed: int | None = None) -> str:
    gate = int(bool(combo["use_change_gate"]))
    run_name = (
        f"run_{combo_idx:05d}"
        f"__e{combo['epochs']}"
        f"__hd{combo['hidden_dim']}"
        f"__cd{combo['cond_dim']}"
        f"__gb{combo['g_num_blocks']}"
        f"__db{combo['d_num_blocks']}"
        f"__gdo{fmt_float_tag(combo['g_dropout'])}"
        f"__ddo{fmt_float_tag(combo['d_dropout'])}"
        f"__gate{gate}"
        f"__ds{combo['d_steps']}"
        f"__gs{combo['g_steps']}"
        f"__dbp{fmt_float_tag(combo['domain_balance_power'])}"
        f"__{combo['loss_preset']}"
    )
    if seed is not None:
        run_name += f"__sd{seed}"
    return run_name


def build_combinations(args) -> list[dict]:
    combos = []
    for (
        epochs,
        hidden_dim,
        cond_dim,
        g_num_blocks,
        d_num_blocks,
        g_dropout,
        d_dropout,
        use_change_gate,
        d_steps,
        g_steps,
        domain_balance_power,
        loss_preset,
    ) in itertools.product(
        args.epochs_list,
        args.hidden_dims,
        args.cond_dims,
        args.g_num_blocks_list,
        args.d_num_blocks_list,
        args.g_dropouts,
        args.d_dropouts,
        args.use_change_gate_list,
        args.d_steps_list,
        args.g_steps_list,
        args.domain_balance_power_list,
        args.loss_presets,
    ):
        combo = {
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "cond_dim": cond_dim,
            "g_num_blocks": g_num_blocks,
            "d_num_blocks": d_num_blocks,
            "g_dropout": g_dropout,
            "d_dropout": d_dropout,
            "use_change_gate": use_change_gate,
            "d_steps": d_steps,
            "g_steps": g_steps,
            "domain_balance_power": domain_balance_power,
            "loss_preset": loss_preset,
        }
        combo.update(LOSS_PRESETS[loss_preset])
        combos.append(combo)
    return combos


def select_combinations(
    all_combos: list[dict],
    shard_id: int,
    num_shards: int,
    index_start: int | None,
    index_end: int | None,
    max_runs: int | None,
) -> list[tuple[int, dict]]:
    selected = []
    for combo_idx, combo in enumerate(all_combos):
        if num_shards > 1 and combo_idx % num_shards != shard_id:
            continue
        if index_start is not None and combo_idx < index_start:
            continue
        if index_end is not None and combo_idx >= index_end:
            continue
        selected.append((combo_idx, combo))
        if max_runs is not None and len(selected) >= max_runs:
            break
    return selected


def append_df_to_csv(csv_path: Path, df_new: pd.DataFrame) -> None:
    if df_new.empty:
        return

    if not csv_path.exists():
        df_new.to_csv(csv_path, index=False)
        return

    df_old = pd.read_csv(csv_path)
    old_cols = list(df_old.columns)
    new_cols = list(df_new.columns)

    if old_cols == new_cols:
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
        return

    merged_cols = old_cols[:]
    for col in new_cols:
        if col not in merged_cols:
            merged_cols.append(col)

    df_old = df_old.reindex(columns=merged_cols)
    df_new = df_new.reindex(columns=merged_cols)
    pd.concat([df_old, df_new], axis=0, ignore_index=True).to_csv(csv_path, index=False)


def load_existing_run_names(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["run_name"])
        return set(df["run_name"].astype(str))
    except Exception:
        return set()


def _pretty_domain_name(domain: str, ref_domain_name: str | None) -> str:
    if ref_domain_name is not None and domain == ref_domain_name:
        return "ref"
    return str(domain)


def _safe_filename(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("'", "")
        .replace('"', "")
    )


def _compute_umap(adata: ad.AnnData, seed: int = 42) -> ad.AnnData:
    a = adata.copy()
    sc.pp.pca(a, random_state=seed)
    sc.pp.neighbors(a, random_state=seed)
    sc.tl.umap(a, random_state=seed)
    return a


def _get_first_existing_metric(metrics: dict, candidate_keys: list[str]) -> tuple[float, str]:
    for k in candidate_keys:
        if k in metrics and pd.notna(metrics[k]):
            return float(metrics[k]), k
    raise KeyError(
        f"None of these metric keys were found: {candidate_keys}. "
        f"Available keys: {list(metrics.keys())}"
    )


def _extract_core_metrics(metrics: dict) -> dict:
    ilisi, _ = _get_first_existing_metric(
        metrics,
        ["iLISI", "iLISI(assay)", "iLISI(assay_eval)"],
    )
    batchkl, _ = _get_first_existing_metric(
        metrics,
        ["BatchKL", "BatchKL(assay)", "BatchKL(assay_eval)"],
    )
    ari, _ = _get_first_existing_metric(
        metrics,
        [
            "ARI",
            "ARI(Leiden vs bio_label)",
            "ARI(Leiden vs cell_type)",
            "ARI(Leiden vs cell_state_label)",
        ],
    )

    # 你这版 metrix 返回的是 ASW_bio_label
    asw_cell_type, _ = _get_first_existing_metric(
        metrics,
        [
            "ASW_cell_type",
            "ASW_bio_label",
            "ASW_bio",
            "ASW(bio_label)",
        ],
    )

    # 你这版 metrix 返回的是 1-ASW_assay_eval，不是 ASW_batch
    one_minus_asw_batch, used_key = _get_first_existing_metric(
        metrics,
        [
            "1-ASW_batch",
            "1-ASW_assay",
            "1-ASW_assay_eval",
        ],
    )

    asw_batch = 1.0 - one_minus_asw_batch

    return {
        "iLISI": ilisi,
        "BatchKL": batchkl,
        "ARI": ari,
        "ASW_cell_type": asw_cell_type,
        "ASW_batch": asw_batch,
        "one_minus_asw_batch": one_minus_asw_batch,
    }


def _check_save_condition(
    core_metrics: dict,
    ilisi_min: float = 1.8,
    batchkl_max: float = 0.5,
    ari_min: float = 0.13,
    asw_cell_type_min: float = 0.25,
    one_minus_asw_batch_min: float = 1.02,
) -> bool:
    return (
        (core_metrics["iLISI"] > ilisi_min)
        and (core_metrics["BatchKL"] < batchkl_max)
        and (core_metrics["ARI"] > ari_min)
        and (core_metrics["ASW_cell_type"] > asw_cell_type_min)
        and (core_metrics["one_minus_asw_batch"] > one_minus_asw_batch_min)
    )

def save_domain_panel_plot(
    adata_dict: dict[str, ad.AnnData],
    plot_path: str | Path,
    ref_domain_name: str | None,
    batch_key: str = "assay_eval",
    bio_key: str = "bio_label",
    seed: int = 42,
) -> None:
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    domain_names = list(adata_dict.keys())
    n_domains = len(domain_names)

    fig, axes = plt.subplots(
        2,
        n_domains,
        figsize=(6.0 * n_domains, 10.0),
        squeeze=False,
    )

    for j, domain in enumerate(domain_names):
        a = _compute_umap(adata_dict[domain], seed=seed)
        title_domain = _pretty_domain_name(domain, ref_domain_name)

        sc.pl.umap(
            a,
            color=batch_key,
            ax=axes[0, j],
            show=False,
            title=f"to {title_domain}\n{batch_key}",
        )
        sc.pl.umap(
            a,
            color=bio_key,
            ax=axes[1, j],
            show=False,
            title=f"to {title_domain}\n{bio_key}",
        )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_one_combo(
    ref_h5ad: str,
    tgt_h5ad: str,
    outdir: Path,
    batch_key: str,
    disease_key: str,
    normal_value: str,
    ref_bio_key: str,
    bio_key: str,
    device: str,
    batch_size: int,
    align_batch_size: int,
    num_workers: int,
    seed: int,
    show_progress: bool,
    run_name: str,
    combo: dict,
    tgt_normal_csv: str,
    tgt_normal_id_col: str,
    tgt_normal_label_col: str,
    tgt_normal_value: str,
    save_passed_h5ad: bool,
    ilisi_min: float,
    batchkl_max: float,
    ari_min: float,
    asw_cell_type_min: float,
    one_minus_asw_batch_min: float,
) -> pd.DataFrame:
    history_dir = outdir / "history"
    plot_dir = outdir / "plots"
    passed_h5ad_dir = outdir / "passed_h5ad"

    history_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    passed_h5ad_dir.mkdir(parents=True, exist_ok=True)

    history_path = history_dir / f"{run_name}.csv"
    plot_path = plot_dir / f"{run_name}__panel.png"

    t0 = time.time()

    model = CorrectModel(
        batch_key=batch_key,
        disease_key=disease_key,
        normal_value=normal_value,
        epochs=combo["epochs"],
        batch_size=batch_size,
        lr_g=combo["lr_g"],
        lr_d=combo["lr_d"],
        hidden_dim=combo["hidden_dim"],
        cond_dim=combo["cond_dim"],
        g_num_blocks=combo["g_num_blocks"],
        d_num_blocks=combo["d_num_blocks"],
        g_dropout=combo["g_dropout"],
        d_dropout=combo["d_dropout"],
        use_change_gate=combo["use_change_gate"],
        lambda_batch=combo["lambda_batch"],
        lambda_state=combo["lambda_state"],
        lambda_rec=combo["lambda_rec"],
        lambda_id=combo["lambda_id"],
        d_steps=combo["d_steps"],
        g_steps=combo["g_steps"],
        domain_balance_power=combo["domain_balance_power"],
        num_workers=num_workers,
        device=device,
        seed=seed,
        tgt_normal_csv=tgt_normal_csv,
        tgt_normal_id_col=tgt_normal_id_col,
        tgt_normal_label_col=tgt_normal_label_col,
        tgt_normal_value=tgt_normal_value,
    )

    model.fit(
        ref_adata_or_path=ref_h5ad,
        tgt_adata_or_path=tgt_h5ad,
        show_progress=show_progress,
    )

    pd.DataFrame(model.history).to_csv(history_path, index=False)

    rows = []
    plot_adatas: dict[str, ad.AnnData] = {}

    for target_domain in model.domain_names_:
        corrected_joint = model.translate_joint(
            ref_adata_or_path=ref_h5ad,
            tgt_adata_or_path=tgt_h5ad,
            target_domain=str(target_domain),
            ref_bio_key=ref_bio_key,
            tgt_bio_key=bio_key,
            batch_size=align_batch_size,
        )

        metrics_raw = evaluate_corrected_adata(
            adata=corrected_joint,
            batch_key="assay_eval",
            bio_key="bio_label",
            seed=seed,
            eval_rep="X_umap",
            k_metric=30,
            ilisi_beta=1.0,
            batchkl_sample_n=100,
            leiden_res=1.0,
        )

        core_metrics = _extract_core_metrics(metrics_raw)

        passed_filter = _check_save_condition(
            core_metrics=core_metrics,
            ilisi_min=ilisi_min,
            batchkl_max=batchkl_max,
            ari_min=ari_min,
            asw_cell_type_min=asw_cell_type_min,
            one_minus_asw_batch_min=one_minus_asw_batch_min,
        )

        saved_h5ad_path = ""
        if save_passed_h5ad and passed_filter:
            domain_tag = _safe_filename(str(target_domain))
            save_path = passed_h5ad_dir / f"{run_name}__to_{domain_tag}.h5ad"
            corrected_joint.write_h5ad(save_path, compression="gzip")
            saved_h5ad_path = str(save_path)

        plot_adatas[str(target_domain)] = corrected_joint

        row = {
            "run_name": run_name,
            "target_domain": str(target_domain),
            "epochs": combo["epochs"],
            "hidden_dim": combo["hidden_dim"],
            "cond_dim": combo["cond_dim"],
            "g_num_blocks": combo["g_num_blocks"],
            "d_num_blocks": combo["d_num_blocks"],
            "g_dropout": combo["g_dropout"],
            "d_dropout": combo["d_dropout"],
            "use_change_gate": bool(combo["use_change_gate"]),
            "d_steps": combo["d_steps"],
            "g_steps": combo["g_steps"],
            "domain_balance_power": combo["domain_balance_power"],
            "lr_g": combo["lr_g"],
            "lr_d": combo["lr_d"],
            "lambda_batch": combo["lambda_batch"],
            "lambda_state": combo["lambda_state"],
            "lambda_rec": combo["lambda_rec"],
            "lambda_id": combo["lambda_id"],
            "loss_preset": combo["loss_preset"],
            "seed": seed,
            "iLISI": core_metrics["iLISI"],
            "BatchKL": core_metrics["BatchKL"],
            "ARI": core_metrics["ARI"],
            "ASW_cell_type": core_metrics["ASW_cell_type"],
            "ASW_batch": core_metrics["ASW_batch"],
            "one_minus_asw_batch": core_metrics["one_minus_asw_batch"],
            "passed_filter": passed_filter,
            "saved_h5ad_path": saved_h5ad_path,
            "status": "ok",
            "error": "",
        }
        rows.append(row)

    save_domain_panel_plot(
        adata_dict=plot_adatas,
        plot_path=plot_path,
        ref_domain_name=model.ref_domain_name_,
        batch_key="assay_eval",
        bio_key="bio_label",
        seed=seed,
    )

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep hyperparameters for CorrectModel. "
            "For each hyperparameter combo, train once, then translate corrected(ref+tgt) "
            "to every domain in model.domain_names_, compute one metric row per domain, "
            "and save one 2xN UMAP panel."
        )
    )

    parser.add_argument("--ref_h5ad", type=str, required=True)
    parser.add_argument("--tgt_h5ad", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--batch_key", type=str, default="assay")
    parser.add_argument("--disease_key", type=str, default="disease")
    parser.add_argument("--normal_value", type=str, default="normal")
    parser.add_argument("--ref_bio_key", type=str, default="cell_type")
    parser.add_argument("--bio_key", type=str, default="cell_state_label")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--align_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed_list", type=str, default="")
    parser.add_argument("--show_progress", action="store_true", default=False)

    parser.add_argument("--tgt_normal_csv", type=str, default="")
    parser.add_argument("--tgt_normal_id_col", type=str, default="Unnamed: 0")
    parser.add_argument("--tgt_normal_label_col", type=str, default="label")
    parser.add_argument("--tgt_normal_value", type=str, default="0")

    parser.add_argument("--epochs_list", type=str, default="70,100,130")
    parser.add_argument("--hidden_dims", type=str, default="512,256")
    parser.add_argument("--cond_dims", type=str, default="128,64")
    parser.add_argument("--g_num_blocks_list", type=str, default="4,6,8")
    parser.add_argument("--d_num_blocks_list", type=str, default="2,3,4")
    parser.add_argument("--g_dropouts", type=str, default="0,0.1")
    parser.add_argument("--d_dropouts", type=str, default="0,0.1")
    parser.add_argument("--use_change_gate_list", type=str, default="0,1")
    parser.add_argument("--d_steps_list", type=str, default="1")
    parser.add_argument("--g_steps_list", type=str, default="2")
    parser.add_argument("--domain_balance_power_list", type=str, default="0.5")
    parser.add_argument(
        "--loss_presets",
        type=str,
        default="q1_push_translation,q2_balanced_translation,q3_stable_but_not_identity",
    )

    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--index_start", type=int, default=None)
    parser.add_argument("--index_end", type=int, default=None)
    parser.add_argument("--max_runs", type=int, default=None)

    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--stop_on_error", action="store_true", default=False)

    parser.add_argument("--save_passed_h5ad", action="store_true", default=False)

    parser.add_argument("--ilisi_min", type=float, default=1.8)
    parser.add_argument("--batchkl_max", type=float, default=0.5)
    parser.add_argument("--ari_min", type=float, default=0.13)
    parser.add_argument("--asw_cell_type_min", type=float, default=0.25)
    parser.add_argument("--one_minus_asw_batch_min", type=float, default=1.02)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    args.epochs_list = parse_int_list(args.epochs_list)
    args.hidden_dims = parse_int_list(args.hidden_dims)
    args.cond_dims = parse_int_list(args.cond_dims)
    args.g_num_blocks_list = parse_int_list(args.g_num_blocks_list)
    args.d_num_blocks_list = parse_int_list(args.d_num_blocks_list)
    args.g_dropouts = parse_float_list(args.g_dropouts)
    args.d_dropouts = parse_float_list(args.d_dropouts)
    args.use_change_gate_list = parse_bool_list(args.use_change_gate_list)
    args.d_steps_list = parse_int_list(args.d_steps_list)
    args.g_steps_list = parse_int_list(args.g_steps_list)
    args.domain_balance_power_list = parse_float_list(args.domain_balance_power_list)
    args.loss_presets = parse_str_list(args.loss_presets)
    args.seed_list = parse_int_list(args.seed_list) if args.seed_list.strip() else [args.seed]

    for preset in args.loss_presets:
        if preset not in LOSS_PRESETS:
            raise ValueError(f"Unknown loss preset: {preset}")

    outdir = Path(args.outdir)
    output_csv = Path(args.output_csv)
    outdir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_combos = build_combinations(args)
    selected = select_combinations(
        all_combos=all_combos,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        index_start=args.index_start,
        index_end=args.index_end,
        max_runs=args.max_runs,
    )

    selected_runs = [
        (combo_idx, combo, seed)
        for combo_idx, combo in selected
        for seed in args.seed_list
    ]

    existing_run_names = load_existing_run_names(output_csv) if args.skip_existing else set()

    print(f"Ref: {args.ref_h5ad}")
    print(f"Tgt: {args.tgt_h5ad}")
    print(f"Outdir: {outdir}")
    print(f"Output CSV: {output_csv}")
    print(f"Total combos: {len(all_combos)}")
    print(f"Selected combos: {len(selected)}")
    print(f"Seed list ({len(args.seed_list)}): {args.seed_list}")
    print(f"Selected runs (combo x seed): {len(selected_runs)}")
    print(f"Shard: {args.shard_id}/{args.num_shards}")
    print(f"save_passed_h5ad: {args.save_passed_h5ad}")
    print(
        "thresholds:",
        {
            "ilisi_min": args.ilisi_min,
            "batchkl_max": args.batchkl_max,
            "ari_min": args.ari_min,
            "asw_cell_type_min": args.asw_cell_type_min,
            "one_minus_asw_batch_min": args.one_minus_asw_batch_min,
        },
    )

    for i, (combo_idx, combo, seed) in enumerate(selected_runs, start=1):
        run_name = make_run_name(combo_idx, combo, seed=seed)

        if args.skip_existing and run_name in existing_run_names:
            print(f"[{i}/{len(selected_runs)}] skip existing: {run_name}")
            continue

        print("=" * 100)
        print(f"[{i}/{len(selected_runs)}] {run_name}")
        print(combo)
        print({"seed": seed})

        seed_everything(seed)
        t0 = time.time()

        try:
            df_rows = run_one_combo(
                ref_h5ad=args.ref_h5ad,
                tgt_h5ad=args.tgt_h5ad,
                outdir=outdir,
                batch_key=args.batch_key,
                disease_key=args.disease_key,
                normal_value=args.normal_value,
                ref_bio_key=args.ref_bio_key,
                bio_key=args.bio_key,
                device=args.device,
                batch_size=args.batch_size,
                align_batch_size=args.align_batch_size,
                num_workers=args.num_workers,
                seed=seed,
                show_progress=args.show_progress,
                run_name=run_name,
                combo=combo,
                tgt_normal_csv=args.tgt_normal_csv,
                tgt_normal_id_col=args.tgt_normal_id_col,
                tgt_normal_label_col=args.tgt_normal_label_col,
                tgt_normal_value=args.tgt_normal_value,
                save_passed_h5ad=args.save_passed_h5ad,
                ilisi_min=args.ilisi_min,
                batchkl_max=args.batchkl_max,
                ari_min=args.ari_min,
                asw_cell_type_min=args.asw_cell_type_min,
                one_minus_asw_batch_min=args.one_minus_asw_batch_min,
            )

            append_df_to_csv(output_csv, df_rows)
            existing_run_names.add(run_name)

            n_passed = int(df_rows["passed_filter"].sum()) if "passed_filter" in df_rows.columns else 0
            print(f"Appended {len(df_rows)} rows to: {output_csv}")
            print(f"Passed domains in this run: {n_passed}")

        except Exception as e:
            err_df = pd.DataFrame(
                [
                    {
                        "run_name": run_name,
                        "target_domain": "",
                        "epochs": combo["epochs"],
                        "hidden_dim": combo["hidden_dim"],
                        "cond_dim": combo["cond_dim"],
                        "g_num_blocks": combo["g_num_blocks"],
                        "d_num_blocks": combo["d_num_blocks"],
                        "g_dropout": combo["g_dropout"],
                        "d_dropout": combo["d_dropout"],
                        "use_change_gate": bool(combo["use_change_gate"]),
                        "d_steps": combo["d_steps"],
                        "g_steps": combo["g_steps"],
                        "domain_balance_power": combo["domain_balance_power"],
                        "lr_g": combo["lr_g"],
                        "lr_d": combo["lr_d"],
                        "lambda_batch": combo["lambda_batch"],
                        "lambda_state": combo["lambda_state"],
                        "lambda_rec": combo["lambda_rec"],
                        "lambda_id": combo["lambda_id"],
                        "loss_preset": combo["loss_preset"],
                        "seed": seed,
                        "iLISI": np.nan,
                        "BatchKL": np.nan,
                        "ARI": np.nan,
                        "ASW_cell_type": np.nan,
                        "ASW_batch": np.nan,
                        "one_minus_asw_batch": np.nan,
                        "passed_filter": False,
                        "saved_h5ad_path": "",
                        "status": "error",
                        "error": repr(e),
                    }
                ]
            )

            append_df_to_csv(output_csv, err_df)
            existing_run_names.add(run_name)

            print(f"[ERROR] {run_name}: {repr(e)}")
            print(f"Appended error row to: {output_csv}")

            if args.stop_on_error:
                raise

        finally:
            gc.collect()
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    print("=" * 100)
    print("Sweep finished.")
    print(f"Final CSV: {output_csv}")


if __name__ == "__main__":
    main()