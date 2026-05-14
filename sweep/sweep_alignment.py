# sweep_alignment.py

from __future__ import annotations

import argparse
import gc
import itertools
import os
import random
import time
import traceback
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from metrix.calculate_adata import evaluate_corrected_adata

try:
    from .train_alignment_gan import train as train_alignment_gan
except ImportError:
    from train_alignment_gan import train as train_alignment_gan


LOSS_PRESETS = {
    "no_tgt_batch": {
        "lambda_adv": 1.0,
        "lambda_pair": 0.0,
        "lambda_con": 0.0,
        "lambda_tgt_batch_D": 1.0,
        "lambda_tgt_batch_G": 0.1,
        "tgt_batch_warmup_epochs": 5,
        "gamma": 0.1,
        "tau": 1.0,
    },

    "batch_conf": {
        "lambda_adv": 2.0,
        "lambda_pair": 0.0,
        "lambda_con": 0.0,
        "lambda_tgt_batch_D": 1.0,
        "lambda_tgt_batch_G": 0.5,
        "tgt_batch_warmup_epochs": 5,
        "gamma": 0.1,
        "tau": 1.0,
    },

    "batch_conf_mid": {
        "lambda_adv": 1.0,
        "lambda_pair": 0.0,
        "lambda_con": 0.0,
        "lambda_tgt_batch_D": 1.0,
        "lambda_tgt_batch_G": 0.1,
        "tgt_batch_warmup_epochs": 5,
        "gamma": 0.1,
        "tau": 1.0,
    },

    "batch_conf_strong": {
        "lambda_adv": 1.0,
        "lambda_pair": 0.0,
        "lambda_con": 0.0,
        "lambda_tgt_batch_D": 1.0,
        "lambda_tgt_batch_G": 0.1,
        "tgt_batch_warmup_epochs": 5,
        "gamma": 0.1,
        "tau": 1.0,
    },

    "adv_pair": {
        "lambda_adv": 1.0,
        "lambda_pair": 0.05,
        "lambda_con": 0.05,
        "lambda_tgt_batch_D": 1.0,
        "lambda_tgt_batch_G": 0.01,
        "tgt_batch_warmup_epochs": 5,
        "gamma": 0.05,
        "tau": 1.0,
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

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


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
    return f"{x:g}".replace("-", "m").replace(".", "p")


def make_run_name(combo_idx: int, combo: dict, seed: int) -> str:
    attn = int(bool(combo["use_attention_ref"]))

    return (
        f"run_{combo_idx:05d}"
        f"__e{combo['epochs']}"
        f"__hd{combo['hidden_dim']}"
        f"__gb{combo['g_blocks']}"
        f"__db{combo['d_blocks']}"
        f"__cd{combo['cond_dim']}"
        f"__ge{combo['g_expansion']}"
        f"__gdo{fmt_float_tag(combo['g_dropout'])}"
        f"__ddo{fmt_float_tag(combo['d_dropout'])}"
        f"__ds{fmt_float_tag(combo['delta_scale'])}"
        f"__nc{combo['n_candidates']}"
        f"__attn{attn}"
        f"__lrg{fmt_float_tag(combo['lr_g'])}"
        f"__lrd{fmt_float_tag(combo['lr_d'])}"
        f"__{combo['loss_preset']}"
        f"__lbd{fmt_float_tag(combo['lambda_tgt_batch_D'])}"
        f"__lbg{fmt_float_tag(combo['lambda_tgt_batch_G'])}"
        f"__bw{combo['tgt_batch_warmup_epochs']}"
        f"__sd{seed}"
    )


def build_combinations(args) -> list[dict]:
    combos = []

    for (
        epochs,
        hidden_dim,
        g_blocks,
        d_blocks,
        g_dropout,
        d_dropout,
        delta_scale,
        n_candidates,
        use_attention_ref,
        lr_g,
        lr_d,
        weight_decay,
        loss_preset,
    ) in itertools.product(
        args.epochs_list,
        args.hidden_dims,
        args.g_blocks_list,
        args.d_blocks_list,
        args.g_dropouts,
        args.d_dropouts,
        args.delta_scales,
        args.n_candidates_list,
        args.use_attention_ref_list,
        args.lr_g_list,
        args.lr_d_list,
        args.weight_decay_list,
        args.loss_presets,
    ):
        combo = {
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "g_blocks": g_blocks,
            "d_blocks": d_blocks,
            "cond_dim": args.cond_dim,
            "g_expansion": args.g_expansion,
            "g_dropout": g_dropout,
            "d_dropout": d_dropout,
            "delta_scale": delta_scale,
            "n_candidates": n_candidates,
            "use_attention_ref": use_attention_ref,
            "lr_g": lr_g,
            "lr_d": lr_d,
            "weight_decay": weight_decay,
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

    merged_cols = old_cols[:]
    for col in new_cols:
        if col not in merged_cols:
            merged_cols.append(col)

    df_old = df_old.reindex(columns=merged_cols)
    df_new = df_new.reindex(columns=merged_cols)

    pd.concat([df_old, df_new], axis=0, ignore_index=True).to_csv(
        csv_path,
        index=False,
    )


def load_existing_run_names(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    try:
        df = pd.read_csv(csv_path, usecols=["run_name"])
        return set(df["run_name"].astype(str))
    except Exception:
        return set()


def get_first_existing_metric(metrics: dict, candidate_keys: list[str]) -> float:
    for k in candidate_keys:
        if k in metrics and pd.notna(metrics[k]):
            return float(metrics[k])

    raise KeyError(
        f"None of these metric keys were found: {candidate_keys}. "
        f"Available keys: {list(metrics.keys())}"
    )


def extract_five_metrics(metrics: dict) -> dict:
    ilisi = get_first_existing_metric(
        metrics,
        ["iLISI", "iLISI(assay)", "iLISI(assay_eval)"],
    )

    batchkl = get_first_existing_metric(
        metrics,
        ["BatchKL", "BatchKL(assay)", "BatchKL(assay_eval)"],
    )

    ari = get_first_existing_metric(
        metrics,
        [
            "ARI",
            "ARI(Leiden vs bio_label)",
            "ARI(Leiden vs cell_type)",
            "ARI(Leiden vs cell_state_label)",
        ],
    )

    asw_cell_type = get_first_existing_metric(
        metrics,
        [
            "ASW_cell_type",
            "ASW_bio_label",
            "ASW_bio",
            "ASW(bio_label)",
        ],
    )

    one_minus_asw_batch = get_first_existing_metric(
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
    }


def check_gene_order(ref_adata: ad.AnnData, tgt_adata: ad.AnnData) -> None:
    if ref_adata.n_vars != tgt_adata.n_vars:
        raise ValueError(
            f"Gene number mismatch: ref={ref_adata.n_vars}, tgt={tgt_adata.n_vars}"
        )

    if not np.array_equal(ref_adata.var_names, tgt_adata.var_names):
        raise ValueError("Gene order mismatch between ref and aligned target.")


def build_joint_adata_for_eval_in_memory(
    ref_h5ad: str,
    aligned_tgt: ad.AnnData,
    batch_key: str,
    ref_bio_key: str,
    tgt_bio_key: str,
    ref_domain_name: str,
    tgt_domain_name: str,
) -> ad.AnnData:
    """
    Build joint AnnData from reference h5ad and in-memory aligned target AnnData.

    Evaluation batch labels:
        ref cells       -> ref_batch
        target assay A  -> tgt_batch_A
        target assay B  -> tgt_batch_B
        ...
    """
    ref = ad.read_h5ad(ref_h5ad)
    tgt = aligned_tgt.copy()

    check_gene_order(ref, tgt)

    ref = ref.copy()

    ref.obs["assay_eval"] = "ref_batch"

    if batch_key in tgt.obs.columns:
        tgt_assay = tgt.obs[batch_key].astype(str).values
        tgt.obs["assay_eval"] = np.asarray(
            [f"tgt_batch_{x}" for x in tgt_assay],
            dtype=object,
        )
    else:
        tgt.obs["assay_eval"] = "tgt_batch_unknown"

    if ref_bio_key not in ref.obs.columns:
        raise KeyError(f"ref_bio_key={ref_bio_key} not found in ref.obs")

    if tgt_bio_key not in tgt.obs.columns:
        raise KeyError(f"tgt_bio_key={tgt_bio_key} not found in target.obs")

    ref.obs["bio_label"] = ref.obs[ref_bio_key].astype(str).values
    tgt.obs["bio_label"] = tgt.obs[tgt_bio_key].astype(str).values

    ref.obs["source_eval"] = ref_domain_name
    tgt.obs["source_eval"] = tgt_domain_name

    joint = ad.concat(
        [ref, tgt],
        axis=0,
        join="inner",
        merge="same",
        label="source",
        keys=["ref", "target"],
        index_unique="-",
    )

    joint.obs["assay_eval"] = joint.obs["assay_eval"].astype("category")
    joint.obs["bio_label"] = joint.obs["bio_label"].astype("category")

    print("[Eval assay_eval counts]")
    print(joint.obs["assay_eval"].value_counts())

    return joint


def compute_metrics_on_joint(
    joint: ad.AnnData,
    seed: int,
    k_metric: int,
    batchkl_sample_n: int,
    leiden_res: float,
) -> dict:
    adata = joint.copy()

    sc.pp.pca(adata, random_state=seed)
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.umap(adata, random_state=seed)

    metrics_raw = evaluate_corrected_adata(
        adata=adata,
        batch_key="assay_eval",
        bio_key="bio_label",
        seed=seed,
        eval_rep="X_umap",
        k_metric=k_metric,
        ilisi_beta=1.0,
        batchkl_sample_n=batchkl_sample_n,
        leiden_res=leiden_res,
    )

    return extract_five_metrics(metrics_raw)


def make_train_args(
    base_args,
    combo: dict,
    run_dir: Path,
    seed: int,
) -> Namespace:
    return Namespace(
        target_h5ad=base_args.target_h5ad,
        ref_h5ad=base_args.ref_h5ad,
        pair_csv=base_args.pair_csv,
        out_dir=str(run_dir),
        phase1_ckpt=base_args.phase1_ckpt,

        # Conditional generator:
        # train_alignment_gan.py uses this as tgt_batch_key.
        tgt_batch_key=base_args.batch_key,
        cond_dim=combo["cond_dim"],
        g_expansion=combo["g_expansion"],

        hidden_dim=combo["hidden_dim"],
        g_blocks=combo["g_blocks"],
        d_blocks=combo["d_blocks"],
        g_dropout=combo["g_dropout"],
        d_dropout=combo["d_dropout"],
        delta_scale=combo["delta_scale"],

        phase1_hidden_dim=base_args.phase1_hidden_dim,
        phase1_latent_dim=base_args.phase1_latent_dim,
        phase1_memory_size=base_args.phase1_memory_size,
        phase1_num_heads=base_args.phase1_num_heads,
        phase1_temperature=base_args.phase1_temperature,
        phase1_dropout=base_args.phase1_dropout,
        phase1_use_memory_bank=base_args.phase1_use_memory_bank,

        lambda_adv=combo["lambda_adv"],
        lambda_pair=combo["lambda_pair"],
        lambda_con=combo["lambda_con"],

        # New target-source adversarial head parameters.
        lambda_tgt_batch_D=combo["lambda_tgt_batch_D"],
        lambda_tgt_batch_G=combo["lambda_tgt_batch_G"],
        tgt_batch_warmup_epochs=combo["tgt_batch_warmup_epochs"],

        gamma=combo["gamma"],
        tau=combo["tau"],

        n_candidates=combo["n_candidates"],
        use_attention_ref=combo["use_attention_ref"],

        epochs=combo["epochs"],
        batch_size=base_args.batch_size,
        lr_g=combo["lr_g"],
        lr_d=combo["lr_d"],
        weight_decay=combo["weight_decay"],
        grad_clip=base_args.grad_clip,

        num_workers=base_args.num_workers,
        pin_memory=base_args.pin_memory,
        save_every=base_args.save_every,
        log_every=base_args.log_every,
        seed=seed,
        device=base_args.device,

        # Sweep mode:
        # do not save h5ad/checkpoints.
        # return aligned AnnData in memory.
        export_aligned=False,
        return_aligned=True,
        save_model=False,
        save_args=False,
    )


def run_one_combo(
    args,
    combo_idx: int,
    combo: dict,
    seed: int,
    run_name: str,
) -> pd.DataFrame:
    run_dir = Path(args.outdir) / "runs" / run_name

    t0 = time.time()

    train_args = make_train_args(
        base_args=args,
        combo=combo,
        run_dir=run_dir,
        seed=seed,
    )

    train_out = train_alignment_gan(train_args)

    if not isinstance(train_out, dict):
        raise RuntimeError(
            "train_alignment_gan should return a dict. "
            "Please make sure train_alignment_gan.py is the updated version."
        )

    aligned_tgt = train_out.get("aligned_adata", None)

    if aligned_tgt is None:
        raise RuntimeError(
            "train_alignment_gan did not return aligned_adata. "
            "Please set return_aligned=True inside make_train_args()."
        )

    joint = build_joint_adata_for_eval_in_memory(
        ref_h5ad=args.ref_h5ad,
        aligned_tgt=aligned_tgt,
        batch_key=args.batch_key,
        ref_bio_key=args.ref_bio_key,
        tgt_bio_key=args.tgt_bio_key,
        ref_domain_name=args.ref_domain_name,
        tgt_domain_name=args.tgt_domain_name,
    )

    metrics = compute_metrics_on_joint(
        joint=joint,
        seed=seed,
        k_metric=args.k_metric,
        batchkl_sample_n=args.batchkl_sample_n,
        leiden_res=args.leiden_res,
    )

    elapsed_sec = time.time() - t0

    row = {
        "run_name": run_name,
        "combo_idx": combo_idx,
        "seed": seed,

        "epochs": combo["epochs"],
        "hidden_dim": combo["hidden_dim"],
        "g_blocks": combo["g_blocks"],
        "d_blocks": combo["d_blocks"],
        "cond_dim": combo["cond_dim"],
        "g_expansion": combo["g_expansion"],
        "g_dropout": combo["g_dropout"],
        "d_dropout": combo["d_dropout"],
        "delta_scale": combo["delta_scale"],
        "n_candidates": combo["n_candidates"],
        "use_attention_ref": bool(combo["use_attention_ref"]),
        "lr_g": combo["lr_g"],
        "lr_d": combo["lr_d"],
        "weight_decay": combo["weight_decay"],

        "batch_key": args.batch_key,
        "ref_bio_key": args.ref_bio_key,
        "tgt_bio_key": args.tgt_bio_key,

        "loss_preset": combo["loss_preset"],
        "lambda_adv": combo["lambda_adv"],
        "lambda_pair": combo["lambda_pair"],
        "lambda_con": combo["lambda_con"],
        "lambda_tgt_batch_D": combo["lambda_tgt_batch_D"],
        "lambda_tgt_batch_G": combo["lambda_tgt_batch_G"],
        "tgt_batch_warmup_epochs": combo["tgt_batch_warmup_epochs"],
        "gamma": combo["gamma"],
        "tau": combo["tau"],

        "iLISI": metrics["iLISI"],
        "BatchKL": metrics["BatchKL"],
        "ARI": metrics["ARI"],
        "ASW_cell_type": metrics["ASW_cell_type"],
        "ASW_batch": metrics["ASW_batch"],

        "elapsed_sec": elapsed_sec,
        "aligned_tgt_path": "",
        "status": "ok",
        "error": "",
    }

    del train_out
    del aligned_tgt
    del joint

    return pd.DataFrame([row])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep hyperparameters for pair-guided alignment-GAN and return five metrics."
    )

    parser.add_argument("--target_h5ad", type=str, required=True)
    parser.add_argument("--ref_h5ad", type=str, required=True)
    parser.add_argument("--pair_csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument(
        "--phase1_ckpt",
        type=str,
        default="/data1011/yuzimu/M2ASDA/ckpt/phase1_G.pth",
    )

    # batch_key has two roles here:
    # 1. train_alignment_gan.py uses it as tgt_batch_key to build G(x, source_domain)
    # 2. evaluation uses it to build assay_eval for target batches
    parser.add_argument("--batch_key", type=str, default="assay")

    parser.add_argument("--ref_bio_key", type=str, default="cell_type")
    parser.add_argument("--tgt_bio_key", type=str, default="cell_state_label")
    parser.add_argument("--ref_domain_name", type=str, default="ref")
    parser.add_argument("--tgt_domain_name", type=str, default="target")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed_list", type=str, default="")

    parser.add_argument("--phase1_hidden_dim", type=int, default=512)
    parser.add_argument("--phase1_latent_dim", type=int, default=128)
    parser.add_argument("--phase1_memory_size", type=int, default=512)
    parser.add_argument("--phase1_num_heads", type=int, default=8)
    parser.add_argument("--phase1_temperature", type=float, default=1.0)
    parser.add_argument("--phase1_dropout", type=float, default=0.0)
    parser.add_argument("--phase1_use_memory_bank", action="store_true", default=False)

    parser.add_argument("--epochs_list", type=str, default="50,80,100")
    parser.add_argument("--hidden_dims", type=str, default="512")
    parser.add_argument("--g_blocks_list", type=str, default="3,4,6")
    parser.add_argument("--d_blocks_list", type=str, default="2,3")

    # Conditional generator fixed parameters.
    # 如果以后要 sweep cond_dim/g_expansion，可以再改成 list。
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--g_expansion", type=int, default=4)

    parser.add_argument("--g_dropouts", type=str, default="0,0.1")
    parser.add_argument("--d_dropouts", type=str, default="0.1,0.2")

    # Conditional residual generator 建议先从小 delta_scale 开始。
    parser.add_argument("--delta_scales", type=str, default="0.1,0.2,0.5")

    parser.add_argument("--n_candidates_list", type=str, default="8,16,32")
    parser.add_argument("--use_attention_ref_list", type=str, default="0")
    parser.add_argument("--lr_g_list", type=str, default="1e-4")
    parser.add_argument("--lr_d_list", type=str, default="1e-4")
    parser.add_argument("--weight_decay_list", type=str, default="1e-4")

    # 推荐先只跑 batch_conf，看新加的 tgt-batch confusion 是否有用。
    parser.add_argument("--loss_presets", type=str, default="batch_conf,batch_conf_strong")

    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=999999)
    parser.add_argument("--log_every", type=int, default=50)

    parser.add_argument("--k_metric", type=int, default=30)
    parser.add_argument("--batchkl_sample_n", type=int, default=100)
    parser.add_argument("--leiden_res", type=float, default=1.0)

    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--index_start", type=int, default=None)
    parser.add_argument("--index_end", type=int, default=None)
    parser.add_argument("--max_runs", type=int, default=None)

    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--stop_on_error", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    seed_everything(args.seed)

    args.epochs_list = parse_int_list(args.epochs_list)
    args.hidden_dims = parse_int_list(args.hidden_dims)
    args.g_blocks_list = parse_int_list(args.g_blocks_list)
    args.d_blocks_list = parse_int_list(args.d_blocks_list)
    args.g_dropouts = parse_float_list(args.g_dropouts)
    args.d_dropouts = parse_float_list(args.d_dropouts)
    args.delta_scales = parse_float_list(args.delta_scales)
    args.n_candidates_list = parse_int_list(args.n_candidates_list)
    args.use_attention_ref_list = parse_bool_list(args.use_attention_ref_list)
    args.lr_g_list = parse_float_list(args.lr_g_list)
    args.lr_d_list = parse_float_list(args.lr_d_list)
    args.weight_decay_list = parse_float_list(args.weight_decay_list)
    args.loss_presets = parse_str_list(args.loss_presets)
    args.seed_list = parse_int_list(args.seed_list) if args.seed_list.strip() else [args.seed]

    for preset in args.loss_presets:
        if preset not in LOSS_PRESETS:
            raise ValueError(
                f"Unknown loss preset: {preset}. "
                f"Available presets: {sorted(LOSS_PRESETS.keys())}"
            )

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
    print(f"Target: {args.target_h5ad}")
    print(f"Pair CSV: {args.pair_csv}")
    print(f"Phase-I ckpt: {args.phase1_ckpt}")
    print(f"Outdir: {outdir}")
    print(f"Output CSV: {output_csv}")
    print(f"Batch key / tgt_batch_key: {args.batch_key}")
    print(f"cond_dim: {args.cond_dim}")
    print(f"g_expansion: {args.g_expansion}")
    print(f"Loss presets: {args.loss_presets}")
    print(f"Total combos: {len(all_combos)}")
    print(f"Selected combos: {len(selected)}")
    print(f"Seed list: {args.seed_list}")
    print(f"Selected runs: {len(selected_runs)}")
    print(f"Shard: {args.shard_id}/{args.num_shards}")
    print("Save h5ad per run: False")
    print("Save checkpoint per run: False")

    for i, (combo_idx, combo, seed) in enumerate(selected_runs, start=1):
        run_name = make_run_name(combo_idx, combo, seed)

        if args.skip_existing and run_name in existing_run_names:
            print(f"[{i}/{len(selected_runs)}] skip existing: {run_name}")
            continue

        print("=" * 100)
        print(f"[{i}/{len(selected_runs)}] {run_name}")
        print(combo)
        print({"seed": seed})

        seed_everything(seed)

        try:
            df_row = run_one_combo(
                args=args,
                combo_idx=combo_idx,
                combo=combo,
                seed=seed,
                run_name=run_name,
            )

            append_df_to_csv(output_csv, df_row)
            existing_run_names.add(run_name)

            print(f"Appended result to: {output_csv}")
            print(
                df_row[
                    [
                        "iLISI",
                        "BatchKL",
                        "ARI",
                        "ASW_cell_type",
                        "ASW_batch",
                    ]
                ].to_string(index=False)
            )

        except Exception as e:
            traceback.print_exc()

            err_row = {
                "run_name": run_name,
                "combo_idx": combo_idx,
                "seed": seed,

                "epochs": combo["epochs"],
                "hidden_dim": combo["hidden_dim"],
                "g_blocks": combo["g_blocks"],
                "d_blocks": combo["d_blocks"],
                "cond_dim": combo["cond_dim"],
                "g_expansion": combo["g_expansion"],
                "g_dropout": combo["g_dropout"],
                "d_dropout": combo["d_dropout"],
                "delta_scale": combo["delta_scale"],
                "n_candidates": combo["n_candidates"],
                "use_attention_ref": bool(combo["use_attention_ref"]),
                "lr_g": combo["lr_g"],
                "lr_d": combo["lr_d"],
                "weight_decay": combo["weight_decay"],

                "batch_key": args.batch_key,
                "ref_bio_key": args.ref_bio_key,
                "tgt_bio_key": args.tgt_bio_key,

                "loss_preset": combo["loss_preset"],
                "lambda_adv": combo["lambda_adv"],
                "lambda_pair": combo["lambda_pair"],
                "lambda_con": combo["lambda_con"],
                "lambda_tgt_batch_D": combo["lambda_tgt_batch_D"],
                "lambda_tgt_batch_G": combo["lambda_tgt_batch_G"],
                "tgt_batch_warmup_epochs": combo["tgt_batch_warmup_epochs"],
                "gamma": combo["gamma"],
                "tau": combo["tau"],

                "iLISI": np.nan,
                "BatchKL": np.nan,
                "ARI": np.nan,
                "ASW_cell_type": np.nan,
                "ASW_batch": np.nan,

                "elapsed_sec": np.nan,
                "aligned_tgt_path": "",
                "status": "error",
                "error": repr(e),
            }

            append_df_to_csv(output_csv, pd.DataFrame([err_row]))
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