import os
import sys
import json
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
from scipy import sparse
from .model import IGReferencePairModule
from .utils import seed_everything


class Phase1EncoderWrapper(nn.Module):
    """
    Wrap a full Phase-I generator when only G.encode(x) is available.
    """

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        return self.generator.encode(x)


def load_phase1_generator_and_encoder(ckpt_path: str):
    """
    Load the pretrained Phase-I generator and return its frozen encoder.

    This function assumes phase1_G.pth was saved as a full torch module.
    If it was saved as a pure state_dict, you need to instantiate
    GeneratorWithMemory first and then load_state_dict.
    """
    print(f"Loading Phase-I checkpoint: {ckpt_path}")

    obj = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(obj, nn.Module):
        G = obj

    elif isinstance(obj, dict):
        if "G" in obj and isinstance(obj["G"], nn.Module):
            G = obj["G"]
        elif "generator" in obj and isinstance(obj["generator"], nn.Module):
            G = obj["generator"]
        elif "model" in obj and isinstance(obj["model"], nn.Module):
            G = obj["model"]
        else:
            keys = list(obj.keys())
            raise ValueError(
                "The checkpoint is a dictionary, but no full generator module "
                "was found under keys 'G', 'generator', or 'model'. "
                f"Available keys: {keys}. "
                "If this is a state_dict, instantiate GeneratorWithMemory with "
                "the original architecture and then load_state_dict."
            )
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)

    # Prefer the actual encoder module.
    if hasattr(G, "extractor") and hasattr(G.extractor, "encoder"):
        encoder = G.extractor.encoder
        print("Using encoder: G.extractor.encoder")

    elif hasattr(G, "encoder") and isinstance(G.encoder, nn.Module):
        encoder = G.encoder
        print("Using encoder: G.encoder")

    elif hasattr(G, "encode"):
        encoder = Phase1EncoderWrapper(G)
        print("Using encoder wrapper around G.encode(x)")

    else:
        raise AttributeError("Cannot find a usable encoder from the loaded Phase-I generator.")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    return G, encoder


def infer_first_linear_input_dim(module: nn.Module):
    """
    Infer the expected input dimension from the first Linear layer.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    return None


def align_gene_order(adata_ref, adata_tgt):
    """
    Align target genes to reference gene order.

    The safest case is that ref and tgt already have identical var_names.
    If not, this function uses shared genes in the reference order.
    """
    ref_genes = adata_ref.var_names.astype(str)
    tgt_genes = adata_tgt.var_names.astype(str)

    if np.array_equal(ref_genes, tgt_genes):
        print("Reference and target gene names are already identical.")
        return adata_ref, adata_tgt

    print("Warning: reference and target var_names are not identical.")
    print("Aligning by shared genes in reference gene order.")

    tgt_gene_set = set(tgt_genes)
    common_genes = [g for g in ref_genes if g in tgt_gene_set]

    if len(common_genes) == 0:
        raise ValueError("No shared genes between reference and target.")

    adata_ref = adata_ref[:, common_genes].copy()
    adata_tgt = adata_tgt[:, common_genes].copy()

    print(f"Shared genes after alignment: {len(common_genes)}")

    return adata_ref, adata_tgt


def load_ref_tgt_and_normal_subset(ref_path: str, tgt_path: str, pred_path: str):
    """
    Load reference and target h5ad files, then keep only predicted-normal target cells.
    """
    print(f"Loading reference h5ad: {ref_path}")
    adata_ref = sc.read_h5ad(ref_path)

    print(f"Loading target h5ad: {tgt_path}")
    adata_tgt = sc.read_h5ad(tgt_path)

    print(f"Loading Phase-I prediction CSV: {pred_path}")
    pred = pd.read_csv(pred_path)

    required_cols = {"cell_id", "pred"}
    missing = required_cols - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")

    if not adata_ref.obs_names.is_unique:
        raise ValueError("Reference obs_names are not unique.")

    if not adata_tgt.obs_names.is_unique:
        raise ValueError("Target obs_names are not unique.")

    pred["cell_id"] = pred["cell_id"].astype(str)
    pred["pred"] = pred["pred"].astype(str).str.lower()

    normal_ids = set(pred.loc[pred["pred"] == "normal", "cell_id"])

    target_ids = adata_tgt.obs_names.astype(str)
    normal_mask = target_ids.isin(normal_ids)

    n_normal = int(normal_mask.sum())

    print(f"Reference cells: {adata_ref.n_obs}")
    print(f"Target cells: {adata_tgt.n_obs}")
    print(f"Predicted-normal target cells matched in h5ad: {n_normal}")

    if n_normal == 0:
        print("Example target obs_names:")
        print(list(adata_tgt.obs_names[:5]))
        print("Example prediction cell_id:")
        print(list(pred["cell_id"].head(5)))
        raise ValueError(
            "No predicted-normal target cells matched target obs_names. "
            "Check whether pred['cell_id'] matches adata_tgt.obs_names."
        )

    adata_tgt_normal = adata_tgt[normal_mask].copy()

    adata_ref, adata_tgt_normal = align_gene_order(adata_ref, adata_tgt_normal)

    X_ref = adata_ref.X
    X_tgt_normal = adata_tgt_normal.X

    print(f"X_ref shape: {X_ref.shape}")
    print(f"X_tgt_normal shape: {X_tgt_normal.shape}")

    return adata_ref, adata_tgt_normal, X_ref, X_tgt_normal


def save_pair_outputs(pair_df, out_dir):
    """
    Save pair output dataframe.
    """
    pair_path = os.path.join(out_dir, "target_ref_pairs_ig.csv")
    pair_df.to_csv(pair_path, index=False)
    print(f"Saved IG pair results to: {pair_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="ckpt/phase1_G.pth",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default="data/ref_clean_colorectum.h5ad",
    )
    parser.add_argument(
        "--tgt_path",
        type=str,
        default="data/tgt_clean_colorectum.h5ad",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        default="output/phase1_pred_ASCs.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/pair_module",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--attn_dim", type=int, default=None)

    # Default is True. Use --no_normalize_latent to turn it off.
    parser.add_argument("--normalize_latent", dest="normalize_latent", action="store_true")
    parser.add_argument("--no_normalize_latent", dest="normalize_latent", action="store_false")
    parser.set_defaults(normalize_latent=True)

    parser.add_argument("--run_ig", action="store_true")
    parser.add_argument("--ig_steps", type=int, default=32)
    parser.add_argument("--internal_batch_size", type=int, default=4)
    parser.add_argument("--score_mode", type=str, default="positive", choices=["positive", "abs"])
    parser.add_argument("--max_targets", type=int, default=None)

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Save arguments for reproducibility.
    args_path = os.path.join(args.out_dir, "train_pair_args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved arguments to: {args_path}")

    print("Loading pretrained Phase-I generator and encoder...")
    G, phase1_encoder = load_phase1_generator_and_encoder(args.ckpt_path)

    print("Loading reference, target, and predicted-normal target cells...")
    adata_ref, adata_tgt_normal, X_ref, X_tgt_normal = load_ref_tgt_and_normal_subset(
        ref_path=args.ref_path,
        tgt_path=args.tgt_path,
        pred_path=args.pred_path,
    )

    expected_input_dim = infer_first_linear_input_dim(phase1_encoder)
    if expected_input_dim is not None:
        print(f"Encoder expected input_dim: {expected_input_dim}")
        print(f"Current data n_vars: {X_ref.shape[1]}")

        if expected_input_dim != X_ref.shape[1]:
            raise ValueError(
                f"Input dimension mismatch: encoder expects {expected_input_dim}, "
                f"but current data has {X_ref.shape[1]} genes. "
                "You need to use the same gene set and gene order as Phase-I training."
            )

    matcher = IGReferencePairModule(
        phase1_encoder=phase1_encoder,
        device=args.device,
        attn_dim=args.attn_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        normalize_latent=args.normalize_latent,
    )

    print("Training pair reconstruction module...")
    matcher.fit(X_ref, X_tgt_normal)

    reconstructor_path = os.path.join(args.out_dir, "pair_reconstructor.pt")
    torch.save(
        {
            "reconstructor_state_dict": matcher.reconstructor.state_dict(),
            "history": matcher.history,
            "attn_dim": args.attn_dim,
            "normalize_latent": args.normalize_latent,
            "z_ref_shape": tuple(matcher.z_ref.shape),
            "z_tgt_shape": tuple(matcher.z_tgt.shape),
            "ref_path": args.ref_path,
            "tgt_path": args.tgt_path,
            "pred_path": args.pred_path,
            "ckpt_path": args.ckpt_path,
        },
        reconstructor_path,
    )
    print(f"Saved trained reconstructor to: {reconstructor_path}")

    history_path = os.path.join(args.out_dir, "pair_reconstructor_history.csv")
    pd.DataFrame(
        {
            "epoch": np.arange(1, len(matcher.history) + 1),
            "recon_loss": matcher.history,
        }
    ).to_csv(history_path, index=False)
    print(f"Saved training history to: {history_path}")

    if args.run_ig:
        print("Running IG-based target-reference matching...")
        results = matcher.match_all(
            ig_steps=args.ig_steps,
            internal_batch_size=args.internal_batch_size,
            score_mode=args.score_mode,
            max_targets=args.max_targets,
        )

        pair_df = pd.DataFrame(results)

        pair_df["target_cell_id"] = [
            adata_tgt_normal.obs_names[i]
            for i in pair_df["target_index"].astype(int).values
        ]
        pair_df["paired_ref_cell_id"] = [
            adata_ref.obs_names[i]
            for i in pair_df["pair_index"].astype(int).values
        ]
        pair_df["attention_ref_cell_id"] = [
            adata_ref.obs_names[i]
            for i in pair_df["attention_pair_index"].astype(int).values
        ]

        # Put important columns first.
        front_cols = [
            "target_cell_id",
            "paired_ref_cell_id",
            "attention_ref_cell_id",
            "target_index",
            "pair_index",
            "attention_pair_index",
            "pair_score",
            "attention_pair_weight",
            "recon_mse",
        ]
        other_cols = [c for c in pair_df.columns if c not in front_cols]
        pair_df = pair_df[front_cols + other_cols]

        save_pair_outputs(pair_df, args.out_dir)

    else:
        print("Skipped IG matching. Add --run_ig if you want to generate pairs now.")


if __name__ == "__main__":
    main()