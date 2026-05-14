import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy import sparse

from .model import Generator_phase2, Subtyper
from .utils import seed_everything


def dense(X):
    return X.toarray().astype(np.float32) if sparse.issparse(X) else np.asarray(X, dtype=np.float32)


def load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def align_genes(ref, tgt):
    genes = [g for g in ref.var_names.astype(str) if g in set(tgt.var_names.astype(str))]
    return ref[:, genes].copy(), tgt[:, genes].copy()


def read_asc_cells(pred_csv, tgt_names):
    df = pd.read_csv(pred_csv)
    cell_col = "cell_id" if "cell_id" in df.columns else df.columns[0]
    label_col = "pred" if "pred" in df.columns else "label"

    df[cell_col] = df[cell_col].astype(str).str.strip()
    label = df[label_col].astype(str).str.strip().str.lower()

    asc = df.loc[label.isin({"1", "true", "asc", "ac", "abnormal", "anomaly", "disease"}), cell_col]
    asc = pd.Index(tgt_names.astype(str)).intersection(pd.Index(asc.astype(str)))

    if len(asc) == 0:
        raise ValueError("No ASC cells matched between pred_csv and target h5ad.")
    return asc


@torch.no_grad()
def run_phase2(ref, tgt, ckpt_path, device, batch_key, ref_bio_key, tgt_bio_key, batch_size, target_domain):
    ckpt = load(ckpt_path, device)
    domain_names = list(ckpt["domain_names"])
    target_idx = domain_names.index(target_domain)

    G = Generator_phase2(
        input_dim=int(ckpt["input_dim"]),
        num_batches=int(ckpt.get("num_domains", len(domain_names))),
        hidden_dim=int(ckpt.get("hidden_dim", 512)),
        cond_dim=int(ckpt.get("cond_dim", 128)),
        num_blocks=int(ckpt.get("g_num_blocks", 6)),
        dropout=float(ckpt.get("g_dropout", 0.1)),
        use_change_gate=bool(ckpt.get("use_change_gate", False)),
    ).to(device)
    G.load_state_dict(ckpt["generator_state_dict"])
    G.eval()

    ref = ref.copy()
    tgt = tgt.copy()

    ref.obs["assay_eval"] = ckpt["ref_domain_name"]
    tgt.obs["assay_eval"] = tgt.obs[batch_key].astype(str).values
    ref.obs["bio_label"] = ref.obs[ref_bio_key].astype(str).values
    tgt.obs["bio_label"] = tgt.obs[tgt_bio_key].astype(str).values
    ref.obs["dataset"] = "ref"
    tgt.obs["dataset"] = "tgt"

    joint = ad.concat([ref, tgt], axis=0, join="inner", merge="same", index_unique=None)

    xs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(dense(joint.X))), batch_size=batch_size)
    for (x,) in loader:
        x = x.to(device)
        c = torch.full((x.shape[0],), target_idx, dtype=torch.long, device=device)
        xs.append(G(x, c).cpu())

    joint.X = torch.cat(xs).numpy()
    return joint


@torch.no_grad()
def run_phase3(adata_asc, phase1_ckpt, phase3_ckpt, device, batch_size):
    G1 = load(phase1_ckpt, device).to(device)
    G1.eval()
    for p in G1.parameters():
        p.requires_grad_(False)

    ckpt = load(phase3_ckpt, device)
    S = Subtyper(
        input_dim=int(ckpt["n_genes"]),
        generator=G1,
        num_types=int(ckpt["num_types"]),
        **ckpt["s_configs"],
    ).to(device)
    S.load_state_dict(ckpt["subtyper_state_dict"])
    S.eval()

    preds, probs = [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(dense(adata_asc.X))), batch_size=batch_size)

    for (x,) in loader:
        x = x.to(device)
        x_hat, z, _ = G1(x, update_mem=False)
        _, q = S(z.detach(), (x - x_hat.detach()).detach())
        preds.append(q.argmax(1).cpu())
        probs.append(q.max(1).values.cpu())

    return torch.cat(preds).numpy(), torch.cat(probs).numpy()


def map_pred_names(pred, true):
    true = pd.Series(true.astype(str))
    pred = pd.Series(pred)
    mp = {}

    for k in sorted(pred.unique()):
        t = true[(pred == k) & true.str.lower().str.contains("tumor")]
        mp[k] = t.value_counts().idxmax() if len(t) else f"subtype_{k}"

    return np.array([mp[x] for x in pred])


def plot_alignment(adata, out):
    adata = adata.copy()
    sc.pp.pca(adata, n_comps=30, random_state=42)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata, random_state=42)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.umap(adata, color="bio_label", ax=ax[0], show=False, frameon=False, title="Corrected | cell_type", size=8)
    sc.pl.umap(adata, color="assay_eval", ax=ax[1], show=False, frameon=False, title="Corrected | batch", size=8)
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_subtype(tgt, out):
    tgt = tgt.copy()
    sc.pp.pca(tgt, n_comps=30, random_state=42)
    sc.pp.neighbors(tgt, n_neighbors=15, n_pcs=30)
    sc.tl.umap(tgt, random_state=42)

    emb = tgt.obsm["X_umap"]
    true_vals = tgt.obs["true_subtype"].astype(str)
    pred_vals = tgt.obs["pred_subtype"].astype(str)

    true_mask = true_vals.str.lower().str.contains("tumor").values
    pred_mask = pred_vals.ne("background").values

    labels = sorted(set(true_vals[true_mask]) | set(pred_vals[pred_mask]))
    cmap = plt.get_cmap("tab10")
    colors = {lab: cmap(i % 10) for i, lab in enumerate(labels)}

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    panels = [
        (ax[0], true_vals, true_mask, f"True ASC subtypes (n={true_mask.sum()})"),
        (ax[1], pred_vals, pred_mask, f"Pred ASC + subtype (n={pred_mask.sum()})"),
    ]

    for a, vals, mask, title in panels:
        a.scatter(emb[~mask, 0], emb[~mask, 1], s=4, c="lightgray", alpha=0.35, linewidths=0)
        for lab in labels:
            m = mask & vals.eq(lab).values
            if m.sum():
                a.scatter(emb[m, 0], emb[m, 1], s=7, c=[colors[lab]], label=lab, linewidths=0)
        a.set_title(title)
        a.set_xlabel("UMAP1")
        a.set_ylabel("UMAP2")
        a.legend(frameon=False, markerscale=2)

    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_path", default="data/ref_clean_colorectum.h5ad")
    p.add_argument("--tgt_path", default="data/tgt_clean_colorectum.h5ad")
    p.add_argument("--phase1_ckpt", default="ckpt/phase1_G.pth")
    p.add_argument("--phase2_ckpt", default="ckpt/phase2.pth")
    p.add_argument("--phase3_ckpt", default="ckpt/phase3.pth")
    p.add_argument("--pred_csv", default="output/phase1_pred_ASCs.csv")
    p.add_argument("--out_dir", default="result")
    p.add_argument("--target_domain", default="10x 3' v2")
    p.add_argument("--batch_key", default="assay")
    p.add_argument("--ref_bio_key", default="cell_type")
    p.add_argument("--tgt_bio_key", default="cell_state_label")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref = sc.read_h5ad(args.ref_path)
    tgt = sc.read_h5ad(args.tgt_path)
    ref.obs_names = ref.obs_names.astype(str)
    tgt.obs_names = tgt.obs_names.astype(str)
    ref, tgt = align_genes(ref, tgt)

    asc_cells = read_asc_cells(args.pred_csv, tgt.obs_names)

    aligned = run_phase2(
        ref, tgt, args.phase2_ckpt, device,
        args.batch_key, args.ref_bio_key, args.tgt_bio_key,
        args.batch_size, args.target_domain,
    )
    plot_alignment(aligned, out / "alignment_umap.png")

    aligned_tgt = aligned[aligned.obs["dataset"].astype(str) == "tgt"].copy()
    adata_asc = aligned_tgt[asc_cells].copy()

    pred, conf = run_phase3(adata_asc, args.phase1_ckpt, args.phase3_ckpt, device, args.batch_size)
    true = adata_asc.obs["bio_label"].astype(str).values
    pred_name = map_pred_names(pred, true)

    aligned_tgt.obs["true_subtype"] = "background"
    aligned_tgt.obs["pred_subtype"] = "background"

    tumor_mask = aligned_tgt.obs["bio_label"].astype(str).str.lower().str.contains("tumor")
    aligned_tgt.obs.loc[tumor_mask, "true_subtype"] = aligned_tgt.obs.loc[tumor_mask, "bio_label"].astype(str).values
    aligned_tgt.obs.loc[adata_asc.obs_names, "pred_subtype"] = pred_name

    pd.DataFrame({
        "cell_id": adata_asc.obs_names.astype(str),
        args.batch_key: adata_asc.obs[args.batch_key].astype(str).values,
        "true_subtype": true,
        "pred_subtype": pred_name,
        "subtype_confidence": conf,
    }).to_csv(out / "asc_subtype.csv", index=False)

    plot_subtype(aligned_tgt, out / "asc_subtype_umap.png")

    print("Saved:", out / "alignment_umap.png")
    print("Saved:", out / "asc_subtype_umap.png")
    print("Saved:", out / "asc_subtype.csv")


if __name__ == "__main__":
    main()