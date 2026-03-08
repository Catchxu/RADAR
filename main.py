from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from discriminator import Discriminator
from generator import Generator
from loss import StarGANLoss


def train_step(
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    criterion,
    x_real,
    c_src,
    c_tgt,
):
    """
    x_real: (B, L)
    c_src:  (B,)
    c_tgt:  (B,)
    """
    # --------------------
    # Train Discriminator
    # --------------------
    discriminator.train()
    generator.train()

    d_optimizer.zero_grad()
    d_loss, d_log = criterion.discriminator_loss(
        generator=generator,
        discriminator=discriminator,
        x_real=x_real,
        c_src=c_src,
        c_tgt=c_tgt,
    )
    d_loss.backward()
    d_optimizer.step()

    # ----------------
    # Train Generator
    # ----------------
    g_optimizer.zero_grad()
    g_loss, g_log = criterion.generator_loss(
        generator=generator,
        discriminator=discriminator,
        x_real=x_real,
        c_src=c_src,
        c_tgt=c_tgt,
    )
    g_loss.backward()
    g_optimizer.step()

    logs = {}
    logs.update({k: float(v) for k, v in d_log.items()})
    logs.update({k: float(v) for k, v in g_log.items()})
    return logs


def _load_adata(adata_or_path: str | Path | ad.AnnData) -> ad.AnnData:
    if isinstance(adata_or_path, ad.AnnData):
        return adata_or_path
    return ad.read_h5ad(str(adata_or_path))


def train_stargan(
    adata_or_path: str | Path | ad.AnnData,
    batch_key: str,
    epochs: int = 100,
    batch_size: int = 128,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    hidden_dim: int = 512,
    cond_dim: int = 128,
    g_num_blocks: int = 6,
    d_num_blocks: int = 3,
    g_dropout: float = 0.1,
    d_dropout: float = 0.0,
    use_change_gate: bool = False,
    lambda_cls: float = 3.0,
    lambda_rec: float = 3.0,
    lambda_id: float = 0.2,
    lambda_change: float = 0.0,
    num_workers: int = 0,
    device: str | torch.device | None = None,
    seed: int = 42,
    show_progress: bool = True,
):
    """
    Train a StarGAN-like model on preprocessed scRNA-seq data.

    Parameters
    ----------
    adata_or_path
        Either a preprocessed AnnData object or a path to an `.h5ad` file.
    batch_key
        Column key in `adata.obs` that defines source/target domains.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    adata = _load_adata(adata_or_path)
    if batch_key not in adata.obs:
        raise KeyError(f"`{batch_key}` not found in `adata.obs`.")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Input AnnData has no cells or no genes.")

    domains = adata.obs[batch_key].astype("category")
    y = domains.cat.codes.to_numpy(np.int64)
    domain_names = list(domains.cat.categories)
    num_domains = len(domain_names)
    if num_domains < 2:
        raise ValueError("Need at least 2 domains in `batch_key` to train StarGAN.")

    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    input_dim = adata.n_vars
    generator = Generator(
        input_dim=input_dim,
        num_batches=num_domains,
        hidden_dim=hidden_dim,
        cond_dim=cond_dim,
        num_blocks=g_num_blocks,
        dropout=g_dropout,
        use_change_gate=use_change_gate,
    ).to(device)
    discriminator = Discriminator(
        input_dim=input_dim,
        num_batches=num_domains,
        hidden_dim=hidden_dim,
        num_blocks=d_num_blocks,
        dropout=d_dropout,
    ).to(device)

    criterion = StarGANLoss(
        lambda_cls=lambda_cls,
        lambda_rec=lambda_rec,
        lambda_id=lambda_id,
        lambda_change=lambda_change,
    )
    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=lr_g)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr_d)

    history = []
    epoch_iterator = tqdm(
        range(1, epochs + 1),
        desc="Training",
        disable=not show_progress,
    )
    for epoch in epoch_iterator:
        running = {}
        n_steps = 0

        batch_iterator = tqdm(
            loader,
            desc=f"Epoch {epoch}/{epochs}",
            leave=False,
            disable=not show_progress,
        )
        for x_real, c_src in batch_iterator:
            x_real = x_real.to(device, non_blocking=True)
            c_src = c_src.to(device, non_blocking=True)

            c_tgt = torch.randint(0, num_domains, c_src.shape, device=device)
            same = c_tgt == c_src
            if same.any():
                c_tgt[same] = (c_tgt[same] + 1) % num_domains

            logs = train_step(
                generator=generator,
                discriminator=discriminator,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                criterion=criterion,
                x_real=x_real,
                c_src=c_src,
                c_tgt=c_tgt,
            )

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + float(v)
            n_steps += 1

        epoch_log = {"epoch": epoch}
        for k, v in running.items():
            epoch_log[k] = v / max(n_steps, 1)
        history.append(epoch_log)
        epoch_iterator.set_postfix(
            g_loss=f"{epoch_log.get('g_loss', 0.0):.4f}",
            d_loss=f"{epoch_log.get('d_loss', 0.0):.4f}",
        )

    generator.domain_names_ = domain_names
    return generator


@torch.no_grad()
def align_adata(
    adata_or_path: str | Path | ad.AnnData,
    generator: Generator,
    batch_key: str,
    target_batch: str,
    domain_names: list[str] | None = None,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
    copy: bool = True,
    overwrite_batch_key: bool = False,
    output_layer: str | None = "X_aligned",
) -> ad.AnnData:
    """
    Align all cells to one target domain using a trained StarGAN generator.

    Parameters
    ----------
    adata_or_path
        Preprocessed AnnData or `.h5ad` path.
    generator
        Trained generator returned by `train_stargan`.
    batch_key
        Domain column in `adata.obs`.
    target_batch
        Target domain name to align all cells into.
    domain_names
        Domain names used during training. If None, use categories from current `adata.obs[batch_key]`.
    overwrite_batch_key
        If True, overwrite `adata.obs[batch_key]` with `target_batch`.
    output_layer
        If not None, write aligned matrix to `adata.layers[output_layer]`.
    """
    adata = _load_adata(adata_or_path)
    if copy:
        adata = adata.copy()

    if batch_key not in adata.obs:
        raise KeyError(f"`{batch_key}` not found in `adata.obs`.")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Input AnnData has no cells or no genes.")

    obs_domains = adata.obs[batch_key].astype("category")
    inferred_names = [str(x) for x in obs_domains.cat.categories]
    if domain_names is None and hasattr(generator, "domain_names_"):
        domain_names = [str(x) for x in generator.domain_names_]
    else:
        domain_names = inferred_names if domain_names is None else [str(x) for x in domain_names]
    domain_to_idx = {name: idx for idx, name in enumerate(domain_names)}

    if target_batch not in domain_to_idx:
        raise ValueError(
            f"`target_batch={target_batch}` is not in domain_names: {domain_names}"
        )

    if hasattr(generator, "input_dim") and adata.n_vars != generator.input_dim:
        raise ValueError(
            f"Feature mismatch: adata.n_vars={adata.n_vars}, generator.input_dim={generator.input_dim}."
        )

    mapped = obs_domains.astype(str).map(domain_to_idx)
    if mapped.isna().any():
        missing = sorted(set(obs_domains.astype(str)[mapped.isna()].tolist()))
        raise ValueError(
            "Found domains in `adata.obs[batch_key]` not present in `domain_names`: "
            f"{missing}"
        )

    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    x = np.asarray(x, dtype=np.float32)

    if device is None:
        device = next(generator.parameters()).device
    else:
        device = torch.device(device)
        generator = generator.to(device)
    generator.eval()

    x_tensor = torch.from_numpy(x)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    target_idx = domain_to_idx[target_batch]
    aligned_chunks = []
    for (x_batch,) in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        c_tgt = torch.full((x_batch.shape[0],), target_idx, dtype=torch.long, device=device)
        x_aligned = generator(x_batch, c_tgt)
        aligned_chunks.append(x_aligned.detach().cpu())

    x_aligned = torch.cat(aligned_chunks, dim=0).numpy()
    adata.X = x_aligned

    if output_layer is not None:
        adata.layers[output_layer] = x_aligned.copy()

    adata.obs[f"{batch_key}_original"] = adata.obs[batch_key].astype(str).values
    if overwrite_batch_key:
        adata.obs[batch_key] = target_batch
    adata.obs[f"{batch_key}_aligned_target"] = target_batch

    return adata


__all__ = ["train_step", "train_stargan", "align_adata"]
