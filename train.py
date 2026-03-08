from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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
    epochs: int = 50,
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
    lambda_cls: float = 1.0,
    lambda_rec: float = 10.0,
    lambda_id: float = 1.0,
    lambda_change: float = 0.0,
    num_workers: int = 0,
    device: str | torch.device | None = None,
    seed: int = 42,
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
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    history = []
    for epoch in range(1, epochs + 1):
        running = {}
        n_steps = 0

        for x_real, c_src in loader:
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

    return {
        "generator": generator,
        "discriminator": discriminator,
        "history": history,
        "domain_names": domain_names,
        "batch_key": batch_key,
    }


__all__ = ["train_step", "train_stargan"]
