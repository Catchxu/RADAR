import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Any

from .utils import seed_everything, update_configs_with_args, set_requires_grad
from .configs import AnomalyConfigs
from .model import GeneratorWithMemory, Discriminator, GMMWithPrior
from .loss import rpgan_G_loss, rpgan_D_loss


class AnomalyModel:
    n_epochs: int
    batch_size: int
    learning_rate: float
    n_critic: int
    loss_weight: Dict[str, float]
    random_state: int
    n_genes: int
    g_configs: Dict[str, Any]
    d_configs: Dict[str, Any]
    gmm_configs: Dict[str, Any]

    def __init__(self, configs: AnomalyConfigs):
        for k, v in configs.__dict__.items():
            setattr(self, k, v)

        seed_everything(self.random_state)
        self._init_model()

        # will be set during training
        self.G_loss = torch.tensor(0.0)
        self.D_loss = torch.tensor(0.0)
        self.gene_names = None
        self.loader = None

    def _init_model(self):
        self.G = GeneratorWithMemory(**self.g_configs).to(self.device)
        self.D = Discriminator(**self.d_configs).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)
        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)

    def _batch_metrics(self, x_real: torch.Tensor) -> Dict[str, float]:
        """
        Compute diagnostic metrics on ONE batch, without updating memory.
        Returns Python floats.
        """
        with torch.no_grad():
            # No memory update for diagnostics
            x_fake, z, _ = self.G(x_real, update_mem=False)
            z_hat = self.G.encode(x_fake)

            x_l1 = (x_fake - x_real).abs().mean()
            z_l1 = (z - z_hat).abs().mean()

            d_real = self.D(x_real)
            d_fake = self.D(x_fake)

            # Flatten in case D outputs (B, 16) or (B,1) etc.
            dr = d_real.reshape(d_real.shape[0], -1)
            df = d_fake.reshape(d_fake.shape[0], -1)

            dr_mean = dr.mean()
            dr_std = dr.std(unbiased=False)

            df_mean = df.mean()
            df_std = df.std(unbiased=False)

            t = df - dr
            t_mean = t.mean()
            t_std = t.std(unbiased=False)

        return {
            "x_l1": float(x_l1.item()),
            "z_l1": float(z_l1.item()),
            "d_real_mean": float(dr_mean.item()),
            "d_real_std": float(dr_std.item()),
            "d_fake_mean": float(df_mean.item()),
            "d_fake_std": float(df_std.item()),
            "t_mean": float(t_mean.item()),
            "t_std": float(t_std.item()),
        }

    def train(self, ref: ad.AnnData):
        tqdm.write("Begin to train RADAR on the reference dataset...")

        self.gene_names = ref.var_names
        train_data = torch.tensor(ref.X, dtype=torch.float32)

        self.loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.G.train()
        self.D.train()

        with tqdm(total=self.n_epochs) as tbar:
            for epoch in range(self.n_epochs):
                tbar.set_description("Training Epochs")

                # epoch accumulators (averages over batches)
                n_batches = 0
                sum_G = 0.0
                sum_D = 0.0
                sums = {
                    "x_l1": 0.0,
                    "z_l1": 0.0,
                    "d_real_mean": 0.0,
                    "d_real_std": 0.0,
                    "d_fake_mean": 0.0,
                    "d_fake_std": 0.0,
                    "t_mean": 0.0,
                    "t_std": 0.0,
                }

                for data in self.loader:
                    data = data.to(self.device)

                    # Update D n_critic times
                    for _ in range(self.n_critic):
                        self.UpdateD(data)

                    # Update G once
                    self.UpdateG(data)

                    # accumulate losses
                    sum_G += float(self.G_loss.item())
                    sum_D += float(self.D_loss.item())

                    # diagnostics (no memory update)
                    m = self._batch_metrics(data)
                    for k in sums:
                        sums[k] += m[k]

                    n_batches += 1

                # epoch averages
                denom = max(n_batches, 1)
                avg_G = sum_G / denom
                avg_D = sum_D / denom
                avg = {k: v / denom for k, v in sums.items()}

                # print per epoch in tqdm postfix
                tbar.set_postfix(
                    G_Loss=avg_G,
                    D_Loss=avg_D,
                    xL1=avg["x_l1"],
                    zL1=avg["z_l1"],
                    dr_m=avg["d_real_mean"],
                    dr_s=avg["d_real_std"],
                    df_m=avg["d_fake_mean"],
                    df_s=avg["d_fake_std"],
                    t_m=avg["t_mean"],
                    t_s=avg["t_std"],
                )
                tbar.update(1)

                self.sch_G.step()
                self.sch_D.step()

        tqdm.write("Training process has been finished.")

    @torch.no_grad()
    def predict(self, tgt: ad.AnnData, run_gmm: bool = True):
        self.check(tgt)

        tqdm.write("Begin to detect anomalies on the target dataset...")
        real_data = torch.tensor(tgt.X, dtype=torch.float32)
        loader = DataLoader(
            real_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        ref_score = self.score(self.loader)
        
        tgt_score = self.score(loader)

        print(f"Ref mean score: {ref_score.mean():.6f}")
        print(f"Target mean score: {tgt_score.mean():.6f}")
        tqdm.write("Anomalous spots have been detected.")

        if run_gmm:
            gmm = GMMWithPrior(ref_score, **self.gmm_configs)
            threshold = gmm.fit(tgt_score=tgt_score)
            tgt_label = [1 if s >= threshold else 0 for s in tgt_score]
            return tgt_score, tgt_label
        else:
            return tgt_score

    def UpdateG(self, data: torch.Tensor):
        set_requires_grad(self.D, False)
        set_requires_grad(self.G, True)

        fake_data, z, _ = self.G(data, update_mem=True)

        loss_adv_G = rpgan_G_loss(
            D=self.D,
            x_real=data,
            x_fake=fake_data,
        )

        self.G_loss = loss_adv_G

        self.opt_G.zero_grad(set_to_none=True)
        self.G_loss.backward()
        self.opt_G.step()

        set_requires_grad(self.D, True)

    def UpdateD(self, data: torch.Tensor):
        set_requires_grad(self.G, False)
        set_requires_grad(self.D, True)

        with torch.no_grad():
            fake_data, _, _ = self.G(data, update_mem=False)

        loss_D = rpgan_D_loss(
            D=self.D,
            x_real=data,
            x_fake=fake_data,
            gamma=float(self.loss_weight["gamma"]),
        )

        self.D_loss = loss_D

        self.opt_D.zero_grad(set_to_none=True)
        self.D_loss.backward()
        self.opt_D.step()

        set_requires_grad(self.G, True)

    def check(self, tgt: ad.AnnData):
        if (tgt.var_names != self.gene_names).any():
            raise RuntimeError("Target and reference data have different genes.")
        if (self.G is None) or (self.D is None):
            raise RuntimeError("Please train the model first.")

    @torch.no_grad()
    def score(self, dataset):
        self.D.eval()
        self.G.eval()
        score = []

        for data in dataset:
            data = data.to(self.device)
            fake_data, z, z_mem = self.G(data, update_mem=False)
            fake_z = self.G.encode(fake_data)

            s = self.cosine_similarity(z, fake_z)
            score.append(s.cpu())

        score = torch.cat(score, dim=0).numpy()
        anomaly_score = 1-score 
        return anomaly_score.reshape(-1)


    def cosine_similarity(self, z: torch.Tensor, fake_z: torch.Tensor):
        dot_product = torch.sum(z * fake_z, dim=1)
        norm_z = torch.norm(z, dim=1)
        norm_fake_z = torch.norm(fake_z, dim=1)
        cosine_sim = dot_product / (norm_z * norm_fake_z + 1e-12)
        return cosine_sim.reshape(-1, 1)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="RADAR (Phase I) with RpGAN+R1+R2 for anomaly detection.")
    configs = AnomalyConfigs()

    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--ref_path", type=str, required=True, help="Path to read the reference h5ad file")
    data_group.add_argument("--tgt_path", type=str, required=True, help="Path to read the target h5ad file")
    data_group.add_argument("--result_path", type=str, default="result.csv", help="Path to save the output csv file")
    data_group.add_argument("--pth_path", type=str, default=None, help="Path to save the trained generator")

    a_group = parser.add_argument_group("AnomalyModel Parameters")
    a_group.add_argument("--n_epochs", type=int, default=configs.n_epochs, help="Number of epochs")
    a_group.add_argument("--batch_size", type=int, default=configs.batch_size, help="Batch size")
    a_group.add_argument("--learning_rate", type=float, default=configs.learning_rate, help="Learning rate")
    a_group.add_argument("--n_critic", type=int, default=configs.n_critic, help="D steps per G step")
    a_group.add_argument("--gamma", type=float, default=configs.gamma, help="R1+R2 penalty weight (paper gamma)")
    a_group.add_argument("--dropout", type=float, default=configs.dropout, help="Dropout rate for G and D")
    a_group.add_argument(
        "--normalization",
        type=int,
        choices=[0, 1],
        default=int(configs.normalization),
        help="Whether to use normalization in G and D: 0=False, 1=True",
    )
    a_group.add_argument(
        "--use_memory_bank",
        type=int,
        choices=[0, 1],
        default=int(getattr(configs, "use_memory_bank", True)),
        help="Whether to use memory bank in Generator: 0=False, 1=True",
    )
    a_group.add_argument(
        "--memory_size",
        type=int,
        default=configs.memory_size,
        help="Memory bank size in GeneratorWithMemory",
    )
    
    a_group.add_argument("--GPU", type=str, default=configs.GPU, help="GPU ID, e.g., cuda:0")
    a_group.add_argument("--random_state", type=int, default=configs.random_state, help="Random seed")
    a_group.add_argument("--n_genes", type=int, default=configs.n_genes, help="Number of genes")
    a_group.add_argument("--no_gmm", action="store_true", help="Disable GMM thresholding")

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["normalization"] = bool(args_dict["normalization"])
    args_dict["use_memory_bank"] = bool(args_dict["use_memory_bank"])
    update_configs_with_args(configs, args_dict, None)
    configs.build()
    configs.clear()

    for key, value in configs.__dict__.items():
        print(f"{key} = {value}")

    ref = sc.read_h5ad(args_dict["ref_path"])

    tgt = sc.read_h5ad(args_dict["tgt_path"])

    model = AnomalyModel(configs)
    model.train(ref)

    if not args_dict["no_gmm"]:
        score, label = model.predict(tgt, True)
        df = pd.DataFrame({"score": score, "label": label}, index=tgt.obs_names)
    else:
        score = model.predict(tgt, False)
        df = pd.DataFrame({"score": score}, index=tgt.obs_names)

    result_path = args_dict["result_path"]
    df.to_csv(result_path)
    print(f"Prediction result has been saved at {result_path}!")

    if args_dict["pth_path"] is not None:
        pth_path = args_dict["pth_path"]
        torch.save(model.G, pth_path)
        print(f"Generator has been saved at {pth_path}!")