import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Any

from .utils import seed_everything, update_configs_with_args, set_requires_grad, to_dense
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

    def _init_model(self):
        self.G = GeneratorWithMemory(**self.g_configs).to(self.device)
        self.D = Discriminator(**self.d_configs).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        self.sch_G = CosineAnnealingLR(self.opt_G, self.n_epochs)
        self.sch_D = CosineAnnealingLR(self.opt_D, self.n_epochs)

    def train(self, ref: ad.AnnData):
        tqdm.write("Begin to train M2ASDA on the reference dataset...")

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

        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description("Training Epochs")

                for data in self.loader:
                    data = data.to(self.device)

                    for _ in range(self.n_critic):
                        self.UpdateD(data)

                    self.UpdateG(data)

                t.set_postfix(G_Loss=float(self.G_loss.item()), D_Loss=float(self.D_loss.item()))
                t.update(1)

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

            # No memory update in scoring
            fake_data, z, _ = self.G(data, update_mem=False)
            fake_z = self.G.encode(fake_data)

            s = self.cosine_similarity(z, fake_z)
            score.append(s.cpu())

        score = torch.cat(score, dim=0).numpy()
        return self.normalize(score)

    def cosine_similarity(self, z: torch.Tensor, fake_z: torch.Tensor):
        dot_product = torch.sum(z * fake_z, dim=1)
        norm_z = torch.norm(z, dim=1)
        norm_fake_z = torch.norm(fake_z, dim=1)
        cosine_sim = dot_product / (norm_z * norm_fake_z + 1e-12)
        return cosine_sim.reshape(-1, 1)

    def normalize(self, score: np.ndarray):
        score = (score.max() - score) / (score.max() - score.min() + 1e-12)
        return score.reshape(-1)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="M2ASDA (Phase I) with RpGAN+R1+R2 for anomaly detection.")
    configs = AnomalyConfigs()

    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--ref_path", type=str, help="Path to read the reference h5ad file")
    data_group.add_argument("--tgt_path", type=str, help="Path to read the target h5ad file")
    data_group.add_argument("--result_path", type=str, default="result.csv", help="Path to save the output csv file")
    data_group.add_argument("--pth_path", type=str, default=None, help="Path to save the trained generator")

    a_group = parser.add_argument_group("AnomalyModel Parameters")
    a_group.add_argument("--n_epochs", type=int, default=configs.n_epochs, help="Number of epochs")
    a_group.add_argument("--batch_size", type=int, default=configs.batch_size, help="Batch size")
    a_group.add_argument("--learning_rate", type=float, default=configs.learning_rate, help="Learning rate")
    a_group.add_argument("--n_critic", type=int, default=configs.n_critic, help="D steps per G step")
    a_group.add_argument("--gamma", type=float, default=configs.gamma, help="R1+R2 penalty weight (paper gamma)")

    a_group.add_argument("--GPU", type=str, default=configs.GPU, help="GPU ID, e.g., cuda:0")
    a_group.add_argument("--random_state", type=int, default=configs.random_state, help="Random seed")
    a_group.add_argument("--n_genes", type=int, default=configs.n_genes, help="Number of genes")
    a_group.add_argument("--no_gmm", action="store_true", help="Disable GMM thresholding")

    args = parser.parse_args()
    args_dict = vars(args)

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
