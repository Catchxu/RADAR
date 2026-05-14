import warnings
import torch
import anndata as ad
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Union, Dict, Any
from torch.utils.data import Dataset


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def clear_warnings(category=FutureWarning):
    def outwrapper(func):
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=category)
                return func(*args, **kwargs)

        return wrapper

    return outwrapper


def select_device(GPU: Union[bool, str] = True):
    if GPU:
        if torch.cuda.is_available():
            if isinstance(GPU, str):
                device = torch.device(GPU)
            else:
                device = torch.device('cuda:0')
        else:
            print("GPU isn't available, and use CPU to train Docs.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device

def update_configs_with_args(configs, args_dict, suffix=None):
    """
    If suffix is None: directly map args -> configs (same key).
    If suffix is not None: only update keys ending with suffix, and strip suffix.
    """
    for key, value in args_dict.items():
        if value is None:
            continue

        if suffix is None:
            config_key = key
        else:
            if not key.endswith(suffix):
                continue
            config_key = key[:-len(suffix)]

        if hasattr(configs, config_key):
            setattr(configs, config_key, value)


@clear_warnings()
def evaluate(y_true, y_score):
    """
    Calculate evaluation metrics
    """
    y_true = pd.Series(y_true)
    y_score = pd.Series(y_score)

    roc_auc = metrics.roc_auc_score(y_true, y_score)
    ap = metrics.average_precision_score(y_true, y_score)

    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thres = np.percentile(y_score, ratio)
    y_pred = (y_score >= thres).astype(int)
    y_true = y_true.astype(int)
    _, _, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return roc_auc, ap, f1

    
class PairDatasetWithBatch(Dataset):
    """
    For Module III training: returns (x_r, x_t, b_onehot)
      x_r: [G]    paired ref cell expression
      x_t: [G]    target cell expression
      b_onehot: [Nb] (paper B_t)
    """
    def __init__(self, ref_pair: ad.AnnData, tgt: ad.AnnData, batch_key: str, num_batches: int):
        if ref_pair.n_obs != tgt.n_obs:
            raise RuntimeError("ref_pair and tgt must have the same n_obs for 1-1 pairing.")

        if batch_key not in tgt.obs:
            raise RuntimeError(
                f"tgt.obs missing '{batch_key}'. Build tgt with ad.concat(..., label='{batch_key}', keys=...)."
            )

        self.x_r = torch.tensor(to_dense(ref_pair.X), dtype=torch.float32)
        self.x_t = torch.tensor(to_dense(tgt.X), dtype=torch.float32)

        codes = tgt.obs[batch_key]
        if hasattr(codes, "cat"):
            codes = codes.cat.codes
        self.batch_id = torch.tensor(np.asarray(codes), dtype=torch.long)

        self.Nb = int(num_batches)

    def __len__(self):
        return self.x_t.size(0)

    def __getitem__(self, i):
        b = self.batch_id[i].item()
        onehot = torch.zeros(self.Nb, dtype=torch.float32)
        if self.Nb == 1:
            onehot[0] = 1.0
        else:
            onehot[b] = 1.0
        return self.x_r[i], self.x_t[i], onehot

    
def set_requires_grad(module: torch.nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

class PairDataset(Dataset):
    def __init__(self, DataA, DataB):
        self.DataA = DataA
        self.DataB = DataB

        if len(self.DataA) != len(self.DataB):
            raise RuntimeError('Input data can not be paired')

    def __len__(self):
        return len(self.DataA)

    def __getitem__(self, index):
        A_sample = self.DataA[index]
        B_sample = self.DataB[index]

        return A_sample, B_sample