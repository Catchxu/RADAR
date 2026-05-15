from .utils import select_device


class AnomalyConfigs:
    def __init__(self):
        self.n_epochs = 300
        self.batch_size = 512
        self.learning_rate = 1e-4
        self.n_critic = 1
        self.gamma = 0.1
        self.lambda_z = 0.1
        self.use_memory_bank = True
        self.memory_size = 512

        self.GPU = "cuda:0"
        self.random_state = 2026
        self.n_genes = 3000

        # new sweepable params
        self.normalization = False
        self.dropout = 0.2

    def build(self):
        self.device = select_device(self.GPU)

        self.loss_weight = {"gamma": self.gamma, "lambda_z": self.lambda_z}

        self.g_configs = {
            "input_dim": self.n_genes,
            "hidden_dim": [1024, 512, 256],
            "latent_dim": 256,
            "memory_size": self.memory_size,
            "temperature": 0.1,
            "normalization": self.normalization,
            "activation": True,
            "dropout": self.dropout,
            "use_memory_bank": self.use_memory_bank,
        }

        self.d_configs = {
            "input_dim": self.n_genes,
            "hidden_dim": [1024, 512, 256],
            "latent_dim": 256,
            "normalization": self.normalization,
            "activation": True,
            "dropout": self.dropout,
        }

        self.gmm_configs = {
            "random_state": self.random_state,
            "max_iter": 100,
            "tol": 1e-5,
            "prior_beta": [1, 10],
        }

    def clear(self):
        if hasattr(self, "GPU"):
            delattr(self, "GPU")

class CorrectConfigs:
    def __init__(self):
        self.n_epochs = 50
        self.batch_size = 256
        self.learning_rate = 1e-4
        self.n_critic = 3
        self.gamma = 0.1
        self.GPU = "cuda:0"
        self.random_state = 2026
        self.n_genes = 3000

    def build(self):
        self.device = select_device(self.GPU)

        self.loss_weight = {"gamma": float(self.gamma)}

        self.g_configs = {
            "input_dim": self.n_genes,
            "hidden_dim": [1024, 512, 256],
            "latent_dim": 256,
            "normalization": True,
            "activation": True,
            "dropout": 0.1,
        }

        self.d_configs = {
            "input_dim": self.n_genes,
            "hidden_dim": [1024, 512, 256],
            "latent_dim": 256,
            "normalization": True,
            "activation": True,
            "dropout": 0.1,
        }

    def clear(self):
        delattr(self, "GPU")
        delattr(self, "gamma")
        

class SubtypeConfigs:
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.weight_decay = 0.0
        self.GPU = "cuda:0"
        self.random_state = 746
        self.n_genes = 3000
        self.n_epochs = 1000

    def build(self):
        self.device = select_device(self.GPU)

        self.s_configs = {
            "alpha": 1.0,
            "kmeans_n_init": 50,
            "num_layers": 2,
            "nheads": 4,
            "ff_hidden_dim": 256,
            "dropout": 0.2,
            "eps": 1e-8,
        }

    def clear(self):
        delattr(self, "GPU")
