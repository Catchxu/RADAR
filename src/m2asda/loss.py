from typing import Callable, Literal
import torch
import torch.nn.functional as F
from torch import Tensor, autograd

def h_logsigmoid(t: Tensor) -> Tensor:
    return -F.softplus(-t)  

def _grad_sq_norm(y: Tensor, x: Tensor) -> Tensor:
    grad = autograd.grad(
        outputs=y.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad.view(grad.size(0), -1).pow(2).sum(dim=1) #shape: (batch_size,)

def rpgan_G_loss(
    D: Callable[[Tensor], Tensor],
    x_real: Tensor,
    x_fake: Tensor,
    h=h_logsigmoid,
) -> Tensor:
    with torch.no_grad():
        d_real = D(x_real)
    d_fake = D(x_fake) #shape: (batch_size, 16)
    adv = h(d_fake - d_real).mean()
    return adv

def rpgan_D_loss(
    D: Callable[[Tensor], Tensor],
    x_real: Tensor,
    x_fake: Tensor,
    gamma: float,
    h=h_logsigmoid,
) -> Tensor:
    x_real = x_real.detach().requires_grad_(True)
    x_fake = x_fake.detach().requires_grad_(True)

    d_real = D(x_real)
    d_fake = D(x_fake)
    adv = h(d_fake - d_real).mean()
    r1 = _grad_sq_norm(d_real, x_real).mean()
    r2 = _grad_sq_norm(d_fake, x_fake).mean()

    return -adv + 0.5 * float(gamma) * (r1 + r2)

class KLLoss(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def kld(self, target, pred):
        log_frac = torch.log(target/(pred + self.epsilon))
        return torch.mean(torch.sum(target*log_frac, dim=1))
    
    def forward(self, p, q):
        return self.kld(p, q)