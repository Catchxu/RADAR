from __future__ import annotations
from typing import Callable, Literal
import torch
import torch.nn as nn
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
    tau=1.0,
) -> Tensor:
    with torch.no_grad():
        d_real = D(x_real)
    d_fake = D(x_fake) 
    t = (d_fake - d_real) / tau
    adv = h(t).mean()
    return adv

def rpgan_D_loss(
    D: Callable[[Tensor], Tensor],
    x_real: Tensor,
    x_fake: Tensor,
    gamma: float,
    h=h_logsigmoid,
    tau=1.0,
) -> Tensor:
    x_real = x_real.detach().requires_grad_(True)
    x_fake = x_fake.detach().requires_grad_(True)

    d_real = D(x_real)
    d_fake = D(x_fake)
    t = (d_fake - d_real) / tau
    adv = h(t).mean()
    r1 = _grad_sq_norm(d_real, x_real).mean()
    r2 = _grad_sq_norm(d_fake, x_fake).mean()

    return -adv + 0.5 * float(gamma) * (r1 + r2)


def gan_G_loss(
    D: Callable[[Tensor], Tensor],
    x_real: Tensor,
    x_fake: Tensor,
    h=h_logsigmoid,
    tau: float = 1.0,
) -> Tensor:

    d_fake = D(x_fake) / tau
    return h(d_fake).mean()


def gan_D_loss(
    D: Callable[[Tensor], Tensor],
    x_real: Tensor,
    x_fake: Tensor,
    gamma: float = 0.0,
    h=h_logsigmoid,
    tau: float = 1.0,
) -> Tensor:

    x_real = x_real.detach()
    x_fake = x_fake.detach()

    d_real = D(x_real) / tau
    d_fake = D(x_fake) / tau

    adv_real = h(-d_real).mean()
    adv_fake = h(d_fake).mean()

    return -(adv_real + adv_fake)


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    logits: (B, C)
    targets: (B,)
    sample_weight: (B,) or None
    """
    per_sample = F.cross_entropy(logits, targets, reduction="none")

    if sample_weight is None:
        return per_sample.mean()

    sample_weight = sample_weight.float().view(-1)
    denom = sample_weight.sum().clamp_min(1e-8)
    return (per_sample * sample_weight).sum() / denom




class phase2RpLoss(nn.Module):
    def __init__(
        self,
        lambda_batch: float = 1.0,
        lambda_state: float = 1.0,
        lambda_rec: float = 10.0,
        lambda_id: float = 1.0,
        gamma: float = 0.05,
        tau: float = 1.0,
        h=h_logsigmoid,
    ):
        super().__init__()
        self.lambda_batch = lambda_batch
        self.lambda_state = lambda_state
        self.lambda_rec = lambda_rec
        self.lambda_id = lambda_id
        self.gamma = gamma
        self.tau = tau
        self.h = h

    def discriminator_loss(
        self,
        generator,
        discriminator,
        x_real: torch.Tensor,
        c_src: torch.Tensor,
        c_tgt: torch.Tensor,
        s_src: torch.Tensor,
        state_weight: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            x_fake = generator(x_real, c_tgt)

        D_adv = lambda x: discriminator(x)[0].view(-1)


        loss_adv = rpgan_D_loss(
            D=D_adv,
            x_real=x_real,
            x_fake=x_fake,
            gamma=self.gamma,
            h=self.h,
            tau=self.tau,
        )


        _, real_batch_logits, real_state_logits = discriminator(x_real)

        loss_batch_real = F.cross_entropy(real_batch_logits, c_src)
        loss_state_real = weighted_cross_entropy(
            real_state_logits,
            s_src,
            sample_weight=state_weight,
        )

        total_loss = (
            loss_adv
            + self.lambda_batch * loss_batch_real
            + self.lambda_state * loss_state_real
        )

        loss_dict = {
            "d_loss": total_loss,
            "d_adv": loss_adv.detach(),   
            "d_batch_real": loss_batch_real.detach(),
            "d_state_real": loss_state_real.detach(),
        }
        return total_loss, loss_dict

    def generator_loss(
        self,
        generator,
        discriminator,
        x_real: torch.Tensor,
        c_src: torch.Tensor,
        c_tgt: torch.Tensor,
        s_src: torch.Tensor,
        state_weight: torch.Tensor | None = None,
    ):
        x_fake = generator(x_real, c_tgt)
        x_recon = generator(x_fake, c_src)
        x_id = generator(x_real, c_src)

        D_adv = lambda x: discriminator(x)[0].view(-1)

        # RpGAN generator loss
        loss_adv = rpgan_G_loss(
            D=D_adv,
            x_real=x_real,
            x_fake=x_fake,
            h=self.h,
            tau=self.tau,
        )

        _, fake_batch_logits, fake_state_logits = discriminator(x_fake)

        loss_batch_fake = F.cross_entropy(fake_batch_logits, c_tgt)
        loss_state_fake = weighted_cross_entropy(
            fake_state_logits,
            s_src,
            sample_weight=state_weight,
        )

        # cycle reconstruction
        loss_rec = F.l1_loss(x_recon, x_real)

        # identity
        loss_id = F.l1_loss(x_id, x_real)

        total_loss = (
            loss_adv
            + self.lambda_batch * loss_batch_fake
            + self.lambda_state * loss_state_fake
            + self.lambda_rec * loss_rec
            + self.lambda_id * loss_id
        )

        loss_dict = {
            "g_loss": total_loss,
            "g_adv": loss_adv.detach(),
            "g_batch_fake": loss_batch_fake.detach(),
            "g_state_fake": loss_state_fake.detach(),
            "g_rec": loss_rec.detach(),
            "g_id": loss_id.detach(),
        }
        return total_loss, loss_dict

class KLLoss(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def kld(self, target, pred):
        log_frac = torch.log(target/(pred + self.epsilon))
        return torch.mean(torch.sum(target*log_frac, dim=1))
    
    def forward(self, p, q):
        return self.kld(p, q)


