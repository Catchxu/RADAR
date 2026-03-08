import torch
import torch.nn as nn
import torch.nn.functional as F


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """
    real_logits: (B, 1)
    fake_logits: (B, 1)
    """
    loss_real = torch.relu(1.0 - real_logits).mean()
    loss_fake = torch.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    fake_logits: (B, 1)
    """
    return -fake_logits.mean()


class StarGANLoss(nn.Module):
    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_rec: float = 10.0,
        lambda_id: float = 1.0,
        lambda_change: float = 0.0,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_rec = lambda_rec
        self.lambda_id = lambda_id
        self.lambda_change = lambda_change

    def discriminator_loss(
        self,
        generator,
        discriminator,
        x_real: torch.Tensor,
        c_src: torch.Tensor,
        c_tgt: torch.Tensor,
    ):
        """
        x_real: (B, L)
        c_src:  (B,)
        c_tgt:  (B,)
        """
        with torch.no_grad():
            x_fake = generator(x_real, c_tgt)

        real_adv, real_cls = discriminator(x_real)
        fake_adv, _ = discriminator(x_fake)

        # Adversarial loss
        loss_adv = discriminator_hinge_loss(real_adv, fake_adv)

        # Domain classification on real samples
        loss_cls_real = F.cross_entropy(real_cls, c_src)

        total_loss = loss_adv + self.lambda_cls * loss_cls_real

        loss_dict = {
            "d_loss": total_loss,
            "d_adv": loss_adv.detach(),
            "d_cls_real": loss_cls_real.detach(),
        }
        return total_loss, loss_dict

    def generator_loss(
        self,
        generator,
        discriminator,
        x_real: torch.Tensor,
        c_src: torch.Tensor,
        c_tgt: torch.Tensor,
    ):
        """
        x_real: (B, L)
        c_src:  (B,)
        c_tgt:  (B,)
        """
        # Translate to target domain
        x_fake = generator(x_real, c_tgt)

        # Translate back to source domain
        x_recon = generator(x_fake, c_src)

        # Identity mapping
        x_id = generator(x_real, c_src)

        fake_adv, fake_cls = discriminator(x_fake)

        # Adversarial loss
        loss_adv = generator_hinge_loss(fake_adv)

        # Domain classification on fake samples
        loss_cls_fake = F.cross_entropy(fake_cls, c_tgt)

        # Reconstruction / cycle loss
        loss_rec = F.l1_loss(x_recon, x_real)

        # Identity loss
        loss_id = F.l1_loss(x_id, x_real)

        # Optional change regularization
        loss_change = F.l1_loss(x_fake, x_real)

        total_loss = (
            loss_adv
            + self.lambda_cls * loss_cls_fake
            + self.lambda_rec * loss_rec
            + self.lambda_id * loss_id
            + self.lambda_change * loss_change
        )

        loss_dict = {
            "g_loss": total_loss,
            "g_adv": loss_adv.detach(),
            "g_cls_fake": loss_cls_fake.detach(),
            "g_rec": loss_rec.detach(),
            "g_id": loss_id.detach(),
            "g_change": loss_change.detach(),
        }
        return total_loss, loss_dict