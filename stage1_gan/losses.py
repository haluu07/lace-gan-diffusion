"""
losses.py — Loss functions for GAN training.

Includes:
  - AdversarialLoss  : Hinge / BCE / LSGAN variants
  - gradient_penalty : WGAN-GP regularization
  - PerceptualLoss   : VGG-based perceptual similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
#  ADVERSARIAL LOSS
# ══════════════════════════════════════════════════════════════════

class AdversarialLoss(nn.Module):
    """
    Multi-variant adversarial loss.

    loss_type options:
      'hinge'  — Hinge loss (stable, used in StyleGAN / SAGAN)
      'bce'    — Classic binary cross-entropy
      'lsgan'  — Least-squares GAN (smooth gradients)
    """

    def __init__(self, loss_type: str = 'hinge'):
        super().__init__()
        assert loss_type in ('hinge', 'bce', 'lsgan'), \
            f"Unknown loss_type: {loss_type}"
        self.loss_type = loss_type

    # ── Discriminator loss ─────────────────────────────────────────
    def d_loss(self, real_pred: torch.Tensor,
               fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Discriminator wants:
          real_pred → high (real)
          fake_pred → low  (fake)
        """
        if self.loss_type == 'hinge':
            return (F.relu(1.0 - real_pred) +
                    F.relu(1.0 + fake_pred)).mean()

        elif self.loss_type == 'bce':
            d_real = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred))
            d_fake = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred))
            return (d_real + d_fake) / 2

        else:  # lsgan
            return (((real_pred - 1) ** 2) + (fake_pred ** 2)).mean() / 2

    # ── Generator loss ─────────────────────────────────────────────
    def g_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Generator wants fake_pred → high (fool the discriminator)."""
        if self.loss_type == 'hinge':
            return -fake_pred.mean()

        elif self.loss_type == 'bce':
            return F.binary_cross_entropy_with_logits(
                fake_pred, torch.ones_like(fake_pred))

        else:  # lsgan
            return ((fake_pred - 1) ** 2).mean()


# ══════════════════════════════════════════════════════════════════
#  GRADIENT PENALTY (WGAN-GP)
# ══════════════════════════════════════════════════════════════════

def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Gradient penalty regularization for Wasserstein GAN-GP.

    Enforces the 1-Lipschitz constraint on D by penalizing
    ||∇D(x̂)||₂ away from 1 on interpolated samples.

    Args:
        discriminator : The Discriminator network.
        real          : Real images  [B, C, H, W].
        fake          : Fake images  [B, C, H, W].
        device        : torch.device.

    Returns:
        Scalar penalty term (add to D loss with weight lambda_gp).
    """
    B = real.size(0)

    # Random interpolation coefficients
    alpha = torch.rand(B, 1, 1, 1, device=device)

    # Interpolated samples: x̂ = α·real + (1-α)·fake
    interpolated = (alpha * real + (1 - alpha) * fake.detach())
    interpolated.requires_grad_(True)

    # Forward through D
    d_interp = discriminator(interpolated)

    # Compute ∇D(x̂) w.r.t. x̂
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten and compute L2 norm per sample
    grads = grads.view(B, -1)
    grad_norm = grads.norm(2, dim=1)

    # Penalty: (||∇D||₂ - 1)²
    return ((grad_norm - 1) ** 2).mean()


# ══════════════════════════════════════════════════════════════════
#  PERCEPTUAL LOSS  (optional — boosts texture quality)
# ══════════════════════════════════════════════════════════════════

class PerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss.

    Computes L1 distance between VGG feature maps of real and fake.
    Helps the generator reproduce fine lace textures that pixel-wise
    losses miss.

    Usage (in training loop):
        perc_loss = PerceptualLoss(device).to(device)
        loss_G += lambda_perc * perc_loss(fake_imgs, real_imgs)
    """

    def __init__(self, device: torch.device):
        super().__init__()
        import torchvision.models as models

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        feats = list(vgg.features)

        # Use three feature stages
        self.s1 = nn.Sequential(*feats[:4]).to(device)    # relu1_2
        self.s2 = nn.Sequential(*feats[4:9]).to(device)   # relu2_2
        self.s3 = nn.Sequential(*feats[9:16]).to(device)  # relu3_3

        # Freeze VGG
        for p in self.parameters():
            p.requires_grad = False

        self.criterion = nn.L1Loss()

        # ImageNet normalization constants
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406],
                                 device=device).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225],
                                device=device).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert GAN output [-1,1] → ImageNet-normalized."""
        x = (x + 1) / 2            # [-1,1] → [0,1]
        return (x - self.mean) / self.std

    def forward(self, fake: torch.Tensor,
                real: torch.Tensor) -> torch.Tensor:
        fake = self._normalize(fake)
        real = self._normalize(real)

        loss = 0.0
        f1, r1 = self.s1(fake), self.s1(real)
        loss += self.criterion(f1, r1)

        f2, r2 = self.s2(f1), self.s2(r1)
        loss += self.criterion(f2, r2)

        f3, r3 = self.s3(f2), self.s3(r2)
        loss += self.criterion(f3, r3)

        return loss / 3
