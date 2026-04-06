"""
train.py — GAN Training loop for Stage 1 lace structure generation.

Features:
  ✓ Mixed precision (fp16) via torch.cuda.amp
  ✓ Gradient clipping (prevents NaN explosion)
  ✓ NaN detection + auto-skip bad batches
  ✓ Gradient penalty every 16 steps (R1 / WGAN-GP)
  ✓ Cosine-annealing LR scheduler
  ✓ Checkpoint save + resume
  ✓ Sample image generation every N epochs
  ✓ TensorBoard logging

Usage:
    python stage1_gan/train.py --config config.yaml
    python stage1_gan/train.py --config config.yaml --resume         # Resume latest
"""

import sys
import argparse
import math
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm

# Allow running from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from stage1_gan.model   import Generator, Discriminator
from stage1_gan.dataset import get_dataloader
from stage1_gan.losses  import AdversarialLoss, gradient_penalty


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_ckpt(G, D, opt_G, opt_D, epoch, path):
    torch.save({
        'epoch': epoch,
        'generator_state_dict':     G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'optimizer_G':              opt_G.state_dict(),
        'optimizer_D':              opt_D.state_dict(),
    }, path)


def load_ckpt(G, D, opt_G, opt_D, path, device):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt['generator_state_dict'])
    D.load_state_dict(ckpt['discriminator_state_dict'])
    opt_G.load_state_dict(ckpt['optimizer_G'])
    opt_D.load_state_dict(ckpt['optimizer_D'])
    return ckpt['epoch']


def has_nan(tensor: torch.Tensor) -> bool:
    """Return True if tensor contains any NaN or Inf values."""
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


@torch.no_grad()
def save_samples(G, z_fixed, out_dir, epoch, device):
    G.eval()
    fake = G(z_fixed)                        # [N, 3, H, W] in [-1,1]
    fake = (fake.clamp(-1, 1) + 1) / 2      # → [0,1]
    grid = vutils.make_grid(fake, nrow=4, padding=2)
    vutils.save_image(grid, f"{out_dir}/epoch_{epoch:04d}.png")
    G.train()


# ══════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════

def train(config: dict, resume: bool = False):
    cfg  = config['stage1_gan']
    dcfg = config['data']
    seed = config['project']['seed']

    # ── Device ────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'━'*55}")
    print(f"  🎨  Lace GAN Training")
    print(f"  Device : {device}")
    if device.type == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {total_vram:.1f} GB")
    print(f"{'━'*55}\n")

    # ── Directories ───────────────────────────────────────────────
    ckpt_dir   = Path(cfg['checkpoint_dir'])
    sample_dir = Path('outputs/gan_samples')
    log_dir    = Path('logs/gan')
    for d in (ckpt_dir, sample_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────
    print("[1/5] Loading dataset …")
    loader = get_dataloader(
        data_dir   = dcfg['processed_dir'],
        image_size = dcfg['image_size'],
        batch_size = dcfg['batch_size'],
        num_workers= dcfg['num_workers'],
        training   = True,
    )
    print(f"  → {len(loader.dataset)} images | {len(loader)} batches/epoch\n")

    # ── Models ────────────────────────────────────────────────────
    print("[2/5] Building models …")
    G = Generator(
        z_dim      = cfg['latent_dim'],
        w_dim      = cfg['latent_dim'],
        ngf        = cfg['ngf'],
        image_size = dcfg['image_size'],
    ).to(device)

    D = Discriminator(ndf=cfg['ndf']).to(device)

    print(f"  Generator    : {sum(p.numel() for p in G.parameters())/1e6:.1f}M params")
    print(f"  Discriminator: {sum(p.numel() for p in D.parameters())/1e6:.1f}M params\n")

    # ── Optimizers ────────────────────────────────────────────────
    print("[3/5] Setting up optimizers …")

    # Lower LR to prevent NaN explosion (was 2e-4, now 5e-5)
    lr_g = cfg.get('lr_g', 5e-5)
    lr_d = cfg.get('lr_d', 5e-5)

    opt_G = torch.optim.Adam(G.parameters(),
                             lr=lr_g,
                             betas=(cfg['beta1'], cfg['beta2']),
                             eps=1e-8)
    opt_D = torch.optim.Adam(D.parameters(),
                             lr=lr_d,
                             betas=(cfg['beta1'], cfg['beta2']),
                             eps=1e-8)

    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=cfg['epochs'], eta_min=1e-7)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_D, T_max=cfg['epochs'], eta_min=1e-7)

    adv_loss = AdversarialLoss(loss_type=cfg.get('loss_type', 'hinge'))
    # GradScaler: fp16 needs dynamic loss scaling to avoid underflow/overflow
    scaler   = GradScaler(enabled=cfg['mixed_precision'])

    # Fixed latents for consistent visual tracking
    z_fixed = torch.randn(16, cfg['latent_dim'], device=device)

    # ── Resume ────────────────────────────────────────────────────
    start_epoch = 0
    latest_ckpt = ckpt_dir / 'latest.pth'
    if resume and latest_ckpt.exists():
        print(f"[4/5] Resuming from {latest_ckpt} …")
        start_epoch = load_ckpt(G, D, opt_G, opt_D, latest_ckpt, device)
        print(f"  → Epoch {start_epoch}\n")
    else:
        print("[4/5] Fresh training start\n")

    # ── Training Loop ─────────────────────────────────────────────
    print("[5/5] Training …\n")
    best_g_loss = float('inf')
    nan_count   = 0          # track consecutive NaN batches

    # TensorBoard (optional — skip gracefully if unavailable)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
        use_tb = True
    except ImportError:
        use_tb = False
        print("  [Info] TensorBoard not available — skipping TB logging.")

    for epoch in range(start_epoch, cfg['epochs']):
        G.train(); D.train()
        total_G = total_D = 0.0
        valid_steps = 0
        n_critic = cfg.get('n_critic', 1)

        pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}/{cfg['epochs']}",
                    dynamic_ncols=True, leave=False)

        for step, real in enumerate(pbar):
            real = real.to(device, non_blocking=True)
            B    = real.size(0)

            # ── Train Discriminator ────────────────────────────
            for _ in range(n_critic):
                opt_D.zero_grad(set_to_none=True)

                with autocast(enabled=cfg['mixed_precision']):
                    z    = torch.randn(B, cfg['latent_dim'], device=device)
                    fake = G(z).detach()

                    real_pred = D(real)
                    fake_pred = D(fake)
                    loss_D    = adv_loss.d_loss(real_pred, fake_pred)

                    # Gradient penalty every 16 steps to save compute
                    if step % 16 == 0:
                        gp     = gradient_penalty(D, real, fake, device)
                        loss_D = loss_D + cfg['lambda_gp'] * gp

                # ▶ NaN guard — skip bad batch
                if has_nan(loss_D):
                    nan_count += 1
                    if nan_count >= 10:
                        print(f"\n  ⚠ Too many NaN batches ({nan_count}). "
                              f"Reducing LR and continuing…")
                        for pg in opt_G.param_groups: pg['lr'] *= 0.5
                        for pg in opt_D.param_groups: pg['lr'] *= 0.5
                        nan_count = 0
                    pbar.set_postfix(G="nan-skip", D="nan-skip")
                    continue

                scaler.scale(loss_D).backward()
                # ▶ Gradient clipping — prevents explosion
                scaler.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                scaler.step(opt_D)

            # ── Train Generator ────────────────────────────────
            opt_G.zero_grad(set_to_none=True)

            with autocast(enabled=cfg['mixed_precision']):
                z    = torch.randn(B, cfg['latent_dim'], device=device)
                fake = G(z)
                fake_pred = D(fake)
                loss_G    = adv_loss.g_loss(fake_pred)

            if has_nan(loss_G):
                nan_count += 1
                pbar.set_postfix(G="nan-skip", D="nan-skip")
                continue

            scaler.scale(loss_G).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            scaler.step(opt_G)
            scaler.update()

            nan_count = 0   # reset on good batch
            total_G += loss_G.item()
            total_D += loss_D.item()
            valid_steps += 1

            pbar.set_postfix(G=f"{loss_G.item():.3f}",
                             D=f"{loss_D.item():.3f}")

        # ── End-of-epoch ──────────────────────────────────────
        sched_G.step()
        sched_D.step()

        if valid_steps == 0:
            print(f"  ⚠  Epoch {epoch+1}: ALL batches NaN — check model!")
            continue

        avg_G = total_G / valid_steps
        avg_D = total_D / valid_steps

        print(f"  Epoch {epoch+1:3d}  |  Loss_G: {avg_G:.4f}  |  "
              f"Loss_D: {avg_D:.4f}  |  "
              f"LR: {sched_G.get_last_lr()[0]:.2e}")

        if use_tb:
            writer.add_scalar('Loss/G', avg_G, epoch)
            writer.add_scalar('Loss/D', avg_D, epoch)

        # Samples
        if (epoch + 1) % cfg['sample_every'] == 0:
            save_samples(G, z_fixed, sample_dir, epoch + 1, device)
            print(f"  💾  Samples → {sample_dir}/epoch_{epoch+1:04d}.png")

        # Checkpoints
        save_ckpt(G, D, opt_G, opt_D, epoch + 1, latest_ckpt)
        if (epoch + 1) % cfg['save_every'] == 0:
            p = ckpt_dir / f"epoch_{epoch+1:04d}.pth"
            save_ckpt(G, D, opt_G, opt_D, epoch + 1, p)
            print(f"  💾  Checkpoint → {p}")

        # Best generator (by G loss)
        if avg_G < best_g_loss and not math.isnan(avg_G):
            best_g_loss = avg_G
            torch.save(G.state_dict(), ckpt_dir / 'best_generator.pth')

    print(f"\n✅  Training complete!")
    print(f"    Best Generator → {ckpt_dir}/best_generator.pth")
    if use_tb:
        writer.close()


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lace GAN")
    parser.add_argument('--config', default='config.yaml',
                        help="Path to config.yaml")
    parser.add_argument('--resume', action='store_true',
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume=args.resume)
