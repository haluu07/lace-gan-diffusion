"""
train_lora.py — Fine-tune Stable Diffusion with LoRA on lace dataset.

LoRA (Low-Rank Adaptation) adds small trainable adapter layers to SD,
teaching it to generate lace-specific textures without modifying the
base model weights.

Benefits over full fine-tuning:
  ✓ 10x less VRAM (12 GB vs 24+ GB)
  ✓ 10x faster (~2-4h vs days)
  ✓ Small output file (~50 MB vs 4 GB)
  ✓ LoRA weights snap onto any SD checkpoint

Requirements:
    pip install diffusers accelerate peft transformers

Usage:
    # Single GPU (GPU 0)
    CUDA_VISIBLE_DEVICES=0 python train_lora.py --config config.yaml

    # Multi-GPU (both A4000s)
    accelerate launch --multi_gpu train_lora.py --config config.yaml
"""

import sys
import os
import argparse
import yaml
import math
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════

class LaceLoRADataset(Dataset):
    """
    Dataset for LoRA training.
    Each image is paired with a text caption describing lace fabric.
    """

    # Caption templates — varied to improve generalization
    CAPTION_TEMPLATES = [
        "a photo of lace fabric with intricate floral patterns",
        "close-up of white lace textile with delicate mesh holes",
        "detailed lace fabric texture with fine thread work",
        "macro photo of lace with intricate woven pattern",
        "high quality lace fabric texture, detailed threads",
        "vintage lace textile with decorative floral motifs",
    ]

    def __init__(self, data_dir: str, image_size: int, tokenizer):
        self.paths = []
        for ext in ('.jpg', '.jpeg', '.png'):
            self.paths += list(Path(data_dir).rglob(f"*{ext}"))
        self.paths = sorted(self.paths)

        if not self.paths:
            raise ValueError(f"No images found in {data_dir}")

        print(f"[LoRA Dataset] {len(self.paths)} images")

        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(image_size,
                              interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Image
        img = Image.open(self.paths[idx]).convert('RGB')
        pixel_values = self.transform(img)

        # Caption — rotate through templates
        caption = self.CAPTION_TEMPLATES[idx % len(self.CAPTION_TEMPLATES)]

        # Tokenize
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
        }


# ══════════════════════════════════════════════════════════════════
#  LoRA INJECTION
# ══════════════════════════════════════════════════════════════════

def inject_lora(unet: UNet2DConditionModel, rank: int = 16, alpha: float = 32):
    """
    Inject LoRA adapters into UNet attention layers using PEFT.

    Each attention projection (Q, K, V, out) gets a low-rank adapter:
      W_new = W_original + (B @ A) * (alpha / rank)
    where A ∈ R^(rank × in), B ∈ R^(out × rank) are the trainable params.

    Args:
        unet : UNet2DConditionModel from diffusers.
        rank : LoRA rank (4 = small, 64 = large; 16 is standard).
        alpha: Scaling factor (usually 2× rank).

    Returns:
        UNet with LoRA adapters (only adapter params are trainable).
    """
    try:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r              = rank,
            lora_alpha     = alpha,
            init_lora_weights = "gaussian",
            target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",    # Attention
                "proj_in", "proj_out",                   # Transformer blocks
            ],
            lora_dropout   = 0.05,
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
        return unet

    except ImportError:
        raise ImportError(
            "PEFT not installed. Run: pip install peft"
        )


# ══════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════

def train_lora(config: dict):
    cfg      = config['lora']
    model_id = cfg['base_model']

    # ── Device ────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float16 if cfg['mixed_precision'] == 'fp16' else torch.float32

    print(f"\n{'━'*55}")
    print(f"  🎨  LoRA Fine-tuning on Lace Dataset")
    print(f"  Model  : {model_id}")
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
    print(f"  Rank   : {cfg['lora_rank']} | Alpha: {cfg['lora_alpha']}")
    print(f"{'━'*55}\n")

    # ── Output dir ────────────────────────────────────────────────
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────
    print("[1/5] Loading base models …")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_enc  = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae       = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet      = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_sch = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze base models (only LoRA adapters will be trained)
    vae.requires_grad_(False)
    text_enc.requires_grad_(False)
    unet.requires_grad_(False)

    # Cast to fp16 for memory
    vae       = vae.to(device, dtype=dtype)
    text_enc  = text_enc.to(device, dtype=dtype)
    unet      = unet.to(device)

    # ── Inject LoRA ───────────────────────────────────────────────
    print("[2/5] Injecting LoRA adapters …")
    unet = inject_lora(unet, rank=cfg['lora_rank'], alpha=cfg['lora_alpha'])

    if cfg.get('gradient_checkpointing', True):
        unet.enable_gradient_checkpointing()

    # ── Dataset ───────────────────────────────────────────────────
    print("[3/5] Loading dataset …")
    dataset = LaceLoRADataset(
        data_dir   = cfg['data_dir'],
        image_size = cfg['image_size'],
        tokenizer  = tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size      = cfg['batch_size'],
        shuffle         = True,
        num_workers     = 4,
        pin_memory      = True,
    )

    # ── Optimizer ─────────────────────────────────────────────────
    print("[4/5] Setting up optimizer …")
    # Only optimize LoRA parameters
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=cfg['learning_rate'])

    num_steps = math.ceil(len(loader) / cfg['gradient_accumulation'])
    total_steps = num_steps * cfg['num_train_epochs']

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg['mixed_precision'] == 'fp16'))

    # ── Training loop ─────────────────────────────────────────────
    print(f"[5/5] Training {cfg['num_train_epochs']} epochs …\n")
    global_step = 0

    for epoch in range(cfg['num_train_epochs']):
        unet.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1:3d}/{cfg['num_train_epochs']}",
                    dynamic_ncols=True, leave=False)

        for step, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(device, dtype=dtype)
            input_ids    = batch['input_ids'].to(device)

            with torch.cuda.amp.autocast(enabled=(cfg['mixed_precision'] == 'fp16')):
                # Encode images → latents
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise and timesteps
                noise      = torch.randn_like(latents)
                timesteps  = torch.randint(
                    0, noise_sch.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()

                # Add noise (forward diffusion)
                noisy_lat  = noise_sch.add_noise(latents, noise, timesteps)

                # Encode text
                enc_hidden = text_enc(input_ids)[0]

                # Predict noise with UNet
                noise_pred = unet(noisy_lat, timesteps,
                                  encoder_hidden_states=enc_hidden).sample

                # MSE loss between predicted and actual noise
                loss = F.mse_loss(noise_pred.float(), noise.float())
                loss = loss / cfg['gradient_accumulation']

            scaler.scale(loss).backward()

            if (step + 1) % cfg['gradient_accumulation'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            total_loss += loss.item() * cfg['gradient_accumulation']
            pbar.set_postfix(loss=f"{loss.item()*cfg['gradient_accumulation']:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1:3d}  |  Loss: {avg_loss:.4f}")

        # Save LoRA every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_path = output_dir / f"epoch_{epoch+1:03d}"
            save_path.mkdir(exist_ok=True)
            # Save only LoRA weights (small ~50MB)
            unet.save_pretrained(save_path)
            print(f"  💾  LoRA saved → {save_path}")

    # ── Final save ────────────────────────────────────────────────
    unet.save_pretrained(output_dir)
    print(f"\n✅  LoRA training complete!")
    print(f"    Weights saved → {output_dir.resolve()}")
    print(f"\n    Next step: update config.yaml:")
    print(f"      lora_path: {output_dir.resolve()}")
    print(f"      use_lora: true")
    print(f"\n    Then run: python inference/pipeline.py --config config.yaml")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tune SD on lace dataset")
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_lora(cfg)
