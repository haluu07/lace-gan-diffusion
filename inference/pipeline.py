"""
pipeline.py — End-to-end inference: GAN → Stable Diffusion.

Stage 1: Load trained Generator → sample random latents → save coarse images.
Stage 2: Load LaceRefiner → img2img refine → save final images.

Usage:
    # Full pipeline
    python inference/pipeline.py --config config.yaml --num_images 10 --style vintage

    # Stage 1 only (no SD)
    python inference/pipeline.py --config config.yaml --skip_diffusion

    # Use specific checkpoint
    python inference/pipeline.py --config config.yaml \
        --checkpoint checkpoints/gan/epoch_0100.pth
"""

import sys
import argparse
import yaml
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stage1_gan.model          import Generator
from stage2_diffusion.refiner  import LaceRefiner
from stage2_diffusion.prompts  import get_prompt, list_styles


# ══════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    Convert [C, H, W] tensor in [-1, 1] to 8-bit PIL RGB image.
    """
    t = t.clamp(-1, 1).cpu()
    t = ((t + 1) / 2 * 255).byte()          # → [0, 255]
    arr = t.permute(1, 2, 0).numpy()        # [H, W, C]
    return Image.fromarray(arr, mode='RGB')


def load_generator(checkpoint: str, cfg_gan: dict,
                   image_size: int, device: torch.device) -> Generator:
    """Load Generator and its weights from a checkpoint file."""
    G = Generator(
        z_dim      = cfg_gan['latent_dim'],
        w_dim      = cfg_gan['latent_dim'],
        ngf        = cfg_gan['ngf'],
        image_size = image_size,
    ).to(device)

    state = torch.load(checkpoint, map_location=device)

    # Handle full training checkpoint vs bare state_dict
    if 'generator_state_dict' in state:
        state = state['generator_state_dict']
    G.load_state_dict(state)
    G.eval()
    print(f"  ✓ Generator loaded from '{checkpoint}'")
    return G


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_pipeline(
    config: dict,
    num_images: int    = 10,
    style: str         = "default",
    checkpoint: str    = None,
    skip_diffusion: bool = False,
    seed: int          = 42,
):
    """
    Run the full GAN → Diffusion lace generation pipeline.

    Args:
        config         : Loaded YAML config dict.
        num_images     : How many lace images to generate.
        style          : Prompt style (see stage2_diffusion/prompts.py).
        checkpoint     : Optional override for GAN checkpoint path.
        skip_diffusion : If True, only run Stage 1.
        seed           : Global random seed.
    """
    cfg_gan  = config['stage1_gan']
    cfg_diff = config['stage2_diffusion']
    cfg_inf  = config['inference']
    img_size = config['data']['image_size']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)

    print(f"\n{'━'*60}")
    print(f"  🎨  Lace Generation Pipeline")
    print(f"  Device     : {device}")
    print(f"  Images     : {num_images}")
    print(f"  Style      : {style}")
    print(f"  Skip SD    : {skip_diffusion}")
    print(f"{'━'*60}\n")

    # ── Output directories ────────────────────────────────────────
    gan_dir   = Path(cfg_inf['output_gan_dir'])
    final_dir = Path(cfg_inf['output_final_dir'])
    gan_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    #  STAGE 1 — GAN Generation
    # ══════════════════════════════════════════════════════════════
    print("▶  STAGE 1 — GAN Generation")
    print("─" * 40)

    ckpt_path = checkpoint or cfg_inf['gan_checkpoint']
    G = load_generator(ckpt_path, cfg_gan, img_size, device)

    gan_paths = []
    with torch.no_grad():
        for i in tqdm(range(num_images), desc="  Generating"):
            z   = torch.randn(1, cfg_gan['latent_dim'], device=device)
            img = G(z)                          # [1, 3, H, W]
            pil = tensor_to_pil(img[0])

            out = gan_dir / f"gan_{i:04d}.png"
            pil.save(out)
            gan_paths.append(out)

    print(f"  ✅  {len(gan_paths)} GAN images → {gan_dir}\n")

    if skip_diffusion:
        print("  ⏭  Diffusion skipped (--skip_diffusion)")
        return gan_paths, []

    # ══════════════════════════════════════════════════════════════
    #  STAGE 2 — Diffusion Refinement
    # ══════════════════════════════════════════════════════════════
    print("▶  STAGE 2 — Diffusion Refinement")
    print("─" * 40)

    prompt, neg_prompt = get_prompt(style)
    print(f"  Prompt : {prompt[:70]}…")
    print(f"  Strength      : {cfg_diff['strength']}")
    print(f"  Guidance scale: {cfg_diff['guidance_scale']}\n")

    refiner = LaceRefiner(
        model_id                = cfg_diff['model_id'],
        device                  = str(device),
        enable_attention_slicing= cfg_diff['enable_attention_slicing'],
        enable_vae_slicing      = cfg_diff['enable_vae_slicing'],
        enable_cpu_offload      = cfg_diff['enable_cpu_offload'],
    )

    refined_paths = refiner.refine_batch(
        image_paths         = gan_paths,
        output_dir          = str(final_dir),
        prompt              = prompt,
        negative_prompt     = neg_prompt,
        strength            = cfg_diff['strength'],
        guidance_scale      = cfg_diff['guidance_scale'],
        num_inference_steps = cfg_diff['num_inference_steps'],
    )

    print(f"\n{'━'*60}")
    print(f"  ✅  Pipeline complete!")
    print(f"  GAN images   → {gan_dir.resolve()}")
    print(f"  Final images → {final_dir.resolve()}")
    print(f"{'━'*60}\n")

    return gan_paths, refined_paths


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lace Generation Pipeline")
    parser.add_argument('--config',         default='config.yaml')
    parser.add_argument('--num_images',     type=int, default=10)
    parser.add_argument('--style',          default='default',
                        choices=list_styles())
    parser.add_argument('--checkpoint',     default=None,
                        help="Override GAN checkpoint path")
    parser.add_argument('--skip_diffusion', action='store_true')
    parser.add_argument('--seed',           type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_pipeline(
        config         = cfg,
        num_images     = args.num_images,
        style          = args.style,
        checkpoint     = args.checkpoint,
        skip_diffusion = args.skip_diffusion,
        seed           = args.seed,
    )
