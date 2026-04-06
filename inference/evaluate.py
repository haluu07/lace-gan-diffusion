"""
evaluate.py — Compute FID & CLIP scores and save visual comparison.

Metrics:
  FID (Fréchet Inception Distance)  — Lower is better.
  CLIP Score                        — Higher is better.

Usage:
    python inference/evaluate.py \
        --real_dir   data/processed \
        --gan_dir    outputs/gan_generated \
        --refined_dir outputs/final \
        --prompt "white lace fabric with intricate floral pattern, detailed"
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))


# ══════════════════════════════════════════════════════════════════
#  FID SCORE
# ══════════════════════════════════════════════════════════════════

def compute_fid(real_dir: str, fake_dir: str,
                batch_size: int = 16) -> float:
    """
    Compute FID between real and generated image directories.

    Requires: pip install pytorch-fid

    Args:
        real_dir  : Directory of real lace images.
        fake_dir  : Directory of generated (GAN or refined) images.
        batch_size: Batch size for Inception-v3 forward pass.

    Returns:
        FID score (float). Lower = more realistic.
    """
    try:
        from pytorch_fid import fid_score
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fid = fid_score.calculate_fid_given_paths(
            paths      = [str(real_dir), str(fake_dir)],
            batch_size = batch_size,
            device     = device,
            dims       = 2048,
        )
        return float(fid)
    except ImportError:
        print("  [FID] pytorch-fid not installed — run: pip install pytorch-fid")
        return float('nan')
    except Exception as e:
        print(f"  [FID] Error: {e}")
        return float('nan')


# ══════════════════════════════════════════════════════════════════
#  CLIP SCORE
# ══════════════════════════════════════════════════════════════════

def compute_clip_score(image_dir: str, prompt: str,
                       max_images: int = 200) -> float:
    """
    Compute average CLIP cosine similarity between images and prompt.

    Requires: pip install open_clip_torch

    Args:
        image_dir : Directory of generated images.
        prompt    : Text description to compare against.
        max_images: Max number of images to evaluate.

    Returns:
        Average CLIP score × 100 (float). Higher = better alignment.
    """
    try:
        import open_clip

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model     = model.to(device).eval()

        text = tokenizer([prompt]).to(device)

        paths = (list(Path(image_dir).glob('*.png')) +
                 list(Path(image_dir).glob('*.jpg')))[:max_images]

        if not paths:
            return float('nan')

        scores = []
        for p in paths:
            img = preprocess(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                i_feat = model.encode_image(img)
                t_feat = model.encode_text(text)
                i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
                t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
                scores.append((i_feat @ t_feat.T).item() * 100)

        return float(np.mean(scores))

    except ImportError:
        print("  [CLIP] open_clip_torch not installed — run: pip install open_clip_torch")
        return float('nan')
    except Exception as e:
        print(f"  [CLIP] Error: {e}")
        return float('nan')


# ══════════════════════════════════════════════════════════════════
#  VISUAL COMPARISON GRID
# ══════════════════════════════════════════════════════════════════

def make_comparison_grid(
    gan_dir: str,
    refined_dir: str,
    output_path: str = "outputs/comparison.png",
    num_samples: int = 6,
):
    """
    Save a side-by-side grid: [GAN | Refined].

    Args:
        gan_dir     : Directory of GAN images.
        refined_dir : Directory of refined images.
        output_path : Where to save the PNG comparison.
        num_samples : Number of image pairs to show.
    """
    gan_paths     = sorted(Path(gan_dir).glob('*.png'))[:num_samples]
    refined_paths = sorted(Path(refined_dir).glob('*.png'))[:num_samples]

    n = min(len(gan_paths), len(refined_paths))
    if n == 0:
        print("  [VIZ] No images found for comparison.")
        return

    fig, axes = plt.subplots(2, n, figsize=(n * 3, 7))
    fig.suptitle("Stage 1: GAN   vs   Stage 2: Refined",
                 fontsize=14, fontweight='bold', y=1.01)

    for i in range(n):
        # Row 0 — GAN output
        axes[0, i].imshow(Image.open(gan_paths[i]))
        axes[0, i].set_title(f"GAN #{i+1}", fontsize=9)
        axes[0, i].axis('off')

        # Row 1 — Refined output
        axes[1, i].imshow(Image.open(refined_paths[i]))
        axes[1, i].set_title(f"Refined #{i+1}", fontsize=9)
        axes[1, i].axis('off')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [VIZ] Comparison grid → {output_path}")


# ══════════════════════════════════════════════════════════════════
#  FULL EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════

def evaluate(
    real_dir: str,
    gan_dir: str,
    refined_dir: str,
    prompt: str,
    output_path: str = "outputs/comparison.png",
    num_visual: int  = 6,
):
    """Run all evaluations and print a summary report."""
    print(f"\n{'━'*55}")
    print(f"  📊  Evaluation Report")
    print(f"{'━'*55}\n")

    # ── FID ──────────────────────────────────────────────────────
    print("Computing FID scores …")
    fid_gan     = compute_fid(real_dir, gan_dir)
    fid_refined = compute_fid(real_dir, refined_dir)

    # ── CLIP ─────────────────────────────────────────────────────
    print("Computing CLIP scores …")
    clip_gan     = compute_clip_score(gan_dir,     prompt)
    clip_refined = compute_clip_score(refined_dir, prompt)

    # ── Visual comparison ─────────────────────────────────────────
    print("Generating visual comparison …")
    make_comparison_grid(gan_dir, refined_dir, output_path, num_visual)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'━'*55}")
    print(f"  {'Metric':<20} {'GAN':>12} {'Refined':>12}")
    print(f"  {'─'*44}")
    print(f"  {'FID ↓':<20} {fid_gan:>12.2f} {fid_refined:>12.2f}")
    print(f"  {'CLIP Score ↑':<20} {clip_gan:>12.2f} {clip_refined:>12.2f}")
    print(f"{'━'*55}")

    if not np.isnan(fid_refined) and not np.isnan(fid_gan):
        delta_fid = fid_gan - fid_refined
        emoji = "✅" if delta_fid > 0 else "⚠️"
        print(f"\n  {emoji}  FID improvement after refinement: {delta_fid:.2f}")

    print(f"\n  Comparison image saved → {output_path}\n")

    return {
        'fid_gan':      fid_gan,
        'fid_refined':  fid_refined,
        'clip_gan':     clip_gan,
        'clip_refined': clip_refined,
    }


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GAN vs Refined")
    parser.add_argument('--real_dir',    required=True)
    parser.add_argument('--gan_dir',     default='outputs/gan_generated')
    parser.add_argument('--refined_dir', default='outputs/final')
    parser.add_argument('--prompt',      default=(
        "white lace fabric with intricate floral pattern, "
        "delicate threads, fine mesh holes, ultra detailed"))
    parser.add_argument('--output',      default='outputs/comparison.png')
    parser.add_argument('--num_visual',  type=int, default=6)
    args = parser.parse_args()

    evaluate(
        real_dir   = args.real_dir,
        gan_dir    = args.gan_dir,
        refined_dir= args.refined_dir,
        prompt     = args.prompt,
        output_path= args.output,
        num_visual = args.num_visual,
    )
