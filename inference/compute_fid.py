"""
compute_fid.py — Tính FID Score cho Stage 1 (GAN) và Stage 2 (Refined).

Usage:
    python inference/compute_fid.py --config config.yaml

Requires:
    pip install pytorch-fid
"""

import sys
import shutil
import argparse
import tempfile
from pathlib import Path

import yaml


def collect_images_to_flat_dir(src_dirs, dst_dir, exts=(".png", ".jpg", ".jpeg")):
    """Copy tất cả ảnh từ nhiều thư mục (có thể lồng nhau) ra 1 thư mục phẳng."""
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in src_dirs:
        for ext in exts:
            for p in Path(src).rglob(f"*{ext}"):
                dst_file = dst / f"{p.parent.name}_{p.name}"
                shutil.copy2(p, dst_file)
                count += 1
    print(f"  Copied {count} images → {dst}")
    return count


def run_fid(real_dir: str, fake_dir: str, device: str = "cpu") -> float:
    """Tính FID giữa 2 thư mục ảnh bằng pytorch_fid."""
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print("[FID] pytorch_fid chưa cài. Chạy: pip install pytorch-fid")
        return -1.0

    fid = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=50,
        device=device,
        dims=2048,          # InceptionV3 feature dim
        num_workers=4,
    )
    return fid


def main():
    parser = argparse.ArgumentParser(description="Compute FID Score")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--device",      default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--real_dir",    default=None,
                        help="Override real images dir (default: config data.processed_dir)")
    parser.add_argument("--gan_dir",     default=None)
    parser.add_argument("--refined_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    real_dir    = args.real_dir    or cfg["data"]["processed_dir"]
    gan_dir     = args.gan_dir     or cfg["inference"]["output_gan_dir"]
    refined_dir = args.refined_dir or cfg["inference"]["output_final_dir"]

    print(f"\n{'━'*55}")
    print(f"  📊  FID Score Computation")
    print(f"  Real images : {real_dir}")
    print(f"  GAN outputs : {gan_dir}")
    print(f"  Refined     : {refined_dir}")
    print(f"  Device      : {args.device}")
    print(f"{'━'*55}\n")

    # Real images có thể nằm trong các subdir → collect về 1 dir phẳng
    with tempfile.TemporaryDirectory() as real_flat:
        print("▶  Collecting real images …")
        n_real = collect_images_to_flat_dir([real_dir], real_flat)

        if n_real < 10:
            print(f"  ⚠ Only {n_real} real images found. FID may be unreliable.")

        # ── Stage 1 FID ───────────────────────────────────────
        print("\n▶  Computing FID — Stage 1 (GAN) …")
        n_gan = len(list(Path(gan_dir).glob("*.png"))) + \
                len(list(Path(gan_dir).glob("*.jpg")))
        if n_gan == 0:
            print(f"  ⚠ No GAN images in {gan_dir}. Run pipeline first.")
            fid_stage1 = -1.0
        else:
            fid_stage1 = run_fid(real_flat, gan_dir, args.device)
            print(f"  ✅  FID Stage 1 (GAN only)  = {fid_stage1:.2f}")

        # ── Stage 2 FID ───────────────────────────────────────
        print("\n▶  Computing FID — Stage 2 (Refined) …")
        n_refined = len(list(Path(refined_dir).glob("*.png"))) + \
                    len(list(Path(refined_dir).glob("*.jpg")))
        if n_refined == 0:
            print(f"  ⚠ No refined images in {refined_dir}. Run pipeline first.")
            fid_stage2 = -1.0
        else:
            fid_stage2 = run_fid(real_flat, refined_dir, args.device)
            print(f"  ✅  FID Stage 2 (Refined)   = {fid_stage2:.2f}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'━'*55}")
    print(f"  📊  RESULTS SUMMARY")
    print(f"{'━'*55}")
    print(f"  FID Stage 1 (GAN)     : {fid_stage1:.2f}")
    print(f"  FID Stage 2 (Refined) : {fid_stage2:.2f}")
    if fid_stage1 > 0 and fid_stage2 > 0:
        delta = fid_stage1 - fid_stage2
        sign  = "↓" if delta > 0 else "↑"
        print(f"  Improvement           : {sign} {abs(delta):.2f}  "
              f"({'better' if delta > 0 else 'worse'} after SD refinement)")
    print(f"\n  ✏  Ghi vào báo cáo:")
    print(f"     FID Stage 1 (GAN) = {fid_stage1:.1f}")
    print(f"     FID Stage 2 (SD+LoRA) = {fid_stage2:.1f}")
    print(f"{'━'*55}\n")


if __name__ == "__main__":
    main()
