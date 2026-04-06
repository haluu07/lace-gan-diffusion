"""
preprocess.py — Dataset preprocessing for lace fabric images.

Steps:
  1. Read images from data/raw/
  2. Resize to target resolution
  3. (Optional) Convert grayscale → RGB
  4. Save to data/processed/

Usage:
    python preprocess.py --size 256
    python preprocess.py --size 512 --raw_dir path/to/raw
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm


SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def preprocess(
    raw_dir: str   = 'data/raw',
    out_dir: str   = 'data/processed',
    image_size: int = 256,
    keep_aspect: bool = False,
):
    """
    Preprocess lace images: resize + convert to RGB.

    Args:
        raw_dir    : Source directory (may be nested).
        out_dir    : Where to save processed images.
        image_size : Target square size.
        keep_aspect: If True, pad to square instead of stretching.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images
    paths = []
    for ext in SUPPORTED:
        paths += list(raw_dir.rglob(f"*{ext}"))
        paths += list(raw_dir.rglob(f"*{ext.upper()}"))
    paths = sorted(set(paths))

    if not paths:
        print(f"❌  No images found in '{raw_dir}'. "
              f"Add lace images and re-run.")
        return

    print(f"Found {len(paths)} images in '{raw_dir}'")
    print(f"Saving {image_size}×{image_size} images to '{out_dir}'\n")

    ok = skipped = 0

    for img_path in tqdm(paths, desc="Processing"):
        try:
            img = Image.open(img_path).convert('RGB')

            if keep_aspect:
                # Resize longest side, then pad to square
                img.thumbnail((image_size, image_size),
                               Image.Resampling.LANCZOS)
                padded = Image.new('RGB', (image_size, image_size),
                                   (255, 255, 255))  # white bg
                x = (image_size - img.width)  // 2
                y = (image_size - img.height) // 2
                padded.paste(img, (x, y))
                img = padded
            else:
                img = img.resize((image_size, image_size),
                                  Image.Resampling.LANCZOS)

            # Mirror directory structure under out_dir
            rel = img_path.relative_to(raw_dir)
            save_path = out_dir / rel.with_suffix('.png')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path, format='PNG')
            ok += 1

        except Exception as e:
            print(f"\n  ⚠️  Skipped '{img_path.name}': {e}")
            skipped += 1

    print(f"\n✅  Done! Processed: {ok} | Skipped: {skipped}")
    print(f"    Images are in: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess lace dataset")
    parser.add_argument('--raw_dir',    default='data/raw')
    parser.add_argument('--out_dir',    default='data/processed')
    parser.add_argument('--size',       type=int, default=256,
                        dest='image_size')
    parser.add_argument('--keep_aspect', action='store_true',
                        help="Pad to square instead of stretching")
    args = parser.parse_args()

    preprocess(
        raw_dir    = args.raw_dir,
        out_dir    = args.out_dir,
        image_size  = args.image_size,
        keep_aspect = args.keep_aspect,
    )
