"""
download_dataset.py — Download DTD lace dataset from Kaggle.

Dataset: Describable Textures Dataset (DTD)
Source : https://www.kaggle.com/datasets/jmexpert/describable-textures-dataset-dtd

Steps:
  1. Download full DTD via kagglehub
  2. Filter only lace-related categories ("lacy", "knitted", "woven", etc.)
  3. Copy filtered images → data/raw/

Usage:
    pip install kagglehub
    python download_dataset.py

    # Only lace category (strictest filter)
    python download_dataset.py --categories lacy

    # Broader fabric-like textures
    python download_dataset.py --categories lacy knitted woven fibrous
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# ── Lace-related texture categories in DTD ────────────────────────────────────
# DTD has 47 categories. These are most relevant for lace fabric generation:
DEFAULT_CATEGORIES = [
    "lacy",      # Direct lace patterns       ← most important
    "knitted",   # Knit fabric                ← similar mesh structure
    "woven",     # Woven fabric               ← similar grid structure
    "fibrous",   # Fibrous textures           ← thread-like detail
    "meshed",    # Mesh/net structures        ← similar to lace holes
    "braided",   # Braided patterns           ← similar interlocking
]


def setup_kaggle_api():
    """
    Check that Kaggle API credentials are configured.

    To set up:
      1. Go to https://www.kaggle.com → Account → API → Create New Token
      2. Download kaggle.json
      3. Place it at:
         - Linux/Mac: ~/.kaggle/kaggle.json
         - Windows  : C:\\Users\\<user>\\.kaggle\\kaggle.json
      4. chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac only)
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        print("❌  Kaggle API key not found!\n")
        print("    Steps to fix:")
        print("    1. Go to: https://www.kaggle.com/settings/account")
        print("    2. Scroll to 'API' → click 'Create New Token'")
        print("    3. Save kaggle.json to:")
        print(f"       {kaggle_json}")
        print("    4. Re-run this script.\n")
        sys.exit(1)
    else:
        print(f"✅  Kaggle API key found at {kaggle_json}")


def download_dtd(download_dir: str = "downloads") -> Path:
    """
    Download the DTD dataset using kagglehub.

    Args:
        download_dir: Where kagglehub caches the download.

    Returns:
        Path to the downloaded dataset root.
    """
    try:
        import kagglehub
    except ImportError:
        print("❌  kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    print("\n📥  Downloading DTD dataset from Kaggle …")
    print("    (This may take a few minutes — dataset is ~600 MB)\n")

    os.environ.setdefault("KAGGLE_HUB_CACHE", str(download_dir))

    path = kagglehub.dataset_download(
        "jmexpert/describable-textures-dataset-dtd"
    )

    dtd_path = Path(path)
    print(f"\n✅  Downloaded to: {dtd_path}")
    return dtd_path


def find_category_dirs(dtd_root: Path, categories: list) -> dict:
    """
    Locate subdirectories for each requested category inside DTD.

    DTD structure:
        dtd/
          images/
            banded/
            braided/
            lacy/        ← we want this
            …

    Args:
        dtd_root  : Root of the extracted DTD dataset.
        categories: List of category names to search for.

    Returns:
        Dict mapping category_name → list of image files found.
    """
    # DTD images are typically under dtd/images/<category>/
    image_roots = list(dtd_root.rglob("images"))
    if not image_roots:
        # Fallback: treat dtd_root itself as root
        image_roots = [dtd_root]

    found = {}
    for cat in categories:
        imgs = []
        for img_root in image_roots:
            cat_dir = img_root / cat
            if cat_dir.is_dir():
                imgs += list(cat_dir.glob("*.jpg"))
                imgs += list(cat_dir.glob("*.png"))
        found[cat] = imgs
        count = len(imgs)
        status = f"✅  {count} images" if count > 0 else "⚠️   Not found"
        print(f"  [{cat:12s}] {status}")

    return found


def copy_to_raw(category_images: dict, raw_dir: str = "data/raw"):
    """
    Copy filtered images into data/raw/<category>/.

    Args:
        category_images: Dict from find_category_dirs.
        raw_dir        : Destination root.
    """
    raw_dir = Path(raw_dir)
    total = 0

    for cat, paths in category_images.items():
        if not paths:
            continue
        dest = raw_dir / cat
        dest.mkdir(parents=True, exist_ok=True)

        for src in paths:
            shutil.copy2(src, dest / src.name)
            total += 1

    print(f"\n✅  Copied {total} images → {raw_dir.resolve()}")
    return total


def summarize(raw_dir: str = "data/raw"):
    """Print per-category image counts in data/raw."""
    raw_dir = Path(raw_dir)
    print(f"\n📊  Dataset summary ({raw_dir}):\n")
    grand_total = 0
    for cat_dir in sorted(raw_dir.iterdir()):
        if cat_dir.is_dir():
            n = len(list(cat_dir.glob("*")))
            print(f"  {cat_dir.name:15s} → {n} images")
            grand_total += n
    print(f"\n  Total: {grand_total} images ready for training.")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download & filter DTD lace dataset from Kaggle"
    )
    parser.add_argument(
        '--categories', nargs='+', default=DEFAULT_CATEGORIES,
        help="DTD categories to include (default: lacy knitted woven fibrous meshed braided)"
    )
    parser.add_argument(
        '--raw_dir', default='data/raw',
        help="Destination folder for filtered images"
    )
    parser.add_argument(
        '--download_dir', default='downloads/dtd',
        help="Local cache folder for kagglehub download"
    )
    parser.add_argument(
        '--lacy_only', action='store_true',
        help="Download ONLY the 'lacy' category (strictest filter)"
    )
    args = parser.parse_args()

    if args.lacy_only:
        args.categories = ["lacy"]

    print("=" * 55)
    print("  🧵  DTD Lace Dataset Downloader")
    print("=" * 55)
    print(f"\n  Categories : {', '.join(args.categories)}")
    print(f"  Output dir : {args.raw_dir}\n")

    # Step 1 — Verify Kaggle API key
    setup_kaggle_api()

    # Step 2 — Download DTD
    dtd_root = download_dtd(args.download_dir)

    # Step 3 — Find category directories
    print(f"\n🔍  Scanning DTD for categories …")
    category_images = find_category_dirs(dtd_root, args.categories)

    # Step 4 — Copy to data/raw/
    print(f"\n📁  Copying to {args.raw_dir} …")
    total = copy_to_raw(category_images, args.raw_dir)

    if total == 0:
        print("\n⚠️  No images were copied. Check that DTD extracted correctly.")
        sys.exit(1)

    # Step 5 — Summary
    summarize(args.raw_dir)

    print("\n🎉  Done! Next step:")
    print("    python preprocess.py --size 256")
    print("    python stage1_gan/train.py --config config.yaml\n")


if __name__ == "__main__":
    main()
