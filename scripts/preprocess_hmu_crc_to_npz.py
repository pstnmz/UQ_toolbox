"""
Convert the HMU-CRC-Hist550K image folder to a single compressed NPZ file.

The HMU-CRC-Hist550K dataset is organized as 8 class sub-directories of
224×224 RGB TIFF/PNG images.  This script reads them all, stacks them into
a uint8 array, and saves a single ``hmu_crc_224.npz`` file that is consistent
with the other benchmark NPZ files (dermamnist-e, amos22, midog).

Usage
-----
  # Estimate compressed size without writing anything
  python scripts/preprocess_hmu_crc_to_npz.py --dry-run

  # Build the NPZ (uses 64 parallel workers by default)
  python scripts/preprocess_hmu_crc_to_npz.py

  # Tune parallelism (96-core machine: use 80-90 workers)
  python scripts/preprocess_hmu_crc_to_npz.py --workers 80

  # Custom source / destination
  python scripts/preprocess_hmu_crc_to_npz.py \\
      --input  Benchmarks/medMNIST/data/HMU-CRC-Hist550K \\
      --output Benchmarks/medMNIST/data/HMU-CRC-Hist550K/hmu_crc_224.npz

NPZ keys
--------
  images   : (N, 224, 224, 3)  uint8   — RGB pixel values
  labels   : (N,)              int64   — class index (0–7)
  classes  : (8,)              str     — class names (alphabetical order)
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Alphabetical order (matches torchvision.datasets.ImageFolder default)
CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "Benchmarks" / "medMNIST" / "data" / "HMU-CRC-Hist550K"
DEFAULT_OUTPUT = DEFAULT_INPUT / "hmu_crc_224.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack HMU-CRC-Hist550K images into a single NPZ file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help=f"Path to HMU-CRC-Hist550K folder (default: {DEFAULT_INPUT})")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help=f"Output NPZ path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--dry-run", action="store_true",
                   help="Count images and estimate compressed size, then exit without writing.")
    p.add_argument("--limit", type=int, default=None,
                   help="Maximum images per class (useful for testing).")
    p.add_argument("--workers", type=int, default=64,
                   help="Number of parallel image-loading threads (default: 64). "
                        "On a 96-core machine, 80 is a good value.")
    return p.parse_args()


def iter_images(root: Path, limit: int | None = None):
    """Yield (image_array, label_idx) for every image found, in class order."""
    for cls in CLASSES:
        cls_dir = root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")
        files = sorted(cls_dir.iterdir())
        if limit is not None:
            files = files[:limit]
        label = CLASS_TO_IDX[cls]
        for fpath in files:
            if fpath.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
                yield fpath, label


def count_images(root: Path, limit: int | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cls in CLASSES:
        cls_dir = root / cls
        files = [f for f in sorted(cls_dir.iterdir())
                 if f.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}]
        counts[cls] = min(len(files), limit) if limit else len(files)
    return counts


def main() -> None:
    args = parse_args()
    root = Path(args.input)
    out_path = Path(args.output)

    if not root.exists():
        print(f"ERROR: Source directory not found: {root}")
        sys.exit(1)

    # Validate class folders
    missing = [c for c in CLASSES if not (root / c).exists()]
    if missing:
        print(f"ERROR: Missing class folders: {missing}")
        sys.exit(1)

    counts = count_images(root, limit=args.limit)
    N = sum(counts.values())
    raw_bytes = N * 224 * 224 * 3
    print(f"\nHMU-CRC-Hist550K  {'(dry-run)' if args.dry_run else ''}")
    print(f"  Source     : {root}")
    print(f"  Output     : {out_path}")
    print(f"  Classes    : {len(CLASSES)}")
    for cls, n in counts.items():
        print(f"    {cls:6s}  {n:>7,} images")
    print(f"  Total      : {N:,} images")
    print(f"  Raw size   : {raw_bytes / 1e9:.1f} GB  ({N} × 224×224×3 uint8)")
    print(f"  RAM needed : ~{raw_bytes / 1e9:.1f} GB  for the image array")
    print(f"  NPZ (est.) : ~{raw_bytes / 1e9 / 5:.0f}–{raw_bytes / 1e9 / 3:.0f} GB  (3–5× compression typical for histology)")

    if args.dry_run:
        print("\nDry-run complete — no files written.")
        return

    # Confirm with user if large dataset
    if N > 10_000 and args.limit is None:
        ans = input(
            f"\nAbout to load {N:,} images (~{raw_bytes / 1e9:.0f} GB) into RAM "
            f"and write compressed NPZ.\nProceed? [y/N] "
        )
        if ans.strip().lower() != "y":
            print("Aborted.")
            return

    print("\nLoading images …")
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("pip install Pillow")

    # Collect all (path, label) pairs upfront so we can pre-allocate
    all_items = list(iter_images(root, limit=args.limit))
    N_actual = len(all_items)
    images = np.empty((N_actual, 224, 224, 3), dtype=np.uint8)
    labels = np.empty(N_actual, dtype=np.int64)

    def _load_one(item):
        fpath, label, pos = item
        img = Image.open(fpath).convert("RGB")
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.BILINEAR)
        return pos, np.asarray(img, dtype=np.uint8), label

    items_with_pos = [(fpath, label, i) for i, (fpath, label) in enumerate(all_items)]

    n_workers = args.workers
    print(f"  Using {n_workers} parallel reader threads …")
    loaded = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_one, item): item for item in items_with_pos}
        for fut in as_completed(futures):
            pos, arr, lbl = fut.result()
            images[pos] = arr
            labels[pos] = lbl
            loaded += 1
            if loaded % 10_000 == 0:
                pct = 100 * loaded / N_actual
                print(f"  {loaded:>7,} / {N_actual:,}  ({pct:.1f}%)", end="\r", flush=True)

    print(f"  {loaded:>7,} / {N_actual:,}  (100.0%)  — done loading")

    print(f"\nSaving → {out_path}  …")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        labels=labels,
        classes=np.array(CLASSES),
    )
    size_gb = out_path.stat().st_size / 1e9
    print(f"Saved  → {out_path}  ({size_gb:.2f} GB)")
    print(f"\nDone.  To upload to HuggingFace Hub run:")
    print(f"  python scripts/upload_to_hub.py --datasets-only --token hf_...")


if __name__ == "__main__":
    main()
