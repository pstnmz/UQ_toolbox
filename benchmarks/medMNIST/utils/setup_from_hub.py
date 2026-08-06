"""
One-command reproducibility setup: download models + datasets from HuggingFace Hub.

Run this once after cloning the repo to get everything needed to reproduce the
benchmark without re-training or re-preprocessing.

Usage
-----
  python scripts/setup_from_hub.py

  # Private repos require an HF read token
  python scripts/setup_from_hub.py --token hf_XXXX

  # Download only models or only datasets
  python scripts/setup_from_hub.py --models-only
  python scripts/setup_from_hub.py --datasets-only

  # Choose specific datasets
  python scripts/setup_from_hub.py --datasets dermamnist-e amos22

What gets downloaded
--------------------
  Models  : 325 checkpoint files into Benchmarks/medMNIST/models/224*224/
            (~59 GB — only models needed for benchmark reproduction)

  Datasets: preprocessed NPZ files for external/custom datasets:
              • Benchmarks/medMNIST/data/ISIC_2018/
                  dermamnist_extended_224_wsitesources.npz   (DermaMNIST-E)
              • Benchmarks/medMNIST/data/AMOS_2022/
                  amos_external_test_224.npz                 (AMOS-2022)

  Standard MedMNIST datasets (bloodmnist, breastmnist, octmnist, …) are
  downloaded automatically by the medmnist package on first benchmark run —
  nothing to do here for those.

After this script completes you can run the full benchmark with:
  python Benchmarks/medMNIST/run_medmnist_benchmark.py --flag breastmnist
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BENCHMARKS_DIR = PROJECT_ROOT / "Benchmarks" / "medMNIST"

sys.path.insert(0, str(BENCHMARKS_DIR / "utils"))
from hub import (
    HF_MODELS_REPO,
    HF_DATASETS_REPO,
    download_all_models,
    download_all_datasets,
    CUSTOM_DATASET_FILES,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download FailCatcher artefacts from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--token", default=None,
                   help="HuggingFace read token (only needed for private repos)")
    p.add_argument("--models-repo", default=HF_MODELS_REPO,
                   help=f"HF model repository (default: {HF_MODELS_REPO})")
    p.add_argument("--datasets-repo", default=HF_DATASETS_REPO,
                   help=f"HF dataset repository (default: {HF_DATASETS_REPO})")
    p.add_argument("--models-only", action="store_true",
                   help="Download models only")
    p.add_argument("--datasets-only", action="store_true",
                   help="Download datasets only")
    p.add_argument("--datasets", nargs="+",
                   choices=list(CUSTOM_DATASET_FILES.keys()),
                   default=None,
                   help="Which custom datasets to download (default: all except hmu-crc). "
                        "Add 'hmu-crc' to also download the ~15-25 GB NPZ.")
    p.add_argument("--models-dir", default=None,
                   help="Override local models directory")
    p.add_argument("--data-dir", default=None,
                   help="Override local data root directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    print("FailCatcher — reproducibility setup")
    print("=" * 60)
    print(f"  Project root  : {PROJECT_ROOT}")
    print(f"  Models repo   : {args.models_repo}")
    print(f"  Datasets repo : {args.datasets_repo}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Download models
    # ------------------------------------------------------------------
    if not args.datasets_only:
        models_dir = (
            Path(args.models_dir)
            if args.models_dir
            else BENCHMARKS_DIR / "models" / "224*224"
        )
        print(f"\n[1/2] Downloading model checkpoints → {models_dir}")
        print("      (This can take a while — ~59 GB total)")
        download_all_models(
            local_dir=models_dir,
            hub_repo=args.models_repo,
            token=token,
        )

    # ------------------------------------------------------------------
    # Download custom datasets
    # ------------------------------------------------------------------
    if not args.models_only:
        data_root = (
            Path(args.data_dir) if args.data_dir else BENCHMARKS_DIR / "data"
        )
        flags = args.datasets  # None → all except hmu-crc (too large for default)
        if flags is None:
            flags = [f for f in CUSTOM_DATASET_FILES.keys() if f != "hmu-crc"]
            print(f"\n[2/2] Downloading custom dataset files → {data_root}")
            print("      (hmu-crc is skipped by default — use --datasets hmu-crc to download it)")
        else:
            print(f"\n[2/2] Downloading custom dataset files → {data_root}")
        download_all_datasets(
            local_root=data_root,
            hub_repo=args.datasets_repo,
            token=token,
            flags=flags,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Setup complete!  You can now run the benchmark:")
    print()
    print("  # Single dataset")
    print("  python Benchmarks/medMNIST/run_medmnist_benchmark.py --flag breastmnist")
    print()
    print("  # All datasets")
    print("  python Benchmarks/medMNIST/launcher_benchmark.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
