"""
Download a DINOv3 model from HuggingFace to a local directory.

Run this on a machine that has internet access, then copy the output
directory to the air-gapped server and pass it to dinov3_projection.py
via --model /path/to/output-dir.

Prerequisites
-------------
    pip install huggingface_hub transformers
    huggingface-cli login
    # Accept Meta's terms at: https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

Usage
-----
    # Download ViT-B/16 (86 MB, recommended)
    python download_dinov3.py --output-dir ./dinov3-vitb16

    # Download ViT-L/16 (1.2 GB, higher quality)
    python download_dinov3.py --model facebook/dinov3-vitl16-pretrain-lvd1689m --output-dir ./dinov3-vitl16

Then on the server:
    python dinov3_projection.py --all-flags --model /path/to/dinov3-vitb16
"""

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="HuggingFace model ID to download.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local directory to save the model files.",
    )
    args = p.parse_args()

    from transformers import AutoImageProcessor, AutoModel

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model} → {args.output_dir} …")

    processor = AutoImageProcessor.from_pretrained(args.model)
    processor.save_pretrained(args.output_dir)
    print("  processor saved.")

    model = AutoModel.from_pretrained(args.model)
    model.save_pretrained(args.output_dir)
    print("  model saved.")

    print(f"\nDone.  Copy {args.output_dir}/ to the server, then run:")
    print(f"  python dinov3_projection.py --all-flags --model {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
