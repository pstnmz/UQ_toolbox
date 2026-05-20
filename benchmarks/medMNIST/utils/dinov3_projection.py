"""
Project medMNIST benchmark datasets into a frozen DINOv3 latent space.

Saves one .npz per (flag, split) containing:
    embeddings  : (N, D) float32 — pooled [CLS] token from DINOv3
    labels      : (N,)   int64   — ground-truth class indices

NOTE: DINOv3 is a gated model on HuggingFace.  Before running:
    1. Accept Meta's licence at https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
    2. Log in locally:  huggingface-cli login

Usage
-----
    # Project train + test splits for one flag
    python dinov3_projection.py --flag organamnist

    # Multiple flags, specific splits
    python dinov3_projection.py --flags organamnist pathmnist breastmnist --splits train test

    # All standard benchmark flags
    python dinov3_projection.py --all-flags --splits train test

    # Larger model
    python dinov3_projection.py --flag organamnist --model facebook/dinov3-vitl16-pretrain-lvd1689m

Available DINOv3 models (all on HuggingFace under facebook/):
    dinov3-vits16-pretrain-lvd1689m    (21 M,  384-dim)
    dinov3-vitb16-pretrain-lvd1689m    (86 M,  768-dim)  ← default
    dinov3-vitl16-pretrain-lvd1689m   (300 M, 1024-dim)
    dinov3-vith16plus-pretrain-lvd1689m(840 M, 1280-dim)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Bypass the institutional HTTPS proxy for HuggingFace.
# System proxy (wpad.curie.fr:443) returns 503 for HTTPS CONNECT tunnels;
# direct access to huggingface.co works fine.
# ---------------------------------------------------------------------------
for _var in ("NO_PROXY", "no_proxy"):
    _existing = os.environ.get(_var, "")
    _hf_hosts = "huggingface.co,*.huggingface.co,hf.co"
    if _hf_hosts not in _existing:
        os.environ[_var] = f"{_existing},{_hf_hosts}".lstrip(",")

# ---------------------------------------------------------------------------
# Workspace root on path so Benchmarks.* imports work from any cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from Benchmarks.medMNIST.utils.datasets import load as _load_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Available flags (subset that have a train split)
# ---------------------------------------------------------------------------
ALL_STANDARD_FLAGS: List[str] = [
    "organamnist",
    "pneumoniamnist",
    "octmnist",
    "pathmnist",
    "bloodmnist",
    "tissuemnist",
    "breastmnist",
    "dermamnist-e-id",   # train split provided via dermamnist-e
]

# Flags that don't have their own train split — map to the flag that does
TRAIN_FLAG_MAP = {
    "dermamnist-e-id": "dermamnist-e",
}

# ---------------------------------------------------------------------------
# DINOv3 encoder
# ---------------------------------------------------------------------------

class DINOv3Encoder:
    """
    Frozen DINOv3 feature extractor.

    Returns the pooled [CLS] embedding (``outputs.pooler_output``) for each
    image.  The ``AutoImageProcessor`` handles all preprocessing (resize,
    normalise), so images can be passed as raw uint8 numpy arrays or PIL images.

    Parameters
    ----------
    model_name : HuggingFace model ID.  Default: facebook/dinov3-vitb16-pretrain-lvd1689m
    device     : PyTorch device string.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str = "cuda",
    ) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError("pip install transformers")

        self.device = torch.device(device)
        self.model_name = model_name

        # Support both HF model IDs and local directory paths
        local = Path(model_name).exists()
        load_kwargs = dict(local_files_only=local)

        print(f"Loading {model_name} …", end=" ", flush=True)
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, **load_kwargs)
            self.model = AutoModel.from_pretrained(model_name, device_map=str(self.device), **load_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"\n\nCould not load DINOv3 model '{model_name}'.\n"
                "This server has no internet access — download the model on a connected "
                "machine and point --model at the local directory.\n"
                "\nDownload command (run on a machine with internet):\n"
                f"  python Benchmarks/medMNIST/utils/download_dinov3.py "
                f"--model {model_name} --output-dir /path/to/save/\n"
            ) from exc
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.embed_dim: int = self.model.config.hidden_size
        print(f"done  (embed_dim={self.embed_dim}, device={self.device})")

    @torch.no_grad()
    def encode_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Encode one batch of images.

        Parameters
        ----------
        images : (B, C, H, W) float32 tensor, benchmark-normalised (μ=σ=0.5,
                 i.e. pixel values in [-1, 1]).

        Returns
        -------
        np.ndarray of shape (B, embed_dim), float32.
        """
        # Undo benchmark normalisation (μ=σ=0.5) → [0, 255] uint8 numpy
        # The AutoImageProcessor then applies its own normalisation internally.
        imgs_01 = (images * 0.5 + 0.5).clamp(0, 1)
        imgs_np = (imgs_01 * 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # (B,H,W,C)

        inputs = self.processor(
            images=list(imgs_np),
            return_tensors="pt",
            do_rescale=False,   # already in [0,255] uint8 — processor rescales internally
        ).to(self.device)

        outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Projection: forward a full split through the encoder and save
# ---------------------------------------------------------------------------

def project_and_save(
    encoder: DINOv3Encoder,
    flag: str,
    split: str,
    output_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    overwrite: bool = False,
) -> Path:
    """
    Extract DINOv3 embeddings for one (flag, split) pair and save to disk.

    Output file: ``{output_dir}/{flag}_{split}.npz``
    Contains arrays:  ``embeddings`` (N, D) float32,  ``labels`` (N,) int64.

    Returns the path to the saved file.
    """
    safe_model = encoder.model_name.split("/")[-1]
    out_path = output_dir / f"{flag}_{split}_{safe_model}.npz"

    if out_path.exists() and not overwrite:
        print(f"  [skip]  {out_path.name}  (already exists, use --overwrite to redo)")
        return out_path

    # Resolve which flag provides this split (train for dermamnist-e-id → dermamnist-e)
    load_flag = TRAIN_FLAG_MAP.get(flag, flag) if split == "train" else flag

    print(f"  Projecting {load_flag}/{split} …", end=" ", flush=True)
    bds = _load_dataset(load_flag, split=split, batch_size=batch_size, num_workers=num_workers)

    all_embs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in bds.loader:
        images, labels = batch[0], batch[1]
        all_embs.append(encoder.encode_batch(images))
        all_labels.append(labels.numpy().reshape(-1))

    embeddings = np.concatenate(all_embs,  axis=0).astype(np.float32)
    labels_arr = np.concatenate(all_labels, axis=0).astype(np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, embeddings=embeddings, labels=labels_arr)
    print(f"{len(embeddings)} samples → {out_path.name}")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--flag",      metavar="FLAG",  help="Single dataset flag.")
    src.add_argument("--flags",     nargs="+", metavar="FLAG", help="One or more dataset flags.")
    src.add_argument("--all-flags", action="store_true", help="Run all standard benchmark flags.")

    p.add_argument(
        "--splits", nargs="+", default=["train", "test"], metavar="SPLIT",
        help="Splits to project (default: train test).",
    )
    p.add_argument(
        "--model",
        default="facebook/dinov2-vitb14",
        help=(
            "HuggingFace model ID or path to a local model directory.\n"
            "Default: facebook/dinov2-vitb14 (public, no gating).\n"
            "DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m) requires manual "
            "approval from Meta — swap in once access is granted."
        ),
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "dinov3_embeddings",
        help="Where to save .npz files (default: results/dinov3_embeddings/).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract even if the output file already exists.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.flag:
        flags = [args.flag]
    elif args.all_flags:
        flags = ALL_STANDARD_FLAGS
    else:
        flags = args.flags

    encoder = DINOv3Encoder(model_name=args.model, device=args.device)

    for flag in flags:
        print(f"\n── {flag} ──")
        for split in args.splits:
            try:
                project_and_save(
                    encoder=encoder,
                    flag=flag,
                    split=split,
                    output_dir=args.output_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                print(f"  [error] {flag}/{split}: {exc}")

    print(f"\nEmbeddings saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
