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

# ---------------------------------------------------------------------------
# Dataset utilities — use the existing data-management infrastructure
# ---------------------------------------------------------------------------
_UTILS_DIR = (
    _REPO_ROOT
    / "Benchmarks" / "medMNIST" / "utils"
    / "data_preprocessing_classification_evaluation"
)
sys.path.insert(0, str(_UTILS_DIR))

from dataset_utils import (          # noqa: E402
    get_transforms,
    apply_specific_corruption,
    apply_random_corruptions,
    get_available_corruptions,
    load_amos_dataset,
    load_amos_for_new_class_shift,
    load_hmu_crc_dataset,
    load_midog_for_new_class_shift,
)
from Benchmarks.medMNIST.utils.train_models_load_datasets import (  # noqa: E402
    get_datasets,
)
from Benchmarks.medMNIST.utils.hub import ensure_medmnist_npz  # noqa: E402

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

# Flags whose images are RGB (color=True for get_transforms)
_COLOR_FLAGS: set[str] = {"pathmnist", "bloodmnist", "dermamnist",
                           "dermamnist-e", "dermamnist-e-id", "dermamnist-e-ext"}

# ---------------------------------------------------------------------------
# Special shift datasets — require custom loaders instead of _load_split
# ---------------------------------------------------------------------------
# Population shift: models trained on a source dataset, evaluated on a related
# target distribution with the same label space.
SPECIAL_FLAGS_POPULATION_SHIFT: List[str] = [
    "amos22",           # AMOS-2022 abdominal CT → OrganaMNIST label space (6 organs)
    "hmu-crc",          # HMU-CRC-Hist550K histology slides → PathMNIST label space
    "dermamnist-e-ext", # dermamnist-e external test center (test-only split)
]
# New-class shift: test set contains OOD classes absent from training.
# Saved NPZ includes a ``binary_gt`` array (0=known, 1=OOD).
SPECIAL_FLAGS_NEW_CLASS: List[str] = [
    "amos22_new_classes",  # AMOS-2022 unmapped organs (OOD) + mapped organs
    "midog",               # MIDOG++ canine tumours (OOD) + PathMNIST test samples
]
ALL_SPECIAL_FLAGS: List[str] = SPECIAL_FLAGS_POPULATION_SHIFT + SPECIAL_FLAGS_NEW_CLASS


def _load_split(
    flag: str,
    split: str,
    batch_size: int = 64,
    num_workers: int = 4,
    corruption: Optional[str] = None,
    severity: int = 3,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader for one (flag, split) pair.

    Delegates to ``get_datasets`` (train_models_load_datasets) for consistent
    dataset construction, ``get_transforms`` (dataset_utils) for normalisation,
    and ``apply_specific_corruption`` / ``apply_random_corruptions`` (dataset_utils)
    for optional medmnistc covariate-shift corruptions.
    """
    from torch.utils.data import DataLoader

    color = flag in _COLOR_FLAGS
    transform, _ = get_transforms(color)

    # dermamnist-e-id / dermamnist-e-ext are sub-views of the base dermamnist-e flag
    load_flag = "dermamnist-e" if flag in ("dermamnist-e-id", "dermamnist-e-ext") else flag
    test_subset = (
        "id"       if flag == "dermamnist-e-id"
        else "external" if flag == "dermamnist-e-ext"
        else "all"
    )

    # For standard medMNIST flags: ensure the 224px NPZ is in ~/.medmnist
    # (tries HF Hub first, then falls back to medmnist's Zenodo download)
    if load_flag not in ("dermamnist-e",):
        ensure_medmnist_npz(load_flag, size=224)

    datasets, _ = get_datasets(
        load_flag,
        im_size=224,
        color=color,
        transform=transform,
        test_subset=test_subset,
    )

    split_idx = {"train": 0, "val": 1, "test": 2}[split]
    dataset = datasets[split_idx]

    # Apply medmnistc corruption if requested
    if corruption is not None:
        if corruption == "random":
            dataset = apply_random_corruptions(dataset, load_flag, severity, seed=42)
        else:
            dataset = apply_specific_corruption(dataset, load_flag, corruption, severity)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

def _load_special_dataset(
    flag: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple:
    """
    Build a DataLoader for a special shift dataset (population-shift or new-class shift).

    Returns
    -------
    loader     : DataLoader yielding (images, labels) batches.
                 images are benchmark-normalised (μ=σ=0.5), labels are int64.
                 For new-class flags, labels are OrganaMNIST / PathMNIST indices
                 for known-class samples and -1 for OOD samples.
    binary_gt  : np.ndarray of shape (N,) with 0=known-class / 1=OOD, or None
                 for population-shift flags where every label is a valid class index.
    """
    from torch.utils.data import DataLoader

    workspace_root = _REPO_ROOT

    if flag == "amos22":
        # Population shift: AMOS CT → OrganaMNIST labels (6 mapped organs only)
        color = False
        transform, _ = get_transforms(color)
        test_dataset, _, _, _, _ = load_amos_dataset(
            transform, None, batch_size=batch_size,
            workspace_root=workspace_root
        )
        return DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ), None

    if flag == "amos22_new_classes":
        # New-class shift: project ONLY unmapped OOD organs (not merged with known samples)
        from Benchmarks.medMNIST.utils.data_preprocessing_classification_evaluation import dataset_utils as du
        color = False
        transform, _ = get_transforms(color)
        
        # Load raw AMOS data
        amos_path = workspace_root / 'benchmarks' / 'medMNIST' / 'Data' / 'AMOS_2022' / 'amos_external_test_224.npz'
        amos_data = np.load(str(amos_path), allow_pickle=True)
        amos_images = amos_data['test_images']
        amos_labels = amos_data['test_labels']
        
        # Extract only unmapped (OOD) organs
        amos_to_organamnist = {0: 10, 1: 5, 2: 4, 5: 6, 9: 9, 13: 0}
        ood_indices = []
        for idx in range(len(amos_labels)):
            amos_organ_id = np.argmax(amos_labels[idx])
            if amos_organ_id not in amos_to_organamnist:
                ood_indices.append(idx)
        
        ood_images = amos_images[ood_indices]
        ood_labels = np.full(len(ood_images), -1, dtype=np.int64)  # -1 = OOD
        
        # Create dataset
        test_dataset = du.AMOSDataset(ood_images, ood_labels, transform=transform)
        loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        return loader, None  # No binary_gt for OOD-only dataset

    if flag == "hmu-crc":
        # Population shift: HMU-CRC histology → PathMNIST labels
        color = True
        transform, _ = get_transforms(color)
        test_dataset, test_loader, _ = load_hmu_crc_dataset(
            transform, None, batch_size=batch_size,
            workspace_root=workspace_root, num_workers=num_workers
        )
        return test_loader, None

    if flag == "dermamnist-e-ext":
        # Population shift: external test center of dermamnist-e (test split only)
        return _load_split(
            "dermamnist-e-ext", split="test",
            batch_size=batch_size, num_workers=num_workers,
        ), None

    if flag == "midog":
        # New-class shift: project ONLY MIDOG++ canine patches (not merged with PathMNIST)
        color = True
        transform, _ = get_transforms(color)
        
        # Load MIDOG patches
        midog_path = workspace_root / 'benchmarks' / 'medMNIST' / 'Data' / 'MIDOG++' / 'midog_canine_patches.npz'
        midog_data = np.load(str(midog_path), allow_pickle=True)
        midog_images = midog_data['images']  # (N, 224, 224, 3) uint8
        midog_labels = np.full(len(midog_images), -1, dtype=np.int64)  # -1 = OOD
        
        # Create custom RGB dataset
        class RGBImageDataset(torch.utils.data.Dataset):
            def __init__(self, images, labels, transform=None):
                self.images = images
                self.labels = labels
                self.transform = transform
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                from PIL import Image as PILImage
                img_pil = PILImage.fromarray(self.images[idx].astype(np.uint8), mode='RGB')
                if self.transform:
                    img_tensor = self.transform(img_pil)
                else:
                    img_tensor = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
                return img_tensor, self.labels[idx]
        
        test_dataset = RGBImageDataset(midog_images, midog_labels, transform=transform)
        loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        return loader, None  # No binary_gt for OOD-only dataset

    raise ValueError(f"Unknown special flag: {flag!r}. "
                     f"Known special flags: {ALL_SPECIAL_FLAGS}")


# ---------------------------------------------------------------------------
# Projection: special shift datasets
# ---------------------------------------------------------------------------

def project_and_save_special(
    encoder: DINOv3Encoder,
    flag: str,
    output_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    overwrite: bool = False,
) -> Path:
    """
    Extract DINOv3 embeddings for a special shift dataset and save to disk.

    Output file: ``{output_dir}/{flag}_{model}.npz``
    Contains arrays:
        embeddings  : (N, D) float32
        labels      : (N,)   int64  (−1 for OOD samples in new-class flags)
        binary_gt   : (N,)   int64  (0=known-class, 1=OOD) — new-class flags only

    Returns the path to the saved file.
    """
    safe_model = encoder.model_name.split("/")[-1]
    out_path = output_dir / f"{flag}_{safe_model}.npz"

    if out_path.exists() and not overwrite:
        print(f"  [skip]  {out_path.name}  (already exists, use --overwrite to redo)")
        return out_path

    print(f"  Projecting {flag} …", end=" ", flush=True)
    loader, binary_gt = _load_special_dataset(flag, batch_size=batch_size, num_workers=num_workers)

    all_embs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        images, labels = batch[0], batch[1]
        all_embs.append(encoder.encode_batch(images))
        all_labels.append(labels.numpy().reshape(-1))

    embeddings = np.concatenate(all_embs,  axis=0).astype(np.float32)
    labels_arr = np.concatenate(all_labels, axis=0).astype(np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict = dict(embeddings=embeddings, labels=labels_arr)
    if binary_gt is not None:
        save_kwargs["binary_gt"] = binary_gt.astype(np.int64)
        shift_type = "new-class shift"
    else:
        shift_type = "population shift"
    np.savez_compressed(out_path, **save_kwargs)
    print(f"{len(embeddings)} samples ({shift_type}) → {out_path.name}")
    return out_path


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
        device: str = "cuda:1",
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
        # Undo benchmark normalisation (μ=σ=0.5) → [0, 1] float32 numpy
        # The AutoImageProcessor applies its own resize + normalisation.
        imgs_01 = (images * 0.5 + 0.5).clamp(0, 1)
        imgs_np = imgs_01.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C) float32

        inputs = self.processor(
            images=list(imgs_np),
            return_tensors="pt",
            do_rescale=False,   # already in [0, 1] float — skip processor rescaling
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
    corruption: Optional[str] = None,
    severity: int = 3,
) -> Path:
    """
    Extract DINOv3 embeddings for one (flag, split) pair and save to disk.

    Output file: ``{output_dir}/{flag}_{split}_{model}[_{corruption}_s{severity}].npz``
    Contains arrays:  ``embeddings`` (N, D) float32,  ``labels`` (N,) int64.

    Returns the path to the saved file.
    """
    safe_model = encoder.model_name.split("/")[-1]
    corrupt_suffix = f"_{corruption}_s{severity}" if corruption is not None else ""
    out_path = output_dir / f"{flag}_{split}_{safe_model}{corrupt_suffix}.npz"

    if out_path.exists() and not overwrite:
        print(f"  [skip]  {out_path.name}  (already exists, use --overwrite to redo)")
        return out_path

    # Resolve which flag provides this split (train for dermamnist-e-id → dermamnist-e)
    load_flag = TRAIN_FLAG_MAP.get(flag, flag) if split == "train" else flag

    corrupt_label = f" [{corruption} s{severity}]" if corruption is not None else ""
    print(f"  Projecting {load_flag}/{split}{corrupt_label} …", end=" ", flush=True)
    loader = _load_split(load_flag, split=split, batch_size=batch_size,
                         num_workers=num_workers,
                         corruption=corruption, severity=severity)

    all_embs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
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
    src.add_argument("--flag",      metavar="FLAG",  help="Single standard or special flag.")
    src.add_argument("--flags",     nargs="+", metavar="FLAG", help="One or more standard flags.")
    src.add_argument("--all-flags", action="store_true", help="Run all standard benchmark flags.")
    src.add_argument(
        "--special-flags", nargs="+", metavar="FLAG",
        choices=ALL_SPECIAL_FLAGS,
        help=(
            "One or more special shift dataset flags. "
            f"Population-shift: {SPECIAL_FLAGS_POPULATION_SHIFT}. "
            f"New-class-shift (saves binary_gt): {SPECIAL_FLAGS_NEW_CLASS}."
        ),
    )
    src.add_argument(
        "--all-special-flags", action="store_true",
        help="Run all special shift datasets (amos22, hmu-crc, amos22_new_classes, midog).",
    )

    p.add_argument(
        "--splits", nargs="+", default=["train", "test"], metavar="SPLIT",
        help="Splits to project for standard flags (default: train test). Ignored for special flags.",
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
    p.add_argument(
        "--corruption", metavar="NAME", default=None,
        help=(
            "medmnistc corruption to apply to the test split, e.g. 'gaussian_noise'. "
            "Use 'random' to pick one corruption randomly per sample. "
            "Omit for clean (uncorrupted) images."
        ),
    )
    p.add_argument(
        "--all-corruptions", action="store_true",
        help="Run every corruption available for each flag (overrides --corruption).",
    )
    p.add_argument(
        "--severity", type=int, default=3, choices=range(1, 6), metavar="1-5",
        help="Corruption severity level 1-5 (default: 3, matching benchmark convention).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Determine whether we're running standard flags or special shift flags
    if getattr(args, "all_special_flags", False):
        special_flags = ALL_SPECIAL_FLAGS
        standard_flags: List[str] = []
    elif getattr(args, "special_flags", None):
        special_flags = args.special_flags
        standard_flags = []
    else:
        special_flags = []
        if args.flag and args.flag in ALL_SPECIAL_FLAGS:
            # Convenience: --flag also accepts special flags
            special_flags = [args.flag]
            standard_flags = []
        elif args.flag:
            standard_flags = [args.flag]
        elif getattr(args, "all_flags", False):
            standard_flags = ALL_STANDARD_FLAGS
        else:
            # --flags may mix standard and special flags — split automatically
            special_flags = [f for f in args.flags if f in ALL_SPECIAL_FLAGS]
            standard_flags = [f for f in args.flags if f not in ALL_SPECIAL_FLAGS]

    encoder = DINOv3Encoder(model_name=args.model, device=args.device)

    # ------------------------------------------------------------------
    # Standard flags (split-based, optional corruptions)
    # ------------------------------------------------------------------
    for flag in standard_flags:
        print(f"\n── {flag} ──")

        # Build the list of corruptions to iterate over
        if args.all_corruptions:
            available = list(get_available_corruptions(flag).keys())
            if not available:
                print(f"  [warn] no medmnistc corruptions available for {flag}, running clean only")
                corruptions_to_run = [None]
            else:
                corruptions_to_run = available
        elif args.corruption:
            corruptions_to_run = [args.corruption]
        else:
            corruptions_to_run = [None]

        for corruption in corruptions_to_run:
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
                        corruption=corruption,
                        severity=args.severity,
                    )
                except Exception as exc:
                    print(f"  [error] {flag}/{split}: {exc}")

    # ------------------------------------------------------------------
    # Special shift flags (no splits, no corruptions)
    # ------------------------------------------------------------------
    for flag in special_flags:
        print(f"\n── {flag} (special shift) ──")
        try:
            project_and_save_special(
                encoder=encoder,
                flag=flag,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            print(f"  [error] {flag}: {exc}")

    print(f"\nEmbeddings saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
