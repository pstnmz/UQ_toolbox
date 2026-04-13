"""
Unified dataset loader for the FailCatcher medMNIST benchmark.

Single entry-point ``load()`` works for all datasets used in the benchmark:
standard MedMNIST datasets, custom extended datasets, and external test sets.
Missing files are downloaded automatically from HuggingFace Hub on first use.

Quick start
-----------
>>> from Benchmarks.medMNIST.utils.datasets import load
>>>
>>> # Standard MedMNIST — uses the medmnist package (auto-downloaded)
>>> ds = load("breastmnist")
>>> ds = load("organamnist", split="train")
>>>
>>> # Extended DermaMNIST  — downloaded from HF Hub on first use
>>> ds = load("dermamnist-e")           # all test samples
>>> ds = load("dermamnist-e-id")        # ID test centers only
>>> ds = load("dermamnist-e-ext")       # external test center only
>>>
>>> # External test sets — downloaded from HF Hub on first use
>>> ds = load("amos22")                 # AMOS-2022 (OrganaMNIST models)
>>> ds = load("midog")                  # MIDOG++ patches (PathMNIST models)
>>>
>>> # Covariate shift (medmnistc corruptions)
>>> ds = load("breastmnist", corruption="gaussian_noise", severity=3)
>>>
>>> # Access the pieces
>>> ds.loader       # torch DataLoader, ready to iterate
>>> ds.dataset      # torch Dataset
>>> ds.info         # dict: task, label, n_channels, n_samples, ...
>>> ds.class_names  # list of class name strings

All flags at a glance
---------------------
Standard MedMNIST (downloaded by the medmnist package):
    bloodmnist   breastmnist   dermamnist   octmnist
    organamnist  pathmnist     pneumoniamnist  tissuemnist

Custom / extended (auto-downloaded from HF Hub):
    dermamnist-e        Full DermaMNIST-E test set
    dermamnist-e-id     DermaMNIST-E — ID test centers only
    dermamnist-e-ext    DermaMNIST-E — external test center only
    amos22              AMOS-2022 external test (OrganaMNIST label space)
    midog               MIDOG++ canine mitosis patches (PathMNIST label space)
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_hub():
    """Load hub.py via file path — robust against any import context."""
    hub_path = Path(__file__).resolve().parent / "hub.py"
    spec = importlib.util.spec_from_file_location("_failcatcher_hub", hub_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _RepeatGray:
    """Expand single-channel tensor to 3-channel (for ResNet/ViT)."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# Entries whose ``source`` is "medmnist" are handled by the medmnist package.
# Entries whose ``source`` is "hub" must be downloaded from HF Hub.
REGISTRY: dict[str, dict] = {
    # --- Standard MedMNIST ---------------------------------------------------
    "bloodmnist": {
        "source": "medmnist",
        "color": True,
        "splits": ["train", "val", "test"],
    },
    "breastmnist": {
        "source": "medmnist",
        "color": False,
        "splits": ["train", "val", "test"],
    },
    "dermamnist": {
        "source": "medmnist",
        "color": True,
        "splits": ["train", "val", "test"],
    },
    "octmnist": {
        "source": "medmnist",
        "color": False,
        "splits": ["train", "val", "test"],
    },
    "organamnist": {
        "source": "medmnist",
        "color": False,
        "splits": ["train", "val", "test"],
    },
    "pathmnist": {
        "source": "medmnist",
        "color": True,
        "splits": ["train", "val", "test"],
    },
    "pneumoniamnist": {
        "source": "medmnist",
        "color": False,
        "splits": ["train", "val", "test"],
    },
    "tissuemnist": {
        "source": "medmnist",
        "color": False,
        "splits": ["train", "val", "test"],
    },
    # --- DermaMNIST-E variants (HF Hub) -------------------------------------
    "dermamnist-e": {
        "source": "hub",
        "color": True,
        "splits": ["train", "val", "test"],
        "test_subset": "all",
        "hub_flag": "dermamnist-e",
        "npz": "dermamnist_extended_224_wsitesources.npz",
        "local_subdir": "ISIC_2018",
        "info": {
            "task": "multi-class",
            "n_channels": 3,
            "label": {
                "0": "actinic keratoses",
                "1": "basal cell carcinoma",
                "2": "benign keratosis-like lesions",
                "3": "dermatofibroma",
                "4": "melanoma",
                "5": "melanocytic nevi",
                "6": "vascular lesions",
            },
        },
    },
    "dermamnist-e-id": {
        "source": "hub",
        "color": True,
        "splits": ["test"],
        "test_subset": "id",
        "hub_flag": "dermamnist-e",
        "npz": "dermamnist_extended_224_wsitesources.npz",
        "local_subdir": "ISIC_2018",
        "info": {
            "task": "multi-class",
            "n_channels": 3,
            "label": {
                "0": "actinic keratoses",
                "1": "basal cell carcinoma",
                "2": "benign keratosis-like lesions",
                "3": "dermatofibroma",
                "4": "melanoma",
                "5": "melanocytic nevi",
                "6": "vascular lesions",
            },
        },
    },
    "dermamnist-e-ext": {
        "source": "hub",
        "color": True,
        "splits": ["test"],
        "test_subset": "external",
        "hub_flag": "dermamnist-e",
        "npz": "dermamnist_extended_224_wsitesources.npz",
        "local_subdir": "ISIC_2018",
        "info": {
            "task": "multi-class",
            "n_channels": 3,
            "label": {
                "0": "actinic keratoses",
                "1": "basal cell carcinoma",
                "2": "benign keratosis-like lesions",
                "3": "dermatofibroma",
                "4": "melanoma",
                "5": "melanocytic nevi",
                "6": "vascular lesions",
            },
        },
    },
    # --- AMOS-2022 external test (HF Hub) -----------------------------------
    "amos22": {
        "source": "hub",
        "color": False,
        "splits": ["test"],
        "hub_flag": "amos22",
        "npz": "amos_external_test_224.npz",
        "local_subdir": "AMOS_2022",
        "info": {
            "task": "multi-class",
            "n_channels": 1,
            "label": {
                "0": "bladder",
                "4": "kidney-left",
                "5": "kidney-right",
                "6": "liver",
                "9": "pancreas",
                "10": "spleen",
            },
            "note": (
                "AMOS labels are remapped to the OrganaMNIST label space. "
                "Models trained on OrganaMNIST are used for this test set."
            ),
        },
    },
    # --- MIDOG++ (HF Hub) ---------------------------------------------------
    "midog": {
        "source": "hub",
        "color": True,
        "splits": ["test"],
        "hub_flag": "midog",
        "npz": "midog_canine_patches.npz",
        "local_subdir": "MIDOG++",
        "info": {
            "task": "multi-class (OOD)",
            "n_channels": 3,
            "label": {
                "0": "canine_lymphoma",
                "1": "canine_cutaneous_mast_cell_tumor",
                "2": "canine_oral_melanoma",
                "3": "canine_mammary_carcinoma",
            },
            "note": (
                "MIDOG++ canine tumor patches. "
                "PathMNIST models are used for this OOD test set."
            ),
        },
    },
    # --- HMU-CRC-Hist550K (dedicated HF Hub repo, ImageFolder layout) ------
    "hmu-crc": {
        "source": "hub",
        "color": True,
        "splits": ["test"],
        "hub_flag": "hmu-crc",
        "npz": "hmu_crc_224.npz",
        "local_subdir": "HMU-CRC-Hist550K",
        "info": {
            "task": "multi-class (OOD, colorectal histology)",
            "n_channels": 3,
            "label": {
                "0": "ADI",  # Adipose
                "1": "DEB",  # Debris
                "2": "LYM",  # Lymphocytes
                "3": "MUC",  # Mucus
                "4": "MUS",  # Smooth muscle
                "5": "NORM", # Normal colon mucosa
                "6": "STR",  # Cancer-associated stroma
                "7": "TUM",  # Colorectal adenocarcinoma epithelium
            },
            "note": (
                "HMU-CRC-Hist550K — 556,000 colorectal histology patches (224x224 RGB). "
                "8 tissue classes. Packed as a single NPZ for convenient download. "
                "PathMNIST models are the closest match for this dataset."
            ),
        },
    },
}

# Canonical aliases
ALIASES: dict[str, str] = {
    "amos": "amos22",
    "amos2022": "amos22",
    "derm-e": "dermamnist-e",
    "derm-e-id": "dermamnist-e-id",
    "derm-e-ext": "dermamnist-e-ext",
    "dermamnist-e-external": "dermamnist-e-ext",
    "midog++": "midog",
    "hmu_crc": "hmu-crc",
    "crc": "hmu-crc",
    "nct-crc": "hmu-crc",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkDataset:
    """
    Container returned by ``load()``.

    Attributes
    ----------
    loader       : DataLoader ready to iterate — yields (image_tensor, label) batches.
    dataset      : The underlying torch.utils.data.Dataset.
    info         : Metadata dict: ``task``, ``label``, ``n_channels``, ``n_samples``.
    flag         : Canonical dataset flag used.
    split        : Which split was loaded.
    class_names  : Ordered list of class name strings.
    image_size   : Spatial resolution of images (H = W).
    """
    loader: DataLoader
    dataset: torch.utils.data.Dataset
    info: dict
    flag: str
    split: str
    class_names: List[str]
    image_size: int

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        return (
            f"BenchmarkDataset(flag={self.flag!r}, split={self.split!r}, "
            f"n={len(self)}, classes={len(self.class_names)}, "
            f"image_size={self.image_size})"
        )


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _data_root() -> Path:
    """Absolute path to Benchmarks/medMNIST/data/."""
    return Path(__file__).resolve().parent.parent / "data"


def _ensure_hub_file(hub_flag: str, npz: str, local_subdir: str) -> Path:
    """Download NPZ from HF Hub if missing; return its local path."""
    local_dir = _data_root() / local_subdir
    local_path = local_dir / npz
    if not local_path.exists():
        hub = _load_hub()
        hub.ensure_dataset_file(hub_flag, npz, local_dir=local_dir)
    return local_path


def _make_transform(color: bool, image_size: int, normalize: bool = True) -> T.Compose:
    ops: list = []
    if image_size != 224:
        ops.append(T.Resize((image_size, image_size)))
    ops.append(T.ToTensor())
    if not color:
        ops.append(_RepeatGray())
    if normalize:
        if color:
            ops.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        else:
            ops.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return T.Compose(ops)


# ---- Standard MedMNIST ----

def _load_medmnist(flag: str, split: str, image_size: int, batch_size: int,
                   num_workers: int, corruption: Optional[str],
                   severity: int, download: bool) -> BenchmarkDataset:
    import medmnist
    from medmnist import INFO

    entry = REGISTRY[flag]
    info = INFO[flag]
    color = entry["color"]
    transform = _make_transform(color, image_size, normalize=True)

    DataClass = getattr(medmnist, info["python_class"])
    dataset = DataClass(split=split, transform=transform, size=image_size, download=download)

    if corruption is not None:
        dataset = _apply_corruption(dataset, flag, corruption, severity)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                        num_workers=num_workers, pin_memory=True)
    class_names = list(info["label"].values())
    return BenchmarkDataset(loader=loader, dataset=dataset, info=info,
                            flag=flag, split=split,
                            class_names=class_names, image_size=image_size)


# ---- DermaMNIST-E ----

def _load_dermamnist_e(flag: str, split: str, image_size: int, batch_size: int,
                        num_workers: int, corruption: Optional[str],
                        severity: int, download: bool) -> BenchmarkDataset:
    entry = REGISTRY[flag]
    test_subset = entry.get("test_subset", "all")
    npz_path = _ensure_hub_file(entry["hub_flag"], entry["npz"], entry["local_subdir"])

    # Use the existing DermaMNIST_E class (it does the NPZ loading)
    _add_utils_to_path()
    from data_preprocessing_classification_evaluation.local_dermamnist_e import DermaMNIST_E

    color = entry["color"]
    transform = _make_transform(color, image_size, normalize=True)
    dataset = DermaMNIST_E(split=split, transform=transform, size=image_size,
                            download=False, root=npz_path.parent,
                            test_subset=(test_subset if split == "test" else "all"))

    if corruption is not None:
        dataset = _apply_corruption(dataset, "dermamnist", corruption, severity)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                        num_workers=num_workers, pin_memory=True)
    info = entry["info"]
    class_names = list(info["label"].values())
    return BenchmarkDataset(loader=loader, dataset=dataset, info=info,
                            flag=flag, split=split,
                            class_names=class_names, image_size=image_size)


# ---- AMOS-2022 ----

def _load_amos22(flag: str, split: str, image_size: int, batch_size: int,
                 num_workers: int, corruption: Optional[str],
                 severity: int, download: bool) -> BenchmarkDataset:
    from torch.utils.data import TensorDataset

    if split != "test":
        raise ValueError("amos22 only has a 'test' split.")

    entry = REGISTRY[flag]
    npz_path = _ensure_hub_file(entry["hub_flag"], entry["npz"], entry["local_subdir"])
    data = np.load(str(npz_path), allow_pickle=True)

    amos_images = data["test_images"]  # (N, H, W, 1) uint8
    amos_labels = data["test_labels"]  # (N, 15)

    # Remap to OrganaMNIST label space (6 mapped organs)
    amos_to_organamnist = {
        0: 10, 1: 5, 2: 4, 5: 6, 9: 9, 13: 0,
    }
    mapped_indices, mapped_labels = [], []
    for idx in range(len(amos_images)):
        organ_id = int(np.argmax(amos_labels[idx]))
        if organ_id in amos_to_organamnist:
            mapped_indices.append(idx)
            mapped_labels.append(amos_to_organamnist[organ_id])

    images_mapped = amos_images[mapped_indices]  # (M, H, W, 1)
    labels_mapped = np.array(mapped_labels)

    transform = _make_transform(color=False, image_size=image_size, normalize=True)

    class _AMOSDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, lbls, tfm):
            self.imgs = imgs
            self.lbls = lbls
            self.tfm = tfm

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            from PIL import Image as PILImage
            img = self.imgs[i].squeeze()
            pil = PILImage.fromarray(img, mode="L")
            return self.tfm(pil), torch.tensor(int(self.lbls[i]), dtype=torch.long)

    dataset = _AMOSDataset(images_mapped, labels_mapped, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    info = entry["info"]
    class_names = list(info["label"].values())
    return BenchmarkDataset(loader=loader, dataset=dataset, info=info,
                            flag=flag, split=split,
                            class_names=class_names, image_size=image_size)


# ---- MIDOG++ ----

def _load_midog(flag: str, split: str, image_size: int, batch_size: int,
                num_workers: int, corruption: Optional[str],
                severity: int, download: bool) -> BenchmarkDataset:
    if split != "test":
        raise ValueError("midog only has a 'test' split.")

    entry = REGISTRY[flag]
    npz_path = _ensure_hub_file(entry["hub_flag"], entry["npz"], entry["local_subdir"])
    data = np.load(str(npz_path), allow_pickle=True)

    images = data["images"]    # (N, H, W, 3) uint8
    labels = data["labels"]    # (N,) int

    color = entry["color"]
    transform = _make_transform(color, image_size, normalize=True)

    class _MIDOGDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, lbls, tfm):
            self.imgs = imgs
            self.lbls = lbls
            self.tfm = tfm

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            from PIL import Image as PILImage
            pil = PILImage.fromarray(self.imgs[i], mode="RGB")
            return self.tfm(pil), torch.tensor(int(self.lbls[i]), dtype=torch.long)

    dataset = _MIDOGDataset(images, labels, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    info = entry["info"]
    class_names = list(info["label"].values())
    return BenchmarkDataset(loader=loader, dataset=dataset, info=info,
                            flag=flag, split=split,
                            class_names=class_names, image_size=image_size)


# ---- HMU-CRC-Hist550K ----

def _load_hmu_crc(flag: str, split: str, image_size: int, batch_size: int,
                  num_workers: int, corruption: Optional[str],
                  severity: int, download: bool) -> BenchmarkDataset:
    """Load HMU-CRC-Hist550K from the pre-packed NPZ file."""
    if split != "test":
        raise ValueError("hmu-crc only supports split='test' (no predefined train/val splits).")

    entry = REGISTRY[flag]
    npz_path = _ensure_hub_file(entry["hub_flag"], entry["npz"], entry["local_subdir"])
    data = np.load(str(npz_path), allow_pickle=True)

    images = data["images"]  # (N, 224, 224, 3) uint8
    labels = data["labels"]  # (N,) int64

    transform = _make_transform(color=True, image_size=image_size, normalize=True)

    class _HMUDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, lbls, tfm):
            self.imgs = imgs
            self.lbls = lbls
            self.tfm = tfm

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            from PIL import Image as PILImage
            pil = PILImage.fromarray(self.imgs[i], mode="RGB")
            return self.tfm(pil), torch.tensor(int(self.lbls[i]), dtype=torch.long)

    dataset = _HMUDataset(images, labels, transform)

    if corruption is not None:
        dataset = _apply_corruption(dataset, "pathmnist", corruption, severity)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    info = entry["info"]
    class_names = list(info["label"].values())
    return BenchmarkDataset(loader=loader, dataset=dataset, info=info,
                            flag=flag, split=split,
                            class_names=class_names, image_size=image_size)


# ---- Corruption wrapper ----

def _apply_corruption(dataset, medmnist_flag: str, corruption: str, severity: int):
    """Apply medmnistc corruption to a dataset, if available."""
    try:
        from medmnistc.corruptions.registry import CORRUPTIONS_DS
        from medmnistc.corruptions.corrupt import corrupt_image
    except ImportError:
        raise ImportError(
            "medmnistc is required for corruption support.\n"
            "Install it with:  pip install medmnistc"
        )
    return _CorruptedWrapper(dataset, medmnist_flag, corruption, severity)


class _CorruptedWrapper(torch.utils.data.Dataset):
    """Wraps a dataset, applying a named medmnistc corruption on-the-fly."""

    def __init__(self, base, flag: str, corruption: str, severity: int):
        self.base = base
        self.flag = flag
        self.corruption = corruption
        self.severity = severity

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        from medmnistc.corruptions.corrupt import corrupt_image
        img_t, label = self.base[i]
        # Convert [-0.5, 0.5] → [0, 255] for corruption → back
        img_np = ((img_t.permute(1, 2, 0).numpy() + 0.5) * 255).astype(np.uint8)
        corrupted = corrupt_image(img_np, self.corruption, self.severity)
        img_out = torch.from_numpy(corrupted).permute(2, 0, 1).float() / 255.0 - 0.5
        return img_out, label


def _add_utils_to_path():
    utils_dir = str(Path(__file__).resolve().parent)
    data_preprocessing_dir = str(Path(__file__).resolve().parent / "data_preprocessing_classification_evaluation")
    for p in [utils_dir, data_preprocessing_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(
    flag: str,
    split: str = "test",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    corruption: Optional[str] = None,
    severity: int = 1,
    download: bool = True,
) -> BenchmarkDataset:
    """
    Load a benchmark dataset by flag name.

    Parameters
    ----------
    flag        : Dataset identifier. See module docstring for full list.
                  Aliases are accepted (e.g. ``"amos2022"``, ``"derm-e-id"``).
    split       : One of ``"train"``, ``"val"``, ``"test"``.
                  External datasets (amos22, midog) only support ``"test"``.
    image_size  : Target image size in pixels (height = width).  Default: 224.
    batch_size  : DataLoader batch size.  Default: 32.
    num_workers : DataLoader worker processes.  Default: 4.
    corruption  : medmnistc corruption name, e.g. ``"gaussian_noise"``,
                  ``"jpeg_compression"``, ``"brightness"``.
                  Requires ``pip install medmnistc``.  Default: None (no corruption).
    severity    : Corruption severity level 1–5.  Only used when
                  ``corruption`` is not None.  Default: 1.
    download    : Automatically download missing files.  Default: True.

    Returns
    -------
    BenchmarkDataset
        A dataclass with ``.loader``, ``.dataset``, ``.info``,
        ``.flag``, ``.split``, ``.class_names``, ``.image_size``.

    Examples
    --------
    >>> ds = load("breastmnist")
    >>> for images, labels in ds.loader:
    ...     pass

    >>> ds = load("dermamnist-e-id")
    >>> print(ds.class_names)

    >>> ds = load("amos22")
    >>> print(len(ds), ds.info["task"])

    >>> ds = load("organamnist", split="train", batch_size=128)

    >>> ds = load("pathmnist", corruption="gaussian_noise", severity=3)
    """
    # Resolve aliases
    canonical = ALIASES.get(flag, flag)
    if canonical not in REGISTRY:
        raise ValueError(
            f"Unknown dataset flag: {flag!r}.\n"
            f"Available: {sorted(REGISTRY)} + aliases: {sorted(ALIASES)}"
        )

    entry = REGISTRY[canonical]

    if split not in entry["splits"]:
        raise ValueError(
            f"{canonical!r} does not have a {split!r} split. "
            f"Available splits: {entry['splits']}"
        )

    if not download and entry["source"] == "hub":
        npz_path = _data_root() / entry["local_subdir"] / entry["npz"]
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {npz_path}\n"
                f"Re-run with download=True or run:  python scripts/setup_from_hub.py"
            )

    source = entry["source"]

    if source == "medmnist":
        return _load_medmnist(canonical, split, image_size, batch_size,
                               num_workers, corruption, severity, download)

    hub_flag = entry.get("hub_flag")

    if hub_flag == "hmu-crc":
        return _load_hmu_crc(canonical, split, image_size, batch_size,
                              num_workers, corruption, severity, download)
    if hub_flag == "dermamnist-e":
        return _load_dermamnist_e(canonical, split, image_size, batch_size,
                                   num_workers, corruption, severity, download)
    if hub_flag == "amos22":
        return _load_amos22(canonical, split, image_size, batch_size,
                             num_workers, corruption, severity, download)
    if hub_flag == "midog":
        return _load_midog(canonical, split, image_size, batch_size,
                            num_workers, corruption, severity, download)

    raise RuntimeError(f"No loader implemented for hub_flag={hub_flag!r}")


def list_datasets() -> None:
    """Print all available dataset flags with a short description."""
    SOURCE_LABELS = {"medmnist": "medmnist pkg", "hub": "HF Hub", "local": "local only"}
    print(f"\n{'Flag':<25} {'Source':<14} {'Splits':<22} {'Color'} {'N classes'}")
    print("-" * 78)
    for flag, entry in REGISTRY.items():
        splits = "/".join(entry["splits"])
        color = "color" if entry["color"] else "gray"
        source = SOURCE_LABELS.get(entry["source"], entry["source"])
        n_cls = len(entry.get("info", {}).get("label", {}))
        n_cls_str = str(n_cls) if n_cls else "—"
        print(f"{flag:<25} {source:<14} {splits:<22} {color:<8} {n_cls_str}")
    if ALIASES:
        print(f"\nAliases: {', '.join(f'{k}→{v}' for k,v in ALIASES.items())}")
    print()
    if ALIASES:
        print(f"\nAliases: {', '.join(f'{k}→{v}' for k,v in ALIASES.items())}")
    print()
