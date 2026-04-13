"""
Hugging Face Hub utilities for FailCatcher benchmark.

Provides download and upload helpers for:
  - Trained model checkpoints  (HF Model Hub)
  - Preprocessed custom datasets (HF Dataset Hub)

HuggingFace repositories
------------------------
  Models  : HF_MODELS_REPO  (default: "pstnmz/FailCatcher-models")
  Datasets: HF_DATASETS_REPO (default: "pstnmz/FailCatcher-datasets")

These can be overridden via environment variables:
  FAILCATCHER_HF_MODELS_REPO
  FAILCATCHER_HF_DATASETS_REPO

Usage
-----
  # Download one model file (called automatically by load_models when file missing)
  from utils.hub import ensure_model_file
  ensure_model_file("breastmnist_resnet18_224_randaug0_fold_0.pt", local_dir="models/224*224")

  # Download a custom dataset NPZ
  from utils.hub import ensure_dataset_file
  ensure_dataset_file("dermamnist-e", "dermamnist_extended_224_wsitesources.npz",
                      local_dir="data/ISIC_2018")

  # Upload everything to HF Hub (run once by repo authors)
  from utils.hub import upload_models, upload_datasets
  upload_models("Benchmarks/medMNIST/models/224*224", token="hf_...")
  upload_datasets({"dermamnist-e": "...", "amos22": "..."}, token="hf_...")
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository identifiers (override via env vars)
# ---------------------------------------------------------------------------
HF_MODELS_REPO: str = os.environ.get(
    "FAILCATCHER_HF_MODELS_REPO", "pstnmz/FailCatcher-models"
)
HF_DATASETS_REPO: str = os.environ.get(
    "FAILCATCHER_HF_DATASETS_REPO", "pstnmz/FailCatcher-datasets"
)

# Mapping: dataset flag → subfolder inside the HF datasets repo
DATASET_SUBFOLDERS: dict[str, str] = {
    "dermamnist-e": "dermamnist-e",
    "amos22": "amos22",
    "midog": "midog",
    "hmu-crc": "hmu-crc",
}

# All custom dataset files that live in the HF datasets repo
# (standard MedMNIST datasets are handled by the medmnist package itself)
CUSTOM_DATASET_FILES: dict[str, list[str]] = {
    "dermamnist-e": ["dermamnist_extended_224_wsitesources.npz"],
    "amos22": ["amos_external_test_224.npz"],
    "midog": ["midog_canine_patches.npz"],
    "hmu-crc": ["hmu_crc_224.npz"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hf_hub_download(repo_id: str, filename: str, repo_type: str,
                     local_dir: str | Path, token: str | None = None) -> Path:
    """Thin wrapper around huggingface_hub.hf_hub_download with useful errors."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download files from the Hub.\n"
            "Install it with:  pip install huggingface_hub"
        ) from exc

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    dest = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=str(local_dir),
        token=token,
    )
    return Path(dest)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def ensure_model_file(
    filename: str,
    local_dir: str | Path,
    hub_repo: str = HF_MODELS_REPO,
    token: str | None = None,
) -> Path:
    """
    Ensure *filename* exists in *local_dir*, downloading from Hub if necessary.

    Parameters
    ----------
    filename  : e.g. "breastmnist_resnet18_224_randaug0_fold_0.pt"
    local_dir : local directory that mirrors the flat Hub repo (models/224*224)
    hub_repo  : HF model repo id
    token     : HF access token (needed for private repos)

    Returns
    -------
    Path to the local file.
    """
    local_path = Path(local_dir) / filename
    if local_path.exists():
        return local_path

    token = token or os.environ.get("HF_TOKEN")
    print(f"[hub] Model not found locally, downloading from {hub_repo}/{filename} …")
    dest = _hf_hub_download(
        repo_id=hub_repo,
        filename=filename,
        repo_type="model",
        local_dir=local_dir,
        token=token,
    )
    print(f"[hub] Saved to {dest}")
    return dest


def download_all_models(
    local_dir: str | Path,
    hub_repo: str = HF_MODELS_REPO,
    token: str | None = None,
) -> None:
    """
    Download **all** model checkpoints from the Hub into *local_dir*.

    This is the bulk download used by ``setup_from_hub.py``.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required.  pip install huggingface_hub"
        ) from exc

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    token = token or os.environ.get("HF_TOKEN")
    print(f"[hub] Downloading all models from {hub_repo} into {local_dir} …")
    snapshot_download(
        repo_id=hub_repo,
        repo_type="model",
        local_dir=str(local_dir),
        token=token,
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )
    print(f"[hub] All models downloaded to {local_dir}")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def ensure_dataset_file(
    flag: str,
    filename: str,
    local_dir: str | Path,
    hub_repo: str = HF_DATASETS_REPO,
    token: str | None = None,
) -> Path:
    """
    Ensure *filename* (for dataset *flag*) exists in *local_dir*.

    Downloads from Hub if the file is missing.

    Parameters
    ----------
    flag      : dataset flag, e.g. 'dermamnist-e' or 'amos22'
    filename  : NPZ filename, e.g. 'dermamnist_extended_224_wsitesources.npz'
    local_dir : directory where the NPZ should live
    hub_repo  : HF dataset repo id
    token     : HF access token
    """
    local_path = Path(local_dir) / filename
    if local_path.exists():
        return local_path

    subfolder = DATASET_SUBFOLDERS.get(flag, flag)
    hub_path = f"{subfolder}/{filename}"

    token = token or os.environ.get("HF_TOKEN")
    print(f"[hub] Dataset file not found locally, downloading {hub_path} from {hub_repo} …")
    dest = _hf_hub_download(
        repo_id=hub_repo,
        filename=hub_path,
        repo_type="dataset",
        local_dir=local_dir,
        token=token,
    )
    # hf_hub_download places files in subdirs; move up if needed
    dest = Path(dest)
    final = Path(local_dir) / filename
    if dest != final and not final.exists():
        import shutil
        shutil.copy2(dest, final)
    print(f"[hub] Saved to {final}")
    return final


def download_all_datasets(
    local_root: str | Path,
    hub_repo: str = HF_DATASETS_REPO,
    token: str | None = None,
    flags: list[str] | None = None,
) -> None:
    """
    Download all custom dataset NPZ files into *local_root*.

    Each flag is placed in its expected subdirectory (e.g. ISIC_2018/, AMOS_2022/).

    Parameters
    ----------
    local_root : root of the data directory (Benchmarks/medMNIST/data)
    flags      : subset of flags to download; None means all
    """
    _LOCAL_DIRS: dict[str, str] = {
        "dermamnist-e": "ISIC_2018",
        "amos22": "AMOS_2022",
        "midog": "MIDOG++",
        "hmu-crc": "HMU-CRC-Hist550K",
    }

    flags = flags or list(CUSTOM_DATASET_FILES.keys())
    token = token or os.environ.get("HF_TOKEN")

    for flag in flags:
        files = CUSTOM_DATASET_FILES.get(flag)
        if not files:
            print(f"[hub] Unknown custom dataset flag: {flag}, skipping.")
            continue
        subdir = _LOCAL_DIRS.get(flag, flag)
        local_dir = Path(local_root) / subdir
        for fname in files:
            ensure_dataset_file(flag, fname, local_dir, hub_repo=hub_repo, token=token)


# ---------------------------------------------------------------------------
# Upload helpers (used by authors to publish to Hub)
# ---------------------------------------------------------------------------

def upload_models(
    models_dir: str | Path,
    hub_repo: str = HF_MODELS_REPO,
    token: str | None = None,
    pattern: str = "*.pt",
    commit_message: str = "Upload trained model checkpoints",
    verbose: bool = True,
    num_workers: int = 1,
) -> None:
    """
    Upload all ``*.pt`` files from *models_dir* to the HF model repo.

    Uses ``upload_large_folder`` which automatically splits the upload into
    multiple commits and can resume interrupted transfers.

    Parameters
    ----------
    models_dir  : local directory containing the .pt files (flat)
    hub_repo    : repo id on HF, e.g. "pstnmz/FailCatcher-models"
    token       : HF write token  (or set HF_TOKEN env var)
    pattern     : glob pattern for files to upload
    verbose     : print a progress report every 30 s (default: True)
    num_workers : parallel upload threads (default: 1, increase cautiously)
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub") from exc

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "A HuggingFace write token is required.  "
            "Pass token= or set the HF_TOKEN environment variable."
        )

    api = HfApi(token=token)
    models_dir = Path(models_dir)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=hub_repo, repo_type="model", exist_ok=True)

    pt_files = sorted(models_dir.glob(pattern))
    if not pt_files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {models_dir}")

    print(
        f"[hub] Uploading {len(pt_files)} files to {hub_repo}\n"
        f"      Using upload_large_folder  (workers={num_workers}, verbose={verbose})\n"
        f"      This will auto-chunk commits and can be safely interrupted/resumed."
    )
    api.upload_large_folder(
        repo_id=hub_repo,
        folder_path=str(models_dir),
        repo_type="model",
        allow_patterns=[pattern],
        num_workers=num_workers,
        print_report=verbose,
        print_report_every=30,
    )
    print(f"[hub] Upload complete → https://huggingface.co/{hub_repo}")


def upload_datasets(
    dataset_files: dict[str, str | Path],
    hub_repo: str = HF_DATASETS_REPO,
    token: str | None = None,
    commit_message: str = "Upload preprocessed dataset files",
) -> None:
    """
    Upload custom dataset NPZ files to the HF dataset repo.

    Parameters
    ----------
    dataset_files : mapping of {flag: local_path_to_npz_or_dir}
                    e.g. {"dermamnist-e": "/path/to/ISIC_2018",
                           "amos22": "/path/to/AMOS_2022/amos_external_test_224.npz"}
    hub_repo      : HF dataset repo id
    token         : HF write token

    Examples
    --------
    upload_datasets({
        "dermamnist-e": "Benchmarks/medMNIST/data/ISIC_2018",
        "amos22":       "Benchmarks/medMNIST/data/AMOS_2022/amos_external_test_224.npz",
    }, token="hf_...")
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub") from exc

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "A HuggingFace write token is required.  "
            "Pass token= or set the HF_TOKEN environment variable."
        )

    api = HfApi(token=token)
    api.create_repo(repo_id=hub_repo, repo_type="dataset", exist_ok=True)

    for flag, local_path in dataset_files.items():
        local_path = Path(local_path)
        subfolder = DATASET_SUBFOLDERS.get(flag, flag)
        expected_files = CUSTOM_DATASET_FILES.get(flag, [])

        if local_path.is_dir():
            # Upload known NPZ files from the directory
            upload_paths = [local_path / f for f in expected_files if (local_path / f).exists()]
        elif local_path.is_file():
            upload_paths = [local_path]
        else:
            print(f"[hub] Skipping {flag}: path not found ({local_path})")
            continue

        for fpath in upload_paths:
            path_in_repo = f"{subfolder}/{fpath.name}"
            print(f"[hub] Uploading {fpath.name} → {hub_repo}/{path_in_repo} …")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=path_in_repo,
                repo_id=hub_repo,
                repo_type="dataset",
                commit_message=f"{commit_message} ({flag})",
            )

    print(f"[hub] Dataset upload complete → https://huggingface.co/datasets/{hub_repo}")