# Benchmarks — medMNIST

This directory contains the full reproducible benchmarking pipeline for evaluating uncertainty quantification methods on MedMNIST datasets and external test sets.

---

## Table of contents

1. [Environment setup](#1-environment-setup)
2. [Data](#2-data)
   - [MedMNIST (automatic download)](#21-medmnist-automatic-download)
   - [AMOS-2022 (external CT test set)](#22-amos-2022-external-ct-test-set)
   - [MIDOG++ (external histology test set)](#23-midog-external-histology-test-set)
   - [DermaMNIST-E (extended dermoscopy)](#24-dermamnist-e-extended-dermoscopy)
3. [Training models](#3-training-models)
   - [ResNet-18](#31-resnet-18)
   - [ViT-B/16](#32-vit-b16)
   - [Moving trained models to the benchmark models/ directory](#33-moving-trained-models)
4. [Running the benchmark](#4-running-the-benchmark)
   - [Single configuration](#41-single-configuration)
   - [Full benchmark (all datasets and setups)](#42-full-benchmark)
   - [GPS pre-computation (TTA_calib)](#43-gps-pre-computation)
5. [Results and outputs](#5-results-and-outputs)
6. [Reproducibility notes](#6-reproducibility-notes)

---

## 1. Environment setup

All scripts assume the project venv. Install from the root:

```bash
# From repo root
pip install -r requirements.txt
pip install -e ToolBox/

# Optional: covariate-shift corruptions
pip install medmnistc
```

Run the pre-flight checklist before benchmarking:
```bash
python ToolBox/tests/pre_run_checklist.py
```

---

## 2. Data

### 2.1 MedMNIST (automatic download)

All MedMNIST datasets are downloaded automatically by the `medmnist` package when first accessed. No manual steps are required.

Supported datasets:
| Flag | Task | Classes |
|---|---|---|
| `breastmnist` | Binary classification | 2 |
| `pneumoniamnist` | Binary classification | 2 |
| `organamnist` | Multi-class | 11 |
| `octmnist` | Multi-class | 4 |
| `pathmnist` | Multi-class | 9 |
| `bloodmnist` | Multi-class | 8 |
| `tissuemnist` | Multi-class | 8 |
| `dermamnist-e` | Multi-class | 7 |

Images are resized to **224×224** for all models. The default download path is managed by the `medmnist` package (`~/.medmnist/` or the path set by `MEDMNIST_ROOT`).

---

### 2.2 AMOS-2022 (external CT test set)

AMOS-2022 is used as an external test set for OrganaMNIST models. The data directory is **gitignored** (`**/data/`) and must be produced locally.

**Step 1 — Download AMOS-2022**

Download the AMOS-2022 dataset from https://amos22.grand-challenge.org/. Place the CT volumes under:
```
Benchmarks/medMNIST/data/AMOS_2022/amos22/
```

**Step 2 — Preprocess into 224×224 patches**

Open and run the preprocessing notebook:
```
Benchmarks/medMNIST/data/AMOS_2022/read_npz.ipynb
```
This extracts 224×224 grayscale patches and saves them as:
```
Benchmarks/medMNIST/data/AMOS_2022/amos_external_test_224.npz
```

The loading code in `dataset_utils.py::load_amos_dataset()` reads this file. **6 organ classes are mapped** to OrganaMNIST labels (spleen, right/left kidney, liver, pancreas, bladder). Unmapped organs are used as unseen classes in `--new-class-shift` mode.

---

### 2.3 MIDOG++ (external histology test set)

MIDOG++ is used as an OOD test set for PathMNIST models in new-class-shift mode. The data directory is **gitignored** and preprocessing is required before benchmarking.

**Step 1 — Download MIDOG++ images**

Download the MIDOG++ dataset from https://midog2022.grand-challenge.org/ and place the TIFF images under:
```
Benchmarks/medMNIST/data/MIDOG++/images/
```

The annotation JSON (`midog_canine_patches.json`) describing patch coordinates is not included in the repository and must be produced or obtained together with the images.

**Step 2 — Generate 224×224 patches**

Use the preprocessing script:
```bash
python Benchmarks/medMNIST/utils/data_preprocessing_classification_evaluation/create_midog_patch_dataset.py \
    --images-dir Benchmarks/medMNIST/data/MIDOG++/images/ \
    --json-file Benchmarks/medMNIST/data/MIDOG++/midog_canine_patches.json \
    --output-dir Benchmarks/medMNIST/data/MIDOG++/patches_individual/ \
    --output-npz Benchmarks/medMNIST/data/MIDOG++/midog_canine_patches.npz
```

This resamples MIDOG++ images to match PathMNIST's physical resolution (0.5 µm/px) before extracting non-overlapping 224×224 patches, to avoid magnification confounds. The output `.npz` is loaded by `dataset_utils.py::load_midog_for_new_class_shift()`.

---

### 2.4 DermaMNIST-E (extended dermoscopy)

DermaMNIST-E is an enhanced version of DermaMNIST (Abhishek et al., 2025) with strict lesion-identity splits. The data directory is **gitignored** and requires a custom preprocessed file that is not available from the standard Zenodo release.

**Required file:**
```
Benchmarks/medMNIST/data/ISIC_2018/dermamnist_extended_224_wsitesources.npz
```

This file extends the standard DermaMNIST-E with a `test_centers` array that tags each test sample by clinical acquisition site. This metadata enables the ID/external split and is not present in the public Zenodo release.

The source images are available from the Harvard Dataverse:
https://dataverse.harvard.edu/file.xhtml?fileId=6924466&version=4.0&toolType=PREVIEW

The file is loaded by `utils/data_preprocessing_classification_evaluation/local_dermamnist_e.py`.

**Benchmark flags:**
- `dermamnist-e-id` — ID centres only (same distribution as training)
- `dermamnist-e-external` — External centre only (distribution shift)

---

## 3. Training models

Models are trained with 5-fold stratified cross-validation (`n_splits=5, seed=42`). Each fold produces one `.pt` checkpoint. **This seed must never change**, as inference and KNN methods depend on the exact same splits.

### 3.1 ResNet-18

**Single dataset, single setup:**
```bash
python Benchmarks/medMNIST/trainings/train_resnet18_medMNIST.py \
    --flag organamnist \
    --color False \
    --batch_size 128 \
    --num_epochs 100 \
    --use_randaugment False \
    --use_dropout False \
    --dropout_rate 0.3 \
    --cuda cuda:0
```

**Training setup flags:**
| `use_randaugment` | `use_dropout` | Benchmark setup name |
|---|---|---|
| False | False | standard (`""`) |
| True | False | `DA` (data augmentation) |
| False | True | `DO` (dropout) |
| True | True | `DADO` (both) |

**Launch all 4 setups for a dataset via the launcher:**

Edit `Benchmarks/medMNIST/trainings/launcher_resnet_training.py` to set the desired flags and run:
```bash
python Benchmarks/medMNIST/trainings/launcher_resnet_training.py
```

### 3.2 ViT-B/16

```bash
python Benchmarks/medMNIST/trainings/train_vit_medMNIST.py \
    --flag organamnist \
    --color False \
    --batch_size 128 \
    --num_epochs 100 \
    --use_randaugment False \
    --use_dropout False \
    --dropout_rate 0.1 \
    --learning_rate 0.0001 \
    --cuda cuda:0
```

Launcher:
```bash
python Benchmarks/medMNIST/trainings/launcher_vit_training.py
```

### 3.3 Moving trained models

After training, five checkpoints are produced per experiment under:
```
Benchmarks/medMNIST/runs/{flag}/{experiment_name}/resnet18_{flag}_224_{fold_idx}.pt
```

Move them to the standard models directory used by the benchmark:
```
Benchmarks/medMNIST/models/224*224/{flag}_{backbone}_{size}_randaug{0|1}_dropout{rate}_fold_{0..4}.pt
```

The notebook `Benchmarks/medMNIST/runs/rename_and_move_models.ipynb` automates this renaming step.

---

## 4. Running the benchmark

### 4.1 Single configuration

```bash
python Benchmarks/medMNIST/run_medmnist_benchmark.py \
    --flag breastmnist \
    --model resnet18 \
    --setup "" \
    --methods MSR MSR_calibrated MLS Ensembling TTA GPS KNN_Raw MCDropout \
    --batch-size 4000 \
    --gpu 0 \
    --per-fold-eval \
    --output-dir ./Benchmarks/medMNIST/results
```

**Available methods:**
`MSR`, `MSR_calibrated`, `MLS`, `Ensembling`, `TTA`, `TTA_calib`, `GPS`, `KNN_Raw`, `KNN_SHAP`, `MCDropout`, `ZScore_Aggregation_per_fold`, `ZScore_Aggregation_ensemble`

**MCDropout** only works with `--setup DO` or `--setup DADO` (models trained with dropout).

**GPS requires a pre-computation step** — see [section 4.3](#43-gps-pre-computation).

**External test sets:**
```bash
# AMOS-2022 (OrganaMNIST models, AMOS test data)
python Benchmarks/medMNIST/run_medmnist_benchmark.py --flag amos2022 --model resnet18 ...

# New-class shift (unseen organ classes)
python Benchmarks/medMNIST/run_medmnist_benchmark.py --flag amos2022 --new-class-shift ...

# MIDOG++ (PathMNIST models, MIDOG OOD test)
python Benchmarks/medMNIST/run_medmnist_benchmark.py --flag midog --new-class-shift ...

# Covariate shift corruptions
python Benchmarks/medMNIST/run_medmnist_benchmark.py \
    --flag organamnist --corruption-severity 3 --corrupt-test
```

List available corruptions for a dataset:
```bash
python Benchmarks/medMNIST/run_medmnist_benchmark.py --flag organamnist --list-corruptions
```

### 4.2 Full benchmark

The launcher iterates over all (dataset × backbone × training setup × method) combinations and dispatches them sequentially:

```bash
python Benchmarks/medMNIST/launcher_benchmark.py \
    --python /home/psteinmetz/venvs/venv_medMNIST/bin/python3.12 \
    --gpu 0 \
    --output-dir ./Benchmarks/medMNIST/results
```

Useful flags:
```bash
# Dry run — print commands without executing
python Benchmarks/medMNIST/launcher_benchmark.py --dry-run

# Only internal test sets
python Benchmarks/medMNIST/launcher_benchmark.py --id-only

# Only ResNet-18, standard and DA setups
python Benchmarks/medMNIST/launcher_benchmark.py --models resnet18 --setups "" DA

# New-class-shift evaluation (AMOS and MIDOG only)
python Benchmarks/medMNIST/launcher_benchmark.py --datasets amos2022 midog --new-class-shift

# Covariate-shift evaluation
python Benchmarks/medMNIST/launcher_benchmark.py --corruption-severity 3 --corrupt-test

# Exclude specific methods
python Benchmarks/medMNIST/launcher_benchmark.py --exclude-methods KNN_SHAP MCDropout
```

### 4.3 GPS pre-computation

GPS requires augmentation predictions on the calibration set to be cached before the greedy search can run. Use the `TTA_calib` method to generate this cache:

```bash
python Benchmarks/medMNIST/run_medmnist_benchmark.py \
    --flag organamnist \
    --model resnet18 \
    --setup "" \
    --methods TTA_calib \
    --batch-size 3500 \
    --gps-calib-samples 5000 \
    --gpu 0
```

Or use the dedicated launcher which runs `TTA_calib` for all backbone/setup combinations of a dataset:

```bash
python Benchmarks/medMNIST/utils/launch_tta_calib.py --flag organamnist --gpu 0
```

After the cache is generated (saved in `results/gps_augment_cache/`), run GPS normally:

```bash
python Benchmarks/medMNIST/run_medmnist_benchmark.py \
    --flag organamnist --model resnet18 --setup "" \
    --methods GPS --gpu 0
```

Or run `TTA_calib GPS` together in one command to generate and use the cache in the same run.

---

## 5. Results and outputs

All outputs are saved under `Benchmarks/medMNIST/results/` (configurable via `--output-dir`):

```
results/
├── {flag}_{backbone}_{setup}_{timestamp}.json   # Per-method metrics (AUROC, AURC, AUGRC)
├── cache/                                        # Cached model predictions (.npz)
│   └── {flag}_{backbone}[_{setup}]{...}_test_results.npz
├── gps_augment_cache/                           # GPS augmentation policy predictions
│   └── {flag}_{backbone}_{setup}_calibration_set/
│       ├── policy_0.npz ... policy_499.npz      # 500 random policy predictions
│       └── gps_policies_cache.pkl               # Cached greedy search results
└── figures/                                     # Evaluation plots
```

**Cache behaviour:** Results are cached per (dataset, backbone, setup, corruption config) to avoid redundant inference. Delete the corresponding `.npz` files in `cache/` to force re-evaluation after changing models.

**Visualisation:** Use the scripts in `utils/viz_benchmark_results/` to generate paper figures from the result JSONs.

---

## 6. Reproducibility notes

The benchmark is designed to be fully reproducible:

- **Random seeds:** All seeds are fixed to `42` (Python, NumPy, PyTorch, and CUDA RNG).
- **Deterministic mode:** `torch.backends.cudnn.deterministic = True` is set at startup.
- **CV splits:** 5-fold `StratifiedKFold(n_splits=5, random_state=42)` is used consistently for both training and KNN method fitting. **Never change these parameters** without retraining all models.
- **GPS subsampling:** Failure-aware subsampling uses `seed=42` for consistent calibration subsets. The subsample size is dataset-specific (see `DATASET_CONFIG` in `launcher_benchmark.py`).
- **Model checkpoints:** Models are loaded from `Benchmarks/medMNIST/models/224*224/` by `train_models_load_datasets.py::load_models()`. Ensure the correct backbone/setup suffix is used.
- **TTA seed:** Fixed TTA uses `seed=42` for the 5 random augmentations.

### Reproducing from scratch

```
1. pip install -r requirements.txt && pip install -e ToolBox/
2. [Optional] Preprocess MIDOG++ patches (see section 2.3)
3. Train all models:
   python Benchmarks/medMNIST/trainings/launcher_resnet_training.py
   python Benchmarks/medMNIST/trainings/launcher_vit_training.py
4. Move/rename model checkpoints using the notebook in runs/
5. Generate GPS caches:
   python Benchmarks/medMNIST/utils/launch_tta_calib.py --flag <dataset> --gpu 0
6. Run the full benchmark:
   python Benchmarks/medMNIST/launcher_benchmark.py --python <python_path> --gpu 0
```
