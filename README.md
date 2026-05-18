# FailCatcher

**FailCatcher** is a uncertainty quantification (UQ) toolkit for PyTorch classification deep learning models, developed and benchmarked on medical imaging datasets from the [MedMNIST](https://medmnist.com/) collection and external test sets. Failure detection benchmark results can be found in the [benchmarks README](benchmarks/README.md).

> **Preprint:** [Steinmetz et al., medRxiv 2026](https://www.medrxiv.org/content/10.64898/2026.05.04.26350496v1) — DOI: 10.64898/2026.05.04.26350496

```bash
pip install FailCatcher
```

The project provides:
- A reusable Python library (`ToolBox/`) implementing uncertainty quantification methods for pytorch classification models.
- A full benchmarking pipeline (`Benchmarks/`), including model training, classification evaluation and failure detection evaluation on a diverse set of distribution shifts images.

---

## Repository structure

```
FailCatcher/
├── ToolBox/                        # UQ library (installable Python package)
│   ├── failure_detection.py        # High-level FailureDetector API
│   ├── UQ_toolbox.py               # Public API aggregator
│   ├── core/                       # Base classes and shared inference utilities
│   ├── methods/                    # UQ method implementations
│   │   ├── tta.py                  # Test-Time Augmentation (TTA) and GPS
│   │   ├── ensemble.py             # Ensemble STD and MC Dropout
│   │   ├── distance.py             # MSR, MLS, and calibration methods
│   │   └── latent.py               # KNN and SHAP latent-space methods
│   ├── search/                     # Greedy Policy Search (GPS) algorithm
│   ├── evaluation/                 # AUROC, AURC, AUGRC metrics and plots
│   ├── visualization/              # Visualization utilities
│   └── tests/                      # Smoke tests and pre-run checks
│
├── Benchmarks/
│   └── medMNIST/
│       ├── launcher_benchmark.py   # Top-level benchmark launcher
│       ├── run_medmnist_benchmark.py # Core benchmark runner (single config)
│       ├── trainings/              # Model training scripts and launchers
│       ├── utils/                  # Data loading, preprocessing, visualization
│       │   ├── train_models_load_datasets.py  # Central data/model utilities
│       │   ├── dataset_utils.py               # External datasets, corruptions
│       │   └── data_preprocessing_classification_evaluation/
│       ├── data/                   # External test datasets (AMOS-2022, MIDOG++, ISIC)
│       ├── models/                 # Trained model checkpoints
│       ├── runs/                   # Training logs and per-run artifacts
│       └── results/                # Benchmark outputs (JSON, figures, cache)
│
└── requirements.txt
```

---

## Quick start

### 1. Install

```bash
pip install FailCatcher
```

For development (editable install from source):
```bash
git clone https://github.com/pstnmz/FailCatcher && pip install -e FailCatcher/
```

### 2. Quick-start tutorial

See [`tutorial.ipynb`](tutorial.ipynb) for a self-contained end-to-end example on CIFAR-10:  
download a model from HuggingFace → inference → MSR uncertainty → AUROC-f / AURC / AUGRC → plots.

### 3. Download models and datasets from HuggingFace

Pre-trained model checkpoints and pre-processed external datasets are available on HuggingFace. Run the one-command setup to skip training and manual preprocessing:

```bash
python scripts/setup_from_hub.py
```

- **Models** (~59 GB, 325 checkpoints): [pstnmz/FailCatcher-models](https://huggingface.co/pstnmz/FailCatcher-models)
- **Datasets** (AMOS-2022, MIDOG++, DermaMNIST-E): [pstnmz/FailCatcher-datasets](https://huggingface.co/datasets/pstnmz/FailCatcher-datasets)

You can download models or datasets independently:
```bash
python scripts/setup_from_hub.py --models-only
python scripts/setup_from_hub.py --datasets-only --datasets amos22 midog dermamnist-e
```

### 4. Train models (alternative to step 3)

See [`Benchmarks/README.md`](Benchmarks/README.md) for the full reproducible training and benchmarking pipeline.

### 5. Run the benchmark

```bash
python Benchmarks/medMNIST/launcher_benchmark.py \
    --python /path/to/your/venv/bin/python \
    --datasets breastmnist organamnist \
    --models resnet18 \
    --setups "" DA \
    --gpu 0
```

---

## UQ methods

| Method | Description |
|---|---|
| **MSR** | Maximum Softmax Response — distance between predicted probability and 1 |
| **MSR-calibrated** | MSR after temperature / Platt scaling calibration |
| **MLS** | Maximum Logit Score — pre-softmax equivalent of MSR |
| **Ensembling** | Standard deviation across 5-fold CV model predictions |
| **TTA** | Test-Time Augmentation — std over random augmentation passes |
| **GPS** | Greedy Policy Search — optimised TTA policy found on the calibration set |
| **KNN-Raw** | k-NN distance in avgpool latent space |
| **KNN-SHAP** | KNN with SHAP-weighted latent features |
| **MC Dropout** | Monte Carlo Dropout at inference time |
| **ZScore Aggregation** | Z-score normalised aggregation of multiple methods |

---

## Datasets

### Internal test sets (MedMNIST)
`breastmnist`, `pneumoniamnist`, `organamnist`, `octmnist`, `pathmnist`, `bloodmnist`, `tissuemnist`, `dermamnist-e`

### External test sets (not in git — see `Benchmarks/README.md` for setup)
- **AMOS-2022** — abdominal CT organ patches mapped to OrganaMNIST classes. Available on [HuggingFace](https://huggingface.co/datasets/pstnmz/FailCatcher-datasets) or preprocessed via `data/AMOS_2022/read_npz.ipynb`.
- **MIDOG++** — mitosis detection histology patches as OOD test for PathMNIST. Available on [HuggingFace](https://huggingface.co/datasets/pstnmz/FailCatcher-datasets) or generated by `utils/data_preprocessing_classification_evaluation/create_midog_patch_dataset.py`.
- **DermaMNIST-E** — extended DermaMNIST with ID and external centre splits. Available on [HuggingFace](https://huggingface.co/datasets/pstnmz/FailCatcher-datasets) or downloaded from [Zenodo](https://zenodo.org/records/12739457), loaded by `utils/data_preprocessing_classification_evaluation/local_dermamnist_e.py`.

---

## Reproducibility

All benchmark results are reproducible from scratch:
- Random seeds fixed to `42` everywhere (training, CV splits, TTA).
- 5-fold StratifiedKFold CV with `seed=42` is consistent between training and inference.
- Model checkpoints, result JSONs, and caches are saved with configuration-specific suffixes.
- See [`Benchmarks/README.md`](Benchmarks/README.md) for step-by-step instructions.

---

## Python version and environment

Tested with **Python 3.12** and the following key packages:

| Package | Version |
|---|---|
| torch | 2.6.0 |
| torchvision | 0.21.0 |
| numpy | 2.1.3 |
| scikit-learn | 1.6.1 |
| monai | 1.5.1 |
| medmnist | 3.0.1 |
| shap | 0.46.0 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |

## License

This project is licensed under CC BY-NC-SA 4.0.

The code is intended for research and academic use only.
Commercial use is prohibited.

For commercial use, please contact the author.
