# ToolBox — FailCatcher UQ Library

**FailCatcher ToolBox** is a post-hoc uncertainty quantification (UQ) library for PyTorch deep learning models. It provides a unified API to quantify prediction uncertainty and detect model failures without retraining.

---

## Installation

From the repository root:
```bash
pip install -e ToolBox/
```

Or directly from the `ToolBox/` directory:
```bash
pip install -e .
```

---

## Usage

### High-level API — `FailureDetector`

The `FailureDetector` class is the recommended entry point. It wraps all UQ methods and handles caching, calibration, and evaluation.

```python
from ToolBox import failure_detection

detector = failure_detection.FailureDetector(
    models=models,             # list of PyTorch models (one per CV fold)
    study_dataset=train_set,   # training set (used by KNN methods)
    calib_dataset=calib_set,   # calibration set (used for calibration methods)
    test_dataset=test_set,
    device=device,
    num_classes=num_classes
)

# Pre-set predictions to avoid redundant inference across methods
detector.set_test_predictions(y_scores, y_true, y_pred)

# Run individual methods
_, metrics = detector.run_msr(y_scores, y_true)
_, metrics = detector.run_ensemble(indiv_scores, y_true)
_, metrics = detector.run_tta(test_dataset_tta, y_true, image_size=224, batch_size=256)
_, metrics = detector.run_gps(test_dataset_tta, y_true, aug_folder=aug_folder, ...)
_, metrics = detector.run_knn_raw(test_loader, train_loaders, y_true, layer_name='avgpool')
_, metrics = detector.run_mcdropout(test_dataset, y_true)

# Save all results to JSON and figures
detector.save_results(output_dir='./results', flag='organamnist', timestamp='...')
```

Each `run_*` method returns `(uncertainties_dict, metrics_dict)` where `metrics_dict` contains AUROC_f, AURC, and AUGRC scores.

### Low-level API — individual method classes

```python
import ToolBox.UQ_toolbox as uq

# Test-Time Augmentation
tta = uq.TTAMethod(transformations=None, n=2, m=45)
scores = tta.compute(model, test_dataset, device, nb_augmentations=5, batch_size=256, image_size=224)

# Greedy Policy Search (requires pre-computed augmentation cache)
gps = uq.GPSMethod(aug_folder, correct_idx_calib, incorrect_idx_calib)
gps.search_policies(num_workers=4, top_k=3)
scores = gps.compute(model, test_dataset, device)

# Ensemble standard deviation
ensemble = uq.EnsembleSTDMethod()
scores = ensemble.compute(models, test_loader, device)

# Maximum Softmax Response
msr = uq.DistanceToHardLabelsMethod()
scores = msr.compute(models, test_loader, device)

# KNN in latent space
knn = uq.KNNLatentMethod(layer_name='avgpool', k=5)
knn.fit(models, train_loaders, device)
scores = knn.compute(models, test_loader, device)
```

---

## Modules

### `failure_detection.py`
High-level `FailureDetector` class. Orchestrates all UQ methods, manages prediction caching, calibration statistics, z-score aggregation, and result saving.

### `UQ_toolbox.py`
Public API aggregator. Imports and re-exports all classes and functions for backward compatibility.

### `core/`
- `base.py` — Abstract base classes: `UQMethod` (defines the `.compute()` interface) and `UQResult`.
- `utils.py` — Shared inference utilities: ensemble forward pass, batch predictions, MONAI cache building.

### `methods/`
- `tta.py` — `TTAMethod` (random augmentation std), `GPSMethod` (optimised policy set).
- `ensemble.py` — `EnsembleSTDMethod` (std across CV folds), `MCDropoutMethod` (Monte Carlo Dropout).
- `distance.py` — `DistanceToHardLabelsMethod` (MSR / MLS), `CalibrationMethod` (temperature / Platt scaling), `TemperatureScaler`.
- `latent.py` — `KNNLatentMethod` (KNN in avgpool feature space), `KNNLatentSHAPMethod` (SHAP-weighted KNN), `HyperplaneDistanceMethod` (SVM margin).

### `search/`
- `greedy.py` — `perform_greedy_policy_search()`: greedy selection of augmentation policies from a pre-computed pool that maximises failure detection AUROC on the calibration set.

### `evaluation/`
- `evaluation.py` — `compute_auroc()`, `compute_aurc()`, `compute_augrc()` (Traub et al., NeurIPS 2024), ROC and risk-coverage curve plotting.

### `visualization/`
- `plots.py` — ROC curves, uncertainty distribution plots, method comparison plots.
- `shap_viz.py` — SHAP feature importance visualisation for KNN-SHAP.

### `tests/`
- `pre_run_checklist.py` — Checks dependencies, CUDA, imports, and disk space before the benchmark.
- `test_basic_functionality.py` — Smoke tests with synthetic data for TTA, Ensemble, Distance, and Visualization.
- `test_imports.py` — Validates both legacy and modular import styles.

---

## Evaluation metrics

| Metric | Description |
|---|---|
| **AUROC_f** | Area Under the ROC Curve for failure detection (correct vs. incorrect predictions) |
| **AURC** | Area Under the Risk-Coverage Curve |
| **AUGRC** | Area Under the Generalised Risk-Coverage Curve (NeurIPS 2024) |

---

## Design notes

- All UQ methods are **post-hoc**: no retraining is required.
- The library works with any PyTorch model that outputs class logits or probabilities.
- MSR and MSR-calibrated scores are computed from **averaged softmax/sigmoid probabilities** across folds (not averaged logits), which is the standard practice for probability-based uncertainty.
- GPS augmentation predictions must be **pre-computed** on the calibration set (see `Benchmarks/README.md` for the `TTA_calib` step) before the greedy search can run.
- KNN methods fit on each fold's **own training data** via the same `StratifiedKFold(n_splits=5, seed=42)` used during training to avoid data leakage.