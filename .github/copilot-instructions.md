ex# UQ Toolbox - AI Agent Instructions

## Project Overview
**UQ_Toolbox** is a post-hoc uncertainty quantification toolkit for PyTorch deep learning models in medical imaging. The project benchmarks multiple UQ methods (GPS, TTA, Ensemble, KNN, MSR calibration) on medMNIST datasets and external test sets like AMOS-2022.

**Architecture**: Two-tier structure
- `FailCatcher/` - Core UQ library with class-based and functional APIs
- `medMNIST/` - Research benchmarking scripts for medical imaging
- `gps_augment/` - Greedy Policy Search augmentation framework (external submodule)

## Critical Patterns & Conventions

### 1. Cross-Validation Splits (CRITICAL)
**The #1 source of bugs**: CV splits MUST match between training and inference.

```python
# Training: train_resnet18_medMNIST.py
fold_gen = tr.CV_fold_generator(..., n_splits=5, seed=42, ...)

# Inference: launch_uq_methods.py
train_loaders = get_cv_train_loaders_for_models(
    study_dataset, models, 
    n_splits=5, seed=42  # ← Currently matches training
)
```

**Why**: KNN methods (`KNNLatentMethod`, `KNNLatentSHAPMethod`) fit on each fold's training data. If splits don't match, models see wrong training data → garbage predictions.

**Current state**: Seeds ARE synchronized (`n_splits=5, seed=42` everywhere). Both training and inference use `StratifiedKFold` from `train_load_datasets_resnet.py` with identical parameters.

**Rules** (to maintain this):
- Never modify `n_splits` or `seed` without retraining ALL models
- Both `CV_fold_generator()` (training) and `get_cv_train_loaders_for_models()` (inference) must use same params
- If you add new splits logic, verify against `train_resnet18_medMNIST.py` line 201

### 2. Class-Based UQ API (Preferred)
All UQ methods inherit from `FailCatcher/core/base.py::UQMethod`:

```python
# Preferred pattern
method = uq.GPSMethod(aug_folder, correct_idx, incorrect_idx)
method.search_policies(num_workers=90, top_k=3)
scores = method.compute(models, test_dataset, device)

# Also supported (legacy)
scores = uq.TTA(policies, models, dataset, device)
```

**Key methods**:
- `TTAMethod` / `GPSMethod` - Test-time augmentation
- `EnsembleSTDMethod` - Ensemble uncertainty
- `KNNLatentMethod` / `KNNLatentSHAPMethod` - Latent space methods
- `DistanceToHardLabelsMethod` / `CalibrationMethod` - Softmax-based

### 3. GPU Memory Management
**Problem**: Batched inference can OOM on large datasets (see terminal error: `CUDA out of memory. Tried to allocate 26.75 GiB`).

**Solutions**:
```python
# 1. Reduce batch_size for augmentation inference
batch_size = 4000  # Default, reduce to 2000 or 1000 if OOM

# 2. Use sequential augmentation (slower but safe)
# In FailCatcher/methods/tta.py::_batched_augmentation_inference
if K * N > 10000:  # Heuristic threshold
    # Fall back to sequential processing
```

**Caching**: Evaluation results are cached in `uq_benchmark_results/cache/` to avoid re-inference. Delete cache files to force re-evaluation.

### 4. Data Flow Architecture
```
medMNIST/train_resnet18_medMNIST.py
  ↓ trains 5 models (CV folds)
  ↓ saves to medMNIST/models/{flag}/fold_{0-4}.pth
  ↓
launch_uq_methods.py
  ↓ loads models via medMNIST/utils/train_load_datasets_resnet.py::load_models()
  ↓ evaluates with FailCatcher.UQ_toolbox methods
  ↓ saves results to uq_benchmark_results/{flag}_{timestamp}.json
```

### 5. GPS Workflow (Multi-Stage)
GPS requires pre-computed augmentation predictions (NOT generated on-the-fly):

```bash
# Stage 1: Generate augmentation pool (gps_augment/get_predictions_randaugment.py)
# Creates: medMNIST/gps_augment/224*224/{flag}_calibration_set/*.npz

# Stage 2: GPS search (launch_uq_methods.py --methods GPS)
gps = uq.GPSMethod(aug_folder, correct_idx, incorrect_idx)
gps.search_policies()  # Greedy search on calibration set
gps.compute()          # Apply best policies to test set
```

**File structure**:
```
medMNIST/gps_augment/224*224/
  breastmnist_calibration_set/
    policy_0.npz  # shape: (N_calib, num_classes)
    policy_1.npz
    ...
    policy_499.npz
```

### 6. Dataset-Specific Quirks

**OrganaMNIST → AMOS-2022 mapping**:
```python
# launch_uq_methods.py, flag='amos22'
amos_to_organamnist = {
    0: 10,  # spleen → spleen
    1: 5,   # right kidney → kidney-right
    # ... 6 total organs mapped
}
# Models trained on OrganaMNIST, tested on AMOS external test set
```

**Color vs Grayscale**:
```python
color = flag in ['dermamnist', 'dermamnist-e', 'pathmnist', 'bloodmnist']
# Non-color datasets use RepeatGrayToRGB() transform
```

**Calibration method selection**:
```python
calib_method = 'platt' if flag in ['breastmnist', 'pneumoniamnist'] else 'temperature'
# Binary: Platt scaling, Multi-class: Temperature scaling
```

### 7. Parallel Processing (KNN SHAP)
`KNNLatentSHAPMethod` supports multi-GPU parallelism:

```python
n_gpus = torch.cuda.device_count()
if n_gpus >= 3:
    parallel = True
    n_jobs = 3  # One GPU per worker
else:
    parallel = False
    n_jobs = 1

knn_shap.fit(..., parallel=parallel, n_jobs=n_jobs)
```

**Implementation**: `FailCatcher/methods/latent.py::_fit_fold_worker_multigpu()` uses `ProcessPoolExecutor` with explicit GPU assignment via `CUDA_VISIBLE_DEVICES`.

### 8. Common Pitfalls

❌ **Don't**:
- Change CV parameters without retraining
- Assume GPS augmentations are generated automatically
- Mix calibration/test sets between methods
- Ignore cached results (delete cache if model changed)

✅ **Do**:
- Check `uq_benchmark_results/cache/` for existing evaluations
- Use `--methods` flag to run specific UQ methods
- Monitor GPU memory with `torch.cuda.max_memory_allocated()`
- Call `torch.cuda.empty_cache()` between methods if OOM

## Key Files & Entry Points

**Main scripts**:
- `launch_uq_methods.py` - Benchmark all UQ methods (primary entry point)
- `medMNIST/train_resnet18_medMNIST.py` - Train CV models
- `save_gps_policies.py` - Generate GPS augmentation pool

**Core library** (`FailCatcher/`):
- `UQ_toolbox.py` - Public API (imports all methods)
- `methods/tta.py` - TTA/GPS implementation
- `methods/latent.py` - KNN/SHAP methods
- `methods/distance.py` - MSR/calibration
- `core/base.py` - Base classes

**Utilities**:
- `medMNIST/utils/train_load_datasets_resnet.py` - Data loading, CV splits, training loop
- `FailCatcher/search/greedy.py` - GPS greedy search algorithm

## Running the Benchmark

```bash
# Full benchmark (all methods)
python launch_uq_methods.py --flag breastmnist

# Specific methods
python launch_uq_methods.py --flag organamnist --methods MSR Ensembling GPS

# External test set
python launch_uq_methods.py --flag amos22 --methods KNN_SHAP

# Control batch size (if OOM)
python launch_uq_methods.py --flag dermamnist --batch-size 2000
```

## Environment Setup
- Python 3.8+
- PyTorch 1.10+ with CUDA
- Key deps: `medmnist`, `shap`, `monai`, `sklearn`
- Virtual env: `venv_medMNIST` (hardcoded in terminal commands)

## Testing & Validation
- No formal test suite (research code)
- Validation: Compare AUC scores in `uq_benchmark_results/*.json`
- Sanity check: Test accuracy should match training logs (~0.60-0.85 depending on dataset)
