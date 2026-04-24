"""
MedMNIST Benchmark Runner using FailCatcher library.

This script demonstrates how to use the generic FailCatcher.benchmark API
for dataset-specific benchmarking.

Usage:
    python run_medmnist_benchmark.py --flag breastmnist --methods MSR Ensembling
    python run_medmnist_benchmark.py --flag organamnist --all-methods
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset

# Import FailCatcher library
import ToolBox
from ToolBox import failure_detection
from ToolBox import UQ_toolbox as uq

# Import medMNIST-specific utilities
from Benchmarks.medMNIST.utils import train_models_load_datasets as tr
from Benchmarks.medMNIST.utils.data_preprocessing_classification_evaluation import dataset_utils


def run_medmnist_benchmark(flag, methods, output_dir='./uq_benchmark_results',
                           batch_size=4000, image_size=224, gpu_id=0, per_fold_eval=True,
                           model_backbone='resnet18', setup='', gps_calib_samples=None,
                           min_failure_ratio=0.3, corruption_severity=0,
                           corrupt_test=False, corrupt_calib=False, new_class_shift=False,
                           concurrent_processes=1, max_loader_workers=16,
                           run_mode='both'):
    """
    Run UQ benchmark on a medMNIST dataset using FailCatcher library.
    
    Args:
        flag: Dataset name (e.g., 'breastmnist', 'organamnist')
        methods: List of method names to run
        output_dir: Output directory for results
        batch_size: Batch size for inference
        image_size: Image size
        gpu_id: GPU device ID to use
        gps_calib_samples: Max samples for GPS calibration (default: None = use all)
        min_failure_ratio: Minimum target proportion of failures (default: 0.3 = 30%)
        per_fold_eval: If True, compute per-fold metrics (mean±std). If False, use ensemble-based evaluation
        model_backbone: Model architecture ('resnet18' or 'vit_b_16')
        setup: Training setup - '' (standard), 'DA', 'DO', or 'DADO'
        corruption_severity: Corruption severity (0=disabled, 1-5=mild to severe covariate shift)
                            When enabled, randomly applies available medmnistc corruptions
        corrupt_test: If True, apply corruption to test set (requires corruption_severity > 0)
        corrupt_calib: If True, apply corruption to calibration set (requires corruption_severity > 0)
        new_class_shift: If True, create artificial test set with new classes (failures) + unanimous correct predictions
                        Only supported for AMOS2022 dataset
        concurrent_processes: Number of benchmark processes running simultaneously on this host
        max_loader_workers: Per-process hard cap for DataLoader workers
        run_mode: Execution mode. One of 'both' (default), 'calib_only', or 'test_only'.
                  'calib_only': skip test-set evaluation, only run calib_detector to collect
                      mean/std stats for Z-score normalization; saves JSON and returns early.
                  'test_only': skip calib_detector runs, only evaluate on the test set.
                  'both': run both calib and test (normal full benchmark).
    """
    print(f"\n{'='*80}")
    print(f"MedMNIST Benchmark: {flag}")
    print(f"Using FailCatcher v{ToolBox.__version__}")
    if new_class_shift:
        print(f"New Class Shift: Evaluating unseen classes (artificial test set)")
        print(f" Test = New/OOD classes (failures) + ALL known-class samples (dynamic success/failure per method)")
    if corruption_severity > 0:
        print(f"Covariate Shift: Random corruptions (severity={corruption_severity}/5)")
        print(f" Test set: {'Corrupted' if corrupt_test else 'Clean'}")
        print(f" Calibration set: {'Corrupted' if corrupt_calib else 'Clean'}")
    print(f"{'='*80}\n")
    
    # Set seeds for reproducibility
    import random
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    # Enable deterministic algorithms for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Deterministic mode enabled (seed=42)\n")
    
    # Get absolute path to workspace root (UQ_Toolbox/)
    workspace_root = Path(__file__).parent.parent.parent.absolute()
    
    # Make output_dir absolute if it's relative
    if not Path(output_dir).is_absolute():
        output_dir = str(workspace_root / output_dir)
    
    # Setup
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Process-aware CPU throttling to avoid oversubscription when multiple benchmarks run in parallel.
    cpu_total = os.cpu_count() or 1
    concurrent_processes = max(1, int(concurrent_processes))
    cpu_budget = max(1, cpu_total // concurrent_processes)
    max_loader_workers = max(1, int(max_loader_workers))
    shared_loader_workers = min(max_loader_workers, max(4, (cpu_budget * 3) // 4))
    loader_pin_memory = device.type == 'cuda'

    # Limit intra-op/inter-op CPU threading per process.
    torch_threads = max(1, min(16, cpu_budget // 2))
    try:
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # set_num_interop_threads can fail if called after parallel work started.
        pass

    print(
        f"CPU budget: total={cpu_total}, concurrent_processes={concurrent_processes}, "
        f"per_process_budget≈{cpu_budget}"
    )
    print(
        f"  Worker config: shared_loader_workers={shared_loader_workers}, "
        f"max_loader_workers={max_loader_workers}, torch_threads={torch_threads}"
    )
    
    # Parse dermamnist-e variants
    base_flag = flag
    test_subset = 'all'  # Default for all datasets
    if flag == 'dermamnist-e-id':
        base_flag = 'dermamnist-e'
        test_subset = 'id'
    elif flag == 'dermamnist-e-external':
        base_flag = 'dermamnist-e'
        test_subset = 'external'
    
    # AMOS uses organamnist models and calibration → use organamnist GPS cache
    # MIDOG uses pathmnist models and calibration → use pathmnist GPS cache
    if flag in ['amos2022', 'amos_external', 'amos22']:
        gps_cache_flag = 'organamnist'
    elif flag == 'midog':
        gps_cache_flag = 'pathmnist'
    else:
        gps_cache_flag = base_flag
    
    color = base_flag in ['dermamnist', 'dermamnist-e', 'pathmnist', 'bloodmnist']
    calib_method = 'platt' if base_flag in ['breastmnist', 'pneumoniamnist'] else 'temperature'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # LOAD DATA AND MODELS (medMNIST-specific)
    # ========================================================================
    print("Loading medMNIST data and models...")
    
    # Transforms
    transform, transform_tta = dataset_utils.get_transforms(color, image_size)
    
    if base_flag not in ['amos2022', 'midog']:
        # Load datasets and models
        models = tr.load_models(base_flag, device=device, size=image_size, 
                               model_backbone=model_backbone, setup=setup)
        [study_dataset, calib_dataset, test_dataset], \
        [_, calib_loader, test_loader], info = \
            tr.load_datasets(base_flag, color, image_size, transform, batch_size, test_subset=test_subset)
        
        [_, calib_dataset_tta, test_dataset_tta], \
        [_, _, _], _ = \
            tr.load_datasets(base_flag, color, image_size, transform_tta, batch_size, test_subset=test_subset)
        
        # Apply corruptions if requested
        if corruption_severity > 0 and (corrupt_test or corrupt_calib):
            print(f"\nApplying covariate shift corruptions...")
            # Map dataset name for corruption (dermamnist-e variants use 'dermamnist')
            if 'dermamnist' in base_flag:
                corruption_flag = 'dermamnist'
            else:
                corruption_flag = base_flag
            
            if corrupt_test:
                print(f" → Corrupting test set (severity={corruption_severity}/5)")
                test_dataset = dataset_utils.apply_random_corruptions(
                    test_dataset, corruption_flag, corruption_severity, cache=True, seed=42
                )
                # For TTA: use return_pil=True to get uint8 tensors for proper augmentation
                test_dataset_tta = dataset_utils.apply_random_corruptions(
                    test_dataset_tta, corruption_flag, corruption_severity, cache=True, seed=42, return_pil=True
                )
                # Rebuild test loader with corrupted dataset
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=shared_loader_workers, pin_memory=loader_pin_memory
                )
            if corrupt_calib:
                print(f" → Corrupting calibration set (severity={corruption_severity}/5)")
                calib_dataset = dataset_utils.apply_random_corruptions(
                    calib_dataset, corruption_flag, corruption_severity, cache=True, seed=42
                )
                # For TTA: use return_pil=True to get uint8 tensors for proper augmentation
                calib_dataset_tta = dataset_utils.apply_random_corruptions(
                    calib_dataset_tta, corruption_flag, corruption_severity, cache=True, seed=42, return_pil=True
                )
                # Rebuild calib loader with corrupted dataset
                calib_loader = DataLoader(
                    calib_dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=shared_loader_workers, pin_memory=loader_pin_memory
                )
    elif base_flag == 'amos2022':
        # Load datasets and models of organamnist and amos2022 as test set
        models = tr.load_models('organamnist', device=device, size=image_size,
                               model_backbone=model_backbone, setup=setup)
        [study_dataset, calib_dataset, _], \
        [_, calib_loader, _], info = \
            tr.load_datasets('organamnist', color, image_size, transform, batch_size)
        
        # Load calibration dataset with TTA transform (for GPS augmentation caching)
        [_, calib_dataset_tta, _], \
        [_, _, _], _ = \
            tr.load_datasets('organamnist', color, image_size, transform_tta, batch_size)
        
        # Load AMOS external test dataset
        if new_class_shift:
            # Load full AMOS dataset including unmapped classes
            test_dataset, test_loader, test_dataset_tta = dataset_utils.load_amos_for_new_class_shift(
                transform, transform_tta, models, device, batch_size,
                workspace_root=Path(__file__).resolve().parent.parent.parent
            )
        else:
            # Load standard AMOS dataset (only mapped classes)
            test_dataset, test_loader, test_dataset_tta, _, _ = dataset_utils.load_amos_dataset(
                transform, transform_tta, batch_size, workspace_root=Path(__file__).resolve().parent.parent.parent
            )
        
        # Apply corruptions if requested
        if corruption_severity > 0 and corrupt_test:
            print(f"\nApplying covariate shift corruptions...")
            print(f" → Corrupting test set (severity={corruption_severity}/5)")
            test_dataset = dataset_utils.apply_random_corruptions(
                test_dataset, flag, corruption_severity, cache=True, seed=42
            )
            # For TTA: use return_pil=True to get uint8 tensors for proper augmentation
            test_dataset_tta = dataset_utils.apply_random_corruptions(
                test_dataset_tta, flag, corruption_severity, cache=True, seed=42, return_pil=True
            )
            # Rebuild test loader
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=shared_loader_workers,
                pin_memory=loader_pin_memory,
            )
    elif base_flag == 'midog':
        # Load PathMNIST models and calibration data, use MIDOG as OOD test set
        models = tr.load_models('pathmnist', device=device, size=image_size, 
                               model_backbone=model_backbone, setup=setup)
        
        # PathMNIST is a color dataset - need color transforms
        pathmnist_transform, pathmnist_transform_tta = dataset_utils.get_transforms(True, image_size)
        
        [study_dataset, calib_dataset, pathmnist_test_dataset], \
        [_, calib_loader, _], info = \
            tr.load_datasets('pathmnist', True, image_size, pathmnist_transform, batch_size)  # color=True for PathMNIST
        
        # Load calibration dataset with TTA transform (for GPS augmentation caching)
        [_, calib_dataset_tta, _], \
        [_, _, _], _ = \
            tr.load_datasets('pathmnist', True, image_size, pathmnist_transform_tta, batch_size)
        
        # Load MIDOG as OOD test set (new class shift paradigm)
        if new_class_shift:
            test_dataset, test_loader, test_dataset_tta = dataset_utils.load_midog_for_new_class_shift(
                pathmnist_transform, pathmnist_transform_tta, models, device, 
                pathmnist_test_dataset,
                batch_size, workspace_root=Path(__file__).resolve().parent.parent.parent
            )
        else:
            raise ValueError("MIDOG dataset only supports new_class_shift mode. Use --new-class-shift flag.")
    print(f" Models: {len(models)} folds")
    # For organamnist: study=train (medMNIST), calib=val (medMNIST)
    # For others: study=80% of (train+val), calib=20% of (train+val)
    study_label = "Train" if flag == 'organamnist' else "Train+val"
    calib_label = "Val" if flag == 'organamnist' else "Calib"
    print(f" {study_label}: {len(study_dataset)}, {calib_label}: {len(calib_dataset)}, Test: {len(test_dataset)}")
    print(f" Task: {info['task']}, Classes: {len(info['label'])}")
    
    # ========================================================================
    # EVALUATE MODELS (or load from cache)
    # ========================================================================
    
    # Cache file paths - include model backbone, setup, and corruption params to avoid mixing results
    cache_dir = os.path.join(output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    setup_suffix = f"_{setup}" if setup else ""
    
    # Add corruption parameters to cache key
    corruption_suffix = ""
    if corruption_severity > 0:
        corruption_suffix = f"_corrupt{corruption_severity}"
        if corrupt_test:
            corruption_suffix += "_test"
        if corrupt_calib:
            corruption_suffix += "_calib"
    
    # Add new class shift to cache key
    new_class_suffix = "_new_class_shift" if new_class_shift else ""
    
    calib_cache_path = os.path.join(cache_dir, f'{flag}_{model_backbone}{setup_suffix}{corruption_suffix}{new_class_suffix}_calib_results.npz')
    test_cache_path = os.path.join(cache_dir, f'{flag}_{model_backbone}{setup_suffix}{corruption_suffix}{new_class_suffix}_test_results.npz')
    
    # Try to load cached results FIRST
    y_true_original = None  # Original class labels for new_class_shift stochastic methods
    cache_loaded = False
    if os.path.exists(calib_cache_path) and os.path.exists(test_cache_path):
        print("\nLoading cached evaluation results...")
        try:
            calib_cache = np.load(calib_cache_path, allow_pickle=True)
            test_cache = np.load(test_cache_path, allow_pickle=True)
            cache_loaded = True
        except (pickle.UnpicklingError, ValueError, EOFError) as e:
            print(f" [WARNING] Cache corrupted ({e.__class__.__name__}), regenerating...")
            # Delete corrupted cache files
            try:
                os.remove(calib_cache_path)
                os.remove(test_cache_path)
            except Exception:
                pass
            cache_loaded = False
    
    if cache_loaded:
        
        # Calibration
        y_true_calib = calib_cache['y_true']
        y_scores_calib = calib_cache['y_scores']
        correct_idx_calib = calib_cache['correct_idx']
        incorrect_idx_calib = calib_cache['incorrect_idx']
        indiv_scores_calib = calib_cache['indiv_scores']  # [N_calib, K, C]
        logits_calib = calib_cache['logits']  # [N_calib, C]
        
        # Test
        y_true = test_cache['y_true']
        y_scores = test_cache['y_scores']
        correct_idx = test_cache['correct_idx']
        incorrect_idx = test_cache['incorrect_idx']
        indiv_scores = test_cache['indiv_scores']  # [N, K, C]
        logits = test_cache['logits']  # [N, C]
        
        # For new class shift: load binary_gt and y_true_original early for dynamic index recomputation
        if new_class_shift:
            if 'binary_gt' in test_cache.files:
                y_true = test_cache['binary_gt']  # 0=known class, 1=OOD
            if 'y_true_original' in test_cache.files:
                y_true_original = test_cache['y_true_original']
            else:
                print(" [WARNING] y_true_original not in cache — delete the test cache file to regenerate.")
        
        # Check if per-fold logits are cached (new format)
        if 'indiv_logits' in test_cache.files:
            indiv_logits = test_cache['indiv_logits']  # [N, K, C]
            indiv_logits_calib = calib_cache['indiv_logits']  # [N_calib, K, C]
        else:
            # Old cache format without per-fold logits
            indiv_logits = None
            indiv_logits_calib = None
        
        # Check if per-fold predictions are cached
        if 'per_fold_predictions' in test_cache.files:
            per_fold_predictions = test_cache['per_fold_predictions']  # [K, N]
            per_fold_predictions_calib = calib_cache['per_fold_predictions']  # [K, N_calib]
            print(f" Loaded per-fold predictions from cache")
        else:
            # Old cache format - compute from indiv_scores
            print(f" [WARNING] Old cache format detected - computing per-fold predictions...")
            # Compute from indiv_scores (still in [N, K, C] format at this point)
            per_fold_predictions = np.argmax(indiv_scores, axis=2).T  # [N, K] → [K, N]
            per_fold_predictions_calib = np.argmax(indiv_scores_calib, axis=2).T  # [N_calib, K] → [K, N_calib]
        
        # Check if per-fold correct/incorrect indices are cached
        if 'per_fold_correct_idx' in test_cache.files:
            per_fold_correct_idx = [arr for arr in test_cache['per_fold_correct_idx']]
            per_fold_incorrect_idx = [arr for arr in test_cache['per_fold_incorrect_idx']]
            per_fold_correct_idx_calib = [arr for arr in calib_cache['per_fold_correct_idx']]
            per_fold_incorrect_idx_calib = [arr for arr in calib_cache['per_fold_incorrect_idx']]
            print(f" Loaded per-fold correct/incorrect indices from cache")
        else:
            # Old cache format - need to compute from per_fold_predictions
            print(f" [WARNING] Computing per-fold indices from predictions...")
            per_fold_correct_idx = []
            per_fold_incorrect_idx = []
            per_fold_correct_idx_calib = []
            per_fold_incorrect_idx_calib = []
            
            for fold_idx in range(per_fold_predictions.shape[0]):
                fold_correct = np.where(per_fold_predictions[fold_idx] == y_true)[0]
                fold_incorrect = np.where(per_fold_predictions[fold_idx] != y_true)[0]
                per_fold_correct_idx.append(fold_correct)
                per_fold_incorrect_idx.append(fold_incorrect)
                
                fold_correct_calib = np.where(per_fold_predictions_calib[fold_idx] == y_true_calib)[0]
                fold_incorrect_calib = np.where(per_fold_predictions_calib[fold_idx] != y_true_calib)[0]
                per_fold_correct_idx_calib.append(fold_correct_calib)
                per_fold_incorrect_idx_calib.append(fold_incorrect_calib)
        
        # For new class shift: always recompute per-fold indices dynamically from predictions
        # incorrect_idx = OOD only; correct_idx = per-fold correct predictions on known-class
        if new_class_shift and y_true_original is not None:
            _known_mask = y_true_original != -1
            _ood_idx = np.where(~_known_mask)[0]  # OOD only — fixed for all folds and methods
            per_fold_correct_idx = []
            per_fold_incorrect_idx = []
            for fold_idx in range(per_fold_predictions.shape[0]):
                _fc = np.where((per_fold_predictions[fold_idx] == y_true_original) & _known_mask)[0]
                per_fold_correct_idx.append(_fc)
                per_fold_incorrect_idx.append(_ood_idx)  # OOD only
        
        # Transpose to [K, N, C] format for per-fold evaluation
        indiv_scores = np.transpose(indiv_scores, (1, 0, 2))  # [K, N, C]
        indiv_scores_calib = np.transpose(indiv_scores_calib, (1, 0, 2))  # [K, N_calib, C]
        if indiv_logits is not None:
            indiv_logits = np.transpose(indiv_logits, (1, 0, 2))  # [K, N, C]
            indiv_logits_calib = np.transpose(indiv_logits_calib, (1, 0, 2))  # [K, N_calib, C]
        
        # Compute y_pred from cached scores (for compatibility with old cache format)
        y_pred = np.argmax(y_scores, axis=1)
        y_pred_calib = np.argmax(y_scores_calib, axis=1)
        
        # For new class shift: ensemble correct = fold intersection; incorrect = OOD only
        if new_class_shift and y_true_original is not None:
            _known_mask = y_true_original != -1
            _all_correct = _known_mask.copy()
            for _fc in per_fold_correct_idx:
                _fold_mask = np.zeros(len(y_true_original), dtype=bool)
                _fold_mask[_fc] = True
                _all_correct &= _fold_mask
            correct_idx = np.where(_all_correct)[0]
            incorrect_idx = np.where(~_known_mask)[0]  # OOD only
        
        # Print summary
        if new_class_shift:
            n_known = int(np.sum(y_true == 0))
            n_ood = int(np.sum(y_true == 1))
            print(f" Loaded cached results (new class shift mode)")
            print(f" Test set: {n_known} known-class + {n_ood} OOD = {len(y_true)} total ({100*n_ood/len(y_true):.1f}% OOD)")
            if y_true_original is not None:
                print(f" Dynamic ensemble: {len(correct_idx)} successes, {len(incorrect_idx)} failures")
            else:
                print(" [WARNING] y_true_original missing — delete cache to regenerate with dynamic indices.")
        else:
            print(f" Loaded cached results")
            print(f" Test accuracy: {len(correct_idx) / len(y_true):.4f}")
    
    else:
        # No cache - evaluate models
        print("\nEvaluating ensemble predictions on test set...")
        y_true, y_scores, y_pred, correct_idx, incorrect_idx, indiv_scores_raw, logits = uq.evaluate_models_on_loader(
            models, test_loader, device, return_logits=True
        )
        
        # For new class shift: replace y_true with binary ground truth for failure detection
        if new_class_shift and hasattr(test_dataset, 'binary_gt'):
            print(" Using binary ground truth for new class shift evaluation")
            y_true_original = y_true.copy()  # Keep original class labels
            y_true = test_dataset.binary_gt  # Binary: 0=known class, 1=OOD
            
            # Preliminary ensemble indices (will be refined to fold intersection after per-fold loop)
            known_mask = y_true_original != -1
            correct_mask = (y_pred == y_true_original) & known_mask
            correct_idx = np.where(correct_mask)[0]
            incorrect_idx = np.where(~known_mask)[0]  # OOD only
        
        # Calibration set
        y_true_calib, y_scores_calib, y_pred_calib, correct_idx_calib, incorrect_idx_calib, indiv_scores_calib_raw, logits_calib = \
            uq.evaluate_models_on_loader(models, calib_loader, device, return_logits=True)
        
        if new_class_shift and hasattr(test_dataset, 'binary_gt'):
            print(f" Test set composition: {len(correct_idx)}/{len(y_true)} ensemble successes "
                  f"({100*len(correct_idx)/len(y_true):.1f}%) — not model accuracy")
        
        # Compute per-fold logits from models
        print("\nComputing per-fold logits for calibration...")
        indiv_logits_raw = []  # Will be [N, K, C]
        indiv_logits_calib_raw = []  # Will be [N_calib, K, C]
        
        for model in models:
            model.eval()
            # Test set
            test_logits_fold = []
            with torch.no_grad():
                for batch in test_loader:
                    if isinstance(batch, dict):
                        images = batch["image"].to(device)
                    else:
                        images = batch[0].to(device)
                    logits_batch = model(images)
                    test_logits_fold.append(logits_batch.cpu().numpy())
            indiv_logits_raw.append(np.concatenate(test_logits_fold, axis=0))  # [N, C]
            
            # Calibration set
            calib_logits_fold = []
            with torch.no_grad():
                for batch in calib_loader:
                    if isinstance(batch, dict):
                        images = batch["image"].to(device)
                    else:
                        images = batch[0].to(device)
                    logits_batch = model(images)
                    calib_logits_fold.append(logits_batch.cpu().numpy())
            indiv_logits_calib_raw.append(np.concatenate(calib_logits_fold, axis=0))  # [N_calib, C]
        
        # Stack to [K, N, C] and [K, N_calib, C]
        indiv_logits_raw = np.stack(indiv_logits_raw, axis=1)  # [N, K, C]
        indiv_logits_calib_raw = np.stack(indiv_logits_calib_raw, axis=1)  # [N_calib, K, C]
        
        # Transpose to [K, N, C] for per-fold evaluation
        indiv_scores = np.transpose(indiv_scores_raw, (1, 0, 2))  # [K, N, C]
        indiv_scores_calib = np.transpose(indiv_scores_calib_raw, (1, 0, 2))  # [K, N_calib, C]
        indiv_logits = np.transpose(indiv_logits_raw, (1, 0, 2))  # [K, N, C]
        indiv_logits_calib = np.transpose(indiv_logits_calib_raw, (1, 0, 2))  # [K, N_calib, C]
        
        # Compute per-fold correct/incorrect indices
        print("\nComputing per-fold correct/incorrect indices...")
        per_fold_correct_idx = []
        per_fold_incorrect_idx = []
        per_fold_correct_idx_calib = []
        per_fold_incorrect_idx_calib = []
        
        for fold_idx in range(len(models)):
            # Test set
            fold_preds = np.argmax(indiv_scores[fold_idx], axis=1)  # [N]
            
            # For new class shift: per-fold correct on known-class; incorrect = OOD only
            if new_class_shift and y_true_original is not None:
                _known_mask = y_true_original != -1
                fold_correct = np.where((fold_preds == y_true_original) & _known_mask)[0]
                fold_incorrect = np.where(~_known_mask)[0]  # OOD only
            else:
                fold_correct = np.where(fold_preds == y_true)[0]
                fold_incorrect = np.where(fold_preds != y_true)[0]
            
            per_fold_correct_idx.append(fold_correct)
            per_fold_incorrect_idx.append(fold_incorrect)
            
            # Calibration set
            fold_preds_calib = np.argmax(indiv_scores_calib[fold_idx], axis=1)  # [N_calib]
            fold_correct_calib = np.where(fold_preds_calib == y_true_calib)[0]
            fold_incorrect_calib = np.where(fold_preds_calib != y_true_calib)[0]
            per_fold_correct_idx_calib.append(fold_correct_calib)
            per_fold_incorrect_idx_calib.append(fold_incorrect_calib)
            
            print(f" Fold {fold_idx}: {len(fold_correct)} correct, {len(fold_incorrect)} incorrect (test)")
        
        # For new class shift: ensemble correct = fold intersection; incorrect = OOD only
        if new_class_shift and y_true_original is not None:
            _known_mask = y_true_original != -1
            _all_correct = _known_mask.copy()
            for _fc in per_fold_correct_idx:
                _fold_mask = np.zeros(len(y_true_original), dtype=bool)
                _fold_mask[_fc] = True
                _all_correct &= _fold_mask
            correct_idx = np.where(_all_correct)[0]
            incorrect_idx = np.where(~_known_mask)[0]  # OOD only
            print(f" Ensemble (fold intersection): {len(correct_idx)} correct, "
                  f"{len(incorrect_idx)} OOD — OOD rate: {100*len(incorrect_idx)/len(y_true):.1f}%")
        
        # Compute per-fold predictions [M, N] for caching
        per_fold_predictions = np.argmax(indiv_scores, axis=2)  # [K, N, C] → [K, N]
        per_fold_predictions_calib = np.argmax(indiv_scores_calib, axis=2)  # [K, N_calib, C] → [K, N_calib]
        
        # Save to cache for next time
        print("\nSaving evaluation results to cache...")
        np.savez_compressed(
            calib_cache_path,
            y_true=y_true_calib,
            y_scores=y_scores_calib,
            y_pred=y_pred_calib,
            correct_idx=correct_idx_calib,
            incorrect_idx=incorrect_idx_calib,
            indiv_scores=indiv_scores_calib_raw,  # Save as [N, K, C]
            logits=logits_calib,
            indiv_logits=indiv_logits_calib_raw,  # Save as [N, K, C]
            per_fold_correct_idx=np.array(per_fold_correct_idx_calib, dtype=object),  # Object array of variable-length arrays
            per_fold_incorrect_idx=np.array(per_fold_incorrect_idx_calib, dtype=object),  # Object array
            per_fold_predictions=per_fold_predictions_calib  # [K, N_calib]
        )
        # Save test cache (with binary_gt if new_class_shift)
        cache_data = dict(
            y_true=y_true,
            y_scores=y_scores,
            y_pred=y_pred,
            correct_idx=correct_idx,
            incorrect_idx=incorrect_idx,
            indiv_scores=indiv_scores_raw,  # Save as [N, K, C]
            logits=logits,
            indiv_logits=indiv_logits_raw,  # Save as [N, K, C]
            per_fold_correct_idx=np.array(per_fold_correct_idx, dtype=object),  # Object array of variable-length arrays
            per_fold_incorrect_idx=np.array(per_fold_incorrect_idx, dtype=object),  # Object array
            per_fold_predictions=per_fold_predictions  # [K, N]
        )
        if new_class_shift and hasattr(test_dataset, 'binary_gt'):
            cache_data['binary_gt'] = test_dataset.binary_gt  # Save binary ground truth for risk computation
            cache_data['y_true_original'] = y_true_original   # Original class labels for stochastic method eval
        np.savez_compressed(test_cache_path, **cache_data)
        print(f" Cached to {cache_dir}")
    
    # ========================================================================
    # CREATE FAILCATCHER DETECTOR
    # ========================================================================
    detector = failure_detection.FailureDetector(
        models=models,
        study_dataset=study_dataset,
        calib_dataset=calib_dataset,
        test_dataset=test_dataset,
        device=device,
        num_classes=len(info['label'])
    )
    
    # Set predictions once to avoid recomputing for each method
    # For new_class_shift: pass pre-computed correct/incorrect indices to avoid recomputation
    if new_class_shift:
        detector.set_test_predictions(y_scores, y_true, y_pred, correct_idx, incorrect_idx,
                                      per_fold_correct_idx, per_fold_incorrect_idx)
    else:
        detector.set_test_predictions(y_scores, y_true, y_pred)
    
    # Set per-fold predictions to avoid redundant vanilla inference
    # This is especially important when running multiple UQ methods
    detector.set_per_fold_predictions(per_fold_predictions)
    print(" Pre-cached per-fold predictions - vanilla inference will be skipped")

    # Separate detector for calibration-only uncertainty statistics.
    # We do NOT save calibration uncertainties; we only persist mean/std summaries.
    calib_detector = failure_detection.FailureDetector(
        models=models,
        study_dataset=study_dataset,
        calib_dataset=calib_dataset,
        test_dataset=calib_dataset,
        device=device,
        num_classes=len(info['label'])
    )
    calib_detector.set_test_predictions(
        y_scores_calib, y_true_calib, y_pred_calib,
        correct_idx_calib, incorrect_idx_calib,
        per_fold_correct_idx_calib, per_fold_incorrect_idx_calib
    )
    calib_detector.set_per_fold_predictions(per_fold_predictions_calib)

    calibration_zscore_stats = {}

    # For amos2022/midog in test_only mode: load pre-computed calibration z-score stats
    # from the corresponding in-distribution dataset results (organamnist / pathmnist).
    # This avoids re-running the expensive calibration pass.
    if run_mode == 'test_only' and base_flag in ['amos2022', 'midog']:
        import json as _json
        import glob as _glob
        calib_source_flag = gps_cache_flag  # 'organamnist' or 'pathmnist'
        setup_suffix_str = f'_{setup}' if setup else ''
        # Filename prefix to match (exclude corrupt runs via negative pattern)
        fname_prefix = f'uq_benchmark_{calib_source_flag}_{model_backbone}{setup_suffix_str}_'
        # Search directories in priority order
        _search_dirs = [
            os.path.join(output_dir),
            os.path.join(output_dir, 'jsons_results', 'in_distribution'),
            str(workspace_root / 'Benchmarks' / 'medMNIST' / 'results'),
            str(workspace_root / 'Benchmarks' / 'medMNIST' / 'results' / 'jsons_results' / 'in_distribution'),
        ]
        _loaded_stats = None
        for _sdir in _search_dirs:
            _candidates = sorted(_glob.glob(os.path.join(_sdir, f'{fname_prefix}*.json')))
            # Exclude corrupt runs (they use a different calib distribution)
            _candidates = [p for p in _candidates if '_corrupt_' not in os.path.basename(p)]
            if not _candidates:
                continue
            # Use the most recent matching file
            for _cpath in reversed(_candidates):
                try:
                    with open(_cpath) as _f:
                        _jdata = _json.load(_f)
                    _stats = _jdata.get('methods', {}).get('Calibration_ZScore_Stats')
                    if _stats:
                        _loaded_stats = _stats
                        print(f" Loaded calibration z-score stats from: {_cpath}")
                        break
                except Exception as _e:
                    print(f" [WARNING] Could not load calib stats from {_cpath}: {_e}")
            if _loaded_stats:
                break
        if _loaded_stats:
            calibration_zscore_stats = _loaded_stats
        else:
            print(f" [WARNING] No calibration z-score stats found for "
                  f"{calib_source_flag} {model_backbone} {setup or 'standard'}.")
            print(f"  ZScore_Aggregation will be skipped. Run without --run-mode test_only "
                  f"or run calib_only mode first.")

    def _store_calibration_stats(method_key: str):
        """Store mean/std stats from calibration uncertainties for later z-score normalization."""
        method_stats = {}

        per_fold_key = f'{method_key}_per_fold'
        ensemble_key = f'{method_key}_ensemble'

        if per_fold_key in calib_detector._uncertainties:
            per_fold_vals = np.asarray(calib_detector._uncertainties[per_fold_key])
            if per_fold_vals.ndim == 2:
                method_stats['per_fold'] = [
                    {
                        'fold': int(fold_idx),
                        'mean': float(np.mean(per_fold_vals[fold_idx])),
                        'std': float(np.std(per_fold_vals[fold_idx]))
                    }
                    for fold_idx in range(per_fold_vals.shape[0])
                ]

        if ensemble_key in calib_detector._uncertainties:
            ensemble_vals = np.asarray(calib_detector._uncertainties[ensemble_key])
            method_stats['ensemble'] = {
                'mean': float(np.mean(ensemble_vals)),
                'std': float(np.std(ensemble_vals))
            }
        elif method_key in calib_detector._uncertainties:
            base_vals = np.asarray(calib_detector._uncertainties[method_key])
            if base_vals.ndim == 1:
                method_stats['ensemble'] = {
                    'mean': float(np.mean(base_vals)),
                    'std': float(np.std(base_vals))
                }

        if method_stats:
            calibration_zscore_stats[method_key] = method_stats
    
    # ========================================================================
    # RUN UQ METHODS using FailCatcher API
    # ========================================================================
    results = {}
    run_test = run_mode != 'calib_only'   # whether to run test-set detector
    run_calib = run_mode != 'test_only'   # whether to run calib-set detector

    if 'MSR' in methods:
        if run_test:
            print("\nRunning MSR...")
            mode_str = "per-fold" if per_fold_eval else "ensemble"
            print(f" Mode: {mode_str} evaluation")
            uncertainties, metrics = detector.run_msr(
                y_scores, y_true, 
                indiv_scores=indiv_scores if per_fold_eval else None,
                logits=logits,
                indiv_logits=indiv_logits if per_fold_eval else None,
                per_fold_evaluation=per_fold_eval
            )
            results['MSR'] = metrics
            if 'auroc_f_mean' in metrics:
                print(f" AUROC: {metrics['auroc_f_mean']:.4f}±{metrics['auroc_f_std']:.4f}, "
                      f"AUGRC: {metrics['augrc_mean']:.6f}±{metrics['augrc_std']:.6f}")
            else:
                print(f" AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")

        if run_calib:
            # Calibration-distribution uncertainty stats (means/stds only)
            calib_detector.run_msr(
                y_scores_calib, y_true_calib,
                indiv_scores=indiv_scores_calib if per_fold_eval else None,
                logits=logits_calib,
                indiv_logits=indiv_logits_calib if per_fold_eval else None,
                per_fold_evaluation=per_fold_eval
            )
            _store_calibration_stats('MSR')

    if 'MSR_calibrated' in methods:
        if run_test:
            print(f"\nRunning MSR-{calib_method}...")
            mode_str = "per-fold" if per_fold_eval else "ensemble"
            print(f" Mode: {mode_str} evaluation")
            uncertainties, metrics = detector.run_msr_calibrated(
                y_scores, y_true, y_scores_calib, y_true_calib,
                logits, logits_calib,
                indiv_logits_test=indiv_logits if (per_fold_eval and indiv_logits is not None) else None,
                indiv_logits_calib=indiv_logits_calib if (per_fold_eval and indiv_logits_calib is not None) else None,
                indiv_scores_test=indiv_scores if per_fold_eval else None,
                indiv_scores_calib=indiv_scores_calib if per_fold_eval else None,
                method=calib_method,
                per_fold_evaluation=per_fold_eval,
                auto_tune_platt=True,  # Enable automatic hyperparameter selection
                verbose_tuning=True    # Print tuning results
            )
            results[f'MSR_{calib_method}'] = metrics
            if 'auroc_f_mean' in metrics:
                print(f" AUROC: {metrics['auroc_f_mean']:.4f}±{metrics['auroc_f_std']:.4f}, "
                      f"AUGRC: {metrics['augrc_mean']:.6f}±{metrics['augrc_std']:.6f}")
            else:
                print(f" AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")

        if run_calib:
            # Calibration-distribution uncertainty stats (means/stds only)
            calib_detector.run_msr_calibrated(
                y_scores_calib, y_true_calib, y_scores_calib, y_true_calib,
                logits_calib, logits_calib,
                indiv_logits_test=indiv_logits_calib if (per_fold_eval and indiv_logits_calib is not None) else None,
                indiv_logits_calib=indiv_logits_calib if (per_fold_eval and indiv_logits_calib is not None) else None,
                indiv_scores_test=indiv_scores_calib if per_fold_eval else None,
                indiv_scores_calib=indiv_scores_calib if per_fold_eval else None,
                method=calib_method,
                per_fold_evaluation=per_fold_eval,
                auto_tune_platt=True,
                verbose_tuning=False
            )
            _store_calibration_stats('MSR_calibrated')
    
    if 'MLS' in methods:
        if run_test:
            print("\nRunning MLS (Maximum Logit Score)...")
            mode_str = "per-fold" if per_fold_eval else "ensemble"
            print(f" Mode: {mode_str} evaluation")
            uncertainties, metrics = detector.run_mls(
                logits, y_true,
                indiv_logits=indiv_logits if per_fold_eval else None,
                per_fold_evaluation=per_fold_eval
            )
            results['MLS'] = metrics
            if 'auroc_f_mean' in metrics:
                print(f" AUROC: {metrics['auroc_f_mean']:.4f}±{metrics['auroc_f_std']:.4f}, "
                      f"AUGRC: {metrics['augrc_mean']:.6f}±{metrics['augrc_std']:.6f}")
            else:
                print(f" AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")

        if run_calib:
            calib_detector.run_mls(
                logits_calib, y_true_calib,
                indiv_logits=indiv_logits_calib if per_fold_eval else None,
                per_fold_evaluation=per_fold_eval
            )
            _store_calibration_stats('MLS')
    
    if 'Ensembling' in methods:
        if run_test:
            print("\nRunning Ensemble STD...")
            uncertainties, metrics = detector.run_ensemble(indiv_scores, y_true)
            results['Ensemble'] = metrics
            print(f" AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")

        if run_calib:
            calib_detector.run_ensemble(indiv_scores_calib, y_true_calib)
            _store_calibration_stats('Ensembling')
    
    if 'TTA' in methods:
        # TTA/GPS batch size - also needs reduction for ViT on large datasets
        tta_gps_batch_size = batch_size
        if model_backbone == 'vit_b_16' and batch_size > 3000:
            tta_gps_batch_size = 3000
            print(f" Note: Using reduced batch size {tta_gps_batch_size} for TTA/GPS with ViT (avoids OOM)")
        if run_test:
            print("\nRunning TTA...")
            mode_str = "per-fold" if per_fold_eval else "ensemble"
            print(f" Mode: {mode_str} evaluation")
            uncertainties, metrics = detector.run_tta(
                test_dataset_tta, y_true,
                image_size=image_size,
                batch_size=tta_gps_batch_size,
                nb_augmentations=5,
                per_fold_evaluation=per_fold_eval,
                seed=42,
                y_true_original=y_true_original if new_class_shift else None,
            )
            results['TTA'] = metrics
            print(f" AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")

        if run_calib:
            calib_detector.run_tta(
                calib_dataset_tta, y_true_calib,
                image_size=image_size,
                batch_size=tta_gps_batch_size,
                nb_augmentations=5,
                per_fold_evaluation=per_fold_eval,
                seed=42
            )
            _store_calibration_stats('TTA')
    
    if 'TTA_calib' in methods:
        print("\nRunning TTA Calibration Caching (BetterRandAugment)...")
        if gps_calib_samples is not None:
            print(f" Subsampling strategy: Keep {gps_calib_samples} samples with min {min_failure_ratio:.0%} failures")
        setup_name = setup if setup else 'standard'
        # Include sample count in folder name if subsampling occurs
        folder_suffix = f'_N{gps_calib_samples}' if gps_calib_samples is not None else ''
        aug_folder = os.path.join(output_dir, 'gps_augment_cache', f'{gps_cache_flag}_{model_backbone}_{setup_name}_calibration_set{folder_suffix}')
        
        # Subsample calibration dataset (failure-aware for GPS)
        # Prioritizes failures (incorrect predictions) to maximize information density
        # Using fixed seed ensures consistency between TTA_calib and GPS search
        # CRITICAL: Pass normalized calib_dataset for accurate ensemble inference,
        #           but return subset of unnormalized calib_dataset_tta for augmentation
        # Returns: (subsampled_dataset, correct_indices, incorrect_indices)
        calib_dataset_tta_subsampled, correct_idx_calib_subsampled, incorrect_idx_calib_subsampled = \
            dataset_utils.subsample_dataset_failure_aware(
                dataset=calib_dataset_tta,
                models=models,
                device=device,
                max_samples=gps_calib_samples,
                min_failure_ratio=min_failure_ratio,
                seed=42,
                batch_size=batch_size,
                eval_dataset=calib_dataset  # Use normalized data for ensemble inference!
            )
        
        # correct_idx_calib_subsampled and incorrect_idx_calib_subsampled 
        # are already computed by subsample_dataset_failure_aware()
        # They are 0-indexed positions within the subsampled dataset
        print(f" GPS will use: {len(correct_idx_calib_subsampled)} correct, {len(incorrect_idx_calib_subsampled)} incorrect indices")
        
        # Determine normalization parameters based on color
        # Note: nb_channels should always be 3 because models expect 3-channel input
        # (grayscale is converted via RepeatGrayToRGB in the dataset)
        nb_channels = 3
        mean = [.5, .5, .5] if color else [.5]
        std = [.5, .5, .5] if color else [.5]
        
        # Use smaller batch size for augmentation caching to avoid OOM
        # Each batch gets multiplied by num_policies, so memory usage is much higher
        aug_batch_size = min(batch_size, 256)  # Conservative for memory safety
        
        # Use MONAI cache with full rate for speed
        # Cache is stored in RAM (CPU memory), not GPU, so it's safe
        print(f" Original calibration set size: {len(calib_dataset_tta)}")
        print(f" Subsampled calibration set size: {len(calib_dataset_tta_subsampled)}")
        print(f" Batch size: {aug_batch_size}")
        
        # Cache augmentation predictions on calibration dataset
        aug_folder = detector.run_augmentation_calibration_caching(
            dataset=calib_dataset_tta_subsampled,  # Use subsampled dataset!
            aug_folder=aug_folder,
            N=2,                      # Number of augmentation ops per policy
            M=45,                     # Magnitude parameter
            num_policies=500,         # Number of random policies to generate
            image_size=image_size,
            batch_size=aug_batch_size,
            nb_channels=nb_channels,
            image_normalization=True,
            mean=mean,
            std=std,
            use_monai_cache=True,
            cache_rate=1.0,  # Full cache in RAM for speed
            cache_num_workers=max(4, min(24, shared_loader_workers // 2)),
            dataloader_workers=max(4, min(16, shared_loader_workers // 2)),  # Increased cap for high-CPU hosts
            dataloader_prefetch=4  # Larger prefetch queue when workers are plentiful
        )
        print(f" Augmentation predictions cached in: {aug_folder}")
        # Note: TTA_calib doesn't produce uncertainty scores, it only caches predictions
        # The cached predictions are used by GPS method
    

    if 'GPS' in methods:
        print("\nRunning GPS...")
    
        # TTA/GPS batch size - also needs reduction for ViT on large datasets
        tta_gps_batch_size = batch_size
        if model_backbone == 'vit_b_16' and batch_size > 3000:
            tta_gps_batch_size = 3000
            print(f" Note: Using reduced batch size {tta_gps_batch_size} for TTA/GPS with ViT (avoids OOM)")
        
        # CRITICAL: Reset test_dataset_tta transform to original state
        # TTA may have modified it, and GPS needs clean dataset
        test_dataset_tta.transform = transform_tta

        # Resolve GPS cache root. Prefer current output dir, but fall back to the
        # medMNIST benchmark results cache location when it already exists.
        gps_cache_root_candidates = [
            os.path.join(output_dir, 'gps_augment_cache'),
            str(workspace_root / 'Benchmarks' / 'medMNIST' / 'results' / 'gps_augment_cache'),
        ]
        # Keep order while removing duplicates
        gps_cache_root_candidates = list(dict.fromkeys(gps_cache_root_candidates))
        
        setup_name = setup if setup else 'standard'
        # Include sample count in folder name if subsampling occurs
        folder_suffix = f'_N{gps_calib_samples}' if gps_calib_samples is not None else ''
        aug_folder_name = f'{gps_cache_flag}_{model_backbone}_{setup_name}_calibration_set{folder_suffix}'
        aug_folder = os.path.join(gps_cache_root_candidates[0], aug_folder_name)

        if 'TTA_calib' not in methods:
            for candidate_root in gps_cache_root_candidates:
                candidate_folder = os.path.join(candidate_root, aug_folder_name)
                if os.path.isdir(candidate_folder):
                    aug_folder = candidate_folder
                    break

            if not os.path.isdir(aug_folder):
                checked_paths = [os.path.join(root, aug_folder_name) for root in gps_cache_root_candidates]
                raise FileNotFoundError(
                    "GPS augmentation cache folder not found. "
                    "Run with TTA_calib first or point --output-dir to a results folder containing gps_augment_cache. "
                    f"Checked: {checked_paths}"
                )
        
        # If TTA_calib was run, use the subsampled indices
        # Otherwise, compute them now (GPS can run independently of TTA_calib)
        if 'TTA_calib' in methods:
            # Use the subsampled indices computed during TTA_calib
            gps_correct_idx = correct_idx_calib_subsampled.tolist()
            gps_incorrect_idx = incorrect_idx_calib_subsampled.tolist()
            print(f" Using subsampled calibration indices from TTA_calib")
        else:
            # GPS running independently - need to subsample and compute indices
            print(f" TTA_calib not run - computing subsampled calibration indices...")
            calib_dataset_tta_subsampled, gps_correct_idx, gps_incorrect_idx = \
                dataset_utils.subsample_dataset_failure_aware(
                    dataset=calib_dataset_tta,
                    models=models,
                    device=device,
                    max_samples=gps_calib_samples,
                    eval_dataset=calib_dataset,  # Use normalized data for ensemble inference!
                    min_failure_ratio=min_failure_ratio,
                    seed=42,
                    batch_size=batch_size
                )
            # Convert numpy arrays to lists for GPS
            gps_correct_idx = gps_correct_idx.tolist()
            gps_incorrect_idx = gps_incorrect_idx.tolist()

        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f" Mode: {mode_str} evaluation")
        if run_test:
            uncertainties, metrics = detector.run_gps(
                test_dataset_tta, y_true,
                aug_folder=aug_folder,
                correct_idx_calib=gps_correct_idx,
                incorrect_idx_calib=gps_incorrect_idx,
                image_size=image_size,
                batch_size=tta_gps_batch_size,
                cache_dir=os.path.dirname(aug_folder),
                per_fold_evaluation=per_fold_eval,
                y_true_original=y_true_original if new_class_shift else None,
            )
            results['GPS'] = metrics
            print(f" AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")

        if run_calib:
            calib_detector.run_gps(
                calib_dataset_tta, y_true_calib,
                aug_folder=aug_folder,
                correct_idx_calib=gps_correct_idx,
                incorrect_idx_calib=gps_incorrect_idx,
                image_size=image_size,
                batch_size=tta_gps_batch_size,
                cache_dir=os.path.dirname(aug_folder),
                per_fold_evaluation=per_fold_eval
            )
            _store_calibration_stats('GPS')
    
    if 'KNN_Raw' in methods:
        print("\nRunning KNN-Raw...")

        # Adaptive batch size for KNN methods based on model architecture
        # KNN requires full forward passes on large datasets which can OOM
        knn_batch_size = min(batch_size, 3000)  # Conservative default for all models
        knn_test_loader = test_loader
        # Further reduce for ViT models which consume significantly more memory
        if model_backbone == 'vit_b_16':
            knn_batch_size = min(batch_size, 4000)  # Reduce to 4000 for ViT to avoid OOM
            print(f" Note: Using reduced batch size {knn_batch_size} for KNN with ViT (avoids OOM)")
        elif knn_batch_size < batch_size:
            print(f" Note: Using reduced batch size {knn_batch_size} for KNN (avoids OOM on large datasets)")
        
        # Create reduced batch size test loader if needed
        if knn_batch_size < batch_size:
            knn_test_loader = DataLoader(
                test_dataset, batch_size=knn_batch_size, shuffle=False,
                num_workers=shared_loader_workers, pin_memory=loader_pin_memory
            )
        
        # Create CV train loaders for KNN methods
        cv_gen = dataset_utils.create_cv_generator(n_splits=5, seed=42, batch_size=knn_batch_size)
        train_loaders = cv_gen(study_dataset, models, knn_batch_size)
        
        # Create KNN-specific calibration loader with reduced batch size (for hyperparameter tuning)
        if knn_batch_size < batch_size:
            knn_calib_loader = DataLoader(
                calib_dataset, batch_size=knn_batch_size, shuffle=False,
                num_workers=shared_loader_workers, pin_memory=loader_pin_memory
            )
        else:
            knn_calib_loader = calib_loader  # Use original loader if batch size is same
            
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f" Mode: {mode_str} evaluation")
        
        # k is selected via grid search on the calibration set for all evaluation scenarios.
        k = None
        k_grid = [1, 5, 10, 20, 50, 100, 200]
        print(f" Using k grid search: {k_grid}")
        
        if run_test:
            uncertainties, metrics = detector.run_knn_raw(
                test_loader=knn_test_loader,
                train_loaders=train_loaders,
                y_true=y_true,
                layer_name='avgpool',
                k=k,
                per_fold_evaluation=per_fold_eval,
                k_grid=k_grid,
                calib_loader=knn_calib_loader,
                y_true_calib=y_true_calib
            )
            results['KNN_Raw'] = metrics
            
            # Print results
            if 'auroc_f_mean' in metrics:
                print(f" AUROC: {metrics['auroc_f_mean']:.4f}±{metrics['auroc_f_std']:.4f}, "
                      f"AUGRC: {metrics['augrc_mean']:.6f}±{metrics['augrc_std']:.6f}")
            else:
                print(f" AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")

        k_for_calib = (metrics['k_selected'] if 'k_selected' in metrics else (5 if k is None else k)) \
                      if run_test else 5
        if run_calib:
            calib_detector.run_knn_raw(
                test_loader=knn_calib_loader,
                train_loaders=train_loaders,
                y_true=y_true_calib,
                layer_name='avgpool',
                k=k_for_calib,
                per_fold_evaluation=per_fold_eval,
                k_grid=None,
                calib_loader=None,
                y_true_calib=None
            )
            _store_calibration_stats('KNN_Raw')
    
    if 'KNN_SHAP' in methods:
        print("\nRunning KNN-SHAP...")
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f" Mode: {mode_str} evaluation")
        parallel_mode = torch.cuda.device_count() >= 3
        n_jobs = 3 if parallel_mode else 1
        
        if run_test:
            uncertainties, metrics = detector.run_knn_shap(
                calib_loader=calib_loader,
                test_loader=knn_test_loader,
                train_loaders=train_loaders,
                y_true=y_true,
                flag=flag,
                layer_name='avgpool',
                k=5,
                n_shap_features=50,
                cache_dir=os.path.join(output_dir, 'shap_cache'),
                parallel=parallel_mode,
                n_jobs=n_jobs,
                per_fold_evaluation=per_fold_eval
            )
            results['KNN_SHAP'] = metrics
            print(f" AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")

        if run_calib:
            calib_detector.run_knn_shap(
                calib_loader=calib_loader,
                test_loader=knn_calib_loader,
                train_loaders=train_loaders,
                y_true=y_true_calib,
                flag=f"{flag}_calib",
                layer_name='avgpool',
                k=5,
                n_shap_features=50,
                cache_dir=os.path.join(output_dir, 'shap_cache'),
                parallel=parallel_mode,
                n_jobs=n_jobs,
                per_fold_evaluation=per_fold_eval
            )
            _store_calibration_stats('KNN_SHAP')
    
    # ========================================================================
    # MC DROPOUT - RUN LAST TO AVOID INTERFERING WITH OTHER METHODS
    # ========================================================================
    # MCDropout is executed last because it modifies dropout layer states
    # which can cause CUDA RNG issues with subsequent DataLoader forking
    # in methods like GPS/TTA that use multiprocessing workers
    if 'MCDropout' in methods:
        print("\nRunning MC Dropout (running last to avoid interference)...")
        mode_str = "per-fold" if per_fold_eval else "ensemble"
        print(f" Mode: {mode_str} evaluation")

        mcd_num_workers = shared_loader_workers
        mcd_pin_memory = loader_pin_memory
        mcd_persistent_workers = mcd_num_workers > 0
        mcd_prefetch_factor = 2 if mcd_num_workers > 0 else None
        print(
            f"  DataLoader: workers={mcd_num_workers}, pin_memory={mcd_pin_memory}, "
            f"persistent_workers={mcd_persistent_workers}, prefetch_factor={mcd_prefetch_factor}"
        )

        try:
            if run_test:
                uncertainties, metrics = detector.run_mcdropout(
                    test_dataset, y_true,
                    batch_size=batch_size,
                    num_samples=30,
                    per_fold_evaluation=per_fold_eval,
                    num_workers=mcd_num_workers,
                    pin_memory=mcd_pin_memory,
                    persistent_workers=mcd_persistent_workers,
                    prefetch_factor=mcd_prefetch_factor,
                    y_true_original=y_true_original if new_class_shift else None,
                )
                results['MCDropout'] = metrics
                print(f" AUROC: {metrics['auroc_f']:.4f}, AUGRC: {metrics['augrc']:.6f}")

            if run_calib:
                calib_detector.run_mcdropout(
                    calib_dataset, y_true_calib,
                    batch_size=batch_size,
                    num_samples=30,
                    per_fold_evaluation=per_fold_eval,
                    num_workers=mcd_num_workers,
                    pin_memory=mcd_pin_memory,
                    persistent_workers=mcd_persistent_workers,
                    prefetch_factor=mcd_prefetch_factor
                )
                _store_calibration_stats('MCDropout')
        except ValueError as e:
            print(f" [SKIPPED] MCDropout: {e}")

    # ========================================================================
    # Z-SCORE AGGREGATION METHODS (CALLABLE VIA --methods)
    # ========================================================================
    aggregation_candidates = [
        'MSR', 'MSR_calibrated', 'MLS', 'Ensembling',
        'GPS', 'KNN_Raw', 'MCDropout'
    ]
    aggregation_methods_ensemble = [
        name for name in aggregation_candidates
        if name in detector._uncertainties
    ]
    aggregation_methods_per_fold = [
        name for name in aggregation_candidates
        if f'{name}_per_fold' in detector._uncertainties
    ]

    # Build calibration-based normalization maps to avoid data leakage.
    # Expected structure in calibration_zscore_stats:
    #   calibration_zscore_stats[method]['ensemble'] -> {'mean': float, 'std': float}
    #   calibration_zscore_stats[method]['per_fold'] -> list[{fold, mean, std}]
    calib_means_ensemble = {}
    calib_stds_ensemble = {}
    calib_means_per_fold = {}
    calib_stds_per_fold = {}

    for method_name, method_stats in calibration_zscore_stats.items():
        ensemble_stats = method_stats.get('ensemble')
        if isinstance(ensemble_stats, dict) and 'mean' in ensemble_stats and 'std' in ensemble_stats:
            calib_means_ensemble[method_name] = float(ensemble_stats['mean'])
            calib_stds_ensemble[method_name] = float(ensemble_stats['std'])

        per_fold_stats = method_stats.get('per_fold')
        if isinstance(per_fold_stats, list) and len(per_fold_stats) > 0:
            sorted_folds = sorted(per_fold_stats, key=lambda x: int(x.get('fold', 0)))
            calib_means_per_fold[method_name] = [float(x['mean']) for x in sorted_folds]
            calib_stds_per_fold[method_name] = [float(x['std']) for x in sorted_folds]

    aggregation_methods_ensemble_calib = [
        name for name in aggregation_methods_ensemble
        if name in calib_means_ensemble and name in calib_stds_ensemble
    ]
    aggregation_methods_per_fold_calib = [
        name for name in aggregation_methods_per_fold
        if name in calib_means_per_fold and name in calib_stds_per_fold
    ]

    if 'ZScore_Aggregation_per_fold' in methods:
        if not per_fold_eval:
            print("\nNote: Skipping ZScore_Aggregation_per_fold (requires --per-fold-eval)")
        elif len(aggregation_methods_per_fold_calib) < 2:
            print("\nNote: Skipping ZScore_Aggregation_per_fold (need at least 2 methods with calibration mean/std)")
        else:
            print("\nRunning ZScore_Aggregation_per_fold...")
            print(f" Sources: {aggregation_methods_per_fold_calib}")
            _, agg_pf_metrics = detector.run_zscore_aggregation_per_fold(
                method_names=aggregation_methods_per_fold_calib,
                means=calib_means_per_fold,
                stds=calib_stds_per_fold,
                use_test_distribution=False,
                aggregation_name='ZScore_Aggregation_per_fold'
            )
            results['ZScore_Aggregation_per_fold'] = agg_pf_metrics
            if 'auroc_f_mean' in agg_pf_metrics:
                print(
                    f"  AUROC: {agg_pf_metrics['auroc_f_mean']:.4f}±{agg_pf_metrics['auroc_f_std']:.4f}, "
                    f"AUGRC: {agg_pf_metrics['augrc_mean']:.6f}±{agg_pf_metrics['augrc_std']:.6f}"
                )
            else:
                print(f" AUROC: {agg_pf_metrics['auroc_f']:.4f}, AUGRC: {agg_pf_metrics['augrc']:.6f}")

    if 'ZScore_Aggregation_ensemble' in methods:
        if len(aggregation_methods_ensemble_calib) < 2:
            print("\nNote: Skipping ZScore_Aggregation_ensemble (need at least 2 methods with calibration mean/std)")
        else:
            print("\nRunning ZScore_Aggregation_ensemble...")
            print(f" Sources: {aggregation_methods_ensemble_calib}")
            _, agg_ens_metrics = detector.run_zscore_aggregation_ensemble(
                method_names=aggregation_methods_ensemble_calib,
                means=calib_means_ensemble,
                stds=calib_stds_ensemble,
                use_test_distribution=False,
                aggregation_name='ZScore_Aggregation_ensemble'
            )
            results['ZScore_Aggregation_ensemble'] = agg_ens_metrics
            print(f" AUROC: {agg_ens_metrics['auroc_f']:.4f}, AUGRC: {agg_ens_metrics['augrc']:.6f}")

    if calibration_zscore_stats:
        detector._results['Calibration_ZScore_Stats'] = calibration_zscore_stats
        print("\nStored calibration z-score stats (means/stds only) for later aggregation.")
    
    # ========================================================================
    # CALIB-ONLY: save stats and return early (skip full benchmark output)
    # ========================================================================
    if run_mode == 'calib_only':
        from datetime import datetime
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        setup_suffix = f"_{setup}" if setup else ""
        calib_stats_path = os.path.join(
            output_dir, f'{flag}_{model_backbone}{setup_suffix}_calib_zscore_stats_{timestamp}.json'
        )
        with open(calib_stats_path, 'w') as f:
            json.dump(calibration_zscore_stats, f, indent=2)
        print(f"\nCalib-only mode: Z-score stats saved to {calib_stats_path}")
        return {}
    
    # ========================================================================
    # SAVE RESULTS AND FIGURES (via FailureDetector)
    # ========================================================================
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build corruption info string for filenames
    corruption_info = None
    if corruption_severity > 0:
        corruption_parts = [f"severity{corruption_severity}"]
        if corrupt_test:
            corruption_parts.append("test")
        if corrupt_calib:
            corruption_parts.append("calib")
        corruption_info = "_".join(corruption_parts)
    
    # Override output directory for new class shift
    if new_class_shift:
        output_dir = os.path.join(output_dir, 'new_class_shifts')
        os.makedirs(output_dir, exist_ok=True)
    
    # Save all results using the detector's save_results method
    saved_paths = detector.save_results(
        output_dir=output_dir,
        flag=flag,
        timestamp=timestamp,
        model_backbone=model_backbone,
        setup=setup,
        corruption_info=corruption_info
    )

    # For new_class_shift: append baseline predictor correct/incorrect indices to the NPZ so
    # that recompute_zscore.py can use a shared, deterministic evaluation set for ZScore
    # aggregation without needing the separate test cache file.
    if new_class_shift and 'metrics_file' in saved_paths:
        with np.load(saved_paths['metrics_file'], allow_pickle=True) as _npf:
            _npz_data = dict(_npf)
        _npz_data['baseline_correct_idx'] = correct_idx
        _npz_data['baseline_incorrect_idx'] = incorrect_idx
        _npz_data['baseline_per_fold_correct_idx'] = np.array(per_fold_correct_idx, dtype=object)
        _npz_data['baseline_per_fold_incorrect_idx'] = np.array(per_fold_incorrect_idx, dtype=object)
        np.savez_compressed(saved_paths['metrics_file'], **_npz_data)
        print(f" Appended baseline_correct_idx / baseline_per_fold_correct_idx to NPZ "
              f"({len(correct_idx)} successes, {len(incorrect_idx)} failures).")
    
    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"{'Method':<20} {'AUROC_f':<20} {'AURC':<20} {'AUGRC':<20} {'Accuracy':<10}")
    print("-"*100)
    for method_name, method_results in results.items():
        # Check if per-fold metrics exist (mean and std)
        if 'auroc_f_mean' in method_results and 'auroc_f_std' in method_results:
            # Per-fold evaluation: show mean±std
            auroc_str = f"{method_results['auroc_f_mean']:.4f}±{method_results['auroc_f_std']:.4f}"
            aurc_str = f"{method_results['aurc_mean']:.6f}±{method_results['aurc_std']:.6f}"
            augrc_str = f"{method_results['augrc_mean']:.6f}±{method_results['augrc_std']:.6f}"
        else:
            # Single evaluation: show just the value
            auroc_str = f"{method_results['auroc_f']:.4f}"
            aurc_str = f"{method_results['aurc']:.6f}"
            augrc_str = f"{method_results['augrc']:.6f}"
        
        print(f"{method_name:<20} "
              f"{auroc_str:<20} "
              f"{aurc_str:<20} "
              f"{augrc_str:<20} "
              f"{method_results['accuracy']:<10.4f}")
    print("="*100)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark FailCatcher UQ methods on medMNIST datasets'
    )
    parser.add_argument(
        '--flag', type=str, required=True,
        choices=['breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'dermamnist-e',
                'dermamnist-e-id', 'dermamnist-e-external', 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'amos2022', 'midog'],
        help='MedMNIST dataset to benchmark. For dermamnist-e, use -id for ID centers or -external for OOD center'
    )
    
    parser.add_argument(
        '--model', type=str, default='resnet18', choices=['resnet18', 'vit_b_16'],
        help='Model backbone to use (default: resnet18)'
    )
    
    parser.add_argument(
        '--setup', type=str,
        choices=['DA', 'DO', 'DADO'], default='',
        help='Load models trained under different setups (DA: data augmentation, DO: dropout, DADO: both). Default is standard training.'
    )
    
    parser.add_argument(
        '--methods', nargs='+',
        default=['MSR', 'MSR_calibrated', 'MLS', 'Ensembling', 'TTA', 'GPS', 'KNN_Raw', 'KNN_SHAP', 'MCDropout'],
        choices=['MSR', 'MSR_calibrated', 'MLS', 'Ensembling', 'TTA', 'GPS', 'TTA_calib', 'KNN_Raw', 'KNN_SHAP', 'MCDropout', 'ZScore_Aggregation_per_fold', 'ZScore_Aggregation_ensemble'],
        help='UQ methods to run (MCDropout runs last to avoid interference with other methods)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./Benchmarks/medMNIST/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4000,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU device ID to use (default: 0)'
    )
    parser.add_argument(
        '--per-fold-eval', action='store_true', default=False,
        help='Use per-fold evaluation (mean±std). If not set, uses ensemble-based evaluation (default: False for backward compatibility)'
    )
    parser.add_argument(
        '--ensemble-eval', dest='per_fold_eval', action='store_false',
        help='Use ensemble-based evaluation (legacy mode)'
    )
    parser.add_argument(
        '--gps-calib-samples', type=int, default=None,
        help='Maximum number of calibration samples for GPS augmentation caching (default: None = use all). Specify a number to subsample (e.g., 2000, 3000).'
    )
    parser.add_argument(
        '--min-failure-ratio', type=float, default=0.3,
        help='Minimum target proportion of failures in GPS calibration subsampling (default: 0.3 = 30%%). Will keep all available failures if less than this ratio.'
    )
    
    # Covariate shift / corruption arguments
    parser.add_argument(
        '--corruption-severity', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
        help='Apply random covariate shift corruptions. 0=disabled (clean), 1=mild to 5=severe (default: 0)'
    )
    parser.add_argument(
        '--corrupt-test', action='store_true', default=False,
        help='Apply corruption to test set (requires --corruption-severity > 0)'
    )
    parser.add_argument(
        '--corrupt-calib', action='store_true', default=False,
        help='Apply corruption to calibration set (requires --corruption-severity > 0)'
    )
    parser.add_argument(
        '--list-corruptions', action='store_true',
        help='List available corruptions for the specified dataset and exit'
    )
    parser.add_argument(
        '--new-class-shift', action='store_true', default=False,
        help='Evaluate new class shift (AMOS and MIDOG only): Create artificial test sets with new/OOD classes + ALL known-class samples. Success/failure computed dynamically per method.'
    )
    parser.add_argument(
        '--concurrent-processes', type=int, default=3,
        help='Number of benchmark processes running simultaneously on this host (used to throttle CPU threads/workers per process).'
    )
    parser.add_argument(
        '--max-loader-workers', type=int, default=48,
        help='Hard cap for DataLoader workers per process (default: 48).'
    )
    parser.add_argument(
        '--run-mode', type=str, choices=['both', 'calib_only', 'test_only'], default='both',
        help="Execution mode: 'both' (default) runs calib+test; 'calib_only' only runs "
             "calib_detector to collect z-score normalization stats (saves JSON, skips test); "
             "'test_only' skips calib_detector and only evaluates on the test set."
    )
    
    args = parser.parse_args()
    
    # Handle --list-corruptions flag
    if args.list_corruptions:
        print(f"\nAvailable corruptions for {args.flag}:")
        corruptions = dataset_utils.list_available_corruptions(args.flag)
        if corruptions:
            print(f" Random corruptions will be applied from this pool:")
            for c in sorted(corruptions):
                print(f" - {c}")
            print(f"\n  Usage example (random corruptions):")
            print(f" python run_medmnist_benchmark.py --flag {args.flag} --corruption-severity 3 --corrupt-test")
            print(f"\n  Each sample gets a random corruption from the pool at the specified severity.")
        else:
            print(f" No corruptions available for {args.flag}")
            if not dataset_utils.MEDMNISTC_AVAILABLE:
                print(f" (medmnistc is not installed - run: pip install medmnistc)")
        sys.exit(0)
    
    run_medmnist_benchmark(
        flag=args.flag,
        methods=args.methods,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gpu_id=args.gpu,
        per_fold_eval=args.per_fold_eval,
        model_backbone=args.model,
        setup=args.setup,
        gps_calib_samples=args.gps_calib_samples,
        min_failure_ratio=args.min_failure_ratio,
        corruption_severity=args.corruption_severity,
        corrupt_test=args.corrupt_test,
        corrupt_calib=args.corrupt_calib,
        new_class_shift=args.new_class_shift,
        concurrent_processes=args.concurrent_processes,
        max_loader_workers=args.max_loader_workers,
        run_mode=args.run_mode
    )
