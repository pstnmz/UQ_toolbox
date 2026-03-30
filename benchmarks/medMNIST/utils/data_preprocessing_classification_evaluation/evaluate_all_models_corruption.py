#!/home/psteinmetz/venvs/venv_medMNIST/bin/python3.12
"""
Comprehensive Model Evaluation Script with MedMNIST-C Corruptions

Evaluates all trained models (standard, DA, DO, DADO) on corrupted test data
and saves detailed metrics for each fold AND for ensembling:
- accuracy
- balanced accuracy  
- AUC (one-vs-rest for multiclass)
- Oracle AURC (theoretical best selective prediction)
- Oracle AUGRC (theoretical best CSF)

This script applies random medmnist-C corruptions with varying severity levels
to evaluate model robustness under covariate shift.

Oracle metrics represent perfect confidence score function performance
and are computed as:
- Oracle AURC = 0.5 * (1 - accuracy)^2
- Oracle AUGRC = 0.5 * (1 - accuracy)^2

For reproducibility, uses the evaluate_model function from train_models_load_datasets.py
and applies deterministic corruptions with seed=42.

Caches corrupted predictions in uq_benchmark_results/cache/ to avoid redundant computation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
import numpy as np
import json
from datetime import datetime
from torchvision import transforms

# Import from UQ_Toolbox
from benchmarks.medMNIST.utils.train_models_load_datasets import (
    load_models, load_datasets, INFO, evaluate_model
)
from benchmarks.medMNIST.utils import dataset_utils


def compute_oracle_augrc(accuracy):
    """Compute oracle (theoretical best) AUGRC given accuracy."""
    return 0.5 * (1.0 - accuracy) ** 2


def compute_oracle_aurc(accuracy):
    """
    Compute oracle (theoretical best) AURC.
    
    Oracle assumes perfect selective prediction:
    reject all errors first, then accept all correct predictions.
    """
    error_rate = 1.0 - accuracy
    return 0.5 * error_rate ** 2


def evaluate_models_with_cache(models, test_loader, data_flag, device, cache_path):
    """
    Evaluate models and use cache for predictions to avoid redundant inference.
    
    Args:
        models: List of trained models
        test_loader: DataLoader for test data
        data_flag: Dataset name
        device: Device to run inference on
        cache_path: Path to cache file
    
    Returns:
        dict: Contains 'ensemble' and 'per_fold' results
    """
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"  ✓ Loading predictions from cache: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        
        y_true = cache['y_true']
        y_scores = cache['y_scores']
        y_pred = cache['y_pred']
        correct_idx = cache['correct_idx']
        incorrect_idx = cache['incorrect_idx']
        indiv_scores = cache['indiv_scores']  # [N, K, C]
        
        # Compute metrics from cached data
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        
        # Ensemble metrics
        ensemble_acc = len(correct_idx) / len(y_true)
        ensemble_balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # AUC (one-vs-rest for multiclass)
        num_classes = y_scores.shape[1]
        if num_classes == 2:
            ensemble_auc = roc_auc_score(y_true, y_scores[:, 1])
        else:
            ensemble_auc = roc_auc_score(y_true, y_scores, multi_class='ovr')
        
        ensemble_metrics = {
            'accuracy': float(ensemble_acc),
            'balanced_accuracy': float(ensemble_balanced_acc),
            'auc': float(ensemble_auc),
            'oracle_aurc': float(compute_oracle_aurc(ensemble_acc)),
            'oracle_augrc': float(compute_oracle_augrc(ensemble_acc)),
            'n_correct': int(len(correct_idx)),
            'n_incorrect': int(len(incorrect_idx)),
            'n_total': int(len(y_true))
        }
        
        # Per-fold metrics
        per_fold_metrics = []
        print(f"  🔍 Per-fold predictions diversity (from cache):")
        for fold_idx in range(indiv_scores.shape[1]):
            fold_scores = indiv_scores[:, fold_idx, :]  # [N, C]
            fold_pred = np.argmax(fold_scores, axis=1)
            fold_correct = (fold_pred == y_true)
            fold_acc = fold_correct.sum() / len(y_true)
            fold_balanced_acc = balanced_accuracy_score(y_true, fold_pred)
            
            # Debug: Check prediction distribution
            unique_preds, pred_counts = np.unique(fold_pred, return_counts=True)
            print(f"    Fold {fold_idx}: acc={fold_acc:.4f}, pred_distribution={dict(zip(unique_preds, pred_counts))}")
            if fold_idx == 0:  # Show sample scores for first fold
                print(f"      Sample scores [0]: {fold_scores[0]}")
                print(f"      Sample scores [1]: {fold_scores[1]}")
                print(f"      Score mean per class: {fold_scores.mean(axis=0)}")
            
            # AUC
            if num_classes == 2:
                fold_auc = roc_auc_score(y_true, fold_scores[:, 1])
            else:
                fold_auc = roc_auc_score(y_true, fold_scores, multi_class='ovr')
            
            fold_metrics = {
                'fold': fold_idx,
                'accuracy': float(fold_acc),
                'balanced_accuracy': float(fold_balanced_acc),
                'auc': float(fold_auc),
                'oracle_aurc': float(compute_oracle_aurc(fold_acc)),
                'oracle_augrc': float(compute_oracle_augrc(fold_acc)),
                'n_correct': int(fold_correct.sum()),
                'n_incorrect': int((~fold_correct).sum()),
                'n_total': int(len(y_true))
            }
            per_fold_metrics.append(fold_metrics)
        
        return {
            'ensemble': ensemble_metrics,
            'per_fold': per_fold_metrics
        }
    
    # No cache - evaluate models and create cache
    print(f"  ⚙️  Evaluating models (no cache found)...")
    
    # Use evaluate_model for ensemble (already handles binary/multi-class, ensemble averaging, etc.)
    print(f"  📊 Evaluating ensemble...")
    ensemble_result = evaluate_model(
        model=models,  # Pass list for ensemble
        test_loader=test_loader,
        data_flag=data_flag,
        device=device,
        output_dir=None,  # Don't save figures/logs
        display_cm=False
    )
    
    # Extract ensemble metrics
    ensemble_acc = ensemble_result['metrics']['accuracy']
    ensemble_balanced_acc = ensemble_result['metrics']['balanced_accuracy']
    ensemble_auc = ensemble_result['metrics']['auc']
    
    ensemble_metrics = {
        'accuracy': float(ensemble_acc),
        'balanced_accuracy': float(ensemble_balanced_acc),
        'auc': float(ensemble_auc),
        'oracle_aurc': float(compute_oracle_aurc(ensemble_acc)),
        'oracle_augrc': float(compute_oracle_augrc(ensemble_acc))
    }
    
    # Evaluate each fold individually to get per-fold metrics AND predictions
    print(f"  📊 Evaluating individual folds...")
    per_fold_metrics = []
    
    # We need to collect predictions manually for caching purposes
    # (evaluate_model doesn't return predictions, only metrics)
    from torch.nn.functional import sigmoid, softmax
    
    num_classes = ensemble_result['num_classes']
    is_binary = (num_classes == 2)
    
    y_true_list = []
    y_scores_ensemble_list = []
    indiv_scores_list = []  # [N, K, C] - all fold predictions
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                labels = batch["label"].squeeze().long()
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.squeeze().long()
            
            # Get predictions from each fold
            batch_indiv_scores = []
            for model in models:
                model.eval()
                logits = model(images)
                
                if is_binary:
                    p = sigmoid(logits).view(-1, 1)  # (B, 1)
                    # Convert to 2-class probabilities: [P(class=0), P(class=1)]
                    probs_class1 = p.squeeze(1)
                    probs_class0 = 1 - probs_class1
                    scores = torch.stack([probs_class0, probs_class1], dim=1)  # [B, 2]
                else:
                    scores = softmax(logits, dim=1)  # (B, C)
                
                batch_indiv_scores.append(scores.cpu().numpy())
            
            # Stack: [B, K, C]
            batch_indiv_scores = np.stack(batch_indiv_scores, axis=1)
            batch_scores_ensemble = batch_indiv_scores.mean(axis=1)  # [B, C]
            
            y_true_list.append(labels.cpu().numpy())
            y_scores_ensemble_list.append(batch_scores_ensemble)
            indiv_scores_list.append(batch_indiv_scores)
    
    # Concatenate
    y_true = np.concatenate(y_true_list)
    y_scores = np.concatenate(y_scores_ensemble_list)
    indiv_scores = np.concatenate(indiv_scores_list, axis=0)  # [N, K, C]
    
    # Compute ensemble predictions
    y_pred = np.argmax(y_scores, axis=1)
    correct_idx = np.where(y_pred == y_true)[0]
    incorrect_idx = np.where(y_pred != y_true)[0]
    
    # Update ensemble metrics with counts
    ensemble_metrics.update({
        'n_correct': int(len(correct_idx)),
        'n_incorrect': int(len(incorrect_idx)),
        'n_total': int(len(y_true))
    })
    
    # Per-fold metrics using evaluate_model
    for fold_idx, model in enumerate(models):
        fold_result = evaluate_model(
            model=model,
            test_loader=test_loader,
            data_flag=data_flag,
            device=device,
            output_dir=None,
            display_cm=False
        )
        
        fold_acc = fold_result['metrics']['accuracy']
        fold_balanced_acc = fold_result['metrics']['balanced_accuracy']
        fold_auc = fold_result['metrics']['auc']
        
        fold_metrics = {
            'fold': fold_idx,
            'accuracy': float(fold_acc),
            'balanced_accuracy': float(fold_balanced_acc),
            'auc': float(fold_auc),
            'oracle_aurc': float(compute_oracle_aurc(fold_acc)),
            'oracle_augrc': float(compute_oracle_augrc(fold_acc)),
            'n_total': int(len(y_true))
        }
        per_fold_metrics.append(fold_metrics)
    
    # Save predictions to cache
    print(f"  💾 Saving predictions to cache: {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        indiv_scores=indiv_scores,
        correct_idx=correct_idx,
        incorrect_idx=incorrect_idx
    )
    
    return {
        'ensemble': ensemble_metrics,
        'per_fold': per_fold_metrics
    }


def evaluate_dataset_with_corruption(dataset_name, model_backbone, setup, corruption_severity, 
                                     output_dir, cache_dir, device, batch_size=128):
    """
    Evaluate models on corrupted test data.
    
    Args:
        dataset_name: Dataset name (e.g., 'breastmnist')
        model_backbone: Model architecture ('resnet18' or 'vit_b_16')
        setup: Training setup ('', 'DA', 'DO', 'DADO')
        corruption_severity: Corruption severity (1-5)
        output_dir: Directory to save results
        cache_dir: Directory to cache predictions
        device: Device to run inference on
        batch_size: Batch size for evaluation
    
    Returns:
        dict: Evaluation results
    """
    # Setup string
    setup_str = setup if setup else 'standard'
    
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name} | Model: {model_backbone} | Setup: {setup_str}")
    print(f"Corruption Severity: {corruption_severity}/5")
    print(f"{'='*80}")
    
    # Load models
    try:
        if dataset_name == 'dermamnist-e-id':
            dataset_name = 'dermamnist-e'
        models = load_models(dataset_name, device=device, size=224, 
                           model_backbone=model_backbone, setup=setup)
        print(f"  ✓ Loaded {len(models)} models")
    except Exception as e:
        print(f"  ✗ Failed to load models: {e}")
        return None
    
    # Load dataset
    color = dataset_name in ['dermamnist', 'dermamnist-e', 'pathmnist', 'bloodmnist']
    
    # Get transforms
    transform_test, _ = dataset_utils.get_transforms(color, image_size=224)
    
    # Load clean dataset
    try:
        [_, _, test_dataset], [_, _, test_loader], info = load_datasets(
            dataset_name, color, im_size=224, 
            transform=transform_test, batch_size=batch_size
        )
        print(f"  ✓ Loaded test dataset: {len(test_dataset)} samples")
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        return None
    if 'dermamnist' in dataset_name:
        dataset_name_for_corruption = 'dermamnist'
    else:
        dataset_name_for_corruption = dataset_name
    # Apply corruption
    print(f"  🔬 Applying random corruptions (severity={corruption_severity})...")
    test_dataset_corrupted = dataset_utils.apply_random_corruptions(
        test_dataset, dataset_name_for_corruption, corruption_severity, 
        cache=True, seed=42
    )
    
    # Create corrupted test loader
    # Optimizations:
    # - persistent_workers=True: Keep workers alive between epochs (faster)
    # - prefetch_factor=2: Workers load batches ahead of time
    # - num_workers=8: More parallel loading (safe now with rng fix)
    test_loader_corrupted = torch.utils.data.DataLoader(
        test_dataset_corrupted, batch_size=batch_size, 
        shuffle=False, num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=3
    )
    
    # Cache path
    setup_suffix = f"_{setup}" if setup else ""
    cache_filename = f"{dataset_name}_{model_backbone}{setup_suffix}_corrupt{corruption_severity}_test_results.npz"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Evaluate with cache
    results = evaluate_models_with_cache(
        models, test_loader_corrupted, dataset_name, device, cache_path
    )
    
    if results is None:
        return None
    
    # Add metadata
    results['metadata'] = {
        'dataset': dataset_name,
        'model': model_backbone,
        'setup': setup_str,
        'corruption_severity': corruption_severity,
        'image_size': 224,
        'num_folds': len(models),
        'num_classes': len(info['label']),
        'task': info['task'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Print summary
    print(f"\n{'─'*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'─'*80}")
    print(f"Ensemble Accuracy: {results['ensemble']['accuracy']:.4f}")
    print(f"Ensemble Oracle AUGRC: {results['ensemble']['oracle_augrc']:.6f}")
    
    # Per-fold statistics
    fold_accs = [f['accuracy'] for f in results['per_fold']]
    fold_mean = np.mean(fold_accs)
    fold_std = np.std(fold_accs)
    print(f"Per-Fold Accuracy: {fold_mean:.4f} ± {fold_std:.4f}")
    print(f"{'─'*80}\n")
    
    return results


def main():
    """Run comprehensive evaluation with corruptions on all datasets and setups."""
    
    # Configuration
    datasets = ['pathmnist'] #[
        #'breastmnist', 'organamnist', 'pneumoniamnist', 
        #'bloodmnist', 'tissuemnist', 'octmnist', 'pathmnist',
        #'dermamnist-e-id'
    #]
    
    model_backbones = ['resnet18', 'vit_b_16']
    setups = ['', 'DA', 'DO', 'DADO']  # '' = standard
    corruption_severities = [3]
    
    # Paths
    repo_root = Path(__file__).parent.parent.parent.parent
    output_base = repo_root / 'benchmarks/medMNIST/utils/comprehensive_evaluation_results/corruption_shift'
    cache_dir = repo_root / 'uq_benchmark_results/cache'
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Tracking
    all_results = {}
    total_configs = len(datasets) * len(model_backbones) * len(setups) * len(corruption_severities)
    completed = 0
    
    print(f"{'='*80}")
    print(f"COMPREHENSIVE CORRUPTION EVALUATION")
    print(f"{'='*80}")
    print(f"Total configurations: {total_configs}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Models: {len(model_backbones)}")
    print(f"  Setups: {len(setups)}")
    print(f"  Corruption Severities: {len(corruption_severities)}")
    print(f"{'='*80}\n")
    
    # Iterate through all configurations
    for dataset in datasets:
        for model in model_backbones:
            for setup in setups:
                for severity in corruption_severities:
                    completed += 1
                    config_key = f"{dataset}_{model}_{setup if setup else 'standard'}_severity{severity}"
                    
                    print(f"\n[{completed}/{total_configs}] Evaluating: {config_key}")
                    
                    try:
                        results = evaluate_dataset_with_corruption(
                            dataset_name=dataset,
                            model_backbone=model,
                            setup=setup,
                            corruption_severity=severity,
                            output_dir=output_base,
                            cache_dir=cache_dir,
                            device=device,
                            batch_size=3000
                        )
                        
                        if results is not None:
                            all_results[config_key] = results
                            
                            # Save individual result
                            setup_suffix = f"_{setup}" if setup else ""
                            result_filename = f"{dataset}_{model}{setup_suffix}_severity{severity}.json"
                            result_path = output_base / result_filename
                            
                            with open(result_path, 'w') as f:
                                json.dump(results, f, indent=2)
                            
                            print(f"  ✓ Saved: {result_filename}")
                        else:
                            print(f"  ✗ Evaluation failed")
                            
                    except Exception as e:
                        print(f"  ✗ Error: {e}")
                        import traceback
                        traceback.print_exc()
    
    # Save summary
    summary_path = output_base / 'all_results_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully evaluated: {len(all_results)}/{total_configs} configurations")
    print(f"Results saved to: {output_base}")
    print(f"Summary: {summary_path}")
    print(f"Cache directory: {cache_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
