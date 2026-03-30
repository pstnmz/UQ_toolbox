#!/home/psteinmetz/venvs/venv_medMNIST/bin/python3.12
"""
Comprehensive Model Evaluation Script

Evaluates all trained models (standard, DA, DO, DADO) on test data and saves
detailed metrics for each fold AND for ensembling:
- accuracy
- balanced accuracy  
- AUC (one-vs-rest for multiclass)
- Oracle AURC (theoretical best selective prediction)
- Oracle AUGRC (theoretical best CSF)

Oracle metrics represent perfect confidence score function performance
and are computed as:
- Oracle AURC = 0.5 * (1 - accuracy)^2
- Oracle AUGRC = 0.5 * (1 - accuracy)^2

For reproducibility, uses the evaluate_model function from train_models_load_datasets.py

Handles these datasets:
- dermamnist-e-id
- dermamnist-e-ood  
- amos22 (AMOS-2022 external test using organamnist models)
- organamnist
- breastmnist
- pneumoniamnist
- bloodmnist
- tissuemnist
- octmnist
- pathmnist
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


def evaluate_single_fold(model, test_loader, data_flag, device):
    """
    Evaluate a single model fold using the standard evaluate_model function.
    
    Returns:
        dict: Metrics from evaluate_model + oracle values
    """
    result = evaluate_model(
        model=model,
        test_loader=test_loader,
        data_flag=data_flag,
        device=device,
        output_dir=None,
        prefix="fold",
        display_cm=False
    )
    
    # Add oracle metrics
    accuracy = result['metrics']['accuracy']
    result['metrics']['oracle_aurc'] = float(compute_oracle_aurc(accuracy))
    result['metrics']['oracle_augrc'] = float(compute_oracle_augrc(accuracy))
    
    return result


def evaluate_ensemble_models(models, test_loader, data_flag, device):
    """
    Evaluate ensemble of models using the standard evaluate_model function.
    
    Returns:
        dict: Metrics from evaluate_model + oracle values
    """
    result = evaluate_model(
        model=models,  # Pass list of models for ensemble
        test_loader=test_loader,
        data_flag=data_flag,
        device=device,
        output_dir=None,
        prefix="ensemble",
        display_cm=False
    )
    
    # Add oracle metrics
    accuracy = result['metrics']['accuracy']
    result['metrics']['oracle_aurc'] = float(compute_oracle_aurc(accuracy))
    result['metrics']['oracle_augrc'] = float(compute_oracle_augrc(accuracy))
    
    return result


def evaluate_dataset_setup(dataset, model_name, setup, device, output_dir):
    """
    Evaluate all models for a specific dataset and setup.
    
    Args:
        dataset: Dataset name (e.g., 'breastmnist', 'dermamnist-e-id')
        model_name: 'resnet18' or 'vit_b_16'
        setup: '' (standard), 'DA', 'DO', or 'DADO'
        device: torch device
        output_dir: Where to save results
    
    Returns:
        dict: Results with per-fold and ensemble metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {dataset} | {model_name} | Setup: {setup or 'standard'}")
    print(f"{'='*80}")
    
    # Handle special dataset mappings
    if dataset == 'dermamnist-e-id':
        load_flag = 'dermamnist-e'
        test_subset = 'id'
    elif dataset == 'dermamnist-e-ood':
        load_flag = 'dermamnist-e'
        test_subset = 'external'
    elif dataset == 'amos22':
        load_flag = 'organamnist'  # Use organamnist models
        test_subset = 'amos'
    else:
        load_flag = dataset
        test_subset = 'all'
    
    # Get dataset info
    try:
        info = INFO[load_flag]
        num_classes = len(info['label'])
        task = info['task']
    except KeyError:
        print(f"  Error: Dataset {load_flag} not found in INFO")
        return None
    
    # Determine if color or grayscale
    color = load_flag in ['dermamnist', 'dermamnist-e', 'pathmnist', 'bloodmnist']
    
    # Load models
    try:
        print(f"\n  Loading models...")
        models = load_models(
            flag=load_flag,
            device=device,
            size=224,
            model_backbone=model_name,
            setup=setup
        )
        print(f"  ✓ Loaded {len(models)} models")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return None
    
    # Load test data
    print(f"  Loading test data (subset: {test_subset})...")
    
    # Prepare transform
    if color:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
    else:
        # Grayscale - need to repeat to 3 channels
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
        ])
    
    try:
        if test_subset == 'amos':
            # Load AMOS-2022 external test dataset
            transform_tta = transforms.Compose([transforms.ToTensor()])
            test_dataset, test_loader, _, _, _ = dataset_utils.load_amos_dataset(
                transform=transform,
                transform_tta=transform_tta,
                batch_size=3500,
                workspace_root=repo_root
            )
            print(f"  ✓ Loaded AMOS test data: {len(test_loader.dataset)} samples")
        else:
            datasets_list, dataloaders, _ = load_datasets(
                dataflag=load_flag,
                color=color,
                im_size=224,
                transform=transform,
                batch_size=3500,
                cache_test=True,
                test_subset=test_subset
            )
            _, _, test_loader = dataloaders
            print(f"  ✓ Loaded test data: {len(test_loader.dataset)} samples")
    except Exception as e:
        print(f"  Error loading test data: {e}")
        return None
    
    # Evaluate per-fold
    print(f"\n  Evaluating per-fold models...")
    per_fold_results = []
    
    for fold_idx, model in enumerate(models):
        print(f"    Fold {fold_idx}...", end=" ")
        
        result = evaluate_single_fold(model, test_loader, load_flag, device)
        metrics = result['metrics']
        
        per_fold_results.append(metrics)
        print(f"acc={metrics['accuracy']:.4f}, oracle_augrc={metrics['oracle_augrc']:.6f}")
    
    # Evaluate ensemble
    print(f"\n  Evaluating ensemble...")
    ensemble_result = evaluate_ensemble_models(models, test_loader, load_flag, device)
    ensemble_metrics = ensemble_result['metrics']
    print(f"    Ensemble: acc={ensemble_metrics['accuracy']:.4f}, oracle_augrc={ensemble_metrics['oracle_augrc']:.6f}")
    
    # Compile results
    results = {
        'dataset': dataset,
        'model': model_name,
        'setup': setup or 'standard',
        'num_classes': num_classes,
        'task': task,
        'per_fold_metrics': per_fold_results,
        'ensemble_metrics': ensemble_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results to appropriate folder
    if dataset == 'amos22':
        # Save to population_shift folder for AMOS
        save_dir = output_dir.parent / 'population_shift'
        save_dir.mkdir(exist_ok=True)
        
        # Also cache predictions
        cache_dir = repo_root / 'uq_benchmark_results' / 'cache'
        cache_dir.mkdir(exist_ok=True, parents=True)
    else:
        # Save to in_distribution folder for regular datasets
        save_dir = output_dir / 'in_distribution'
        save_dir.mkdir(exist_ok=True)
    
    setup_suffix = f"_{setup}" if setup else "_standard"
    filename = f"comprehensive_metrics_{dataset}_{model_name}{setup_suffix}.json"
    output_path = save_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  ✓ Saved results to: {output_path}")
    
    return results


def main():
    """Run comprehensive evaluation for all datasets and setups."""
    
    # Configuration
    datasets = [
        # 'dermamnist-e-id',
        # 'dermamnist-e-ood',
        # 'organamnist',
        # 'breastmnist',
        # 'pneumoniamnist',
        # 'bloodmnist',
        # 'tissuemnist',
        # 'octmnist',
        # 'pathmnist',
        'amos22'
    ]
    
    model_names = ['resnet18', 'vit_b_16']
    setups = ['', 'DA', 'DO', 'DADO']  # Empty string = standard
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(__file__).parent / 'comprehensive_evaluation_results'
    output_dir.mkdir(exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")
    
    # Track results
    all_results = []
    failed_evaluations = []
    
    # Run evaluations
    print(f"\n{'='*80}")
    print(f"Starting comprehensive evaluation")
    print(f"{'='*80}")
    print(f"Datasets: {len(datasets)}")
    print(f"Models: {model_names}")
    print(f"Setups: {['standard' if s == '' else s for s in setups]}")
    print(f"Total evaluations: {len(datasets) * len(model_names) * len(setups)}")
    
    for dataset in datasets:
        for model_name in model_names:
            for setup in setups:
                try:
                    result = evaluate_dataset_setup(
                        dataset=dataset,
                        model_name=model_name,
                        setup=setup,
                        device=device,
                        output_dir=output_dir
                    )
                    
                    if result is not None:
                        all_results.append(result)
                    else:
                        failed_evaluations.append((dataset, model_name, setup))
                
                except Exception as e:
                    print(f"\n  ✗ FAILED: {dataset} | {model_name} | {setup}")
                    print(f"    Error: {e}")
                    failed_evaluations.append((dataset, model_name, setup))
                    continue
    
    # Summary
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful evaluations: {len(all_results)}")
    print(f"Failed evaluations: {len(failed_evaluations)}")
    
    if failed_evaluations:
        print(f"\nFailed:")
        for dataset, model, setup in failed_evaluations:
            print(f"  - {dataset} | {model} | {setup or 'standard'}")
    
    # Create summary file
    summary = {
        'total_evaluations': len(all_results),
        'failed_evaluations': len(failed_evaluations),
        'failed_list': [
            {'dataset': d, 'model': m, 'setup': s or 'standard'}
            for d, m, s in failed_evaluations
        ],
        'results_dir': str(output_dir),
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_path}")
    print(f"✓ All results saved in: {output_dir}")


if __name__ == '__main__':
    main()
