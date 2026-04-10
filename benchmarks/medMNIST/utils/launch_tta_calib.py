#!/usr/bin/env python3
"""
Launcher script to run TTA_calib for all model backbones and setups.

This script runs the augmentation calibration caching step for GPS preprocessing
across different model configurations (backbones and training setups).

Usage:
    # Run on GPU 0 for breastmnist
    python launch_tta_calib.py --flag breastmnist --gpu 0
    
    # Run for specific models/setups
    python launch_tta_calib.py --flag breastmnist --gpu 1 --models resnet18 --setups DA DADO
    
    # Dry run to see what would be executed
    python launch_tta_calib.py --flag breastmnist --dry-run
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def run_tta_calib(flag, model, setup, gpu_id, batch_size=4000, gps_calib_samples=None, 
                  min_failure_ratio=0.3, dry_run=False):
    """
    Run TTA_calib for a specific model configuration.
    
    Args:
        flag: Dataset name
        model: Model backbone ('resnet18' or 'vit_b_16')
        setup: Training setup ('', 'DA', 'DO', 'DADO')
        gpu_id: GPU device ID
        batch_size: Batch size for inference
        gps_calib_samples: Max calibration samples for GPS (None = use all)
        min_failure_ratio: Minimum target proportion of failures (default: 0.3)
        dry_run: If True, only print the command without executing
    """
    setup_str = setup if setup else "standard"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        "run_medmnist_benchmark.py",
        "--flag", flag,
        "--model", model,
        "--methods", "TTA_calib",
        "--batch-size", str(batch_size),
        "--gpu", str(gpu_id),
    ]
    
    # Only add --gps-calib-samples if explicitly set
    if gps_calib_samples is not None:
        cmd.extend(["--gps-calib-samples", str(gps_calib_samples)])
    
    # Add min-failure-ratio
    cmd.extend(["--min-failure-ratio", str(min_failure_ratio)])
    
    # Add setup if not standard
    if setup:
        cmd.extend(["--setup", setup])
    
    # Keep environment clean (no need for CUDA_VISIBLE_DEVICES since we pass --gpu)
    env = os.environ.copy()
    
    # Print info
    config_name = f"{flag}_{model}_{setup if setup else 'standard'}"
    print(f"\n{'='*80}")
    print(f"[{timestamp}] Running TTA_calib: {config_name}")
    print(f"GPU: {gpu_id}, Batch size: {batch_size}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    if dry_run:
        print(" [DRY RUN - not executing]\n")
        return 0
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print(f"\n Completed: {config_name}\n")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n Failed: {config_name} (exit code: {e.returncode})\n")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Launch TTA_calib for multiple model configurations'
    )
    
    parser.add_argument(
        '--flag', type=str, required=True,
        choices=['breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'dermamnist-e',
                'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist', 'amos2022'],
        help='MedMNIST dataset'
    )
    
    parser.add_argument(
        '--models', nargs='+',
        default=['resnet18', 'vit_b_16'],
        choices=['resnet18', 'vit_b_16'],
        help='Model backbones to run (default: all)'
    )
    
    parser.add_argument(
        '--setups', nargs='+',
        default=['', 'DA', 'DO', 'DADO'],
        choices=['', 'DA', 'DO', 'DADO'],
        help='Training setups to run (default: all). Empty string is standard training.'
    )
    
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU device ID to use (default: 0)'
    )
    
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='Batch size for augmentation inference (default: 256, conservative for memory safety)'
    )
    
    parser.add_argument(
        '--gps-calib-samples', type=int, default=None,
        help='Maximum calibration samples for GPS augmentation (default: None = use all). Use 2000-3000 to subsample large datasets.'
    )
    
    parser.add_argument(
        '--min-failure-ratio', type=float, default=0.3,
        help='Minimum target proportion of failures in GPS calibration (default: 0.3 = 30%%). Actual ratio may be lower if not enough failures.'
    )
    
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print commands without executing them'
    )
    
    parser.add_argument(
        '--continue-on-error', action='store_true',
        help='Continue running remaining configurations if one fails'
    )
    
    args = parser.parse_args()
    
    # Build list of all configurations to run
    configs = []
    for model in args.models:
        for setup in args.setups:
            configs.append((model, setup))
    
    total_configs = len(configs)
    print(f"\n{'='*80}")
    print(f"TTA_calib Launcher")
    print(f"{'='*80}")
    print(f"Dataset: {args.flag}")
    print(f"GPU: {args.gpu}")
    print(f"Configurations to run: {total_configs}")
    print(f"{'='*80}\n")
    
    # Display all configurations
    print("Configurations:")
    for i, (model, setup) in enumerate(configs, 1):
        setup_str = setup if setup else "standard"
        print(f" {i}. {model:12s} - {setup_str}")
    print()
    
    if args.dry_run:
        print("DRY RUN MODE - Commands will be printed but not executed\n")
    
    # Run each configuration
    results = []
    for i, (model, setup) in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"# Configuration {i}/{total_configs}")
        print(f"{'#'*80}")
        
        returncode = run_tta_calib(
            flag=args.flag,
            model=model,
            setup=setup,
            gpu_id=args.gpu,
            batch_size=args.batch_size,
            gps_calib_samples=args.gps_calib_samples,
            min_failure_ratio=args.min_failure_ratio,
            dry_run=args.dry_run
        )
        
        results.append((model, setup, returncode))
        
        # Stop on error unless continue-on-error is set
        if returncode != 0 and not args.continue_on_error and not args.dry_run:
            print(f"\n Stopped due to error in configuration {i}/{total_configs}")
            break
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Setup':<10} {'Status':<10}")
    print(f"{'-'*80}")
    
    success_count = 0
    for model, setup, returncode in results:
        setup_str = setup if setup else "standard"
        status = " Success" if returncode == 0 else f" Failed ({returncode})"
        if returncode == 0:
            success_count += 1
        print(f"{model:<15} {setup_str:<10} {status:<10}")
    
    print(f"{'-'*80}")
    print(f"Completed: {success_count}/{len(results)}")
    print(f"{'='*80}\n")
    
    # Exit with error if any failed
    if success_count < len(results) and not args.dry_run:
        sys.exit(1)


if __name__ == '__main__':
    main()
