#!/usr/bin/env python3
"""
Comprehensive benchmark launcher for medMNIST UQ methods.

Automatically runs all combinations of:
- Datasets (internal test sets and external test sets)
- Model backbones (ResNet-18, ViT-B/16)
- Training setups (standard, DA, DO, DADO)
- UQ methods (with setup-specific filtering)

Usage:
    # Run full benchmark
    python launcher_benchmark.py --python /path/to/venv/bin/python --gpu 0
    
    # Dry run (just print commands)
    python launcher_benchmark.py --dry-run
    
    # Run specific datasets only
    python launcher_benchmark.py --datasets breastmnist organamnist
    
    # Run specific models only
    python launcher_benchmark.py --models resnet18
    
    # Run specific setups only (use "" for standard)
    python launcher_benchmark.py --setups "" DA DO
    
    # New class shift evaluation (AMOS only)
    python launcher_benchmark.py --datasets amos2022 --new-class-shift
    
    # Covariate shift evaluation
    python launcher_benchmark.py --corruption-severity 3 --corrupt-test
"""

import subprocess
import argparse
from pathlib import Path
from typing import List, Dict
import sys


# =============================================================================
# Configuration
# =============================================================================

# Dataset definitions
DATASETS_ID = [
    'organamnist', 
    'pneumoniamnist', 
    'dermamnist-e-id', 
    'octmnist',  
    'pathmnist',
    'bloodmnist', 
    'tissuemnist', 
    'breastmnist'
]

DATASETS_EXTERNAL = [
    'amos2022',  # OrganaMNIST → AMOS-2022 external test
    'dermamnist-e-external',  # DermaMNIST-E external centers,
    'hmu-crc'
    
]

# Datasets supporting new class shift evaluation
DATASETS_NEW_CLASS_SHIFT = [
    'amos2022',  # Only AMOS has new/unseen classes (9 unmapped organs)
    'midog'  # MIDOG++ external test set
]

# Model backbones
MODEL_BACKBONES = ['resnet18', 'vit_b_16']

# Training setups
TRAINING_SETUPS = ['', 'DA', 'DO', 'DADO']  # '' = standard training

# UQ methods
ALL_METHODS = [
    'MSR', 
    'MSR_calibrated', 
    'MLS',
    'Ensembling',
    'TTA', 
    'GPS', 
    'KNN_Raw', 
    #'KNN_SHAP', 
    'MCDropout',
    'ZScore_Aggregation_per_fold',
    'ZScore_Aggregation_ensemble'
]

# Setup constraints: which methods work with which setups
SETUP_METHOD_COMPATIBILITY = {
    '': [m for m in ALL_METHODS if m != 'MCDropout'],  # Standard: no dropout
    'DA': [m for m in ALL_METHODS if m != 'MCDropout'],  # DA only: no dropout
    'DO': ALL_METHODS,  # Dropout: all methods work
    'DADO': ALL_METHODS,  # DA + Dropout: all methods work
}

# Dataset-specific configurations
DATASET_CONFIG = {
    'breastmnist': {'batch_size': 4000, 'gps_subsample': None},  # Binary, small → no subsampling
    'pneumoniamnist': {'batch_size': 4000, 'gps_subsample': None},  # Binary, small → no subsampling
    'organamnist': {'batch_size': 3500, 'gps_subsample': 5000},  # 11 classes, large
    'octmnist': {'batch_size': 4000, 'gps_subsample': 5000},
    'pathmnist': {'batch_size': 3500, 'gps_subsample': 5000},
    'bloodmnist': {'batch_size': 4000, 'gps_subsample': 5000},
    'tissuemnist': {'batch_size': 3500, 'gps_subsample': 5000},
    'dermamnist-e-id': {'batch_size': 4000, 'gps_subsample': 5000},
    'dermamnist-e-external': {'batch_size': 3500, 'gps_subsample': 5000},
    'amos2022': {'batch_size': 3500, 'gps_subsample': 5000},
    'midog': {'batch_size': 3500, 'gps_subsample': 5000},
    'hmu-crc': {'batch_size': 3500, 'gps_subsample': 5000},
}


# =============================================================================
# Command Generation
# =============================================================================

def get_methods_for_setup(setup: str, exclude_methods: List[str] = None) -> List[str]:
    """
    Get compatible methods for a given training setup.
    
    Args:
        setup: Training setup ('', 'DA', 'DO', 'DADO')
        exclude_methods: Optional list of methods to exclude
    
    Returns:
        List of compatible method names
    """
    methods = SETUP_METHOD_COMPATIBILITY[setup].copy()
    
    if exclude_methods:
        methods = [m for m in methods if m not in exclude_methods]
    
    return methods


def generate_command(
    dataset: str,
    model: str,
    setup: str,
    methods: List[str],
    python_path: str,
    script_path: Path,
    gpu: int,
    per_fold_eval: bool = True,
    output_dir: str = './uq_benchmark_results',
    new_class_shift: bool = False,
    run_mode: str = 'both',
    max_loader_workers: int = 48,
    concurrent_processes: int = 1,
    **kwargs
) -> str:
    """
    Generate a benchmark command for a specific configuration.
    
    Args:
        dataset: Dataset name (e.g., 'breastmnist')
        model: Model backbone ('resnet18' or 'vit_b_16')
        setup: Training setup ('', 'DA', 'DO', 'DADO')
        methods: List of UQ methods to run
        python_path: Path to Python executable
        script_path: Path to run_medmnist_benchmark.py
        gpu: GPU device ID
        per_fold_eval: Use per-fold evaluation
        output_dir: Output directory for results
        new_class_shift: Enable new class shift evaluation (only for amos2022 and midog)
        **kwargs: Additional arguments (batch_size, gps_subsample, etc.)
    
    Returns:
        Command string
    """
    # Get dataset-specific config
    config = DATASET_CONFIG.get(dataset, {'batch_size': 256, 'gps_subsample': 5000})
    batch_size = kwargs.get('batch_size', config['batch_size'])
    gps_subsample = kwargs.get('gps_subsample', config['gps_subsample'])
    
    # Build command
    cmd_parts = [
        python_path,
        str(script_path),
        f"--flag {dataset}",
        f"--model {model}",
    ]
    
    # Add setup if not standard
    if setup:
        cmd_parts.append(f"--setup {setup}")
    
    # Add methods
    methods_str = ' '.join(methods)
    cmd_parts.append(f"--methods {methods_str}")
    
    # Add other arguments
    cmd_parts.append(f"--batch-size {batch_size}")
    cmd_parts.append(f"--gpu {gpu}")
    cmd_parts.append(f"--output-dir {output_dir}")
    
    # Per-fold evaluation flag
    if per_fold_eval:
        cmd_parts.append("--per-fold-eval")
    
    # GPS subsampling (always set min-failure-ratio for reproducibility)
    if gps_subsample is not None and 'GPS' in methods:
        cmd_parts.append(f"--gps-calib-samples {gps_subsample}")
    cmd_parts.append("--min-failure-ratio 0.3")
    
    # Corruption parameters (covariate shift)
    corruption_severity = kwargs.get('corruption_severity', 0)
    if corruption_severity > 0:
        cmd_parts.append(f"--corruption-severity {corruption_severity}")
        if kwargs.get('corrupt_test', False):
            cmd_parts.append("--corrupt-test")
        if kwargs.get('corrupt_calib', False):
            cmd_parts.append("--corrupt-calib")
    
    # New class shift evaluation (only for amos2022 and midog)
    if new_class_shift and dataset in DATASETS_NEW_CLASS_SHIFT:
        cmd_parts.append("--new-class-shift")
    
    # Run mode
    if run_mode != 'both':
        cmd_parts.append(f"--run-mode {run_mode}")

    # Worker / parallelism hints
    cmd_parts.append(f"--max-loader-workers {max_loader_workers}")
    if concurrent_processes > 1:
        cmd_parts.append(f"--concurrent-processes {concurrent_processes}")

    return ' '.join(cmd_parts)


def generate_all_commands(
    datasets: List[str],
    models: List[str],
    setups: List[str],
    python_path: str,
    script_path: Path,
    gpu: int,
    exclude_methods: List[str] = None,
    per_fold_eval: bool = True,
    output_dir: str = './uq_benchmark_results',
    corruption_severity: int = 0,
    corrupt_test: bool = False,
    corrupt_calib: bool = False,
    new_class_shift: bool = False,
    run_mode: str = 'both',
    max_loader_workers: int = 48,
    concurrent_processes: int = 1
) -> List[Dict]:
    """
    Generate all benchmark commands for specified configurations.
    
    Returns:
        List of dicts with 'config' (metadata) and 'command' (command string)
    """
    commands = []
    
    for dataset in datasets:
        # Skip new_class_shift for datasets that don't support it
        if new_class_shift and dataset not in DATASETS_NEW_CLASS_SHIFT:
            print(f"[WARNING] Skipping {dataset}: new_class_shift only supported for {DATASETS_NEW_CLASS_SHIFT}")
            continue
            
        for model in models:
            for setup in setups:
                # Get compatible methods for this setup
                methods = get_methods_for_setup(setup, exclude_methods)
                
                if not methods:
                    print(f"[WARNING] Skipping {dataset}/{model}/{setup or 'standard'}: no compatible methods")
                    continue
                
                cmd = generate_command(
                    dataset=dataset,
                    model=model,
                    setup=setup,
                    methods=methods,
                    python_path=python_path,
                    script_path=script_path,
                    gpu=gpu,
                    per_fold_eval=per_fold_eval,
                    output_dir=output_dir,
                    corruption_severity=corruption_severity,
                    corrupt_test=corrupt_test,
                    corrupt_calib=corrupt_calib,
                    new_class_shift=new_class_shift,
                    run_mode=run_mode,
                    max_loader_workers=max_loader_workers,
                    concurrent_processes=concurrent_processes
                )
                
                commands.append({
                    'config': {
                        'dataset': dataset,
                        'model': model,
                        'setup': setup or 'standard',
                        'methods': methods,
                        'num_methods': len(methods),
                        'new_class_shift': new_class_shift and dataset in DATASETS_NEW_CLASS_SHIFT
                    },
                    'command': cmd
                })
    
    return commands


# =============================================================================
# Execution
# =============================================================================

def run_commands(commands: List[Dict], dry_run: bool = False, verbose: bool = True):
    """
    Execute generated commands sequentially.
    
    Args:
        commands: List of command dicts
        dry_run: If True, only print commands without executing
        verbose: Print progress information
    """
    total = len(commands)
    
    print(f"\n{'='*80}")
    print(f"Benchmark Launcher: {total} configurations to run")
    print(f"{'='*80}\n")
    
    if dry_run:
        print("DRY RUN MODE - Commands will not be executed\n")
    
    for idx, cmd_dict in enumerate(commands, 1):
        config = cmd_dict['config']
        command = cmd_dict['command']
        
        print(f"\n[{idx}/{total}] Running configuration:")
        print(f" Dataset: {config['dataset']}")
        print(f" Model: {config['model']}")
        print(f" Setup: {config['setup']}")
        print(f" Methods ({config['num_methods']}): {', '.join(config['methods'])}")
        if config.get('new_class_shift'):
            print(f" Mode: New class shift evaluation")
        print(f"\n  Command: {command}\n")
        
        if dry_run:
            print(" Skipped (dry run)\n")
            continue
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=not verbose,
                text=True
            )
            print(f" Completed successfully\n")
        except subprocess.CalledProcessError as e:
            print(f" [ERROR] Failed with exit code {e.returncode}")
            if not verbose and e.stderr:
                print(f" Error output:\n{e.stderr}")
            print(f" Continuing to next configuration...\n")
            continue
    
    print(f"\n{'='*80}")
    print(f"Benchmark complete: {total} configurations processed")
    print(f"{'='*80}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive medMNIST UQ benchmark launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Python and script paths
    parser.add_argument(
        '--python', type=str, 
        default=sys.executable,
        help='Path to Python executable (default: current Python)'
    )
    parser.add_argument(
        '--script', type=str,
        default=str(Path(__file__).parent / 'run_medmnist_benchmark.py'),
        help='Path to run_medmnist_benchmark.py script'
    )
    
    # Dataset selection
    parser.add_argument(
        '--datasets', nargs='+',
        default=DATASETS_ID + DATASETS_EXTERNAL,
        choices=DATASETS_ID + DATASETS_EXTERNAL,
        help='Datasets to benchmark (default: all)'
    )
    parser.add_argument(
        '--id-only', action='store_true',
        help='Run only internal test sets (ID datasets)'
    )
    parser.add_argument(
        '--external-only', action='store_true',
        help='Run only external test sets'
    )
    
    # Model and setup selection
    parser.add_argument(
        '--models', nargs='+',
        default=MODEL_BACKBONES,
        choices=MODEL_BACKBONES,
        help='Model backbones to test (default: all)'
    )
    parser.add_argument(
        '--setups', nargs='+',
        default=TRAINING_SETUPS,
        choices=TRAINING_SETUPS,
        metavar='SETUP',
        help='Training setups to test (default: all). Use "" or "standard" for standard training'
    )
    
    # Method selection
    parser.add_argument(
        '--exclude-methods', nargs='+',
        default=[],
        choices=ALL_METHODS,
        help='Methods to exclude from all runs'
    )
    
    # Execution options
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU device ID (default: 0)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./uq_benchmark_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--per-fold-eval', action='store_true', default=True,
        help='Use per-fold evaluation (default: True)'
    )
    parser.add_argument(
        '--ensemble-eval', dest='per_fold_eval', action='store_false',
        help='Use ensemble evaluation instead of per-fold'
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
    
    # New class shift evaluation
    parser.add_argument(
        '--new-class-shift', action='store_true', default=False,
        help='Enable new class shift evaluation (only for amos2022 and midog: evaluates on unseen organ classes)'
    )
    
    # Control flags
    parser.add_argument(
        '--run-mode', type=str, choices=['both', 'calib_only', 'test_only'], default='both',
        help="Execution mode passed to run_medmnist_benchmark: 'both' (default) runs calib+test; "
             "'calib_only' collects z-score normalization stats only; "
             "'test_only' skips calib_detector."
    )
    parser.add_argument(
        '--max-loader-workers', type=int, default=48,
        help='Hard cap for DataLoader workers per spawned process (default: 48).'
    )
    parser.add_argument(
        '--concurrent-processes', type=int, default=1,
        help='Number of benchmark processes running simultaneously on this host '
             '(used to throttle CPU per process, default: 1).'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print commands without executing them'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress command output (only show summaries)'
    )
    
    args = parser.parse_args()
    
    # Handle dataset selection
    if args.id_only:
        datasets = [d for d in args.datasets if d in DATASETS_ID]
    elif args.external_only:
        datasets = [d for d in args.datasets if d in DATASETS_EXTERNAL]
    else:
        datasets = args.datasets
    
    # Handle "standard" setup alias
    setups = args.setups
    if 'standard' in setups:
        setups = ['' if s == 'standard' else s for s in setups]
    
    # Validate paths
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"[ERROR] Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Generate commands
    commands = generate_all_commands(
        datasets=datasets,
        models=args.models,
        setups=setups,
        python_path=args.python,
        script_path=script_path,
        gpu=args.gpu,
        exclude_methods=args.exclude_methods,
        per_fold_eval=args.per_fold_eval,
        output_dir=args.output_dir,
        corruption_severity=args.corruption_severity,
        corrupt_test=args.corrupt_test,
        corrupt_calib=args.corrupt_calib,
        new_class_shift=args.new_class_shift,
        run_mode=args.run_mode,
        max_loader_workers=args.max_loader_workers,
        concurrent_processes=args.concurrent_processes
    )
    
    if not commands:
        print("[ERROR] No valid configurations to run")
        sys.exit(1)
    
    # Execute
    run_commands(commands, dry_run=args.dry_run, verbose=not args.quiet)


if __name__ == '__main__':
    main()