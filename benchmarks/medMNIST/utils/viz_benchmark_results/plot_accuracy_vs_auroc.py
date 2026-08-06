"""
Generate scatter plot showing relationship between balanced accuracy and AUROC_f.

For each fold, configuration, dataset, and method, plot:
- X-axis: Balanced accuracy of that fold
- Y-axis: AUROC_f of the method on that fold

Loads pre-computed results from:
- Benchmarks/medMNIST/results/classification_results/*.json (classification metrics including balanced accuracy)
- Benchmarks/medMNIST/results/full_results/all_metrics_*.npz (AUROC_f values)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict
import warnings
import seaborn as sns
import pandas as pd
warnings.filterwarnings('ignore')


def load_comprehensive_metrics(json_file):
    """
    Load comprehensive metrics from JSON file.
    
    Returns:
        dict with balanced accuracy per fold and ensemble
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Handle two different JSON structures:
        # 1. Standard format: {'per_fold_metrics': [...], 'ensemble_metrics': {...}}
        # 2. Corruption format: {'per_fold': [...], 'ensemble': {...}}
        
        per_fold_acc = []
        
        # Try standard format first
        if 'per_fold_metrics' in data:
            for fold_data in data['per_fold_metrics']:
                if 'balanced_accuracy' in fold_data:
                    per_fold_acc.append(fold_data['balanced_accuracy'])
        # Try corruption format
        elif 'per_fold' in data:
            for fold_data in data['per_fold']:
                if 'balanced_accuracy' in fold_data:
                    per_fold_acc.append(fold_data['balanced_accuracy'])
        
        # Extract ensemble balanced accuracy
        ensemble_acc = None
        if 'ensemble_metrics' in data and 'balanced_accuracy' in data['ensemble_metrics']:
            ensemble_acc = data['ensemble_metrics']['balanced_accuracy']
        elif 'ensemble' in data and 'balanced_accuracy' in data['ensemble']:
            ensemble_acc = data['ensemble']['balanced_accuracy']
        
        return {
            'per_fold_acc': per_fold_acc,
            'ensemble_acc': ensemble_acc
        }
    except Exception as e:
        print(f"Failed to load {json_file}: {e}")
        return None


def find_cache_file(cache_dir, dataset, model, config, shift_name):
    """Find the test_test cache file for a given configuration."""
    if shift_name == 'in_distribution':
        if config == 'standard':
            patterns = [f"{dataset}_{model}_test_test_results.npz", f"{dataset}_{model}_test_results.npz"]
        else:
            patterns = [f"{dataset}_{model}_{config}_test_test_results.npz", f"{dataset}_{model}_{config}_test_results.npz"]
    elif shift_name == 'corruption':
        if config == 'standard':
            patterns = [f"{dataset}_{model}_corrupt3_test_test_results.npz", f"{dataset}_{model}_corrupt3_test_results.npz"]
        else:
            patterns = [f"{dataset}_{model}_{config}_corrupt3_test_test_results.npz", f"{dataset}_{model}_{config}_corrupt3_test_results.npz"]
    elif shift_name == 'new_class_shift':
        # New class shift has special cache naming with _new_class_shift suffix
        if config == 'standard':
            patterns = [f"{dataset}_{model}_new_class_shift_test_results.npz"]
        else:
            patterns = [f"{dataset}_{model}_{config}_new_class_shift_test_results.npz"]
    else:  # population
        if config == 'standard':
            patterns = [f"{dataset}_{model}_test_test_results.npz", f"{dataset}_{model}_test_results.npz"]
        else:
            patterns = [f"{dataset}_{model}_{config}_test_test_results.npz", f"{dataset}_{model}_{config}_test_results.npz"]
    
    for pattern in patterns:
        cache_file = cache_dir / pattern
        if cache_file.exists():
            return cache_file
    return None


def load_cache_data(cache_file):
    """Load y_true, y_pred, and per_fold_predictions from cache file."""
    try:
        data = np.load(cache_file, allow_pickle=True)
        result = {
            'y_true': data['y_true'],
            'y_pred': data.get('y_pred', None),
            'per_fold_predictions': data.get('per_fold_predictions', None),
            'binary_gt': data.get('binary_gt', None)  # For new_class_shift
        }
        return result
    except:
        return None


def load_uq_metrics(json_file):
    """
    Load UQ metrics from JSON file.
    
    Returns:
        dict of method_name -> {per_fold: [...], ensemble: scalar}
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        methods_data = {}
        
        # Process methods (note: key is 'methods' not 'results')
        if 'methods' in data:
            for method_name, method_results in data['methods'].items():
                methods_data[method_name] = {}
                
                # Get per-fold AUROC_f from per_fold_metrics array
                if 'per_fold_metrics' in method_results:
                    per_fold_auroc = [fold['auroc_f'] for fold in method_results['per_fold_metrics']]
                    methods_data[method_name]['per_fold'] = per_fold_auroc
                
                # Get ensemble AUROC_f (top-level auroc_f is the ensemble result)
                if 'auroc_f' in method_results:
                    methods_data[method_name]['ensemble'] = method_results['auroc_f']
        
        return methods_data
    except Exception as e:
        print(f"Failed to load {json_file}: {e}")
        return {}


def parse_filename_to_key(filename):
    """
    Parse filename to extract (dataset, model, config).
    
    Args:
        filename: filename to parse (JSON format)
    
    Returns:
        tuple: (dataset, model, config)
    """
    import re
    
    # Handle different filename patterns:
    # 1. comprehensive_metrics_bloodmnist_resnet18_DADO.json
    # 2. uq_benchmark_bloodmnist_resnet18_20260116_095229.json
    # 3. bloodmnist_resnet18_DADO_severity3.json (corruption)
    
    if filename.startswith('comprehensive_metrics_'):
        name = filename.replace('comprehensive_metrics_', '').replace('.json', '')
    elif filename.startswith('uq_benchmark_'):
        name = filename.replace('uq_benchmark_', '').replace('.json', '')
        # Remove timestamp
        name = re.sub(r'_\d{8}_\d{6}$', '', name)
    else:
        # Corruption format: {dataset}_{model}_{config}_severity3.json
        name = filename.replace('.json', '').replace('_severity3', '')
    
    # Parse parts
    parts = name.split('_')
    
    # Find model name
    if 'vit' in parts:
        model_idx = parts.index('vit')
        model = 'vit_b_16'
        dataset = '_'.join(parts[:model_idx])
        config_parts = parts[model_idx+3:]
    elif 'resnet18' in parts:
        model_idx = parts.index('resnet18')
        model = 'resnet18'
        dataset = '_'.join(parts[:model_idx])
        config_parts = parts[model_idx+1:]
    else:
        return None
    
    # Determine config
    if config_parts and config_parts[0] in ['DA', 'DO', 'DADO']:
        config = config_parts[0]
    else:
        config = 'standard'
    
    return (dataset, model, config)


def collect_all_data_points(workspace_root):
    """
    Collect all (balanced_accuracy, auroc_f) pairs from pre-computed JSON results.
    
    Returns:
        dict: {
            'per_fold': list of data dicts
            'ensemble': list of data dicts
        }
    """
    comprehensive_dir = workspace_root / 'benchmarks' / 'medMNIST' / 'results' / 'classification_results'
    uq_results_dir = workspace_root / 'benchmarks' / 'medMNIST' / 'results' / 'jsons_results'
    
    per_fold_data = []
    ensemble_data = []
    
    # Shifts to process
    shift_dirs = {
        'in_distribution': ('in_distribution', 'in_distribution'),
        'corruption': ('corruption_shifts', 'corruption_shifts'),
        'population': ('population_shift', 'population_shifts')
    }
    
    for shift_name, (comprehensive_subdir, uq_subdir) in shift_dirs.items():
        comprehensive_shift_dir = comprehensive_dir / comprehensive_subdir
        uq_shift_dir = uq_results_dir / uq_subdir
        
        if not comprehensive_shift_dir.exists() or not uq_shift_dir.exists():
            print(f"Skipping {shift_name}: directories not found")
            continue
        
        print(f"\nProcessing {shift_name}...")
        
        # Find all comprehensive metrics JSON files
        # Different naming patterns for different shifts:
        # - in_distribution & population: comprehensive_metrics_*.json
        # - corruption: {dataset}_{model}_{config}_severity3.json
        if shift_name == 'corruption':
            json_files = list(comprehensive_shift_dir.glob('*_severity3.json'))
        else:
            json_files = list(comprehensive_shift_dir.glob('comprehensive_metrics_*.json'))
        print(f" Found {len(json_files)} classification JSON files")
        
        for json_file in json_files:
            # Parse filename
            key = parse_filename_to_key(json_file.name)
            if key is None:
                continue
            
            dataset, model, config = key
            
            # Handle dataset name mismatches between comprehensive and UQ files
            # (comprehensive uses different names than UQ for population shifts)
            dataset_mapping = {
                'amos22': 'amos2022',
                'dermamnist-e-ood': 'dermamnist-e-external'
            }
            uq_dataset = dataset_mapping.get(dataset, dataset)
            
            # Load classification metrics
            class_metrics = load_comprehensive_metrics(json_file)
            if class_metrics is None:
                continue
            # Don't skip if per_fold is missing - may still have ensemble data
            if not class_metrics['per_fold_acc'] and class_metrics['ensemble_acc'] is None:
                continue
            
            # Find corresponding UQ metrics JSON file
            if shift_name == 'corruption':
                # Corruption files have special naming: uq_benchmark_{dataset}_{model}_{config}_corrupt_severity3_test_*.json
                if config != 'standard':
                    uq_json_files = list(uq_shift_dir.glob(f'uq_benchmark_{uq_dataset}_{model}_{config}_corrupt_severity3_*.json'))
                else:
                    uq_json_files = list(uq_shift_dir.glob(f'uq_benchmark_{uq_dataset}_{model}_corrupt_severity3_*.json'))
            else:
                # Standard naming for in_distribution and population
                uq_json_files = list(uq_shift_dir.glob(f'uq_benchmark_{uq_dataset}_{model}_*.json'))
            
            if shift_name != 'corruption':
                if config != 'standard':
                    uq_json_files = [f for f in uq_json_files if f'{model}_{config}_' in f.name]
                else:
                    uq_json_files = [f for f in uq_json_files if not any(c in f.name for c in ['_DA_', '_DO_', '_DADO_'])]
            
            if not uq_json_files:
                continue
            
            # Use the most recent file
            uq_json_file = sorted(uq_json_files)[-1]
            
            # Load UQ metrics
            uq_metrics = load_uq_metrics(uq_json_file)
            if not uq_metrics:
                continue
            
            # Process per-fold data
            num_folds = len(class_metrics['per_fold_acc'])
            
            for method_name, method_data in uq_metrics.items():
                if 'per_fold' not in method_data:
                    continue
                
                auroc_per_fold = method_data['per_fold']
                
                # Match folds
                for fold_idx in range(min(num_folds, len(auroc_per_fold))):
                    fold_acc = class_metrics['per_fold_acc'][fold_idx]
                    fold_auroc = auroc_per_fold[fold_idx]
                    
                    if not np.isnan(fold_acc) and not np.isnan(fold_auroc):
                        per_fold_data.append({
                            'balanced_accuracy': fold_acc,
                            'auroc_f': fold_auroc,
                            'method': method_name,
                            'dataset': dataset,
                            'model': model,
                            'config': config,
                            'shift': shift_name,
                            'fold': fold_idx
                        })
            
            # Process ensemble data (use ensemble classification accuracy and ensemble UQ score)
            if class_metrics['ensemble_acc'] is not None:
                for method_name, method_data in uq_metrics.items():
                    if 'ensemble' in method_data:
                        ensemble_acc = class_metrics['ensemble_acc']
                        ensemble_auroc = method_data['ensemble']
                        
                        if not np.isnan(ensemble_acc) and not np.isnan(ensemble_auroc):
                            ensemble_data.append({
                                'balanced_accuracy': ensemble_acc,
                                'auroc_f': ensemble_auroc,
                                'method': method_name,
                                'dataset': dataset,
                                'model': model,
                                'config': config,
                                'shift': shift_name
                            })

    
    return {'per_fold': per_fold_data, 'ensemble': ensemble_data}


def create_scatter_plot(data, output_dir):
    """
    Create scatter plot of balanced accuracy vs AUROC_f.
    
    Args:
        data: dict with 'per_fold' and 'ensemble' lists
        output_dir: directory to save plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Combine all data
    all_data = data['per_fold'] + data['ensemble']
    
    # Get methods with ORIGINAL names for color assignment (to match radar plots)
    methods_original = sorted(set(d['method'] for d in all_data))
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods_original)))
    method_colors_original = dict(zip(methods_original, colors))
    
    # Now rename methods for display
    name_mapping = {
        'MSR_calibrated': 'MSR-S',
        'KNN_Raw': 'KNN',
        'Ensembling': 'DE',
        'MCDropout': 'MCD',
        'ZScore Agg + Ens': 'Mean Agg + Ens',
        'ZScore_Aggregation_per_fold': 'Mean Agg',
        'ZScore_Aggregation_ensemble': 'Mean Agg + Ens'
    }
    
    for d in all_data:
        if d['method'] in name_mapping:
            d['method'] = name_mapping[d['method']]
    
    # Create color map with display names
    method_colors = {}
    for orig_name, color in method_colors_original.items():
        display_name = name_mapping.get(orig_name, orig_name)
        method_colors[display_name] = color
    
    # Extract unique methods - reorder for display but keep color mapping consistent
    all_methods = set(d['method'] for d in all_data)
    mean_agg_methods = ['Mean Agg', 'Mean Agg + Ens']
    regular_methods = sorted([m for m in all_methods if m not in mean_agg_methods])
    methods = regular_methods + [m for m in mean_agg_methods if m in all_methods]
    
    # Plot each method
    for method in methods:
        method_data = [d for d in all_data if d['method'] == method]
        x = [d['balanced_accuracy'] for d in method_data]
        y = [d['auroc_f'] for d in method_data]
        
        # Use different markers for different method types
        if method == 'Mean Agg':
            # Red star for Mean Aggregation
            ax.scatter(x, y, s=150, marker='*', c='red',
                      label='Mean Agg', alpha=0.8, edgecolors='darkred', linewidths=0.5)
            continue
        elif method == 'Mean Agg + Ens':
            # Diamond for Mean Aggregation + Ensemble
            ax.scatter(x, y, s=150, marker='$\u26A1$', c="#f4c97a",
                      label='Mean Agg+Ens', alpha=0.8, edgecolors='black', linewidths=0.3)
            continue
        elif method == 'DE':
            marker = 's'  # Square for ensemble
            alpha = 0.7
            size = 50
            edgecolor = 'white'
            linewidth = 0.5
        else:
            marker = 'o'  # Circle for others
            alpha = 0.7
            size = 20
            edgecolor = 'white'
            linewidth = 0.5
        
        ax.scatter(x, y, c=[method_colors[method]], marker=marker, s=size, 
                  alpha=alpha, label=method, edgecolors=edgecolor, linewidths=linewidth)
    
    # Add diagonal reference line (perfect correlation)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, linewidth=1, label='y=x reference')
    
    ax.set_xlabel('Balanced Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUROC (Failure Detection)', fontsize=14, fontweight='bold')
    #ax.set_title('Relationship between Model Accuracy and Failure Detection Performance\nAcross All Methods, Configurations, and Shifts', 
     #           fontsize=16, fontweight='bold', pad=20)
    
    legend = ax.legend(loc='upper left', ncol=2, fontsize=10, frameon=True, framealpha=0.9, markerscale=2.0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    output_path = output_dir / 'accuracy_vs_auroc_scatter.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Scatter plot saved to {output_path}")
    
    plt.close(fig)
    
    # Also create separate plots by shift type
    create_scatter_by_shift(data, output_dir)


def create_scatter_by_shift(data, output_dir):
    """Create separate scatter plots for each shift type."""
    all_data = data['per_fold'] + data['ensemble']
    
    # Get methods with ORIGINAL names for color assignment (to match radar plots)
    methods_original = sorted(set(d['method'] for d in all_data))
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods_original)))
    method_colors_original = dict(zip(methods_original, colors))
    
    # Now rename methods for display
    name_mapping = {
        'MSR_calibrated': 'MSR-S',
        'KNN_Raw': 'KNN',
        'Ensembling': 'DE',
        'MCDropout': 'MCD',
        'ZScore Agg + Ens': 'Mean Agg + Ens',
        'ZScore_Aggregation_per_fold': 'Mean Agg',
        'ZScore_Aggregation_ensemble': 'Mean Agg + Ens'
    }
    
    for d in all_data:
        if d['method'] in name_mapping:
            d['method'] = name_mapping[d['method']]
    
    # Create color map with display names
    method_colors = {}
    for orig_name, color in method_colors_original.items():
        display_name = name_mapping.get(orig_name, orig_name)
        method_colors[display_name] = color
    
    # Custom order for shifts: in_distribution first, then corruption, then population
    shift_order = ['in_distribution', 'corruption', 'population']
    shifts = [s for s in shift_order if s in set(d['shift'] for d in all_data)]
    
    # Create figure with custom layout: 2 top, legend bottom left, population bottom right
    fig = plt.figure(figsize=(16, 12))
    
    # Top row: 2 subplots
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    # Bottom right: population subplot
    ax3 = plt.subplot(2, 2, 4)
    
    axes_list = [ax1, ax2, ax3]
    
    # Extract unique methods - reorder for display but keep color mapping consistent
    all_methods = set(d['method'] for d in all_data)
    mean_agg_methods = ['Mean Agg', 'Mean Agg + Ens']
    regular_methods = sorted([m for m in all_methods if m not in mean_agg_methods])
    methods = regular_methods + [m for m in mean_agg_methods if m in all_methods]

    for idx, shift in enumerate(shifts):
        if idx >= len(axes_list):
            break
        ax = axes_list[idx]
        shift_data = [d for d in all_data if d['shift'] == shift]
        
        for method in methods:
            method_data = [d for d in shift_data if d['method'] == method]
            if not method_data:
                continue
            
            x = [d['balanced_accuracy'] for d in method_data]
            y = [d['auroc_f'] for d in method_data]
            
            # Use different markers for different method types
            if method == 'Mean Agg':
                # Red star for Mean Aggregation
                ax.scatter(x, y, s=150, marker='*', c='red',
                          label='Mean Agg', alpha=0.8, edgecolors='darkred', linewidths=0.5)
                continue
            elif method == 'Mean Agg + Ens':
                # Diamond for Mean Aggregation + Ensemble
                ax.scatter(x, y, s=150, marker='$\u26A1$', c="#f4c97a",
                          label='Mean Agg+Ens', alpha=0.8, edgecolors='black', linewidths=0.3)
                continue
            elif method == 'DE':
                marker = 's'  # Square for ensemble
                alpha = 0.7
                size = 50
                edgecolor = 'white'
                linewidth = 0.5
            else:
                marker = 'o'  # Circle for others
                alpha = 0.7
                size = 20
                edgecolor = 'white'
                linewidth = 0.5
            
            ax.scatter(x, y, c=[method_colors[method]], marker=marker, s=size, 
                      alpha=alpha, label=method, edgecolors=edgecolor, linewidths=linewidth)
        
        # Add diagonal reference line (perfect correlation)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, linewidth=1)
        
        ax.set_xlabel('Balanced Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('AUROC (Failure Detection)', fontsize=12, fontweight='bold')
        
        # Add "Shift" suffix for corruption and population
        title = shift.replace("_", " ").title()
        if shift in ['corruption', 'population']:
            title += ' Shifts'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add shared legend in bottom left position
    # Collect handles and labels from first subplot
    handles, labels = axes_list[0].get_legend_handles_labels()
    # Create invisible subplot for legend in bottom left
    ax_legend = plt.subplot(2, 2, 3)
    ax_legend.axis('off')
    ax_legend.legend(handles, labels, loc='center', ncol=2, fontsize=12, 
                    frameon=True, framealpha=0.95, title='Methods', title_fontsize=14, markerscale=2.0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'accuracy_vs_auroc_by_shift.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Shift-wise scatter plot saved to {output_path}")
    
    plt.close(fig)


def create_per_method_correlation_plots(data, output_dir):
    """
    One subplot per UQ method with all shifts (ID, CS, PS) merged into a single cloud.
    Layout: 2 rows × 5 columns (10 methods).
    """
    all_data = data['per_fold'] + data['ensemble']

    # Rename methods for display
    name_mapping = {
        'MSR_calibrated': 'MSR-S',
        'KNN_Raw': 'KNN',
        'Ensembling': 'DE',
        'MCDropout': 'MCD',
        'ZScore Agg + Ens': 'Mean Agg + Ens',
        'ZScore_Aggregation_per_fold': 'Mean Agg',
        'ZScore_Aggregation_ensemble': 'Mean Agg + Ens'
    }
    for d in all_data:
        if d['method'] in name_mapping:
            d['method'] = name_mapping[d['method']]

    # Fixed display order (consistent with the per-shift version)
    methods = ['MSR', 'MSR-S', 'MLS', 'TTA', 'GPS', 'MCD', 'KNN', 'DE', 'Mean Agg', 'Mean Agg + Ens']

    # Colors keyed by method name (alphabetical, for consistency with other plots)
    all_methods_alpha = sorted(set(d['method'] for d in all_data))
    method_colors = dict(zip(all_methods_alpha,
                             plt.cm.tab20(np.linspace(0, 1, len(all_methods_alpha)))))

    n_cols = 5
    n_rows = (len(methods) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))

    all_corr_values = []

    for method_idx, method in enumerate(methods):
        row = method_idx // n_cols
        col = method_idx % n_cols
        ax = axes[row, col]

        # All shifts merged
        method_data = [d for d in all_data if d['method'] == method]
        x = np.array([d['balanced_accuracy'] for d in method_data])
        y = np.array([d['auroc_f'] for d in method_data])

        # Scatter — same style as the per-shift version
        if method == 'Mean Agg':
            ax.scatter(x, y, s=150, marker='*', c='red',
                       alpha=0.7, edgecolors='darkred', linewidths=0.5)
        elif method == 'Mean Agg + Ens':
            ax.scatter(x, y, s=150, marker='$\u26A1$', c="#f4c97a",
                       alpha=0.7, edgecolors='black', linewidths=0.5)
        else:
            ax.scatter(x, y, c=[method_colors.get(method, '#888888')], marker='o', s=50,
                       alpha=0.7, edgecolors='white', linewidths=0.5)

        # Trend line + annotation
        if len(x) > 1:
            corr = np.corrcoef(x, y)[0, 1]
            if not np.isnan(corr):
                all_corr_values.append(corr)
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), "r--", alpha=0.5, linewidth=2)
            ax.text(0.4, 0.2, f'r = {corr:.2f}\ny = {z[0]:.2f}x + {z[1]:.2f}',
                    transform=ax.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # x=y diagonal
        ax.plot([0.4, 1.0], [0.4, 1.0], 'k--', alpha=0.3, linewidth=1, zorder=0)

        ax.set_title(method, fontsize=13, fontweight='bold')
        ax.set_xlim([0.4, 1.0])
        ax.set_ylim([0.4, 1.0])
        ax.grid(True, alpha=0.3, linestyle='--')

        if col == 0:
            ax.set_ylabel('AUROC_f', fontsize=11, fontweight='bold')
        if row == n_rows - 1:
            ax.set_xlabel('Balanced Accuracy', fontsize=11, fontweight='bold')

    # Hide unused subplots
    for idx in range(len(methods), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')

    plt.tight_layout()

    if all_corr_values:
        mean_r = np.mean(all_corr_values)
        std_r = np.std(all_corr_values)
        print(f"\nMean Pearson r over {len(all_corr_values)} subplots (all shifts merged): "
              f"{mean_r:.4f} ± {std_r:.4f}")

    output_path = output_dir / 'per_method_correlation_all_shifts.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Merged per-method correlation plot saved to {output_path}")
    plt.close(fig)


def create_method_correlation_pairplots(data, output_dir):
    """
    Create pairplots showing correlations between methods for each shift type.
    Each pairplot shows how different methods' AUROC_f scores correlate with each other.
    
    Note: Separates per-fold and ensemble results since they represent different test cases.
    
    Args:
        data: dict with 'per_fold' and 'ensemble' lists
        output_dir: directory to save plots
    """
    # Rename methods for display
    name_mapping = {
        'MSR_calibrated': 'MSR-S',
        'KNN_Raw': 'KNN',
        'Ensembling': 'DE',
        'MCDropout': 'MCD',
        'ZScore Agg + Ens': 'Mean Agg + Ens',
        'ZScore_Aggregation_per_fold': 'Mean Agg',
        'ZScore_Aggregation_ensemble': 'Mean Agg + Ens'
    }
    
    # Define ensemble methods (these should only correlate with other ensemble results)
    ensemble_methods = ['DE', 'Mean Agg + Ens']
    
    # Process each shift type separately
    shift_info = [
        ('in_distribution', 'In-Distribution'),
        ('corruption', 'Corruption Shifts'),
        ('population', 'Population Shifts')
    ]
    
    for shift_type, shift_title in shift_info:
        # Process per-fold data (exclude ensemble methods)
        per_fold_data = [d for d in data['per_fold'] if d['shift'] == shift_type]
        
        # Rename methods
        for d in per_fold_data:
            if d['method'] in name_mapping:
                d['method'] = name_mapping[d['method']]
        
        if not per_fold_data:
            continue
        
        # Create a DataFrame where each row is a unique test case (dataset, config, fold)
        # and each column is a method's AUROC_f score
        test_cases = defaultdict(dict)
        for d in per_fold_data:
            # Exclude ensemble methods from per-fold data
            if d['method'] in ensemble_methods:
                continue
            key = (d['dataset'], d['config'], d.get('fold', 'NA'))
            test_cases[key][d['method']] = d['auroc_f']
        
        # Convert to DataFrame
        rows = []
        for (dataset, config, fold), method_scores in test_cases.items():
            row = {'dataset': dataset, 'config': config, 'fold': fold}
            row.update(method_scores)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Get method columns (exclude metadata columns and ensemble methods)
        method_cols = [col for col in df.columns 
                      if col not in ['dataset', 'config', 'fold'] 
                      and col not in ensemble_methods]
        
        # Only keep methods that have data for at least some test cases
        method_cols = [col for col in method_cols if df[col].notna().sum() > 0]
        
        # Sort methods alphabetically (to match color scheme)
        method_cols = sorted(method_cols)
        
        print(f"\n{shift_title}: Creating pairplot with {len(method_cols)} per-fold methods and {len(df)} test cases")
        
        # Create pairplot
        if len(method_cols) < 2:
            print(f" Skipping {shift_title} - need at least 2 methods")
            continue
        
        # Set up the plot style
        sns.set_style("whitegrid")
        
        # Create pairplot with smaller markers and transparency
        g = sns.pairplot(
            df[method_cols],
            diag_kind='kde',
            plot_kws={'alpha': 0.2, 's': 20, 'edgecolor': 'none'},
            diag_kws={'linewidth': 2},
            corner=False
        )
        
        # Add title
        g.fig.suptitle(f'{shift_title}: Per-Fold Method AUROC_f Correlations', 
                      fontsize=16, fontweight='bold', y=1.0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f'method_correlation_pairplot_{shift_type}.png'
        g.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f" Pairplot saved to {output_path}")
        
        plt.close()


def create_sample_level_pairplots(workspace_root, output_dir):
    """
    Create pairplots showing correlations between methods at the sample level.
    Each point is an individual sample's uncertainty score across all methods.
    
    Args:
        workspace_root: path to workspace root
        output_dir: directory to save plots
        
    Returns:
        shift_dataframes: dict mapping shift names to their complete dataframes
    """
    # Rename methods for display
    name_mapping = {
        'MSR_calibrated': 'MSR-S',
        'KNN_Raw': 'KNN',
        'Ensembling': 'DE',
        'MCDropout': 'MCD',
        'ZScore Agg + Ens': 'Mean Agg + Ens',
        'ZScore_Aggregation_per_fold': 'Mean Agg',
        'ZScore_Aggregation_ensemble': 'Mean Agg + Ens'
    }
    
    # Dictionary to store complete dataframes for paired limit computation
    # Keys: shift_dir_name, Values: df_complete
    shift_dataframes = {}
    
    # Dictionary to store axis limits for paired shifts
    # Keys: ('id_cs', method) or ('ps_ncs', method)
    # Values: {'xlim': (min, max), 'ylim': (min, max)}
    paired_axis_limits = {}
    
    # Process each shift type including new_class_shifts
    shift_info = [
        ('in_distribution', 'In-Distribution'),
        ('corruption_shifts', 'Corruption Shifts'),
        ('population_shifts', 'Population Shifts'),
        ('new_class_shifts', 'New Class Shifts')
    ]
    
    for shift_dir_name, shift_title in shift_info:
        print(f"\n{shift_title}: Loading sample-level scores...")
        
        uq_shift_dir = workspace_root / 'Benchmarks' / 'medMNIST' / 'results' / 'full_results' / shift_dir_name
        
        if not uq_shift_dir.exists():
            print(f" Directory not found: {uq_shift_dir}")
            continue
        
        # Collect all sample-level scores across all NPZ files
        # Structure: {(dataset, config, fold, sample_idx): {method: score}}
        sample_scores = defaultdict(dict)
        
        # Find all NPZ files
        npz_files = sorted(uq_shift_dir.glob('all_metrics_*.npz'), key=lambda p: p.name)
        print(f" Found {len(npz_files)} NPZ files")
        
        for npz_file in npz_files:
            # Parse filename to get dataset, model, config
            filename = npz_file.name
            if filename.startswith('all_metrics_'):
                name = filename.replace('all_metrics_', '').replace('.json', '').replace('.npz', '')
                
                # Remove timestamp and severity patterns
                import re
                name = re.sub(r'_\d{8}_\d{6}$', '', name)
                name = name.replace('_corrupt3', '')
                name = name.replace('_corrupt_severity3', '')
                
                parts = name.split('_')
                
                # Find model
                if 'resnet18' in parts:
                    model_idx = parts.index('resnet18')
                    model = 'resnet18'
                elif 'vit' in parts and 'b' in parts and '16' in parts:
                    model_idx = parts.index('vit')
                    model = 'vit_b_16'
                else:
                    continue
                
                dataset = '_'.join(parts[:model_idx])
                config_parts = parts[model_idx+1:] if model == 'resnet18' else parts[model_idx+3:]
                config = config_parts[0] if config_parts and config_parts[0] in ['DA', 'DO', 'DADO'] else 'standard'
                
                # Load NPZ data
                try:
                    npz_data = np.load(npz_file, allow_pickle=True)

                    # For new_class_shifts, correctness labels are embedded in the NPZ itself
                    # (baseline_per_fold_correct_idx / baseline_per_fold_incorrect_idx).
                    # Other shifts need a separate cache file for y_true / per_fold_predictions.
                    if shift_dir_name == 'new_class_shifts':
                        # Reconstruct per-fold correctness from embedded index arrays
                        per_fold_correct_idx = npz_data['baseline_per_fold_correct_idx']  # object array (5,)
                        n_folds = len(per_fold_correct_idx)
                        # n_samples per fold inferred from any _per_fold score array
                        ref_key = next(k for k in npz_data.keys() if k.endswith('_per_fold') and npz_data[k].ndim == 2)
                        n_samples = npz_data[ref_key].shape[1]

                        # Build per-fold boolean correct arrays
                        fold_correct_masks = []
                        for fi in range(n_folds):
                            mask = np.zeros(n_samples, dtype=bool)
                            mask[per_fold_correct_idx[fi]] = True
                            fold_correct_masks.append(mask)

                        # Process each method
                        for method_name in sorted(npz_data.keys()):
                            if npz_data[method_name].ndim != 2 or npz_data[method_name].shape[0] != n_folds:
                                continue
                            base_method_name = method_name.replace('_per_fold', '')
                            display_name = name_mapping.get(base_method_name, base_method_name)
                            scores = npz_data[method_name]

                            for fold_idx in range(n_folds):
                                fold_scores = scores[fold_idx]
                                correct_mask = fold_correct_masks[fold_idx]
                                for sample_idx in range(min(len(fold_scores), n_samples)):
                                    key = (dataset, model, config, fold_idx, sample_idx)
                                    sample_scores[key][display_name] = fold_scores[sample_idx]
                                    if 'correct' not in sample_scores[key]:
                                        sample_scores[key]['correct'] = bool(correct_mask[sample_idx])

                    else:
                        # Standard path: load y_true / per_fold_predictions from cache file
                        cache_dir = uq_shift_dir.parent.parent / 'cache'
                        shift_name_short = shift_dir_name.replace('_shifts', '')
                        cache_file = find_cache_file(cache_dir, dataset, model, config, shift_name_short)

                        if not cache_file or not cache_file.exists():
                            continue

                        cache_data = load_cache_data(cache_file)
                        if cache_data is None or cache_data['per_fold_predictions'] is None:
                            continue

                        n_folds = len(cache_data['per_fold_predictions'])
                        n_samples = len(cache_data['y_true'])
                        y_true = cache_data['y_true']
                        binary_gt = cache_data.get('binary_gt', None)

                        for method_name in sorted(npz_data.keys()):
                            if npz_data[method_name].shape[0] != 5:
                                continue
                            base_method_name = method_name.replace('_per_fold', '')
                            display_name = name_mapping.get(base_method_name, base_method_name)
                            scores = npz_data[method_name]

                            for fold_idx in range(n_folds):
                                fold_scores = scores[fold_idx]
                                y_pred_fold = cache_data['per_fold_predictions'][fold_idx]
                                n_samples_safe = min(len(fold_scores), len(y_pred_fold), len(y_true))
                                for sample_idx in range(n_samples_safe):
                                    key = (dataset, model, config, fold_idx, sample_idx)
                                    sample_scores[key][display_name] = fold_scores[sample_idx]
                                    if 'correct' not in sample_scores[key]:
                                        if binary_gt is not None:
                                            sample_scores[key]['correct'] = (binary_gt[sample_idx] == 0)
                                        else:
                                            sample_scores[key]['correct'] = (y_true[sample_idx] == y_pred_fold[sample_idx])
                
                except Exception as e:
                    print(f" Warning: Failed to process {npz_file.name}: {e}")
                    continue
        
        if not sample_scores:
            print(f" No sample scores found")
            continue
        
        # Convert to DataFrame
        rows = []
        for (dataset, model, config, fold, sample_idx), method_dict in sorted(sample_scores.items()):
            row = {'dataset': dataset, 'model': model, 'config': config, 'fold': fold, 'sample_idx': sample_idx}
            row.update(method_dict)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Get method columns
        method_cols = [col for col in df.columns 
                      if col not in ['dataset', 'model', 'config', 'fold', 'sample_idx', 'correct']]
        print(method_cols)
        # Only keep methods with sufficient data
        method_cols = [col for col in method_cols if df[col].notna().sum() > 100]
        method_cols = ['MSR', 'MSR-S', 'MLS', 'TTA', 'GPS', 'MCD', 'KNN', 'ZScore_Aggregation']
        # Remove samples with missing values in any method.
        # Keep metadata columns needed for grouped normalization and later stratification.
        df_complete = df[method_cols + ['correct', 'dataset', 'model', 'config', 'fold']].dropna()

        # Z-score normalization per method within each (dataset, model, config, fold) group.
        group_cols = ['dataset', 'model', 'config', 'fold']
        for method in method_cols:
            grouped = df_complete.groupby(group_cols, sort=True)[method]
            means = grouped.transform('mean')
            stds = grouped.transform('std')
            df_complete[method] = np.where(stds > 0, (df_complete[method] - means) / stds, 0.0)
        
        # Convert correct boolean to string labels for better legend
        df_complete['Prediction'] = df_complete['correct'].map({True: 'Correct', False: 'Incorrect'})
        
        print(f" Found {len(df_complete)} complete samples across {len(method_cols)} methods")
        
        if len(method_cols) < 2 or len(df_complete) < 10:
            print(f" Skipping - insufficient data")
            continue
        df_complete = df_complete.rename(columns={'ZScore_Aggregation': 'Mean Agg'})
        # Subsample if too many points (for visualization performance)
        if len(df_complete) > 10000:
            # Get unique datasets
            datasets = sorted(df_complete['dataset'].unique())
            n_datasets = len(datasets)
            samples_per_dataset = 10000 // n_datasets
            
            # Target: 50/50 balance between correct and incorrect for clear visualization
            target_incorrect_ratio = 0.5
            
            sampled_dfs = []
            for dataset in datasets:
                df_dataset = df_complete[df_complete['dataset'] == dataset].sort_values(
                    by=['dataset', 'config', 'fold'], kind='mergesort'
                )
                
                # Count correct and incorrect for this dataset
                n_correct = df_dataset['correct'].sum()
                n_incorrect = (~df_dataset['correct']).sum()
                
                # If dataset has fewer samples than target, take all
                if len(df_dataset) <= samples_per_dataset:
                    sampled_dfs.append(df_dataset)
                else:
                    target_incorrect = int(samples_per_dataset * target_incorrect_ratio)
                    
                    # Take all incorrect if less than target, otherwise sample
                    if n_incorrect <= target_incorrect:
                        df_incorrect = df_dataset[~df_dataset['correct']]
                        n_correct_needed = samples_per_dataset - len(df_incorrect)
                        df_correct_sample = df_dataset[df_dataset['correct']].sample(n=min(n_correct_needed, n_correct), random_state=42)
                    else:
                        df_incorrect = df_dataset[~df_dataset['correct']].sample(n=target_incorrect, random_state=42)
                        df_correct_sample = df_dataset[df_dataset['correct']].sample(n=samples_per_dataset-target_incorrect, random_state=42)
                    
                    sampled_dfs.append(pd.concat([df_correct_sample, df_incorrect]))
            
            df_complete = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Store dataframe for paired limit computation
        shift_dataframes[shift_dir_name] = df_complete.copy()
        
        # Create pairplot
        sns.set_style("whitegrid")
        
        # Explicitly shuffle the dataframe to ensure random mixing of red/green points
        df_complete = df_complete.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Define custom color palette (green for correct, red for incorrect)
        palette = {'Correct': 'green', 'Incorrect': 'red'}
        method_cols = ['MSR', 'MSR-S', 'MLS', 'TTA', 'GPS', 'MCD', 'KNN', 'Mean Agg']

        g = sns.pairplot(
            df_complete,
            vars=method_cols,
            hue='Prediction',
            palette=palette,
            diag_kind='kde',
            plot_kws={'alpha': 0.2, 's': 20, 'edgecolor': 'none'},
            diag_kws={'linewidth': 2},
            corner=False
        )
    
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f'sample_level_pairplot_{shift_dir_name}.png'
        g.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f" Sample-level pairplot saved to {output_path}")
        
        plt.close()
    
    # === STEP 2: Compute paired axis limits from both shifts ===
    print("\n" + "="*80)
    print("Computing paired axis limits for corner plots...")
    print("="*80)
    
    paired_axis_limits = {}
    
    # Pair 1: ID + CS
    if 'in_distribution' in shift_dataframes and 'corruption_shifts' in shift_dataframes:
        df_id = shift_dataframes['in_distribution']
        df_cs = shift_dataframes['corruption_shifts']
        
        # Get common methods
        id_methods = [c for c in df_id.columns if c not in ['dataset', 'model', 'config', 'fold', 'sample_idx', 'correct', 'Prediction']]
        cs_methods = [c for c in df_cs.columns if c not in ['dataset', 'model', 'config', 'fold', 'sample_idx', 'correct', 'Prediction']]
        common_methods = list(set(id_methods) & set(cs_methods))
        
        for method in common_methods:
            min_val = min(df_id[method].min(), df_cs[method].min())
            max_val = max(df_id[method].max(), df_cs[method].max())
            range_val = max_val - min_val
            xlim = (min_val - 0.05 * range_val, max_val + 0.05 * range_val)
            paired_axis_limits[('id_cs', method)] = {'xlim': xlim, 'ylim': xlim}
            print(f" ID+CS {method}: [{xlim[0]:.3f}, {xlim[1]:.3f}]")
    
    # Pair 2: PS + NCS (will be completed in amos2022 function)
    if 'population_shifts' in shift_dataframes:
        df_ps = shift_dataframes['population_shifts']
        ps_methods = [c for c in df_ps.columns if c not in ['dataset', 'model', 'config', 'fold', 'sample_idx', 'correct', 'Prediction']]
        
        # Store for later pairing with amos2022
        for method in ps_methods:
            min_val = df_ps[method].min()
            max_val = df_ps[method].max()
            range_val = max_val - min_val
            xlim = (min_val - 0.05 * range_val, max_val + 0.05 * range_val)
            # Temporary limits, will be updated when amos2022 is processed
            paired_axis_limits[('ps_ncs', method)] = {'xlim': xlim, 'ylim': xlim}
    
    # === STEP 3: Create corner plots with paired limits ===
    print("\n" + "="*80)
    print("Creating corner plots with paired limits...")
    print("="*80)
    
    for shift_dir_name in shift_dataframes.keys():
        if shift_dir_name not in ['in_distribution', 'corruption_shifts', 'population_shifts']:
            continue
            
        df_complete = shift_dataframes[shift_dir_name]
        method_cols = [c for c in df_complete.columns if c not in ['dataset', 'model', 'config', 'fold', 'sample_idx', 'correct', 'Prediction']]
        
        # Define custom color palette
        palette = {'Correct': 'green', 'Incorrect': 'red'}
        
        # Create corner version - lower-left for ID/population, upper-right for CS/newclass
        if shift_dir_name in ['in_distribution', 'population_shifts']:
            # Lower-left triangle (standard corner=True)
            print(f" Creating LOWER-LEFT corner pairplot (same samples)...")
            
            # Use PairGrid for consistency with lower-right plots
            df_plot = df_complete.sample(frac=1, random_state=42).reset_index(drop=True)
            g_corner = sns.PairGrid(
                df_plot,
                vars=method_cols,
                diag_sharey=False
            )

            def scatter_shuffled(x, y, **kwargs):
                ax = plt.gca()
                idx = x.index  
                colors = df_plot.loc[idx, "Prediction"].map(palette).values

                ax.scatter(
                    x.values,
                    y.values,
                    c=colors,
                    alpha=0.2,
                    s=20,
                    edgecolors="none",
                    rasterized=True
                )

            g_corner.map_lower(scatter_shuffled)

            def diag_kde_by_class(x, **kwargs):
                ax = plt.gca()
                pred = df_plot.loc[x.index, "Prediction"]

                levels = pred.dropna().unique()
                
                try:
                    levels = np.sort(levels)
                except Exception:
                    pass

                for lvl in levels:
                    xs = x[pred == lvl]
                    if xs.size < 2:
                        continue
                    sns.kdeplot(
                        x=xs,
                        ax=ax,
                        fill=True,
                        linewidth=2,
                        color=palette.get(lvl, None),
                        alpha=0.35,
                        common_norm=False
                    )

            g_corner.map_diag(diag_kde_by_class)

            
            # Remove unused upper triangle axes (they might already be None with corner=True)
            for i in range(len(method_cols)):
                for j in range(len(method_cols)):
                    if j > i and g_corner.axes[i, j] is not None:  # Upper triangle
                        g_corner.axes[i, j].set_visible(False)
            
            # Apply paired axis limits
            pair_key = 'id_cs' if shift_dir_name == 'in_distribution' else 'ps_ncs'
            print(f" Applying paired limits for {pair_key}...")
            
            for i, method_i in enumerate(method_cols):
                for j, method_j in enumerate(method_cols):
                    if j <= i:  # Lower triangle and diagonal
                        ax = g_corner.axes[i, j]
                        if i == j:  # Diagonal
                            xlim = paired_axis_limits.get((pair_key, method_i), {}).get('xlim')
                            if xlim:
                                ax.set_xlim(xlim)
                        else:  # Off-diagonal
                            xlim = paired_axis_limits.get((pair_key, method_j), {}).get('xlim')
                            ylim = paired_axis_limits.get((pair_key, method_i), {}).get('ylim')
                            if xlim:
                                ax.set_xlim(xlim)
                            if ylim:
                                ax.set_ylim(ylim)
            
            plt.tight_layout()
            output_path_corner = output_dir / f'sample_level_pairplot_{shift_dir_name}_corner_lowerleft.png'
            g_corner.savefig(output_path_corner, dpi=300, bbox_inches='tight')
            print(f" Lower-left corner pairplot saved to {output_path_corner}")
            plt.close()
            
        elif shift_dir_name in ['corruption_shifts', 'new_class_shifts']:
            # Upper-right triangle (custom PairGrid)
            print(f" Creating UPPER-RIGHT corner pairplot (same samples)...")
            
            # Shuffle again to ensure proper z-order mixing
            # Use PairGrid for consistency with lower-right plots
            df_plot = df_complete.sample(frac=1, random_state=42).reset_index(drop=True)

            g_corner = sns.PairGrid(
                df_plot,
                vars=method_cols,
                diag_sharey=False
            )

            def scatter_shuffled(x, y, **kwargs):
                ax = plt.gca()
                idx = x.index  
                colors = df_plot.loc[idx, "Prediction"].map(palette).values

                ax.scatter(
                    x.values,
                    y.values,
                    c=colors,
                    alpha=0.2,
                    s=20,
                    edgecolors="none",
                    rasterized=True
                )

            g_corner.map_upper(scatter_shuffled)

            def diag_kde_by_class(x, **kwargs):
                ax = plt.gca()
                pred = df_plot.loc[x.index, "Prediction"]

                levels = pred.dropna().unique()
                # optionnel: ordre stable si bool ou 0/1
                try:
                    levels = np.sort(levels)
                except Exception:
                    pass

                for lvl in levels:
                    xs = x[pred == lvl]
                    if xs.size < 2:
                        continue
                    sns.kdeplot(
                        x=xs,
                        ax=ax,
                        fill=True,
                        linewidth=2,
                        color=palette.get(lvl, None),
                        alpha=0.35,
                        common_norm=False
                    )

            g_corner.map_diag(diag_kde_by_class)
            
            # Remove unused lower triangle axes
            for i in range(len(method_cols)):
                for j in range(len(method_cols)):
                    if j < i and g_corner.axes[i, j] is not None:  # Lower triangle
                        g_corner.axes[i, j].set_visible(False)
            
            # Apply matching axis limits from paired shift
            pair_key = 'id_cs' if shift_dir_name == 'corruption_shifts' else 'ps_ncs'
            for i, method_i in enumerate(method_cols):
                for j, method_j in enumerate(method_cols):
                    if j > i:  # Upper triangle only
                        ax = g_corner.axes[i, j]
                        # Get stored limits
                        xlim_i = paired_axis_limits.get((pair_key, method_j), {}).get('xlim')
                        ylim_i = paired_axis_limits.get((pair_key, method_i), {}).get('ylim')
                        if xlim_i:
                            ax.set_xlim(xlim_i)
                        if ylim_i:
                            ax.set_ylim(ylim_i)
            
            # Apply limits to diagonal plots
            for i, method in enumerate(method_cols):
                ax_diag = g_corner.axes[i, i]
                xlim = paired_axis_limits.get((pair_key, method), {}).get('xlim')
                if xlim:
                    ax_diag.set_xlim(xlim)
            
            # Add x-ticks on top for first row (all columns)
            for j in range(len(method_cols)):
                if g_corner.axes[0, j] is not None:
                    ax = g_corner.axes[0, j]
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position('top')
                    ax.tick_params(axis='x', which='both', top=True, labeltop=True)
            
            # Add y-ticks on right for last column (all rows)
            for i in range(len(method_cols)):
                if g_corner.axes[i, -1] is not None:
                    ax = g_corner.axes[i, -1]
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position('right')
                    ax.tick_params(axis='y', which='both', left=False, labelleft=False, right=True, labelright=True)
            
            # Add method labels on top of columns and right side of rows
            for i, method in enumerate(method_cols):
                # Top labels for each column
                ax_top = g_corner.axes[0, i]
                ax_top.set_title(method, fontsize=10)
                
                # Right labels for each row using the rightmost plot in that row
                ax_right = g_corner.axes[i, -1]  # Rightmost column
                ax_twin = ax_right.twinx()  # Create secondary y-axis on the right
                ax_twin.set_ylabel(method, fontsize=10, rotation=270, labelpad=15)
                ax_twin.set_yticks([])  # Hide ticks on the twin axis
            
            plt.tight_layout()
            output_path_corner = output_dir / f'sample_level_pairplot_{shift_dir_name}_corner_upperright.png'
            g_corner.savefig(output_path_corner, dpi=300, bbox_inches='tight')
            print(f" Upper-right corner pairplot saved to {output_path_corner}")
            plt.close()

    return shift_dataframes


def create_combined_pairplot(shift_dataframe_a, shift_dataframe_b, output_dir, shift_a_name='ID', shift_b_name='CS'):
    """
    Create a combined pairplot with shift_a data in lower-left and shift_b data in upper-right.
    Both shifts share the same diagonal.
    
    Args:
        shift_dataframe_a: DataFrame for first shift (lower-left)
        shift_dataframe_b: DataFrame for second shift (upper-right)
        output_dir: directory to save plot
        shift_a_name: display name for shift A
        shift_b_name: display name for shift B
    """
    print("\n" + "="*80)
    print(f"Creating combined {shift_a_name} (lower-left) + {shift_b_name} (upper-right) pairplot")
    print("="*80)
    
    if shift_dataframe_a is None or shift_dataframe_b is None or shift_dataframe_a.empty or shift_dataframe_b.empty:
        print(f"\n Skipping combined {shift_a_name}+{shift_b_name} pairplot - missing data")
        return
    
    df_a = shift_dataframe_a
    df_b = shift_dataframe_b
    
    # Get all methods (union from both shifts)
    id_methods = [c for c in df_a.columns if c not in ['dataset', 'model', 'config', 'fold', 'sample_idx', 'correct', 'Prediction']]
    cs_methods = [c for c in df_b.columns if c not in ['dataset', 'model', 'config', 'fold', 'sample_idx', 'correct', 'Prediction']]
    all_methods = list(set(id_methods) | set(cs_methods))  # Union instead of intersection
    
    if len(all_methods) < 2:
        print(" Insufficient methods")
        return
    
    # # Sort alphabetically, but put Mean Agg at the end
    # all_methods = sorted(all_methods)
    # if 'Mean Agg' in all_methods:
    #     all_methods.remove('Mean Agg')
    #     all_methods.append('Mean Agg')
    all_methods = ['MSR', 'MSR-S', 'MLS', 'TTA', 'GPS', 'MCD', 'KNN', 'Mean Agg']
    
    print(f" All methods ({len(all_methods)}): {', '.join(all_methods)}")
    print(f" {shift_a_name} samples: {len(df_a)}, {shift_b_name} samples: {len(df_b)}")
    
    # Shuffle both dataframes - only include columns that exist in each
    id_cols_to_keep = [m for m in all_methods if m in df_a.columns] + ['Prediction']
    cs_cols_to_keep = [m for m in all_methods if m in df_b.columns] + ['Prediction']
    
    df_a_plot = df_a[id_cols_to_keep].sample(frac=1, random_state=42).reset_index(drop=True)
    df_b_plot = df_b[cs_cols_to_keep].sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Define color palette
    palette = {'Correct': 'green', 'Incorrect': 'red'}
    
    # Create figure with extended grid for padding
    # Need grid_size = n_methods + 2 to accommodate both CS and ID distributions
    grid_size = len(all_methods) + 2
    fig, axes = plt.subplots(grid_size, grid_size, 
                             figsize=(2.2*grid_size, 2.5*grid_size))
    
    # Make axes array 2D if needed
    if grid_size == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot grid with proper diagonal alignment
    # For method at index m (0-7):
    #   - Column in grid: m+1 (columns 1-8)
    #   - Row m, column m+1: CS KDE (upper-right diagonal)
    #   - Row m+1, column m+1: Blank with method name (main diagonal)
    #   - Row m+2, column m+1: ID KDE (lower-left diagonal)
    #   - Upper triangle (i < j, excluding KDE): CS scatter
    #   - Lower triangle (i > j, excluding KDE): ID scatter
    
    for i in range(grid_size):
        for j in range(grid_size):
            ax = axes[i, j]
            
            # Column j corresponds to method index (j-1) if j in [1, 8]
            if j < 1 or j > len(all_methods):
                ax.axis('off')
                continue
            
            method_col_idx = j - 1
            method_col = all_methods[method_col_idx]
            
            # Check if this is a CS KDE position: i == method_col_idx
            if i == method_col_idx:
                if method_col not in df_b_plot.columns:
                    ax.axis('off')
                    continue
                
                x_cs = df_b_plot[method_col]
                pred_cs = df_b_plot['Prediction']
                
                for pred_class in ['Correct', 'Incorrect']:
                    xs = x_cs[pred_cs == pred_class]
                    if xs.size >= 2:
                        sns.kdeplot(x=xs, ax=ax, fill=True, linewidth=2, 
                                   color=palette[pred_class], alpha=0.35, 
                                   common_norm=False, linestyle='-')
                
                ax.set_ylabel(shift_b_name, fontsize=11, fontweight='bold', rotation=270, labelpad=10)
                ax.yaxis.set_label_position('right')
                ax.set_yticklabels([])
                

                    
                ax.grid(True, alpha=0.2, linestyle='--')
            
            # Check if this is blank diagonal: i == method_col_idx + 1
            elif i == method_col_idx + 1:
                ax.axis('off')
                ax.text(0.5, 0.5, method_col, 
                       transform=ax.transAxes, 
                       fontsize=11, fontweight='bold',
                       ha='center', va='center')
            
            # Check if this is ID KDE position: i == method_col_idx + 2
            elif i == method_col_idx + 2:
                if method_col not in df_a_plot.columns:
                    ax.axis('off')
                    continue
                
                x_id = df_a_plot[method_col]
                pred_id = df_a_plot['Prediction']
                
                for pred_class in ['Correct', 'Incorrect']:
                    xs = x_id[pred_id == pred_class]
                    if xs.size >= 2:
                        sns.kdeplot(x=xs, ax=ax, fill=True, linewidth=2, 
                                   color=palette[pred_class], alpha=0.35, 
                                   common_norm=False, linestyle='-')
                
                ax.set_ylabel(shift_a_name, fontsize=11, fontweight='bold', labelpad=-7.5)
                ax.yaxis.set_label_position('left')
                ax.set_yticklabels([])
                
                if i == grid_size - 1:
                    ax.set_xlabel(method_col, fontsize=10)
                else:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
                    
                ax.grid(True, alpha=0.2, linestyle='--')
            
            # Upper triangle: CS scatter (i < method_col_idx, excluding special positions)
            elif i < method_col_idx:
                # Row i could be: CS KDE row (i), blank row (i+1), or ID KDE row (i+2) for method at index i
                # We need to get the proper row method
                
                # Check if column i+1 exists and maps to a method
                if i + 1 < 1 or i + 1 > len(all_methods):
                    ax.axis('off')
                    continue
                
                method_row_idx = i
                method_row = all_methods[method_row_idx]
                
                if method_col not in df_b_plot.columns or method_row not in df_b_plot.columns:
                    ax.axis('off')
                    continue
                
                x = df_b_plot[method_col].values
                y = df_b_plot[method_row].values
                colors = df_b_plot['Prediction'].map(palette).values
                
                ax.scatter(x, y, c=colors, alpha=0.2, s=20, edgecolor='none', rasterized=True)
                
                if method_row_idx == 0:
                    # Labels
                    ax.set_xlabel(method_col, fontsize=10)
                    ax.xaxis.set_label_position('top')
                    ax.xaxis.tick_top()
                
                if j == len(all_methods):
                    ax.set_ylabel(method_row, fontsize=10)
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.tick_right()
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
                    
                ax.grid(True, alpha=0.2, linestyle='--')
            
            # Lower triangle: ID scatter (i > method_col_idx + 2, excluding special positions)
            elif i > method_col_idx + 2:
                # Map row to method: row i corresponds to method at i-2
                method_row_idx = i - 2
                if method_row_idx >= len(all_methods):
                    ax.axis('off')
                    continue
                
                method_row = all_methods[method_row_idx]
                
                if method_col not in df_a_plot.columns or method_row not in df_a_plot.columns:
                    ax.axis('off')
                    continue
                
                x = df_a_plot[method_col].values
                y = df_a_plot[method_row].values
                colors = df_a_plot['Prediction'].map(palette).values
                
                ax.scatter(x, y, c=colors, alpha=0.2, s=20, edgecolor='none', rasterized=True)
                
                # Labels
                if j == 1:
                    ax.set_ylabel(method_row, fontsize=10)
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
                
                if i == grid_size - 1:
                    ax.set_xlabel(method_col, fontsize=10)
                else:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
                    
                ax.grid(True, alpha=0.2, linestyle='--')
            
            else:
                ax.axis('off')
    if shift_a_name == 'ID':
        shift_a_name_full = 'In-Distribution'
    elif shift_a_name == 'PS':
        shift_a_name_full = 'Population Shifts'
    if shift_b_name == 'CS':
        shift_b_name_full = 'Corruption Shifts'
    elif shift_b_name == 'NCS':
        shift_b_name_full = 'New Class Shifts'

    # Add vertical axis titles instead of box legend
    # Left side: shift A name
    fig.text(0.125, 0.25, shift_a_name_full, 
             transform=fig.transFigure, fontsize=18, fontweight='bold',
             verticalalignment='center', rotation=90)
    
    # Right side: shift B name (reversed)
    fig.text(0.92, 0.75, shift_b_name_full, 
             transform=fig.transFigure, fontsize=18, fontweight='bold',
             verticalalignment='center', rotation=270)
    
    # Make each method column share the same x-axis across both ID and CS
    # (same limits for the same method in both shift evaluations)
    for j in range(1, len(all_methods) + 1):
        # Collect x-limits from all subplots in this column
        col_xlims = []
        for i in range(grid_size):
            ax = axes[i, j]
            try:
                # Check if axis has content (collections from scatter plots or lines from KDE)
                if len(ax.collections) > 0 or len(ax.lines) > 0:
                    col_xlims.append(ax.get_xlim())
            except:
                pass
        
        if col_xlims:
            col_xmin = min(lim[0] for lim in col_xlims)
            col_xmax = max(lim[1] for lim in col_xlims)
            col_xlim = (col_xmin, col_xmax)
            
            # Apply to all subplots in this column
            for i in range(grid_size):
                ax = axes[i, j]
                try:
                    if len(ax.collections) > 0 or len(ax.lines) > 0:
                        ax.set_xlim(col_xlim)
                except:
                    pass
    
    # Adjust spacing to maximize subplot size
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.98, wspace=0.15, hspace=0.15)
    
    output_path = output_dir / f'sample_level_pairplot_{shift_a_name}_{shift_b_name}_combined.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n   Combined {shift_a_name}+{shift_b_name} pairplot saved to {output_path}")
    
    plt.close()


def main():
    """Main function."""
    print("=" * 80)
    print("Balanced Accuracy vs AUROC_f Scatter Plot Generator")
    print("=" * 80)
    
    # Get workspace root
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent.parent
    
    print(f"Workspace root: {workspace_root}")
    
    # Create output directory
    output_dir = script_dir / 'scatter_plots'
    output_dir.mkdir(exist_ok=True)
    
    # Collect all data points from pre-computed results
    data = collect_all_data_points(workspace_root)
    
    # # Create scatter plots
    # create_scatter_plot(data, output_dir)
    
    # Create per-method correlation plots
    create_per_method_correlation_plots(data, output_dir)
    
    # # Create method correlation pairplots
    # create_method_correlation_pairplots(data, output_dir)
    
    # Create sample-level pairplots
    shift_dataframes = create_sample_level_pairplots(workspace_root, output_dir)
    
    # Create combined pairplots
    if 'in_distribution' in shift_dataframes and 'corruption_shifts' in shift_dataframes:
        create_combined_pairplot(shift_dataframes['in_distribution'], shift_dataframes['corruption_shifts'], output_dir, 'ID', 'CS')
    
    if 'population_shifts' in shift_dataframes and 'new_class_shifts' in shift_dataframes:
        create_combined_pairplot(shift_dataframes['population_shifts'], shift_dataframes['new_class_shifts'], output_dir, 'PS', 'NCS')
    
    print("\n" + "=" * 80)
    print(" All plots generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
