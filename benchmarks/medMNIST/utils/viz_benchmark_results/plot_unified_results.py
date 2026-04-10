"""
Unified visualization of UQ benchmark results across all shift types.

Creates 3 figures:
1. AUROC_f Radar Plots: 3 rows (ID, CS, PS/NCS) × 2 columns (ResNet18, ViT)
2. AUGRC Radar Plots: Same layout as AUROC_f
3. Heatmaps: 3 rows (ID, CS, PS/NCS) × 2 columns (AUROC_f, AUGRC)
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import functions from existing scripts
from generate_radar_plots import create_radar_plot_on_axis, parse_results_directory


def compute_metric_means(results_dict, metric='auroc_f'):
    """
    Compute mean metric values for each method across all datasets.
    
    Args:
        results_dict: Dict structure:
            {
                'id': {
                    'resnet18': {
                        'breastmnist_standard': {'MSR': 0.85, 'GPS': 0.90, ...},
                        'pneumoniamnist_standard': {...},
                        ...
                    },
                    'vit_b_16': {...}
                },
                'corruption': {...},
                'population': {...}
            }
        metric: 'auroc_f' or 'augrc'
    
    Returns:
        Dict structure:
            {
                'id': {
                    'resnet18': {
                        'auroc_f' or 'augrc': {'MSR': 0.85, 'GPS': 0.90, ...}
                    },
                    'vit_b_16': {...}
                },
                'corruption': {...},
                'population': {...}
            }
    """
    all_means = {}
    
    shift_keys = ['id', 'corruption', 'population']
    model_names = ['resnet18', 'vit_b_16']
    
    print(f"\nComputing {metric.upper()} means...")
    
    for shift_key in shift_keys:
        results = results_dict.get(shift_key, {})
        
        if not results:
            print(f" No results for {shift_key}")
            continue
        
        all_means[shift_key] = {}
        
        for model_name in model_names:
            if model_name not in results:
                print(f" No {model_name} results for {shift_key}")
                continue
            
            model_results = results[model_name]
            
            # Get all unique methods across datasets
            all_methods = set()
            for dataset_data in model_results.values():
                all_methods.update(dataset_data.keys())
            all_methods = sorted(all_methods)
            
            # Get all dataset keys
            dataset_keys = list(model_results.keys())
            
            print(f" {shift_key}/{model_name}: {len(dataset_keys)} datasets, {len(all_methods)} methods")
            
            # Compute mean values and std for each method
            method_means = {}
            method_stds = {}
            for method_name in all_methods:
                values = []
                for dataset_key in dataset_keys:
                    val = model_results[dataset_key].get(method_name, np.nan)
                    # For AUGRC: invert so that 0 (best) -> 0.5 (edge) and 0.5 (worst) -> 0 (center)
                    if metric == 'augrc' and not np.isnan(val):
                        val = 0.5 - val
                    values.append(val)
                
                # Filter out NaN values
                valid_values = [v for v in values if not np.isnan(v)]
                
                if len(valid_values) == 0:
                    method_means[method_name] = 0.0
                    method_stds[method_name] = 0.0
                    continue
                
                # Compute mean and std (no normalization)
                method_means[method_name] = np.mean(valid_values)
                method_stds[method_name] = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
            
            # Store method means and stds under metric key
            all_means[shift_key][model_name] = {
                metric: method_means,
                f'{metric}_std': method_stds
            }
    
    print(f" Computed means for {len(all_means)} shift types")
    return all_means


def load_and_parse_results(results_dirs, metric='auroc_f'):
    """
    Load and parse results from multiple directories.
    
    Args:
        results_dirs: Dict with keys 'id', 'corruption', 'population', 'new_class'
        metric: Metric to extract ('auroc_f' or 'augrc')
    
    Returns:
        Dict with keys 'id', 'corruption', 'population' (merged pop + new_class)
    """
    parsed = {}
    
    # Load ID and corruption shifts normally
    for shift_type in ['id', 'corruption']:
        dir_path = results_dirs.get(shift_type)
        if dir_path is not None and dir_path.exists():
            parsed[shift_type] = parse_results_directory(dir_path, metric=metric)
        else:
            parsed[shift_type] = {}
    
    # Load and merge population and new_class shifts
    pop_results = {}
    new_class_results = {}
    
    if results_dirs.get('population') is not None and results_dirs['population'].exists():
        pop_results = parse_results_directory(results_dirs['population'], metric=metric)
    
    if results_dirs.get('new_class') is not None and results_dirs['new_class'].exists():
        new_class_results = parse_results_directory(results_dirs['new_class'], metric=metric)
    
    # Merge population and new_class (prefix new_class datasets with "new_class_")
    merged = {}
    for model in ['resnet18', 'vit_b_16']:
        merged[model] = {}
        # Add population shift datasets
        if model in pop_results:
            for dataset in pop_results[model]:
                merged[model][dataset] = pop_results[model][dataset]
        # Add new class shift datasets (prefix with "new_class_")
        if model in new_class_results:
            for dataset in new_class_results[model]:
                new_dataset_name = f"new_class_{dataset}"
                merged[model][new_dataset_name] = new_class_results[model][dataset]
    
    parsed['population'] = merged
    
    return parsed


def _load_json_files(dir_path):
    """Load all JSON files from a directory."""
    json_files = list(dir_path.glob('uq_benchmark_*.json'))
    all_data = {}
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Parse filename: uq_benchmark_{dataset}_{model}_{setup}_{timestamp}.json
            filename = json_file.stem
            parts = filename.replace('uq_benchmark_', '').rsplit('_', 1)
            
            if len(parts) == 2:
                prefix, timestamp = parts
                prefix_parts = prefix.split('_')
                
                if len(prefix_parts) >= 2:
                    setup = prefix_parts[-1]
                    model = prefix_parts[-2]
                    dataset = '_'.join(prefix_parts[:-2])
                    key = f"{dataset}_{model}_{setup}"
                    all_data[key] = data
    
    return all_data

def create_radar_figure(results_auroc, results_augrc, metric='auroc_f', aggregation='mean', 
                        results_dir=None, comp_eval_dirs=None):
    """
    Create unified radar plot figure for one metric (AUROC_f or AUGRC).
    
    Layout: 3 rows (ID, CS, PS/NCS) × 2 columns (ResNet18, ViT)
    
    Args:
        results_auroc: Dict with keys 'id', 'corruption', 'population'
        results_augrc: Dict with keys 'id', 'corruption', 'population'
        metric: 'auroc_f' or 'augrc'
        aggregation: Aggregation strategy
        results_dir: Main results directory
        comp_eval_dirs: Dict of comprehensive evaluation directories
        
    Returns:
        fig: matplotlib figure
    """
    results_map = results_auroc if metric == 'auroc_f' else results_augrc
    
    # Create figure with 3 rows and 2 columns
    fig = plt.figure(figsize=(16, 21))
    gs = fig.add_gridspec(3, 2, hspace=0.15, wspace=0.30,
                         left=0.10, right=0.90, top=0.96, bottom=0.04)
    
    # Create all radar axes
    radar_axes = []
    for row in range(3):
        for col in range(2):
            ax = fig.add_subplot(gs[row, col], projection='polar')
            radar_axes.append(ax)
    
    # Define shift types and their settings
    shift_configs = [
        ('id', 'in_distribution', 'In Distribution', 0),
        ('corruption', 'corruption_shifts', 'Corruption Shifts', 1),
        ('population', 'population_shift', 'Population / New Class Shift', 2)
    ]
    
    model_names = ['resnet18', 'vit_b_16']
    
    all_handles = []
    all_labels = []
    
    print(f"\nGenerating {metric.upper()} radar plots...")
    
    for shift_key, shift_name, shift_label, row in shift_configs:
        results = results_map.get(shift_key, {})
        
        if not results:
            print(f" No results for {shift_key}")
            continue
        
        # Get comprehensive evaluation directory
        comp_eval_dir = comp_eval_dirs.get(shift_key) if comp_eval_dirs else None
        
        # Add shift type label centered between the two columns
        if row == 0:
            fig.text(0.5, 0.967, shift_label.upper(), ha='center', va='center', 
                    fontsize=15, fontweight='bold')
        elif row == 1:
            fig.text(0.5, 0.646, shift_label.upper(), ha='center', va='center', 
                    fontsize=15, fontweight='bold')
        else:  # row == 2
            shift_label = 'POPULATION /\n NEW CLASS SHIFTS'
            fig.text(0.5, 0.319, shift_label, ha='center', va='center', 
                    fontsize=15, fontweight='bold')
        
        for col, model_name in enumerate(model_names):
            ax_idx = row * 2 + col
            ax = radar_axes[ax_idx]
            
            if model_name not in results:
                print(f" No {model_name} results for {shift_key}")
                continue
            
            model_results = results[model_name]
            
            # Generate radar plot (uses pre-computed Mean_Aggregation from JSON)
            handles, labels, method_surfaces, method_angles = create_radar_plot_on_axis(
                ax, model_results, model_name,
                runs_dir=comp_eval_dir,
                metric=metric, aggregation=aggregation, shift=shift_name
            )
            
            # Collect legend info from first subplot
            if ax_idx == 0 and handles and labels:
                all_handles = handles
                all_labels = labels
            
            # Add subplot title
            model_display = 'RESNET18' if model_name == 'resnet18' else 'VIT_B_16'
            metric_display = 'AUROC F' if metric == 'auroc_f' else 'AUGRC'
            if model_name == 'resnet18':
                ax.set_title(f'{model_display}\nPer-fold {aggregation.capitalize()}\n{metric_display}',
                            fontsize=14, fontweight='bold', y=0.95, x=-0.15, ha='left')
            else:
                ax.set_title(f'{model_display}\nPer-fold {aggregation.capitalize()}\n{metric_display}',
                            fontsize=14, fontweight='bold', pad=0.95, x=1.15, ha='right')
    
    # Add legend
    if all_handles and all_labels:
        # Move "Mean_Aggregation" to bottom of legend
        if 'ZScore_Aggregation_per_fold' in all_labels:
            idx = all_labels.index('ZScore_Aggregation_per_fold')
            all_labels.append(all_labels.pop(idx))
            all_handles.append(all_handles.pop(idx))
        
        # Rename method labels
        all_labels = [label.replace('KNN_Raw', 'KNN')
                           .replace('Ensembling', 'DE')
                           .replace('MCDropout', 'MCD')
                           .replace('MSR_calibrated', 'MSR-S')
                           .replace('ZScore_Aggregation_per_fold', 'Mean Agg')
                      for label in all_labels]
        
        fig.legend(all_handles, all_labels, loc='center', ncol=2,
                  fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.70))
    
    return fig


def create_combined_radar_histogram_figure_id_cs(results_auroc, results_augrc, all_means, 
                                           aggregation='mean', results_dir=None, comp_eval_dirs=None, metric='auroc_f'):
    """
    Create combined radar + histogram figure for ID and CS (Figure 1).
    
    Layout: 3 rows × 2 columns
    - Row 0: ID radars (ResNet18 left, ViT right)
    - Row 1: Histograms (ID left, CS right)
    - Row 2: CS radars (ResNet18 left, ViT right)
    
    Args:
        results_auroc: Dict with keys 'id', 'corruption', 'population'
        results_augrc: Dict with keys 'id', 'corruption', 'population' (for radar plotting)
        all_means: Mean radar values dict from radar computation
        aggregation: Aggregation strategy
        results_dir: Main results directory
        comp_eval_dirs: Dict of comprehensive evaluation directories
        metric: 'auroc_f' or 'augrc'
        
    Returns:
        fig: matplotlib figure
    """
    results_map = results_auroc if metric == 'auroc_f' else results_augrc
    
    # Create figure with 2 rows × 3 columns
    # Row 0: ID (ResNet18 radar, histogram, ViT radar)
    # Row 1: CS (ResNet18 radar, histogram, ViT radar)
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.05,
                         height_ratios=[1, 1],  # 2 equal rows for ID and CS
                         width_ratios=[1, 0.3, 1],  # Cols: R18 radar, histogram, ViT radar
                         left=0.08, right=0.96, top=0.86, bottom=0.10)
    
    # Create radar axes in columns 0 and 2 for both rows
    radar_axes = {}
    for row in range(2):
        for col in [0, 2]:  # Radars in columns 0 and 2
            ax = fig.add_subplot(gs[row, col], projection='polar')
            radar_axes[(row, col)] = ax
    
    # Define shift types and their settings (only ID and CS)
    shift_configs = [
        ('id', 'in_distribution', 'ID', 0),  # row 0
        ('corruption', 'corruption_shifts', 'CS', 1)  # row 1
    ]
    
    model_names = ['resnet18', 'vit_b_16']
    model_display_names = {
        'resnet18': 'ResNet18',
        'vit_b_16': 'ViT-B/16'
    }
    
    all_handles = []
    all_labels = []
    
    metric_display = 'AUROC_f' if metric == 'auroc_f' else 'AUGRC'
    print(f"\nGenerating ID+CS radar + histogram figure ({metric_display})...")
    
    # Add row labels at left
    fig.text(0.10, 0.70, 'ID', ha='center', va='center', fontsize=18, fontweight='bold', rotation=90)
    fig.text(0.10, 0.27, 'CS', ha='center', va='center', fontsize=18, fontweight='bold', rotation=90)
    
    # Add column titles at top
    fig.text(0.17, 0.92, f'ResNet18\n{metric_display}', ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.52, 0.92, 'Mean Radar Value', ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.68, 0.92, f'ViT-B/16\n{metric_display}', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Plot radars for both models × both shifts
    plot_idx = 0
    for shift_key, shift_name, shift_label, row in shift_configs:
        results = results_map.get(shift_key, {})
        
        if not results:
            continue
        
        # Get comprehensive evaluation directory
        comp_eval_dir = comp_eval_dirs.get(shift_key) if comp_eval_dirs else None
        
        for model_idx, model_name in enumerate(model_names):
            if model_name not in results:
                continue
            
            col = 0 if model_idx == 0 else 2  # Column 0 for ResNet18, 2 for ViT
            ax = radar_axes[(row, col)]
            
            # Call create_radar_plot_on_axis (from generate_radar_plots.py)
            result = create_radar_plot_on_axis(
                ax=ax,
                model_results=results[model_name],
                model_name=model_name,
                runs_dir=comp_eval_dir,
                metric=metric,
                aggregation=aggregation,
                shift=shift_name
            )
            
            # Handle case where no data is available
            if result is None:
                print(f" No data returned for {model_name}/{shift_key}/{metric}")
                continue
            
            handles, labels, method_surfaces, method_angles = result
            
            # Store method means for histogram generation
            if shift_key not in all_means:
                all_means[shift_key] = {}
            if model_name not in all_means[shift_key]:
                all_means[shift_key][model_name] = {}
            all_means[shift_key][model_name][metric] = method_surfaces
            
            # Collect handles and labels for legend
            if handles and labels and plot_idx == 0:
                all_handles = handles
                all_labels = labels
            
            plot_idx += 1
    
    # Plot histograms in row 1 (one for each shift)
    # Get method order and colors (same as radar plots)
    method_display_order = {
        'MSR': 9,
        'MSR_calibrated': 8,
        'MLS': 7,
        'TTA': 6,
        'GPS': 5,
        'MCDropout': 4,
        'KNN_Raw': 3,
        'Ensembling': 2,
        'ZScore_Aggregation_per_fold': 1,
        'ZScore_Aggregation_ensemble': 0
    }
    
    # Get all methods across all shifts for color mapping
    all_methods_set = set()
    for shift_data in all_means.values():
        for model_data in shift_data.values():
            all_methods_set.update(model_data.get(metric, {}).keys())
    
    # Sort methods for color mapping
    all_methods_sorted_for_colors = sorted(all_methods_set)
    all_methods_sorted_for_colors.append(all_methods_sorted_for_colors.pop(7))
    all_methods_sorted_for_colors.insert(7, all_methods_sorted_for_colors.pop(8))
    # Create color mapping matching radar plots
    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, len(all_methods_sorted_for_colors)))
    method_colors = {method: colors_tab20[i] for i, method in enumerate(all_methods_sorted_for_colors)}
    
    # Override Mean_Aggregation to red
    method_colors['ZScore_Aggregation_per_fold'] = 'red'
    
    # Create histogram for each shift (one per row, in column 1 - middle)
    for shift_key, shift_name, shift_label, row in shift_configs:
        shift_data = all_means.get(shift_key, {})
        
        if not shift_data:
            continue
        
        # Create histogram in column 1 (middle) for this row
        ax = fig.add_subplot(gs[row, 1])
        
        # Get data for both models
        resnet_data = shift_data.get('resnet18', {})
        vit_data = shift_data.get('vit_b_16', {})
        
        # Get mean values and stds for the specified metric for both models
        resnet_metric = resnet_data.get(metric, {})
        vit_metric = vit_data.get(metric, {})
        resnet_std = resnet_data.get(f'{metric}_std', {})
        vit_std = vit_data.get(f'{metric}_std', {})
        
        # Get all methods (union of both models)
        all_methods = sorted(set(list(resnet_metric.keys()) + list(vit_metric.keys())), 
                           key=lambda m: method_display_order.get(m, 999))
        
        if not all_methods:
            continue
        

        # Prepare data for both models
        resnet_values = [resnet_metric.get(m, 0.0) for m in all_methods]
        vit_values = [vit_metric.get(m, 0.0) for m in all_methods]
        resnet_errors = [resnet_std.get(m, 0.0) for m in all_methods]
        vit_errors = [vit_std.get(m, 0.0) for m in all_methods]
        
        # Get colors for each method
        bar_colors = [method_colors.get(m, 'gray') for m in all_methods]
        
        # Create offset bar positions with reduced spacing (horizontal bars now)
        y = np.arange(len(all_methods)) * 0.07  # Very tight spacing between bars
        bar_height = 0.04  # Narrow bars
        offset = bar_height * 0.2  # Offset to separate error bars
        
        # Plot ResNet18 on primary (bottom) x-axis - horizontal bars going right
        bars_resnet = ax.barh(y-offset, resnet_values, height=bar_height, color=bar_colors, 
                         alpha=0.8, edgecolor='black', linewidth=0.5, label='ResNet18')
        ax.errorbar(resnet_values, y - offset, xerr=resnet_errors, fmt='none', ecolor='#2E4057',
                   elinewidth=0.8, capsize=0, capthick=0.8, alpha=0.7, zorder=10)
        
        # Create secondary (top) x-axis for ViT - bars going left from top (reversed axis)
        ax2 = ax.twiny()
        bars_vit = ax2.barh(y+offset, vit_values, height=bar_height, color=bar_colors,
                             alpha=0.8, edgecolor='black', linewidth=0.5, label='ViT')
        ax2.errorbar(vit_values, y + offset, xerr=vit_errors, fmt='none', ecolor='#2E4057',
                    elinewidth=0.8, capsize=0, capthick=0.8, alpha=0.7, zorder=10)
        
        # Format x-axes based on metric (horizontal bars now)
        if metric == 'augrc':
            # AUGRC: Raw AUGRC values (0-0.5, lower is better)
            ax.set_xlim(0.35, 0.65)
            ax.set_xlabel('ResNet18', fontsize=12, fontweight='bold', x=0.28)
            
            resnet_ticks = [0.5, 0.475, 0.45, 0.425, 0.4, 0.375, 0.35]
            resnet_labels = ['0%', '', '5%', '', '10%', '', '15%']
            ax.set_xticks(resnet_ticks)
            ax.set_xticklabels(resnet_labels, fontsize=9)
            ax.tick_params(axis='x', which='both', length=0)  # Remove tick marks
            
            # Format top x-axis (ViT - reversed so bars meet in center)
            ax2.set_xlim(0.65, 0.35)  # Reversed
            ax2.set_xlabel('ViT', fontsize=12, fontweight='bold', x=0.75)
            
            vit_ticks = [0.35, 0.375, 0.4, 0.425, 0.45, 0.475,0.5]
            vit_labels = ['15%', '', '10%', '', '5%', '', '0%']
            ax2.set_xticks(vit_ticks)
            ax2.set_xticklabels(vit_labels, fontsize=9)
            ax2.tick_params(axis='x', which='both', top=True, labeltop=True, pad=6, length=0)  # Remove tick marks
            
            # Separator line at 0.2
            separator_x = 0.5
        else:
            # AUROC_f: Raw AUROC_f values (0.4-1.0)
            ax.set_xlim(0.6, 1.2)
            ax.set_xlabel('ResNet18', fontsize=12, fontweight='bold', x=0.28)
            
            resnet_ticks = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
            resnet_labels = ['.6', '', '.7', '', '.8', '', '.9']
            ax.set_xticks(resnet_ticks)
            ax.set_xticklabels(resnet_labels, fontsize=9)
            ax.tick_params(axis='x', which='both', length=0)  # Remove tick marks
            
            # Format top x-axis (ViT - reversed so bars meet in center)
            ax2.set_xlim(1.2, 0.6)  # Reversed
            ax2.set_xlabel('ViT', fontsize=12, fontweight='bold', x=0.75)
            
            vit_ticks = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
            vit_labels = ['.9', '', '.8', '', '.7', '', '.6']
            ax2.set_xticks(vit_ticks)
            ax2.set_xticklabels(vit_labels, fontsize=9)
            ax2.tick_params(axis='x', which='both', top=True, labeltop=True, pad=6, length=0)  # Remove tick marks
            
            separator_x = 0.9
        
        # Add visual separator line (now vertical)
        ax.axvline(x=separator_x, color='lightgray', linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)
        
        # Remove y-tick labels and set tight y-axis limits
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', length=0)  # Remove y-axis tick marks
        ax2.tick_params(axis='y', which='both', length=0)  # Remove y-axis tick marks
        
        # Set y-axis limits to fit bars tightly with small margins
        if len(y) > 0:
            margin = bar_height * 3.5
            ax.set_ylim(y[0] - margin, y[-1] + margin)
            ax2.set_ylim(y[0] - margin, y[-1] + margin)
        
        # Grid on primary axis only (now horizontal grid for vertical layout)
        ax.grid(axis='x', alpha=0.5, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax2.grid(axis='x', alpha=0.5, linestyle='--', zorder=0)
        ax2.set_axisbelow(True)
        
        # Hide spines strategically
        for spine in ax.spines.values():
            if spine.spine_type in ['right', 'bottom', 'top']:
                spine.set_visible(False)
        # Keep right spine for ax visible
        ax.spines['left'].set_visible(True)
        
        # For ax2, hide everything except top spine
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.spines['right'].set_visible(True)
        ax2.spines['right'].set_linewidth(0.8)
        ax2.spines['right'].set_color('black')
        
        # Draw custom spine segment for bottom axis only (ResNet18)
        if metric == 'augrc':
            # AUGRC: spine from 0.0 to 0.5
            ax.plot([0.35, 0.5], [0, 0], 
                    color='black', linewidth=0.8, clip_on=False, zorder=100, 
                    transform=ax.get_xaxis_transform())
            ax2.plot([0.35, 0.5], [1, 1], 
                    color='black', linewidth=0.8, clip_on=False, zorder=100, 
                    transform=ax2.get_xaxis_transform())
        else:
            # AUROC_f: spine from 0.4 to 1.0
            ax.plot([0.6, 0.9], [0, 0], 
                    color='black', linewidth=0.8, clip_on=False, zorder=100, 
                    transform=ax.get_xaxis_transform())
            # AUROC_f: spine from 0.4 to 1.0
            ax2.plot([0.6, 0.9], [1, 1], 
                    color='black', linewidth=0.8, clip_on=False, zorder=100, 
                    transform=ax2.get_xaxis_transform())
    
    # Add legend at the bottom
    if all_handles and all_labels:
        # Reorder legend items to match method_display_order
        # Create tuples of (original_label, handle) and sort by display order
        label_handle_pairs = list(zip(all_labels, all_handles))
        sorted_pairs = sorted(label_handle_pairs, 
                             key=lambda x: method_display_order.get(x[0], 999))
        
        if sorted_pairs:
            all_labels, all_handles = zip(*sorted_pairs)
            all_labels = list(all_labels)
            all_handles = list(all_handles)
        
        # Rename method labels AFTER sorting
        all_labels = [label.replace('KNN_Raw', 'KNN')
                           .replace('Ensembling', 'DE')
                           .replace('MCDropout', 'MCD')
                           .replace('MSR_calibrated', 'MSR-S')
                           .replace('ZScore Agg + Ens', 'Mean Agg + Ens')
                           .replace('ZScore_Aggregation_per_fold', 'Mean Agg')
                      for label in all_labels]
        
        # Manually reorder for ncol=5 column-wise filling
        if len(all_labels) == 10:
            reordered_labels = [all_labels[i] for i in [7, 3, 8, 2, 6, 1, 5, 0, 4, 9]]
            reordered_handles = [all_handles[i] for i in [7, 3, 8, 2, 6, 1, 5, 0, 4, 9]]
        else:
            reordered_labels = all_labels
            reordered_handles = all_handles
        
        fig.legend(reordered_handles, reordered_labels, loc='lower center', ncol=5,
                  fontsize=11, frameon=True, bbox_to_anchor=(0.52, 0.455), framealpha=1)
    
    return fig


def create_combined_radar_histogram_figure_ncs_ps(results_auroc, results_augrc, all_means, 
                                           aggregation='mean', results_dir=None, comp_eval_dirs=None, metric='auroc_f'):
    """
    Create combined radar + histogram figure for NCS and PS combined (Figure 2).
    
    Layout: 2 rows × 3 columns (same as ID+CS figure)
    - Row 0: AUROC_F → ResNet18 radar, histogram, ViT radar
    - Row 1: AUGRC → ResNet18 radar, histogram, ViT radar
    
    Args:
        results_auroc: Dict with keys 'id', 'corruption', 'population'
        results_augrc: Dict with keys 'id', 'corruption', 'population' (for radar plotting)
        all_means: Mean radar values dict from radar computation
        aggregation: Aggregation strategy
        results_dir: Main results directory
        comp_eval_dirs: Dict of comprehensive evaluation directories
        metric: 'auroc_f' or 'augrc' (ignored - both metrics are shown)
        
    Returns:
        fig: matplotlib figure
    """
    # Get combined population data (already merged in main)
    combined_results_auroc = results_auroc.get('population', {})
    combined_results_augrc = results_augrc.get('population', {})
    
    if not combined_results_auroc and not combined_results_augrc:
        print(f" Warning: No population (NCS+PS) results found")
        return None
    
    # Create figure with 2 rows × 3 columns (matching ID+CS layout)
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.05,
                         height_ratios=[1, 1],  # Two equal rows
                         width_ratios=[1, 0.3, 1],  # ResNet18 radar, histogram, ViT radar
                         left=0.08, right=0.96, top=0.86, bottom=0.10)
    
    # Create radar axes: rows 0,1 cols 0,2 (skip middle column 1 for histograms)
    radar_axes = {}
    for row in [0, 1]:
        for col in [0, 2]:  # Columns 0 and 2 for radars
            ax = fig.add_subplot(gs[row, col], projection='polar')
            radar_axes[(row, col)] = ax
    
    model_names = ['resnet18', 'vit_b_16']
    model_display_names = {
        'resnet18': 'ResNet18',
        'vit_b_16': 'ViT-B/16'
    }
    
    all_handles = []
    all_labels = []
    
    print(f"\\nGenerating NCS+PS combined radar + histogram figure (both metrics)...")
    
    # Add row labels at left (matching ID+CS position)
    fig.text(0.10, 0.70, 'AUROC_f', ha='center', va='center', fontsize=18, fontweight='bold', rotation=90)
    fig.text(0.10, 0.27, 'AUGRC', ha='center', va='center', fontsize=18, fontweight='bold', rotation=90)
    
    # Add column titles at top (matching ID+CS)
    fig.text(0.17, 0.92, 'ResNet18\nNCS + PS', ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.52, 0.92, 'Mean Radar Value', ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.68, 0.92, 'ViT-B/16\nNCS + PS', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Plot radars for both metrics and both models
    metrics_config = [
        ('auroc_f', combined_results_auroc, 'AUROC_f', 0),  # row 0
        ('augrc', combined_results_augrc, 'AUGRC', 1)       # row 1
    ]
    
    for metric_key, combined_results, metric_display, row in metrics_config:
        if not combined_results:
            continue
            
        for col_idx, model_name in enumerate(model_names):
            if model_name not in combined_results:
                continue
            
            # Map model index to grid column (0->0, 1->2, skipping middle column 1)
            col = 0 if col_idx == 0 else 2
            ax = radar_axes[(row, col)]
            
            # Get comprehensive evaluation directory (use population_shift as representative)
            comp_eval_dir = comp_eval_dirs.get('population') if comp_eval_dirs else None
            
            # Call create_radar_plot_on_axis (from generate_radar_plots.py)
            result = create_radar_plot_on_axis(
                ax=ax,
                model_results=combined_results[model_name],
                model_name=model_name,
                runs_dir=comp_eval_dir,
                metric=metric_key,
                aggregation=aggregation,
                shift='population_shift'  # Use as representative
            )
            
            # Handle case where no data is available
            if result is None:
                print(f" No data returned for {model_name}/population/{metric_key}")
                continue
            
            handles, labels, method_surfaces, method_angles = result
            
            # Store method means for histogram generation
            if 'population' not in all_means:
                all_means['population'] = {}
            if model_name not in all_means['population']:
                all_means['population'][model_name] = {}
            all_means['population'][model_name][metric_key] = method_surfaces
            
            # Collect handles and labels for legend (from first radar only)
            if handles and labels and row == 0 and col == 0:
                all_handles = handles
                all_labels = labels
    
    # Plot histograms in middle column (col 1) - one for each row/metric
    # Get method order and colors (same as radar plots)
    method_display_order = {
        'MSR': 9,
        'MSR_calibrated': 8,
        'MLS': 7,
        'TTA': 6,
        'GPS': 5,
        'MCDropout': 4,
        'KNN_Raw': 3,
        'Ensembling': 2,
        'ZScore_Aggregation_per_fold': 1,
        'ZScore_Aggregation_ensemble': 0
    }
    
    # Get all methods across all shifts for color mapping
    all_methods_set = set()
    for shift_data in all_means.values():
        for model_data in shift_data.values():
            for metric_key in ['auroc_f', 'augrc']:
                all_methods_set.update(model_data.get(metric_key, {}).keys())
    
    # Sort methods ALPHABETICALLY for consistent color mapping
    all_methods_sorted_for_colors = sorted(all_methods_set)
    all_methods_sorted_for_colors.append(all_methods_sorted_for_colors.pop(7))
    all_methods_sorted_for_colors.insert(7, all_methods_sorted_for_colors.pop(8))
    
    # Create color mapping matching radar plots
    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, len(all_methods_sorted_for_colors)))
    method_colors = {method: colors_tab20[i] for i, method in enumerate(all_methods_sorted_for_colors)}
    
    # Override Mean_Aggregation to red
    method_colors['ZScore_Aggregation_per_fold'] = 'red'
    
    # Get population shift data (which contains both PS and NCS)
    shift_data = all_means.get('population', {})
    
    if not shift_data:
        print(" Warning: No population mean data for histogram")
    else:
        # Get data for both models
        resnet_data = shift_data.get('resnet18', {})
        vit_data = shift_data.get('vit_b_16', {})
        
        # Create histograms for both metrics (one per row in column 1)
        metrics_config = [
            ('auroc_f', 'AUROC_f', 0),  # row 0
            ('augrc', 'AUGRC', 1)        # row 1
        ]
        
        for metric_key, metric_display, row in metrics_config:
            # Get mean values and stds for this metric for both models
            resnet_metric = resnet_data.get(metric_key, {})
            vit_metric = vit_data.get(metric_key, {})
            resnet_std = resnet_data.get(f'{metric_key}_std', {})
            vit_std = vit_data.get(f'{metric_key}_std', {})
            
            # Get all methods (union of both models)
            all_methods = sorted(set(list(resnet_metric.keys()) + list(vit_metric.keys())), 
                               key=lambda m: method_display_order.get(m, 999))
            
            if not all_methods:
                continue
            
            # Create histogram axis in column 1 (middle)
            ax = fig.add_subplot(gs[row, 1])
            
            # Prepare data for both models
            resnet_values = [resnet_metric.get(m, 0.0) for m in all_methods]
            vit_values = [vit_metric.get(m, 0.0) for m in all_methods]
            resnet_errors = [resnet_std.get(m, 0.0) for m in all_methods]
            vit_errors = [vit_std.get(m, 0.0) for m in all_methods]
            
            # Get colors for each method
            bar_colors = [method_colors.get(m, 'gray') for m in all_methods]
            
            # Create offset bar positions with reduced spacing (horizontal bars)
            y = np.arange(len(all_methods)) * 0.07  # Very tight spacing between bars
            bar_height = 0.04  # Narrow bars
            offset = bar_height * 0.35  # Offset to separate error bars
            
            # Plot ResNet18 on primary (bottom) x-axis - horizontal bars going right
            bars_resnet = ax.barh(y-offset, resnet_values, height=bar_height, color=bar_colors, 
                             alpha=0.8, edgecolor='black', linewidth=0.5, label='ResNet18')
            ax.errorbar(resnet_values, y - offset, xerr=resnet_errors, fmt='none', ecolor='#2E4057',
                       elinewidth=0.8, capsize=0, capthick=0.8, alpha=0.7, zorder=10)
            
            # Create secondary (top) x-axis for ViT - bars going left from top (reversed axis)
            ax2 = ax.twiny()
            bars_vit = ax2.barh(y+offset, vit_values, height=bar_height, color=bar_colors,
                                 alpha=0.8, edgecolor='black', linewidth=0.5, label='ViT')
            ax2.errorbar(vit_values, y + offset, xerr=vit_errors, fmt='none', ecolor='#2E4057',
                        elinewidth=0.8, capsize=0, capthick=0.8, alpha=0.7, zorder=10)
            
            # Format x-axes based on metric (horizontal bars)
            if metric_key == 'augrc':
                ax.set_xlim(0.225, 0.475)
                ax.set_xlabel('ResNet18', fontsize=12, fontweight='bold', x=0.28)
                
                resnet_ticks = [0.35, 0.325, 0.3, 0.275, 0.25, 0.225]
                resnet_labels = ['15%', '', '20%', '', '25%', '']
                ax.set_xticks(resnet_ticks)
                ax.set_xticklabels(resnet_labels, fontsize=9)
                ax.tick_params(axis='x', which='both', length=0)  # Remove tick marks
                
                # Format top x-axis (ViT - reversed so bars meet in center)
                ax2.set_xlim(0.475, 0.225)  # Reversed
                ax2.set_xlabel('ViT', fontsize=12, fontweight='bold', x=0.75)
                
                vit_ticks = [0.225, 0.25, 0.275, 0.3, 0.325, 0.35]
                vit_labels = ['', '25%', '', '20%', '', '15%']
                ax2.set_xticks(vit_ticks)
                ax2.set_xticklabels(vit_labels, fontsize=9)
                ax2.tick_params(axis='x', which='both', top=True, labeltop=True, pad=6, length=0)  # Remove tick marks
                
                separator_x = 0.35
            else:
                # AUROC_f: Raw AUROC_f values (0.4-1.0)
                ax.set_xlim(0.6, 1.2)
                ax.set_xlabel('ResNet18', fontsize=12, fontweight='bold', x=0.28)
                
                resnet_ticks = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
                resnet_labels = ['.6', '', '.7', '', '.8', '', '.9']
                ax.set_xticks(resnet_ticks)
                ax.set_xticklabels(resnet_labels, fontsize=9)
                ax.tick_params(axis='x', which='both', length=0)  # Remove tick marks
                
                # Format top x-axis (ViT - reversed so bars meet in center)
                ax2.set_xlim(1.2, 0.6)  # Reversed
                ax2.set_xlabel('ViT', fontsize=12, fontweight='bold', x=0.75)
                
                vit_ticks = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
                vit_labels = ['.9', '', '.8', '', '.7', '', '.6']
                ax2.set_xticks(vit_ticks)
                ax2.set_xticklabels(vit_labels, fontsize=9)
                ax2.tick_params(axis='x', which='both', top=True, labeltop=True, pad=6, length=0)  # Remove tick marks
                
                separator_x = 0.9
            
            # Add visual separator line (vertical)
            ax.axvline(x=separator_x, color='lightgray', linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)
            
            # Remove y-tick labels and set tight y-axis limits
            ax.set_yticks(y)
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', length=0)  # Remove y-axis tick marks
            ax2.tick_params(axis='y', which='both', length=0)  # Remove y-axis tick marks
            
            # Set y-axis limits to fit bars tightly with small margins
            if len(y) > 0:
                margin = bar_height * 3.5
                ax.set_ylim(y[0] - margin, y[-1] + margin)
                ax2.set_ylim(y[0] - margin, y[-1] + margin)
            
            # Grid on primary axis only (horizontal grid for vertical layout)
            ax.grid(axis='x', alpha=0.5, linestyle='--', zorder=0)
            ax.set_axisbelow(True)
            ax2.grid(axis='x', alpha=0.5, linestyle='--', zorder=0)
            ax2.set_axisbelow(True)
            
            # Hide spines strategically
            for spine in ax.spines.values():
                if spine.spine_type in ['right', 'bottom', 'top']:
                    spine.set_visible(False)
            # Keep left spine for ax visible
            ax.spines['left'].set_visible(True)
            
            # For ax2, hide everything except right spine
            for spine in ax2.spines.values():
                spine.set_visible(False)
            ax2.spines['right'].set_visible(True)
            ax2.spines['right'].set_linewidth(0.8)
            ax2.spines['right'].set_color('black')
            
            # Draw custom spine segment for bottom axis only (ResNet18)
            if metric_key == 'augrc':
                ax.plot([0.225, 0.35], [0, 0], 
                        color='black', linewidth=0.8, clip_on=False, zorder=100, 
                        transform=ax.get_xaxis_transform())
                ax2.plot([0.225, 0.35], [1, 1], 
                        color='black', linewidth=0.8, clip_on=False, zorder=100, 
                        transform=ax2.get_xaxis_transform())
            else:
                ax.plot([0.6, 0.9], [0, 0], 
                        color='black', linewidth=0.8, clip_on=False, zorder=100, 
                        transform=ax.get_xaxis_transform())
                ax2.plot([0.6, 0.9], [1, 1], 
                        color='black', linewidth=0.8, clip_on=False, zorder=100, 
                        transform=ax2.get_xaxis_transform())
    
    # Add legend at the bottom
    if all_handles and all_labels:
        # Reorder legend items to match method_display_order
        # Create tuples of (original_label, handle) and sort by display order
        label_handle_pairs = list(zip(all_labels, all_handles))
        sorted_pairs = sorted(label_handle_pairs, 
                             key=lambda x: method_display_order.get(x[0], 999))
        
        if sorted_pairs:
            all_labels, all_handles = zip(*sorted_pairs)
            all_labels = list(all_labels)
            all_handles = list(all_handles)
        
        # Rename method labels AFTER sorting
        all_labels = [label.replace('KNN_Raw', 'KNN')
                           .replace('Ensembling', 'DE')
                           .replace('MCDropout', 'MCD')
                           .replace('MSR_calibrated', 'MSR-S')
                           .replace('ZScore Agg + Ens', 'Mean Agg + Ens')
                           .replace('ZScore_Aggregation_per_fold', 'Mean Agg')
                      for label in all_labels]
        
        # Manually reorder for ncol=5 column-wise filling
        if len(all_labels) == 10:
            reordered_labels = [all_labels[i] for i in [7, 3, 8, 2, 6, 1, 5, 0, 4, 9]]
            reordered_handles = [all_handles[i] for i in [7, 3, 8, 2, 6, 1, 5, 0, 4, 9]]

        else:
            reordered_labels = all_labels
            reordered_handles = all_handles
        
        fig.legend(reordered_handles, reordered_labels, loc='lower center', ncol=5,
                  fontsize=11, frameon=True, bbox_to_anchor=(0.52, 0.455), framealpha=1)
    
    return fig


def compute_mean_agg_row(results, display_names, shift_name, metric, aggregation):
    """
    Compute Mean_Aggregation row for heatmap using pre-computed values from JSON.
    
    Args:
        results: Dict with pre-loaded results [model][dataset][method] structure
        display_names: List of display names matching heatmap columns
        shift_name: Shift type name
        metric: 'auroc_f' or 'augrc'
        aggregation: Aggregation strategy name
    
    Returns:
        np.array: Row of differences (ensemble - per-fold)
    """
    agg_row_list = []
    metric_key = f'{metric}_mean' if metric == 'augrc' else metric
    
    for display_name in display_names:
        # Parse display_name to extract model and dataset
        clean_name = display_name.replace('_corrupt_severity3_test', '').replace('_test', '').replace('_external', '')
        
        setup = 'standard'
        for setup_name in ['DADO', 'DO', 'DA']:
            if setup_name in clean_name:
                setup = setup_name
                clean_name = clean_name.replace('_' + setup_name, '')
                break
        
        model = 'unknown'
        if '_vit_b_16' in clean_name:
            model = 'vit_b_16'
            clean_name = clean_name.replace('_vit_b_16', '')
        elif '_resnet18' in clean_name:
            model = 'resnet18'
            clean_name = clean_name.replace('_resnet18', '')
        
        dataset = clean_name
        dataset_key = dataset if setup == 'standard' else f"{dataset}_{setup}"
        
        # Load pre-computed values from results dict
        try:
            dataset_results = results.get(model, {}).get(dataset_key, {})
            
            # Get Mean_Aggregation_Ensemble and Mean_Aggregation values
            ensemble_val = dataset_results.get('Mean_Aggregation_Ensemble', {}).get(metric_key, np.nan)
            perfold_val = dataset_results.get('Mean_Aggregation', {}).get(metric_key, np.nan)
            
            diff = ensemble_val - perfold_val if not np.isnan(ensemble_val) and not np.isnan(perfold_val) else np.nan
        except:
            diff = np.nan
        
        agg_row_list.append(diff)
    
    return np.array(agg_row_list).reshape(1, -1)


def parse_display_names(display_names):
    """Parse display names into setup, model, dataset lists."""
    setups = []
    models = []
    datasets = []
    setup_names = ['DA', 'DO', 'DADO']
    model_names = ['resnet18', 'vit_b_16']
    
    for name in display_names:
        clean_name = name.replace('_corrupt_severity3_test', '').replace('_test', '').replace('_external', '')
        
        setup = 'standard'
        col_without_setup = clean_name
        for setup_name in setup_names:
            if clean_name.endswith('_' + setup_name):
                setup = setup_name
                col_without_setup = clean_name[:-len(setup_name)-1]
                break
        
        model = None
        dataset = None
        for model_name in model_names:
            if col_without_setup.endswith('_' + model_name):
                model = model_name
                dataset = col_without_setup[:-len(model_name)-1]
                break
        
        if model is None:
            parts = col_without_setup.split('_')
            model = parts[-1] if len(parts) > 0 else ''
            dataset = '_'.join(parts[:-1]) if len(parts) > 1 else ''
        
        setups.append(setup)
        models.append(model)
        datasets.append(dataset)
    
    return setups, models, datasets


def add_hierarchical_labels(ax, ticks, setups, models, datasets):
    """Add model and dataset labels below x-axis."""
    # Group by dataset and model
    groups_detailed = []
    i = 0
    while i < len(datasets):
        curr_dataset = datasets[i]
        curr_model = models[i]
        start = i
        while i < len(datasets) and datasets[i] == curr_dataset and models[i] == curr_model:
            i += 1
        groups_detailed.append((start, i-1, curr_dataset, curr_model))
    
    # Add model names
    for (s, e, dname, mname) in groups_detailed:
        if not mname:
            continue
        center = (ticks[s] + ticks[e]) / 2.0
        ax.text(center, -0.25, mname, transform=ax.get_xaxis_transform(), 
                ha='center', va='top', fontsize=9, fontweight='normal')
    
    # Group by dataset only
    dataset_groups = []
    i = 0
    while i < len(datasets):
        curr_dataset = datasets[i]
        start = i
        while i < len(datasets) and datasets[i] == curr_dataset:
            i += 1
        dataset_groups.append((start, i-1, curr_dataset))
    
    # Add dataset names
    for (s, e, dname) in dataset_groups:
        if not dname:
            continue
        center = (ticks[s] + ticks[e]) / 2.0
        ax.text(center, -0.38, dname, transform=ax.get_xaxis_transform(), 
                ha='center', va='top', fontsize=9, fontweight='bold')


def main(aggregation='mean'):
    """Main function to generate unified figures."""
    
    print("=" * 80)
    print("Unified Results Visualization")
    print(f"Aggregation: {aggregation.upper()}")
    print("=" * 80)
    
    # Get paths
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent.parent
    results_dir = workspace_root / 'Benchmarks' / 'medMNIST' / 'results'
    
    # Define all results directories
    results_dirs = {
        'id': results_dir / 'jsons_results' / 'in_distribution',
        'corruption': results_dir / 'jsons_results' / 'corruption_shifts',
        'population': results_dir / 'jsons_results' / 'population_shifts',
        'new_class': results_dir / 'jsons_results' / 'new_class_shifts'
    }
    
    # Merge population and new_class for visualization
    # (they will be combined in the same row as PS/NCS)
    
    # Define comprehensive evaluation directories
    comp_eval_base = workspace_root / 'benchmarks' / 'medMNIST' / 'utils' / 'comprehensive_evaluation_results'
    comp_eval_dirs = {
        'id': comp_eval_base / 'in_distribution',
        'corruption': comp_eval_base / 'corruption_shifts',
        'population': comp_eval_base / 'population_shifts'
    }
    
    print(f"Workspace root: {workspace_root}")
    print(f"Results directory: {results_dir}")
    
    # Create output directory
    output_dir = results_dir / 'figures' / 'unified'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1. Load all results
    # ==========================================
    print("\n" + "=" * 80)
    print("Loading results...")
    print("=" * 80)
    
    results_auroc = load_and_parse_results(results_dirs, metric='auroc_f')
    results_augrc = load_and_parse_results(results_dirs, metric='augrc')
    
    # Merge population and new_class results
    for metric_results in [results_auroc, results_augrc]:
        if 'population' in metric_results and 'new_class' in metric_results:
            pop_res = metric_results['population']
            new_res = metric_results['new_class']
            
            # Merge both into 'population' key
            merged = {}
            for model in ['resnet18', 'vit_b_16']:
                merged[model] = {}
                if model in pop_res:
                    merged[model].update(pop_res[model])
                if model in new_res:
                    merged[model].update(new_res[model])
            
            metric_results['population'] = merged


    
    # ==========================================
    # 2. Compute AUROC_f means
    # ==========================================
    print("\n" + "=" * 80)
    print("Computing AUROC_f means...")
    print("=" * 80)
    
    means_auroc = compute_metric_means(results_auroc, metric='auroc_f')
    
    # ==========================================
    # 3. Compute AUGRC means
    # ==========================================
    print("\n" + "=" * 80)
    print("Computing AUGRC means...")
    print("=" * 80)
    
    means_augrc = compute_metric_means(results_augrc, metric='augrc')
    
    # Both means_auroc and means_augrc have structure: [shift][model][metric][method]
    # Merge them by combining the metric subdicts
    all_means = {}
    for shift_key in ['id', 'corruption', 'population']:
        all_means[shift_key] = {}
        for model_name in ['resnet18', 'vit_b_16']:
            all_means[shift_key][model_name] = {}
            
            # Get AUROC_f means from means_auroc
            auroc_data = means_auroc.get(shift_key, {}).get(model_name, {})
            if 'auroc_f' in auroc_data:
                all_means[shift_key][model_name]['auroc_f'] = auroc_data['auroc_f']
                all_means[shift_key][model_name]['auroc_f_std'] = auroc_data.get('auroc_f_std', {})
            else:
                # Old structure: it's a flat dict of methods
                all_means[shift_key][model_name]['auroc_f'] = auroc_data
            
            # Get AUGRC means from means_augrc
            augrc_data = means_augrc.get(shift_key, {}).get(model_name, {})
            if 'augrc' in augrc_data:
                all_means[shift_key][model_name]['augrc'] = augrc_data['augrc']
                all_means[shift_key][model_name]['augrc_std'] = augrc_data.get('augrc_std', {})
            else:
                # Old structure: it's a flat dict of methods
                all_means[shift_key][model_name]['augrc'] = augrc_data

    # ==========================================
    # 6. Create combined radar + histogram figures (ID+CS)
    # ==========================================
    print("\n" + "=" * 80)
    print("Creating ID+CS radar + histogram figures...")
    print("=" * 80)
    
    # Figure 1: ID+CS AUROC_f
    fig_id_cs_auroc = create_combined_radar_histogram_figure_id_cs(
        results_auroc, results_augrc, all_means,
        aggregation=aggregation,
        results_dir=results_dir,
        comp_eval_dirs=comp_eval_dirs,
        metric='auroc_f'
    )
    
    output_path = output_dir / f'id_cs_auroc_f_radars_with_histograms_{aggregation}.png'
    fig_id_cs_auroc.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Saved ID+CS AUROC_f to {output_path}")
    plt.close(fig_id_cs_auroc)
    
    # Figure 1: ID+CS AUGRC
    fig_id_cs_augrc = create_combined_radar_histogram_figure_id_cs(
        results_auroc, results_augrc, all_means,
        aggregation=aggregation,
        results_dir=results_dir,
        comp_eval_dirs=comp_eval_dirs,
        metric='augrc'
    )
    
    output_path = output_dir / f'id_cs_augrc_radars_with_histograms_{aggregation}.png'
    fig_id_cs_augrc.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Saved ID+CS AUGRC to {output_path}")
    plt.close(fig_id_cs_augrc)
    
    # ==========================================
    # 7. Create combined radar + histogram figure (NCS+PS - both metrics)
    # ==========================================
    print("\n" + "=" * 80)
    print("Creating NCS+PS combined radar + histogram figure (both metrics)...")
    print("=" * 80)
    
    # Figure 2: NCS+PS (both AUROC_f and AUGRC in one 3x2 figure)
    fig_ncs_ps_combined = create_combined_radar_histogram_figure_ncs_ps(
        results_auroc, results_augrc, all_means,
        aggregation=aggregation,
        results_dir=results_dir,
        comp_eval_dirs=comp_eval_dirs,
        metric='auroc_f'  # Parameter kept for compatibility but not used internally
    )
    
    if fig_ncs_ps_combined:
        output_path = output_dir / f'ncs_ps_combined_radars_with_histograms_{aggregation}.png'
        fig_ncs_ps_combined.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n Saved NCS+PS combined (both metrics) to {output_path}")
        plt.close(fig_ncs_ps_combined)
    
    print("\n" + "=" * 80)
    print(" All figures saved!")
    print("=" * 80)


if __name__ == '__main__':
    aggregation = sys.argv[1] if len(sys.argv) > 1 else 'mean'
    
    if aggregation not in ['mean', 'min', 'max', 'vote']:
        print(f"Unknown aggregation: {aggregation}")
        print("Usage: python plot_unified_results.py [mean|min|max|vote]")
        sys.exit(1)
    
    main(aggregation=aggregation)
