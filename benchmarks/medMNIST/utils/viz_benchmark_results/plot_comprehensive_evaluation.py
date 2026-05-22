#!/home/psteinmetz/venvs/venv_medMNIST/bin/python3.12
"""
Visualize Comprehensive Evaluation Results

Creates boxplots for balanced accuracy across datasets, models, and training setups
from the comprehensive evaluation JSON files.
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# Set style
plt.rcParams['font.size'] = 10

# Dataset names for display
DATASET_DISPLAY_NAMES = {
    'bloodmnist': 'Blood',
    'breastmnist': 'Breast',
    'dermamnist-e-id': 'Derma-e-ID',
    'dermamnist-e-ood': 'Derma-e-OOD',
    'octmnist': 'OCT',
    'organamnist': 'Organ',
    'pneumoniamnist': 'Pneumonia',
    'tissuemnist': 'Tissue',
    'pathmnist': 'Path',
    'amos22': 'AMOS-2022'
}

# Setup colors and markers (matching notebook style)
SETUP_STYLES = {
    'standard': {'color': '#1f77b4', 'marker': 'o', 'label': 'Standard'},
    'DA': {'color': '#ff7f0e', 'marker': 's', 'label': 'DA'},
    'DO': {'color': '#2ca02c', 'marker': '^', 'label': 'DO'},
    'DADO': {'color': '#d62728', 'marker': 'D', 'label': 'DADO'}
}


def load_all_results(results_dir):
    """
    Load all comprehensive evaluation results from JSON files.
    
    Returns:
        dict: Nested dict with structure:
              {dataset: {model: {setup: {shift_type: {'per_fold': [...], 'ensemble': {...}}}}}}
              where shift_type is 'in_distribution', 'corruption', or 'population'
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    # Load in-distribution results
    in_dist_dir = os.path.join(results_dir, "in_distribution")
    if os.path.exists(in_dist_dir):
        json_files = glob.glob(os.path.join(in_dist_dir, "comprehensive_metrics_*.json"))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            dataset = data['dataset']
            model = data['model']
            setup = data['setup']
            
            per_fold_bacc = [fold['balanced_accuracy'] for fold in data['per_fold_metrics']]
            ensemble_bacc = data['ensemble_metrics']['balanced_accuracy']
            
            results[dataset][model][setup]['in_distribution'] = {
                'per_fold_bacc': per_fold_bacc,
                'ensemble_bacc': ensemble_bacc,
                'per_fold_acc': [fold['accuracy'] for fold in data['per_fold_metrics']],
                'ensemble_acc': data['ensemble_metrics']['accuracy'],
                'per_fold_auc': [fold['auc'] for fold in data['per_fold_metrics']],
                'ensemble_auc': data['ensemble_metrics']['auc'],
            }
    
    # Load corruption shift results (severity 3)
    corruption_dir = os.path.join(results_dir, "corruption_shifts")
    if os.path.exists(corruption_dir):
        json_files = glob.glob(os.path.join(corruption_dir, "*_severity3.json"))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            dataset = data['metadata']['dataset']
            model = data['metadata']['model']
            setup_str = data['metadata']['setup']
            
            # Fix dermamnist-e → dermamnist-e-id mapping
            # (corruption files are named dermamnist-e-id but contain "dermamnist-e" in metadata)
            if dataset == 'dermamnist-e' and 'dermamnist-e-id' in os.path.basename(json_file):
                dataset = 'dermamnist-e-id'
            
            per_fold_bacc = [fold['balanced_accuracy'] for fold in data['per_fold']]
            ensemble_bacc = data['ensemble']['balanced_accuracy']
            
            results[dataset][model][setup_str]['corruption'] = {
                'per_fold_bacc': per_fold_bacc,
                'ensemble_bacc': ensemble_bacc,
            }
    
    # Load population shift results
    population_dir = os.path.join(results_dir, "population_shift")
    if os.path.exists(population_dir):
        json_files = glob.glob(os.path.join(population_dir, "comprehensive_metrics_*.json"))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            dataset = data['dataset']
            model = data['model']
            setup = data['setup']
            
            per_fold_bacc = [fold['balanced_accuracy'] for fold in data['per_fold_metrics']]
            ensemble_bacc = data['ensemble_metrics']['balanced_accuracy']
            
            # Store population shift results
            # amos22 → shows in organamnist's population column
            # dermamnist-e-ood → shows in dermamnist-e-id's population column  
            # pathmnist from population_shift → shows in pathmnist's population column
            if dataset == 'amos22':
                results['organamnist'][model][setup]['population'] = {
                    'per_fold_bacc': per_fold_bacc,
                    'ensemble_bacc': ensemble_bacc,
                }
            elif dataset == 'dermamnist-e-ood':
                results['dermamnist-e-id'][model][setup]['population'] = {
                    'per_fold_bacc': per_fold_bacc,
                    'ensemble_bacc': ensemble_bacc,
                }
            elif dataset == 'hmu-crc':
                    results['pathmnist'][model][setup]['population'] = {
                        'per_fold_bacc': per_fold_bacc,
                        'ensemble_bacc': ensemble_bacc,
                    }
    
    return results


def create_boxplot_for_dataset_column(ax, dataset_name, dataset_data, shift_type, is_title_row=False):
    """
    Create box plot for a single dataset and shift type (one cell in the grid).
    
    Args:
        ax: Matplotlib axis
        dataset_name: Name of dataset
        dataset_data: Dict with model/setup/shift data
        shift_type: 'in_distribution', 'corruption', or 'population'
        is_title_row: If True, this is the first row and should show column titles
    """
    models = ['resnet18', 'vit_b_16']
    setups = ['standard', 'DA', 'DO', 'DADO']
    
    positions = {
        'resnet18': {'standard': 1, 'DA': 2, 'DO': 3, 'DADO': 4},
        'vit_b_16': {'standard': 6, 'DA': 7, 'DO': 8, 'DADO': 9}
    }
    
    # Check if data exists for this shift type
    has_data = False
    for model in models:
        for setup in setups:
            if setup in dataset_data.get(model, {}) and shift_type in dataset_data[model][setup]:
                has_data = True
                break
        if has_data:
            break
    
    if not has_data:
        # Draw "N/A" for missing data
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
               fontsize=20, color='gray', transform=ax.transAxes)
        ax.set_xlim(0, 10)
        ax.set_ylim(0.4, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        return
    
    # Collect data for box plots
    all_boxes_data = []
    all_positions = []
    
    for model in models:
        for setup in setups:
            if setup in dataset_data.get(model, {}) and shift_type in dataset_data[model][setup]:
                fold_values = dataset_data[model][setup][shift_type]['per_fold_bacc']
                all_boxes_data.append(fold_values)
                all_positions.append(positions[model][setup])
    
    # Create box plots
    if all_boxes_data:
        # Color based on shift type
        if shift_type == 'in_distribution':
            box_color = 'lightgray'
        elif shift_type == 'corruption':
            box_color = 'lightcoral'
        else:  # population
            box_color = 'lightskyblue'
        
        bp = ax.boxplot(all_boxes_data, positions=all_positions, widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2))
        
        for patch in bp['boxes']:
            patch.set_facecolor(box_color)
            patch.set_alpha(0.4)
            patch.set_linewidth(1.2)
    
    # Overlay ensemble points
    for model in models:
        for setup in setups:
            if setup in dataset_data.get(model, {}) and shift_type in dataset_data[model][setup]:
                ensemble_bacc = dataset_data[model][setup][shift_type]['ensemble_bacc']
                pos = positions[model][setup]
                
                style = SETUP_STYLES[setup]
                ax.plot(pos, ensemble_bacc, marker=style['marker'], 
                       color=style['color'], markersize=8, 
                       markeredgecolor='black', markeredgewidth=1.2, zorder=3)
    
    # Formatting
    ax.set_xlim(0, 10)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks([2.5, 7.5])
    ax.set_xticklabels(['R18', 'ViT'], fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)


def create_all_datasets_detailed_boxplot(results, output_dir):
    """
    Create one big figure with all datasets, each showing detailed x-axis labels.
    Groups: ID | Corruption | Population for each model within each dataset subplot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    datasets_to_plot = [
        'pathmnist', 'dermamnist-e-id', 'octmnist', 'pneumoniamnist',
        'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist'
    ]
    
    # Filter to available datasets
    datasets_to_plot = [d for d in datasets_to_plot if d in results]
    
    n_datasets = len(datasets_to_plot)
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    # Tighter figure with shared x-axis
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 5*n_rows), sharex=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, dataset_name in enumerate(datasets_to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        dataset_data = results[dataset_name]
        
        models = ['resnet18', 'vit_b_16']
        setups = ['standard', 'DA', 'DO', 'DADO']
        shift_types = ['in_distribution', 'corruption', 'population']
        shift_labels = ['', '_C', '_P']
        
        all_boxes_data = []
        all_positions = []
        all_colors = []
        all_labels = []
        
        # New layout: group by shift type instead of by model
        # For each shift: R18 setups, then VIT setups
        pos = 0
        gap_between_setups = 0.5  # Gap between individual setups
        gap_between_models = 0.6  # Gap between R18 and VIT within same shift
        gap_between_shifts = 0.8  # Larger gap between different shift types
        
        for shift_idx, (shift_type, shift_label) in enumerate(zip(shift_types, shift_labels)):
            if shift_idx > 0:
                pos += gap_between_shifts
            
            for model_idx, model in enumerate(models):
                if model_idx > 0:
                    pos += gap_between_models
                
                # Always create positions for all 4 setups, even if no data
                for setup in setups:
                    if setup in dataset_data.get(model, {}) and shift_type in dataset_data[model][setup]:
                        fold_values = dataset_data[model][setup][shift_type]['per_fold_bacc']
                        all_boxes_data.append(fold_values)
                        all_positions.append(pos)
                        
                        # All boxes same gray color
                        all_colors.append('lightgray')
                        
                        # Shorter labels
                        model_short = 'R18' if model == 'resnet18' else 'ViT'
                        if setup == 'standard':
                            setup_short = 'S'
                        elif setup == 'DADO':
                            setup_short = 'DADO'
                        else:
                            setup_short = setup
                        all_labels.append(f"{model_short}_{setup_short}{shift_label}")
                    else:
                        # Empty position for alignment
                        all_boxes_data.append(None)
                        all_positions.append(pos)
                        all_colors.append('white')
                        all_labels.append('')
                    
                    pos += gap_between_setups
        
        # Add grey background for empty positions (where there are no boxes)
        # Cover the full width including gaps between positions
        for data, pos_val in zip(all_boxes_data, all_positions):
            if data is None:  # Empty position
                # Add a light grey rectangle covering full width (half gap on each side)
                ax.axvspan(pos_val - gap_between_setups/2, pos_val + gap_between_setups/2, 
                          facecolor='lightgrey', alpha=0.2, zorder=0)
        
        # Create box plots (only for non-None data)
        valid_boxes = [(data, pos, color) for data, pos, color in zip(all_boxes_data, all_positions, all_colors) if data is not None]
        
        if valid_boxes:
            valid_data, valid_pos, valid_colors = zip(*valid_boxes)
            bp = ax.boxplot(valid_data, positions=valid_pos, widths=0.4,  # Even tighter boxes
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=2.0),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5))
            
            for patch, color in zip(bp['boxes'], valid_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.4)
                patch.set_linewidth(1.5)
        
        # Overlay ensemble points
        pos = 0
        for shift_idx, shift_type in enumerate(shift_types):
            if shift_idx > 0:
                pos += gap_between_shifts
            
            for model_idx, model in enumerate(models):
                if model_idx > 0:
                    pos += gap_between_models
                
                # Always iterate through all setups for alignment
                for setup in setups:
                    if setup in dataset_data.get(model, {}) and shift_type in dataset_data[model][setup]:
                        ensemble_bacc = dataset_data[model][setup][shift_type]['ensemble_bacc']
                        
                        style = SETUP_STYLES[setup]
                        ax.plot(pos, ensemble_bacc, marker=style['marker'], 
                               color=style['color'], markersize=10,  # Bigger markers
                               markeredgecolor='black', markeredgewidth=1.5, zorder=3)
                    
                    pos += gap_between_setups
        
        # Formatting
        # Display name with special case for dermamnist-e-id → Derma-e
        display_name = 'Derma-e' if dataset_name == 'dermamnist-e-id' else DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
        # Only show y-label for left column
        if col == 0:
            ax.set_ylabel('Balanced Accuracy', fontsize=16, fontweight='bold')
        # No title - dataset name will be inside plot
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        
        # Add dataset name inside plot (bottom right)
        ax.text(0.98, 0.04, display_name, transform=ax.transAxes, 
               fontsize=20, fontweight='bold', ha='right', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))
        
        # Only show x-ticks on bottom row with simplified labels
        is_bottom_row = (row == n_rows - 1)
        if is_bottom_row:
            # Create simplified labels: just setup names repeated for each shift
            simplified_labels = []
            for i, label in enumerate(all_labels):
                if label:  # Non-empty label
                    # Extract setup from label (e.g., "R18_S" -> "S", "ViT_DADO_C" -> "DADO")
                    parts = label.split('_')
                    if len(parts) >= 2:
                        setup_part = parts[1].replace('_C', '').replace('_P', '')
                        simplified_labels.append(setup_part)
                    else:
                        simplified_labels.append('')
                else:
                    simplified_labels.append('')
            
            ax.set_xticks(all_positions)
            ax.set_xticklabels(simplified_labels, rotation=90, ha='center', fontsize=17, fontweight='bold')
        else:
            ax.set_xticks([])
        
        # Add vertical lines to separate groups (shift types and models)
        # Find positions between shifts (larger gaps)
        if len(all_positions) > 0:
            positions_arr = np.array(all_positions)
            diffs = np.diff(positions_arr)
            separator_positions = []
            model_separator = None  # The line between ResNet18 and VIT
            
            for i, diff in enumerate(diffs):
                if diff > 1.0:  # Gap detected (shift or model boundary)
                    sep_pos = (positions_arr[i] + positions_arr[i+1]) / 2
                    separator_positions.append(sep_pos)
                    # The largest gap is between models
                    if diff > 2.5:  # gap_between_models = 3.0
                        model_separator = sep_pos
            
            # Draw all separator lines
            for sep_pos in separator_positions:
                if sep_pos == model_separator:
                    # Bolder line between ResNet18 and VIT
                    ax.axvline(x=sep_pos, color='gray', linestyle='-', linewidth=2.5, alpha=0.7, zorder=1)
                else:
                    # Lighter lines between shift types
                    ax.axvline(x=sep_pos, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1)
    
    # Hide unused subplots
    for idx in range(n_datasets, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    # Add table-like header at the top for BOTH columns
    # New layout: Each column has ID (R18+VIT), CS (R18+VIT), PS (R18+VIT)
    # LEFT COLUMN (col=0): spans 0.05 to 0.50
    fig.text(0.07, 0.945, 'R18-ID', ha='center', va='top', fontsize=22, fontweight='bold')
    fig.text(0.1452, 0.945, 'ViT-ID', ha='center', va='top', fontsize=22, fontweight='bold')
    
    fig.text(0.2250, 0.945, 'R18-CS', ha='center', va='top', fontsize=22, fontweight='bold')
    fig.text(0.3100, 0.945, 'ViT-CS', ha='center', va='top', fontsize=22, fontweight='bold')
    
    fig.text(0.3900, 0.945, 'R18-PS', ha='center', va='top', fontsize=22, fontweight='bold')
    fig.text(0.4621, 0.945, 'ViT-PS', ha='center', va='top', fontsize=22, fontweight='bold')
    
    # RIGHT COLUMN (col=1): spans 0.50 to 0.95
    fig.text(0.5630, 0.945, 'R18-ID', ha='center', va='top', fontsize=22, fontweight='bold')
    fig.text(0.6390, 0.945, 'ViT-ID', ha='center', va='top', fontsize=22, fontweight='bold')
    
    fig.text(0.7270, 0.945, 'R18-CS', ha='center', va='top', fontsize=22, fontweight='bold')
    fig.text(0.7900, 0.945, 'ViT-CS', ha='center', va='top', fontsize=22, fontweight='bold')
    
    fig.text(0.8800, 0.945, 'R18-PS', ha='center', va='top', fontsize=22, fontweight='bold')
    fig.text(0.9600, 0.945, 'ViT-PS', ha='center', va='top', fontsize=22, fontweight='bold')
    
    # Create legend and place it inside Pathmnist plot (first plot, row=0, col=0)
    legend_elements = [
        mpatches.Patch(facecolor='lightgray', edgecolor='black', alpha=0.4, 
                      label='Fold variability')
    ]
    for setup, style in SETUP_STYLES.items():
        legend_elements.append(
            plt.Line2D([0], [0], marker=style['marker'], color='w', 
                      markerfacecolor=style['color'], markeredgecolor='black',
                      markersize=11, markeredgewidth=1.5, label=f"{style['label']} (ensemble)")
        )
    
    # Place legend inside Pathmnist plot (row=0, col=0) at bottom left
    axes[0, 0].legend(handles=legend_elements, loc='lower left', ncol=1, 
                     fontsize=17, frameon=True, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    output_file = os.path.join(output_dir, 'all_datasets_boxplots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def create_detailed_boxplot_per_dataset(results, output_dir):
    """
    Create one detailed boxplot per dataset with clear x-axis labels.
    Groups: ID | Corruption | Population for each model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    datasets_to_plot = [
        'pathmnist', 'dermamnist-e-id', 'octmnist', 'pneumoniamnist',
        'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist'
    ]
    
    for dataset_name in datasets_to_plot:
        if dataset_name not in results:
            continue
        
        dataset_data = results[dataset_name]
        
        fig, ax = plt.subplots(figsize=(24, 6))
        
        models = ['resnet18', 'vit_b_16']
        setups = ['standard', 'DA', 'DO', 'DADO']
        shift_types = ['in_distribution', 'corruption', 'population']
        shift_labels = ['', '_corrupt', '_pop']
        
        all_boxes_data = []
        all_positions = []
        all_colors = []
        all_labels = []
        
        pos = 0
        gap_between_shifts = 1.5
        gap_between_models = 3.0
        
        for model_idx, model in enumerate(models):
            if model_idx > 0:
                pos += gap_between_models
            
            for shift_idx, (shift_type, shift_label) in enumerate(zip(shift_types, shift_labels)):
                if shift_idx > 0:
                    pos += gap_between_shifts
                
                # Check if this shift type has any data
                has_shift_data = False
                for setup in setups:
                    if setup in dataset_data.get(model, {}) and shift_type in dataset_data[model][setup]:
                        has_shift_data = True
                        break
                
                if not has_shift_data:
                    continue
                
                # Add boxes for each setup
                for setup in setups:
                    if setup in dataset_data.get(model, {}) and shift_type in dataset_data[model][setup]:
                        fold_values = dataset_data[model][setup][shift_type]['per_fold_bacc']
                        all_boxes_data.append(fold_values)
                        all_positions.append(pos)
                        
                        # Color by shift type
                        if shift_type == 'in_distribution':
                            all_colors.append('lightgray')
                        elif shift_type == 'corruption':
                            all_colors.append('lightcoral')
                        else:  # population
                            all_colors.append('lightskyblue')
                        
                        # Label
                        model_short = 'R18' if model == 'resnet18' else 'ViT'
                        all_labels.append(f"{model_short}_{setup}{shift_label}")
                        
                        pos += 1
        
        # Create box plots
        if all_boxes_data:
            bp = ax.boxplot(all_boxes_data, positions=all_positions, widths=0.7,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=1.5),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2))
            
            for patch, color in zip(bp['boxes'], all_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.4)
                patch.set_linewidth(1.2)
        
        # Overlay ensemble points
        pos = 0
        for model_idx, model in enumerate(models):
            if model_idx > 0:
                pos += gap_between_models
            
            for shift_idx, shift_type in enumerate(shift_types):
                if shift_idx > 0:
                    pos += gap_between_shifts
                
                has_shift_data = False
                for setup in setups:
                    if setup in dataset_data.get(model, {}) and shift_type in dataset_data[model][setup]:
                        has_shift_data = True
                        break
                
                if not has_shift_data:
                    continue
                
                for setup in setups:
                    if setup in dataset_data.get(model, {}) and shift_type in dataset_data[model][setup]:
                        ensemble_bacc = dataset_data[model][setup][shift_type]['ensemble_bacc']
                        
                        style = SETUP_STYLES[setup]
                        ax.plot(pos, ensemble_bacc, marker=style['marker'], 
                               color=style['color'], markersize=9, 
                               markeredgecolor='black', markeredgewidth=1.2, zorder=3)
                        
                        pos += 1
        
        # Formatting
        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Balanced Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name), 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add vertical lines to separate groups
        if len(all_positions) > 0:
            # Find gaps to draw separators
            positions_arr = np.array(all_positions)
            diffs = np.diff(positions_arr)
            separator_positions = []
            for i, diff in enumerate(diffs):
                if diff > 1.2:  # Gap detected
                    separator_positions.append((positions_arr[i] + positions_arr[i+1]) / 2)
            
            for sep_pos in separator_positions:
                ax.axvline(x=sep_pos, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='lightgray', edgecolor='black', alpha=0.4, 
                          label='ID fold variability'),
            mpatches.Patch(facecolor='lightcoral', edgecolor='black', alpha=0.4, 
                          label='Corruption fold variability'),
            mpatches.Patch(facecolor='lightskyblue', edgecolor='black', alpha=0.4, 
                          label='Population fold variability')
        ]
        for setup, style in SETUP_STYLES.items():
            legend_elements.append(
                plt.Line2D([0], [0], marker=style['marker'], color='w', 
                          markerfacecolor=style['color'], markeredgecolor='black',
                          markersize=8, markeredgewidth=1.2, label=f"{style['label']} (ensemble)")
            )
        
        ax.legend(handles=legend_elements, loc='lower left', ncol=7, 
                 fontsize=9, frameon=True)
        
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'{dataset_name}_detailed_boxplot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f" Saved: {output_file}")
        plt.close()


def create_three_column_layout(results, output_dir):
    """
    Create a 3-column layout: ID | Corruption | Population Shift
    with one row per dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define datasets and their shift availability
    dataset_rows = [
        ('dermamnist-e-id', True, True, True),  # id, corruption, population
        ('pathmnist', True, True, True),       # id, corruption, population
        ('octmnist', True, True, False),         # id, corruption
        ('pneumoniamnist', True, True, False),   # id, corruption
        ('breastmnist', True, True, False),      # id, corruption
        ('bloodmnist', True, True, False),       # id, corruption
        ('tissuemnist', True, True, False),      # id, corruption
        ('organamnist', True, True, True),       # id, corruption, population
    ]
    
    n_rows = len(dataset_rows)
    n_cols = 3
    
    fig = plt.figure(figsize=(18, 3.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.25,
                          left=0.08, right=0.98, top=0.96, bottom=0.04)
    
    for row_idx, (dataset_name, has_id, has_corruption, has_population) in enumerate(dataset_rows):
        dataset_data = results.get(dataset_name, {})
        
        # Column 0: In-Distribution
        ax0 = fig.add_subplot(gs[row_idx, 0])
        if has_id:
            create_boxplot_for_dataset_column(ax0, dataset_name, dataset_data, 'in_distribution')
        else:
            ax0.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                    fontsize=20, color='gray', transform=ax0.transAxes)
            ax0.set_xlim(0, 10)
            ax0.set_ylim(0.4, 1.0)
            ax0.set_xticks([])
            ax0.set_yticks([])
        
        # Add dataset label on left
        display_name = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
        ax0.set_ylabel(display_name, fontsize=11, fontweight='bold', rotation=0, 
                      ha='right', va='center', labelpad=15)
        
        # Column titles (only on first row)
        if row_idx == 0:
            ax0.set_title('In-Distribution', fontsize=12, fontweight='bold', pad=10)
        
        # Column 1: Corruption Shift
        ax1 = fig.add_subplot(gs[row_idx, 1])
        if has_corruption:
            create_boxplot_for_dataset_column(ax1, dataset_name, dataset_data, 'corruption')
        else:
            ax1.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                    fontsize=20, color='gray', transform=ax1.transAxes)
            ax1.set_xlim(0, 10)
            ax1.set_ylim(0.4, 1.0)
            ax1.set_xticks([])
            ax1.set_yticks([])
        
        if row_idx == 0:
            ax1.set_title('Corruption Shift\n(severity 3)', fontsize=12, fontweight='bold', pad=10)
        
        # Column 2: Population Shift
        ax2 = fig.add_subplot(gs[row_idx, 2])
        if has_population:
            create_boxplot_for_dataset_column(ax2, dataset_name, dataset_data, 'population')
            # Add subtitle for population shift source
            if dataset_name == 'organamnist':
                subtitle = '(AMOS-2022)'
            elif dataset_name == 'dermamnist-e-id':
                subtitle = '(Derma-e-OOD)'
            elif dataset_name == 'pathmnist':
                subtitle = '(HMU-CRC)'
            else:
                subtitle = ''
            
            if subtitle and row_idx == 0:
                ax2.set_title(f'Population Shift\n{subtitle}', fontsize=12, fontweight='bold', pad=10)
            elif row_idx == 0:
                ax2.set_title('Population Shift', fontsize=12, fontweight='bold', pad=10)
        else:
            ax2.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                    fontsize=20, color='gray', transform=ax2.transAxes)
            ax2.set_xlim(0, 10)
            ax2.set_ylim(0.4, 1.0)
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            if row_idx == 0:
                ax2.set_title('Population Shift', fontsize=12, fontweight='bold', pad=10)
        
        # Only show y-axis labels on leftmost column
        if True:  # Always show y-axis
            ax0.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax0.set_yticklabels(['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=9)
        
        ax1.set_yticks([])
        ax2.set_yticks([])
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgray', edgecolor='black', alpha=0.4, 
                      label='ID fold variability'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='black', alpha=0.4, 
                      label='Corruption fold variability'),
        mpatches.Patch(facecolor='lightskyblue', edgecolor='black', alpha=0.4, 
                      label='Population fold variability')
    ]
    for setup, style in SETUP_STYLES.items():
        legend_elements.append(
            plt.Line2D([0], [0], marker=style['marker'], color='w', 
                      markerfacecolor=style['color'], markeredgecolor='black',
                      markersize=9, markeredgewidth=1.2, label=f"{style['label']} (ensemble)")
        )
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=7, 
              fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.995))
    
    output_file = os.path.join(output_dir, 'all_datasets_three_column_layout.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def create_all_datasets_boxplots(results, output_dir):
    """
    Create box plots for all datasets with 3-column layout:
    Column 1: In-distribution
    Column 2: Corruption shift
    Column 3: Population shift
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define dataset rows with their shift configurations
    # Format: (dataset_id, dataset_corruption, dataset_population, display_name)
    dataset_rows = [
        ('pathmnist', 'pathmnist', 'hmu-crc', 'Path'),
        ('dermamnist-e-id', 'dermamnist-e-id', 'dermamnist-e-ood', 'Derma-e'),
        ('octmnist', 'octmnist', None, 'OCT'),
        ('pneumoniamnist', 'pneumoniamnist', None, 'Pneumonia'),
        ('breastmnist', 'breastmnist', None, 'Breast'),
        ('bloodmnist', 'bloodmnist', None, 'Blood'),
        ('tissuemnist', 'tissuemnist', None, 'Tissue'),
        ('organamnist', 'organamnist', 'organamnist', 'Organ (+ AMOS)')
    ]
    
    n_rows = len(dataset_rows)
    n_cols = 3  # ID, Corruption, Population
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows))
    
    for row_idx, (ds_id, ds_corrupt, ds_pop, display_name) in enumerate(dataset_rows):
        # Column 0: In-distribution
        ax_id = axes[row_idx, 0]
        if ds_id in results and 'in_distribution' in str(results[ds_id]):
            has_data = create_boxplot_for_dataset_column(ax_id, ds_id, results[ds_id], 'in_distribution')
        else:
            ax_id.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                      fontsize=20, color='gray', transform=ax_id.transAxes)
            ax_id.set_xlim(0, 10)
            ax_id.set_ylim(0.4, 1.0)
            has_data = False
        
        # Row label on leftmost column
        if row_idx == 0:
            ax_id.set_title('In-Distribution', fontsize=12, fontweight='bold', pad=10)
        ax_id.set_ylabel(f'{display_name}\n\nBalanced Accuracy', fontsize=11, fontweight='bold')
        
        # Column 1: Corruption shift
        ax_corrupt = axes[row_idx, 1]
        if ds_corrupt and ds_corrupt in results:
            has_data = create_boxplot_for_dataset_column(ax_corrupt, ds_corrupt, results[ds_corrupt], 'corruption')
        else:
            ax_corrupt.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                           fontsize=20, color='gray', transform=ax_corrupt.transAxes)
            ax_corrupt.set_xlim(0, 10)
            ax_corrupt.set_ylim(0.4, 1.0)
        
        if row_idx == 0:
            ax_corrupt.set_title('Corruption Shift', fontsize=12, fontweight='bold', pad=10)
        ax_corrupt.set_ylabel('')  # No ylabel for middle column
        
        # Column 2: Population shift
        ax_pop = axes[row_idx, 2]
        if ds_pop and ds_pop in results:
            has_data = create_boxplot_for_dataset_column(ax_pop, ds_pop, results[ds_pop], 'population')
        else:
            ax_pop.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       fontsize=20, color='gray', transform=ax_pop.transAxes)
            ax_pop.set_xlim(0, 10)
            ax_pop.set_ylim(0.4, 1.0)
        
        if row_idx == 0:
            title = 'Population Shift'
            if ds_pop == 'dermamnist-e-ood':
                title += '\n(Derma-e-OOD)'
            elif ds_pop == 'organamnist':
                title += '\n(AMOS-2022)'
            elif ds_pop == 'hmu-crc':
                title += '\n(HMU-CRC)'
            ax_pop.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax_pop.set_ylabel('')  # No ylabel for rightmost column
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgray', edgecolor='black', alpha=0.3, 
                      label='Fold variability (box plot)')
    ]
    for setup, style in SETUP_STYLES.items():
        legend_elements.append(
            plt.Line2D([0], [0], marker=style['marker'], color='w', 
                      markerfacecolor=style['color'], markeredgecolor='black',
                      markersize=10, label=f"{style['label']} (ensemble)")
        )
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=6, 
              fontsize=11, frameon=True, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_file = os.path.join(output_dir, 'all_datasets_boxplots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgray', edgecolor='black', alpha=0.3, 
                      label='ID: Fold variability'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='black', alpha=0.3, 
                      label='Corruption (severity 3): Fold variability')
    ]
    for setup, style in SETUP_STYLES.items():
        legend_elements.append(
            plt.Line2D([0], [0], marker=style['marker'], color='w', 
                      markerfacecolor=style['color'], markeredgecolor='black',
                      markersize=10, label=f"{style['label']} (ensemble)")
        )
    # Add population shift indicator
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='none', markeredgecolor='gray',
                  markersize=10, markeredgewidth=2.5, 
                  label="Population shift (ensemble)")
    )
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, 
              fontsize=11, frameon=True, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_file = os.path.join(output_dir, 'all_datasets_boxplots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def create_ensemble_scatter_plot(results, output_dir):
    """
    Create scatter plot comparing ensemble vs per-fold mean balanced accuracy.
    Includes all shift types: in-distribution, corruption, and population.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Define shift type styles
    shift_styles = {
        'in_distribution': {'facecolor': 'full', 'alpha': 0.7, 'label': 'In-Distribution'},
        'corruption': {'facecolor': 'none', 'alpha': 1.0, 'label': 'Corruption Shift'},
        'population': {'facecolor': 'full', 'alpha': 0.5, 'label': 'Population Shift'}
    }
    
    # Collect data - all shift types
    for dataset in results:
        for model in results[dataset]:
            for setup in results[dataset][model]:
                for shift_type in ['in_distribution', 'corruption', 'population']:
                    if shift_type not in results[dataset][model][setup]:
                        continue
                    
                    data = results[dataset][model][setup][shift_type]
                    fold_mean = np.mean(data['per_fold_bacc'])
                    ensemble_bacc = data['ensemble_bacc']
                    
                    style = SETUP_STYLES[setup]
                    shift_style = shift_styles[shift_type]
                    
                    # Plot point with different marker fills for shift types
                    if model == 'resnet18':
                        marker_size = 120
                        edgewidth = 2.0
                    else:  # vit_b_16
                        marker_size = 150
                        edgewidth = 2.5
                    
                    # Handle different fill styles for shift types
                    if shift_style['facecolor'] == 'full':
                        # In-distribution and population: filled (different alpha)
                        ax.scatter(fold_mean, ensemble_bacc, 
                                  marker=style['marker'], s=marker_size,
                                  color=style['color'], alpha=shift_style['alpha'],
                                  edgecolors='black', linewidths=edgewidth)
                    else:  # 'none' - hollow for corruption
                        # Corruption: hollow
                        ax.scatter(fold_mean, ensemble_bacc, 
                                  marker=style['marker'], s=marker_size,
                                  facecolors='none', edgecolors=style['color'],
                                  alpha=shift_style['alpha'], linewidths=edgewidth+0.5)
    
    # Diagonal line (y=x)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=2, label='y=x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Formatting
    ax.set_xlabel('Per-Fold Mean Balanced Accuracy', fontsize=16, fontweight='bold')
    ax.set_ylabel('Ensemble Balanced Accuracy', fontsize=16, fontweight='bold')
    ax.set_title('Ensemble vs Per-Fold Mean Performance\n(All Datasets and Shift Types)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Create custom legend with two columns: Setup styles and Shift types
    legend_elements = []
    
    # Setup markers
    legend_elements.append(plt.Line2D([0], [0], linestyle='', label='Training Setup:', 
                                     marker='', markersize=0))
    for setup, style in SETUP_STYLES.items():
        legend_elements.append(
            plt.Line2D([0], [0], marker=style['marker'], color='w',
                      markerfacecolor=style['color'], markeredgecolor='black',
                      markersize=11, markeredgewidth=1.5, label=f"  {style['label']}")
        )
    
    # Separator
    legend_elements.append(plt.Line2D([0], [0], linestyle='', label='', marker='', markersize=0))
    
    # Shift type markers
    legend_elements.append(plt.Line2D([0], [0], linestyle='', label='Shift Type:', 
                                     marker='', markersize=0))
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor='gray', markeredgecolor='black',
                  markersize=11, markeredgewidth=1.5, label='  In-Distribution')
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor='none', markeredgecolor='gray',
                  markersize=11, markeredgewidth=2.5, label='  Corruption Shift')
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor='gray', markeredgecolor='black',
                  markersize=11, markeredgewidth=1.5, label='  Population Shift',
                  alpha=0.5)
    )
    
    # Separator
    legend_elements.append(plt.Line2D([0], [0], linestyle='', label='', marker='', markersize=0))
    
    # Model size markers
    legend_elements.append(plt.Line2D([0], [0], linestyle='', label='Model:', 
                                     marker='', markersize=0))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='gray', markeredgecolor='black',
                                     markersize=9, label='  ResNet18'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='gray', markeredgecolor='black',
                                     markersize=11, label='  ViT-B/16'))
    
    # Separator and diagonal
    legend_elements.append(plt.Line2D([0], [0], linestyle='', label='', marker='', markersize=0))
    legend_elements.append(plt.Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='y=x'))
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, frameon=True,
             framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'ensemble_vs_folds_scatter.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def create_metric_comparison_table(results, output_dir):
    """
    Create a summary table with mean ± std for each configuration.
    """
    import pandas as pd
    
    rows = []
    
    for dataset in sorted(results.keys()):
        for model in sorted(results[dataset].keys()):
            for setup in ['standard', 'DA', 'DO', 'DADO']:
                if setup in results[dataset][model]:
                    # Only use in-distribution data for table
                    if 'in_distribution' not in results[dataset][model][setup]:
                        continue
                        
                    data = results[dataset][model][setup]['in_distribution']
                    
                    bacc_values = data['per_fold_bacc']
                    acc_values = data.get('per_fold_acc', [])
                    auc_values = data.get('per_fold_auc', [])
                    
                    row = {
                        'Dataset': dataset,
                        'Model': model,
                        'Setup': setup,
                        'Balanced Acc (mean±std)': f"{np.mean(bacc_values):.4f}±{np.std(bacc_values):.4f}",
                        'Ensemble Balanced Acc': f"{data['ensemble_bacc']:.4f}",
                    }
                    
                    if acc_values:
                        row['Accuracy (mean±std)'] = f"{np.mean(acc_values):.4f}±{np.std(acc_values):.4f}"
                        row['Ensemble Accuracy'] = f"{data['ensemble_acc']:.4f}"
                    
                    if auc_values:
                        row['AUC (mean±std)'] = f"{np.mean(auc_values):.4f}±{np.std(auc_values):.4f}"
                        row['Ensemble AUC'] = f"{data['ensemble_auc']:.4f}"
                    
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_file = os.path.join(output_dir, 'summary_table.csv')
    df.to_csv(csv_file, index=False)
    print(f" Saved: {csv_file}")
    
    return df


def main():
    """Main execution function."""
    
    # Paths
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    results_dir = repo_root / 'Benchmarks/medMNIST/results/classification_results'
    output_dir = repo_root / 'Benchmarks/medMNIST/results/classification_results/figures'
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION VISUALIZATION")
    print("="*80)
    print(f"Loading results from: {results_dir}")
    
    # Load results
    results = load_all_results(results_dir)
    
    n_datasets = len(results)
    n_configs = sum(len(models) * len(setups) 
                    for models in results.values() 
                    for setups in models.values())
    
    print(f" Loaded {n_configs} configurations across {n_datasets} datasets")
    print(f" Datasets: {', '.join(sorted(results.keys()))}")
    
    # Create visualizations
    print(f"\nGenerating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. Creating comprehensive boxplot for all datasets...")
    create_all_datasets_detailed_boxplot(results, output_dir)
    
    print("\n2. Creating ensemble vs per-fold scatter plot...")
    create_ensemble_scatter_plot(results, output_dir)
    
    print("\n3. Creating summary table...")
    df = create_metric_comparison_table(results, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Generated:")
    print(f" - 1 comprehensive boxplot with all datasets")
    print(f" - 1 ensemble vs per-fold scatter plot")
    print(f" - 1 summary CSV table")
    print("="*80)


if __name__ == '__main__':
    main()
