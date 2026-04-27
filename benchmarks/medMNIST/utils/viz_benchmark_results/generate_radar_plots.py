"""
Generate radar plots for UQ benchmark results across datasets and model configurations.

Creates two large radar plots (ResNet18 and ViT) showing mean AUROC_f values
for different UQ methods across all dataset-setup combinations.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def parse_results_directory(results_dir='./', metric='auroc_f'):
    """
    Parse all JSON result files in the directory.
    
    Args:
        results_dir: Directory containing JSON result files
        metric: Metric to extract - 'auroc_f' or 'augrc'
    
    Returns:
        dict: Nested dictionary structure:
            {
                'resnet18': {
                    'breastmnist_standard': {
                        'MSR': 0.75,
                        'MSR_platt': 0.82,
                        ...
                    },
                    ...
                },
                'vit_b_16': {
                    ...
                }
            }
    """
    results = defaultdict(lambda: defaultdict(dict))
    
    results_dir = Path(results_dir)
    json_files = list(results_dir.glob('uq_benchmark_*.json'))
    
    print(f"Found {len(json_files)} result files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            flag = data.get('flag', '')
            model_backbone = data.get('model_backbone', 'unknown')
            setup = data.get('setup', 'standard')
            
            # Create dataset-setup key
            if setup and setup != 'standard':
                dataset_key = f"{flag}_{setup}"
            else:
                dataset_key = f"{flag}_standard"
            
            # Extract method results
            methods_data = data.get('methods', {})
            for method_name, method_results in methods_data.items():
                # Get specified metric (use mean if available, otherwise single value)
                metric_mean_key = f'{metric}_mean'
                if metric_mean_key in method_results:
                    value = method_results[metric_mean_key]
                elif metric in method_results:
                    value = method_results[metric]
                else:
                    continue
                
                results[model_backbone][dataset_key][method_name] = float(value)
            
            print(f" Loaded: {model_backbone} - {dataset_key} ({len(methods_data)} methods)")
        
        except Exception as e:
            print(f" Failed to parse {json_file.name}: {e}")
    
    return dict(results)


def get_dataset_accuracy(results_dir, dataset_key, model_name='resnet18'):
    """
    Get the test accuracy for a dataset from its JSON file.
    Look in comprehensive_evaluation_results/in_distribution for accuracy data.
    
    Args:
        results_dir: Path to results directory
        dataset_key: Dataset key (e.g., 'breastmnist_standard', 'breastmnist_DA')
        model_name: Model name to filter by
    
    Returns:
        float: Test accuracy, or 0 if not found
    """
    # Navigate to comprehensive evaluation folder
    workspace_root = Path(results_dir).parent
    comp_eval_dir = workspace_root / 'benchmarks' / 'medMNIST' / 'utils' / 'comprehensive_evaluation_results' / 'in_distribution'
    
    if not comp_eval_dir.exists():
        return 0.0
    
    # Parse dataset_key to extract base name and setup
    # e.g., 'breastmnist_standard' -> ('breastmnist', 'standard')
    # e.g., 'breastmnist_DA' -> ('breastmnist', 'DA')
    if '_' in dataset_key:
        parts = dataset_key.rsplit('_', 1)
        base_name = parts[0]
        setup = parts[1] if parts[1] in ['standard', 'DA', 'DO', 'DADO'] else 'standard'
        # If the split didn't give us a valid setup, treat whole key as base name
        if setup == 'standard' and parts[1] != 'standard':
            base_name = dataset_key
            setup = 'standard'
    else:
        base_name = dataset_key
        setup = 'standard'
    
    # Build pattern based on setup
    if setup == 'standard':
        # For standard setup, filename doesn't have setup suffix
        pattern = f"comprehensive_metrics_{base_name}_{model_name}_standard.json"
        json_file = comp_eval_dir / pattern
        json_files = [json_file] if json_file.exists() else []
    else:
        # For DA/DO/DADO, filename includes setup
        pattern = f"comprehensive_metrics_{base_name}_{model_name}_{setup}.json"
        json_file = comp_eval_dir / pattern
        json_files = [json_file] if json_file.exists() else []
    
    if not json_files:
        return 0.0
    
    # Use the most recent file
    json_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        ensemble_metrics = data.get('ensemble_metrics', {})
        return ensemble_metrics.get('balanced_accuracy', 0.0)
    except:
        return 0.0


def get_ensemble_accuracy_from_runs(comp_eval_dir, dataset_key, model_name='resnet18', shift='corruption_shifts'):
    """
    Get ensemble balanced accuracy from comprehensive_evaluation_results/in_distribution
    
    Args:
        comp_eval_dir: Path to comprehensive evaluation directory
        dataset_key: Dataset key (e.g., 'tissuemnist_standard', 'tissuemnist_DA')
        model_name: Model name (e.g., 'resnet18', 'vit_b_16')
    
    Returns:
        float: Ensemble balanced accuracy, or 0 if not found
    """
    comp_eval_dir = Path(comp_eval_dir)
    
    # Parse dataset_key to get base name and setup
    if '_' in dataset_key:
        parts = dataset_key.rsplit('_', 1)
        base_name = parts[0]
        setup = parts[1] if parts[1] in ['standard', 'DA', 'DO', 'DADO'] else 'standard'
        if setup == 'standard' and parts[1] != 'standard':
            base_name = dataset_key
            setup = 'standard'
    else:
        base_name = dataset_key
        setup = 'standard'
    
    # Build pattern based on setup
    if shift == 'in_distribution':
        prefix = 'comprehensive_metrics_'
        suffix = ''
        standard = '_standard'
    else:
        prefix = ''
        suffix = '_severity3'
        standard = ''
        
    if setup == 'standard':
        pattern = f"{prefix}{base_name}_{model_name}{suffix}{standard}.json"
        json_file = comp_eval_dir / pattern
        json_files = [json_file] if json_file.exists() else []
    else:
        # For DA/DO/DADO, filename includes setup
        pattern = f"{prefix}{base_name}_{model_name}_{setup}{suffix}.json"
        json_file = comp_eval_dir / pattern
        json_files = [json_file] if json_file.exists() else []
    
    if not json_files:
        return 0.0
    
    # Use the most recent file
    json_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        if shift == 'corruption_shifts':
            ensemble_metrics = data.get('ensemble', {})
        else:
            ensemble_metrics = data.get('ensemble_metrics', {})
        return ensemble_metrics.get('balanced_accuracy', 0.0)
    except:
        return 0.0


def augrc_log_transform(value, max_display=0.30, scale_factor=50.0):
    """
    Transform AUGRC values with scaling that gives MORE space near edges (good values).
    Maps [0, max_display] where 0 is at edges (best) and max_display is at center (worst).
    Uses power > 1 (1.5) for MORE space at edges, LESS at center.
    
    Args:
        value: Original AUGRC value
        max_display: Maximum display value (center), typically 0.30 for standard shifts, 0.45 for population/new_class
        scale_factor: Controls overall scale
    
    Returns:
        Transformed value
    """
    # Use power 1.5 transform: scale * (max_display - value)^1.5
    # This gives MORE space near edges (0) and LESS at center (0.30)
    # When value = max_display: result = 0 (center)
    # When value = 0: result = scale * max_display^1.5 (edge)
    # Power 1.5 is less aggressive than square (2.0) but still emphasizes edges
    return scale_factor * (max_display - np.array(value)) ** 1.5


def create_radar_plot_on_axis(ax, model_results, model_name, runs_dir=None, metric='auroc_f', aggregation='mean', shift='corruption_shifts'):
    """
    Create a radar plot on a given axis for a single model showing all dataset-setup combinations.
    
    Args:
        ax: Matplotlib polar axis to plot on
        model_results: dict mapping dataset_key -> method -> metric_value (includes pre-computed ZScore aggregation methods)
        model_name: Name of the model (e.g., 'resnet18', 'vit_b_16')
        runs_dir: Path to runs directory (for ensemble balanced accuracy)
        metric: Metric to plot - 'auroc_f' or 'augrc'
        aggregation: Aggregation strategy - 'mean', 'min', 'max', or 'vote' (for display labeling)
        shift: Shift type - 'corruption_shifts' or 'in_distribution'
    """
    # Group datasets by family (base dataset name)
    dataset_families = defaultdict(list)
    for dataset_key in model_results.keys():
        # Extract base dataset name (before _setup)
        base_name = dataset_key.rsplit('_', 1)[0] if '_' in dataset_key else dataset_key
        # If it ends with _standard, remove it to get true base name
        if base_name.endswith('_standard'):
            base_name = base_name.replace('_standard', '')
        dataset_families[base_name].append(dataset_key)
    
    # Sort setups within each family: standard first, then alphabetically
    setup_order = {'standard': 0, 'DA': 1, 'DO': 2, 'DADO': 3}
    for base_name in dataset_families:
        dataset_families[base_name].sort(key=lambda k: (
            setup_order.get(k.split('_')[-1], 99),  # Setup priority
            k  # Then alphabetically
        ))
    
    # Manual ordering by classification performance (easiest to hardest)
    # Order: tissuemnist, dermamnist-e, breastmnist, pneumoniamnist, octmnist, pathmnist, organamnist, bloodmnist
    preferred_order = [
        'tissuemnist', 'dermamnist-e', 'breastmnist', 'pneumoniamnist', 
        'octmnist', 'pathmnist', 'organamnist', 'bloodmnist'
    ]
    
    # Build dataset list following the preferred order
    dataset_keys = []
    for base_name in preferred_order:
        if base_name in dataset_families:
            dataset_keys.extend(dataset_families[base_name])
    
    # Add any remaining datasets not in the preferred order
    for base_name in sorted(dataset_families.keys()):
        if base_name not in preferred_order:
            dataset_keys.extend(dataset_families[base_name])
    
    num_datasets = len(dataset_keys)
    
    if num_datasets == 0:
        print(f"No data for {model_name}, skipping radar plot")
        return
    
    # ZScore aggregation methods are loaded from JSON files.
    print(f" Using ZScore aggregation values from JSON files")
    
    # Get all unique methods across datasets
    all_methods = set()
    for dataset_data in model_results.values():
        all_methods.update(dataset_data.keys())
    all_methods = sorted(all_methods)
    all_methods.append(all_methods.pop(7))
    all_methods.insert(7, all_methods.pop(8))
    
    print(f"\nCreating radar plot for {model_name}")
    print(f" Datasets: {num_datasets}")
    print(f" Methods: {len(all_methods)}")
    print(f" Dataset order: {dataset_keys[:5]}... (showing first 5)")
    print(f" Dataset families found: {sorted(dataset_families.keys())}")
    print(f" Preferred order: {preferred_order}")
    
    # Cluster setups from same dataset with smaller angles
    # Allocate space per dataset family, then subdivide for setups
    angles = []
    dataset_labels = []  # Will store just the setup name (standard, DA, DO, DADO)
    family_angles = []  # Store angles for family name labels
    family_names = []
    
    # Use the actual dataset_keys order (which includes all datasets)
    # Group by family on the fly
    processed_families = set()
    current_family_idx = 0
    num_families_total = len(dataset_families)
    angle_per_family = 2 * np.pi / num_families_total
    within_family_factor = 0.85  # Wider spacing within family for better visibility
    
    i = 0
    while i < len(dataset_keys):
        dataset_key = dataset_keys[i]
        # Extract base family name
        base_name = dataset_key.rsplit('_', 1)[0] if '_' in dataset_key else dataset_key
        if base_name.endswith('_standard'):
            base_name = base_name.replace('_standard', '')
        
        if base_name not in processed_families:
            processed_families.add(base_name)
            family_datasets = dataset_families[base_name]
            
            family_center = current_family_idx * angle_per_family
            family_size = len(family_datasets)
            
            # Store family info for outer labels
            family_angles.append(family_center)
            family_names.append(base_name)
            
            if family_size == 1:
                angles.append(family_center)
                setup_name = family_datasets[0].split('_')[-1]
                # Replace 'standard' with 'S'
                if setup_name == 'standard':
                    setup_name = 'S'
                dataset_labels.append(setup_name)
            else:
                # Distribute setups within family's allocated space
                family_span = angle_per_family * within_family_factor
                for j, ds_key in enumerate(family_datasets):
                    offset = (j - (family_size - 1) / 2) * (family_span / family_size)
                    angle = (family_center + offset) % (2 * np.pi)  # Ensure positive
                    angles.append(angle)
                    setup_name = ds_key.split('_')[-1]
                    # Replace 'standard' with 'S'
                    if setup_name == 'standard':
                        setup_name = 'S'
                    dataset_labels.append(setup_name)
            
            current_family_idx += 1
            i += len(family_datasets)
        else:
            i += 1
    
    # Ensure all angles are positive and within [0, 2π]
    angles = [(a % (2 * np.pi)) for a in angles]
    
    # Rotate dataset positions anticlockwise by 30° for population/new_class shifts
    if shift in ['population_shift', 'new_class_shift']:
        rotation_offset = -np.pi / 6  # 30 degrees in radians
        angles = [(a + rotation_offset) % (2 * np.pi) for a in angles]
        family_angles = [(a + rotation_offset) % (2 * np.pi) for a in family_angles]
    
    # Store number of datasets before adding closing point
    num_angle_points = len(angles)
    angles = angles + [angles[0]]  # Complete the circle
    
    print(f" Angle range: {min(angles):.2f} to {max(angles):.2f} radians")
    print(f" Families: {len(family_names)}")
    print(f" Angle points (before close): {num_angle_points}, dataset_keys: {num_datasets}")
    
    # Color map for methods
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_methods)))
    
    # Pre-compute the innermost ring display value for AUGRC ID/CS out-of-bounds clamping.
    # Out-of-bounds points (AUGRC > 0.3) will be drawn at the innermost ring so the
    # connecting line is visible (reaching toward the center = worst performance).
    radar_center_display = None
    if metric == 'augrc' and shift not in ['population_shift', 'new_class_shift']:
        _augrc_max_pre = 0.3
        # 0.25 is the worst tick shown; its transformed value is the innermost ring (y_min)
        radar_center_display = float(augrc_log_transform(
            np.array([0.25]), max_display=_augrc_max_pre, scale_factor=50.0
        )[0])
    
    # Plot each method (lines only, no fill)
    for method_idx, method_name in enumerate(all_methods):
        values = []
        for dataset_key in dataset_keys:
            metric_value = model_results[dataset_key].get(method_name, np.nan)
            values.append(metric_value)
        
        # For visualization: apply display transform only for AUGRC with ID/CS shifts
        # This is purely visual - uses ORIGINAL AUGRC values (0-0.5, lower is better)
        values_display = values.copy()
        out_of_bounds_mask = [False] * len(values)
        if metric == 'augrc' and shift not in ['population_shift', 'new_class_shift']:
            augrc_max_display = 0.3
            # Mark values beyond the display limit
            out_of_bounds_mask = [
                (not np.isnan(v)) and (v > augrc_max_display) for v in values
            ]
            # Build display values: out-of-bounds → innermost ring (radar_center_display)
            # so the connecting line is drawn reaching toward the center
            values_display = []
            for v, oob in zip(values, out_of_bounds_mask):
                if oob:
                    values_display.append(radar_center_display)
                elif np.isnan(v):
                    values_display.append(np.nan)
                else:
                    values_display.append(
                        float(augrc_log_transform(np.array([v]), max_display=augrc_max_display, scale_factor=50.0)[0])
                    )
        
        # Complete the circle for display
        values_display += values_display[:1]
        
        # Plot ZScore_Aggregation_per_fold with lightning icon, others with lines
        if method_name == 'ZScore_Aggregation_per_fold':
            # Plot lightning bolt markers without connecting lines
            ax.scatter(angles[:-1], values_display[:-1], s=270, marker='*', 
                       color='red', label=method_name,# + ' (MSR - MSR_calibrated - MLS - GPS - KNN_Raw - MC_Dropout)', 
                        alpha=0.9, zorder=99,edgecolors='black', linewidths=0.5)
        elif method_name == 'ZScore_Aggregation_ensemble':
            # Red star marker for ZScore aggregation + ensemble
            ax.scatter(angles[:-1], values_display[:-1], s=300, marker='$\u26A1$', 
                       color=colors[method_idx], label='ZScore Agg + Ens',
                       alpha=0.9, zorder=100, edgecolors='black', linewidths=0.5)
        else:
            if method_name == "MSR_calibrated":
                method_name = "MSR-S"
            # Plot line (always, including clamped out-of-bounds points that go to center)
            ax.plot(angles, values_display, '-', linewidth=1.5, label=method_name,
                    color=colors[method_idx], alpha=0.85)
            # Plot markers only for in-bounds points (skip clamped positions)
            in_bounds_angles = [a for a, oob in zip(angles[:-1], out_of_bounds_mask) if not oob]
            in_bounds_values = [v for v, oob in zip(values_display[:-1], out_of_bounds_mask) if not oob]
            if in_bounds_angles:
                ax.plot(in_bounds_angles, in_bounds_values, 'o',
                        color=colors[method_idx], markersize=7, markeredgewidth=1,
                        markeredgecolor='white', alpha=0.85)
    
    # Set labels - setup names only (increased font size for readability)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dataset_labels, size=8, fontweight='medium', 
                       rotation=0, ha='center')
    
    # Ensure full circle is visible - CRITICAL for proper display
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_thetalim(0, 2 * np.pi)  # Force full circle view
    
    # Set y-axis range based on metric
    if metric == 'auroc_f':
        y_min, y_max = 0.4, 1.0
        tick_step = 0.1
        y_ticks = np.arange(y_min, y_max + 0.05, tick_step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:g}' for y in y_ticks], size=12, fontweight='medium')
    elif metric == 'augrc' and shift in ['population_shift', 'new_class_shift']:
        # Linear scale for PS/NCS - displaying original AUGRC values
        # AUGRC: 0 (best, edge) to 0.5 (worst, center) - lower is better
        y_min, y_max = 0.4, 0.03
        tick_step = 0.05
        y_ticks = np.arange(y_max, y_min + 0.01, tick_step)  # From 0.03 to 0.4
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:g}' for y in y_ticks], size=12, fontweight='medium')
    else:  # augrc for ID and CS - log transform for MORE space at edges (good performance)
        # Original AUGRC values (0-0.5, lower is better)
        augrc_max_display = 0.3
        # Original AUGRC tick values from bad (center) to good (edge)
        original_augrc_ticks = np.array([0.25, 0.2, 0.15, 0.10, 0.05, 0.01, 0.0])
        # Transform original AUGRC values using log transform (for positioning)
        transformed_ticks = augrc_log_transform(original_augrc_ticks, max_display=augrc_max_display, scale_factor=50.0)
        
        # Set limits in transformed space
        y_min, y_max = transformed_ticks[0], transformed_ticks[-1]
        ax.set_yticks(transformed_ticks)
        # Format labels using original AUGRC values (not inverted)
        tick_labels = []
        for y in original_augrc_ticks:
            if y < 0.01:
                tick_labels.append(f'{y:.3f}'.rstrip('0').rstrip('.'))
            elif y == 0.0:
                tick_labels.append('')  # Empty label for 0.0 (only for ID/CS shifts)
            else:
                tick_labels.append(f'{y:g}')
        ax.set_yticklabels(tick_labels, size=12, fontweight='medium')
    
    ax.set_ylim(y_min, y_max)
    ax.set_rlim(y_min, y_max)  # Also set radial limits explicitly
    
    # Set radial axis label position based on shift type
    if shift in ['population_shift', 'new_class_shift']:
        # PS/NCS: around 7 o'clock (between derma-e-ext and new_class_amos)
        ax.set_rlabel_position(200)
    else:
        # ID and CS: around 8 o'clock (between blood and organa)
        ax.set_rlabel_position(230)
    
    # Determine max_display for AUGRC based on shift type (for label positioning)
    if metric == 'augrc':
        if shift in ['population_shift', 'new_class_shift']:
            augrc_max_display = 0.4
        else:
            augrc_max_display = 0.3
    
    # Add dataset family names (larger, further out, with background)
    # Position based on metric scale for visibility (AFTER y_max is defined)
    if metric == 'auroc_f':
        label_position = 1.16
    elif metric == 'augrc' and shift in ['population_shift', 'new_class_shift']:
        # Linear scale for PS/NCS - position relative to max value
        label_position = -0.05
    else:  # augrc for ID/CS - position beyond edge in transformed space
        # Edge is at 0 which transforms to large value (y_max)
        # We want to be beyond that, so add extra offset to y_max
        label_position = y_max + 1.85  # Beyond the edge
    
    for angle, name in zip(family_angles, family_names):
        # Add subtle background box for family names
        angle_positive = angle % (2 * np.pi)  # Ensure positive angle
        # Remove 'mnist' suffix for cleaner display
        display_name = name.replace('mnist', '')
        
        # Custom angular adjustment for specific datasets to avoid overlap
        custom_angle = angle_positive
        if name == 'bloodmnist':
            # Shift blood label slightly around the circle to avoid overlap with pneumonia
            custom_angle = angle_positive + 0.35  # Rotate slightly clockwise
                
        elif name == 'new_class_amos2022':
            display_name = 'new class\namos2022'
            custom_angle = angle_positive -0.09
        elif name == 'new_class_midog':
            display_name = 'new class\nmidog++'
            custom_angle = angle_positive + 0.08
        elif name == 'dermamnist-e-external':
            display_name = 'derma-e\n-external'
            custom_angle = angle_positive + 0.07
        elif name == 'dermamnist-e-id':
            #display_name = 'derma-e\n-id'
            custom_angle = angle_positive + 0.18
        elif name == 'pneumoniamnist':
            custom_angle = angle_positive + 0.35
        elif name == 'amos2022':
            custom_angle = angle_positive - 0.1
        elif name == 'organamnist':
            custom_angle = angle_positive + 0.1
        elif name == 'breastmnist':
            custom_angle = angle_positive + 0.07
        elif name == 'octmnist':
            custom_angle = angle_positive - 0.05
            
        if name == 'tissuemnist' and metric == 'auroc_f':
            label_position_tissue = 1.12
            ax.text(custom_angle, label_position_tissue, display_name, 
                    horizontalalignment='center', verticalalignment='center',
                    size=12, fontweight='bold', transform=ax.transData,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
        elif name == 'tissuemnist' and metric == 'augrc':
            label_position_tissue = y_max + 1.7
            ax.text(custom_angle, label_position_tissue, display_name, 
                    horizontalalignment='center', verticalalignment='center',
                    size=12, fontweight='bold', transform=ax.transData,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
        elif name == 'octmnist' and metric == 'auroc_f':
            label_position_oct = 1.12
            ax.text(custom_angle, label_position_oct, display_name, 
                    horizontalalignment='center', verticalalignment='center',
                    size=12, fontweight='bold', transform=ax.transData,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
        elif name == 'pneumoniamnist' and metric == 'auroc_f':
            label_position_pneum = 1.1
            ax.text(custom_angle, label_position_pneum, display_name, 
                    horizontalalignment='center', verticalalignment='center',
                    size=12, fontweight='bold', transform=ax.transData,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
        elif name == 'bloodmnist' and metric == 'auroc_f':
            label_position_blood = 1.1
            ax.text(custom_angle, label_position_blood, display_name, 
                    horizontalalignment='center', verticalalignment='center',
                    size=12, fontweight='bold', transform=ax.transData,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
        elif name == 'pathmnist' and metric == 'auroc_f':
            label_position_path = 1.12
            ax.text(custom_angle, label_position_path, display_name, 
                    horizontalalignment='center', verticalalignment='center',
                    size=12, fontweight='bold', transform=ax.transData,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
        elif name == 'organamnist' and metric == 'auroc_f':
            label_position_organa = 1.17
            ax.text(custom_angle, label_position_organa, display_name, 
                    horizontalalignment='center', verticalalignment='center',
                    size=12, fontweight='bold', transform=ax.transData,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
        elif name == 'organamnist' and metric == 'augrc':
            label_position_organa = y_max + 2
            ax.text(custom_angle, label_position_organa, display_name, 
                    horizontalalignment='center', verticalalignment='center',
                    size=12, fontweight='bold', transform=ax.transData,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
            
        elif name == 'dermamnist-e-id' and metric == 'augrc':
            label_position_derma = y_max + 2.2
            ax.text(custom_angle, label_position_derma, display_name, 
                    horizontalalignment='center', verticalalignment='center',
                    size=12, fontweight='bold', transform=ax.transData,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
            
        else:
            ax.text(custom_angle, label_position, display_name, 
                    horizontalalignment='center', verticalalignment='center',
                    size=12, fontweight='bold', transform=ax.transData,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='gray', alpha=0.8))
    
    # Remove y-axis label for cleaner display
    metric_label = metric.upper().replace('_', ' ')
    ax.set_ylabel('', size=13, fontweight='bold', labelpad=35)
    
    # Enhanced grid
    ax.grid(True, linewidth=1.0, alpha=0.5, linestyle='--')
    ax.set_axisbelow(True)  # Grid behind data
    
    # Get legend handles and labels to return
    handles, labels = ax.get_legend_handles_labels()
    
    # Compute mean values for each method (return for histogram generation)
    method_means = {}
    
    # For each method, compute mean radar value (no normalization)
    for method_idx, method_name in enumerate(all_methods):
        values = []
        for dataset_key in dataset_keys:
            val = model_results[dataset_key].get(method_name, np.nan)
            # For AUGRC: invert so that 0 (best) -> 0.5 (edge) and 0.5 (worst) -> 0 (center)
            if metric == 'augrc' and not np.isnan(val):
                val = 0.5 - val
            values.append(val)
        
        # Filter out NaN values
        valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
        
        if len(valid_indices) == 0:  # Need at least 1 point
            method_means[method_name] = 0.0
            continue
        
        # Get filtered values
        filtered_values = [values[i] for i in valid_indices]
        
        # Compute simple mean (no normalization)
        method_means[method_name] = np.mean(filtered_values)
    
    return handles, labels, method_means, {}


def generate_summary_table(results, output_path):
    """
    Generate a summary table showing all results in CSV format.
    
    Args:
        results: Parsed results dictionary
        output_path: Path to save CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Model', 'Dataset', 'Method', 'AUROC_f', 'AUGRC'])
        
        # Data
        for model_name, model_results in sorted(results.items()):
            for dataset_key, dataset_data in sorted(model_results.items()):
                for method_name, auroc in sorted(dataset_data.items()):
                    writer.writerow([model_name, dataset_key, method_name, f"{auroc:.4f}"])
    
    print(f"\n Summary table saved to {output_path}")


def compute_mean_aggregation_metric(results_dir, dataset_key, model_name, metric='auroc_f', aggregation='mean', shift='in_distribution', use_ensemble=False):
    """
    Compute specified metric for aggregation of z-scored UQ methods.
    
    Computes per-fold aggregation by default, or ensemble-based if use_ensemble=True:
    - Per-fold (use_ensemble=False): For each fold: z-score normalize methods -> aggregate -> compute metric -> average across folds
    - Ensemble (use_ensemble=True): Z-score normalize ensemble scores -> aggregate -> compute single metric with ensemble indices
    
    Methods used: MSR, MSR_calibrated, MLS, GPS, KNN_Raw, MC_Dropout (when available)
    Excludes: TTA and Ensembling
    
    Args:
        results_dir: Path to results directory
        dataset_key: Dataset key (e.g., 'breastmnist_standard')
        model_name: Model name (e.g., 'resnet18', 'vit_b_16')
        metric: Metric to compute - 'auroc_f' or 'augrc'
        aggregation: Aggregation strategy - 'mean', 'min', 'max', or 'vote'
        shift: Shift type - 'in_distribution', 'corruption_shifts', 'population_shift', 'new_class_shift'
        use_ensemble: If True, use ensemble scores/indices. If False, use per-fold scores/indices (default)
    
    Returns:
        float: Metric value (single for ensemble, average across folds for per-fold), or NaN if not found
    """
    results_dir = Path(results_dir)
    
    # Check if this is a new_class dataset
    is_new_class = dataset_key.startswith('new_class_')
    if is_new_class:
        dataset_key = dataset_key.replace('new_class_', '', 1)
    
    # Parse dataset_key to get base dataset name and setup
    if '_' in dataset_key and dataset_key.split('_')[-1] in ['standard', 'DA', 'DO', 'DADO']:
        parts = dataset_key.rsplit('_', 1)
        dataset_name = parts[0]
        setup = parts[1]
    else:
        dataset_name = dataset_key
        setup = 'standard'
    
    # Construct npz filename pattern
    if setup == 'standard':
        pattern = f"all_metrics_{dataset_name}_{model_name}_*.npz"
    else:
        pattern = f"all_metrics_{dataset_name}_{model_name}_{setup}_*.npz"
    
    # Search in appropriate subdirectory based on shift type
    if is_new_class:
        search_dir = results_dir / 'new_class_shifts'
    elif shift == 'corruption_shifts':
        search_dir = results_dir / 'corruption_shifts'
    elif shift == 'population_shift':
        search_dir = results_dir / 'population_shifts'
    else:  # in_distribution
        # Try multiple directory structures
        if (results_dir / 'id').exists():
            search_dir = results_dir / 'id'
        elif (results_dir / 'in_distribution').exists():
            search_dir = results_dir / 'in_distribution'
        else:
            search_dir = results_dir / 'id_results' / 'log_results'
    
    # Find matching npz files
    all_npz_files = list(search_dir.glob(pattern))
    
    # Filter for standard setup (exclude DA, DO, DADO)
    if '_' in dataset_key and dataset_key.split('_')[-1] == 'standard':
        npz_files = [f for f in all_npz_files if not any(
            f'_{s}_' in f.name for s in ['DA', 'DO', 'DADO']
        )]
    else:
        npz_files = all_npz_files
    
    if not npz_files:
        return np.nan
    
    # Use most recent file
    npz_file = max(npz_files, key=lambda p: p.stat().st_mtime)
    
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        # Get cache file path
        cache_dir = results_dir / 'cache'
        if is_new_class:
            if setup == 'standard':
                cache_pattern = f"{dataset_name}_{model_name}_new_class_shift_test_results.npz"
            else:
                cache_pattern = f"{dataset_name}_{model_name}_{setup}_new_class_shift_test_results.npz"
        elif shift == 'corruption_shifts':
            if setup == 'standard':
                cache_pattern_v2 = f"{dataset_name}_{model_name}_corrupt3_test_test_results.npz"
                cache_pattern = f"{dataset_name}_{model_name}_corrupt3_test_results.npz"
                if (cache_dir / cache_pattern_v2).exists():
                    cache_pattern = cache_pattern_v2
            else:
                cache_pattern_v2 = f"{dataset_name}_{model_name}_{setup}_corrupt3_test_test_results.npz"
                cache_pattern = f"{dataset_name}_{model_name}_{setup}_corrupt3_test_results.npz"
                if (cache_dir / cache_pattern_v2).exists():
                    cache_pattern = cache_pattern_v2
        else:
            if setup == 'standard':
                cache_pattern_v2 = f"{dataset_name}_{model_name}_test_test_results.npz"
                cache_pattern = f"{dataset_name}_{model_name}_test_results.npz"
                if (cache_dir / cache_pattern_v2).exists():
                    cache_pattern = cache_pattern_v2
            else:
                cache_pattern_v2 = f"{dataset_name}_{model_name}_{setup}_test_test_results.npz"
                cache_pattern = f"{dataset_name}_{model_name}_{setup}_test_results.npz"
                if (cache_dir / cache_pattern_v2).exists():
                    cache_pattern = cache_pattern_v2
        
        cache_file_path = cache_dir / cache_pattern
        if not cache_file_path.exists():
            return np.nan
        
        cache_data = np.load(cache_file_path, allow_pickle=True)
        
        if use_ensemble:
            # ENSEMBLE MODE: Use ensemble scores with ensemble correct/incorrect indices
            # Use _ensemble suffix methods which represent logit averaging (average logits first, then softmax)
            # Include Ensembling_ensemble since it represents fold-to-fold variability
            method_keys = [k for k in data.keys() 
                          if k.endswith('_ensemble')
                          and k not in ['TTA_ensemble']]  # Only exclude TTA
            
            if not method_keys:
                return np.nan
            
            # Z-score normalize ensemble scores
            normalized_arrays = []
            for method_name in method_keys:
                uncertainties = data[method_name]
                mean_val = np.mean(uncertainties)
                std_val = np.std(uncertainties)
                
                if std_val > 0:
                    z_score = (uncertainties - mean_val) / std_val
                    normalized_arrays.append(z_score)
            
            if not normalized_arrays:
                return np.nan
            
            # Apply aggregation
            stacked = np.stack(normalized_arrays, axis=0)
            if aggregation == 'mean':
                aggregated = np.mean(stacked, axis=0)
            elif aggregation == 'min':
                aggregated = np.min(stacked, axis=0)
            elif aggregation == 'max':
                aggregated = np.max(stacked, axis=0)
            elif aggregation == 'vote':
                aggregated = np.sum(stacked > 0, axis=0) / len(normalized_arrays)
            else:
                return np.nan
            
            # Get ensemble correct/incorrect indices
            if 'correct_idx' in cache_data and 'incorrect_idx' in cache_data:
                correct_idx = np.asarray(cache_data['correct_idx'], dtype=int)
                incorrect_idx = np.asarray(cache_data['incorrect_idx'], dtype=int)
            elif 'per_fold_correct_idx' in cache_data:
                correct_idx = np.asarray(cache_data['per_fold_correct_idx'][0], dtype=int)
                incorrect_idx = np.asarray(cache_data['per_fold_incorrect_idx'][0], dtype=int)
            else:
                return np.nan
            
            n_samples = len(correct_idx) + len(incorrect_idx)
            failure_labels = np.zeros(n_samples)
            failure_labels[incorrect_idx] = 1
            
            if len(failure_labels) != len(aggregated):
                return np.nan
            
            # Compute metric
            if metric == 'auroc_f':
                from sklearn.metrics import roc_auc_score
                result = roc_auc_score(failure_labels, aggregated)
            elif metric == 'augrc':
                if 'y_pred' in cache_data and 'y_true' in cache_data:
                    predictions = cache_data['y_pred']
                    labels = cache_data['y_true']
                    
                    import sys
                    sys.path.insert(0, str(results_dir.parent))
                    from ToolBox.evaluation.evaluation import compute_augrc
                    
                    result, _ = compute_augrc(aggregated, predictions, labels,
                                             correct_idx=correct_idx, incorrect_idx=incorrect_idx)
                else:
                    return np.nan
            else:
                return np.nan
            
            return result
        
        else:
            # PER-FOLD MODE: Use per-fold scores with per-fold correct/incorrect indices
            per_fold_keys = [k for k in data.keys() 
                            if k.endswith('_per_fold') 
                            and k not in ['TTA_per_fold', 'Ensembling_per_fold']]
            
            if not per_fold_keys:
                return np.nan
            
            # Get number of folds
            num_folds = len(data[per_fold_keys[0]])
            
            # Get per-fold correct/incorrect indices
            if 'per_fold_correct_idx' in cache_data:
                per_fold_correct = [np.asarray(idx, dtype=int) for idx in cache_data['per_fold_correct_idx']]
                per_fold_incorrect = [np.asarray(idx, dtype=int) for idx in cache_data['per_fold_incorrect_idx']]
            else:
                return np.nan
            
            # For AUGRC, we also need per-fold predictions
            if metric == 'augrc':
                if 'per_fold_predictions' in cache_data:
                    per_fold_predictions = cache_data['per_fold_predictions']
                else:
                    per_fold_predictions = None
                y_true = cache_data['y_true']
            
            # Compute metric for each fold
            fold_metrics = []
            
            for fold_idx in range(num_folds):
                # Z-score normalize each method for this fold
                normalized_arrays = []
                
                for key in per_fold_keys:
                    uncertainties = data[key][fold_idx]
                    mean_val = np.mean(uncertainties)
                    std_val = np.std(uncertainties)
                    
                    if std_val > 0:
                        z_score = (uncertainties - mean_val) / std_val
                        normalized_arrays.append(z_score)
                
                if not normalized_arrays:
                    continue
                
                # Apply aggregation strategy
                stacked = np.stack(normalized_arrays, axis=0)
                if aggregation == 'mean':
                    aggregated = np.mean(stacked, axis=0)
                elif aggregation == 'min':
                    aggregated = np.min(stacked, axis=0)
                elif aggregation == 'max':
                    aggregated = np.max(stacked, axis=0)
                elif aggregation == 'vote':
                    aggregated = np.sum(stacked > 0, axis=0) / len(normalized_arrays)
                else:
                    continue
                
                # Get indices for this fold
                correct_idx = per_fold_correct[fold_idx]
                incorrect_idx = per_fold_incorrect[fold_idx]
                
                # Build failure labels
                n_samples = len(correct_idx) + len(incorrect_idx)
                failure_labels = np.zeros(n_samples)
                failure_labels[incorrect_idx] = 1
                
                if len(failure_labels) != len(aggregated):
                    continue
                
                # Compute metric for this fold
                if metric == 'auroc_f':
                    from sklearn.metrics import roc_auc_score
                    fold_metric = roc_auc_score(failure_labels, aggregated)
                elif metric == 'augrc':
                    if per_fold_predictions is None:
                        continue
                    
                    import sys
                    sys.path.insert(0, str(results_dir.parent))
                    from ToolBox.evaluation.evaluation import compute_augrc
                    
                    predictions = per_fold_predictions[fold_idx]
                    labels = y_true
                    
                    fold_metric, _ = compute_augrc(aggregated, predictions, labels, 
                                                   correct_idx=correct_idx, incorrect_idx=incorrect_idx)
                else:
                    continue
                
                fold_metrics.append(fold_metric)
            
            if not fold_metrics:
                return np.nan
            
            # Average across folds
            return np.mean(fold_metrics)
        
    except Exception as e:
        import traceback
        print(f" Warning: Could not compute mean aggregation for {dataset_key}: {e}")
        traceback.print_exc()
        return np.nan



def aggregate_uq_methods(npz_path, aggregation='mean', methods=None, output_path=None):
    """
    Aggregate multiple UQ methods by z-score normalizing each method and combining.
    
    Args:
        npz_path: Path to the npz file containing UQ method uncertainties
        aggregation: Aggregation strategy - 'mean', 'max', 'min', or 'vote'
        methods: List of method names to aggregate (None = all methods)
        output_path: Optional path to save aggregated results
    
    Returns:
        dict: Dictionary with aggregated uncertainties and per-method z-scores
            {
                'aggregated': array of aggregated uncertainties,
                'z_scores': dict of {method_name: z_scored_array},
                'methods_used': list of method names,
                'aggregation': aggregation type
            }
    """
    # Load the npz file
    data = np.load(npz_path, allow_pickle=True)
    
    # Get available methods (exclude '_per_fold' and '_ensemble' suffixes)
    all_keys = list(data.keys())
    method_keys = [k for k in all_keys if not k.endswith('_per_fold') and not k.endswith('_ensemble')]
    
    if methods is None:
        methods = method_keys
    else:
        # Filter to only requested methods that exist
        methods = [m for m in methods if m in method_keys]
    
    if not methods:
        raise ValueError(f"No valid methods found in {npz_path}. Available: {method_keys}")
    
    print(f"\nAggregating UQ methods from: {npz_path}")
    print(f" Methods to aggregate: {methods}")
    print(f" Aggregation strategy: {aggregation}")
    
    # Z-score normalize each method
    z_scores = {}
    normalized_arrays = []
    
    for method_name in methods:
        uncertainties = data[method_name]
        
        # Z-score normalization: (x - mean) / std
        mean = np.mean(uncertainties)
        std = np.std(uncertainties)
        
        if std == 0:
            print(f" Warning: {method_name} has zero std, skipping")
            continue
        
        z_score = (uncertainties - mean) / std
        z_scores[method_name] = z_score
        normalized_arrays.append(z_score)
        
        print(f" {method_name}: mean={mean:.4f}, std={std:.4f}")
    
    if not normalized_arrays:
        raise ValueError("No valid methods after normalization")
    
    # Stack all z-scored arrays (shape: num_methods x num_samples)
    stacked = np.stack(normalized_arrays, axis=0)
    
    # Aggregate across methods
    if aggregation == 'mean':
        aggregated = np.mean(stacked, axis=0)
    elif aggregation == 'max':
        aggregated = np.max(stacked, axis=0)
    elif aggregation == 'min':
        aggregated = np.min(stacked, axis=0)
    elif aggregation == 'vote':
        # Majority vote: count how many methods have z-score > 0 (above average uncertainty)
        aggregated = np.sum(stacked > 0, axis=0) / len(normalized_arrays)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Use 'mean', 'max', 'min', or 'vote'")
    
    print(f" Aggregated shape: {aggregated.shape}")
    print(f" Aggregated range: [{aggregated.min():.4f}, {aggregated.max():.4f}]")
    
    result = {
        'aggregated': aggregated,
        'z_scores': z_scores,
        'methods_used': [m for m in methods if m in z_scores],
        'aggregation': aggregation
    }
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        np.savez_compressed(
            output_path,
            aggregated=aggregated,
            **z_scores,
            methods_used=result['methods_used'],
            aggregation=aggregation
        )
        print(f" Saved to {output_path}")
    
    return result


def main(aggregation='mean', shift='corruption_shifts'):
    """Main function to generate separate radar plots based on shift type.
    
    Creates TWO separate figures:
    
    Figure 1 (ID + CS): 2x2 layout
    - Top row: AUROC_f (ResNet18 left, ViT right) for ID + CS datasets
    - Bottom row: AUGRC (ResNet18 left, ViT right) for ID + CS datasets
    
    Figure 2 (NCS + PS): 1x2 layout  
    - Top row: AUROC_f (ResNet18 left, ViT right) for NCS + PS datasets
    - Bottom row: AUGRC (ResNet18 left, ViT right) for NCS + PS datasets
    
    Args:
        aggregation: Aggregation strategy - 'mean', 'min', 'max', or 'vote'
        shift: Controls which figures to generate - 'all' generates both, 
               or specific shift type for single figure
    """
    
    print("=" * 80)
    print("UQ Benchmark Radar Plot Generator (Split Figures)")
    print(f"Aggregation: {aggregation.upper()}")
    print("=" * 80)
    
    # Get the workspace root (5 levels up from script: utils -> viz_benchmark_results -> utils -> medMNIST -> benchmarks -> UQ_Toolbox)
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent.parent
    results_dir = workspace_root / 'uq_benchmark_results'
    
    print(f"Workspace root: {workspace_root}")
    
    # Create output directory for plots in the script's directory
    output_dir = script_dir / 'radar_plots'
    output_dir.mkdir(exist_ok=True)
    
    # Determine which figures to generate
    if shift == 'all':
        shift_groups = [
            {'name': 'ID_CS', 'shifts': ['in_distribution', 'corruption_shifts'], 'title': 'In-Distribution & Corruption Shifts', 'layout': 'separate'},
            {'name': 'NCS_PS', 'shifts': ['new_class_shift', 'population_shift'], 'title': 'New Class & Population Shifts', 'layout': 'combined'}
        ]
    else:
        # Single shift type (backward compatibility)
        if shift == 'corruption_shifts':
            shift_name = 'CS'
            title = 'Corruption Shifts'
        elif shift == 'in_distribution':
            shift_name = 'ID'
            title = 'In-Distribution'
        elif shift == 'population_shift':
            shift_name = 'PS'
            title = 'Population Shift'
        elif shift == 'new_class_shift':
            shift_name = 'NCS'
            title = 'New Class Shift'
        else:
            shift_name = shift
            title = shift.replace('_', ' ').title()
        shift_groups = [{'name': shift_name, 'shifts': [shift], 'title': title, 'layout': 'single'}]
    
    # Process each shift group (creates separate figures)
    for group in shift_groups:
        print(f"\n{'='*80}")
        print(f"Processing: {group['title']}")
        print(f"{'='*80}")
        
        layout = group.get('layout', 'single')
        
        # Load results for each shift type separately (for 'separate' layout) or combined
        if layout == 'separate':
            # Load each shift separately - for ID/CS split display
            shift_results = {}
            for shift_type in group['shifts']:
                shift_results_dir = results_dir / 'jsons_results' / shift_type
                
                if not shift_results_dir.exists():
                    print(f" Warning: Directory not found: {shift_results_dir}")
                    continue
                
                print(f" Loading from: {shift_results_dir}")
                
                # Parse results for this shift type
                auroc_results = parse_results_directory(shift_results_dir, metric='auroc_f')
                augrc_results = parse_results_directory(shift_results_dir, metric='augrc')
                
                shift_results[shift_type] = {
                    'auroc_f': auroc_results,
                    'augrc': augrc_results
                }
            
            if not shift_results:
                print(f" ERROR: No results found for {group['title']}")
                continue
        else:
            # Combine results from all shifts in this group - for NCS/PS combined display
            combined_auroc = {}
            combined_augrc = {}
            
            for shift_type in group['shifts']:
                shift_results_dir = results_dir / 'jsons_results' / shift_type
                
                if not shift_results_dir.exists():
                    print(f" Warning: Directory not found: {shift_results_dir}")
                    continue
                
                print(f" Loading from: {shift_results_dir}")
                
                # Parse results for this shift type
                auroc_results = parse_results_directory(shift_results_dir, metric='auroc_f')
                augrc_results = parse_results_directory(shift_results_dir, metric='augrc')
                
                # Merge into combined results
                for model_name in auroc_results:
                    if model_name not in combined_auroc:
                        combined_auroc[model_name] = {}
                    combined_auroc[model_name].update(auroc_results[model_name])
                
                for model_name in augrc_results:
                    if model_name not in combined_augrc:
                        combined_augrc[model_name] = {}
                    combined_augrc[model_name].update(augrc_results[model_name])
            
            if not combined_auroc and not combined_augrc:
                print(f" ERROR: No results found for {group['title']}")
                continue
        
        model_names = ['resnet18', 'vit_b_16']
        all_handles = []
        all_labels = []
        
        # Different layouts based on group type
        if layout == 'separate':
            # Figure 1: 2x2 layout - ID and CS as separate radars
            # Top row: ID (ResNet18 left, ViT right)
            # Bottom row: CS (ResNet18 left, ViT right)
            fig = plt.figure(figsize=(24, 24))
            axes = [
                fig.add_subplot(2, 2, 1, projection='polar'),  # Top-left: ResNet18 ID
                fig.add_subplot(2, 2, 2, projection='polar'),  # Top-right: ViT ID
                fig.add_subplot(2, 2, 3, projection='polar'),  # Bottom-left: ResNet18 CS
                fig.add_subplot(2, 2, 4, projection='polar'),  # Bottom-right: ViT CS
            ]
            
            # Generate subplot for each shift × model combination
            plot_idx = 0
            for shift_idx, shift_type in enumerate(group['shifts']):
                if shift_type not in shift_results:
                    continue
                    
                for col, model_name in enumerate(model_names):
                    ax = axes[plot_idx]
                    
                    # Get results for this shift and model (using AUROC for display)
                    results = shift_results[shift_type]['auroc_f']
                    if not results or model_name not in results:
                        plot_idx += 1
                        continue
                    
                    model_results = results[model_name]
                    
                    print(f"\n  {'='*76}")
                    print(f" Generating {shift_type.upper()} plot for {model_name}...")
                    print(f" {'='*76}")
                    
                    # Get comp_eval_dir for this shift
                    comp_eval_dir = workspace_root / 'benchmarks' / 'medMNIST' / 'utils' / 'comprehensive_evaluation_results' / shift_type
                    
                    # Generate plot on this axis (only AUROC_f for separate layout)
                    handles, labels, method_surfaces, method_angles = create_radar_plot_on_axis(
                        ax, model_results, model_name, 
                        runs_dir=comp_eval_dir, 
                        metric='auroc_f', aggregation=aggregation, shift=shift_type
                    )
                    
                    # Collect legend info from first subplot only
                    if plot_idx == 0:
                        all_handles = handles
                        all_labels = labels
                    
                    # Add subplot title
                    model_display = model_name.replace('_', ' ').upper()
                    shift_display = 'ID' if shift_type == 'in_distribution' else 'CS'
                    ax.set_title(f'{model_display} - {shift_display}\\nPer-fold {aggregation.capitalize()} AUROC F',
                                fontsize=16, fontweight='bold', pad=40, x=-0.1, ha='left')
                    
                    plot_idx += 1
        
        else:
            # Figure 2: 1x2 layout - PS and NCS combined on same radar
            # Left: ResNet18 with both PS+NCS
            # Right: ViT with both PS+NCS
            fig = plt.figure(figsize=(24, 12))
            axes = [
                fig.add_subplot(1, 2, 1, projection='polar'),  # Left: ResNet18 PS+NCS
                fig.add_subplot(1, 2, 2, projection='polar'),  # Right: ViT PS+NCS
            ]
            
            # Generate subplot for each model (using combined PS+NCS data)
            for col, model_name in enumerate(model_names):
                ax = axes[col]
                
                results = combined_auroc  # Use AUROC for combined display
                if not results or model_name not in results:
                    continue
                
                model_results = results[model_name]
                
                print(f"\n  {'='*76}")
                print(f" Generating combined PS+NCS plot for {model_name}...")
                print(f" {'='*76}")
                
                # Use the first shift type for display parameters
                plot_shift = group['shifts'][0]
                
                # Get comp_eval_dir for this shift
                comp_eval_dir = workspace_root / 'benchmarks' / 'medMNIST' / 'utils' / 'comprehensive_evaluation_results' / plot_shift
                
                # Generate plot on this axis (AUROC_f only)
                handles, labels, method_surfaces, method_angles = create_radar_plot_on_axis(
                    ax, model_results, model_name, 
                    runs_dir=comp_eval_dir, 
                    metric='auroc_f', aggregation=aggregation, shift=plot_shift
                )
                
                # Collect legend info from first subplot only
                if col == 0:
                    all_handles = handles
                    all_labels = labels
                
                # Add subplot title
                model_display = model_name.replace('_', ' ').upper()
                ax.set_title(f'{model_display} - PS + NCS\\nPer-fold {aggregation.capitalize()} AUROC F',
                            fontsize=16, fontweight='bold', pad=40, x=-0.1, ha='left')
        
        # Move "ZScore_Aggregation_per_fold" to bottom of legend
        if 'ZScore_Aggregation_per_fold' in all_labels:
            idx = all_labels.index('ZScore_Aggregation_per_fold')
            all_labels.append(all_labels.pop(idx))
            all_handles.append(all_handles.pop(idx))
        
        # Add shared legend at the bottom center
        fig.legend(all_handles, all_labels, loc='lower center', ncol=5, 
                  fontsize=14, frameon=True, bbox_to_anchor=(0.5, -0.02))
        
        # Add main title
        fig.suptitle(f'CSF Performances - {group["title"]}', fontsize=24, fontweight='bold', y=0.98)
        
        # Adjust spacing
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save combined figure with group-specific name
        output_path = output_dir / f'radar_plots_{group["name"].lower()}_{aggregation}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n   Plot saved to {output_path}")
        
        plt.close(fig)
        
        # Generate summary table for this group
        summary_path = output_dir / f'results_summary_{group["name"].lower()}.csv'
        if layout == 'separate':
            # For separate layout, use results from first shift type
            first_shift = group['shifts'][0]
            if first_shift in shift_results:
                generate_summary_table(shift_results[first_shift]['auroc_f'], summary_path)
        else:
            # For combined layout, use combined results
            generate_summary_table(combined_auroc if combined_auroc else combined_augrc, summary_path)
    
    print("\n" + "=" * 80)
    print(" All plots generated successfully!")
    print("=" * 80)


def create_rank_radar_plot_on_axis(ax, ranked_results, model_name, aggregation='mean', shift='in_distribution'):
    """
    Create radar plot showing method ranks on a given axis.
    
    Args:
        ax: Matplotlib polar axis to plot on
        ranked_results: Dict[dataset][method] = rank (1 = best at center, higher = worse)
        model_name: Model name for display
        aggregation: Aggregation strategy name
        shift: Shift type name
        
    Returns:
        tuple: (handles, labels) for legend
    """
    from collections import defaultdict
    from scipy.stats import rankdata
    
    # Group datasets by family (base name without _standard, _DA, _DO, _DADO)
    dataset_families = defaultdict(list)
    for ds_key in ranked_results.keys():
        # Extract base dataset name (before _setup)
        base_name = ds_key.rsplit('_', 1)[0] if '_' in ds_key else ds_key
        # If it ends with _standard, remove it to get true base name
        if base_name.endswith('_standard'):
            base_name = base_name.replace('_standard', '')
        dataset_families[base_name].append(ds_key)
    
    # Sort setups within each family: standard first, then alphabetically
    setup_order = {'standard': 0, 'DA': 1, 'DO': 2, 'DADO': 3}
    for base_name in dataset_families:
        dataset_families[base_name].sort(key=lambda k: (
            setup_order.get(k.split('_')[-1], 99),
            k
        ))
    
    # Manual ordering by classification performance
    preferred_order = [
        'tissuemnist', 'dermamnist-e', 'breastmnist', 'pneumoniamnist', 
        'octmnist', 'pathmnist', 'organamnist', 'bloodmnist',
        'dermamnist-e-id', 'dermamnist-e-external', 'amos2022', 'new_class_amos2022'
    ]
    
    # Build dataset list following the preferred order
    dataset_keys = []
    for base_name in preferred_order:
        if base_name in dataset_families:
            dataset_keys.extend(dataset_families[base_name])
    
    # Add any remaining datasets not in the preferred order
    for base_name in sorted(dataset_families.keys()):
        if base_name not in preferred_order:
            dataset_keys.extend(dataset_families[base_name])
    
    num_datasets = len(dataset_keys)
    
    if num_datasets == 0:
        print(f" No datasets found for {model_name}")
        return None, None
    
    # Get all unique methods
    all_methods = set()
    for dataset_data in ranked_results.values():
        all_methods.update(dataset_data.keys())
    all_methods = sorted(all_methods)
    
    print(f" Plotting {len(all_methods)} methods across {num_datasets} datasets (ranks)")
    
    # Calculate angles for datasets (with grouped spacing)
    angles = []
    dataset_labels = []
    family_angles = []
    family_names = []
    
    # Process datasets sequentially, grouping by family on the fly
    processed_families = set()
    current_family_idx = 0
    num_families_total = len(dataset_families)
    angle_per_family = 2 * np.pi / num_families_total
    within_family_factor = 0.85  # Portion of angle_per_family used for setups within family
    
    i = 0
    while i < len(dataset_keys):
        dataset_key = dataset_keys[i]
        # Extract base family name
        base_name = dataset_key.rsplit('_', 1)[0] if '_' in dataset_key else dataset_key
        if base_name.endswith('_standard'):
            base_name = base_name.replace('_standard', '')
        
        if base_name not in processed_families:
            processed_families.add(base_name)
            family_datasets = dataset_families[base_name]
            
            family_center = current_family_idx * angle_per_family
            family_size = len(family_datasets)
            
            # Store family info for outer labels
            family_angles.append(family_center)
            family_names.append(base_name)
            
            if family_size == 1:
                angles.append(family_center)
                setup_name = family_datasets[0].split('_')[-1]
                if setup_name == 'standard':
                    setup_name = 'S'
                dataset_labels.append(setup_name)
            else:
                # Distribute setups within family's allocated space
                family_span = angle_per_family * within_family_factor
                for j, ds_key in enumerate(family_datasets):
                    offset = (j - (family_size - 1) / 2) * (family_span / family_size)
                    angle = (family_center + offset) % (2 * np.pi)
                    angles.append(angle)
                    setup_name = ds_key.split('_')[-1]
                    if setup_name == 'standard':
                        setup_name = 'S'
                    dataset_labels.append(setup_name)
            
            current_family_idx += 1
            i += len(family_datasets)
        else:
            i += 1
    
    # Rotate dataset positions for population/new_class shifts
    if shift in ['population_shift', 'new_class_shift']:
        rotation_offset = -np.pi / 6
        angles = [(a + rotation_offset) % (2 * np.pi) for a in angles]
        family_angles = [(a + rotation_offset) % (2 * np.pi) for a in family_angles]
    
    # Complete the circle
    angles = angles + [angles[0]]
    
    # Color map for methods
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_methods)))
    
    # Plot each method
    for method_idx, method_name in enumerate(all_methods):
        values = []
        for dataset_key in dataset_keys:
            rank = ranked_results[dataset_key].get(method_name, np.nan)
            # Add +1 offset for standard and DA setups (7 methods) to align with DO/DADO (8 methods)
            setup = dataset_key.split('_')[-1]
            if setup in ['standard', 'DA']:
                rank = rank + 1 if not np.isnan(rank) else np.nan
            values.append(rank)
        
        # Complete the circle
        values += values[:1]
        
        # Plot based on method type
        if method_name == 'ZScore_Aggregation_per_fold':
            ax.scatter(angles[:-1], values[:-1], s=270, marker='*', 
                       color='red', label=method_name, alpha=0.6, zorder=99,
                       edgecolors='black', linewidths=0.5)
        elif method_name == 'ZScore_Aggregation_ensemble':
            ax.scatter(angles[:-1], values[:-1], s=300, marker='$\u26A1$', 
                       color=colors[method_idx], label='ZScore Agg + Ens',
                       alpha=0.9, zorder=100, edgecolors='black', linewidths=0.5)
        else:
            display_name = method_name
            if method_name == "MSR_calibrated":
                display_name = "MSR-S"
            ax.plot(angles, values, 'o-', linewidth=2, label=display_name, 
                    color=colors[method_idx], markersize=8, markeredgewidth=2,
                    markeredgecolor='white', alpha=0.85)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dataset_labels, size=9, fontweight='medium', 
                       rotation=0, ha='center')
    
    # Set up circular display
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetalim(0, 2 * np.pi)
    
    # Set y-axis range for ranks (inverted: worst at center, best at edge)
    y_min, y_max = 0, 11
    y_ticks = np.arange(1, 11, 1)
    ax.set_yticks(y_ticks)
    # Show radial position (1 at center, 10 at edge)
    ax.set_yticklabels([f'{int(y)}' for y in y_ticks], size=11, fontweight='medium')
    
    ax.set_ylim(y_min, y_max)
    ax.set_rlim(y_min, y_max)
    
    # Set radial axis label position based on shift type
    if shift in ['population_shift', 'new_class_shift']:
        ax.set_rlabel_position(200)
    else:
        ax.set_rlabel_position(240)
    
    # Add dataset family names
    label_position = 12.5
    
    print(f" DEBUG: Adding {len(family_names)} family labels at radius {label_position}")
    for angle, name in zip(family_angles, family_names):
        # Add subtle background box for family names
        angle_positive = angle % (2 * np.pi)  # Ensure positive angle
        # Remove 'mnist' suffix for cleaner display
        display_name = name.replace('mnist', '')
        
        print(f" Family: {name} -> {display_name}, angle: {np.degrees(angle_positive):.1f}°")
        
        # Custom angular adjustment for specific datasets to avoid overlap
        custom_angle = angle_positive
        if name == 'bloodmnist':
            # Shift blood label slightly around the circle to avoid overlap with pneumonia
            custom_angle = angle_positive - 0.27  # Rotate slightly clockwise
                
        elif name == 'new_class_amos2022':
            display_name = 'new class\namos2022'
            custom_angle = angle_positive - 0.16
        elif name == 'dermamnist-e-external':
            display_name = 'derma-e\n-external'
            custom_angle = angle_positive - 0.1
        elif name == 'dermamnist-e-id':
            #display_name = 'derma-e\n-id'
            custom_angle = angle_positive + 0.18
        elif name == 'pneumoniamnist':
            custom_angle = angle_positive + 0.25
        elif name == 'amos2022':
            custom_angle = angle_positive - 0.25
        elif name == 'organamnist':
            custom_angle = angle_positive + 0.1
        elif name == 'breastmnist':
            custom_angle = angle_positive + 0.07
        elif name == 'octmnist':
            custom_angle = angle_positive - 0.05
        elif name == 'pathmnist':
            custom_angle = angle_positive - 0.25
        
        print(f" -> Adjusted angle: {np.degrees(custom_angle):.1f}°")
            
        # For ranks, use consistent label_position for all datasets
        ax.text(custom_angle, label_position, display_name, 
                horizontalalignment='center', verticalalignment='center',
                size=12, fontweight='bold', transform=ax.transData,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='gray', alpha=0.8))
    
    # Set axis label
    ax.set_ylabel('Rank', fontsize=12, fontweight='bold', labelpad=25)
    
    # Grid
    ax.grid(True, linewidth=1.0, alpha=0.5, linestyle='--')
    ax.set_axisbelow(True)
    
    # Get legend handles and labels to return
    handles, labels = ax.get_legend_handles_labels()
    
    return handles, labels


if __name__ == '__main__':
    import sys
    
    # Usage: python generate_radar_plots.py [mean|min|max|vote] [all|in_distribution|corruption_shifts|new_class_shift|population_shift]
    # Default: python generate_radar_plots.py mean all
    #   -> Generates both ID+CS figure and NCS+PS figure
    aggregation = sys.argv[1] if len(sys.argv) > 1 else 'mean'
    shift = sys.argv[2] if len(sys.argv) > 2 else 'all'
    
    if aggregation not in ['mean', 'min', 'max', 'vote']:
        print(f"Unknown aggregation: {aggregation}")
        print("Usage: python generate_radar_plots.py [mean|min|max|vote] [all|in_distribution|corruption_shifts|new_class_shift|population_shift]")
        sys.exit(1)
    
    # Generate radar plots with specified aggregation
    # shift='all' will create both figures (ID+CS and NCS+PS)
    # or specify individual shift type for single figure
    main(aggregation=aggregation, shift=shift)

