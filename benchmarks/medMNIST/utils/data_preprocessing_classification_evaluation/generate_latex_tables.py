#!/usr/bin/env python3
"""
Generate LaTeX tables for ViT vs ResNet18 comparison across all shifts.
Creates 4 tables: ResNet18 AUROC-F, ViT AUROC-F, ResNet18 AUGRC, ViT AUGRC.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Reuse the same directory and dataset definitions
ID_DIR = Path("Benchmarks/medMNIST/results/full_results/in_distribution")
CS_DIR = Path("Benchmarks/medMNIST/results/full_results/corruption_shifts")
PS_DIR = Path("Benchmarks/medMNIST/results/full_results/population_shifts")
NCS_DIR = Path("Benchmarks/medMNIST/results/full_results/new_class_shifts")

DATASETS_ID = {
    "dermamnist-e-id": "DermaMNIST",
    "breastmnist": "BreastMNIST",
    "tissuemnist": "TissueMNIST",
    "pneumoniamnist": "PneumoniaMNIST",
    "pathmnist": "PathMNIST",
    "octmnist": "OctMNIST",
    "organamnist": "OrganAMNIST",
    "bloodmnist": "BloodMNIST"
}

DATASETS_PS = {
    "dermamnist-e-external": "DermaMNIST",
    "amos2022": "OrganAMNIST"
}

DATASETS_NCS = {
    "amos2022": "OrganAMNIST",
    "midog": "MIDOG++"
}

SETUPS = ["", "DA", "DO", "DADO"]
MODELS = ["resnet18", "vit_b_16"]

# UQ Methods mapping (display name -> JSON key patterns)
UQ_METHODS = {
    "MSR": ["MSR"],
    "MSR-S": ["MSR_calibrated"],
    "MLS": ["MLS"],
    "TTA": ["TTA"],
    "GPS": ["GPS"],
    "KNN": ["KNN_Raw"],
    "MCD": ["MCDropout"],
    "Mean Agg": ["ZScore_Aggregation_per_fold"],
    "DE": ["Ensembling"],
    "Mean Agg+Ens": ["ZScore_Aggregation_ensemble"]
}


def find_json_files_by_pattern(directory, pattern_parts, exclude_patterns=None):
    """Find JSON files matching pattern parts."""
    all_files = list(directory.glob("uq_benchmark_*.json"))
    matching_files = []
    for f in all_files:
        if all(part in f.stem for part in pattern_parts):
            if exclude_patterns:
                if any(excl in f.stem for excl in exclude_patterns):
                    continue
            matching_files.append(f)
    
    if len(matching_files) == 0:
        return None
    elif len(matching_files) == 1:
        return matching_files[0]
    else:
        return max(matching_files, key=lambda p: p.stem.split('_')[-1])


def find_file_for_shift(shift_type, dataset_id, model, setup):
    """Find file for given shift type, dataset, model, and setup."""
    pattern_parts = [dataset_id, model]
    
    # Setup-specific exclusions
    if setup == "":
        exclude_patterns = ['_DA_', '_DO_', '_DADO_']
        setup_suffix = ""
    elif setup == "DA":
        pattern_parts.append("_DA_")
        exclude_patterns = ['_DADO_']
        setup_suffix = "_DA"
    elif setup == "DO":
        pattern_parts.append("_DO_")
        exclude_patterns = ['_DADO_', '_DA_']
        setup_suffix = "_DO"
    elif setup == "DADO":
        pattern_parts.append("_DADO_")
        exclude_patterns = None
        setup_suffix = "_DADO"
    
    if shift_type == "ID":
        directory = ID_DIR
    elif shift_type == "CS":
        directory = CS_DIR
        pattern_parts.append('corrupt_severity3_test')
        if setup == "":
            exclude_patterns = ['_DA_corrupt', '_DO_corrupt', '_DADO_corrupt']
    elif shift_type == "PS":
        directory = PS_DIR
        if exclude_patterns is None:
            exclude_patterns = []
        exclude_patterns.extend(['corrupt', 'new_class_shift'])
    elif shift_type == "NCS":
        directory = NCS_DIR
    else:
        return None
    
    return find_json_files_by_pattern(directory, pattern_parts, exclude_patterns)


def extract_method_metrics(json_path, metric_type="auroc_f"):
    """Extract metrics for all methods from JSON file.
    
    For fold column: use auroc_f_mean/augrc_mean and std
    For ensemble column: use main auroc_f/augrc field (first field in method)
    Special case: Mean_Aggregation_Ensemble only populates ensemble column
    
    Returns dict: {method_name: {"mean": X, "std": Y, "ensemble": Z}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    methods = data.get("methods", {})
    results = {}
    
    for method_name, method_data in methods.items():
        if not isinstance(method_data, dict):
            continue
        
        if metric_type == "auroc_f":
            mean_key = "auroc_f_mean"
            std_key = "auroc_f_std"
            ens_key = "auroc_f"  # Main ensemble result
        else:  # augrc
            mean_key = "augrc_mean"
            std_key = "augrc_std"
            ens_key = "augrc"  # Main ensemble result
        
        # Special handling for Mean_Aggregation_Ensemble - only ensemble value
        if method_name == "Mean_Aggregation_Ensemble":
            mean_val = method_data.get(ens_key)  # This goes in ensemble column
            results[method_name] = {
                "mean": None,  # No fold average for this method
                "std": None,
                "ensemble": mean_val  # Put the value in ensemble column
            }
        else:
            mean_val = method_data.get(mean_key)
            std_val = method_data.get(std_key)  # May be None for some methods
            ens_val = method_data.get(ens_key)
            
            results[method_name] = {
                "mean": mean_val,
                "std": std_val,
                "ensemble": ens_val
            }
    
    return results


def match_method_name(json_method_name, uq_methods_map):
    """Match JSON method name to standardized UQ method name."""
    # Try exact match first
    for standard_name, patterns in uq_methods_map.items():
        for pattern in patterns:
            if json_method_name == pattern:
                return standard_name
    
    # Fall back to case-insensitive substring match
    for standard_name, patterns in uq_methods_map.items():
        for pattern in patterns:
            if pattern.lower() in json_method_name.lower():
                return standard_name
    return None


def collect_all_data(model, metric_type):
    """Collect all data for one model and one metric type.
    
    Returns nested dict: 
    {shift_type: {dataset: {setup: {method: {"mean": X, "std": Y, "ensemble": Z}}}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    # Collect ID data
    for dataset_id, dataset_base in DATASETS_ID.items():
        for setup in SETUPS:
            setup_label = setup if setup != "" else "S"
            json_file = find_file_for_shift("ID", dataset_id, model, setup)
            
            if json_file:
                method_metrics = extract_method_metrics(json_file, metric_type)
                for json_method, metrics in method_metrics.items():
                    std_method = match_method_name(json_method, UQ_METHODS)
                    if std_method:
                        data["ID"][dataset_base][setup_label][std_method] = metrics
    
    # Collect CS data
    for dataset_id, dataset_base in DATASETS_ID.items():
        for setup in SETUPS:
            setup_label = setup if setup != "" else "S"
            json_file = find_file_for_shift("CS", dataset_id, model, setup)
            
            if json_file:
                method_metrics = extract_method_metrics(json_file, metric_type)
                for json_method, metrics in method_metrics.items():
                    std_method = match_method_name(json_method, UQ_METHODS)
                    if std_method:
                        data["CS"][dataset_base][setup_label][std_method] = metrics
    
    # Collect PS data
    for dataset_id, dataset_base in DATASETS_PS.items():
        for setup in SETUPS:
            setup_label = setup if setup != "" else "S"
            json_file = find_file_for_shift("PS", dataset_id, model, setup)
            
            if json_file:
                method_metrics = extract_method_metrics(json_file, metric_type)
                for json_method, metrics in method_metrics.items():
                    std_method = match_method_name(json_method, UQ_METHODS)
                    if std_method:
                        data["PS"][dataset_base][setup_label][std_method] = metrics
    
    # Collect NCS data
    for dataset_id, dataset_base in DATASETS_NCS.items():
        for setup in SETUPS:
            setup_label = setup if setup != "" else "S"
            json_file = find_file_for_shift("NCS", dataset_id, model, setup)
            
            if json_file:
                method_metrics = extract_method_metrics(json_file, metric_type)
                for json_method, metrics in method_metrics.items():
                    std_method = match_method_name(json_method, UQ_METHODS)
                    if std_method:
                        data["NCS"][dataset_base][setup_label][std_method] = metrics
    
    return data


def format_value_bold_max(value, std=None, is_max=False, metric_type="auroc_f"):
    """Format a value with optional std, bolding if is_max, using fixed-width box.
    
    For AUROC-F: round to 2 decimals, bold max
    For AUGRC: multiply by 1000, round to 1 decimal, bold min (smallest)
    """
    if value is None:
        return "\\makebox[2.5em][c]{---}"
    
    if metric_type == "augrc":
        # Multiply by 1000 and round to 1 decimal
        value = value * 1000
        if std is not None:
            std = std * 1000
            formatted = f"{value:.1f}${{\\scriptstyle\\pm}}${std:.1f}"
        else:
            formatted = f"{value:.1f}"
    else:  # auroc_f
        # Round to 2 decimals
        if std is not None:
            formatted = f"{value:.2f}${{\\scriptstyle\\pm}}${std:.2f}"
        else:
            formatted = f"{value:.2f}"
    
    if is_max:
        formatted = f"\\textbf{{{formatted}}}"
    
    # Use fixed-width box for consistent alignment
    # 4.5em for fold values (with ±), 2.5em for ensemble values (no ±)
    width = "4.5em" if std is not None else "2.5em"
    return f"\\makebox[{width}][c]{{{formatted}}}"


def get_display_name(dataset, shift):
    """Get the display name for a dataset in a table.
    
    For PS/NCS tables, show 'amos2022' instead of 'organamnist'.
    """
    if shift in ['PS', 'NCS'] and dataset == 'OrganAMNIST':
        return 'AMOS2022'
    return dataset


def generate_latex_table(model_name, metric_name, data, metric_type="auroc_f", shift_filter=None, 
                        include_opening=True, include_closing=True):
    """Generate LaTeX table code for one model and one metric.
    
    Args:
        shift_filter: List of shifts to include (e.g., ["ID"], ["CS"], ["PS", "NCS"])
        include_opening: Whether to include \\begin{landscape} and \\begin{table}[p]
        include_closing: Whether to include \\end{table} and \\end{landscape}
    """
    
    # Determine if we're looking for min (AUGRC) or max (AUROC-F)
    find_min = (metric_type == "augrc")
    
    # Determine all methods present in the data
    all_methods = set()
    for shift_data in data.values():
        for dataset_data in shift_data.values():
            for setup_data in dataset_data.values():
                all_methods.update(setup_data.keys())
    
    # Sort methods by our preferred order
    method_order = list(UQ_METHODS.keys())
    methods = [m for m in method_order if m in all_methods]
    
    # Build column structure
    all_shifts = ["ID", "CS", "PS", "NCS"]
    shifts = shift_filter if shift_filter else all_shifts
    
    # For PS/NCS tables, group datasets per shift (since they don't overlap)
    # For ID/CS tables, group datasets globally (since they're the same datasets)
    is_ps_ncs = set(shifts) == {"PS", "NCS"}
    
    if is_ps_ncs:
        # Group datasets per shift, but combine PS-pathmnist and NCS-organamnist into one subtable
        dataset_groups = []
        ps_datasets = sorted(list(data.get('PS', {}).keys())) if 'PS' in data else []
        ncs_datasets = sorted(list(data.get('NCS', {}).keys())) if 'NCS' in data else []
        
        # Group PS datasets first (in pairs)
        for i in range(0, len(ps_datasets), 2):
            if i + 1 < len(ps_datasets):
                # Full pair from PS
                dataset_groups.append(('PS', ps_datasets[i:i+2]))
            else:
                # Last PS dataset is alone - combine with NCS if available
                if ncs_datasets:
                    # Mixed group: PS-pathmnist and NCS-organamnist on same line
                    dataset_groups.append((['PS', 'NCS'], [ps_datasets[i], ncs_datasets[0]]))
                    ncs_datasets = ncs_datasets[1:]  # Remove the used NCS dataset
                else:
                    # Just the lone PS dataset
                    dataset_groups.append(('PS', [ps_datasets[i]]))
        
        # Add remaining NCS datasets (if any weren't combined above)
        for i in range(0, len(ncs_datasets), 2):
            dataset_groups.append(('NCS', ncs_datasets[i:i+2]))
    else:
        # Original logic: group datasets globally
        # Define custom order to ensure tissuemnist is last for odd-number handling
        dataset_order = ['BloodMNIST', 'BreastMNIST', 'DermaMNIST', 'OctMNIST', 
                         'OrganAMNIST', 'PneumoniaMNIST', 'TissueMNIST']
        
        all_datasets = set()
        # Only collect datasets from the shifts we're actually using
        for shift in shifts:
            if shift in data:
                all_datasets.update(data[shift].keys())
        
        # Sort by custom order, putting any unknown datasets at the end alphabetically
        sorted_datasets = sorted(list(all_datasets), 
                                key=lambda x: (dataset_order.index(x) if x in dataset_order 
                                              else len(dataset_order), x))
        
        # Split datasets into groups of 2
        dataset_groups = []
        for i in range(0, len(sorted_datasets), 2):
            dataset_groups.append(sorted_datasets[i:i+2])
    
    latex = []
    
    # Only add opening tags if requested
    if include_opening:
        latex.append("\\begin{landscape}")
        latex.append("\\begin{table}[p]")
        latex.append("\\vspace{-2cm}")  # Reduce top spacing
    
    latex.append("\\centering")
    latex.append("\\tiny")
    
    # Determine shift label for caption
    if shift_filter:
        if shift_filter == ["ID"]:
            shift_label = "In-Distribution"
        elif shift_filter == ["CS"]:
            shift_label = "Corruption Shifts"
        elif set(shift_filter) == {"PS", "NCS"}:
            shift_label = "Population \\& New Class Shifts"
        else:
            shift_label = ", ".join(shift_filter)
    else:
        shift_label = "All Shifts"
    
    caption_text = f"{model_name} {metric_name} - {shift_label}"
    if metric_type == "augrc":
        caption_text += " ($\\times 10^3$, lower is better)"
    latex.append(f"\\caption{{{caption_text}}}")
    
    # Create label-safe shift string
    shift_suffix = "_".join(shift_filter).lower() if shift_filter else "all"
    latex.append(f"\\label{{tab:{model_name.lower().replace(' ', '_')}_{metric_name.lower().replace('-', '_')}_{shift_suffix}}}")
    
    # Maximum columns: 1 method + 4 shifts × 2 datasets × 4 setups × 2 values = 1 + 64 = 65
    # But we'll have max 2 datasets per subtable, so: 1 + 4 shifts × 2 datasets × 4 setups × 2 values = 1 + 64/2 = 33
    # Realistically per 2 datasets across all shifts: 1 method + variable data columns
    
    # Process each dataset group
    for group_idx, dataset_group_item in enumerate(dataset_groups):
        # Extract dataset list and shifts for this group
        if is_ps_ncs:
            # dataset_group_item is either (shift, [datasets]) or ([shift1, shift2], [datasets])
            shift_or_shifts, dataset_group = dataset_group_item
            if isinstance(shift_or_shifts, list):
                # Mixed shifts (e.g., PS and NCS combined)
                # Create mapping: shift -> datasets for this mixed group
                group_shifts = shift_or_shifts
                shift_to_datasets = {}
                for idx, shift in enumerate(group_shifts):
                    if idx < len(dataset_group):
                        shift_to_datasets[shift] = [dataset_group[idx]]
                    else:
                        shift_to_datasets[shift] = []
            else:
                # Single shift
                group_shifts = [shift_or_shifts]
                shift_to_datasets = {shift_or_shifts: dataset_group}
        else:
            # dataset_group_item is [datasets]
            dataset_group = dataset_group_item
            group_shifts = shifts  # Iterate all shifts for this group
            shift_to_datasets = None  # Not used for ID/CS tables
        
        # Count columns for this group
        total_cols = 1  # Method column
        col_specs = "l"
        
        # Check if this is the last group with only 1 dataset
        # Only apply padding for ID and CS tables, not for PS/NCS
        is_last_group = (group_idx == len(dataset_groups) - 1)
        is_id_or_cs = shift_filter and ("ID" in shift_filter or "CS" in shift_filter) and not ("PS" in shift_filter or "NCS" in shift_filter)
        needs_padding = is_last_group and len(dataset_group) == 1 and is_id_or_cs
        
        # Track if this group has fewer than 2 datasets (need right-align padding)
        datasets_in_group = 0
        for shift in group_shifts:
            if shift not in data:
                continue
            datasets_for_shift = shift_to_datasets[shift] if shift_to_datasets else dataset_group
            for dataset in datasets_for_shift:
                if dataset in data[shift]:
                    datasets_in_group += 1
                    break  # Count dataset only once across shifts
        
        # Calculate columns needed for full 2-dataset width (for alignment)
        max_cols_per_shift = {}
        for shift in group_shifts:
            if shift not in data:
                continue
            # Count max columns for this shift (4 setups × 2 values = 8 columns per dataset)
            max_cols = 0
            # Get datasets for this shift (use mapping for mixed-shift groups)
            datasets_for_shift = shift_to_datasets[shift] if shift_to_datasets else dataset_group
            for dataset in datasets_for_shift:
                if dataset in data[shift]:
                    for setup in ["S", "DA", "DO", "DADO"]:
                        if setup in data[shift][dataset]:
                            max_cols += 2
            max_cols_per_shift[shift] = max_cols
        
        # Build column spec for actual data
        for shift in group_shifts:
            if shift not in data:
                continue
            datasets_for_shift = shift_to_datasets[shift] if shift_to_datasets else dataset_group
            for dataset in datasets_for_shift:
                if dataset in data[shift]:
                    for setup in ["S", "DA", "DO", "DADO"]:
                        if setup in data[shift][dataset]:
                            total_cols += 2
                            col_specs += "cc"
        
        # Add padding columns to column spec if needed
        if needs_padding:
            # Add 8 columns per shift (4 setups × 2 values = one full dataset)
            for shift in group_shifts:
                if shift in data:
                    for _ in range(8):
                        col_specs += "c"
                        total_cols += 1
        
        # Start subtable
        if group_idx > 0:
            latex.append("\\\\[2ex]")  # Add space between subtables
        
        latex.append(f"\\begin{{tabular}}{{{col_specs}}}")
        
        # Calculate actual data columns (excluding padding) for partial rules
        actual_data_cols = 1  # Method column
        for shift in group_shifts:
            if shift not in data:
                continue
            datasets_for_shift = shift_to_datasets[shift] if shift_to_datasets else dataset_group
            for dataset in datasets_for_shift:
                if dataset in data[shift]:
                    for setup in ["S", "DA", "DO", "DADO"]:
                        if setup in data[shift][dataset]:
                            actual_data_cols += 2
        
        # Use partial rule if this group has padding, otherwise full rule
        if needs_padding:
            latex.append(f"\\cmidrule(lr){{1-{actual_data_cols}}}")
        else:
            latex.append("\\toprule")
        
        # First subtable: full headers; continuation subtables: only dataset names
        if group_idx == 0:
            # Header row 1: Shift types
            header1 = ["\\multirow{3}{*}{\\textbf{Method}}"]
            for shift in group_shifts:
                if shift not in data:
                    continue
                n_cols = 0
                datasets_for_shift = shift_to_datasets[shift] if shift_to_datasets else dataset_group
                for dataset in datasets_for_shift:
                    if dataset in data[shift]:
                        for setup in ["S", "DA", "DO", "DADO"]:
                            if setup in data[shift][dataset]:
                                n_cols += 2
                if n_cols > 0:
                    header1.append(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{shift}}}}}")
            latex.append(" & ".join(header1) + " \\\\")
            if total_cols > 1:
                latex.append("\\cmidrule(lr){2-" + str(total_cols) + "}")
            
            # Header row 2: Datasets
            header2 = [""]
            for shift in group_shifts:
                if shift not in data:
                    continue
                datasets_for_shift = shift_to_datasets[shift] if shift_to_datasets else dataset_group
                for dataset in datasets_for_shift:
                    if dataset in data[shift]:
                        n_cols = 0
                        for setup in ["S", "DA", "DO", "DADO"]:
                            if setup in data[shift][dataset]:
                                n_cols += 2
                        if n_cols > 0:
                            display_name = get_display_name(dataset, shift)
                            header2.append(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textit{{{display_name}}}}}")
            latex.append(" & ".join(header2) + " \\\\")
            
            # Header row 3: Setups
            header3 = [""]
            for shift in group_shifts:
                if shift not in data:
                    continue
                datasets_for_shift = shift_to_datasets[shift] if shift_to_datasets else dataset_group
                for dataset in datasets_for_shift:
                    if dataset in data[shift]:
                        for setup in ["S", "DA", "DO", "DADO"]:
                            if setup in data[shift][dataset]:
                                header3.extend([f"\\multicolumn{{1}}{{c}}{{{setup}}}", ""])
            latex.append(" & ".join(header3) + " \\\\")
            
            # Header row 4: Fold/Ensemble
            header4 = [""]
            for shift in group_shifts:
                if shift not in data:
                    continue
                datasets_for_shift = shift_to_datasets[shift] if shift_to_datasets else dataset_group
                for dataset in datasets_for_shift:
                    if dataset in data[shift]:
                        for setup in ["S", "DA", "DO", "DADO"]:
                            if setup in data[shift][dataset]:
                                header4.extend(["Fold", "Ens"])
            latex.append(" & ".join(header4) + " \\\\")
        else:
            # Continuation: dataset names (and shift type for PS/NCS tables)
            if is_ps_ncs:
                # For PS/NCS, show shift type as first header row, then dataset names
                # Build shift header
                header_shift = ["\\textbf{Method}"]
                for shift in group_shifts:
                    if shift not in data:
                        continue
                    n_cols = 0
                    datasets_for_shift = shift_to_datasets[shift] if shift_to_datasets else dataset_group
                    for dataset in datasets_for_shift:
                        if dataset in data[shift]:
                            for setup in ["S", "DA", "DO", "DADO"]:
                                if setup in data[shift][dataset]:
                                    n_cols += 2
                    if n_cols > 0:
                        header_shift.append(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{shift}}}}}")
                latex.append(" & ".join(header_shift) + " \\\\")
                
                # Then dataset names
                header_datasets = [""]
            else:
                # For ID/CS, only show dataset names
                header_datasets = ["\\textbf{Method}"]
            
            # Note: needs_padding is already calculated earlier with proper is_id_or_cs check
            
            for shift in group_shifts:
                if shift not in data:
                    continue
                datasets_for_shift = shift_to_datasets[shift] if shift_to_datasets else dataset_group
                for dataset in datasets_for_shift:
                    if dataset in data[shift]:
                        n_cols = 0
                        for setup in ["S", "DA", "DO", "DADO"]:
                            if setup in data[shift][dataset]:
                                n_cols += 2
                        if n_cols > 0:
                            display_name = get_display_name(dataset, shift)
                            header_datasets.append(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textit{{{display_name}}}}}")
                
                # Add empty header space for padding if needed
                if needs_padding:
                    # Add 8 columns per shift (4 setups × 2 values)
                    padding_cols = 8
                    if padding_cols > 0:
                        header_datasets.append(f"\\multicolumn{{{padding_cols}}}{{c}}{{}}")  # Empty header
            
            latex.append(" & ".join(header_datasets) + " \\\\")
        
        # Use partial rule if this group has padding, otherwise full rule
        if needs_padding:
            latex.append(f"\\cmidrule(lr){{1-{actual_data_cols}}}")
        else:
            latex.append("\\midrule")
        
        # Data rows for this group - one per method
        for method in methods:
            row = [method]
            
            # Generate cells for this group's datasets only
            for shift in group_shifts:
                if shift not in data:
                    continue
                datasets_for_shift = shift_to_datasets[shift] if shift_to_datasets else dataset_group
                for dataset in datasets_for_shift:
                    if dataset in data[shift]:
                        for setup in ["S", "DA", "DO", "DADO"]:
                            if setup in data[shift][dataset]:
                                method_data = data[shift][dataset][setup].get(method, {})
                                fold_mean = method_data.get("mean")
                                fold_std = method_data.get("std")
                                ens_val = method_data.get("ensemble")
                                
                                # Find max/min for this column (all methods, this setup)
                                if find_min:
                                    fold_extreme = np.inf
                                    ens_extreme = np.inf
                                else:
                                    fold_extreme = -np.inf
                                    ens_extreme = -np.inf
                                
                                for m in methods:
                                    m_data = data[shift][dataset][setup].get(m, {})
                                    m_fold = m_data.get("mean")
                                    m_ens = m_data.get("ensemble")
                                    if m_fold is not None:
                                        if find_min:
                                            fold_extreme = min(fold_extreme, m_fold)
                                        else:
                                            fold_extreme = max(fold_extreme, m_fold)
                                    if m_ens is not None:
                                        if find_min:
                                            ens_extreme = min(ens_extreme, m_ens)
                                        else:
                                            ens_extreme = max(ens_extreme, m_ens)
                                
                                is_fold_extreme = (fold_mean is not None and fold_mean == fold_extreme)
                                is_ens_extreme = (ens_val is not None and ens_val == ens_extreme)
                                
                                row.append(format_value_bold_max(fold_mean, fold_std, is_fold_extreme, metric_type))
                                row.append(format_value_bold_max(ens_val, None, is_ens_extreme, metric_type))
            
            # If this is the last group with only 1 dataset in ID/CS tables, add padding for left-alignment
            if needs_padding:
                # Add 8 cells per shift (4 setups × 2 columns each) with phantom content to match width
                for shift in group_shifts:
                    if shift in data:
                        # Add cells with phantom content to maintain column width
                        for _ in range(8):
                            row.append("\\phantom{0.00${{\\scriptstyle\\pm}}$0.00}")
            
            latex.append(" & ".join(row) + " \\\\")
        
        # Use partial rule if this group has padding, otherwise full rule
        if needs_padding:
            latex.append(f"\\cmidrule(lr){{1-{actual_data_cols}}}")
        else:
            latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
    
    # Only add closing tags if requested
    if include_closing:
        latex.append("\\end{table}")
        latex.append("\\end{landscape}")
    
    latex.append("")
    
    return "\n".join(latex)


def main():
    """Generate all 4 tables."""
    
    output_file = "results_tables.tex"
    
    with open(output_file, 'w') as f:
        # Document preamble
        f.write("% Add these to your LaTeX preamble:\n")
        f.write("% \\usepackage{booktabs}\n")
        f.write("% \\usepackage{multirow}\n")
        f.write("% \\usepackage{pdflscape}\n")
        f.write("% \\usepackage{geometry}\n")
        f.write("% \\geometry{a4paper, margin=0.5in}\n\n")
        
        # Define shift type groups
        shift_configs = [
            (["ID"], "ID"),
            (["CS"], "CS"),
            (["PS", "NCS"], "PS_NCS")
        ]
        
        # Define model configs
        model_configs = [
            ("resnet18", "ResNet18"),
            ("vit_b_16", "ViT-B-16")
        ]
        
        # Define metric configs
        metric_configs = [
            ("auroc_f", "AUROC-F"),
            ("augrc", "AUGRC")
        ]
        
        # Reorganized structure: alternate models, combine PS/NCS on same page
        # For each metric:
        #   For each shift type (ID, CS, PS/NCS):
        #     For each model (ResNet18, ViT):
        #       Generate table
        #       Add page break EXCEPT when it's PS/NCS and going from R18 to ViT
        
        for metric_key, metric_name in metric_configs:
            for shift_filter, shift_name in shift_configs:
                is_ps_ncs = shift_name == "PS_NCS"
                
                for idx, (model_key, model_name) in enumerate(model_configs):
                    is_first_model = (idx == 0)
                    is_last_model = (idx == len(model_configs) - 1)
                    
                    print(f"Generating table for {model_name} {metric_name} - {shift_name}...")
                    data = collect_all_data(model_key, metric_key)
                    
                    # For PS/NCS, use custom opening/closing to keep both models on same page
                    if is_ps_ncs:
                        if is_first_model:
                            # First PS/NCS table: include opening, exclude closing
                            table = generate_latex_table(model_name, metric_name, data, metric_key, 
                                                        shift_filter=shift_filter, 
                                                        include_opening=True, include_closing=False)
                        elif is_last_model:
                            # Last PS/NCS table: exclude opening, include closing
                            table = generate_latex_table(model_name, metric_name, data, metric_key, 
                                                        shift_filter=shift_filter, 
                                                        include_opening=False, include_closing=True)
                        else:
                            # Middle tables: no wrappers
                            table = generate_latex_table(model_name, metric_name, data, metric_key, 
                                                        shift_filter=shift_filter, 
                                                        include_opening=False, include_closing=False)
                        
                        f.write(table)
                        
                        # Add spacing between PS/NCS tables on same page
                        if not is_last_model:
                            f.write("\n\\vspace{4ex}\n\n")
                    else:
                        # Normal tables: full wrappers, page breaks between
                        table = generate_latex_table(model_name, metric_name, data, metric_key, 
                                                    shift_filter=shift_filter)
                        f.write(table)
                        f.write("\n\n")
    
    print(f"\nLaTeX tables saved to: {output_file}")
    print(f"\nGenerated 12 tables (2 metrics × 3 shift types × 2 models)")
    print("\nTable organization:")
    print(" AUROC-F: R18-ID, ViT-ID, R18-CS, ViT-CS, [R18-PS/NCS + ViT-PS/NCS on same page]")
    print(" AUGRC:   R18-ID, ViT-ID, R18-CS, ViT-CS, [R18-PS/NCS + ViT-PS/NCS on same page]")
    print("\nNote: Each table may have multiple subtables (one per 2 datasets)")
    print("PS/NCS tables for both models are kept on the same page with \\vspace{2ex} separation")


if __name__ == "__main__":
    main()
