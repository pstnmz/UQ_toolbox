#!/usr/bin/env python3
"""
Generate LaTeX table for classification results across all shifts.
Columns: Datasets × Models × Setups × (Fold/Ens)
Rows: Shifts × Metrics (b_acc, AUC)
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Data directories
ID_DIR = Path("benchmarks/medMNIST/utils/comprehensive_evaluation_results/in_distribution")
CS_DIR = Path("benchmarks/medMNIST/utils/comprehensive_evaluation_results/corruption_shifts")
PS_DIR = Path("benchmarks/medMNIST/utils/comprehensive_evaluation_results/population_shift")

# Datasets to include (in order)
DATASETS = [
    ("bloodmnist", "BloodMNIST"),
    ("breastmnist", "BreastMNIST"),
    ("dermamnist-e", "DermaMNIST"),  # Use dermamnist-e-id for ID, dermamnist-e-ood for PS
    ("octmnist", "OctMNIST"),
    ("organamnist", "OrganAMNIST"),
    ("pneumoniamnist", "PneumoniaMNIST"),
    ("tissuemnist", "TissueMNIST"),
    ("pathmnist", "PathMNIST")
]

MODELS = [
    ("resnet18", "ResNet18"),
    ("vit_b_16", "ViT-B-16")
]

SETUPS = [
    ("standard", "S"),
    ("DA", "DA"),
    ("DO", "DO"),
    ("DADO", "DADO")
]

SHIFTS = [
    ("ID", "In-Distribution"),
    ("CS", "Corruption Shifts"),
    ("PS", "Population Shifts")
]

METRICS = [
    ("balanced_accuracy", "b\\_acc"),
    ("auc", "AUC")
]


def find_json_file(shift_type, dataset_id, model, setup):
    """Find the JSON file for given shift, dataset, model, and setup."""
    
    # Map dataset IDs for special cases
    if shift_type in ["ID", "CS"] and dataset_id == "dermamnist-e":
        dataset_search = "dermamnist-e-id"
    elif shift_type == "PS" and dataset_id == "dermamnist-e":
        dataset_search = "dermamnist-e-ood"
    elif shift_type == "PS" and dataset_id == "organamnist":
        dataset_search = "amos22"
    else:
        dataset_search = dataset_id
    
    # For ID and PS: always include setup suffix (including "_standard")
    # For CS: use "" for standard, otherwise "_SETUP"
    if shift_type in ["ID", "PS"]:
        setup_suffix = f"_{setup}"
    else:  # CS
        setup_suffix = "" if setup == "standard" else f"_{setup}"
    
    # Build filename pattern
    if shift_type == "ID":
        directory = ID_DIR
        filename = f"comprehensive_metrics_{dataset_search}_{model}{setup_suffix}.json"
    elif shift_type == "CS":
        directory = CS_DIR
        filename = f"{dataset_search}_{model}{setup_suffix}_severity3.json"
    elif shift_type == "PS":
        directory = PS_DIR
        filename = f"comprehensive_metrics_{dataset_search}_{model}{setup_suffix}.json"
    else:
        return None
    
    filepath = directory / filename
    return filepath if filepath.exists() else None


def extract_metrics(json_path, metric_name):
    """Extract fold and ensemble metrics from JSON file.
    
    Returns: (fold_mean, fold_std, ensemble_value)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if "per_fold_metrics" in data:
        # ID and PS format
        per_fold = data["per_fold_metrics"]
        ensemble = data.get("ensemble_metrics", {})
    elif "per_fold" in data:
        # CS format
        per_fold = data["per_fold"]
        ensemble = data.get("ensemble", {})
    else:
        return None, None, None
    
    # Extract fold values
    fold_values = [fold.get(metric_name) for fold in per_fold if metric_name in fold]
    
    if fold_values:
        fold_mean = np.mean(fold_values)
        fold_std = np.std(fold_values, ddof=1) if len(fold_values) > 1 else 0.0
    else:
        fold_mean = None
        fold_std = None
    
    # Extract ensemble value
    ensemble_value = ensemble.get(metric_name)
    
    return fold_mean, fold_std, ensemble_value


def format_value(value, std=None, is_bold=False, precision=2):
    """Format a value with optional std, bolding if is_bold.
    
    Args:
        value: The value to format
        std: Standard deviation (optional)
        is_bold: Whether to bold the value (for max in column)
        precision: Number of decimal places (default 2)
    """
    if value is None:
        return "\\makebox[2.5em][c]{---}"
    
    # Check for NaN
    if np.isnan(value):
        return "\\makebox[2.5em][c]{---}"
    
    if std is not None:
        if np.isnan(std):
            formatted = f"{value:.{precision}f}"
            width = "2.5em"
        else:
            formatted = f"{value:.{precision}f}${{\\scriptstyle\\pm}}${std:.{precision}f}"
            width = "4.5em"
    else:
        formatted = f"{value:.{precision}f}"
        width = "2.5em"
    
    if is_bold:
        formatted = f"\\textbf{{{formatted}}}"
    
    return f"\\makebox[{width}][c]{{{formatted}}}"


def collect_all_data():
    """Collect all classification data.
    
    Returns nested dict: 
    {shift: {metric: {dataset: {model: {setup: {"fold_mean": X, "fold_std": Y, "ensemble": Z}}}}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    
    for shift, _ in SHIFTS:
        # Skip PS for datasets without external test sets
        skip_datasets = []
        if shift == "PS":
            skip_datasets = ["bloodmnist", "breastmnist", "octmnist", "pneumoniamnist", 
                           "tissuemnist", "pathmnist"]
        
        for dataset_id, _ in DATASETS:
            if dataset_id in skip_datasets:
                continue
                
            for model, _ in MODELS:
                for setup, setup_label in SETUPS:
                    json_file = find_json_file(shift, dataset_id, model, setup)
                    
                    if json_file:
                        for metric_key, _ in METRICS:
                            fold_mean, fold_std, ensemble_val = extract_metrics(json_file, metric_key)
                            # Store with setup_label (S, DA, DO, DADO) not setup (standard, DA, DO, DADO)
                            data[shift][metric_key][dataset_id][model][setup_label] = {
                                "fold_mean": fold_mean,
                                "fold_std": fold_std,
                                "ensemble": ensemble_val
                            }
    
    return data


def generate_latex_table(data):
    """Generate LaTeX table code for classification results.
    
    Creates one subtable per dataset with structure:
    Rows: Shift × Metric
    Columns: Model × Setup × (Fold/Ens)
    All datasets stacked vertically with headers only for the first.
    """
    
    latex = []
    
    # Document setup
    latex.append("% Add these to your LaTeX preamble:")
    latex.append("% \\usepackage{booktabs}")
    latex.append("% \\usepackage{multirow}")
    latex.append("% \\usepackage{pdflscape}")
    latex.append("% \\usepackage{geometry}")
    latex.append("% \\geometry{a4paper, margin=0.3in}")
    latex.append("")
    
    latex.append("\\begin{landscape}")
    latex.append("\\begin{table}[t!]")
    latex.append("\\vspace*{-2cm}")
    latex.append("\\centering")
    latex.append("\\tiny")
    latex.append("\\caption{Classification Performance Across All Datasets and Shifts (Balanced Accuracy and AUC)}")
    latex.append("\\label{tab:classification_all}")
    
    # Column spec: 1 (metric) + models × setups × 2 (fold/ens)
    # 2 models × 4 setups × 2 values = 16 + 1 = 17 columns
    col_spec = "l" + "cc" * (len(MODELS) * len(SETUPS))
    
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header row 1: Models (spanning setups) - only for first dataset
    header1 = ["\\multirow{2}{*}{\\textbf{Shift:Metric}}"]
    for model, model_name in MODELS:
        n_cols = len(SETUPS) * 2  # 4 setups × 2 values = 8
        header1.append(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{model_name}}}}}")
    latex.append(" & ".join(header1) + " \\\\")
    latex.append("\\cmidrule(lr){2-" + str(1 + len(MODELS) * len(SETUPS) * 2) + "}")
    
    # Header row 2: Setups
    header2 = [""]
    for model, _ in MODELS:
        for setup, setup_label in SETUPS:
            header2.append(f"\\multicolumn{{2}}{{c}}{{{setup_label}}}")
    latex.append(" & ".join(header2) + " \\\\")
    
    # Header row 3: Fold/Ens
    header3 = [""]
    for model, _ in MODELS:
        for setup, _ in SETUPS:
            header3.extend(["Fold", "Ens"])
    latex.append(" & ".join(header3) + " \\\\")
    latex.append("\\midrule")
    
    # Generate one subtable per dataset
    for dataset_idx, (dataset_id, dataset_name) in enumerate(DATASETS):
        if dataset_idx > 0:
            latex.append("\\midrule")
        
        # Dataset name row (no headers - they're at the top for all datasets)
        latex.append(f"\\multicolumn{{{1 + len(MODELS) * len(SETUPS) * 2}}}{{l}}{{\\textbf{{{dataset_name}}}}} \\\\")
        
        # Data rows: one row per shift × metric combination
        for shift_idx, (shift, shift_name) in enumerate(SHIFTS):
            # Check if this dataset has data for this shift  
            has_shift_data = (shift in data and 
                            any(metric_key in data[shift] and dataset_id in data[shift][metric_key] 
                                for metric_key, _ in METRICS))
            
            if not has_shift_data:
                continue  # Skip shifts with no data (e.g., PS for most datasets)
            
            for metric_idx, (metric_key, metric_label) in enumerate(METRICS):
                row_label = f"{shift}: {metric_label}"
                row = [row_label]
                
                # Check if this specific metric has data
                has_data = (shift in data and 
                           metric_key in data[shift] and 
                           dataset_id in data[shift][metric_key])
                
                if has_data:
                    # Find max values for bolding (across all models/setups)
                    fold_max = -np.inf
                    ens_max = -np.inf
                    for m, _ in MODELS:
                        for s, s_label in SETUPS:
                            m_data = data[shift][metric_key][dataset_id][m].get(s_label, {})
                            m_fold = m_data.get("fold_mean")
                            m_ens = m_data.get("ensemble")
                            if m_fold is not None and not np.isnan(m_fold):
                                fold_max = max(fold_max, m_fold)
                            if m_ens is not None and not np.isnan(m_ens):
                                ens_max = max(ens_max, m_ens)
                    
                    # Generate cells for each model × setup
                    for model, _ in MODELS:
                        for setup, setup_label in SETUPS:
                            metrics = data[shift][metric_key][dataset_id][model].get(setup_label, {})
                            fold_mean = metrics.get("fold_mean")
                            fold_std = metrics.get("fold_std")
                            ensemble_val = metrics.get("ensemble")
                            
                            is_fold_max = (fold_mean is not None and not np.isnan(fold_mean) and fold_mean == fold_max)
                            is_ens_max = (ensemble_val is not None and not np.isnan(ensemble_val) and ensemble_val == ens_max)
                            
                            row.append(format_value(fold_mean, fold_std, is_fold_max))
                            row.append(format_value(ensemble_val, None, is_ens_max))
                else:
                    # No data for this metric - fill with dashes
                    for model, _ in MODELS:
                        for setup, _ in SETUPS:
                            row.append("---")
                            row.append("---")
                
                latex.append(" & ".join(row) + " \\\\")
            
            # Add midrule after each shift except the last one with data
            if shift_idx < len(SHIFTS) - 1:
                # Check if there's another shift with data
                remaining_shifts = SHIFTS[shift_idx + 1:]
                has_more_data = any(
                    s in data and any(
                        metric_key in data[s] and dataset_id in data[s][metric_key] 
                        for metric_key, _ in METRICS
                    )
                    for s, _ in remaining_shifts
                )
                if has_more_data:
                    latex.append("\\midrule")
    
    # Close the single table after all datasets
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("\\end{landscape}")
    latex.append("")
    
    return "\n".join(latex)


def main():
    """Generate the classification results table."""
    
    print("Collecting classification data...")
    data = collect_all_data()
    
    print("Generating LaTeX tables...")
    latex_table = generate_latex_table(data)
    
    output_file = "classification_results_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to: {output_file}")
    print("\nTable structure:")
    print(f"  Single table with all {len(DATASETS)} datasets stacked vertically")
    print(f"  Columns: Models (2) × Setups (4) × (Fold/Ens) = 16 + 1 metric column")
    print(f"  Rows: One subtable per dataset, with Shifts × Metrics (2) rows each")
    print("\nNote: Population Shifts only show data for DermaMNIST and OrganAMNIST (AMOS22)")


if __name__ == "__main__":
    main()
