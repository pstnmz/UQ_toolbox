#!/usr/bin/env python3
"""
Comparative analysis of classification results.

Generates:
1. Mean bAcc differences: organamnist ID vs amos22 PS, pathmnist ID vs hmu-crc PS
2. ViT vs ResNet18 breakdown by shift (ID + Corruption)
2b. ViT vs ResNet18 on POPULATION SHIFTS (AMOS22, HMU-CRC, DermaExt)
3. DA vs Standard breakdown by shift (ID, Corruption, Population)
4. DO vs Standard breakdown by shift (ID, Corruption, Population)
5. Ensemble vs Mean Fold bAcc by shift (ID, Corruption, Population)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Data directories
RESULTS_DIR = Path("/workspace/Benchmarks/medMNIST/results/classification_results")
ID_DIR = RESULTS_DIR / "in_distribution"
CS_DIR = RESULTS_DIR / "corruption_shifts"
PS_DIR = RESULTS_DIR / "population_shift"

def load_json(filepath):
    """Load and parse JSON file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except:
        return None

def extract_bacc(data, metric_name="balanced_accuracy"):
    """Extract fold mean, fold std, and ensemble bAcc from JSON data."""
    if not data:
        return None, None, None
    
    # Handle different JSON structures
    if "per_fold_metrics" in data:
        per_fold = data["per_fold_metrics"]
        ensemble = data.get("ensemble_metrics", {})
    elif "per_fold" in data:
        per_fold = data["per_fold"]
        ensemble = data.get("ensemble", {})
    else:
        return None, None, None
    
    fold_values = [f.get(metric_name) for f in per_fold if metric_name in f]
    
    if fold_values:
        fold_mean = np.mean(fold_values)
        fold_std = np.std(fold_values, ddof=1) if len(fold_values) > 1 else 0.0
    else:
        fold_mean, fold_std = None, None
    
    ensemble_val = ensemble.get(metric_name)
    
    return fold_mean, fold_std, ensemble_val

def load_dataset_results(dataset_name, shift_type):
    """Load all results for a dataset across all models and setups."""
    results = {}
    
    if shift_type == "id":
        directory = ID_DIR
        pattern = f"comprehensive_metrics_{dataset_name}_*_*.json"
    elif shift_type == "corruption":
        directory = CS_DIR
        pattern = f"{dataset_name}_*_*_severity3.json"
    elif shift_type == "ps":
        directory = PS_DIR
        pattern = f"comprehensive_metrics_{dataset_name}_*_*.json"
    else:
        return results
    
    for filepath in directory.glob(pattern):
        data = load_json(filepath)
        if not data:
            continue
        
        name = filepath.stem
        if "comprehensive_metrics_" in name:
            name = name.replace("comprehensive_metrics_", "").replace(dataset_name + "_", "")
        else:
            name = name.replace("_severity3", "").replace(dataset_name + "_", "")
        
        parts = name.split("_")
        model = "_".join([p for p in parts if p in ["resnet18", "vit", "b", "16"]]).replace("_b_16", "_b_16") or "resnet18"
        setup = "_".join([p for p in parts if p in ["DA", "DO", "DADO", "standard"]]) or "standard"
        
        if "vit" in name:
            model = "vit_b_16"
        elif "resnet18" in name:
            model = "resnet18"
        
        if "DADO" in name:
            setup = "DADO"
        elif "DO" in name and "DADO" not in name:
            setup = "DO"
        elif "DA" in name and "DADO" not in name:
            setup = "DA"
        else:
            setup = "standard"
        
        fold_mean, fold_std, ens_val = extract_bacc(data)
        results[(model, setup)] = {
            "fold_mean": fold_mean,
            "fold_std": fold_std,
            "ensemble": ens_val
        }
    
    return results

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*90}")
    print(f"{title}")
    print(f"{'='*90}")

def print_subsection(subtitle):
    """Print a formatted subsection header."""
    print(f"\n{subtitle}")
    print("-" * len(subtitle))

# ============================================================================
# 1. ID vs PS differences (organamnist/pathmnist)
# ============================================================================

print_section("1. ID vs POPULATION SHIFT DIFFERENCES")

for dataset_id, dataset_ps, dataset_name in [
    ("organamnist", "amos22", "OrganAMNIST/AMOS22"),
    ("pathmnist", "hmu-crc", "PathMNIST/HMU-CRC"),
]:
    print_subsection(f"{dataset_name}")
    
    id_results = load_dataset_results(dataset_id, "id")
    ps_results = load_dataset_results(dataset_ps, "ps")
    
    differences_fold = []
    differences_ens = []
    
    for (model, setup) in sorted(set(list(id_results.keys()) + list(ps_results.keys()))):
        id_data = id_results.get((model, setup), {})
        ps_data = ps_results.get((model, setup), {})
        
        id_fold = id_data.get("fold_mean")
        id_ens = id_data.get("ensemble")
        ps_fold = ps_data.get("fold_mean")
        ps_ens = ps_data.get("ensemble")
        
        if id_fold and ps_fold:
            diff_fold = id_fold - ps_fold
            differences_fold.append(diff_fold)
            print(f"  {model:10s} {setup:8s}  |  ID fold: {id_fold:.4f}, PS fold: {ps_fold:.4f}  →  Δ: {diff_fold:+.4f}")
        
        if id_ens and ps_ens:
            diff_ens = id_ens - ps_ens
            differences_ens.append(diff_ens)
    
    if differences_fold:
        print(f"\n  Fold mean difference (ID - PS):      {np.mean(differences_fold):+.4f} ± {np.std(differences_fold):.4f}")
    if differences_ens:
        print(f"  Ensemble mean difference (ID - PS):  {np.mean(differences_ens):+.4f} ± {np.std(differences_ens):.4f}")

# ============================================================================
# 2. ViT vs ResNet18 by shift
# ============================================================================

print_section("2. ViT vs ResNet18 BREAKDOWN BY SHIFT")

for shift_name, shift_key, directory in [
    ("In-Distribution", "id", ID_DIR),
    ("Corruption Shifts (s3)", "corruption", CS_DIR),
]:
    print_subsection(f"{shift_name}")
    
    vit_folds = []
    vit_ens = []
    r18_folds = []
    r18_ens = []
    
    # Load all files and extract ViT vs R18
    for filepath in directory.glob("*.json"):
        data = load_json(filepath)
        if not data:
            continue
        
        fold_mean, fold_std, ens_val = extract_bacc(data)
        
        if "vit_b_16" in filepath.name or "vit" in filepath.name:
            if fold_mean:
                vit_folds.append(fold_mean)
            if ens_val:
                vit_ens.append(ens_val)
        elif "resnet18" in filepath.name or "resnet18" in filepath.name:
            if fold_mean:
                r18_folds.append(fold_mean)
            if ens_val:
                r18_ens.append(ens_val)
    
    print(f"  ViT-B-16:       Fold mean: {np.mean(vit_folds):.4f} ± {np.std(vit_folds):.4f}  |  Ensemble: {np.mean(vit_ens):.4f} ± {np.std(vit_ens):.4f}")
    print(f"  ResNet18:       Fold mean: {np.mean(r18_folds):.4f} ± {np.std(r18_folds):.4f}  |  Ensemble: {np.mean(r18_ens):.4f} ± {np.std(r18_ens):.4f}")
    print(f"  ViT advantage:  Fold: {np.mean(vit_folds) - np.mean(r18_folds):+.4f}  |  Ensemble: {np.mean(vit_ens) - np.mean(r18_ens):+.4f}")

# ============================================================================
# 2b. ViT vs ResNet18 on POPULATION SHIFTS (AMOS22, HMU-CRC, DermaExt)
# ============================================================================

print_section("2b. ViT vs ResNet18 ON POPULATION SHIFTS")

for dataset_ps, dataset_name in [
    ("amos22", "AMOS22 (OrganaMNIST)"),
    ("hmu-crc", "HMU-CRC (PathMNIST)"),
    ("dermamnist-e-ood", "DermaExt (DermaMNIST-E)"),
]:
    print_subsection(f"{dataset_name}")
    
    vit_folds = []
    vit_ens = []
    r18_folds = []
    r18_ens = []
    
    # Load all files from PS folder for this dataset
    for filepath in PS_DIR.glob(f"*.json"):
        if dataset_ps not in filepath.name:
            continue
        
        data = load_json(filepath)
        if not data:
            continue
        
        fold_mean, fold_std, ens_val = extract_bacc(data)
        
        if "vit_b_16" in filepath.name or "vit" in filepath.name:
            if fold_mean:
                vit_folds.append(fold_mean)
            if ens_val:
                vit_ens.append(ens_val)
        elif "resnet18" in filepath.name:
            if fold_mean:
                r18_folds.append(fold_mean)
            if ens_val:
                r18_ens.append(ens_val)
    
    if vit_folds and r18_folds:
        print(f"  ViT-B-16:       Fold mean: {np.mean(vit_folds):.4f} ± {np.std(vit_folds):.4f}  |  Ensemble: {np.mean(vit_ens):.4f} ± {np.std(vit_ens):.4f}")
        print(f"  ResNet18:       Fold mean: {np.mean(r18_folds):.4f} ± {np.std(r18_folds):.4f}  |  Ensemble: {np.mean(r18_ens):.4f} ± {np.std(r18_ens):.4f}")
        print(f"  ViT advantage:  Fold: {np.mean(vit_folds) - np.mean(r18_folds):+.4f}  |  Ensemble: {np.mean(vit_ens) - np.mean(r18_ens):+.4f}")

# ============================================================================
# 3. DA vs Standard by shift
# ============================================================================

print_section("3. DA vs STANDARD BREAKDOWN BY SHIFT")

for shift_name, shift_key, directory in [
    ("In-Distribution", "id", ID_DIR),
    ("Corruption Shifts (s3)", "corruption", CS_DIR),
    ("Population Shifts", "ps", PS_DIR),
]:
    print_subsection(f"{shift_name}")
    
    da_folds = []
    da_ens = []
    std_folds = []
    std_ens = []
    
    for filepath in directory.glob("*.json"):
        data = load_json(filepath)
        if not data:
            continue
        
        fold_mean, fold_std, ens_val = extract_bacc(data)
        
        is_da = "_DA_" in filepath.name or "_DA.json" in filepath.name
        is_do = "_DO_" in filepath.name or "_DO.json" in filepath.name
        is_dado = "_DADO_" in filepath.name or "_DADO.json" in filepath.name
        is_standard = not (is_da or is_do or is_dado)
        
        if is_da and not is_dado:
            if fold_mean:
                da_folds.append(fold_mean)
            if ens_val:
                da_ens.append(ens_val)
        elif is_standard:
            if fold_mean:
                std_folds.append(fold_mean)
            if ens_val:
                std_ens.append(ens_val)
    
    print(f"  Standard:       Fold mean: {np.mean(std_folds):.4f} ± {np.std(std_folds):.4f}  |  Ensemble: {np.mean(std_ens):.4f} ± {np.std(std_ens):.4f}")
    print(f"  DA:             Fold mean: {np.mean(da_folds):.4f} ± {np.std(da_folds):.4f}  |  Ensemble: {np.mean(da_ens):.4f} ± {np.std(da_ens):.4f}")
    print(f"  DA improvement: Fold: {np.mean(da_folds) - np.mean(std_folds):+.4f}  |  Ensemble: {np.mean(da_ens) - np.mean(std_ens):+.4f}")

# ============================================================================
# 4. DO vs Standard by shift
# ============================================================================

print_section("4. DO vs STANDARD BREAKDOWN BY SHIFT")

for shift_name, shift_key, directory in [
    ("In-Distribution", "id", ID_DIR),
    ("Corruption Shifts (s3)", "corruption", CS_DIR),
    ("Population Shifts", "ps", PS_DIR),
]:
    print_subsection(f"{shift_name}")
    
    do_folds = []
    do_ens = []
    std_folds = []
    std_ens = []
    
    for filepath in directory.glob("*.json"):
        data = load_json(filepath)
        if not data:
            continue
        
        fold_mean, fold_std, ens_val = extract_bacc(data)
        
        is_do = "_DO_" in filepath.name or "_DO.json" in filepath.name
        is_da = "_DA_" in filepath.name or "_DA.json" in filepath.name
        is_dado = "_DADO_" in filepath.name or "_DADO.json" in filepath.name
        is_standard = not (is_da or is_do or is_dado)
        
        if is_do and not is_dado:
            if fold_mean:
                do_folds.append(fold_mean)
            if ens_val:
                do_ens.append(ens_val)
        elif is_standard:
            if fold_mean:
                std_folds.append(fold_mean)
            if ens_val:
                std_ens.append(ens_val)
    
    print(f"  Standard:       Fold mean: {np.mean(std_folds):.4f} ± {np.std(std_folds):.4f}  |  Ensemble: {np.mean(std_ens):.4f} ± {np.std(std_ens):.4f}")
    print(f"  DO:             Fold mean: {np.mean(do_folds):.4f} ± {np.std(do_folds):.4f}  |  Ensemble: {np.mean(do_ens):.4f} ± {np.std(do_ens):.4f}")
    print(f"  DO improvement: Fold: {np.mean(do_folds) - np.mean(std_folds):+.4f}  |  Ensemble: {np.mean(do_ens) - np.mean(std_ens):+.4f}")

# ============================================================================
# 5. Ensemble vs Mean Fold by shift
# ============================================================================

print_section("5. ENSEMBLE vs MEAN FOLD BREAKDOWN BY SHIFT")

for shift_name, shift_key, directory in [
    ("In-Distribution", "id", ID_DIR),
    ("Corruption Shifts (s3)", "corruption", CS_DIR),
    ("Population Shifts", "ps", PS_DIR),
]:
    print_subsection(f"{shift_name}")
    
    fold_means = []
    ens_vals = []
    
    for filepath in directory.glob("*.json"):
        data = load_json(filepath)
        if not data:
            continue
        
        fold_mean, fold_std, ens_val = extract_bacc(data)
        
        if fold_mean:
            fold_means.append(fold_mean)
        if ens_val:
            ens_vals.append(ens_val)
    
    ensemble_avg = np.mean(ens_vals)
    fold_avg = np.mean(fold_means)
    improvement = ensemble_avg - fold_avg
    
    print(f"  Mean fold bAcc:     {fold_avg:.4f} ± {np.std(fold_means):.4f}")
    print(f"  Ensemble bAcc:      {ensemble_avg:.4f} ± {np.std(ens_vals):.4f}")
    print(f"  Ensemble gain:      {improvement:+.4f} ({improvement/fold_avg*100:+.2f}%)")

print("\n")

if __name__ == "__main__":
    pass
