"""
Multiple regression analysis: AUROC_f ~ FID + balanced_accuracy for each UQ method.

For each of the 10 methods, fit:
    AUROC_f = β₀ + β₁*FID + β₂*bAcc + ε

Reports both raw and standardized (β*) coefficients.
Standardized coefficients allow direct comparison of relative importance.

Data is filtered to exclude new-class shift (only ID, corruption, population shift).

Output:
    - Regression coefficients (raw and standardized) for each method
    - R² and adjusted R² goodness-of-fit
    - VIF (multicollinearity check)
    - Summary table of all methods
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DEFAULT_FID_PKL = (
    _HERE.parents[1] / "results" / "dinov3_embeddings" / "fid_results.pkl"
)
_DEFAULT_JSON_DIR = (
    _HERE.parents[1] / "results" / "jsons_results"
)

# ---------------------------------------------------------------------------
# Method names and display order
# ---------------------------------------------------------------------------
METHODS_TO_PLOT = [
    ("MSR",                          "MSR"),
    ("MSR_calibrated",               "MSR-S"),
    ("MLS",                          "MLS"),
    ("Ensembling",                   "DE"),
    ("GPS",                          "GPS"),
    ("TTA",                          "TTA"),
    ("MCDropout",                    "MCD"),
    ("KNN_Raw",                      "KNN"),
    ("ZScore_Aggregation_per_fold",  "Agg(fold)"),
    ("ZScore_Aggregation_ensemble",  "Agg(ens)"),
]

# Methods that use ensemble results only (no per_fold)
ENSEMBLE_ONLY = {"Ensembling", "ZScore_Aggregation_ensemble"}

# Population-shift dataset → FID key mapping
PS_TO_FID_DATASET = {
    "amos2022":              "organamnist",
    "hmu-crc":               "pathmnist",
    "dermamnist-e-external": "dermamnist-e-id",
}

# Dataset mapping for bAcc lookup
CLS_TO_FID_DATASET = {
    "dermamnist-e": "dermamnist-e-id",
}


# ---------------------------------------------------------------------------
# Data loading (replicate from scatter plot scripts)
# ---------------------------------------------------------------------------

def _json_files_by_shift(json_dir: Path) -> dict[str, list[Path]]:
    """Group JSON files by shift type."""
    groups = {
        "id":         sorted((json_dir / "in_distribution").glob("*.json")),
        "corruption": sorted((json_dir / "corruption_shifts").glob("*.json")),
        "pop_shift":  sorted((json_dir / "population_shifts").glob("*.json")),
        "new_class":  sorted((json_dir / "new_class_shifts").glob("*.json")),
    }
    return groups


def _map_json_to_fid_dataset(shift_type: str, dataset: str) -> Optional[str]:
    """Map JSON dataset name to FID dataset name."""
    if shift_type in ("id", "corruption"):
        return dataset
    elif shift_type == "pop_shift":
        return PS_TO_FID_DATASET.get(dataset)
    elif shift_type == "new_class":
        # Map new-class dataset names to training dataset
        new_class_to_train = {
            "amos2022": "organamnist",
            "midog": "pathmnist",
        }
        return new_class_to_train.get(dataset)
    return None


def collect_method_points_with_bacc(
    fid_results: dict,
    json_dir: Path,
    cls_dir: Path,
    fid_key: str = "fid",  # "fid" for raw, "fid_norm" for normalized
    debug: bool = False,
) -> dict[str, list[dict]]:
    """
    Collect scatter points for each method, INCLUDING balanced accuracy.
    
    Returns dict:
      {
        "MSR": [
          {fid, auroc_f, bacc, shift, dataset, model, ...},
          ...
        ],
        ...
      }
    
    Data is filtered to exclude new_class shift.
    fid_key: "fid" for raw FID distance, "fid_norm" for normalized FID
    debug: if True, print matched bAcc values
    """
    points_by_method: dict[str, list[dict]] = {
        method: [] for method, _ in METHODS_TO_PLOT
    }
    
    bacc_matches = 0
    bacc_misses = 0

    json_files = _json_files_by_shift(json_dir)

    for shift_type, paths in json_files.items():
        # Skip new-class shift
        if shift_type == "new_class":
            continue

        for path in paths:
            with open(path) as f:
                d = json.load(f)

            dataset = d["flag"]
            fid_dataset = _map_json_to_fid_dataset(shift_type, dataset)
            if fid_dataset is None:
                continue
            if fid_dataset not in fid_results:
                continue

            # Determine FID key
            if shift_type == "id":
                fid_shift_key = "id"
            elif shift_type == "corruption":
                fid_shift_key = "random_s3"
            elif shift_type == "pop_shift":
                fid_shift_key = "population_shift"
            else:
                continue

            if fid_shift_key not in fid_results[fid_dataset]:
                continue

            fid_entry = fid_results[fid_dataset][fid_shift_key]
            fid_abs = float(fid_entry.get("fid_distance", np.nan))
            fid_normalized = float(fid_entry.get("normalized_fid", np.nan))

            # Load balanced accuracy from classification results
            bacc_ensemble, bacc_per_fold = _load_bacc_for_dataset(cls_dir, dataset, shift_type, d.get("model_backbone", ""), d.get("setup", ""))
            if bacc_ensemble is None:
                bacc_misses += 1
                if debug:
                    print(f"  ⚠ No bAcc for {dataset} ({shift_type})")
                continue

            bacc_matches += 1
            if debug:
                print(f"  ✓ bAcc={bacc_ensemble:.4f} for {dataset} ({shift_type})")

            fid_val = fid_normalized if fid_key == "fid_norm" else fid_abs

            # Extract points for each method
            for method_key, _ in METHODS_TO_PLOT:
                if method_key not in d.get("methods", {}):
                    continue

                method_data = d["methods"][method_key]

                if method_key in ENSEMBLE_ONLY:
                    # One point: ensemble auroc_f + ensemble bAcc
                    auroc_f = method_data.get("auroc_f")
                    if auroc_f is not None:
                        points_by_method[method_key].append(
                            dict(fid=fid_val, auroc_f=auroc_f, bacc=bacc_ensemble, shift=shift_type, dataset=dataset)
                        )
                elif "per_fold_metrics" in method_data:
                    # Per-fold: pair fold_auroc[i] with fold_bacc[i]
                    fold_aurocs = [m.get("auroc_f") for m in method_data["per_fold_metrics"]]
                    fold_baccs = bacc_per_fold if bacc_per_fold and len(bacc_per_fold) == len(fold_aurocs) else [bacc_ensemble] * len(fold_aurocs)
                    for auroc_f, bacc_f in zip(fold_aurocs, fold_baccs):
                        if auroc_f is not None:
                            points_by_method[method_key].append(
                                dict(fid=fid_val, auroc_f=auroc_f, bacc=bacc_f, shift=shift_type, dataset=dataset)
                            )
                else:
                    # Fallback: ensemble auroc_f + ensemble bAcc
                    auroc_f = method_data.get("auroc_f")
                    if auroc_f is not None:
                        points_by_method[method_key].append(
                            dict(fid=fid_val, auroc_f=auroc_f, bacc=bacc_ensemble, shift=shift_type, dataset=dataset)
                        )
    
    if debug:
        print(f"\nData collection summary: {bacc_matches} matched, {bacc_misses} missed")
        print(f"Points by method:")
        for method_key, method_label in METHODS_TO_PLOT:
            print(f"  {method_label:15}: {len(points_by_method[method_key])} points")

    return points_by_method


def _load_bacc_for_dataset(
    cls_dir: Path,
    dataset: str,
    shift_type: str,
    model_backbone: str = "",
    setup: str = "",
) -> tuple[Optional[float], Optional[list[float]]]:
    """
    Load balanced accuracy for a given dataset and shift type.
    Returns (ensemble_bacc, per_fold_bacc_list).
    per_fold_bacc_list is a list of 5 fold-level bAcc values (or None).

    Uses model_backbone + setup to find the exact matching classification file.
    Handles dataset name mappings:
      amos2022 → amos22
      dermamnist-e-external → dermamnist-e-ood
    """
    pop_shift_name_map = {
        "amos2022": "amos22",
        "dermamnist-e-external": "dermamnist-e-ood",
    }
    # setup="" means standard config
    config = setup if setup else "standard"

    def _read(path):
        with open(path) as f:
            return json.load(f)

    def _parse_id_pop(d):
        ensemble = float(d["ensemble_metrics"]["balanced_accuracy"])
        per_fold = [float(x["balanced_accuracy"]) for x in d.get("per_fold_metrics", []) if "balanced_accuracy" in x] or None
        return ensemble, per_fold

    def _parse_corruption(d):
        ensemble = float(d["ensemble"]["balanced_accuracy"])
        per_fold = [float(x["balanced_accuracy"]) for x in d.get("per_fold", []) if "balanced_accuracy" in x] or None
        return ensemble, per_fold

    if shift_type == "id":
        path = cls_dir / "in_distribution" / f"comprehensive_metrics_{dataset}_{model_backbone}_{config}.json"
        if path.exists():
            d = _read(path)
            if "ensemble_metrics" in d:
                return _parse_id_pop(d)

    elif shift_type == "corruption":
        if setup:
            path = cls_dir / "corruption_shifts" / f"{dataset}_{model_backbone}_{setup}_severity3.json"
        else:
            path = cls_dir / "corruption_shifts" / f"{dataset}_{model_backbone}_severity3.json"
        if path.exists():
            d = _read(path)
            if "ensemble" in d:
                return _parse_corruption(d)

    elif shift_type == "pop_shift":
        search_dataset = pop_shift_name_map.get(dataset, dataset)
        path = cls_dir / "population_shift" / f"comprehensive_metrics_{search_dataset}_{model_backbone}_{config}.json"
        if path.exists():
            d = _read(path)
            if "ensemble_metrics" in d:
                return _parse_id_pop(d)

    return None, None


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def fit_multivar_regression(
    points: list[dict],
) -> dict:
    """
    Fit: AUROC_f = β₀ + β₁*FID + β₂*bAcc (both raw and standardized)
    
    Returns:
        {
            'intercept': β₀ (raw),
            'fid_coeff': β₁ (raw),
            'bacc_coeff': β₂ (raw),
            'fid_coeff_std': β₁ (standardized),
            'bacc_coeff_std': β₂ (standardized),
            'r_squared': R²,
            'adjusted_r_squared': adj R²,
            'n': number of points,
            'fid_pval': p-value for FID coefficient,
            'bacc_pval': p-value for bAcc coefficient,
            'vif_fid': Variance Inflation Factor for FID,
            'vif_bacc': Variance Inflation Factor for bAcc,
        }
    """
    if not points or len(points) < 3:
        return None

    # Drop NaN values
    points = [p for p in points if all(
        not np.isnan(p.get(k, np.nan))
        for k in ['fid', 'bacc', 'auroc_f']
    )]

    if len(points) < 3:
        return None

    X = np.array([[p['fid'], p['bacc']] for p in points], dtype=np.float64)
    y = np.array([p['auroc_f'] for p in points], dtype=np.float64)

    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # Fit regression using lstsq (raw coefficients)
    coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    intercept, fid_coeff, bacc_coeff = coeffs

    # Compute R²
    y_pred = X_with_intercept @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Compute R² for FID only
    X_fid_only = np.column_stack([np.ones(len(X)), X[:, 0]])
    coeffs_fid, _, _, _ = np.linalg.lstsq(X_fid_only, y, rcond=None)
    y_pred_fid = X_fid_only @ coeffs_fid
    ss_res_fid = np.sum((y - y_pred_fid) ** 2)
    r2_fid_only = 1 - (ss_res_fid / ss_tot) if ss_tot > 0 else 0

    # Compute R² for bAcc only
    X_bacc_only = np.column_stack([np.ones(len(X)), X[:, 1]])
    coeffs_bacc, _, _, _ = np.linalg.lstsq(X_bacc_only, y, rcond=None)
    y_pred_bacc = X_bacc_only @ coeffs_bacc
    ss_res_bacc = np.sum((y - y_pred_bacc) ** 2)
    r2_bacc_only = 1 - (ss_res_bacc / ss_tot) if ss_tot > 0 else 0

    # Compute incremental R² (unique contributions)
    delta_r2_fid = r_squared - r2_bacc_only   # FID contribution when bAcc already in model
    delta_r2_bacc = r_squared - r2_fid_only   # bAcc contribution when FID already in model

    # Compute adjusted R²
    n = len(y)
    p = 2  # Number of predictors (FID, bAcc)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared

    # Compute standard errors and p-values
    mse = ss_res / (n - p - 1) if n > p + 1 else 0
    var_covar = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * mse if mse > 0 else np.zeros((3, 3))
    se = np.sqrt(np.diag(var_covar))

    fid_t_stat = fid_coeff / se[1] if se[1] > 0 else 0
    bacc_t_stat = bacc_coeff / se[2] if se[2] > 0 else 0
    fid_pval = 2 * (1 - stats.t.cdf(abs(fid_t_stat), n - p - 1)) if n > p + 1 else np.nan
    bacc_pval = 2 * (1 - stats.t.cdf(abs(bacc_t_stat), n - p - 1)) if n > p + 1 else np.nan

    # Standardize X and y for standardized coefficients
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0, ddof=1)
    y_mean = np.mean(y)
    y_std = np.std(y, ddof=1)
    
    X_std_scaled = (X - X_mean) / X_std
    y_std_scaled = (y - y_mean) / y_std
    
    X_std_with_intercept = np.column_stack([np.ones(len(X_std_scaled)), X_std_scaled])
    coeffs_std, _, _, _ = np.linalg.lstsq(X_std_with_intercept, y_std_scaled, rcond=None)
    fid_coeff_std, bacc_coeff_std = coeffs_std[1], coeffs_std[2]

    # Compute VIF (Variance Inflation Factor)
    # VIF_j = 1 / (1 - R²_j) where R²_j is from regressing predictor j on all others
    fid_col = X[:, 0]
    bacc_col = X[:, 1]
    
    # Regress FID ~ bAcc
    fid_X = np.column_stack([np.ones(len(bacc_col)), bacc_col])
    fid_coeffs, _, _, _ = np.linalg.lstsq(fid_X, fid_col, rcond=None)
    fid_pred = fid_X @ fid_coeffs
    fid_ss_res = np.sum((fid_col - fid_pred) ** 2)
    fid_ss_tot = np.sum((fid_col - np.mean(fid_col)) ** 2)
    fid_r_squared = 1 - (fid_ss_res / fid_ss_tot) if fid_ss_tot > 0 else 0
    vif_fid = 1 / (1 - fid_r_squared) if fid_r_squared < 1 else np.inf

    # Regress bAcc ~ FID
    bacc_X = np.column_stack([np.ones(len(fid_col)), fid_col])
    bacc_coeffs, _, _, _ = np.linalg.lstsq(bacc_X, bacc_col, rcond=None)
    bacc_pred = bacc_X @ bacc_coeffs
    bacc_ss_res = np.sum((bacc_col - bacc_pred) ** 2)
    bacc_ss_tot = np.sum((bacc_col - np.mean(bacc_col)) ** 2)
    bacc_r_squared = 1 - (bacc_ss_res / bacc_ss_tot) if bacc_ss_tot > 0 else 0
    vif_bacc = 1 / (1 - bacc_r_squared) if bacc_r_squared < 1 else np.inf

    return {
        'intercept': intercept,
        'fid_coeff': fid_coeff,
        'bacc_coeff': bacc_coeff,
        'fid_coeff_std': fid_coeff_std,
        'bacc_coeff_std': bacc_coeff_std,
        'r_squared': r_squared,
        'r2_fid_only': r2_fid_only,
        'r2_bacc_only': r2_bacc_only,
        'delta_r2_fid': delta_r2_fid,
        'delta_r2_bacc': delta_r2_bacc,
        'adjusted_r_squared': adj_r_squared,
        'n': n,
        'fid_pval': fid_pval,
        'bacc_pval': bacc_pval,
        'vif_fid': vif_fid,
        'vif_bacc': vif_bacc,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fid-pkl",  type=Path, default=_DEFAULT_FID_PKL)
    p.add_argument("--json-dir", type=Path, default=_DEFAULT_JSON_DIR)
    p.add_argument("--cls-dir",  type=Path, default=_HERE.parents[1] / "results" / "classification_results")
    args = p.parse_args()

    with open(args.fid_pkl, "rb") as f:
        fid_results = pickle.load(f)

    # Run regression for both raw and normalized FID
    for fid_type, fid_key in [("Raw FID", "fid"), ("Normalized FID", "fid_norm")]:
        print("\n" + "=" * 90)
        print(f"Multiple Regression: AUROC_f ~ {fid_type} + balanced_accuracy")
        print("(New-class shift data excluded)")
        print("=" * 90)

        points_by_method = collect_method_points_with_bacc(fid_results, args.json_dir, args.cls_dir, fid_key=fid_key, debug=False)

        results_list = []

        for method_key, method_label in METHODS_TO_PLOT:
            points = points_by_method[method_key]
            if not points:
                print(f"\n{method_label:15} — No data")
                continue

            reg = fit_multivar_regression(points)
            if reg is None:
                print(f"\n{method_label:15} — Insufficient data (< 3 points)")
                continue
            
            # Compute Pearson correlations for sanity check
            auroc_vals = np.array([p['auroc_f'] for p in points])
            fid_vals = np.array([p['fid'] for p in points])
            bacc_vals = np.array([p['bacc'] for p in points])
            
            r_auroc_fid = np.corrcoef(auroc_vals, fid_vals)[0, 1]
            r_auroc_bacc = np.corrcoef(auroc_vals, bacc_vals)[0, 1]
            r_fid_bacc = np.corrcoef(fid_vals, bacc_vals)[0, 1]

            print(f"\n{method_label:15}")
            print(f"  Intercept:         {reg['intercept']:8.4f}")
            print(f"  {fid_type} coeff (raw):   {reg['fid_coeff']:8.4f}  (std: {reg['fid_coeff_std']:8.4f}, p={reg['fid_pval']:.2e}, VIF={reg['vif_fid']:.4f})")
            print(f"  bAcc coeff (raw):  {reg['bacc_coeff']:8.4f}  (std: {reg['bacc_coeff_std']:8.4f}, p={reg['bacc_pval']:.2e}, VIF={reg['vif_bacc']:.4f})")
            print(f"  R²:                {reg['r_squared']:8.4f}")
            print(f"  Adj. R²:           {reg['adjusted_r_squared']:8.4f}")
            print(f"  N:                 {reg['n']:8d}")
            print(f"  ")
            print(f"  [Pearson correlation sanity check]")
            print(f"    AUROC vs {fid_type}:  r = {r_auroc_fid:8.4f}  (r² ≈ {r_auroc_fid**2:.4f})")
            print(f"    AUROC vs bAcc:     r = {r_auroc_bacc:8.4f}  (r² ≈ {r_auroc_bacc**2:.4f})")
            print(f"    {fid_type} vs bAcc:     r = {r_fid_bacc:8.4f}  (collinearity check)")

            results_list.append({
                'Method': method_label,
                'Intercept': reg['intercept'],
                'FID coeff': reg['fid_coeff'],
                'FID coeff (std)': reg['fid_coeff_std'],
                'FID p-val': reg['fid_pval'],
                'FID VIF': reg['vif_fid'],
                'bAcc coeff': reg['bacc_coeff'],
                'bAcc coeff (std)': reg['bacc_coeff_std'],
                'bAcc p-val': reg['bacc_pval'],
                'bAcc VIF': reg['vif_bacc'],
                'R² (both)': reg['r_squared'],
                'R² (FID only)': reg['r2_fid_only'],
                'R² (bAcc only)': reg['r2_bacc_only'],
                'ΔR² (FID)': reg['delta_r2_fid'],
                'ΔR² (bAcc)': reg['delta_r2_bacc'],
                'Coeff ratio (|FID|/bAcc)': abs(reg['fid_coeff_std']) / abs(reg['bacc_coeff_std']) if reg['bacc_coeff_std'] != 0 else np.nan,
                'Adj. R²': reg['adjusted_r_squared'],
                'N': reg['n'],
                'r(AUROC,FID)': r_auroc_fid,
                'r(AUROC,bAcc)': r_auroc_bacc,
                'r(FID,bAcc)': r_fid_bacc,
            })

        # Summary table
        print("\n" + "=" * 90)
        print("SUMMARY TABLE")
        print("=" * 90)

        df = pd.DataFrame(results_list)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}' if not pd.isna(x) else 'nan'))

        # Save summary to CSV
        fid_suffix = "raw_fid" if fid_key == "fid" else "normalized_fid"
        output_csv = _HERE.parents[1] / "results" / f"multivar_regression_summary_{fid_suffix}.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, float_format='%.6f')
        print(f"\nSummary saved to {output_csv}")


if __name__ == "__main__":
    main()
