"""
Scatter plots: each UQ method's AUROC_f vs FID distance.

10 subplots (2×5 layout), one per method:
  MSR, MSR-S (MSR_calibrated), MLS, TTA, GPS,
  MCD (MCDropout), KNN (KNN_Raw), DE (Ensembling), Mean Agg, Mean Agg + Ens

Two figures:
  1. Absolute FID distance vs AUROC_f
  2. Normalized FID (ID = 1.0) vs AUROC_f

Points are colored by shift type (ID / Corruption / Population shift / New-class).
All shift types are included: in_distribution, corruption_shifts, population_shifts, new_class_shifts.

Usage
-----
    python method_auroc_vs_fid_scatter.py
    python method_auroc_vs_fid_scatter.py --metrics fid kid
    python method_auroc_vs_fid_scatter.py --metrics mahalanobis --fid-pkl /path/to/fid_results.pkl
    python method_auroc_vs_fid_scatter.py --fid-pkl /path/to/fid_results.pkl \
                                          --json-dir /path/to/jsons_results
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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
_DEFAULT_OUT = _HERE.parents[1] / "results" / "figures"

# ---------------------------------------------------------------------------
# Method names and display order
# ---------------------------------------------------------------------------
METHODS_TO_PLOT = [
    ("MSR",                          "MSR"),
    ("MSR_calibrated",               "MSR-S"),
    ("MLS",                          "MLS"),
    ("TTA",                          "TTA"),
    ("GPS",                          "GPS"),
    ("MCDropout",                    "MCD"),
    ("KNN_Raw",                      "KNN"),
    ("Ensembling",                   "DE"),
    ("ZScore_Aggregation_per_fold",  "Mean Agg"),
    ("ZScore_Aggregation_ensemble",  "Mean Agg + Ens"),
]

# Methods that use ensemble results only (no per_fold)
ENSEMBLE_ONLY = {"Ensembling", "ZScore_Aggregation_ensemble"}

# Population-shift dataset → FID key mapping
PS_TO_FID_DATASET = {
    "amos2022":              "organamnist",
    "hmu-crc":               "pathmnist",
    "dermamnist-e-external": "dermamnist-e-id",
}

# ---------------------------------------------------------------------------
# Aesthetics
# ---------------------------------------------------------------------------
SHIFT_STYLE = {
    "id":          dict(color="#4C72B0", label="ID"),
    "corruption":  dict(color="#DD8452", label="CS"),
    "pop_shift":   dict(color="#55A868", label="PS"),
    "new_class":   dict(color="#C44E52", label="NCS"),
}

DATASET_LABELS = {
    "organamnist":    "OrganA",
    "pneumoniamnist": "Pneumonia",
    "octmnist":       "OCT",
    "pathmnist":      "Path",
    "bloodmnist":     "Blood",
    "tissuemnist":    "Tissue",
    "breastmnist":    "Breast",
    "dermamnist-e-id": "Derma",
}

# ---------------------------------------------------------------------------
# Distance metric configurations
# ---------------------------------------------------------------------------
DISTANCE_METRICS = {
    "fid": dict(
        abs_key="fid",      norm_key="fid_norm",
        abs_xlabel="FID (*10$^{-2}$)",
        norm_xlabel="FID (norm, log)",
        label="FID",        file_prefix="fid",
    ),
    "kid": dict(
        abs_key="kid",      norm_key="kid_norm",
        abs_xlabel="KID distance (*10$^{-2}$)",
        norm_xlabel="KID (norm, log)",
        label="KID",        file_prefix="kid",
    ),
    "mahalanobis": dict(
        abs_key="maha",     norm_key="maha_norm",
        abs_xlabel="Mahalanobis distance (*10$^{-2}$)",
        norm_xlabel="Mahalanobis (norm, log)",
        label="Mahalanobis", file_prefix="mahalanobis",
    ),
}


# ---------------------------------------------------------------------------
# Data loading
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
    """
    Map JSON dataset name to FID dataset name.
    
    For ID and corruption: dataset as-is.
    For pop_shift: use PS_TO_FID_DATASET mapping.
    For new_class: map OOD dataset names back to training dataset.
    """
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


def collect_method_points(
    fid_results: dict,
    json_dir: Path,
) -> dict[str, list[dict]]:
    """
    Collect scatter points for each method.
    
    Returns dict:
      {
        "MSR": [
          {fid, fid_norm, auroc_f, shift, dataset, ...},
          ...
        ],
        ...
      }
    """
    points_by_method: dict[str, list[dict]] = {
        method: [] for method, _ in METHODS_TO_PLOT
    }

    json_files = _json_files_by_shift(json_dir)

    for shift_type, paths in json_files.items():
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
                fid_key = "id"
            elif shift_type == "corruption":
                fid_key = "random_s3"
            elif shift_type == "pop_shift":
                fid_key = "population_shift"
            elif shift_type == "new_class":
                fid_key = "new_classes"
            else:
                continue

            if fid_key not in fid_results[fid_dataset]:
                continue

            fid_entry = fid_results[fid_dataset][fid_key]
            fid_abs  = float(fid_entry["fid_distance"])
            fid_norm = float(fid_entry["normalized_fid"])
            kid_abs  = float(fid_entry.get("kid",                    np.nan))
            kid_norm = float(fid_entry.get("normalized_kid",         np.nan))
            maha_abs = float(fid_entry.get("mahalanobis",            np.nan))
            maha_norm= float(fid_entry.get("normalized_mahalanobis", np.nan))

            # Extract points for each method
            for method_key, _ in METHODS_TO_PLOT:
                if method_key not in d.get("methods", {}):
                    continue

                method_data = d["methods"][method_key]

                # Use per_fold if available, else ensemble
                if method_key in ENSEMBLE_ONLY:
                    auroc_values = [method_data.get("auroc_f")]
                elif "per_fold_metrics" in method_data:
                    auroc_values = [
                        m.get("auroc_f")
                        for m in method_data["per_fold_metrics"]
                        if m.get("auroc_f") is not None
                    ]
                else:
                    auroc_values = [method_data.get("auroc_f")]

                for auroc_f in auroc_values:
                    if auroc_f is None:
                        continue
                    points_by_method[method_key].append(
                        dict(
                            fid=fid_abs,   fid_norm=fid_norm,
                            kid=kid_abs,   kid_norm=kid_norm,
                            maha=maha_abs, maha_norm=maha_norm,
                            auroc_f=auroc_f,
                            shift=shift_type,
                            dataset=dataset,
                        )
                    )

    return points_by_method


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _method_subplot(
    ax: plt.Axes,
    points: list[dict],
    method_name: str,
    x_key: str = "fid",
    xlabel: str = "FID",
    xlog: bool = False,    show_xlabel: bool = True,
    show_ylabel: bool = True,) -> None:
    """Draw one method's scatter subplot with Pearson correlation and linear regression."""
    if not points:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="0.5")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[list(ax.spines.keys())].set_visible(False)
        return

    # Drop NaN x-values (metric may not be in pickle yet)
    points = [p for p in points if not np.isnan(p.get(x_key, np.nan))]
    if not points:
        ax.text(0.5, 0.5, "No data\n(metric missing)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.5")
        return

    # Collect all x, y values for correlation and regression
    all_xs = []
    all_ys = []

    for shift_type, style in SHIFT_STYLE.items():
        pts = [p for p in points if p["shift"] == shift_type]
        if not pts:
            continue
        # Scale absolute distance metrics by 100 for more reasonable r values
        xs = [p[x_key] / 100 if x_key in ["fid", "kid", "maha"] else p[x_key] for p in pts]
        ys = [p["auroc_f"] for p in pts]
        all_xs.extend(xs)
        all_ys.extend(ys)
        ax.scatter(xs, ys, s=25, alpha=0.6, color=style["color"],
                   label=style["label"], edgecolors="white", linewidths=0.4)

    # Calculate Pearson correlation
    if len(all_xs) > 1:
        r, p_val = stats.pearsonr(all_xs, all_ys)

        # Fit linear regression
        if xlog:
            log_xs = np.log10(all_xs)
            coeffs = np.polyfit(log_xs, all_ys, 1)
        else:
            coeffs = np.polyfit(all_xs, all_ys, 1)
        slope, intercept = coeffs[0], coeffs[1]

        ax._pearson_r = r
        ax._p_val = p_val
        ax._slope = slope
        ax._intercept = intercept
        ax._xlog = xlog

    if xlog:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))

    ax.set_xlabel(xlabel if show_xlabel else "", fontsize=13, fontweight="bold")
    ax.set_ylabel("AUROC_f" if show_ylabel else "", fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(0.3, 1)

    # Plot regression line after scale is set
    if len(all_xs) > 1:
        x_range = np.array(ax.get_xlim())
        if ax._xlog:
            y_fit = ax._slope * np.log10(x_range) + ax._intercept
        else:
            y_fit = ax._slope * x_range + ax._intercept
        ax.plot(x_range, y_fit, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)

        text_str = f"r = {ax._pearson_r:.2f}\ny = {ax._slope:.2f}x + {ax._intercept:.2f}"
        ax.text(0.4, 0.2, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    ax.set_title(method_name, fontsize=13, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def plot(
    fid_results: dict,
    json_dir: Path,
    output_dir: Path,
    metrics: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    points_by_method = collect_method_points(fid_results, json_dir)

    n_methods = len(METHODS_TO_PLOT)
    n_rows = 2
    n_cols = 5
    assert n_methods == n_rows * n_cols, f"Expected {n_rows*n_cols} methods, got {n_methods}"

    # Filter metrics if specified
    metrics_to_plot = {k: v for k, v in DISTANCE_METRICS.items() if metrics is None or k in metrics}

    for metric_key, metric in metrics_to_plot.items():
        for use_norm, x_key, xlabel, suffix in [
            (False, metric["abs_key"],  metric["abs_xlabel"],  "_absolute"),
            (True,  metric["norm_key"], metric["norm_xlabel"], "_normalized"),
        ]:
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(16, 8),
                constrained_layout=True,
            )
            axes = axes.flatten()

            for idx, (method_key, method_label) in enumerate(METHODS_TO_PLOT):
                _method_subplot(
                    axes[idx],
                    points_by_method[method_key],
                    method_label,
                    x_key=x_key,
                    xlabel=xlabel,
                    xlog=use_norm,
                    show_xlabel=(idx >= n_cols),
                    show_ylabel=(idx % n_cols == 0),
                )

            handles = [
                plt.scatter([], [], s=40, color=s["color"], label=s["label"],
                           edgecolors="white", linewidths=0.5)
                for s in SHIFT_STYLE.values()
            ]
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=4,
                fontsize=13,
                framealpha=0.9,
                bbox_to_anchor=(0.5, -0.05),
            )

            title_norm = "Normalized (log scale)" if use_norm else "Absolute"
            fig.suptitle(
                f"AUROC_f vs {metric['label']}",
                fontsize=16, fontweight="bold",
            )

            for ext in ("pdf", "png"):
                out = output_dir / f"method_auroc_vs_{metric['file_prefix']}{suffix}.{ext}"
                fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
                print(f"Saved → {out}")
            plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fid-pkl",  type=Path, default=_DEFAULT_FID_PKL)
    p.add_argument("--json-dir", type=Path, default=_DEFAULT_JSON_DIR)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--metrics",  nargs="+", choices=list(DISTANCE_METRICS.keys()),
                   help="Distance metrics to plot (default: all). Choose from: fid, kid, mahalanobis")
    args = p.parse_args()

    with open(args.fid_pkl, "rb") as f:
        fid_results = pickle.load(f)

    plot(fid_results, args.json_dir, args.output_dir, metrics=args.metrics)


if __name__ == "__main__":
    main()
