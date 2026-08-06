"""
Scatter plots: DINOv3-FID distance vs balanced accuracy.

Two panels:
  1. Absolute FID distance  vs  balanced_accuracy
  2. Normalized FID         vs  balanced_accuracy   (ID test = 1 per dataset)

Markers:  x = individual CV fold,  o = ensemble
Colour  : shift type  (ID / Corruption / Population shift)

Usage
-----
    python fid_vs_bacc_scatter.py
    python fid_vs_bacc_scatter.py --metrics fid kid
    python fid_vs_bacc_scatter.py --results-dir /path/to/classification_results \
                                   --fid-pkl /path/to/fid_results.pkl \
                                   --metrics mahalanobis
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
_DEFAULT_CLS_DIR  = _HERE.parents[1] / "results" / "classification_results"
_DEFAULT_FID_PKL  = _HERE.parents[1] / "results" / "dinov3_embeddings" / "fid_results.pkl"
_DEFAULT_OUT      = _HERE.parents[1] / "results" / "figures"

# ---------------------------------------------------------------------------
# Population-shift dataset → FID key mapping
# (classification results use the shift dataset name; FID is stored under the
#  training dataset that the models were trained on)
# ---------------------------------------------------------------------------
PS_TO_FID_DATASET = {
    "amos22":           "organamnist",
    "hmu-crc":          "pathmnist",
    "dermamnist-e-ood": "dermamnist-e-id",
}
# NOTE: new-class shift datasets (amos22_new_classes, midog) are intentionally
# excluded from these plots.  They have FID entries under the 'new_classes'
# sub-key in fid_results.pkl, but no paired classification results exist
# (models are not evaluated on OOD classes they were never trained on in a
# balanced-accuracy sense).  Any unknown ps_dataset not in PS_TO_FID_DATASET
# is skipped with a warning in _bacc_pop_shift().

# ---------------------------------------------------------------------------
# Aesthetics
# ---------------------------------------------------------------------------
SHIFT_STYLE = {
    "id":         dict(color="#4C72B0", label="ID"),
    "corruption": dict(color="#DD8452", label="CS"),
    "pop_shift":  dict(color="#55A868", label="PS")
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

# One colour per dataset for annotation
_DS_COLORS = plt.cm.tab10(np.linspace(0, 1, len(DATASET_LABELS)))
DS_COLOR = dict(zip(DATASET_LABELS.keys(), _DS_COLORS))

# ---------------------------------------------------------------------------
# Distance metric configurations
# ---------------------------------------------------------------------------
DISTANCE_METRICS = {
    "fid": dict(
        abs_key="fid",      norm_key="fid_norm",
        abs_xlabel="FID to training set (absolute)",
        norm_xlabel="Normalized FID  (ID test = 1.0, log scale)",
        label="FID",        file_prefix="fid",
    ),
    "kid": dict(
        abs_key="kid",      norm_key="kid_norm",
        abs_xlabel="KID distance to training set (absolute)",
        norm_xlabel="Normalized KID  (ID test = 1.0, log scale)",
        label="KID",        file_prefix="kid",
    ),
    "mahalanobis": dict(
        abs_key="maha",     norm_key="maha_norm",
        abs_xlabel="Mahalanobis distance to training distribution (absolute)",
        norm_xlabel="Normalized Mahalanobis  (ID test = 1.0, log scale)",
        label="Mahalanobis", file_prefix="mahalanobis",
    ),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _bacc_id(cls_dir: Path) -> list[dict]:
    """
    Returns list of records from in_distribution/:
      {dataset, model, setup, fid_key='id', folds=[bacc,...], ensemble_bacc}
    """
    records = []
    for path in sorted((cls_dir / "in_distribution").glob("comprehensive_metrics_*.json")):
        with open(path) as f:
            d = json.load(f)
        dataset = d["dataset"]
        records.append(dict(
            dataset=dataset,
            model=d["model"],
            setup=d["setup"],
            fid_key="id",
            shift="id",
            folds=[m["balanced_accuracy"] for m in d["per_fold_metrics"]],
            ensemble=d["ensemble_metrics"]["balanced_accuracy"],
        ))
    return records


def _bacc_corruption(cls_dir: Path) -> list[dict]:
    """Returns records from corruption_shifts/ (severity3 only)."""
    records = []
    # Map classification dataset names to FID dataset names for corruption shifts
    cs_to_fid_dataset = {"dermamnist-e": "dermamnist-e-id"}
    
    for path in sorted((cls_dir / "corruption_shifts").glob("*_severity3.json")):
        with open(path) as f:
            d = json.load(f)
        meta = d["metadata"]
        cls_dataset = meta["dataset"]
        # Map to FID dataset name
        fid_dataset = cs_to_fid_dataset.get(cls_dataset, cls_dataset)
        
        records.append(dict(
            dataset=fid_dataset,        # FID dataset name
            model=meta["model"],
            setup=meta["setup"],
            fid_key="random_s3",
            shift="corruption",
            folds=[m["balanced_accuracy"] for m in d["per_fold"]],
            ensemble=d["ensemble"]["balanced_accuracy"],
        ))
    return records


def _bacc_pop_shift(cls_dir: Path) -> list[dict]:
    """Returns records from population_shift/, remapping dataset → FID dataset."""
    records = []
    for path in sorted((cls_dir / "population_shift").glob("comprehensive_metrics_*.json")):
        with open(path) as f:
            d = json.load(f)
        ps_dataset = d["dataset"]
        fid_dataset = PS_TO_FID_DATASET.get(ps_dataset)
        if fid_dataset is None:
            print(f"  [skip] population_shift/{ps_dataset}: no FID mapping (new-class shift or unknown)")
            continue
        records.append(dict(
            dataset=fid_dataset,       # FID is stored under the training dataset
            ps_dataset=ps_dataset,
            model=d["model"],
            setup=d["setup"],
            fid_key="population_shift",
            shift="pop_shift",
            folds=[m["balanced_accuracy"] for m in d["per_fold_metrics"]],
            ensemble=d["ensemble_metrics"]["balanced_accuracy"],
        ))
    return records


def collect_points(
    fid_results: dict,
    cls_dir: Path,
) -> list[dict]:
    """
    Build a flat list of scatter points.
    Each entry: {fid, fid_norm, kid, kid_norm, maha, maha_norm, bacc, shift, dataset, model, is_ensemble}
    """
    all_records = (
        _bacc_id(cls_dir)
        + _bacc_corruption(cls_dir)
        + _bacc_pop_shift(cls_dir)
    )

    pts: list[dict] = []

    for rec in all_records:
        ds = rec["dataset"]
        fid_key = rec["fid_key"]
        if ds not in fid_results or fid_key not in fid_results[ds]:
            continue
        entry = fid_results[ds][fid_key]
        dists = dict(
            fid      = float(entry.get("fid_distance",             np.nan)),
            fid_norm = float(entry.get("normalized_fid",           np.nan)),
            kid      = float(entry.get("kid",                      np.nan)),
            kid_norm = float(entry.get("normalized_kid",           np.nan)),
            maha     = float(entry.get("mahalanobis",              np.nan)),
            maha_norm= float(entry.get("normalized_mahalanobis",   np.nan)),
        )
        common = dict(shift=rec["shift"], dataset=ds, model=rec["model"])

        for bacc in rec["folds"]:
            pts.append(dict(**common, **dists, bacc=bacc, is_ensemble=False))
        pts.append(dict(**common, **dists, bacc=rec["ensemble"], is_ensemble=True))

    return pts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scatter_panel(
    ax: plt.Axes,
    points: list[dict],
    xlabel: str,
    title: str,
    x_key: str = "fid",
    xlog: bool = False,
    model_filter: str | None = None,
) -> None:
    """Draw one scatter panel with correlation and linear regression.
    If model_filter is set, only those points are drawn."""
    if model_filter is not None:
        points = [p for p in points if p["model"] == model_filter]
    # Drop NaN x-values (metric may not be in pickle yet)
    points = [p for p in points if not np.isnan(p.get(x_key, np.nan))]

    # Collect all points for correlation/regression
    all_xs = []
    all_ys = []

    for shift, style in SHIFT_STYLE.items():
        # folds
        xs = [p[x_key] for p in points if p["shift"] == shift and not p["is_ensemble"]]
        ys = [p["bacc"] for p in points if p["shift"] == shift and not p["is_ensemble"]]
        if xs:
            ax.scatter(xs, ys, marker="x", s=30, linewidths=1.2,
                       color=style["color"], alpha=0.55, zorder=3)
            all_xs.extend(xs)
            all_ys.extend(ys)
        # ensemble
        xe = [p[x_key] for p in points if p["shift"] == shift and p["is_ensemble"]]
        ye = [p["bacc"] for p in points if p["shift"] == shift and p["is_ensemble"]]
        if xe:
            ax.scatter(xe, ye, marker="o", s=55, linewidths=1.0,
                       color=style["color"], edgecolors="white", alpha=0.9,
                       zorder=4, label=style["label"])
            all_xs.extend(xe)
            all_ys.extend(ye)

    # Calculate Pearson correlation
    if len(all_xs) > 1:
        r, p_val = stats.pearsonr(all_xs, all_ys)
        
        # Fit linear regression
        if xlog:
            # Log-linear regression: fit log(x) vs y
            log_xs = np.log10(all_xs)
            coeffs = np.polyfit(log_xs, all_ys, 1)
            slope, intercept = coeffs[0], coeffs[1]
        else:
            # Linear regression: fit x vs y
            coeffs = np.polyfit(all_xs, all_ys, 1)
            slope, intercept = coeffs[0], coeffs[1]
        
        # Store for plotting after scale is set
        ax._pearson_r = r
        ax._p_val = p_val
        ax._slope = slope
        ax._intercept = intercept
        ax._xlog = xlog

    # Dataset text annotations on ensemble points only (avoid clutter)
    for p in points:
        if not p["is_ensemble"]:
            continue
        ds = p["dataset"]
        ax.annotate(
            DATASET_LABELS.get(ds, ds),
            xy=(p[x_key], p["bacc"]),
            xytext=(4, 2), textcoords="offset points",
            fontsize=5.5, color=DS_COLOR.get(ds, "0.4"),
            alpha=0.8,
        )

    if xlog:
        valid_xs = [x for x in all_xs if x > 0]
        if valid_xs:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.set_ylim(0.4, 1)
    
    # Plot regression line after scale is set
    if len(all_xs) > 1:
        x_range = np.array(ax.get_xlim())
        if ax._xlog:
            # Log-linear: fit log(x) vs y
            log_x_range = np.log10(x_range)
            y_fit = ax._slope * log_x_range + ax._intercept
        else:
            # Linear: fit x vs y
            y_fit = ax._slope * x_range + ax._intercept
        ax.plot(x_range, y_fit, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
        
        # Add correlation text to plot
        text_str = f"r = {ax._pearson_r:.3f}\n(p = {ax._p_val:.2e})"
        ax.text(0.95, 0.05, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.5))

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # Legend: shift types (from scatter handles) + fold/ensemble marker legend
    shift_handles = [
        plt.scatter([], [], marker="o", color=s["color"], label=s["label"],
                    edgecolors="white", s=55)
        for s in SHIFT_STYLE.values()
    ]
    marker_handles = [
        plt.scatter([], [], marker="x", color="0.4", s=30, linewidths=1.2, label="Fold"),
        plt.scatter([], [], marker="o", color="0.4", s=55, edgecolors="white",
                    linewidths=1.0, label="Ensemble"),
    ]
    ax.legend(
        handles=shift_handles + marker_handles,
        fontsize=8, framealpha=0.7, loc="lower left",
        ncol=2,
    )


def plot(
    fid_results: dict,
    cls_dir: Path,
    output_dir: Path,
    metrics: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pts = collect_points(fid_results, cls_dir)

    models = [("resnet18", "ResNet-18"), ("vit_b_16", "ViT-B/16")]

    # Filter metrics if specified
    metrics_to_plot = {k: v for k, v in DISTANCE_METRICS.items() if metrics is None or k in metrics}

    for metric_key, metric in metrics_to_plot.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

        for col, (model_key, model_label) in enumerate(models):
            _scatter_panel(
                axes[0, col], pts,
                xlabel=metric["abs_xlabel"],
                title=f"{model_label}  —  Absolute {metric['label']} vs Balanced accuracy",
                x_key=metric["abs_key"],
                xlog=False,
                model_filter=model_key,
            )
            _scatter_panel(
                axes[1, col], pts,
                xlabel=metric["norm_xlabel"],
                title=f"{model_label}  —  Normalized {metric['label']} vs Balanced accuracy",
                x_key=metric["norm_key"],
                xlog=True,
                model_filter=model_key,
            )

        fig.suptitle(
            f"Distribution shift ({metric['label']}) vs classification performance  —  all datasets × setups",
            fontsize=12, fontweight="bold",
        )

    
        for ext in ("pdf", "png"):
            out = output_dir / f"{metric['file_prefix']}_vs_bacc_scatter.{ext}"
            fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
            print(f"Saved → {out}")
        plt.close(fig)


def plot_fid_absolute_standalone(fid_results: dict, cls_dir: Path, output_dir: Path) -> None:
    """Standalone paper-quality figure: absolute FID vs balanced accuracy, one panel per model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pts = collect_points(fid_results, cls_dir)
    models = [("resnet18", "ResNet-18"), ("vit_b_16", "ViT-B/16")]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

    for ax, (model_key, model_label) in zip(axes, models):
        mpts = [p for p in pts if p["model"] == model_key and not np.isnan(p.get("fid", np.nan))]

        all_xs, all_ys = [], []
        for shift, style in SHIFT_STYLE.items():
            xs = [p["fid"] for p in mpts if p["shift"] == shift and not p["is_ensemble"]]
            ys = [p["bacc"] for p in mpts if p["shift"] == shift and not p["is_ensemble"]]
            if xs:
                ax.scatter(xs, ys, marker="x", s=45, linewidths=1.5,
                           color=style["color"], alpha=0.55, zorder=3)
                all_xs.extend(xs); all_ys.extend(ys)
            xe = [p["fid"] for p in mpts if p["shift"] == shift and p["is_ensemble"]]
            ye = [p["bacc"] for p in mpts if p["shift"] == shift and p["is_ensemble"]]
            if xe:
                ax.scatter(xe, ye, marker="o", s=80, linewidths=1.2,
                           color=style["color"], edgecolors="white", alpha=0.9,
                           zorder=4, label=style["label"])
                all_xs.extend(xe); all_ys.extend(ye)

        # Dataset annotations on ensemble points
        for p in mpts:
            if not p["is_ensemble"]:
                continue
            ds = p["dataset"]
            ax.annotate(
                DATASET_LABELS.get(ds, ds),
                xy=(p["fid"], p["bacc"]),
                xytext=(5, 3), textcoords="offset points",
                fontsize=9, color=DS_COLOR.get(ds, "0.4"), alpha=0.85,
            )

        # Pearson + regression line
        if len(all_xs) > 1:
            r, p_val = stats.pearsonr(all_xs, all_ys)
            coeffs = np.polyfit(all_xs, all_ys, 1)
            x_range = np.array([min(all_xs), max(all_xs)])
            ax.plot(x_range, np.polyval(coeffs, x_range),
                    color="gray", linestyle="--", linewidth=1.8, alpha=0.7, zorder=1)
            ax.text(0.95, 0.05,
                    f"r = {r:.3f}\n(p = {p_val:.2e})",
                    transform=ax.transAxes, fontsize=12,
                    va="bottom", ha="right",
                    bbox=dict(boxstyle="round", facecolor="white",
                              alpha=0.85, edgecolor="gray", linewidth=0.5))

        ax.set_xlabel("FID to training set", fontsize=15)
        ax.set_ylabel("Balanced accuracy", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.set_ylim(0.4, 1)
        ax.set_title(f"{model_label}  —  FID vs Balanced accuracy",
                     fontsize=16, fontweight="bold", pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.xaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        shift_handles = [
            plt.scatter([], [], marker="o", color=s["color"], label=s["label"],
                        edgecolors="white", s=80)
            for s in SHIFT_STYLE.values()
        ]
        marker_handles = [
            plt.scatter([], [], marker="x", color="0.4", s=45, linewidths=1.5, label="Fold"),
            plt.scatter([], [], marker="o", color="0.4", s=80, edgecolors="white",
                        linewidths=1.2, label="Ensemble"),
        ]
        ax.legend(handles=shift_handles + marker_handles,
                  fontsize=13, framealpha=0.7, loc="lower left", ncol=2)

    for ext in ("pdf", "png"):
        out = output_dir / f"fid_absolute_vs_bacc_standalone.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fid-pkl",      type=Path, default=_DEFAULT_FID_PKL)
    p.add_argument("--results-dir",  type=Path, default=_DEFAULT_CLS_DIR)
    p.add_argument("--output-dir",   type=Path, default=_DEFAULT_OUT)
    p.add_argument("--metrics",      nargs="+", choices=list(DISTANCE_METRICS.keys()),
                   help="Distance metrics to plot (default: all). Choose from: fid, kid, mahalanobis")
    args = p.parse_args()

    with open(args.fid_pkl, "rb") as f:
        fid_results = pickle.load(f)

    plot(fid_results, args.results_dir, args.output_dir, metrics=args.metrics)
    plot_fid_absolute_standalone(fid_results, args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
