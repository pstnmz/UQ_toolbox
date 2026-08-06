"""
Visualization of DINOv3-FID distances between test distributions and training data.

Two barplots:
  1. Absolute FID distance to train for each dataset × shift type
  2. Normalized FID (relative to the ID test split = 1.0) for non-ID splits

Usage
-----
    python distances_vs_perf.py
    python distances_vs_perf.py --metrics fid kid
    python distances_vs_perf.py --metrics mahalanobis --output-dir /path/to/out
    python distances_vs_perf.py --fid-pkl /path/to/fid_results.pkl --metrics fid
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DEFAULT_PKL = (
    _HERE.parents[1] / "results" / "dinov3_embeddings" / "fid_results.pkl"
)
_DEFAULT_OUT = _HERE.parents[1] / "results" / "figures"

# ---------------------------------------------------------------------------
# Aesthetics
# ---------------------------------------------------------------------------
SHIFT_META = {
    "id":               dict(label="ID",          color="#4C72B0"),
    "random_s3":        dict(label="CS",  color="#DD8452"),
    "population_shift": dict(label="PS", color="#55A868"),
    "new_classes":      dict(label="NCS",  color="#C44E52"),
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


def _dataset_label(flag: str) -> str:
    return DATASET_LABELS.get(flag, flag)


# ---------------------------------------------------------------------------
# Distance metric configurations (FID / KID / Mahalanobis)
# ---------------------------------------------------------------------------
DISTANCE_METRICS = {
    "fid": dict(
        abs_key="fid_distance",
        norm_key="normalized_fid",
        abs_ylabel="FID distance to train set",
        norm_ylabel="FID / FID_ID  (ID test = 1.0)",
        suptitle="DINOv3-FID distribution distances across datasets and shift types",
        file_prefix="fid",
    ),
    "kid": dict(
        abs_key="kid",
        norm_key="normalized_kid",
        abs_ylabel="KID distance to train set",
        norm_ylabel="KID / KID_ID  (ID test = 1.0)",
        suptitle="DINOv3-KID (Kernel Inception Distance) across datasets and shift types",
        file_prefix="kid",
    ),
    "mahalanobis": dict(
        abs_key="mahalanobis",
        norm_key="normalized_mahalanobis",
        abs_ylabel="Mahalanobis distance to train distribution",
        norm_ylabel="Mahalanobis / Mahalanobis_ID  (ID test = 1.0)",
        suptitle="DINOv3-Mahalanobis distribution distances across datasets and shift types",
        file_prefix="mahalanobis",
    ),
}


# ---------------------------------------------------------------------------
# Core plotting helpers
# ---------------------------------------------------------------------------

def _grouped_barplot(
    ax: plt.Axes,
    datasets: list[str],
    shift_data: dict[str, dict[str, float]],  # shift_key → {dataset → value}
    shifts_present: list[str],
    ylabel: str,
    title: str,
) -> None:
    """Draw grouped bars: one group per dataset, one bar per shift type."""
    n_datasets = len(datasets)
    n_shifts = len(shifts_present)
    width = 0.8 / n_shifts          # bar width
    x = np.arange(n_datasets)

    for i, shift in enumerate(shifts_present):
        meta = SHIFT_META[shift]
        values = [shift_data[shift].get(ds, np.nan) for ds in datasets]
        offset = (i - (n_shifts - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width,
            label=meta["label"],
            color=meta["color"],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on top of each bar (offset is multiplicative for log compat)
        for bar, v in zip(bars, values):
            if np.isnan(v):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.04,
                f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=6.5, color="0.3",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_dataset_label(ds) for ds in datasets], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, framealpha=0.7, loc="upper left")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_distances(
    fid_results: dict,
    output_dir: Path,
    metric_key: str = "fid",
) -> None:
    """Generate barplot figure for one distance metric (fid | kid | mahalanobis)."""
    cfg = DISTANCE_METRICS[metric_key]
    abs_key  = cfg["abs_key"]
    norm_key = cfg["norm_key"]

    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(fid_results.keys())
    all_shifts = [s for s in SHIFT_META if any(s in fid_results[ds] for ds in datasets)]

    abs_data:  dict[str, dict[str, float]] = {s: {} for s in all_shifts}
    norm_data: dict[str, dict[str, float]] = {s: {} for s in all_shifts if s != "id"}

    for ds, shifts in fid_results.items():
        for shift, vals in shifts.items():
            if shift not in abs_data or abs_key not in vals:
                continue
            abs_data[shift][ds] = float(vals[abs_key])
            if shift != "id" and norm_key in vals:
                norm_data[shift][ds] = float(vals[norm_key])

    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(10, len(datasets) * 1.4), 10),
        constrained_layout=True,
    )

    _grouped_barplot(
        axes[0], datasets, abs_data, all_shifts,
        ylabel=cfg["abs_ylabel"],
        title=f"Absolute {metric_key.upper()}: test distributions → training data",
    )

    non_id_shifts = [s for s in all_shifts if s != "id"]
    _grouped_barplot(
        axes[1], datasets, norm_data, non_id_shifts,
        ylabel=cfg["norm_ylabel"],
        title=f"Normalized {metric_key.upper()}  (relative to ID test split)",
    )

    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axes[1].axhline(1.0, color=SHIFT_META["id"]["color"], linewidth=1.2,
                    linestyle="--", alpha=0.7, label="ID baseline (= 1.0)")
    axes[1].legend(fontsize=8, framealpha=0.7, loc="upper left")

    fig.suptitle(cfg["suptitle"], fontsize=13, fontweight="bold", y=1.01)

    for ext in ("pdf", "png"):
        out = output_dir / f"{cfg['file_prefix']}_distances_barplots.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"Saved → {out}")
    plt.close(fig)


# Keep old name as alias for backward compatibility
def plot_fid_distances(fid_results: dict, output_dir: Path) -> None:
    plot_distances(fid_results, output_dir, "fid")


def plot_fid_absolute_standalone(fid_results: dict, output_dir: Path) -> None:
    """Standalone paper-quality barplot of absolute FID distances only."""
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(fid_results.keys())
    all_shifts = [s for s in SHIFT_META if any(s in fid_results[ds] for ds in datasets)]

    abs_data: dict[str, dict[str, float]] = {s: {} for s in all_shifts}
    for ds, shifts in fid_results.items():
        for shift, vals in shifts.items():
            if shift in abs_data and "fid_distance" in vals:
                abs_data[shift][ds] = float(vals["fid_distance"])

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.4), 6), constrained_layout=True)

    n_datasets = len(datasets)
    n_shifts = len(all_shifts)
    width = 0.8 / n_shifts
    x = np.arange(n_datasets)

    for i, shift in enumerate(all_shifts):
        meta = SHIFT_META[shift]
        values = [abs_data[shift].get(ds, np.nan) for ds in datasets]
        offset = (i - (n_shifts - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width,
            label=meta["label"],
            color=meta["color"],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, v in zip(bars, values):
            if np.isnan(v):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.04,
                f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=9, color="0.3",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_dataset_label(ds) for ds in datasets], fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("FID to train set", fontsize=15)
    ax.set_title("DINOv3-FID distribution distances", fontsize=16, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=13, framealpha=0.7, loc="upper left")

    for ext in ("pdf", "png"):
        out = output_dir / f"fid_absolute_standalone.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fid-pkl",    type=Path, default=_DEFAULT_PKL)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--metrics",    nargs="+", choices=list(DISTANCE_METRICS.keys()),
                   help="Distance metrics to plot (default: all). Choose from: fid, kid, mahalanobis")
    args = p.parse_args()

    with open(args.fid_pkl, "rb") as f:
        fid_results = pickle.load(f)

    metrics_to_plot = args.metrics if args.metrics else list(DISTANCE_METRICS.keys())
    for metric_key in metrics_to_plot:
        plot_distances(fid_results, args.output_dir, metric_key)

    plot_fid_absolute_standalone(fid_results, args.output_dir)


if __name__ == "__main__":
    main()
