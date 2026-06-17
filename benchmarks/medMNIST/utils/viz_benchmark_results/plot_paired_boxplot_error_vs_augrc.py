#!/usr/bin/env python3
"""Paired boxplot: error rate vs ZScore aggregation AU-GRC.

Compact (narrow) alternative to plot_aggregationAUGRC_vs_standolone_error_rate.py.
Shows the same data collapsed into two side-by-side boxplots with individual
points overlaid and lines connecting each matched (same config) pair.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── data-loading (mirrors the wide-plot script) ────────────────────────────────

SHIFT_LABELS = {
    "in_distribution": "ID",
    "corruption_shifts": "Corruption",
    "population_shifts": "Population",
    "new_class_shifts": "New Class",
}

SHIFT_ORDER = ["in_distribution", "corruption_shifts", "population_shifts", "new_class_shifts"]

BACKBONE_TO_LABEL = {
    "resnet18": "R18",
    "vit_b_16": "ViT",
}

DATASET_DISPLAY = {
    "dermamnist-e-external": "dermamnist-e-ext",
    "midog": "midog ++",
}

SETUP_TO_LABEL = {
    "": "S",
    "DA": "DA",
    "DO": "DO",
    "DADO": "DADO",
}

SHIFT_COLORS = {
    "in_distribution": "#4f97c6",
    "corruption_shifts": "#e07b39",
    "population_shifts": "#5ab56e",
    "new_class_shifts": "#b06bbf",
}


@dataclass
class RunRecord:
    shift: str
    dataset: str
    backbone: str
    setup: str
    timestamp: str
    error_rate: float
    zscore_augrc: float
    source_file: Path


def load_records(jsons_root: Path) -> List[RunRecord]:
    records: List[RunRecord] = []
    for path in sorted(jsons_root.rglob("*.json")):
        shift_folder = path.parent.name
        if shift_folder not in SHIFT_LABELS:
            continue

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        test_accuracy = data.get("test_accuracy")
        methods = data.get("methods", {})
        zscore = methods.get("ZScore_Aggregation_ensemble", {})
        zscore_augrc = zscore.get("augrc")

        if test_accuracy is None or zscore_augrc is None:
            continue

        dataset = str(data.get("flag", "unknown"))
        dataset = DATASET_DISPLAY.get(dataset, dataset)
        backbone_raw = str(data.get("model_backbone", "unknown"))
        setup_raw = str(data.get("setup", ""))
        timestamp = str(data.get("timestamp", ""))

        backbone_label = BACKBONE_TO_LABEL.get(backbone_raw, backbone_raw)
        setup_label = SETUP_TO_LABEL.get(setup_raw, setup_raw if setup_raw else "S")

        records.append(
            RunRecord(
                shift=shift_folder,
                dataset=dataset,
                backbone=backbone_label,
                setup=setup_label,
                timestamp=timestamp,
                error_rate=1.0 - float(test_accuracy),
                zscore_augrc=float(zscore_augrc),
                source_file=path,
            )
        )

    return records


def keep_latest(records: List[RunRecord]) -> List[RunRecord]:
    latest: Dict[Tuple[str, str, str, str], RunRecord] = {}
    for rec in records:
        key = (rec.shift, rec.dataset, rec.backbone, rec.setup)
        prev = latest.get(key)
        if prev is None or rec.timestamp > prev.timestamp:
            latest[key] = rec
    return list(latest.values())


# ── figure ─────────────────────────────────────────────────────────────────────

POS_ERROR = 0.0
POS_AUGRC = 1.0
JITTER_STD = 0.022


def create_paired_boxplot(records: List[RunRecord], output_path: Path, title: str) -> None:
    rng = np.random.default_rng(42)

    # Pre-compute jitter so line endpoints match dot positions exactly.
    jitter_err = rng.normal(0.0, JITTER_STD, size=len(records))
    jitter_aug = rng.normal(0.0, JITTER_STD, size=len(records))

    error_vals = np.array([r.error_rate for r in records])
    augrc_vals = np.array([r.zscore_augrc for r in records])

    fig, ax = plt.subplots(figsize=(4, 4.2))

    # 1. Connecting lines (drawn first, behind everything).
    for i, rec in enumerate(records):
        color = SHIFT_COLORS.get(rec.shift, "#888888")
        ax.plot(
            [POS_ERROR + jitter_err[i], POS_AUGRC + jitter_aug[i]],
            [rec.error_rate, rec.zscore_augrc],
            color=color,
            linewidth=0.6,
            alpha=0.30,
            zorder=1,
            solid_capstyle="round",
        )

    # 2. Boxplots.
    _bp_kw = dict(
        widths=0.30,
        patch_artist=True,
        zorder=3,
        showfliers=False,  # fliers shown as individual scatter points
    )
    ax.boxplot(
        [error_vals],
        positions=[POS_ERROR],
        boxprops=dict(facecolor="#c6dff0", color="#2f6f99", linewidth=1.3),
        medianprops=dict(color="#1a4f72", linewidth=2.2),
        whiskerprops=dict(color="#2f6f99", linewidth=1.0, linestyle="--"),
        capprops=dict(color="#2f6f99", linewidth=1.2),
        **_bp_kw,
    )
    ax.boxplot(
        [augrc_vals],
        positions=[POS_AUGRC],
        boxprops=dict(facecolor="#fde8cc", color="#c85a00", linewidth=1.3),
        medianprops=dict(color="#7a3200", linewidth=2.2),
        whiskerprops=dict(color="#c85a00", linewidth=1.0, linestyle="--"),
        capprops=dict(color="#c85a00", linewidth=1.2),
        **_bp_kw,
    )

    # 3. Individual scatter points (on top).
    for i, rec in enumerate(records):
        color = SHIFT_COLORS.get(rec.shift, "#888888")
        ax.scatter(
            POS_ERROR + jitter_err[i],
            rec.error_rate,
            s=14,
            color=color,
            alpha=0.80,
            zorder=5,
            linewidths=0,
        )
        ax.scatter(
            POS_AUGRC + jitter_aug[i],
            rec.zscore_augrc,
            s=14,
            color=color,
            alpha=0.80,
            zorder=5,
            linewidths=0,
        )

    # Axes cosmetics.
    ax.set_xlim(-0.50, 1.50)
    ax.set_xticks([POS_ERROR, POS_AUGRC])
    ax.set_xticklabels(["Baseline (1 − Acc)", "Mean Agg + Ens AUGRC"], fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Error Rate", fontsize=8)
    ax.set_title(title, fontsize=8, pad=8, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.30, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: distribution shifts.
    present_shifts = {r.shift for r in records}
    handles = [
        mpatches.Patch(facecolor=SHIFT_COLORS[s], label=SHIFT_LABELS[s], alpha=0.85)
        for s in SHIFT_ORDER
        if s in present_shifts
    ]
    ax.legend(
        handles=handles,
        fontsize=7,
        loc="upper right",
        frameon=True,
        framealpha=0.85,
        edgecolor="#cccccc",
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.0,
        handletextpad=0.4,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_json_root = root / "results" / "jsons_results"
    default_out = root / "results" / "figures" / "paired_boxplot_error_vs_augrc.png"
    parser = argparse.ArgumentParser(description="Paired boxplot: error rate vs ZScore AU-GRC.")
    parser.add_argument("--json-root", type=Path, default=default_json_root)
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument(
        "--title",
        type=str,
        default="Error rate Baseline vs after Failure Detection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.json_root)
    if not records:
        raise RuntimeError(f"No valid JSON records found in: {args.json_root}")
    latest = keep_latest(records)
    create_paired_boxplot(latest, args.output, args.title)
    print(f"Saved: {args.output}")
    print(f"Configs plotted: {len(latest)}")


if __name__ == "__main__":
    main()
