#!/usr/bin/env python3
"""Plot error rate vs ZScore aggregation AU-GRC across all benchmark JSON results.

This script scans all JSON files under:
Benchmarks/medMNIST/results/jsons_results/

For each tested configuration, it extracts:
- root key: test_accuracy
- nested key: methods["ZScore_Aggregation_ensemble"]["augrc"]

It then creates a large horizontal scatter plot with a multi-level x-axis:
1) setup: S, DA, DO, DADO (one label per configuration tick)
2) backbone: R18, ViT (each covers 4 setup ticks)
3) dataset name (each covers 8 ticks)
4) shift setup: ID, Corruption, Population, New Class
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath


SHIFT_LABELS = {
	"in_distribution": "ID",
	"corruption_shifts": "Corruption",
	"population_shifts": "Population",
	"new_class_shifts": "New Class",
}

SHIFT_ORDER = ["in_distribution", "corruption_shifts", "population_shifts", "new_class_shifts"]
SETUP_ORDER = ["S", "DA", "DO", "DADO"]
BACKBONE_ORDER = ["R18", "ViT"]

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


def build_plot_index(records: List[RunRecord]):
	by_key = {(r.shift, r.dataset, r.backbone, r.setup): r for r in records}

	datasets_per_shift: Dict[str, List[str]] = {}
	for shift in SHIFT_ORDER:
		ds = sorted({r.dataset for r in records if r.shift == shift})
		if ds:
			datasets_per_shift[shift] = ds

	x_positions: List[int] = []
	setup_labels: List[str] = []
	y_error: List[float] = []
	y_augrc: List[float] = []

	backbone_ticks: List[float] = []
	backbone_labels: List[str] = []
	dataset_ticks: List[float] = []
	dataset_labels: List[str] = []
	shift_ticks: List[float] = []
	shift_labels: List[str] = []
	shift_spans: List[Tuple[float, float]] = []

	boundary_after_x: List[float] = []

	x = 0
	for shift in SHIFT_ORDER:
		datasets = datasets_per_shift.get(shift, [])
		if not datasets:
			continue

		shift_start = x
		for dataset in datasets:
			dataset_start = x
			for backbone in BACKBONE_ORDER:
				bb_start = x
				for setup in SETUP_ORDER:
					key = (shift, dataset, backbone, setup)
					rec = by_key.get(key)

					x_positions.append(x)
					setup_labels.append(setup)
					if rec is None:
						y_error.append(np.nan)
						y_augrc.append(np.nan)
					else:
						y_error.append(rec.error_rate)
						y_augrc.append(rec.zscore_augrc)
					x += 1

				bb_end = x - 1
				backbone_ticks.append((bb_start + bb_end) / 2.0)
				backbone_labels.append(backbone)
				boundary_after_x.append(x - 0.5)

			dataset_end = x - 1
			dataset_ticks.append((dataset_start + dataset_end) / 2.0)
			dataset_labels.append(dataset)
			boundary_after_x.append(x - 0.5)

		shift_end = x - 1
		shift_ticks.append((shift_start + shift_end) / 2.0)
		shift_labels.append(SHIFT_LABELS[shift])
		shift_spans.append((shift_start - 0.5, shift_end + 0.5))
		boundary_after_x.append(x - 0.5)

	return {
		"x": np.array(x_positions, dtype=float),
		"setup_labels": setup_labels,
		"y_error": np.array(y_error, dtype=float),
		"y_augrc": np.array(y_augrc, dtype=float),
		"backbone_ticks": backbone_ticks,
		"backbone_labels": backbone_labels,
		"dataset_ticks": dataset_ticks,
		"dataset_labels": dataset_labels,
		"shift_ticks": shift_ticks,
		"shift_labels": shift_labels,
		"shift_spans": shift_spans,
		"boundaries": sorted(set(boundary_after_x)),
	}


def apply_secondary_axis(ax, ticks, labels, pad, fontsize, rotation=0, ha="center", fontweight="normal"):
	sec = ax.secondary_xaxis("bottom")
	sec.spines["bottom"].set_position(("outward", pad))
	sec.spines["bottom"].set_visible(False)
	sec.set_xticks(ticks)
	sec.set_xticklabels(labels, fontsize=fontsize, rotation=rotation, ha=ha, fontweight=fontweight)
	sec.tick_params(axis="x", length=0, pad=2)
	return sec


def lightning_marker_path() -> MplPath:
	# Stylized lightning bolt marker (normalized around center).
	verts = np.array(
		[
			(-0.10, 1.00),
			(0.20, 0.30),
			(0.02, 0.30),
			(0.25, -1.00),
			(-0.25, -0.10),
			(-0.02, -0.10),
			(-0.10, 1.00),
		],
		dtype=float,
	)
	codes = [
		MplPath.MOVETO,
		MplPath.LINETO,
		MplPath.LINETO,
		MplPath.LINETO,
		MplPath.LINETO,
		MplPath.LINETO,
		MplPath.CLOSEPOLY,
	]
	return MplPath(verts, codes)


def create_plot(index_data, output_path: Path, title: str):
	x = index_data["x"]
	y_error = index_data["y_error"]
	y_augrc = index_data["y_augrc"]
	lightning_color = plt.cm.tab20(17)
	lightning_edge = "#2a2208"

	n_points = len(x)
	fig_w = max(16, 0.105 * n_points)
	fig, ax = plt.subplots(figsize=(fig_w, 7.4))

	valid_error = np.isfinite(y_error)
	valid_augrc = np.isfinite(y_augrc)

	ax.scatter(
		x[valid_error],
		y_error[valid_error],
		s=42,
		color="#4f97c6",
		edgecolor="#2f6f99",
		linewidths=0.55,
		alpha=0.94,
		label="Error rate (1 - test_accuracy)",
		zorder=3,
	)

	for xi, yi in zip(x[valid_augrc], y_augrc[valid_augrc]):
		txt = ax.text(
			xi,
			yi,
			"",
			ha="center",
			va="center",
			fontsize=12,
			color=lightning_color,
			alpha=0.95,
			zorder=5,
		)
		txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground=lightning_edge), pe.Normal()])

	for b in index_data["boundaries"]:
		ax.axvline(b, color="#c7c7c7", linewidth=0.4, alpha=0.45, zorder=1)

	ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.28, zorder=0)
	ax.set_xlim(-0.5, max(x) + 0.5)
	ax.set_ylabel("Error Rate", fontsize=11, fontweight="bold")
	ax.set_title(title, fontsize=13, pad=10, fontweight="bold")
	ax.spines["bottom"].set_visible(False)

	ax.set_xticks(x)
	ax.set_xticklabels(index_data["setup_labels"], fontsize=8, rotation=90, ha="right")
	ax.tick_params(axis="x", length=0, pad=2)

	apply_secondary_axis(
		ax,
		index_data["backbone_ticks"],
		index_data["backbone_labels"],
		pad=26,
		fontsize=8,
		rotation=18,
		ha="right",
	)
	apply_secondary_axis(
		ax,
		index_data["dataset_ticks"],
		index_data["dataset_labels"],
		pad=44,
		fontsize=8,
		rotation=12,
		ha="right",
	)
	apply_secondary_axis(
		ax,
		index_data["shift_ticks"],
		index_data["shift_labels"],
		pad=74,
		fontsize=11,
		fontweight="bold",
	)

	for _, x1 in index_data["shift_spans"][:-1]:
		ax.axvline(x1, color="#202020", linewidth=1.9, alpha=0.95, zorder=2)

	for x0, x1 in index_data["shift_spans"]:
		ax.hlines(
			y=-0.20,
			xmin=x0,
			xmax=x1,
			transform=ax.get_xaxis_transform(),
			colors="#303030",
			linewidth=2.0,
			alpha=0.9,
			clip_on=False,
			zorder=6,
		)

	err_handle = Line2D(
		[],
		[],
		linestyle="None",
		marker="o",
		markersize=7,
        alpha=0.9,
		markerfacecolor="#4f97c6",
		markeredgecolor="#2f6f99",
		markeredgewidth=0.55,
		label="Error rate (1 - test_accuracy)",
	)
	bolt_handle = Line2D(
		[],
		[],
		linestyle="None",
		marker="$$",
		markersize=12,
		alpha=0.9,
		markerfacecolor=lightning_color,
		markeredgecolor=lightning_edge,
		markeredgewidth=0.7,
		label="Mean Agg + Ens AUGRC",
	)
	ax.legend(handles=[err_handle, bolt_handle], loc="upper left", frameon=True, fontsize=9)
	fig.subplots_adjust(bottom=0.25, left=0.06, right=0.995, top=0.91)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
	plt.close(fig)


def parse_args() -> argparse.Namespace:
	root = Path(__file__).resolve().parents[2]
	default_json_root = root / "results" / "jsons_results"
	default_out = root / "results" / "figures" / "error_vs_zscore_augrc_multilevel.png"
	parser = argparse.ArgumentParser(description="Plot error rate and ZScore aggregation AU-GRC.")
	parser.add_argument("--json-root", type=Path, default=default_json_root)
	parser.add_argument("--output", type=Path, default=default_out)
	parser.add_argument(
		"--title",
		type=str,
		default="Test Error Rate (1-Acc) vs AUGRC Mean Aggregation + Ensemble Across Configurations",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	records = load_records(args.json_root)
	if not records:
		raise RuntimeError(f"No valid JSON records found in: {args.json_root}")

	latest_records = keep_latest(records)
	index_data = build_plot_index(latest_records)
	if len(index_data["x"]) == 0:
		raise RuntimeError("No configurations available to plot after indexing.")

	create_plot(index_data, args.output, args.title)
	print(f"Saved figure: {args.output}")
	print(f"Plotted configurations: {len(index_data['x'])}")
	print(f"Records scanned: {len(records)} | latest unique records used: {len(latest_records)}")


if __name__ == "__main__":
	main()
