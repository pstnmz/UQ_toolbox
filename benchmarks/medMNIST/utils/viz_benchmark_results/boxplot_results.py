from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

try:
	from scipy.stats import mannwhitneyu
except Exception:
	mannwhitneyu = None


SHIFT_TO_SUBFOLDERS = {
	"ID": ["in_distribution"],
	"CS": ["corruption_shifts"],
	"PS": ["population_shifts"],
	"NCS": ["new_class_shifts"],
}

BACKBONE_LABELS = {
	"resnet": "R18",
	"vit": "ViT",
}

BASELINE_MEAN_METHODS = ["KNN_Raw", "GPS", "MLS", "MSR", "MSR_calibrated", "TTA"]
TARGET_METHOD = "ZScore_Aggregation_ensemble"
BOX_ERROR = "Error Rate"
BOX_AUGRC = "Mean Agg + Ens AUGRC"
SIGNIFICANCE_ALPHA = 0.05


def parse_args() -> argparse.Namespace:
	script_dir = Path(__file__).resolve().parent
	repo_root = script_dir.parents[3]
	default_json_root = repo_root / "Benchmarks" / "medMNIST" / "results" / "jsons_results"
	default_output = (
		repo_root
		/ "Benchmarks"
		/ "medMNIST"
		/ "results"
		/ "figures"
		/ "boxplot_error_rate_vs_mean_agg_ens_augrc.png"
	)

	parser = argparse.ArgumentParser(
		description=(
			"Create side-by-side boxplots of model error rate (1 - test_accuracy) versus "
			"ZScore_Aggregation_ensemble/augrc across ID, CS, PS, and NCS shifts for "
			"R18 and ViT backbones."
		)
	)
	parser.add_argument("--json-root", type=Path, default=default_json_root)
	parser.add_argument("--output", type=Path, default=default_output)
	return parser.parse_args()


def _detect_backbone_label(model_backbone: Optional[str]) -> Optional[str]:
	if not model_backbone:
		return None
	backbone_l = model_backbone.lower()
	if "resnet" in backbone_l:
		return BACKBONE_LABELS["resnet"]
	if "vit" in backbone_l:
		return BACKBONE_LABELS["vit"]
	return None


def _safe_float(value: object) -> Optional[float]:
	if value is None:
		return None
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def iter_shift_files(json_root: Path, subfolders: Iterable[str]) -> Iterable[Path]:
	for subfolder in subfolders:
		folder = json_root / subfolder
		if not folder.exists():
			continue
		yield from folder.glob("*.json")


def iter_parsed_runs(json_root: Path) -> Iterable[Dict[str, object]]:
	for shift_label, subfolders in SHIFT_TO_SUBFOLDERS.items():
		for json_path in iter_shift_files(json_root, subfolders):
			try:
				with json_path.open("r", encoding="utf-8") as f:
					payload = json.load(f)
			except (OSError, json.JSONDecodeError):
				continue

			backbone = _detect_backbone_label(payload.get("model_backbone"))
			if backbone is None:
				continue

			methods = payload.get("methods", {})
			if not isinstance(methods, dict):
				continue

			yield {
				"shift": shift_label,
				"backbone": backbone,
				"run_id": json_path.stem,
				"payload": payload,
				"methods": methods,
			}


def build_dataframe(json_root: Path) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []

	for run in iter_parsed_runs(json_root):
		shift_label = run["shift"]
		backbone = run["backbone"]
		payload = run["payload"]
		methods = run["methods"]
		group = f"{shift_label} | {backbone}"

		test_acc = _safe_float(payload.get("test_accuracy"))
		if test_acc is not None:
			rows.append(
				{
					"shift": shift_label,
					"backbone": backbone,
					"group": group,
					"box_type": BOX_ERROR,
					"value": 1.0 - test_acc,
				}
			)

		target_score = _safe_float(methods.get(TARGET_METHOD, {}).get("augrc"))
		if target_score is not None:
			rows.append(
				{
					"shift": shift_label,
					"backbone": backbone,
					"group": group,
					"box_type": BOX_AUGRC,
					"value": target_score,
				}
			)

	return pd.DataFrame(rows)


def permutation_mean_diff_pvalue(
	x: np.ndarray,
	y: np.ndarray,
	rng: np.random.Generator,
	n_perm: int = 50000,
) -> float:
	if x.size == 0 or y.size == 0:
		return float("nan")
	obs = abs(float(np.mean(x) - np.mean(y)))
	pooled = np.concatenate([x, y])
	nx = x.size
	extreme = 0
	for _ in range(n_perm):
		perm = rng.permutation(pooled)
		stat = abs(float(np.mean(perm[:nx]) - np.mean(perm[nx:])))
		if stat >= obs:
			extreme += 1
	return float((extreme + 1) / (n_perm + 1))


def summarize_distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame()

	rng = np.random.default_rng(42)
	rows: List[Dict[str, object]] = []

	for (shift, backbone), gdf in df.groupby(["shift", "backbone"], sort=False):
		agg = gdf.loc[gdf["box_type"] == BOX_AUGRC, "value"].dropna().to_numpy(dtype=float)
		ind = gdf.loc[gdf["box_type"] == BOX_ERROR, "value"].dropna().to_numpy(dtype=float)
		if agg.size == 0 or ind.size == 0:
			continue

		mwu_p = float("nan")
		if mannwhitneyu is not None:
			try:
				mwu_p = float(mannwhitneyu(agg, ind, alternative="two-sided").pvalue)
			except Exception:
				mwu_p = float("nan")

		rows.append(
			{
				"shift": shift,
				"backbone": backbone,
				"group": f"{shift} | {backbone}",
				"n_augrc": int(agg.size),
				"n_error_rate": int(ind.size),
				"mean_augrc": float(np.mean(agg)),
				"mean_error_rate": float(np.mean(ind)),
				"median_augrc": float(np.median(agg)),
				"median_error_rate": float(np.median(ind)),
				"delta_mean_augrc_minus_error": float(np.mean(agg) - np.mean(ind)),
				"perm_mean_diff_p": permutation_mean_diff_pvalue(agg, ind, rng=rng),
				"mannwhitney_p": mwu_p,
			}
		)

	return pd.DataFrame(rows)


def create_plot(df: pd.DataFrame, output_path: Path, stats_df: Optional[pd.DataFrame] = None) -> None:
	order = [
		"ID | R18",
		"ID | ViT",
		"CS | R18",
		"CS | ViT",
		"PS | R18",
		"PS | ViT",
		"NCS | R18",
		"NCS | ViT",
	]
	hue_order = [BOX_ERROR, BOX_AUGRC]
	legend_labels = {
		BOX_ERROR: "Error Rate (1 - test_acc)",
		BOX_AUGRC: "Mean Agg + Ens AUGRC",
	}

	sns.set_theme(style="whitegrid", context="talk")
	alpha = 0.8
	palette = {
		BOX_ERROR: (0.3098, 0.5922, 0.7765, alpha),
		BOX_AUGRC: (0.8706, 0.8706, 0.5961, alpha),
	}

	fig, ax = plt.subplots(figsize=(18, 8))
	group_positions = {
		"ID | R18": 0.0,
		"ID | ViT": 1.0,
		"CS | R18": 3.0,
		"CS | ViT": 4.0,
		"PS | R18": 6.0,
		"PS | ViT": 6.65,
		"NCS | R18": 7.25,
		"NCS | ViT": 7.9,
	}
	thin_groups = {"PS | R18", "PS | ViT", "NCS | R18", "NCS | ViT"}
	base_width = 0.32
	thin_width = 0.22
	offsets = {
		BOX_ERROR: -0.17,
		BOX_AUGRC: 0.17,
	}

	for box_type in hue_order:
		subset = df[df["box_type"] == box_type]
		for grp in order:
			vals = subset.loc[subset["group"] == grp, "value"].dropna().to_numpy()
			if vals.size == 0:
				continue
			bp = ax.boxplot(
				[vals],
				positions=[group_positions[grp] + offsets[box_type]],
				widths=[thin_width if grp in thin_groups else base_width],
				patch_artist=True,
				showfliers=False,
			)
			for patch in bp["boxes"]:
				patch.set_facecolor(palette[box_type])
				patch.set_edgecolor(palette[box_type])
			for median in bp["medians"]:
				median.set_color("black")
			for whisker in bp["whiskers"]:
				whisker.set_color(palette[box_type])
			for cap in bp["caps"]:
				cap.set_color(palette[box_type])

	ax.set_title("Baseline Error Rate vs Average Error Rate after Failure Detection", fontweight="bold")
	ax.set_xlabel("Shift Type and Backbone", labelpad=40, fontweight="bold")
	ax.set_ylabel("Error Rate", fontweight="bold")

	# First line on ticks: backbone names (not rotated).
	tick_backbones = [
		"R18",
		"ViT",
		"R18",
		"ViT",
		"R18",
		"ViT",
		"R18",
		"ViT",
	]
	ax.set_xticks([group_positions[g] for g in order])
	ax.set_xticklabels(tick_backbones, rotation=0, ha="center")
	ax.set_xlim(-0.8, 8.5)

	# Second line: merged shift labels centered under each backbone pair.
	for xpos, shift in [(0.5, "ID"), (3.5, "CS"), (6.325, "PS"), (7.575, "NCS")]:
		ax.text(
			xpos,
			-0.11,
			shift,
			ha="center",
			va="top",
			transform=ax.get_xaxis_transform(),
			fontsize=18,
		)

	handles = [
		mpatches.Patch(facecolor=palette[name], edgecolor=palette[name], label=legend_labels[name])
		for name in hue_order
	]
	ax.legend(handles=handles, title="", loc="best")
	ax.xaxis.grid(False)
	ax.grid(axis="y", alpha=0.25)

	# Add significance markers per shift/backbone group.
	if stats_df is not None and not stats_df.empty:
		y_span = float(df["value"].max() - df["value"].min())
		if y_span <= 0:
			y_span = 0.1
		extra_top = 0.0
		for _, row in stats_df.iterrows():
			group = row.get("group")
			if group not in group_positions:
				continue

			mwu_p = row.get("mannwhitney_p")
			perm_p = row.get("perm_mean_diff_p")
			p_value = mwu_p if pd.notna(mwu_p) else perm_p
			if pd.isna(p_value) or float(p_value) >= SIGNIFICANCE_ALPHA:
				continue

			gvals = df.loc[df["group"] == group, "value"]
			if gvals.empty:
				continue

			x_left = group_positions[group] + offsets[BOX_ERROR]
			x_right = group_positions[group] + offsets[BOX_AUGRC]
			y = float(gvals.max()) + 0.02 * y_span
			h = 0.012 * y_span
			ax.plot([x_left, x_left, x_right, x_right], [y, y + h, y + h, y], color="black", linewidth=1.2)
			ax.text(
				(group_positions[group]),
				y + h + 0.005 * y_span,
				"*",
				ha="center",
				va="bottom",
				fontsize=16,
				fontweight="bold",
			)
			extra_top = max(extra_top, (y + h + 0.03 * y_span) - float(df["value"].max()))

		if extra_top > 0:
			cur_min, cur_max = ax.get_ylim()
			ax.set_ylim(cur_min, cur_max + extra_top)

	plt.tight_layout()
	plt.subplots_adjust(bottom=0.27)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.close()


def main() -> None:
	args = parse_args()
	df = build_dataframe(args.json_root)
	if df.empty:
		raise RuntimeError(f"No valid data parsed from {args.json_root}")
	stats_df = summarize_distribution_stats(df)

	create_plot(df, args.output, stats_df=stats_df)
	counts = (
		df.groupby(["shift", "backbone", "box_type"])["value"]
		.size()
		.reset_index(name="n")
	)
	print("Saved figure:", args.output)
	print("\nSamples per box:")
	print(counts.to_string(index=False))
	if not stats_df.empty:
		print("\nDistribution comparison (Error Rate vs Mean Agg + Ens AUGRC, by shift/backbone):")
		print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
	main()
