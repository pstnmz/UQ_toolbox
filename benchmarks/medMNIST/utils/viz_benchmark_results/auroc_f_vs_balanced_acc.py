from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def normalize_setup(value: Optional[str]) -> str:
	if value is None:
		return "standard"
	value = str(value).strip()
	return value if value else "standard"


def parse_dataset_model_setup(name_core: str) -> Optional[Tuple[str, str, str]]:
	if "_vit_b_16" in name_core:
		dataset, rest = name_core.split("_vit_b_16", 1)
		model = "vit_b_16"
	elif "_resnet18" in name_core:
		dataset, rest = name_core.split("_resnet18", 1)
		model = "resnet18"
	else:
		return None

	setup = normalize_setup(rest.lstrip("_"))
	if setup not in {"standard", "DA", "DO", "DADO"}:
		setup = "standard"
	return dataset, model, setup


def remap_dataset_to_uq_name(dataset: str) -> str:
	mapping = {
		"amos22": "amos2022",
		"dermamnist-e-ood": "dermamnist-e-external",
	}
	return mapping.get(dataset, dataset)


def load_classification_balanced_accuracy(workspace_root: Path) -> Dict[Tuple[str, str, str, str], float]:
	cls_root = workspace_root / "Benchmarks" / "medMNIST" / "results" / "classification_results"
	shift_to_folder = {
		"in_distribution": "in_distribution",
		"corruption_shifts": "corruption_shifts",
		"population_shifts": "population_shift",
	}

	result: Dict[Tuple[str, str, str, str], float] = {}
	for shift_key, folder_name in shift_to_folder.items():
		folder = cls_root / folder_name
		if not folder.exists():
			continue

		for json_path in folder.glob("*.json"):
			name = json_path.name
			if shift_key == "corruption_shifts":
				name_core = name.replace(".json", "").replace("_severity3", "")
			else:
				name_core = name.replace("comprehensive_metrics_", "").replace(".json", "")

			parsed = parse_dataset_model_setup(name_core)
			if parsed is None:
				continue
			dataset, model, setup = parsed
			dataset = remap_dataset_to_uq_name(dataset)

			try:
				with json_path.open("r", encoding="utf-8") as f:
					payload = json.load(f)
			except (OSError, json.JSONDecodeError):
				continue

			bal_acc = None
			if isinstance(payload.get("ensemble_metrics"), dict):
				bal_acc = payload["ensemble_metrics"].get("balanced_accuracy")
			if bal_acc is None and isinstance(payload.get("ensemble"), dict):
				bal_acc = payload["ensemble"].get("balanced_accuracy")
			if bal_acc is None:
				continue

			key = (shift_key, dataset, model, setup)
			result[key] = float(bal_acc)

	return result


def collect_scatter_points(workspace_root: Path):
	uq_root = workspace_root / "Benchmarks" / "medMNIST" / "results" / "jsons_results"
	cls_index = load_classification_balanced_accuracy(workspace_root)

	individual_x = []
	individual_y = []
	agg_x = []
	agg_y = []

	for shift_key in ["in_distribution", "corruption_shifts", "population_shifts"]:
		shift_folder = uq_root / shift_key
		if not shift_folder.exists():
			continue

		for json_path in shift_folder.glob("*.json"):
			try:
				with json_path.open("r", encoding="utf-8") as f:
					payload = json.load(f)
			except (OSError, json.JSONDecodeError):
				continue

			dataset = payload.get("flag")
			model = payload.get("model_backbone")
			setup = normalize_setup(payload.get("setup"))
			if not dataset or not model:
				continue

			key = (shift_key, str(dataset), str(model), setup)
			bal_acc = cls_index.get(key)
			if bal_acc is None:
				continue

			methods = payload.get("methods", {})
			if not isinstance(methods, dict):
				continue

			for method_name, method_data in methods.items():
				if not isinstance(method_data, dict):
					continue
				if method_name == "ZScore_Aggregation_ensemble":
					continue
				if method_name.startswith("ZScore_Aggregation") or method_name == "Calibration_ZScore_Stats":
					continue

				if method_name == "Ensembling":
					score = method_data.get("auroc_f")
				else:
					score = method_data.get("auroc_f_mean")

				if score is None:
					continue

				individual_x.append(bal_acc)
				individual_y.append(float(score))

			agg_score = methods.get("ZScore_Aggregation_ensemble", {}).get("auroc_f")
			if agg_score is not None:
				agg_x.append(bal_acc)
				agg_y.append(float(agg_score))

	return individual_x, individual_y, agg_x, agg_y


def create_plot(workspace_root: Path) -> Path:
	individual_x, individual_y, agg_x, agg_y = collect_scatter_points(workspace_root)
	if not individual_x and not agg_x:
		raise RuntimeError("No valid points found. Check result file availability and naming.")

	fig, ax = plt.subplots(figsize=(10, 10))
	label_fs = 18
	title_fs = 20
	tick_fs = 15
	legend_fs = 14
	annotation_fs = 14

	if individual_x:
		ax.scatter(
			individual_x,
			individual_y,
			c="#6f6f6f",
			alpha=0.55,
			s=20,
			edgecolors="none",
			label="Individual Methods",
		)

	if agg_x:
		ax.scatter(
			agg_x,
			agg_y,
			marker="$⚡$",
			c="#f4c97a",
			s=170,
			alpha=0.62,
			edgecolors="black",
			linewidths=0.3,
			label="Mean Agg + Ens",
		)

	# Fit and draw global regression across all displayed points.
	x_all = np.asarray(individual_x + agg_x, dtype=float)
	y_all = np.asarray(individual_y + agg_y, dtype=float)
	if x_all.size >= 2 and np.std(x_all) > 0:
		slope, intercept = np.polyfit(x_all, y_all, 1)
		r = float(np.corrcoef(x_all, y_all)[0, 1])
		x_line = np.linspace(float(np.min(x_all)), float(np.max(x_all)), 100)
		y_line = slope * x_line + intercept
		ax.plot(x_line, y_line, color="#2d2d2d", linestyle="-", linewidth=2.0, label="Linear fit")

		eq_text = f"y = {slope:.3f}x + {intercept:.3f}\nr = {r:.3f}"
		ax.text(
			0.02,
			0.98,
			eq_text,
			transform=ax.transAxes,
			ha="left",
			va="top",
			fontsize=annotation_fs,
			bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "#999999"},
		)

	ax.set_xlabel("Balanced Accuracy", fontweight="bold", fontsize=label_fs)
	ax.set_ylabel("AUROC-F", fontweight="bold", fontsize=label_fs)
	ax.set_title("Balanced Accuracy vs AUROC-F", fontweight="bold", fontsize=title_fs)
	ax.tick_params(axis="both", which="major", labelsize=tick_fs)
	ax.grid(True, linestyle="--", alpha=0.25)
	ax.legend(loc="best", frameon=True, fontsize=legend_fs)

	output_path = (
		workspace_root
		/ "Benchmarks"
		/ "medMNIST"
		/ "results"
		/ "figures"
		/ "auroc_f_vs_balanced_acc_scatter.png"
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)

	print(f"Saved figure: {output_path}")
	print(f"Individual points: {len(individual_x)}")
	print(f"Mean Agg + Ens points: {len(agg_x)}")
	return output_path


def main() -> None:
	script_dir = Path(__file__).resolve().parent
	workspace_root = script_dir.parents[3]
	create_plot(workspace_root)


if __name__ == "__main__":
	main()
