import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


AUROC_MEAN_METHODS = [
	"MSR",
	"MSR_calibrated",
	"MLS",
	"GPS",
	"KNN_Raw",
	"ZScore_Aggregation_per_fold",
]

AUROC_DIRECT_METHODS = ["Ensembling", "ZScore_Aggregation_ensemble"]

SHIFT_DIRS = {
	"ID": "in_distribution",
	"corruption shifts": "corruption_shifts",
	"population shifts": "population_shifts",
	"new class shifts": "new_class_shifts",
}


@dataclass(frozen=True)
class RunRecord:
	shift_type: str
	folder_name: str
	file_path: Path
	timestamp: str
	dataset: str
	backbone: str
	setup: str
	best_auroc: float
	best_augrc: float


def parse_backbone(raw: str) -> str:
	low = raw.lower()
	if "vit" in low:
		return "vit_b_16"
	return "resnet18"


def best_auroc_and_augrc(methods: Dict) -> Tuple[float, float]:
	auroc_candidates: List[float] = []
	augrc_candidates: List[float] = []

	for method_name in AUROC_MEAN_METHODS:
		method_dict = methods.get(method_name)
		if not isinstance(method_dict, dict):
			continue
		auroc = method_dict.get("auroc_f_mean")
		augrc = method_dict.get("augrc_mean")
		if isinstance(auroc, (int, float)):
			auroc_candidates.append(float(auroc))
		if isinstance(augrc, (int, float)):
			augrc_candidates.append(float(augrc))

	for method_name in AUROC_DIRECT_METHODS:
		method_dict = methods.get(method_name)
		if not isinstance(method_dict, dict):
			continue
		auroc = method_dict.get("auroc_f")
		augrc = method_dict.get("augrc")
		if isinstance(auroc, (int, float)):
			auroc_candidates.append(float(auroc))
		if isinstance(augrc, (int, float)):
			augrc_candidates.append(float(augrc))

	if not auroc_candidates or not augrc_candidates:
		raise ValueError("No valid AUROC/AUGRC candidates found in methods dictionary")

	return max(auroc_candidates), min(augrc_candidates)


def format_stats(values: List[float]) -> str:
	if not values:
		return "n=0"
	mu = mean(values)
	sd = pstdev(values) if len(values) > 1 else 0.0
	return f"{mu:+.6f} +/- {sd:.6f} (n={len(values)})"


def load_latest_records(json_root: Path) -> List[RunRecord]:
	latest: Dict[Tuple[str, str, str, str], RunRecord] = {}

	for shift_type, folder in SHIFT_DIRS.items():
		folder_path = json_root / folder
		for file_path in sorted(folder_path.glob("*.json")):
			with file_path.open("r", encoding="utf-8") as f:
				payload = json.load(f)

			methods = payload.get("methods", {})
			best_auroc, best_augrc = best_auroc_and_augrc(methods)

			timestamp = str(payload.get("timestamp", ""))
			dataset = str(payload.get("flag", "")).strip()
			setup = str(payload.get("setup", "")).strip() or "base"
			backbone = parse_backbone(str(payload.get("model_backbone", "")))

			rec = RunRecord(
				shift_type=shift_type,
				folder_name=folder,
				file_path=file_path,
				timestamp=timestamp,
				dataset=dataset,
				backbone=backbone,
				setup=setup,
				best_auroc=best_auroc,
				best_augrc=best_augrc,
			)

			key = (shift_type, dataset, setup, backbone)
			previous = latest.get(key)
			if previous is None or rec.timestamp > previous.timestamp:
				latest[key] = rec

	return list(latest.values())


def grouped(records: Iterable[RunRecord]) -> Dict[Tuple[str, str, str], Dict[str, RunRecord]]:
	buckets: Dict[Tuple[str, str, str], Dict[str, RunRecord]] = {}
	for rec in records:
		key = (rec.shift_type, rec.dataset, rec.setup)
		buckets.setdefault(key, {})[rec.backbone] = rec
	return buckets


def vit_minus_resnet_by_shift(records: List[RunRecord], metric_name: str) -> Dict[str, List[float]]:
	diffs_by_shift: Dict[str, List[float]] = {k: [] for k in SHIFT_DIRS}
	for (shift_type, _dataset, _setup), by_backbone in grouped(records).items():
		if "vit_b_16" not in by_backbone or "resnet18" not in by_backbone:
			continue
		vit = by_backbone["vit_b_16"]
		res = by_backbone["resnet18"]
		v = vit.best_auroc if metric_name == "auroc" else vit.best_augrc
		r = res.best_auroc if metric_name == "auroc" else res.best_augrc
		diffs_by_shift[shift_type].append(v - r)
	return diffs_by_shift


def id_minus_corruption_by_backbone(records: List[RunRecord], metric_name: str) -> Dict[str, List[float]]:
	by_key: Dict[Tuple[str, str, str], Dict[str, RunRecord]] = {}
	for rec in records:
		key = (rec.dataset, rec.setup, rec.backbone)
		by_key.setdefault(key, {})[rec.shift_type] = rec

	out: Dict[str, List[float]] = {"resnet18": [], "vit_b_16": []}
	for (_dataset, _setup, backbone), shift_map in by_key.items():
		if "ID" not in shift_map or "corruption shifts" not in shift_map:
			continue
		id_rec = shift_map["ID"]
		corr_rec = shift_map["corruption shifts"]
		id_val = id_rec.best_auroc if metric_name == "auroc" else id_rec.best_augrc
		corr_val = corr_rec.best_auroc if metric_name == "auroc" else corr_rec.best_augrc
		out[backbone].append(id_val - corr_val)
	return out


def cross_dataset_pair_diffs(
	records: List[RunRecord],
	id_dataset: str,
	shift_dataset: str,
	shift_type: str,
	metric_name: str,
) -> Dict[str, List[float]]:
	by_exact: Dict[Tuple[str, str, str, str], RunRecord] = {}
	for rec in records:
		key = (rec.shift_type, rec.dataset, rec.setup, rec.backbone)
		by_exact[key] = rec

	out: Dict[str, List[float]] = {"resnet18": [], "vit_b_16": []}
	for backbone in ["resnet18", "vit_b_16"]:
		setups = {r.setup for r in records if r.backbone == backbone}
		for setup in sorted(setups):
			id_key = ("ID", id_dataset, setup, backbone)
			sh_key = (shift_type, shift_dataset, setup, backbone)
			if id_key not in by_exact or sh_key not in by_exact:
				continue
			id_rec = by_exact[id_key]
			sh_rec = by_exact[sh_key]
			id_val = id_rec.best_auroc if metric_name == "auroc" else id_rec.best_augrc
			sh_val = sh_rec.best_auroc if metric_name == "auroc" else sh_rec.best_augrc
			out[backbone].append(id_val - sh_val)

	return out


def arbitrary_pair_diffs(
	records: List[RunRecord],
	shift_type_a: str,
	dataset_a: str,
	shift_type_b: str,
	dataset_b: str,
	metric_name: str,
) -> Dict[str, List[float]]:
	"""Compute (metric_a - metric_b) per backbone, matched by setup."""
	by_exact: Dict[Tuple[str, str, str, str], RunRecord] = {}
	for rec in records:
		key = (rec.shift_type, rec.dataset, rec.setup, rec.backbone)
		by_exact[key] = rec

	out: Dict[str, List[float]] = {"resnet18": [], "vit_b_16": []}
	for backbone in ["resnet18", "vit_b_16"]:
		setups = {r.setup for r in records if r.backbone == backbone}
		for setup in sorted(setups):
			key_a = (shift_type_a, dataset_a, setup, backbone)
			key_b = (shift_type_b, dataset_b, setup, backbone)
			if key_a not in by_exact or key_b not in by_exact:
				continue
			rec_a = by_exact[key_a]
			rec_b = by_exact[key_b]
			val_a = rec_a.best_auroc if metric_name == "auroc" else rec_a.best_augrc
			val_b = rec_b.best_auroc if metric_name == "auroc" else rec_b.best_augrc
			out[backbone].append(val_a - val_b)
	return out


def mean_by_backbone(
	records: List[RunRecord],
	shift_type: str,
	dataset: str,
	metric_name: str,
) -> Dict[str, List[float]]:
	"""Collect per-backbone metric values for a specific (shift_type, dataset)."""
	out: Dict[str, List[float]] = {"resnet18": [], "vit_b_16": []}
	for rec in records:
		if rec.shift_type == shift_type and rec.dataset == dataset:
			val = rec.best_auroc if metric_name == "auroc" else rec.best_augrc
			out[rec.backbone].append(val)
	return out


def format_abs(values: List[float]) -> str:
	if not values:
		return "n=0"
	mu = mean(values)
	sd = pstdev(values) if len(values) > 1 else 0.0
	return f"{mu:.6f} +/- {sd:.6f} (n={len(values)})"


def print_absolute_table(title: str, values: Dict[str, List[float]]) -> None:
	print(f"\n{title}")
	for backbone in ["resnet18", "vit_b_16"]:
		print(f"- {backbone}: {format_abs(values.get(backbone, []))}")


def print_shift_table(title: str, diffs: Dict[str, List[float]]) -> None:
	print(f"\n{title}")
	for shift_type in ["ID", "corruption shifts", "population shifts", "new class shifts"]:
		print(f"- {shift_type}: {format_stats(diffs.get(shift_type, []))}")


def print_backbone_table(title: str, diffs: Dict[str, List[float]]) -> None:
	print(f"\n{title}")
	for backbone in ["resnet18", "vit_b_16"]:
		print(f"- {backbone}: {format_stats(diffs.get(backbone, []))}")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--json-root",
		type=Path,
		default=Path("Benchmarks/medMNIST/results/jsons_results"),
		help="Root directory with in_distribution/corruption_shifts/population_shifts/new_class_shifts",
	)
	args = parser.parse_args()

	records = load_latest_records(args.json_root)

	print(f"Loaded latest comparable records: {len(records)}")
	print("Selection rule: keep latest timestamp per (shift_type, dataset, setup, backbone).")

	diff_vr_auroc = vit_minus_resnet_by_shift(records, metric_name="auroc")
	print_shift_table(
		"ViT - ResNet best-CSF AUROC-F difference by shift type:",
		diff_vr_auroc,
	)

	diff_vr_augrc = vit_minus_resnet_by_shift(records, metric_name="augrc")
	print_shift_table(
		"ViT - ResNet best-CSF AUGRC difference by shift type (lower is better):",
		diff_vr_augrc,
	)

	diff_idcorr_auroc = id_minus_corruption_by_backbone(records, metric_name="auroc")
	print_backbone_table(
		"ID - Corruption best-CSF AUROC-F difference by backbone:",
		diff_idcorr_auroc,
	)

	diff_idcorr_augrc = id_minus_corruption_by_backbone(records, metric_name="augrc")
	print_backbone_table(
		"ID - Corruption best-CSF AUGRC difference by backbone (lower is better):",
		diff_idcorr_augrc,
	)

	derm_auroc = cross_dataset_pair_diffs(
		records,
		id_dataset="dermamnist-e-id",
		shift_dataset="dermamnist-e-external",
		shift_type="population shifts",
		metric_name="auroc",
	)
	print_backbone_table(
		"dermamnist-e-id (ID) - dermamnist-e-external (population shifts) AUROC-F:",
		derm_auroc,
	)

	derm_augrc = cross_dataset_pair_diffs(
		records,
		id_dataset="dermamnist-e-id",
		shift_dataset="dermamnist-e-external",
		shift_type="population shifts",
		metric_name="augrc",
	)
	print_backbone_table(
		"dermamnist-e-id (ID) - dermamnist-e-external (population shifts) AUGRC:",
		derm_augrc,
	)

	organ_auroc = cross_dataset_pair_diffs(
		records,
		id_dataset="organamnist",
		shift_dataset="amos2022",
		shift_type="population shifts",
		metric_name="auroc",
	)
	print_backbone_table(
		"organamnist (ID) - amos2022 (population shifts) AUROC-F:",
		organ_auroc,
	)

	organ_augrc = cross_dataset_pair_diffs(
		records,
		id_dataset="organamnist",
		shift_dataset="amos2022",
		shift_type="population shifts",
		metric_name="augrc",
	)
	print_backbone_table(
		"organamnist (ID) - amos2022 (population shifts) AUGRC:",
		organ_augrc,
	)

	# ── AMOS2022: OOD detection (new class shifts) vs failure detection (population shifts) ──
	print("\n" + "=" * 70)
	print("AMOS2022 — OOD detection vs failure detection")
	print("=" * 70)

	amos_ncs_auroc = mean_by_backbone(records, "new class shifts", "amos2022", "auroc")
	print_absolute_table(
		"amos2022 new class shifts best-CSF AUROC-F (OOD detection):",
		amos_ncs_auroc,
	)

	amos_ps_auroc = mean_by_backbone(records, "population shifts", "amos2022", "auroc")
	print_absolute_table(
		"amos2022 population shifts best-CSF AUROC-F (failure detection on known labels):",
		amos_ps_auroc,
	)

	amos_ncs_vs_ps = arbitrary_pair_diffs(
		records,
		shift_type_a="new class shifts", dataset_a="amos2022",
		shift_type_b="population shifts", dataset_b="amos2022",
		metric_name="auroc",
	)
	print_backbone_table(
		"amos2022 new class shifts − population shifts AUROC-F (OOD − failure detection):",
		amos_ncs_vs_ps,
	)

	# ── MIDOG (OOD) vs PathMNIST ID (failure detection) ──
	print("\n" + "=" * 70)
	print("MIDOG (OOD) vs PathMNIST in-distribution (failure detection)")
	print("=" * 70)

	midog_ncs_auroc = mean_by_backbone(records, "new class shifts", "midog", "auroc")
	print_absolute_table(
		"midog new class shifts best-CSF AUROC-F (OOD detection):",
		midog_ncs_auroc,
	)

	pathmnist_id_auroc = mean_by_backbone(records, "ID", "pathmnist", "auroc")
	print_absolute_table(
		"pathmnist ID best-CSF AUROC-F (failure detection):",
		pathmnist_id_auroc,
	)

	midog_vs_pathmnist = arbitrary_pair_diffs(
		records,
		shift_type_a="new class shifts", dataset_a="midog",
		shift_type_b="ID",              dataset_b="pathmnist",
		metric_name="auroc",
	)
	print_backbone_table(
		"midog (new class shifts) − pathmnist (ID) AUROC-F (OOD − failure detection):",
		midog_vs_pathmnist,
	)


if __name__ == "__main__":
	main()
