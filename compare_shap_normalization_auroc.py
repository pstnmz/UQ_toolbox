"""
Loop over every cached KNN-SHAP fold (shap_cache_*.npz) and compute pooled RAW (unnormalized)
KNN-distance AUROC, comparing adaptive SHAP-mass feature selection at several cumulative
importance thresholds (default 70/80/90%) instead of a fixed top-50 cutoff.

Optionally (--enable-early-layer) also computes a second KNN block on an early conv layer's
activation maps: per-channel PyRadiomics first-order features -> concatenated per-image vector
-> StandardScaler+PCA(80%) -> KNN, then combines it with the penultimate-layer KNN score.
This block runs on the full train/test set and is reported/cached separately from the
mass-threshold comparison above.

Usage:
    python compare_shap_normalization_auroc.py
    python compare_shap_normalization_auroc.py --mass-thresholds 0.6 0.75 0.9
    python compare_shap_normalization_auroc.py --shap-cache-dir /path --output results.csv
    python compare_shap_normalization_auroc.py --enable-early-layer --early-layer-name layer1
"""
import argparse
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from Benchmarks.medMNIST.utils import train_models_load_datasets as tr
from Benchmarks.medMNIST.utils.data_preprocessing_classification_evaluation import dataset_utils
from ToolBox.methods.latent import (
    extract_latent_space_and_compute_shap_importance,
    get_layer_from_model,
)

DEFAULT_SHAP_CACHE_DIR = '/workspace/uq_benchmark_results/shap_cache'
DEFAULT_RESULTS_CACHE_DIR = '/workspace/uq_benchmark_results/cache'
DEFAULT_RADIOMICS_CACHE_DIR = '/workspace/uq_benchmark_results/radiomics_cache'
IMAGE_SIZE = 224

# Longest-name-first so 'dermamnist-e-id' matches before the shorter 'dermamnist-e'/'dermamnist'.
KNOWN_FLAGS = sorted(
    ['breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist-e-external', 'dermamnist-e-id',
     'dermamnist-e', 'dermamnist', 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist'],
    key=len, reverse=True,
)
KNOWN_BACKBONES = ['vit_b_16', 'resnet18']
COLOR_FLAGS = {'dermamnist', 'dermamnist-e', 'pathmnist', 'bloodmnist', 'hmu-crc'}
# ponytail: amos2022/midog/hmu-crc use special external-test-set loading (see
# run_medmnist_benchmark.py); not supported here, skip them rather than reimplementing.
UNSUPPORTED_FLAGS = {'amos2022', 'midog', 'hmu-crc'}

SHAP_CACHE_RE = re.compile(r'^shap_cache_(?P<namespace>.+)_fold(?P<fold>\d+)_bg(?P<bg>\d+)\.npz$')


def parse_namespace(namespace):
    """Split '{flag}_{backbone}{_setup}' into (flag, backbone, setup) or None if unparseable."""
    for flag in KNOWN_FLAGS:
        prefix = flag + '_'
        if not namespace.startswith(prefix):
            continue
        remainder = namespace[len(prefix):]
        for backbone in KNOWN_BACKBONES:
            if remainder == backbone:
                return flag, backbone, ''
            if remainder.startswith(backbone + '_'):
                return flag, backbone, remainder[len(backbone) + 1:]
    return None


def _test_parse_namespace():
    """Minimal self-check for the naming-convention parser (no framework needed)."""
    assert parse_namespace('organamnist_resnet18') == ('organamnist', 'resnet18', '')
    assert parse_namespace('organamnist_resnet18_DA') == ('organamnist', 'resnet18', 'DA')
    assert parse_namespace('dermamnist-e-id_vit_b_16_DADO') == ('dermamnist-e-id', 'vit_b_16', 'DADO')
    assert parse_namespace('not_a_real_dataset_resnet18') is None


def discover_shap_cache_files(shap_cache_dir):
    """Group cached SHAP fold files by (flag, backbone, setup), keyed off filename only."""
    groups = {}
    for path in sorted(Path(shap_cache_dir).glob('shap_cache_*.npz')):
        m = SHAP_CACHE_RE.match(path.name)
        if not m:
            print(f" [SKIP] Unrecognized filename pattern: {path.name}")
            continue
        parsed = parse_namespace(m.group('namespace'))
        if parsed is None:
            print(f" [SKIP] Could not parse dataset/backbone/setup from: {path.name}")
            continue
        flag, backbone, setup = parsed
        if flag in UNSUPPORTED_FLAGS:
            print(f" [SKIP] {path.name}: {flag} requires special external test-set loading, unsupported here")
            continue
        key = (flag, backbone, setup)
        groups.setdefault(key, []).append((int(m.group('fold')), int(m.group('bg')), path))
    return groups


def load_train_features(shap_cache_path):
    """Get features_train/labels_train from the SHAP cache, or None if this cache was
    generated in shap_only mode (which skips train-feature extraction entirely)."""
    cache = np.load(shap_cache_path, allow_pickle=True)
    if 'features_train' in cache.files and 'labels_train' in cache.files:
        return cache['features_train'], cache['labels_train']
    return None


def pooled_auroc(correct_by_class, incorrect_by_class, classes):
    scores, labels_bin = [], []
    for c in classes:
        scores.append(correct_by_class[c])
        labels_bin.append(np.zeros(len(correct_by_class[c])))
        scores.append(incorrect_by_class[c])
        labels_bin.append(np.ones(len(incorrect_by_class[c])))
    scores = np.concatenate(scores)
    labels_bin = np.concatenate(labels_bin)
    if len(np.unique(labels_bin)) < 2:
        return float('nan')
    return roc_auc_score(labels_bin, scores)


def select_features_by_mass(shap_values, labels_calib, class_idx, mass_threshold):
    """Minimal top-k feature indices whose cumulative |SHAP| importance covers
    >= mass_threshold of this class's total mass (adaptive, vs. the fixed top-50 elsewhere)."""
    if shap_values.ndim == 3:
        class_shap = shap_values[:, :, class_idx]
    else:
        class_shap = shap_values[labels_calib == class_idx, :]
    mean_abs = np.mean(np.abs(class_shap), axis=0)
    order = np.argsort(mean_abs)[::-1]
    sorted_vals = mean_abs[order]
    total = sorted_vals.sum()
    if total <= 0:
        return order[:1], 1
    cumulative = np.cumsum(sorted_vals) / total
    k = int(np.searchsorted(cumulative, mass_threshold) + 1)
    k = max(1, min(k, len(order)))
    return order[:k], k


# ============================================================================
# EARLY-LAYER RADIOMICS KNN (opt-in via --enable-early-layer)
# ============================================================================

_FIRST_ORDER_SETTINGS = {'force2D': True, 'force2Ddimension': 0}
# Starting subset of first-order features: Entropy needs a discretized histogram (the
# expensive part), the rest are direct voxel-array statistics - restricting to these instead
# of enableAllFeatures() cuts per-channel cost ~4.5x (benchmarked: 6.6ms -> 1.5ms/channel).
_FIRST_ORDER_FEATURE_NAMES = ['Mean', 'Variance', 'Entropy', 'Skewness', 'Kurtosis', 'Energy']


def _radiomics_vector_for_image(channel_maps):
    """One image's [C, H, W] activation map -> concatenated PyRadiomics first-order
    features (Mean/Variance/Entropy/Skewness/Kurtosis/Energy per channel) as a single
    fixed-length vector. Runs in a worker process."""
    import SimpleITK as sitk
    from radiomics.firstorder import RadiomicsFirstOrder
    vec = []
    for c in range(channel_maps.shape[0]):
        chan = np.ascontiguousarray(channel_maps[c], dtype=np.float64)
        image = sitk.GetImageFromArray(chan)
        mask = sitk.GetImageFromArray(np.ones_like(chan, dtype=np.uint8))
        extractor = RadiomicsFirstOrder(image, mask, **_FIRST_ORDER_SETTINGS)
        extractor.disableAllFeatures()
        for name in _FIRST_ORDER_FEATURE_NAMES:
            extractor.enableFeatureByName(name)
        vec.extend(float(v) for v in extractor.execute().values())
    return np.asarray(vec, dtype=np.float64)


def compute_radiomics_features(activations, n_jobs=None):
    """[N, C, H, W] activations -> [N, C*6] first-order radiomics feature matrix.
    ponytail: PyRadiomics has no batched/vectorized path - each channel needs its own
    SimpleITK Image+Mask object (~1.5ms/channel with this restricted feature set), so this
    is still CPU-bound and scales with N*C. Parallelized across images via ProcessPoolExecutor;
    the on-disk cache avoids recomputing across runs."""
    n_jobs = n_jobs or max(1, (os.cpu_count() or 4) - 1)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        vectors = list(ex.map(_radiomics_vector_for_image, list(activations)))
    return np.stack(vectors)



def extract_early_layer_activations(model, data_loader, device, layer_name):
    """Hook layer_name and collect its raw [N, C, H, W] activations + labels for every
    batch in data_loader (no flattening - radiomics needs the 2D spatial map per channel)."""
    layer = get_layer_from_model(model, layer_name)
    model.eval()
    activations, labels_list = [], []

    def hook(module, inp, out):
        activations.append(out.detach().cpu())

    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                batch = (batch['image'], batch['label'])
            images = batch[0].to(device, non_blocking=True)
            labels_t = batch[1].view(-1).long()
            model(images)
            labels_list.extend(labels_t.numpy().tolist())
    handle.remove()
    return torch.cat(activations, dim=0).numpy(), np.array(labels_list)


def get_or_compute_radiomics(cache_dir, cache_key, model, data_loader, device, layer_name, n_jobs):
    """Load cached (features, labels) for this cache_key, or compute + persist them."""
    cache_path = Path(cache_dir) / f'radiomics_{cache_key}.npz'
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached['features'], cached['labels']
    activations, labels = extract_early_layer_activations(model, data_loader, device, layer_name)
    features = compute_radiomics_features(activations, n_jobs=n_jobs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=features, labels=labels)
    return features, labels


def compute_early_layer_and_combined_auroc(early_features_train, early_labels_train, early_features_test,
                                            true_labels, predicted_classes,
                                            penultimate_features_train, penultimate_labels_train,
                                            penultimate_features_test, shap_cache_path=None, mass_threshold=0.8):
    """Per class present in the test set: fit StandardScaler+PCA(80% var) on early-layer
    radiomics train features, KNN-distance the test set; separately compute the
    penultimate-layer KNN distance using features selected by SHAP-mass threshold
    (default 80%); combine the two via simple averaging (both are RAW, unnormalized distances).
    Returns pooled AUROC for early-only, penultimate-only, and combined."""
    early_df_train = pd.DataFrame(early_features_train)
    early_df_test = pd.DataFrame(early_features_test)
    pen_df_train = pd.DataFrame(penultimate_features_train)
    pen_df_test = pd.DataFrame(penultimate_features_test)

    # Load SHAP info for mass-threshold feature selection (penultimate layer only)
    pen_selected_features_per_class = None
    if shap_cache_path is not None:
        try:
            cache = np.load(shap_cache_path, allow_pickle=True)
            shap_values = cache['shap_values']
            labels_calib = cache['labels']
            if shap_values.ndim == 3 and len(np.unique(labels_calib)) == 2:
                shap_values = shap_values.squeeze(-1)
            # Pre-compute selected features for each class at the given mass threshold
            pen_selected_features_per_class = {}
            for c in range(len(cache['selected_features_per_class'])):
                mass_idx, k = select_features_by_mass(shap_values, labels_calib, c, mass_threshold)
                pen_selected_features_per_class[c] = mass_idx
        except Exception as e:
            print(f"  [WARN] Could not load SHAP features for mass-threshold selection: {e}; using all features")
            pen_selected_features_per_class = None

    early_correct, early_incorrect = {}, {}
    pen_correct, pen_incorrect = {}, {}
    comb_correct, comb_incorrect = {}, {}
    classes_present = []

    for c in np.unique(predicted_classes):
        early_train_mask = (early_labels_train == c)
        pen_train_mask = (penultimate_labels_train == c)
        pred_mask = (predicted_classes == c)
        if early_train_mask.sum() == 0 or pen_train_mask.sum() == 0 or pred_mask.sum() == 0:
            continue
        true_here = true_labels[pred_mask]

        e_scaler = StandardScaler()
        e_train_pca = PCA(n_components=0.8).fit(e_scaler.fit_transform(early_df_train[early_train_mask].values))
        e_train_std = e_scaler.transform(early_df_train[early_train_mask].values)
        e_train_low = e_train_pca.transform(e_train_std)
        e_knn = NearestNeighbors(n_neighbors=min(5, len(e_train_low)))
        e_knn.fit(e_train_low)
        e_test_low = e_train_pca.transform(e_scaler.transform(early_df_test[pred_mask].values))
        e_dist = e_knn.kneighbors(e_test_low)[0].mean(axis=1)

        # Penultimate layer: use mass-threshold-selected features if available
        if pen_selected_features_per_class is not None and c in pen_selected_features_per_class:
            pen_feats = pen_selected_features_per_class[c]
            pen_train_subset = pen_df_train[pen_train_mask].iloc[:, pen_feats].values
            pen_test_subset = pen_df_test[pred_mask].iloc[:, pen_feats].values
        else:
            pen_train_subset = pen_df_train[pen_train_mask].values
            pen_test_subset = pen_df_test[pred_mask].values

        p_scaler = StandardScaler()
        p_train_std = p_scaler.fit_transform(pen_train_subset)
        p_pca = PCA(n_components=0.9)
        p_train_low = p_pca.fit_transform(p_train_std)
        p_knn = NearestNeighbors(n_neighbors=min(5, len(p_train_low)))
        p_knn.fit(p_train_low)
        p_test_low = p_pca.transform(p_scaler.transform(pen_test_subset))
        p_dist = p_knn.kneighbors(p_test_low)[0].mean(axis=1)

        classes_present.append(c)
        early_correct[c], early_incorrect[c] = e_dist[true_here == c], e_dist[true_here != c]
        pen_correct[c], pen_incorrect[c] = p_dist[true_here == c], p_dist[true_here != c]
        combined = (e_dist + p_dist) / 2
        comb_correct[c], comb_incorrect[c] = combined[true_here == c], combined[true_here != c]

    if not classes_present:
        return None
    return {
        'n_classes': len(classes_present),
        'auroc_early': pooled_auroc(early_correct, early_incorrect, classes_present),
        'auroc_penultimate': pooled_auroc(pen_correct, pen_incorrect, classes_present),
        'auroc_combined': pooled_auroc(comb_correct, comb_incorrect, classes_present),
    }


def compute_metrics_for_fold(fold_idx, shap_cache_path, test_cache,
                              features_test, labels_test_fresh, features_train, labels_train,
                              mass_thresholds):
    """Pooled RAW (unnormalized) KNN-distance AUROC per fold, comparing adaptive SHAP-mass
    feature selection at each of mass_thresholds (e.g. 0.7/0.8/0.9) plus a full-feature (no
    selection) baseline — no top-50 cutoff, no distance normalization."""
    cache = np.load(shap_cache_path, allow_pickle=True)
    selected_features_per_class = cache['selected_features_per_class']
    labels_calib = cache['labels']
    n_classes = len(selected_features_per_class)

    shap_values = cache['shap_values']
    if shap_values.ndim == 3 and len(np.unique(labels_calib)) == 2:
        shap_values = shap_values.squeeze(-1)

    labels_test = test_cache['y_true']
    predicted_classes = test_cache['per_fold_predictions'][fold_idx]
    if len(labels_test) != len(features_test):
        raise ValueError("cached test set size must match extracted features")

    train_df = pd.DataFrame(features_train, columns=[f"Feature_{i}" for i in range(features_train.shape[1])])
    test_df = pd.DataFrame(features_test.numpy(), columns=[f"Feature_{i}" for i in range(features_test.shape[1])])
    all_feats = list(train_df.columns)

    # 'all' = every feature, no SHAP-based selection; mass thresholds = adaptive selection.
    modes = ['all'] + list(mass_thresholds)
    correct_by_mode = {m: {} for m in modes}
    incorrect_by_mode = {m: {} for m in modes}
    n_features_by_mode = {m: {} for m in modes}
    classes_present = []
    skipped_classes = {}  # class -> reason it couldn't contribute to pooled AUROC this fold

    for c in range(n_classes):
        train_mask = (labels_train == c)
        if train_mask.sum() == 0:
            skipped_classes[c] = 'no_train_samples'
            continue

        pred_mask = (predicted_classes == c)
        if pred_mask.sum() == 0:
            skipped_classes[c] = 'never_predicted'
            continue
        true_here = labels_test[pred_mask]
        # No per-class both-outcomes requirement: with RAW (unnormalized) distances there's no
        # per-class statistic to compute, so a class that's always right/wrong this fold still
        # contributes valid distances to whichever side of the pooled set it has samples for.
        # pooled_auroc() only needs the FULL pooled set (across all classes) to have both labels.

        classes_present.append(c)
        for mode in modes:
            if mode == 'all':
                feats = all_feats
                n_features_by_mode[mode][c] = len(feats)
            else:
                mass_idx, k = select_features_by_mass(shap_values, labels_calib, c, mode)
                feats = [f"Feature_{i}" for i in mass_idx]
                n_features_by_mode[mode][c] = k

            scaler = StandardScaler()
            train_std = scaler.fit_transform(train_df[train_mask][feats].values)
            pca = PCA(n_components=0.9)
            train_pca = pca.fit_transform(train_std)
            knn = NearestNeighbors(n_neighbors=min(5, len(train_pca)))
            knn.fit(train_pca)

            test_pca = pca.transform(scaler.transform(test_df[pred_mask][feats].values))
            distances, _ = knn.kneighbors(test_pca)
            avg_distances = distances.mean(axis=1)
            correct_by_mode[mode][c] = avg_distances[true_here == c]
            incorrect_by_mode[mode][c] = avg_distances[true_here != c]

    if not classes_present:
        return None

    result = {'n_classes': len(classes_present), 'n_classes_total': n_classes,
              'skipped_classes': skipped_classes}
    for mode in modes:
        key = 'all' if mode == 'all' else f'mass{int(mode * 100)}'
        result[f'auroc_{key}'] = pooled_auroc(correct_by_mode[mode], incorrect_by_mode[mode], classes_present)
        result[f'avg_n_features_{key}'] = float(np.mean([n_features_by_mode[mode][c] for c in classes_present]))
        result[f'n_features_{key}_by_class'] = {c: n_features_by_mode[mode][c] for c in classes_present}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shap-cache-dir', default=DEFAULT_SHAP_CACHE_DIR)
    parser.add_argument('--results-cache-dir', default=DEFAULT_RESULTS_CACHE_DIR)
    parser.add_argument('--output', default='/workspace/uq_benchmark_results/knn_shap_normalization_comparison.csv')
    parser.add_argument('--mass-thresholds', type=float, nargs='+', default=[0.2, 0.5, 0.7, 0.8, 0.9],
                        help='Cumulative |SHAP| importance mass thresholds to compare, RAW '
                             'distances only (default: 0.7 0.8 0.9). A full-feature (no '
                             'selection) baseline is always included alongside these.')
    parser.add_argument('--enable-early-layer', action='store_true', default=False,
                        help='Also compute a KNN block on an early conv layer (PyRadiomics '
                             'first-order features per channel) and combine it with the '
                             'penultimate-layer KNN score. CPU-bound; runs on the full '
                             'train/test set, cached to disk.')
    parser.add_argument('--early-layer-name', default='layer1')
    parser.add_argument('--radiomics-n-jobs', type=int, default=None,
                        help='Worker processes for PyRadiomics extraction (default: cpu_count - 1).')
    parser.add_argument('--radiomics-cache-dir', default=DEFAULT_RADIOMICS_CACHE_DIR)
    parser.add_argument('--early-layer-output', default='/workspace/uq_benchmark_results/knn_early_layer_combined.csv')
    args = parser.parse_args()
    mass_thresholds = sorted(args.mass_thresholds)

    _test_parse_namespace()

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    groups = discover_shap_cache_files(args.shap_cache_dir)
    print(f"Found {sum(len(v) for v in groups.values())} SHAP cache files across {len(groups)} "
          f"(dataset, backbone, setup) groups\n")

    results_cache_dir = Path(args.results_cache_dir)
    rows = []
    early_rows = []
    dataset_cache = {}  # (base_flag, test_subset) -> (study_dataset, test_dataset, test_loader)
    early_train_loader_cache = {}  # (base_flag, test_subset) -> full-study-set DataLoader for early-layer extraction

    for (flag, backbone, setup), fold_entries in groups.items():
        if flag !='dermamnist-e-id':
            continue
        else:
            setup_suffix = f'_{setup}' if setup else ''
            namespace = f'{flag}_{backbone}{setup_suffix}'
            test_cache_path = results_cache_dir / f'{namespace}_test_results.npz'
            if not test_cache_path.exists():
                print(f" [SKIP] {namespace}: missing test result cache in {results_cache_dir}")
                continue

            # dermamnist-e-id/external are test-set subsets of the 'dermamnist-e' models/study;
            # models and datasets are keyed by the base flag, only the test subset differs.
            base_flag, test_subset = flag, 'all'
            if flag == 'dermamnist-e-id':
                base_flag, test_subset = 'dermamnist-e', 'id'
            elif flag == 'dermamnist-e-external':
                base_flag, test_subset = 'dermamnist-e', 'external'

            print(f"=== {namespace} ({len(fold_entries)} fold(s)) ===")
            try:
                models = tr.load_models(base_flag, device=device, size=IMAGE_SIZE, model_backbone=backbone, setup=setup)
                color = base_flag in COLOR_FLAGS
                dataset_key = (base_flag, test_subset)
                if dataset_key not in dataset_cache:
                    transform, _ = dataset_utils.get_transforms(color, IMAGE_SIZE)
                    [study_dataset, _, test_dataset], [_, _, test_loader], _ = tr.load_datasets(
                        base_flag, color, IMAGE_SIZE, transform, batch_size=256, test_subset=test_subset
                    )
                    dataset_cache[dataset_key] = (study_dataset, test_dataset, test_loader)
                study_dataset, test_dataset, test_loader = dataset_cache[dataset_key]

                if args.enable_early_layer and dataset_key not in early_train_loader_cache:
                    early_train_loader_cache[dataset_key] = DataLoader(study_dataset, batch_size=64, shuffle=False)
            except Exception as e:
                print(f" [SKIP] {namespace}: failed to load models/dataset ({e})")
                continue

            test_cache = np.load(test_cache_path, allow_pickle=True)
            group_rows = []
            group_early_rows = []

            for fold_idx, bg_size, shap_cache_path in sorted(fold_entries):
                if fold_idx >= len(models):
                    print(f" [SKIP] fold {fold_idx}: only {len(models)} models loaded")
                    continue
                train_data = load_train_features(shap_cache_path)
                if train_data is None:
                    print(f" [SKIP] fold {fold_idx} (bg{bg_size}): cache has no features_train "
                        f"(generated with --shap-only, not a full run)")
                    continue
                features_train, labels_train = train_data
                try:
                    model_fold = models[fold_idx]
                    model_fold.eval()
                    layer = get_layer_from_model(model_fold, 'avgpool')
                    features_test, labels_test_fresh, _, _ = extract_latent_space_and_compute_shap_importance(
                        model_fold, test_loader, device, layer, importance=False
                    )
                    metrics = compute_metrics_for_fold(
                        fold_idx, shap_cache_path, test_cache,
                        features_test, labels_test_fresh, features_train, labels_train,
                        mass_thresholds
                    )
                except Exception as e:
                    print(f" [FAIL] fold {fold_idx} (bg{bg_size}): {e}")
                    continue

                if metrics is None:
                    print(f" [SKIP] fold {fold_idx}: no classes with both correct and incorrect test predictions")
                    continue

                print(f"  fold {fold_idx} (bg{bg_size}, {metrics['n_classes']}/{metrics['n_classes_total']} classes):")
                if metrics['skipped_classes']:
                    skipped_info = ', '.join(f"C{c}={reason}" for c, reason in sorted(metrics['skipped_classes'].items()))
                    print(f"    skipped {len(metrics['skipped_classes'])} class(es) this fold: {skipped_info}")
                for key, label in [('all', 'all features')] + [(f'mass{int(t*100)}', f'mass-{int(t*100)}%') for t in mass_thresholds]:
                    by_class = metrics[f'n_features_{key}_by_class']
                    feats_info = ', '.join(f"C{c}={k}" for c, k in sorted(by_class.items()))
                    print(f"    {label}: RAW AUROC={metrics[f'auroc_{key}']:.4f}  "
                        f"avg_features={metrics[f'avg_n_features_{key}']:.1f}  ({feats_info})")
                row = {'flag': flag, 'backbone': backbone, 'setup': setup or 'standard',
                    'fold': fold_idx, 'bg_size': bg_size,
                    **{k: v for k, v in metrics.items()
                        if not k.endswith('_by_class') and k != 'skipped_classes'}}
                rows.append(row)
                group_rows.append(row)

                if args.enable_early_layer:
                    train_loader_full = early_train_loader_cache[dataset_key]
                    cache_key_train = f'{namespace}_fold{fold_idx}_{args.early_layer_name}_train'
                    cache_key_test = f'{namespace}_fold{fold_idx}_{args.early_layer_name}_test'
                    try:
                        early_features_train, early_labels_train = get_or_compute_radiomics(
                            args.radiomics_cache_dir, cache_key_train, model_fold, train_loader_full,
                            device, args.early_layer_name, args.radiomics_n_jobs
                        )
                        early_features_test, _ = get_or_compute_radiomics(
                            args.radiomics_cache_dir, cache_key_test, model_fold, test_loader,
                            device, args.early_layer_name, args.radiomics_n_jobs
                        )
                        predicted_classes_full = test_cache['per_fold_predictions'][fold_idx]
                        true_labels_full = test_cache['y_true']
                        pen_features_test_full = features_test.numpy()
                        early_metrics = compute_early_layer_and_combined_auroc(
                            early_features_train, early_labels_train, early_features_test,
                            true_labels_full, predicted_classes_full,
                            features_train, labels_train, pen_features_test_full,
                            shap_cache_path, mass_threshold=0.8
                        )
                    except Exception as e:
                        print(f"    [FAIL] early-layer block for fold {fold_idx}: {e}")
                        early_metrics = None

                    if early_metrics is not None:
                        print(f"    early-layer ({args.early_layer_name}, {early_metrics['n_classes']} classes): "
                            f"early={early_metrics['auroc_early']:.4f}  "
                            f"penultimate={early_metrics['auroc_penultimate']:.4f}  "
                            f"combined={early_metrics['auroc_combined']:.4f}")
                        early_row = {
                            'flag': flag, 'backbone': backbone, 'setup': setup or 'standard', 'fold': fold_idx,
                            **early_metrics
                        }
                        early_rows.append(early_row)
                        group_early_rows.append(early_row)

            if group_rows:
                group_df = pd.DataFrame(group_rows)
                keys = ['all'] + [f'mass{int(t*100)}' for t in mass_thresholds]
                metric_cols = [f'auroc_{k}' for k in keys] + [f'avg_n_features_{k}' for k in keys]
                means = group_df[metric_cols].mean()
                print(f"  MEAN ({len(group_rows)} fold(s)):")
                for key, label in [('all', 'all features')] + [(f'mass{int(t*100)}', f'mass-{int(t*100)}%') for t in mass_thresholds]:
                    print(f"    {label}: RAW AUROC={means[f'auroc_{key}']:.4f}  "
                        f"avg_features={means[f'avg_n_features_{key}']:.1f}")
                
                # Early-layer mean (if enabled and present for this group)
                if group_early_rows:
                    early_group_df = pd.DataFrame(group_early_rows)
                    early_means = early_group_df[['auroc_early', 'auroc_penultimate', 'auroc_combined']].mean()
                    print(f"  EARLY-LAYER MEAN ({len(group_early_rows)} fold(s)):")
                    print(f"    early={early_means['auroc_early']:.4f}  "
                        f"penultimate={early_means['auroc_penultimate']:.4f}  "
                        f"combined={early_means['auroc_combined']:.4f}")
                print()

    if not rows:
        print("\nNo results computed (no matching cache files found or all skipped).")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("SUMMARY (pooled AUROC per fold)")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nMean across all folds:")
    keys = ['all'] + [f'mass{int(t*100)}' for t in mass_thresholds]
    summary_cols = [f'auroc_{k}' for k in keys] + [f'avg_n_features_{k}' for k in keys]
    print(df[summary_cols].mean().to_string(float_format=lambda x: f"{x:.4f}"))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved full results to {args.output}")

    if early_rows:
        early_df = pd.DataFrame(early_rows)
        print("\n" + "=" * 100)
        print(f"EARLY-LAYER ({args.early_layer_name}) + COMBINED SUMMARY")
        print("=" * 100)
        print(early_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nMean across all folds:")
        print(early_df[['auroc_early', 'auroc_penultimate', 'auroc_combined']]
              .mean().to_string(float_format=lambda x: f"{x:.4f}"))
        Path(args.early_layer_output).parent.mkdir(parents=True, exist_ok=True)
        early_df.to_csv(args.early_layer_output, index=False)
        print(f"\nSaved early-layer results to {args.early_layer_output}")


if __name__ == '__main__':
    main()
