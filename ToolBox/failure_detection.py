"""
Failure detection and uncertainty quantification runner for FailCatcher.

This module provides a clean, dataset-agnostic API for detecting failures
and quantifying uncertainty in PyTorch models using various UQ methods.

Example:
    >>> from FailCatcher import failure_detection
    >>> 
    >>> detector = failure_detection.FailureDetector(
    ...     models=models,
    ...     study_dataset=train_dataset,
    ...     calib_dataset=calib_dataset,
    ...     test_dataset=test_dataset,
    ...     device='cuda'
    ... )
    >>> 
    >>> # Run MSR
    >>> results = detector.run_msr(y_scores_test, y_true_test)
    >>> 
    >>> # Run ensemble
    >>> results = detector.run_ensemble(indiv_scores_test)
    >>> 
    >>> # Cache augmentations for GPS
    >>> aug_folder = detector.run_augmentation_calibration_caching(
    ...     aug_folder='gps_augment/dataset_calib',
    ...     N=2, M=45, num_policies=500
    ... )
    >>> 
    >>> # Run GPS with cached augmentations
    >>> results = detector.run_gps(
    ...     test_dataset, y_true, aug_folder, 
    ...     correct_idx_calib, incorrect_idx_calib
    ... )
    >>> 
    >>> # Run KNN-SHAP
    >>> results = detector.run_knn_shap(
    ...     calib_loader=calib_loader,
    ...     test_loader=test_loader,
    ...     train_loaders=train_loaders
    ... )
"""

import os
import numpy as np
import torch
from typing import List, Callable, Optional, Dict, Any, Tuple, Union
from torch.utils.data import DataLoader
import time

from . import UQ_toolbox as uq


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.elapsed = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.name:
            print(f"{self.name}: {self.elapsed:.2f}s")


class FailureDetector:
    """
    Failure detection and uncertainty quantification runner.
    
    This class provides a dataset-agnostic interface for running and evaluating
    uncertainty quantification methods on PyTorch models to detect failures.
    
    Args:
        models: List of PyTorch models (e.g., CV fold models)
        study_dataset: Full training dataset (for CV splitting)
        calib_dataset: Calibration dataset
        test_dataset: Test dataset
        device: torch.device or str
        num_classes: Number of classes in classification task
        
    Example:
        >>> detector = FailureDetector(
        ...     models=fold_models,
        ...     study_dataset=train_data,
        ...     calib_dataset=calib_data,
        ...     test_dataset=test_data,
        ...     device='cuda'
        ... )
    """
    
    def __init__(
        self,
        models: List[torch.nn.Module],
        study_dataset: torch.utils.data.Dataset,
        calib_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        device: torch.device = None,
        num_classes: int = None
    ):
        self.models = models
        self.study_dataset = study_dataset
        self.calib_dataset = calib_dataset
        self.test_dataset = test_dataset
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Move models to device
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        # Cache for predictions (computed once)
        self._test_predictions_cache = None
        self._test_loader_cache = None
        self._per_fold_predictions_cache = None  # Cache per-fold predictions [M, N] to avoid recomputing
        
        # Storage for computed uncertainties and predictions
        self._uncertainties = {}
        self._uncertainties_normalized = {}  # Store normalized versions for fair comparison
        self._predictions_per_fold = {}  # Store per-fold predictions for accurate ROC curves
        self._results = {}
    
    def set_per_fold_predictions(
        self,
        per_fold_predictions: np.ndarray
    ):
        """
        Set pre-computed per-fold predictions to avoid vanilla inference.
        
        This is much more efficient than recomputing predictions for each UQ method.
        The predictions can be computed once during initial caching and reused.
        
        Args:
            per_fold_predictions: Per-fold predictions [M, N] where M = num_folds, N = num_samples
        """
        self._per_fold_predictions_cache = per_fold_predictions
        print(f" Set per-fold predictions cache {per_fold_predictions.shape}")
    
    def set_test_predictions(
        self,
        y_scores: np.ndarray,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        correct_idx: Optional[np.ndarray] = None,
        incorrect_idx: Optional[np.ndarray] = None,
        per_fold_correct_idx: Optional[list] = None,
        per_fold_incorrect_idx: Optional[list] = None
    ):
        """
        Set pre-computed test predictions to avoid recomputing.
        
        Args:
            y_scores: Ensemble probability scores [N, num_classes]
            y_true: True labels [N]
            y_pred: Predicted labels [N] (optional, will compute from y_scores if not provided)
            correct_idx: Pre-computed correct prediction indices (optional)
            incorrect_idx: Pre-computed incorrect prediction indices (optional)
            per_fold_correct_idx: List of correct indices for each fold (optional)
            per_fold_incorrect_idx: List of incorrect indices for each fold (optional)
        """
        if y_pred is None:
            y_pred = np.argmax(y_scores, axis=1)
        
        # Use pre-computed indices if provided, otherwise compute from y_pred vs y_true
        if correct_idx is None or incorrect_idx is None:
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        self._test_predictions_cache = {
            'y_scores': y_scores,
            'y_true': y_true,
            'y_pred': y_pred,
            'correct_idx': correct_idx,
            'incorrect_idx': incorrect_idx,
            'per_fold_correct_idx': per_fold_correct_idx,
            'per_fold_incorrect_idx': per_fold_incorrect_idx
        }

    def run_zscore_aggregation(
        self,
        method_names: Optional[List[str]] = None,
        score_dict: Optional[Dict[str, np.ndarray]] = None,
        means: Optional[Union[List[Any], Dict[str, Any]]] = None,
        stds: Optional[Union[List[Any], Dict[str, Any]]] = None,
        use_test_distribution: bool = True,
        aggregation_name: str = 'ZScore_Aggregation',
        mode: str = 'ensemble',
        predictions: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aggregate multiple uncertainty scores by z-scoring each score and averaging.

                Two normalization source modes are supported:
        - use_test_distribution=True: compute mean/std directly from each score array.
        - use_test_distribution=False: use externally provided means/stds.

                Aggregation modes:
                - mode='ensemble': build one score per method using method_ensemble when available,
                    otherwise mean(method_per_fold), otherwise method main score.
                - mode='per_fold': aggregate each fold separately (z-score + average over methods),
                    then average folds for the main score.

        The aggregated score and its metrics are stored in self._uncertainties and
        self._results under aggregation_name, so save_results() persists them in
        both NPZ and JSON outputs.
        """
        timer = Timer(f"{aggregation_name} computation")

        with timer:
            source_scores = self._uncertainties if score_dict is None else score_dict

            if mode not in ['ensemble', 'per_fold']:
                raise ValueError("mode must be either 'ensemble' or 'per_fold'")

            if method_names is None:
                method_names = []
                for key, value in source_scores.items():
                    if not isinstance(value, np.ndarray):
                        continue
                    if key.endswith('_per_fold') or key.endswith('_ensemble') or key.endswith('_per_fold_internal'):
                        continue
                    if mode == 'ensemble' and value.ndim == 1:
                        method_names.append(key)
                    elif mode == 'per_fold' and (value.ndim == 1 or f'{key}_per_fold' in source_scores):
                        method_names.append(key)

            if not method_names:
                raise ValueError("No methods provided for z-score aggregation")

            eps = 1e-10

            if mode == 'ensemble':
                ordered_scores = []
                for name in method_names:
                    ensemble_key = f'{name}_ensemble'
                    per_fold_key = f'{name}_per_fold'

                    if ensemble_key in source_scores:
                        arr = np.asarray(source_scores[ensemble_key])
                    elif per_fold_key in source_scores:
                        arr = np.asarray(source_scores[per_fold_key]).mean(axis=0)
                    elif name in source_scores:
                        arr = np.asarray(source_scores[name])
                    else:
                        raise ValueError(f"Method '{name}' not found in provided scores")

                    if arr.ndim != 1:
                        raise ValueError(
                            f"Method '{name}' must resolve to 1D scores [N], got shape {arr.shape}"
                        )
                    ordered_scores.append(arr)

                n_samples = len(ordered_scores[0])
                for idx, arr in enumerate(ordered_scores):
                    if len(arr) != n_samples:
                        raise ValueError(
                            f"All score arrays must have same length. "
                            f"'{method_names[idx]}' has length {len(arr)} != {n_samples}"
                        )

                resolved_means = []
                resolved_stds = []
                if use_test_distribution:
                    for arr in ordered_scores:
                        resolved_means.append(float(np.mean(arr)))
                        resolved_stds.append(float(np.std(arr)))
                else:
                    if means is None or stds is None:
                        raise ValueError(
                            "means and stds must be provided when use_test_distribution=False"
                        )

                    if isinstance(means, dict):
                        resolved_means = [float(means[name]) for name in method_names]
                    else:
                        if len(means) != len(method_names):
                            raise ValueError("means length must match method_names length")
                        resolved_means = [float(v) for v in means]

                    if isinstance(stds, dict):
                        resolved_stds = [float(stds[name]) for name in method_names]
                    else:
                        if len(stds) != len(method_names):
                            raise ValueError("stds length must match method_names length")
                        resolved_stds = [float(v) for v in stds]

                zscored = []
                for arr, mu, sigma in zip(ordered_scores, resolved_means, resolved_stds):
                    if abs(sigma) < eps:
                        z_arr = np.zeros_like(arr, dtype=float)
                    else:
                        z_arr = (arr - mu) / sigma
                    zscored.append(z_arr)

                zscored = np.stack(zscored, axis=0)  # [num_methods, N]
                aggregated_for_metrics = np.mean(zscored, axis=0)  # [N]
                aggregated_scores = aggregated_for_metrics

            else:
                # per_fold mode
                num_folds = None
                ordered_scores = []  # each entry [M, N]

                for name in method_names:
                    per_fold_key = f'{name}_per_fold'

                    if per_fold_key in source_scores:
                        arr = np.asarray(source_scores[per_fold_key])  # [M, N]
                    else:
                        raise ValueError(
                            f"Method '{name}' has no per-fold scores ('{per_fold_key}'). "
                            "Per-fold aggregation only supports methods with explicit per-fold outputs."
                        )

                    if arr.ndim != 2:
                        raise ValueError(
                            f"Method '{name}' must resolve to per-fold [M, N], got shape {arr.shape}"
                        )

                    if num_folds is None:
                        num_folds = arr.shape[0]
                    elif arr.shape[0] != num_folds:
                        raise ValueError(
                            f"All methods must have same fold count. "
                            f"'{name}' has {arr.shape[0]} folds != {num_folds}"
                        )

                    ordered_scores.append(arr)

                n_samples = ordered_scores[0].shape[1]
                for idx, arr in enumerate(ordered_scores):
                    if arr.shape[1] != n_samples:
                        raise ValueError(
                            f"All score arrays must have same sample count. "
                            f"'{method_names[idx]}' has {arr.shape[1]} != {n_samples}"
                        )

                # Means/stds per method and per fold: [num_methods, M]
                resolved_means = []
                resolved_stds = []

                if use_test_distribution:
                    for arr in ordered_scores:
                        resolved_means.append(np.mean(arr, axis=1))
                        resolved_stds.append(np.std(arr, axis=1))
                else:
                    if means is None or stds is None:
                        raise ValueError(
                            "means and stds must be provided when use_test_distribution=False"
                        )

                    def _resolve_external(values, name, idx):
                        if isinstance(values, dict):
                            raw = values[name]
                        else:
                            if len(values) != len(method_names):
                                raise ValueError("External values length must match method_names")
                            raw = values[idx]

                        raw_arr = np.asarray(raw, dtype=float)
                        if raw_arr.ndim == 0:
                            return np.full((num_folds,), float(raw_arr))
                        if raw_arr.ndim == 1 and len(raw_arr) == num_folds:
                            return raw_arr
                        raise ValueError(
                            f"External values for '{name}' must be scalar or length-{num_folds}"
                        )

                    for idx, name in enumerate(method_names):
                        resolved_means.append(_resolve_external(means, name, idx))
                        resolved_stds.append(_resolve_external(stds, name, idx))

                # Z-score each method for each fold independently, then average methods
                zscored = []
                for arr, mu_vec, sd_vec in zip(ordered_scores, resolved_means, resolved_stds):
                    z_arr = np.zeros_like(arr, dtype=float)
                    for fold_idx in range(num_folds):
                        sigma = float(sd_vec[fold_idx])
                        if abs(sigma) < eps:
                            z_arr[fold_idx] = 0.0
                        else:
                            z_arr[fold_idx] = (arr[fold_idx] - float(mu_vec[fold_idx])) / sigma
                    zscored.append(z_arr)

                zscored = np.stack(zscored, axis=0)  # [num_methods, M, N]
                aggregated_per_fold = np.mean(zscored, axis=0)  # [M, N]
                aggregated_scores = np.mean(aggregated_per_fold, axis=0)  # [N]
                aggregated_for_metrics = aggregated_per_fold

        # Resolve labels/predictions from cache when not explicitly provided
        if labels is None or predictions is None:
            if self._test_predictions_cache is None:
                raise ValueError(
                    "predictions/labels not provided and test prediction cache is empty"
                )
            if labels is None:
                labels = self._test_predictions_cache['y_true']
            if predictions is None:
                predictions = self._test_predictions_cache['y_pred']

        # Prefer cached correctness indices when available. This is required for
        # new_class_shift, where labels may be binary failure flags while
        # predictions are multiclass IDs.
        correct_idx = None
        incorrect_idx = None
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache.get('correct_idx')
            incorrect_idx = self._test_predictions_cache.get('incorrect_idx')

        if correct_idx is None or incorrect_idx is None:
            correct_idx = np.where(predictions == labels)[0]
            incorrect_idx = np.where(predictions != labels)[0]

        # Per-fold metrics require per-fold predictions for fold-wise AUROC/AURC/AUGRC.
        # Reuse cached per-fold predictions if available.
        predictions_per_fold = None
        if mode == 'per_fold':
            if self._per_fold_predictions_cache is not None:
                predictions_per_fold = self._per_fold_predictions_cache
            elif self._test_predictions_cache is not None:
                cached_pf = self._test_predictions_cache.get('per_fold_predictions')
                if cached_pf is not None:
                    predictions_per_fold = cached_pf

        metrics = self._compute_metrics(
            aggregated_for_metrics,
            correct_idx,
            incorrect_idx,
            predictions,
            labels,
            predictions_per_fold=predictions_per_fold,
            ensemble_uncertainties=aggregated_scores if mode == 'per_fold' else None
        )
        metrics['time_seconds'] = timer.elapsed
        metrics['aggregation_sources'] = method_names
        metrics['aggregation_mode'] = mode

        if mode == 'ensemble':
            metrics['zscore_means'] = {
                name: float(mu) for name, mu in zip(method_names, resolved_means)
            }
            metrics['zscore_stds'] = {
                name: float(sd) for name, sd in zip(method_names, resolved_stds)
            }
        else:
            metrics['zscore_means'] = {
                name: [float(v) for v in mu_vec]
                for name, mu_vec in zip(method_names, resolved_means)
            }
            metrics['zscore_stds'] = {
                name: [float(v) for v in sd_vec]
                for name, sd_vec in zip(method_names, resolved_stds)
            }

        self._uncertainties[aggregation_name] = aggregated_scores
        if mode == 'per_fold':
            self._uncertainties[f'{aggregation_name}_per_fold'] = aggregated_for_metrics
            self._uncertainties[f'{aggregation_name}_ensemble'] = aggregated_scores
        self._results[aggregation_name] = metrics

        return aggregated_for_metrics, metrics

    def run_zscore_aggregation_per_fold(
        self,
        method_names: Optional[List[str]] = None,
        score_dict: Optional[Dict[str, np.ndarray]] = None,
        means: Optional[Union[List[Any], Dict[str, Any]]] = None,
        stds: Optional[Union[List[Any], Dict[str, Any]]] = None,
        use_test_distribution: bool = True,
        aggregation_name: str = 'ZScore_Aggregation_per_fold',
        predictions: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Dedicated per-fold z-score aggregation entrypoint."""
        return self.run_zscore_aggregation(
            method_names=method_names,
            score_dict=score_dict,
            means=means,
            stds=stds,
            use_test_distribution=use_test_distribution,
            aggregation_name=aggregation_name,
            mode='per_fold',
            predictions=predictions,
            labels=labels
        )

    def run_zscore_aggregation_ensemble(
        self,
        method_names: Optional[List[str]] = None,
        score_dict: Optional[Dict[str, np.ndarray]] = None,
        means: Optional[Union[List[Any], Dict[str, Any]]] = None,
        stds: Optional[Union[List[Any], Dict[str, Any]]] = None,
        use_test_distribution: bool = True,
        aggregation_name: str = 'ZScore_Aggregation_ensemble',
        predictions: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Dedicated ensemble z-score aggregation entrypoint."""
        return self.run_zscore_aggregation(
            method_names=method_names,
            score_dict=score_dict,
            means=means,
            stds=stds,
            use_test_distribution=use_test_distribution,
            aggregation_name=aggregation_name,
            mode='ensemble',
            predictions=predictions,
            labels=labels
        )
    
    def _get_per_fold_predictions(self, batch_size):
        """
        Get per-fold predictions (vanilla, no augmentations).
        Computes once and caches for reuse across UQ methods.
        
        If per-fold predictions are already set via set_per_fold_predictions(),
        returns the cached values WITHOUT running vanilla inference.
        This saves significant time when running multiple UQ methods.
        
        Uses self.test_dataset (normalized) to ensure predictions match
        the cached ensemble predictions used for all benchmarks.
        
        Args:
            batch_size: Batch size for inference
        
        Returns:
            np.ndarray: Per-fold predictions [M, N]
        """
        if self._per_fold_predictions_cache is not None:
            return self._per_fold_predictions_cache
        
        # Compute via vanilla inference
        print(" Computing per-fold predictions (vanilla inference, no augmentations)...")
        predictions_per_fold = []
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        for model_idx, model in enumerate(self.models):
            fold_preds = []
            for batch_imgs, _ in test_loader:
                with torch.no_grad():
                    model.eval()
                    batch_imgs = batch_imgs.to(self.device)
                    logits = model(batch_imgs)
                    if logits.shape[1] == 1:
                        probs = torch.sigmoid(logits)
                        probs = torch.cat([1 - probs, probs], dim=1)
                    else:
                        probs = torch.softmax(logits, dim=1)
                    fold_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
            predictions_per_fold.append(np.concatenate(fold_preds))
        
        self._per_fold_predictions_cache = np.array(predictions_per_fold)  # [M, N]
        print(f" Cached per-fold predictions [{len(self.models)} folds × {len(predictions_per_fold[0])} samples]")
        return self._per_fold_predictions_cache
    
    def run_msr(
        self,
        y_scores: np.ndarray,
        y_true: np.ndarray,
        indiv_scores: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
        indiv_logits: Optional[np.ndarray] = None,
        per_fold_evaluation: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run Maximum Softmax Response (MSR).
        
        Args:
            y_scores: Ensemble probability scores [N, num_classes].
                     MSR is computed directly from these probabilities.
            y_true: True labels [N]
            indiv_scores: Optional individual model scores [num_models, N, num_classes]
            logits: Optional ensemble logits [N, C] (kept for API compatibility)
            indiv_logits: Optional per-fold logits [num_models, N, C]
            per_fold_evaluation: If True and indiv_scores provided, compute per-fold metrics.
                                If False, use ensemble predictions (legacy behavior)
        
        Returns:
            tuple: (uncertainties, metrics_dict)
                If per_fold_evaluation=True and indiv_scores provided: uncertainties is [num_folds, N]
                Otherwise: uncertainties is [N]
        """
        # MSR policy: use averaged softmax/sigmoid scores directly.
        # Keep logits arguments for backward-compatible call sites.
        scores_for_msr = y_scores
        indiv_scores_for_msr = indiv_scores

        timer = Timer("MSR computation")
        with timer:
            if indiv_scores_for_msr is not None and per_fold_evaluation:
                # Compute MSR per-fold
                uncertainties_per_fold = []
                predictions_per_fold = []
                for fold_idx in range(indiv_scores_for_msr.shape[0]):
                    fold_scores = indiv_scores_for_msr[fold_idx]  # [N, num_classes]
                    fold_uncertainties = uq.distance_to_hard_labels_computation(fold_scores)
                    fold_predictions = np.argmax(fold_scores, axis=1)
                    uncertainties_per_fold.append(fold_uncertainties)
                    predictions_per_fold.append(fold_predictions)
                uncertainties = np.array(uncertainties_per_fold)  # [num_folds, N]
                predictions_per_fold = np.array(predictions_per_fold)  # [num_folds, N]
            else:
                # Ensemble-based: single uncertainty from ensemble
                uncertainties = uq.distance_to_hard_labels_computation(scores_for_msr)
                predictions_per_fold = None
        
        # Use cached predictions if available
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            y_pred = np.argmax(scores_for_msr, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        # Compute TRUE ensemble uncertainties BEFORE metrics (needed for correct JSON values)
        ensemble_uncertainties = None
        if uncertainties.ndim == 2:
            ensemble_uncertainties = uq.distance_to_hard_labels_computation(scores_for_msr)  # [N]
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true, 
                                       predictions_per_fold=predictions_per_fold,
                                       ensemble_uncertainties=ensemble_uncertainties)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        if uncertainties.ndim == 2:
            # Per-fold: store averaged (for metrics), per-fold (for multi-curve plots), and ensemble (for reference)
            self._uncertainties['MSR'] = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['MSR_per_fold'] = uncertainties  # [num_folds, N] for plotting all curves
            self._uncertainties['MSR_ensemble'] = ensemble_uncertainties  # [N] TRUE ensemble
            self._predictions_per_fold['MSR'] = predictions_per_fold  # [num_folds, N] for accurate ROC curves
        else:
            # Ensemble: store single uncertainty
            self._uncertainties['MSR'] = uncertainties
        self._results['MSR'] = metrics
        
        return uncertainties, metrics
    
    def run_mls(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        indiv_logits: Optional[np.ndarray] = None,
        per_fold_evaluation: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run Maximum Logit Score (MLS) - negative of maximum unnormalized logit.
        
        Args:
            logits: Ensemble logits [N, C] (unnormalized scores)
            y_true: True labels [N]
            indiv_logits: Optional individual model logits [num_models, N, C]
            per_fold_evaluation: If True and indiv_logits provided, compute per-fold metrics.
                                If False, use ensemble predictions (legacy behavior)
        
        Returns:
            tuple: (uncertainties, metrics_dict)
                If per_fold_evaluation=True and indiv_logits provided: uncertainties is [num_folds, N]
                Otherwise: uncertainties is [N]
        """
        from .methods.distance import maximum_logit_score_computation
        
        timer = Timer("MLS computation")
        with timer:
            if indiv_logits is not None and per_fold_evaluation:
                # Compute MLS per-fold
                uncertainties_per_fold = []
                predictions_per_fold = []
                for fold_idx in range(indiv_logits.shape[0]):
                    fold_logits = indiv_logits[fold_idx]  # [N, C]
                    fold_uncertainties = maximum_logit_score_computation(fold_logits)
                    # Predictions from logits
                    if fold_logits.ndim == 1 or fold_logits.shape[1] == 1:
                        # Binary: threshold at 0
                        fold_predictions = (fold_logits.ravel() > 0).astype(int)
                    else:
                        # Multiclass: argmax
                        fold_predictions = np.argmax(fold_logits, axis=1)
                    uncertainties_per_fold.append(fold_uncertainties)
                    predictions_per_fold.append(fold_predictions)
                uncertainties = np.array(uncertainties_per_fold)  # [num_folds, N]
                predictions_per_fold = np.array(predictions_per_fold)  # [num_folds, N]
            else:
                # Ensemble-based: single uncertainty from ensemble
                uncertainties = maximum_logit_score_computation(logits)
                predictions_per_fold = None
        
        # Use cached predictions if available
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            # Compute predictions from logits
            if logits.ndim == 1 or logits.shape[1] == 1:
                y_pred = (logits.ravel() > 0).astype(int)
            else:
                y_pred = np.argmax(logits, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        # Compute TRUE ensemble uncertainties BEFORE metrics (needed for correct JSON values)
        ensemble_uncertainties = None
        if uncertainties.ndim == 2:
            ensemble_uncertainties = maximum_logit_score_computation(logits)  # [N]
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true, 
                                       predictions_per_fold=predictions_per_fold,
                                       ensemble_uncertainties=ensemble_uncertainties)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        if uncertainties.ndim == 2:
            # Per-fold: store averaged (for metrics), per-fold (for multi-curve plots), and ensemble (for reference)
            self._uncertainties['MLS'] = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['MLS_per_fold'] = uncertainties  # [num_folds, N] for plotting all curves
            self._uncertainties['MLS_ensemble'] = ensemble_uncertainties  # [N] TRUE ensemble
            self._predictions_per_fold['MLS'] = predictions_per_fold  # [num_folds, N] for accurate ROC curves
        else:
            # Ensemble: store single uncertainty
            self._uncertainties['MLS'] = uncertainties
        self._results['MLS'] = metrics
        
        return uncertainties, metrics
    
    def run_msr_calibrated(
        self,
        y_scores_test: np.ndarray,
        y_true_test: np.ndarray,
        y_scores_calib: np.ndarray,
        y_true_calib: np.ndarray,
        logits_test: Optional[np.ndarray] = None,
        logits_calib: Optional[np.ndarray] = None,
        indiv_logits_test: Optional[np.ndarray] = None,
        indiv_logits_calib: Optional[np.ndarray] = None,
        indiv_scores_test: Optional[np.ndarray] = None,
        indiv_scores_calib: Optional[np.ndarray] = None,
        method: str = 'temperature',
        per_fold_evaluation: bool = True,
        auto_tune_platt: bool = False,
        verbose_tuning: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run calibrated MSR using post-hoc calibration.
        
        Args:
            y_scores_test: Ensemble test probabilities [N, C]
            y_true_test: Test labels [N]
            y_scores_calib: Ensemble calibration probabilities [N_calib, C]
            y_true_calib: Calibration labels [N_calib]
            logits_test: Ensemble test logits (for temperature) [N, C]
            logits_calib: Ensemble calibration logits (for temperature) [N_calib, C]
            indiv_logits_test: Per-fold test logits [num_folds, N, C]
            indiv_logits_calib: Per-fold calibration logits [num_folds, N_calib, C]
            indiv_scores_test: Per-fold test scores [num_folds, N, C]
            indiv_scores_calib: Per-fold calibration scores [num_folds, N_calib, C]
            method: 'temperature', 'platt', or 'isotonic'
            per_fold_evaluation: If True and indiv_* provided, compute per-fold metrics.
                                If False, use ensemble predictions (legacy behavior)
        
        Returns:
            tuple: (uncertainties, metrics_dict)
                If per_fold_evaluation=True and indiv_* provided: uncertainties is [num_folds, N]
                Otherwise: uncertainties is [N]
        """
        from .methods.distance import posthoc_calibration
        from .core.utils import apply_calibration
        
        calibrated_scores_per_fold = []

        timer = Timer(f"MSR-{method} calibration")
        with timer:
            if per_fold_evaluation and method == 'temperature' and indiv_logits_test is not None and indiv_logits_calib is not None:
                # Per-fold calibration with temperature scaling
                uncertainties_per_fold = []
                predictions_per_fold = []
                num_folds = indiv_logits_test.shape[0]
                
                for fold_idx in range(num_folds):
                    fold_logits_calib = indiv_logits_calib[fold_idx]
                    fold_logits_test = indiv_logits_test[fold_idx]
                    
                    # Fit calibration on this fold's calibration set
                    _, calibration_model = posthoc_calibration(
                        fold_logits_calib, y_true_calib, method,
                        auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                    )
                    
                    # Get fold's test scores
                    fold_scores_test = indiv_scores_test[fold_idx] if indiv_scores_test is not None else None
                    if fold_scores_test is None:
                        # Compute from logits if not provided
                        import torch
                        logits_tensor = torch.from_numpy(fold_logits_test)
                        if logits_tensor.ndim == 1 or (logits_tensor.ndim == 2 and logits_tensor.shape[1] == 1):
                            probs_pos = torch.sigmoid(logits_tensor.reshape(-1, 1)).numpy().ravel()
                            fold_scores_test = np.stack([1.0 - probs_pos, probs_pos], axis=1)
                        else:
                            fold_scores_test = torch.softmax(logits_tensor, dim=1).numpy()
                    
                    # Apply calibration
                    calibrated_scores = apply_calibration(
                        fold_scores_test, calibration_model, method, logits=fold_logits_test
                    )
                    
                    # Compute uncertainty and predictions
                    fold_uncertainties = uq.distance_to_hard_labels_computation(calibrated_scores)
                    if calibrated_scores.ndim == 1:
                        calibrated_scores_per_fold.append(np.stack([1.0 - calibrated_scores, calibrated_scores], axis=1))
                    else:
                        calibrated_scores_per_fold.append(calibrated_scores)
                    # Handle both binary (1D) and multiclass (2D) calibrated scores
                    if calibrated_scores.ndim == 1:
                        # Binary: calibrated_scores is [N] (prob of positive class)
                        fold_predictions = (calibrated_scores > 0.5).astype(int)
                    else:
                        # Multiclass: calibrated_scores is [N, C]
                        fold_predictions = np.argmax(calibrated_scores, axis=1)
                    uncertainties_per_fold.append(fold_uncertainties)
                    predictions_per_fold.append(fold_predictions)
                
                uncertainties = np.array(uncertainties_per_fold)  # [num_folds, N]
                predictions_per_fold = np.array(predictions_per_fold)  # [num_folds, N]
                
            elif per_fold_evaluation and method != 'temperature' and indiv_scores_test is not None and indiv_scores_calib is not None:
                # Per-fold calibration with Platt/Isotonic
                # CRITICAL: Use original predictions, don't recalculate from calibrated scores
                # batch_size is not needed here since predictions are already cached
                original_predictions_per_fold = self._get_per_fold_predictions(batch_size=1024)
                
                uncertainties_per_fold = []
                predictions_per_fold = []
                num_folds = indiv_scores_test.shape[0]
                
                for fold_idx in range(num_folds):
                    fold_scores_calib = indiv_scores_calib[fold_idx]
                    fold_scores_test = indiv_scores_test[fold_idx]
                    
                    # Fit calibration on this fold's calibration set
                    _, calibration_model = posthoc_calibration(
                        fold_scores_calib, y_true_calib, method,
                        auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                    )
                    
                    # Apply calibration to scores (for uncertainty computation)
                    calibrated_scores = apply_calibration(
                        fold_scores_test, calibration_model, method
                    )
                    
                    # Compute uncertainty from calibrated scores
                    fold_uncertainties = uq.distance_to_hard_labels_computation(calibrated_scores)

                    if calibrated_scores.ndim == 1:
                        calibrated_scores_per_fold.append(np.stack([1.0 - calibrated_scores, calibrated_scores], axis=1))
                    else:
                        calibrated_scores_per_fold.append(calibrated_scores)
                    
                    # Use ORIGINAL predictions, NOT from calibrated scores
                    # Calibration should only affect confidence scores, not predictions
                    fold_predictions = original_predictions_per_fold[fold_idx]
                    
                    uncertainties_per_fold.append(fold_uncertainties)
                    predictions_per_fold.append(fold_predictions)
                
                uncertainties = np.array(uncertainties_per_fold)  # [num_folds, N]
                predictions_per_fold = np.array(predictions_per_fold)  # [num_folds, N]
                
            else:
                # Ensemble mode: calibrate the ENSEMBLE predictions (not individual folds)
                # This is the correct approach for deep ensembles - treat ensemble as a single model
                if method == 'temperature':
                    if logits_calib is None:
                        raise ValueError("Temperature scaling requires logits")
                    _, calibration_model = posthoc_calibration(
                        logits_calib, y_true_calib, method,
                        auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                    )
                    calibrated_scores = apply_calibration(y_scores_test, calibration_model, method, logits=logits_test)
                else:
                    _, calibration_model = posthoc_calibration(
                        y_scores_calib, y_true_calib, method,
                        auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                    )
                    calibrated_scores = apply_calibration(y_scores_test, calibration_model, method)
                
                # Compute uncertainty on calibrated ensemble predictions
                uncertainties = uq.distance_to_hard_labels_computation(calibrated_scores)
                predictions_per_fold = None
        
        # CRITICAL: Use ORIGINAL predictions, not calibrated scores
        # Calibration should only adjust confidence scores, not change predictions
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred_calibrated = self._test_predictions_cache['y_pred']
        else:
            # Use original predictions from ensemble/model, NOT from calibrated scores
            # This ensures accuracy stays consistent across all CSF methods
            raise RuntimeError("MSR_calibrated requires cached predictions. Run run_msr() first.")
        
        # Compute TRUE ensemble uncertainties BEFORE metrics (needed for correct JSON values)
        ensemble_uncertainties = None
        if uncertainties.ndim == 2:
            if per_fold_evaluation and len(calibrated_scores_per_fold) > 0:
                # Average calibrated fold probabilities first, then compute MSR.
                calibrated_ensemble = np.mean(np.stack(calibrated_scores_per_fold, axis=0), axis=0)
                ensemble_uncertainties = uq.distance_to_hard_labels_computation(calibrated_ensemble)
            else:
                # Fallback for non-per-fold mode.
                if method == 'temperature':
                    if logits_calib is None:
                        raise ValueError("Temperature scaling requires ensemble logits")
                    _, calib_model = posthoc_calibration(
                        logits_calib, y_true_calib, method,
                        auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                    )
                    calibrated_ensemble = apply_calibration(y_scores_test, calib_model, method, logits=logits_test)
                else:
                    _, calib_model = posthoc_calibration(
                        y_scores_calib, y_true_calib, method,
                        auto_tune_platt=auto_tune_platt, verbose=verbose_tuning
                    )
                    calibrated_ensemble = apply_calibration(y_scores_test, calib_model, method)
                
                ensemble_uncertainties = uq.distance_to_hard_labels_computation(calibrated_ensemble)
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx,
                                       y_pred_calibrated, y_true_test,
                                       predictions_per_fold=predictions_per_fold,
                                       ensemble_uncertainties=ensemble_uncertainties)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        if uncertainties.ndim == 2:
            # Per-fold: store averaged (for metrics), per-fold (for multi-curve plots), and ensemble (for reference)
            self._uncertainties['MSR_calibrated'] = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['MSR_calibrated_per_fold'] = uncertainties  # [num_folds, N]
            self._uncertainties['MSR_calibrated_ensemble'] = ensemble_uncertainties  # [N] TRUE ensemble
            self._predictions_per_fold['MSR_calibrated'] = predictions_per_fold  # [num_folds, N] for accurate ROC curves
        else:
            # Ensemble: store single uncertainty
            self._uncertainties['MSR_calibrated'] = uncertainties
        self._results['MSR_calibrated'] = metrics
        
        return uncertainties, metrics
    
    def run_ensemble(
        self,
        indiv_scores: np.ndarray,
        y_true: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run ensemble standard deviation.
        
        Args:
            indiv_scores: Individual model scores [K, N, C] or [N, K, C]
            y_true: True labels [N]
        
        Returns:
            tuple: (uncertainties, metrics_dict)
        """
        timer = Timer("Ensemble STD computation")
        with timer:
            # Handle both [K, N, C] and [N, K, C] formats
            if indiv_scores.ndim != 3:
                raise ValueError(f"indiv_scores must be 3D, got shape {indiv_scores.shape}")
            
            K, N, C = indiv_scores.shape
            # If K looks like number of samples (much larger than C), assume [N, K, C] format
            if K > 100 and K > C * 10:
                # Already in [N, K, C] format
                pass
            else:
                # Assume [K, N, C] format, transpose to [N, K, C]
                indiv_scores = np.transpose(indiv_scores, (1, 0, 2))
            
            uncertainties = uq.ensembling_stds_computation(indiv_scores)
            
            # Ensure output is numpy array
            if isinstance(uncertainties, list):
                uncertainties = np.array(uncertainties)
        
        # Use cached predictions if available
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            y_scores = np.mean(indiv_scores, axis=0)
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        self._uncertainties['Ensembling'] = uncertainties
        self._results['Ensembling'] = metrics
        
        return uncertainties, metrics
    
    def run_mcdropout(
        self,
        test_dataset: torch.utils.data.Dataset,
        y_true: np.ndarray,
        batch_size: int = 256,
        num_samples: int = 5,
        per_fold_evaluation: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        seed: int = 42,
        y_true_original: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run Monte Carlo Dropout uncertainty quantification.
        
        Requires models to have dropout layers. Performs multiple stochastic forward passes
        with dropout enabled at test time to estimate epistemic uncertainty.
        
        Args:
            test_dataset: Test dataset (with transforms applied)
            y_true: True labels [N]
            batch_size: Batch size for inference
            num_samples: Number of MC dropout samples per model (default: 5)
            per_fold_evaluation: If True, compute per-model uncertainty then average (per-fold evaluation)
                                If False, average models first then compute uncertainty (ensemble evaluation)
            num_workers: Number of DataLoader workers
            pin_memory: Whether to enable pinned host memory for faster host->device copies
            persistent_workers: Keep workers alive across iterations (only effective when num_workers > 0)
            prefetch_factor: Number of batches prefetched per worker (only effective when num_workers > 0)
            seed: Random seed for MC dropout sampling. Ensures ensemble-level metrics are
                  reproducible across runs regardless of prior RNG state (default: 42).
        
        Returns:
            tuple: (uncertainties, metrics_dict)
                If per_fold_evaluation=True: uncertainties is [num_folds, N]
                Otherwise: uncertainties is [N]
        """
        from torch.utils.data import DataLoader

        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = persistent_workers
            if prefetch_factor is not None:
                loader_kwargs['prefetch_factor'] = prefetch_factor
        
        # Seed torch RNG before MC sampling so that ensemble-level metrics are
        # reproducible regardless of which other methods ran before MCDropout.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        print(f" MC Dropout seed: {seed}")

        timer = Timer(f"MC Dropout computation ({num_samples} samples)")
        with timer:
            mcdropout = uq.MCDropoutMethod(num_samples=num_samples)
            test_loader = DataLoader(test_dataset, **loader_kwargs)
            
            # Always compute per-model uncertainties first.
            # Also returns per-fold mean predictions (argmax of mean MC probs per fold).
            per_fold_uncertainties, per_fold_mean_preds_mcd = mcdropout.compute(
                self.models, test_loader, self.device,
                ensemble_mode=True,
                return_per_fold=True
            )  # [M, N], [M, N]
            
            if per_fold_evaluation:
                uncertainties = per_fold_uncertainties  # [M, N]
            else:
                uncertainties = np.mean(per_fold_uncertainties, axis=0)  # [N]
        
        # Per-fold predictions: use MCD's own mean predictions (NOT baseline single-pass)
        # so that fold-level failures are defined by argmax(mean MC softmax) per fold.
        if per_fold_evaluation:
            predictions_per_fold = per_fold_mean_preds_mcd  # [M, N]
        else:
            predictions_per_fold = None

        # Use cached ensemble predictions for ensemble-level correct/incorrect indices
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            test_loader = DataLoader(test_dataset, **loader_kwargs)
            y_scores = self._get_ensemble_predictions(test_loader)
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]

        # For new_class_shift: replace binary_gt-based indices with stochastic-aware ones.
        # Per-fold: correct iff mean MCD prediction == original class label (known class only).
        # Ensemble: intersection of per-fold correct (unanimously correct under MCD).
        pf_correct_override = None
        pf_incorrect_override = None
        if y_true_original is not None:
            pf_correct_override, pf_incorrect_override, correct_idx, incorrect_idx = \
                self._compute_new_class_shift_indices(per_fold_mean_preds_mcd, y_true_original)

        # use_cached_per_fold_indices=False: derive fold failures from MCD mean predictions,
        # not from the baseline cached per-fold predictions.
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true,
                                       predictions_per_fold=predictions_per_fold,
                                       use_cached_per_fold_indices=False,
                                       per_fold_correct_idx_override=pf_correct_override,
                                       per_fold_incorrect_idx_override=pf_incorrect_override)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results (matching TTA pattern)
        if per_fold_evaluation:
            # Per-fold mode: store per-fold [M, N], averaged [N], and ensemble reference [N]
            averaged_uncertainties = np.mean(uncertainties, axis=0)  # [N]
            self._uncertainties['MCDropout'] = averaged_uncertainties
            self._uncertainties['MCDropout_per_fold'] = uncertainties  # [M, N]
            self._uncertainties['MCDropout_ensemble'] = averaged_uncertainties  # [N]
            self._predictions_per_fold['MCDropout'] = per_fold_mean_preds_mcd  # [M, N] (MCD mean preds)
        else:
            # Ensemble mode: store single averaged uncertainty [N]
            self._uncertainties['MCDropout'] = uncertainties  # [N]
            self._uncertainties['MCDropout_per_fold_internal'] = per_fold_uncertainties  # [M, N]
        
        self._results['MCDropout'] = metrics
        
        return uncertainties, metrics
    
    def run_tta(
        self,
        test_dataset: torch.utils.data.Dataset,
        y_true: np.ndarray,
        image_size: int = 224,
        batch_size: int = 4000,
        nb_augmentations: int = 5,
        n: int = 2,
        m: int = 9,
        nb_channels: int = 3,
        mean: float = 0.5,
        std: float = 0.5,
        per_fold_evaluation: bool = True,
        seed: Optional[int] = None,
        y_true_original: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run Test-Time Augmentation (TTA).
        
        Args:
            test_dataset: Test dataset (with transforms applied)
            y_true: True labels [N]
            image_size: Image size
            batch_size: Batch size for inference
            nb_augmentations: Number of augmentations
            n: RandAugment num_ops
            m: RandAugment magnitude
            nb_channels: Number of image channels
            mean: Normalization mean
            std: Normalization std
            per_fold_evaluation: If True, compute per-model uncertainty then average (per-fold evaluation)
                                If False, average models first then compute uncertainty (ensemble evaluation)
            seed: Random seed for augmentations. If None (default), generates a time-based seed
                  for different augmentations on each run. Set to an integer for reproducibility.
        
        Returns:
            tuple: (uncertainties, metrics_dict)
        """
        # Generate time-based seed if not provided (for non-deterministic augmentations)
        if seed is None:
            seed = int(time.time_ns() % (2**31))  # Use nanosecond timestamp modulo 2^31
            print(f" Using random seed for augmentations: {seed}")
        
        timer = Timer("TTA computation")
        with timer:
            tta = uq.TTAMethod(
                transformations=None,  # Random policies
                n=n,
                m=m,
                nb_augmentations=nb_augmentations,
                nb_channels=nb_channels,
                image_size=image_size,
                image_normalization=True,
                mean=mean,
                std=std,
                batch_size=batch_size
            )
            
            # TTA ALWAYS computes the same way:
            # For each model: apply augmentations and compute std across augmentations
            # This gives us per-model uncertainties [M, N]
            
            # Compute per-model TTA uncertainties and per-fold mean predictions
            _, per_fold_uncertainties, per_fold_mean_preds_tta = tta.compute(
                self.models, test_dataset, self.device, 
                ensemble_mode=True,  # Compute per-model
                return_per_fold=True,  # Return full [M, N] array
                seed=seed  # Pass seed to TTA
            )  # stds [M,N], per_fold_uncertainties [M,N], per_fold_mean_preds_tta [M,N]
            
            # Difference between modes: what we return and how we compute metrics
            if per_fold_evaluation:
                # Per-fold mode: return [M, N] for per-fold metrics computation
                uncertainties = per_fold_uncertainties  # [M, N]
            else:
                # Ensemble mode: average across models and return [N]
                uncertainties = np.mean(per_fold_uncertainties, axis=0)  # [N]
        
        # Use cached ensemble predictions for ensemble-level correct/incorrect indices
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            y_scores = self._get_ensemble_predictions(test_loader)
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]

        # For new_class_shift: replace binary_gt-based indices with stochastic-aware ones.
        # Per-fold: correct iff mean TTA prediction == original class label (known class only).
        # Ensemble: intersection of per-fold correct (unanimously correct under TTA).
        pf_correct_override = None
        pf_incorrect_override = None
        if y_true_original is not None:
            pf_correct_override, pf_incorrect_override, correct_idx, incorrect_idx = \
                self._compute_new_class_shift_indices(per_fold_mean_preds_tta, y_true_original)

        # Compute metrics based on mode
        if per_fold_evaluation:
            # Per-fold: use TTA's own mean predictions to define fold-level failures.
            # use_cached_per_fold_indices=False so we don't use baseline cached indices.
            metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true,
                                           predictions_per_fold=per_fold_mean_preds_tta,
                                           use_cached_per_fold_indices=False,
                                           per_fold_correct_idx_override=pf_correct_override,
                                           per_fold_incorrect_idx_override=pf_incorrect_override)
        else:
            # Ensemble: compute single metrics using ensemble predictions only
            metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true,
                                           predictions_per_fold=None)
        
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        if per_fold_evaluation:
            # Per-fold mode: store per-fold [M, N], averaged [N], and ensemble reference [N]
            averaged_uncertainties = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['TTA'] = averaged_uncertainties
            self._uncertainties['TTA_per_fold'] = uncertainties  # [M, N] for plotting all curves
            # Ensemble baseline: same averaged uncertainties, but evaluated against ensemble predictions
            self._uncertainties['TTA_ensemble'] = averaged_uncertainties  # [N] for ensemble ROC baseline
            self._predictions_per_fold['TTA'] = per_fold_mean_preds_tta  # [M, N] (TTA mean preds)
        else:
            # Ensemble mode: store single averaged uncertainty [N]
            # DO NOT store per-fold data (it will cause plotting to show per-fold curves)
            self._uncertainties['TTA'] = uncertainties  # [N] averaged - this is the ensemble result
            # Store per-fold data with different key for internal reference only (won't be plotted)
            self._uncertainties['TTA_per_fold_internal'] = per_fold_uncertainties  # [M, N] for reference
        
        self._results['TTA'] = metrics
        
        return uncertainties, metrics
    
    def run_augmentation_calibration_caching(
        self,
        dataset: torch.utils.data.Dataset,
        aug_folder: str,
        N: int = 2,
        M: int = 45,
        num_policies: int = 500,
        image_size: int = 224,
        batch_size: int = 128,
        nb_channels: int = 3,
        image_normalization: bool = True,
        mean: float = 0.5,
        std: float = 0.5,
        use_monai_cache: bool = True,
        cache_rate: float = 1.0,
        cache_num_workers: int = 8,
        dataloader_workers: int = 8,
        dataloader_prefetch: int = 8
    ) -> str:
        """
        Run BetterRandAugment on calibration dataset and cache results for GPS.
        
        This function applies random augmentation policies to the calibration dataset,
        computes model predictions for each augmented version, and saves them to disk
        as .npz files. These cached predictions are then used by GPS for greedy policy search.
        
        Args:
            dataset: Calibration dataset (should use transform WITHOUT normalization, e.g., calib_dataset_tta)
            aug_folder: Output directory for cached augmentation predictions
            N: Number of augmentation operations per policy (BetterRandAugment)
            M: Magnitude parameter for augmentations (0-100 typical)
            num_policies: Number of random augmentation policies to generate
            image_size: Input image size
            batch_size: Batch size for inference
            nb_channels: Number of image channels (1 for grayscale, 3 for RGB)
            image_normalization: Whether to apply normalization
            mean: Normalization mean (single value or per-channel)
            std: Normalization std (single value or per-channel)
            use_monai_cache: Whether to use MONAI CacheDataset for faster loading
            cache_rate: Fraction of dataset to cache in memory (0-1)
            cache_num_workers: Number of workers for cache building
            dataloader_workers: Number of workers for DataLoader
            dataloader_prefetch: Prefetch factor for DataLoader
        
        Returns:
            str: Path to the output folder containing cached predictions
        
        Example:
            >>> detector = FailureDetector(models, train_data, calib_data, test_data, device)
            >>> aug_folder = detector.run_augmentation_calibration_caching(
            ...     dataset=calib_dataset_tta,  # Use unnormalized dataset!
            ...     aug_folder='gps_augment/breastmnist_calib',
            ...     N=2, M=45, num_policies=500,
            ...     use_monai_cache=True, batch_size=128
            ... )
            >>> # Later use for GPS:
            >>> uncertainties, metrics = detector.run_gps(
            ...     test_dataset, y_true, aug_folder, correct_idx, incorrect_idx
            ... )
        """
        print(f"\n Caching augmentation predictions for GPS...")
        print(f" Output folder: {aug_folder}")
        print(f" Calibration set size: {len(dataset)}")
        print(f" Policies: {num_policies}, N={N}, M={M}")
        print(f" MONAI cache: {use_monai_cache}, Cache rate: {cache_rate}")
        
        timer = Timer("Augmentation caching")
        with timer:
            uq.apply_randaugment_and_store_results(
                dataset=dataset,
                models=self.models,
                N=N,
                M=M,
                num_policies=num_policies,
                device=self.device,
                folder_name=aug_folder,
                image_normalization=image_normalization,
                mean=mean,
                std=std,
                nb_channels=nb_channels,
                image_size=image_size,
                batch_size=batch_size,
                use_monai_cache=use_monai_cache,
                cache_rate=cache_rate,
                cache_num_workers=cache_num_workers,
                dataloader_workers=dataloader_workers,
                dataloader_prefetch=dataloader_prefetch
            )
        
        # Verify files were created
        import glob
        npz_files = glob.glob(os.path.join(aug_folder, f"N{N}_M{M}_*.npz"))
        print(f"Cached {len(npz_files)} augmentation prediction files")
        
        if len(npz_files) == 0:
            raise RuntimeError(f"No .npz files created in {aug_folder}")
        
        return aug_folder
    
    def run_gps(
        self,
        test_dataset: torch.utils.data.Dataset,
        y_true: np.ndarray,
        aug_folder: str,
        correct_idx_calib: List[int],
        incorrect_idx_calib: List[int],
        image_size: int = 224,
        batch_size: int = 4000,
        max_iterations: int = 5,
        num_workers: int = 90,
        num_searches: int = 30,
        top_k: int = 3,
        seed: int = 64,
        nb_channels: int = 3,
        mean: float = 0.5,
        std: float = 0.5,
        cache_dir: Optional[str] = None,
        per_fold_evaluation: bool = True,
        y_true_original: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run Greedy Policy Search (GPS).
        
        Args:
            test_dataset: Test dataset
            y_true: True labels [N]
            aug_folder: Folder containing pre-computed augmentation predictions
            correct_idx_calib: Indices of correctly classified calibration samples
            incorrect_idx_calib: Indices of incorrectly classified calibration samples
            image_size: Image size
            batch_size: Batch size
            max_iterations: Max GPS iterations
            num_workers: Number of parallel workers for search
            num_searches: Number of searches per iteration
            top_k: Top-k policies to select
            seed: Random seed
            nb_channels: Number of image channels
            mean: Normalization mean
            std: Normalization std
            cache_dir: Directory to cache policies
            per_fold_evaluation: If True, compute per-model uncertainty then average (per-fold evaluation)
                                If False, average models first then compute uncertainty (ensemble evaluation)
        
        Returns:
            tuple: (uncertainties, metrics_dict)
        """
        import os
        import pickle
        import hashlib
        
        timer = Timer("GPS computation")
        with timer:
            # Initialize GPS
            gps = uq.GPSMethod(
                aug_folder=aug_folder,
                correct_calib=correct_idx_calib,
                incorrect_calib=incorrect_idx_calib,
                max_iter=max_iterations
            )
            
            # Cache logic
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_key_parts = [
                    aug_folder,
                    str(sorted(correct_idx_calib)),
                    str(sorted(incorrect_idx_calib)),
                    str(max_iterations),
                    str(num_workers),
                    str(num_searches),
                    str(top_k),
                    str(seed)
                ]
                cache_key = hashlib.md5('_'.join(cache_key_parts).encode()).hexdigest()
                cache_file = os.path.join(cache_dir, f'gps_policies_{cache_key}.pkl')
                
                if os.path.exists(cache_file):
                    print(f" Loading cached GPS policies...")
                    with open(cache_file, 'rb') as f:
                        gps.policies = pickle.load(f)
                    print(f" Loaded {len(gps.policies)} policy groups from cache")
                else:
                    gps.search_policies(
                        num_workers=num_workers,
                        num_searches=num_searches,
                        top_k=top_k,
                        seed=seed
                    )
                    with open(cache_file, 'wb') as f:
                        pickle.dump(gps.policies, f)
                    print(f" Cached {len(gps.policies)} policy groups")
            else:
                gps.search_policies(
                    num_workers=num_workers,
                    num_searches=num_searches,
                    top_k=top_k,
                    seed=seed
                )
            
            # Compute GPS uncertainties
            # GPS always applies selected policies to each model and computes avg(std(models))
            # This is consistent with TTA but uses ensemble-optimized policies from search
            
            print(f"\n  Applying {len(gps.policies)} selected policies to test set...")
            print(f" Computing per-model uncertainties, then averaging: avg(std(models))...")
            
            # CRITICAL: Reset random state before applying augmentations for reproducibility
            # Augmentation operations (Rotate, ShearX, etc.) use random.random() internally
            # and the global state may have been modified by previous methods
            import random as pyrandom
            pyrandom.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Always compute per-model uncertainties first.
            # Also returns per-fold mean predictions (argmax of mean GPS softmax per fold).
            _, per_fold_uncertainties, per_fold_mean_preds_gps = gps.compute(
                self.models, test_dataset, self.device,
                n=2, m=45,
                nb_channels=nb_channels,
                image_size=image_size,
                image_normalization=True,
                mean=mean,
                std=std,
                batch_size=batch_size,
                ensemble_mode=True,  # Compute per-model std
                return_per_fold=True  # Return [M, N]
            )  # stds [M,N], per_fold_uncertainties [M,N], per_fold_mean_preds_gps [M,N]
            
            if per_fold_evaluation:
                uncertainties = per_fold_uncertainties  # [M, N] for per-fold metrics
            else:
                uncertainties = np.mean(per_fold_uncertainties, axis=0)  # [N] averaged
        
        # Use cached ensemble predictions for ensemble-level correct/incorrect indices
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            y_scores = self._get_ensemble_predictions(test_loader)
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]

        # For new_class_shift: replace binary_gt-based indices with stochastic-aware ones.
        # Per-fold: correct iff mean GPS prediction == original class label (known class only).
        # Ensemble: intersection of per-fold correct (unanimously correct under GPS).
        pf_correct_override = None
        pf_incorrect_override = None
        if y_true_original is not None:
            pf_correct_override, pf_incorrect_override, correct_idx, incorrect_idx = \
                self._compute_new_class_shift_indices(per_fold_mean_preds_gps, y_true_original)

        if per_fold_evaluation:
            # Per-fold: use GPS's own mean predictions to define fold-level failures.
            # use_cached_per_fold_indices=False so we don't use baseline cached indices.
            metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true,
                                           predictions_per_fold=per_fold_mean_preds_gps,
                                           use_cached_per_fold_indices=False,
                                           per_fold_correct_idx_override=pf_correct_override,
                                           per_fold_incorrect_idx_override=pf_incorrect_override)
        else:
            metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true,
                                           predictions_per_fold=None)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results (matching TTA pattern)
        if per_fold_evaluation:
            # Per-fold mode: store per-fold [M, N], averaged [N], and ensemble reference [N]
            averaged_uncertainties = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['GPS'] = averaged_uncertainties
            self._uncertainties['GPS_per_fold'] = uncertainties  # [M, N] for plotting all curves
            # Ensemble baseline: same averaged uncertainties, but evaluated against ensemble predictions
            self._uncertainties['GPS_ensemble'] = averaged_uncertainties  # [N] for ensemble ROC baseline
            self._predictions_per_fold['GPS'] = per_fold_mean_preds_gps  # [M, N] (GPS mean preds)
        else:
            # Ensemble mode: store single averaged uncertainty [N]
            # DO NOT store per-fold data (would trigger per-fold plotting)
            self._uncertainties['GPS'] = uncertainties  # [N] averaged - this is avg(std(models))
            # Store per-fold with internal suffix for potential debugging
            self._uncertainties['GPS_per_fold_internal'] = per_fold_uncertainties  # [M, N]
        
        self._results['GPS'] = metrics
        
        return uncertainties, metrics
    
    def run_knn_raw(
        self,
        test_loader: DataLoader,
        train_loaders: List[DataLoader],
        y_true: np.ndarray,
        layer_name: str = 'avgpool',
        k: int = 5,
        per_fold_evaluation: bool = True,
        k_grid: Optional[List[int]] = None,
        calib_loader: Optional[DataLoader] = None,
        y_true_calib: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run KNN in raw latent space with optional hyperparameter tuning.
        
        Args:
            test_loader: Test data loader (or single DataLoader if 1 model)
            train_loaders: List of train data loaders (one per model fold)
                          Or single DataLoader if models is a single model
            y_true: True labels [N]
            layer_name: Layer name for feature extraction
            k: Number of nearest neighbors (used if k_grid is None)
            per_fold_evaluation: If True, compute per-fold metrics. If False, average uncertainties
            k_grid: Optional list of k values for grid search (e.g., [1, 5, 10, 20, 50, 100, 200])
                   If provided, will perform hyperparameter tuning on calibration set
            calib_loader: Calibration data loader (required if k_grid is provided)
            y_true_calib: True labels for calibration set (required if k_grid is provided)
        
        Returns:
            tuple: (uncertainties, metrics_dict)
                If per_fold_evaluation=True: uncertainties is [num_folds, N]
                Otherwise: uncertainties is [N] (averaged)
        """
        timer = Timer("KNN-Raw computation")
        
        # Hyperparameter tuning on calibration set (if requested)
        k_selected = k  # Default
        if k_grid is not None:
            if calib_loader is None or y_true_calib is None:
                raise ValueError("calib_loader and y_true_calib required for hyperparameter tuning")
            
            print(f" Grid search for k in {k_grid}...")
            timer_tuning = Timer("Hyperparameter tuning")
            
            with timer_tuning:
                # OPTIMIZATION: Fit PCA once on training data, then try all k values
                # Step 1: Extract features and fit PCA (once per fold)
                knn_method = uq.KNNLatentMethod(layer_name=layer_name, k=1)  # k doesn't matter for feature extraction
                knn_method.fit(self.models, train_loaders, self.device)
                
                # Step 2: Extract and transform calibration features + store training features
                from .methods.latent import extract_latent_space_and_compute_shap_importance, get_layer_from_model
                
                calib_features_transformed = []
                train_features_transformed = []  # Store for grid search
                
                for fold_idx, (model, fitted, train_loader) in enumerate(zip(self.models, knn_method.fitted_models, train_loaders)):
                    layer = fitted['layer']
                    
                    # Extract and transform training features (needed for grid search)
                    train_features, _, _, _ = extract_latent_space_and_compute_shap_importance(
                        model, train_loader, self.device, layer, importance=False
                    )
                    train_features_std = fitted['scaler'].transform(train_features.numpy())
                    train_features_pca = fitted['pca'].transform(train_features_std)
                    train_features_transformed.append(train_features_pca)
                    
                    # Extract and transform calibration features
                    calib_features, _, _, _ = extract_latent_space_and_compute_shap_importance(
                        model, calib_loader, self.device, layer, importance=False
                    )
                    calib_features_std = fitted['scaler'].transform(calib_features.numpy())
                    calib_features_pca = fitted['pca'].transform(calib_features_std)
                    calib_features_transformed.append(calib_features_pca)
                
                # Get calibration predictions for AUROC computation
                y_scores_calib = self._get_ensemble_predictions(calib_loader)
                y_pred_calib = np.argmax(y_scores_calib, axis=1)
                correct_idx_calib = np.where(y_pred_calib == y_true_calib)[0]
                incorrect_idx_calib = np.where(y_pred_calib != y_true_calib)[0]
                
                # Check if we have failures on calibration set
                if len(incorrect_idx_calib) == 0:
                    print(f" [WARNING] WARNING: Perfect accuracy on calibration set - cannot tune hyperparameters!")
                    print(f" Using default k=5 (no failures to optimize against)")
                    k_selected = 5  # Use k=5 as default for perfect models
                    best_auroc = None  # No AUROC computed
                else:
                    # Step 3: Grid search over k values (fast, only KNN fitting)
                    best_auroc = -1
                    best_k = k_grid[0]
                    
                    for k_candidate in k_grid:
                        # Fit KNN with k_candidate on ALREADY PCA-transformed training features
                        knn_models = []
                        for fold_idx in range(len(self.models)):
                            from sklearn.neighbors import NearestNeighbors
                            train_features = train_features_transformed[fold_idx]
                            knn_model = NearestNeighbors(n_neighbors=k_candidate, metric='euclidean')
                            knn_model.fit(train_features)
                            knn_models.append(knn_model)
                        
                        # Compute uncertainties on calibration set
                        # Use transformed calibration features
                        uncertainties_calib_per_fold = []
                        for fold_idx, knn_model in enumerate(knn_models):
                            calib_feats = calib_features_transformed[fold_idx]
                            distances, _ = knn_model.kneighbors(calib_feats)
                            # Uncertainty = mean distance to k nearest neighbors
                            uncertainty_fold = np.mean(distances, axis=1)  # [N_calib]
                            uncertainties_calib_per_fold.append(uncertainty_fold)
                        
                        # Average across folds for ensemble evaluation
                        uncertainties_calib = np.mean(uncertainties_calib_per_fold, axis=0)  # [N_calib]
                        
                        # Evaluate AUROC on calibration set
                        from sklearn.metrics import roc_auc_score
                        is_correct_calib = np.zeros(len(y_true_calib))
                        is_correct_calib[correct_idx_calib] = 1
                        
                        # Higher uncertainty = more likely to fail → invert for AUROC
                        auroc_calib = roc_auc_score(1 - is_correct_calib, uncertainties_calib)
                        
                        print(f" k={k_candidate}: AUROC={auroc_calib:.4f}")
                        
                        if auroc_calib > best_auroc:
                            best_auroc = auroc_calib
                            best_k = k_candidate
                    
                    k_selected = best_k
            
            # Print selection result after timer exits
            if best_auroc is not None:
                print(f" Selected k={k_selected} (AUROC={best_auroc:.4f}) in {timer_tuning.elapsed:.2f}s")
        
        # Fit final model with selected k
        with timer:
            knn_method = uq.KNNLatentMethod(layer_name=layer_name, k=k_selected)
            knn_method.fit(self.models, train_loaders, self.device)
            uncertainties = knn_method.compute(self.models, test_loader, self.device, 
                                              return_per_fold=per_fold_evaluation)
        
        # Use cached predictions if available, otherwise compute
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            y_scores = self._get_ensemble_predictions(test_loader)
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        # Get per-fold predictions if needed for per-fold metrics
        predictions_per_fold = None
        if uncertainties.ndim == 2:  # Per-fold uncertainties
            predictions_per_fold = self._get_per_fold_predictions(test_loader)
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true,
                                       predictions_per_fold=predictions_per_fold)
        metrics['time_seconds'] = timer.elapsed
        
        # Store selected k if tuning was performed
        if k_grid is not None:
            metrics['k_selected'] = k_selected
        
        # Store results
        if uncertainties.ndim == 2:
            # Per-fold: store averaged (for metrics), per-fold (for multi-curve plots), and ensemble reference
            averaged_uncertainties = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['KNN_Raw'] = averaged_uncertainties
            self._uncertainties['KNN_Raw_per_fold'] = uncertainties  # [num_folds, N]
            self._uncertainties['KNN_Raw_ensemble'] = averaged_uncertainties  # [N] for ensemble ROC baseline
            self._predictions_per_fold['KNN_Raw'] = predictions_per_fold  # [M, N]
        else:
            # Ensemble: store single uncertainty
            self._uncertainties['KNN_Raw'] = uncertainties
        self._results['KNN_Raw'] = metrics
        
        # Note: Z-score normalization now happens per-fold inside KNNLatentMethod.compute()
        # No post-hoc normalization needed
        
        return uncertainties, metrics
    
    def run_knn_shap(
        self,
        calib_loader: DataLoader,
        test_loader: DataLoader,
        train_loaders: List[DataLoader],
        y_true: np.ndarray,
        flag: str = 'dataset',
        layer_name: str = 'avgpool',
        k: int = 5,
        n_shap_features: int = 50,
        cache_dir: Optional[str] = None,
        parallel: bool = False,
        n_jobs: int = 2,
        per_fold_evaluation: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run KNN in SHAP-selected latent space.
        
        Args:
            calib_loader: Calibration data loader (for SHAP)
            test_loader: Test data loader (or single DataLoader if 1 model)
            train_loaders: List of train data loaders (one per model fold)
                          Or single DataLoader if models is a single model
            y_true: True labels [N]
            flag: Dataset identifier (for caching)
            layer_name: Layer name for feature extraction
            k: Number of nearest neighbors
            n_shap_features: Number of top SHAP features
            cache_dir: Directory to cache SHAP values
            parallel: Enable parallel processing across folds
            n_jobs: Number of parallel workers
            per_fold_evaluation: If True, compute per-fold metrics. If False, average uncertainties
        
        Returns:
            tuple: (uncertainties, metrics_dict)
                If per_fold_evaluation=True: uncertainties is [num_folds, N]
                Otherwise: uncertainties is [N] (averaged)
        """
        timer = Timer("KNN-SHAP computation")
        with timer:
            # Create method with parallelization
            knn_method = uq.KNNLatentSHAPMethod(
                layer_name=layer_name,
                k=k,
                n_shap_features=n_shap_features,
                cache_dir=cache_dir,
                parallel=parallel,
                n_jobs=n_jobs
            )
            
            # Fit and compute uncertainties
            knn_method.fit(self.models, train_loaders, calib_loader, self.device, flag=flag)
            uncertainties = knn_method.compute(self.models, test_loader, self.device,
                                              return_per_fold=per_fold_evaluation)
        
        # Use cached predictions if available, otherwise compute
        if self._test_predictions_cache is not None:
            correct_idx = self._test_predictions_cache['correct_idx']
            incorrect_idx = self._test_predictions_cache['incorrect_idx']
            y_pred = self._test_predictions_cache['y_pred']
        else:
            y_scores = self._get_ensemble_predictions(test_loader)
            y_pred = np.argmax(y_scores, axis=1)
            correct_idx = np.where(y_pred == y_true)[0]
            incorrect_idx = np.where(y_pred != y_true)[0]
        
        # Get per-fold predictions if needed for per-fold metrics
        predictions_per_fold = None
        if uncertainties.ndim == 2:  # Per-fold uncertainties
            predictions_per_fold = self._get_per_fold_predictions(test_loader)
        
        metrics = self._compute_metrics(uncertainties, correct_idx, incorrect_idx, y_pred, y_true,
                                       predictions_per_fold=predictions_per_fold)
        metrics['time_seconds'] = timer.elapsed
        
        # Store results
        if uncertainties.ndim == 2:
            # Per-fold: store averaged (for metrics), per-fold (for multi-curve plots), and ensemble reference
            averaged_uncertainties = np.mean(uncertainties, axis=0)  # [N] averaged across folds
            self._uncertainties['KNN_SHAP'] = averaged_uncertainties
            self._uncertainties['KNN_SHAP_per_fold'] = uncertainties  # [num_folds, N]
            self._uncertainties['KNN_SHAP_ensemble'] = averaged_uncertainties  # [N] for ensemble ROC baseline
            self._predictions_per_fold['KNN_SHAP'] = predictions_per_fold  # [M, N]
        else:
            # Ensemble: store single uncertainty
            self._uncertainties['KNN_SHAP'] = uncertainties
        self._results['KNN_SHAP'] = metrics
        
        # Note: Z-score normalization now happens per-fold inside KNNLatentSHAPMethod.compute()
        # No post-hoc normalization needed
        
        return uncertainties, metrics
    
    def _get_ensemble_predictions(self, data_loader: DataLoader) -> np.ndarray:
        """Helper to get ensemble predictions from data loader."""
        all_preds = []
        
        for model in self.models:
            model.eval()
            preds_fold = []
            
            with torch.no_grad():
                for batch in data_loader:
                    if isinstance(batch, dict):
                        images = batch['image']
                    else:
                        images = batch[0]
                    
                    images = images.to(self.device)
                    logits = model(images)
                    
                    # Handle both binary and multiclass
                    if logits.shape[1] == 1:
                        # Binary classification with single output
                        probs = torch.sigmoid(logits)  # [B, 1]
                        # Convert to 2-class format: [B, 2]
                        probs = torch.cat([1 - probs, probs], dim=1)
                    else:
                        # Multi-class
                        probs = torch.softmax(logits, dim=1)  # [B, C]
                    
                    preds_fold.append(probs.cpu().numpy())
            
            all_preds.append(np.concatenate(preds_fold, axis=0))
        
        # Average across models
        ensemble_preds = np.mean(all_preds, axis=0)
        return ensemble_preds
    
    def _compute_new_class_shift_indices(
        self,
        per_fold_mean_preds: np.ndarray,
        y_true_original: np.ndarray,
    ) -> Tuple[List, List, np.ndarray, np.ndarray]:
        """
        Derive stochastic-aware success/failure indices for new_class_shift evaluation.

        For each fold k, a sample is "correct" iff:
          - it is a *known*-class sample (y_true_original[i] != -1), AND
          - the stochastic mean prediction for that fold matches the original class label.
        New-class OOD samples (y_true_original == -1) are always failures.

        The ensemble level uses the *intersection* across all folds (unanimous stochastic
        agreement), mirroring the deterministic-ensemble logic for baseline methods.

        Args:
            per_fold_mean_preds: [M, N] argmax of mean stochastic softmax per fold
            y_true_original:     [N] original class labels; -1 for OOD/new-class samples

        Returns:
            per_fold_correct_idx   – list of M index arrays (correct per fold)
            per_fold_incorrect_idx – list of M index arrays (incorrect per fold)
            ensemble_correct_idx   – correct in ALL folds
            ensemble_incorrect_idx – complement of ensemble_correct_idx
        """
        M = per_fold_mean_preds.shape[0]
        known_mask = y_true_original != -1  # True = known-class sample

        ood_idx = np.where(~known_mask)[0]  # OOD samples — always incorrect, for all methods

        per_fold_correct_idx = []
        per_fold_incorrect_idx = []
        all_folds_correct = known_mask.copy()

        for k in range(M):
            fold_correct_mask = (per_fold_mean_preds[k] == y_true_original) & known_mask
            per_fold_correct_idx.append(np.where(fold_correct_mask)[0])
            per_fold_incorrect_idx.append(ood_idx)  # OOD only — not contaminated by hard known-class misclassifications
            all_folds_correct &= fold_correct_mask

        ensemble_correct_idx = np.where(all_folds_correct)[0]
        ensemble_incorrect_idx = ood_idx  # OOD only
        return per_fold_correct_idx, per_fold_incorrect_idx, ensemble_correct_idx, ensemble_incorrect_idx

    def _compute_metrics(
        self,
        uncertainties: np.ndarray,
        correct_idx: np.ndarray,
        incorrect_idx: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        predictions_per_fold: np.ndarray = None,
        ensemble_uncertainties: np.ndarray = None,
        use_cached_per_fold_indices: bool = True,
        per_fold_correct_idx_override: Optional[List] = None,
        per_fold_incorrect_idx_override: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics.
        
        Args:
            use_cached_per_fold_indices: If True (default), per-fold correct/incorrect indices
                are read from _test_predictions_cache (baseline predictions). Set to False for
                methods like TTA/GPS/MCDropout that define per-fold failures from their own
                mean predictions — in that case pass the method's mean predictions via
                predictions_per_fold and let compute_all_metrics_per_fold derive the indices.
            per_fold_correct_idx_override: When provided, replaces both the cached and
                the derived per-fold correct indices.  Used by stochastic methods under
                new_class_shift so that success/failure is judged by the stochastic mean
                prediction vs the original class label rather than by binary_gt.
            per_fold_incorrect_idx_override: Companion to per_fold_correct_idx_override.
        """
        # Check if uncertainties are per-fold [num_folds, N] or averaged [N]
        if uncertainties.ndim == 2:
            # Ensure per-fold predictions exist for fold-wise metrics.
            if predictions_per_fold is None:
                if self._per_fold_predictions_cache is not None:
                    predictions_per_fold = self._per_fold_predictions_cache
                elif self._test_predictions_cache is not None:
                    cached_pf = self._test_predictions_cache.get('per_fold_predictions')
                    if cached_pf is not None:
                        predictions_per_fold = cached_pf

            if predictions_per_fold is None:
                # Conservative fallback to keep execution alive; fold-level metrics will
                # effectively mirror ensemble predictions when true per-fold preds are unavailable.
                predictions_per_fold = np.tile(predictions[None, :], (uncertainties.shape[0], 1))

            # Get per-fold correct/incorrect indices from cache if available.
            # Skip when use_cached_per_fold_indices=False so that compute_all_metrics_per_fold
            # derives fold failures from predictions_per_fold (the method's own mean predictions).
            per_fold_correct_idx = None
            per_fold_incorrect_idx = None
            if use_cached_per_fold_indices and self._test_predictions_cache is not None:
                per_fold_correct_idx = self._test_predictions_cache.get('per_fold_correct_idx')
                per_fold_incorrect_idx = self._test_predictions_cache.get('per_fold_incorrect_idx')

            # Explicit overrides take priority (used by stochastic methods under new_class_shift).
            if per_fold_correct_idx_override is not None:
                per_fold_correct_idx = per_fold_correct_idx_override
            if per_fold_incorrect_idx_override is not None:
                per_fold_incorrect_idx = per_fold_incorrect_idx_override
            
            # Per-fold uncertainties: compute metrics per fold, then aggregate
            metrics = uq.compute_all_metrics_per_fold(
                uncertainties, predictions, labels, 
                predictions_per_fold=predictions_per_fold,
                ensemble_uncertainties=ensemble_uncertainties,
                per_fold_correct_idx=per_fold_correct_idx,
                per_fold_incorrect_idx=per_fold_incorrect_idx,
                ensemble_correct_idx=correct_idx,
                ensemble_incorrect_idx=incorrect_idx
            )
        else:
            # Single uncertainty array: compute standard metrics
            metrics = uq.compute_all_metrics(
                uncertainties, predictions, labels, correct_idx, incorrect_idx
            )
        return metrics
    
    def save_results(
        self, 
        output_dir: str,
        flag: str = None,
        timestamp: str = None,
        model_backbone: str = None,
        setup: str = None,
        corruption_info: str = None
    ) -> Dict[str, str]:
        """
        Save all computed uncertainties, evaluation plots, and results summary.
        
        Args:
            output_dir: Base directory for saving results
            flag: Dataset/experiment identifier (optional, for filenames)
            timestamp: Timestamp string (optional, will generate if not provided)
            model_backbone: Model architecture identifier (e.g., 'resnet18', 'vit_b_16')
            setup: Training setup identifier (e.g., '', 'DA', 'DO', 'DADO')
            
        Returns:
            Dictionary with paths to saved files:
                - 'metrics_file': Path to .npz file with all uncertainties
                - 'figures_dir': Directory containing evaluation plots
                - 'summary_file': Path to JSON results summary
        """
        from datetime import datetime
        import json
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if flag is None:
            flag = 'results'
        
        # Build filename suffix with model configuration
        config_suffix = ''
        if model_backbone:
            config_suffix += f'_{model_backbone}'
        if setup:
            config_suffix += f'_{setup}'
        if corruption_info:
            config_suffix += f'_corrupt_{corruption_info}'
        
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        
        # Save all metric values (uncertainties) to .npz
        if self._uncertainties:
            metrics_file = os.path.join(output_dir, f'all_metrics_{flag}{config_suffix}_{timestamp}.npz')
            np.savez_compressed(metrics_file, **self._uncertainties)
            saved_paths['metrics_file'] = metrics_file
            print(f"\nAll metric values saved to: {metrics_file}")
        
        # Generate and save evaluation plots for each method
        if self._uncertainties and self._test_predictions_cache is not None:
            figures_dir = os.path.join(output_dir, 'figures', flag, timestamp)
            os.makedirs(figures_dir, exist_ok=True)
            
            y_pred = self._test_predictions_cache['y_pred']
            y_true = self._test_predictions_cache['y_true']
            correct_idx = self._test_predictions_cache.get('correct_idx')
            incorrect_idx = self._test_predictions_cache.get('incorrect_idx')
            per_fold_correct_idx = self._test_predictions_cache.get('per_fold_correct_idx')
            per_fold_incorrect_idx = self._test_predictions_cache.get('per_fold_incorrect_idx')
            
            print(f"\nGenerating evaluation plots...")
            for method_name, metric_values in self._uncertainties.items():
                # Skip per-fold entries (they're paired with main entries)
                if (method_name.endswith('_per_fold') or 
                    method_name.endswith('_ensemble') or
                    method_name.endswith('_per_fold_internal')):
                    continue
                
                try:
                    # Check if per-fold data exists
                    per_fold_key = f'{method_name}_per_fold'
                    ensemble_key = f'{method_name}_ensemble'
                    
                    uncertainties_per_fold = None
                    ensemble_uncertainties = None
                    predictions_per_fold = None
                    
                    if per_fold_key in self._uncertainties:
                        uncertainties_per_fold = self._uncertainties[per_fold_key]
                    
                    if ensemble_key in self._uncertainties:
                        ensemble_uncertainties = self._uncertainties[ensemble_key]
                    
                    if method_name in self._predictions_per_fold:
                        predictions_per_fold = self._predictions_per_fold[method_name]
                    
                    fig_paths = uq.save_all_evaluation_plots(
                        uncertainties=metric_values,
                        predictions=y_pred,
                        labels=y_true,
                        method_name=method_name,
                        output_dir=figures_dir,
                        uncertainties_per_fold=uncertainties_per_fold,
                        ensemble_uncertainties=ensemble_uncertainties,
                        predictions_per_fold=predictions_per_fold,
                        model_backbone=model_backbone,
                        setup=setup,
                        corruption_info=corruption_info,
                        correct_idx=correct_idx,
                        incorrect_idx=incorrect_idx,
                        per_fold_correct_idx=per_fold_correct_idx,
                        per_fold_incorrect_idx=per_fold_incorrect_idx
                    )
                    print(f" {method_name}: {len(fig_paths)} plots saved")
                except Exception as e:
                    print(f" [WARNING] Failed to generate plots for {method_name}: {e}")
            
            saved_paths['figures_dir'] = figures_dir
            print(f"Figures saved to: {figures_dir}")
        
        # Save JSON summary
        if self._results and self._test_predictions_cache is not None:
            y_true = self._test_predictions_cache['y_true']
            correct_idx = self._test_predictions_cache['correct_idx']
            
            summary = {
                'flag': flag,
                'model_backbone': model_backbone,
                'setup': setup,
                'corruption_info': corruption_info,
                'timestamp': timestamp,
                'test_accuracy': float(len(correct_idx) / len(y_true)),
                'test_size': len(y_true),
                'methods': self._results,
                'metrics_file': saved_paths.get('metrics_file'),
                'figures_dir': saved_paths.get('figures_dir')
            }
            
            summary_file = os.path.join(output_dir, f'uq_benchmark_{flag}{config_suffix}_{timestamp}.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            saved_paths['summary_file'] = summary_file
            print(f"\nResults summary saved to: {summary_file}")
        
        return saved_paths
