"""
Evaluation metrics for uncertainty quantification methods.

Includes:
- AUROC_f: Area Under ROC Curve (for failure prediction)
- AURC: Area Under Risk-Coverage Curve
- AUGRC: Area Under Generalized Risk-Coverage Curve (Traub et al., NeurIPS 2024)

AUGRC measures the average rate of silent failures (undetected errors) across all working points.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pathlib import Path


def compute_auroc(uncertainties, correct_idx, incorrect_idx):
    """
    Compute AUROC for failure prediction (classification: correct vs incorrect).
    
    Args:
        uncertainties: Array of uncertainty scores [N]
        correct_idx: Indices of correctly classified samples
        incorrect_idx: Indices of incorrectly classified samples
    
    Returns:
        float: AUROC score (higher = better uncertainty quantification)
    """
    # Create binary labels: 1 = incorrect (failure), 0 = correct
    n_samples = len(uncertainties)
    labels = np.zeros(n_samples)
    labels[list(incorrect_idx)] = 1
    
    # AUROC: higher uncertainty should predict failures
    auroc = roc_auc_score(labels, uncertainties)
    
    return auroc


def compute_roc_curve(uncertainties, correct_idx, incorrect_idx):
    """
    Compute ROC curve for failure prediction.
    
    Args:
        uncertainties: Array of uncertainty scores [N]
        correct_idx: Indices of correctly classified samples
        incorrect_idx: Indices of incorrectly classified samples
    
    Returns:
        tuple: (fpr, tpr, thresholds, auroc)
    """
    n_samples = len(uncertainties)
    labels = np.zeros(n_samples)
    labels[list(incorrect_idx)] = 1
    
    fpr, tpr, thresholds = roc_curve(labels, uncertainties)
    auroc = roc_auc_score(labels, uncertainties)
    
    return fpr, tpr, thresholds, auroc


def compute_aurc(uncertainties, predictions, labels, num_bins=1000, correct_idx=None, incorrect_idx=None):
    """
    Compute Area Under Risk-Coverage Curve (AURC).
    
    Risk-Coverage curve plots classification error vs coverage, where samples
    are rejected in order of decreasing uncertainty. AURC measures the quality
    of uncertainty estimates for selective prediction.
    
    Args:
        uncertainties: Array of uncertainty scores [N] (higher = more uncertain)
        predictions: Array of predicted labels [N]
        labels: Array of true labels [N]
        num_bins: Number of coverage bins (default: 1000)
        correct_idx: Optional pre-computed indices of correct predictions
        incorrect_idx: Optional pre-computed indices of incorrect predictions
    
    Returns:
        float: AURC score (lower = better, range [0, 1])
        dict: Additional metrics including:
            - coverages: Coverage values
            - risks: Risk (error rate) at each coverage
            - optimal_risk: Risk when covering only correct predictions
    """
    n_samples = len(uncertainties)
    
    # Sort by uncertainty (descending: most uncertain first)
    sorted_indices = np.argsort(uncertainties)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Compute cumulative errors and coverage
    # Use pre-computed indices if provided (for new_class_shift where predictions are multi-class but labels are binary)
    if correct_idx is not None and incorrect_idx is not None:
        errors_array = np.zeros(n_samples, dtype=float)
        # Ensure indices are integer arrays (might be object dtype from cache)
        incorrect_idx_int = np.asarray(incorrect_idx, dtype=int)
        errors_array[incorrect_idx_int] = 1.0
        errors = errors_array[sorted_indices]
    else:
        errors = (sorted_predictions != sorted_labels).astype(float)
    
    # Coverage percentages — vectorized O(N) via reverse cumulative sum
    # cum_errors[i] = sum of errors[i:] (i.e. errors among the i-th to last kept samples)
    cum_errors = np.cumsum(errors[::-1])[::-1]          # [N]
    n_kept = np.arange(n_samples, 0, -1, dtype=float)   # [N, N-1, ..., 1]
    risks_main = cum_errors / n_kept                     # risk at each rejection threshold
    coverages_main = n_kept / n_samples                  # coverage at each threshold

    # Append the all-rejected point (coverage=0, risk=0)
    coverages = np.append(coverages_main, 0.0)
    risks = np.append(risks_main, 0.0)

    # Already monotone (coverage descending then 0); sort ascending for trapezoid
    sorted_idx = np.argsort(coverages)
    coverages = coverages[sorted_idx]
    risks = risks[sorted_idx]
    
    aurc = np.trapezoid(risks, coverages)
    
    # Optimal risk: if we could perfectly identify errors
    total_errors = errors.sum()
    optimal_risk = total_errors / n_samples
    
    return aurc, {
        'coverages': coverages,
        'risks': risks,
        'optimal_risk': optimal_risk,
        'n_samples': n_samples,
        'n_errors': int(total_errors)
    }


def compute_augrc(uncertainties, predictions, labels, num_bins=1000, correct_idx=None, incorrect_idx=None, _precomputed_aurc=None):
    """
    Compute Area Under Generalized Risk-Coverage Curve (AUGRC).
    
    From Traub et al. (NeurIPS 2024): "Overcoming Common Flaws in the Evaluation 
    of Selective Classification Systems"
    
    AUGRC evaluates selective classification in terms of the rate of silent failures
    (undetected errors) averaged over all working points. Unlike AURC, it uses the
    Generalized Risk P(fail, accept) instead of Selective Risk P(fail|accept).
    
    The formula is:
        AUGRC = (1 - AUROC_f) * acc * (1 - acc) + 0.5 * (1 - acc)^2
    
    Where:
    - AUROC_f: AUROC for failure prediction (how well uncertainties rank errors)
    - acc: Overall classification accuracy
    
    This metric:
    - Is bounded to [0, 0.5], where lower is better
    - Is directly interpretable as average risk of silent failures
    - Maintains monotonicity with respect to both AUROC_f and accuracy
    - Resolves the excessive weighting of high-confidence failures in AURC
    
    Args:
        uncertainties: Array of uncertainty scores [N] (higher = more uncertain)
        predictions: Array of predicted labels [N]
        labels: Array of true labels [N]
        num_bins: Number of coverage bins (default: 1000, not used but kept for compatibility)
        correct_idx: Optional pre-computed indices of correct predictions
        incorrect_idx: Optional pre-computed indices of incorrect predictions
    
    Returns:
        float: AUGRC score (0-0.5, lower is better)
        dict: Additional metrics including:
            - auroc_f: AUROC for failure prediction
            - acc: Overall accuracy
            - error_rate: Classification error rate (1 - acc)
            - aurc: Raw AURC value (for comparison)
            - coverages: Coverage values from risk-coverage curve
            - risks: Risk at each coverage
    """
    n_samples = len(uncertainties)
    
    # Use pre-computed indices if provided (for new_class_shift where predictions are multi-class but labels are binary)
    if correct_idx is not None and incorrect_idx is not None:
        n_errors = len(incorrect_idx)
    else:
        errors = (predictions != labels).astype(float)
        n_errors = errors.sum()
        correct_idx = np.where(~errors.astype(bool))[0]
        incorrect_idx = np.where(errors.astype(bool))[0]
    
    # Compute accuracy from error count and sample count
    acc = 1.0 - (n_errors / n_samples)
    error_rate = n_errors / n_samples
    
    # Compute AUROC_f (AUROC for failure prediction)
    if len(incorrect_idx) > 0 and len(correct_idx) > 0:
        auroc_f = compute_auroc(uncertainties, correct_idx, incorrect_idx)
    else:
        # Edge case: perfect model or completely wrong model
        auroc_f = 0.5  # Random baseline
    
    # Compute AUGRC using the formula from Traub et al. (Equation 7)
    # AUGRC = (1 - AUROC_f) * acc * (1 - acc) + 0.5 * (1 - acc)^2
    term1 = (1.0 - auroc_f) * acc * (1.0 - acc)
    term2 = 0.5 * (1.0 - acc) ** 2
    augrc = term1 + term2
    
    # Also compute AURC for comparison (legacy metric)
    # Accepts pre-computed result to avoid a redundant O(N) pass when called from compute_all_metrics
    if _precomputed_aurc is not None:
        aurc, aurc_metrics = _precomputed_aurc
    else:
        aurc, aurc_metrics = compute_aurc(uncertainties, predictions, labels, num_bins, correct_idx, incorrect_idx)

    return augrc, {
        'auroc_f': float(auroc_f),
        'acc': float(acc),
        'error_rate': float(error_rate),
        'n_errors': int(n_errors),
        'aurc': float(aurc),  # For comparison with old metric
        'coverages': aurc_metrics['coverages'],
        'risks': aurc_metrics['risks']
    }


def compute_all_metrics(uncertainties, predictions, labels, correct_idx=None, incorrect_idx=None):
    """
    Compute all UQ evaluation metrics.
    
    Args:
        uncertainties: Array of uncertainty scores [N] (higher = more uncertain)
        predictions: Array of predicted labels [N]
        labels: Array of true labels [N]
        correct_idx: Optional indices of correct predictions (computed if None)
        incorrect_idx: Optional indices of incorrect predictions (computed if None)
    
    Returns:
        dict: All metrics including AUROC_f, AURC, AUGRC, and auxiliary information
    """
    # Compute correct/incorrect indices if not provided
    if correct_idx is None or incorrect_idx is None:
        errors = (predictions != labels)
        correct_idx = np.where(~errors)[0]
        incorrect_idx = np.where(errors)[0]

    # Filter to only labeled samples when excluded samples exist (e.g., new_class_shift
    # "hard" known-class samples that are neither correct nor OOD). Without this,
    # excluded samples get label=0 in AUROC, degrading the score.
    correct_idx_arr = np.asarray(correct_idx, dtype=int)
    incorrect_idx_arr = np.asarray(incorrect_idx, dtype=int)
    n_relevant = len(correct_idx_arr) + len(incorrect_idx_arr)
    if n_relevant < len(uncertainties):
        relevant_idx = np.concatenate([correct_idx_arr, incorrect_idx_arr])
        uncertainties = uncertainties[relevant_idx]
        if predictions is not None:
            predictions = predictions[relevant_idx]
        if labels is not None:
            labels = labels[relevant_idx]
        correct_idx = np.arange(len(correct_idx_arr))
        incorrect_idx = np.arange(len(correct_idx_arr), n_relevant)

    # AUROC_f (for failure prediction)
    if len(incorrect_idx) > 0 and len(correct_idx) > 0:
        auroc_f = compute_auroc(uncertainties, correct_idx, incorrect_idx)
    else:
        auroc_f = np.nan
    
    # AURC and AUGRC — compute_aurc only once, pass result into compute_augrc to avoid double work
    aurc, aurc_metrics = compute_aurc(uncertainties, predictions, labels, correct_idx=correct_idx, incorrect_idx=incorrect_idx)
    augrc, augrc_metrics = compute_augrc(uncertainties, predictions, labels, correct_idx=correct_idx, incorrect_idx=incorrect_idx,
                                         _precomputed_aurc=(aurc, aurc_metrics))
    
    return {
        'auroc_f': float(auroc_f),
        'aurc': float(aurc),
        'augrc': float(augrc),
        'accuracy': float(augrc_metrics['acc']),
        'error_rate': float(augrc_metrics['error_rate']),
        'n_correct': len(correct_idx),
        'n_incorrect': len(incorrect_idx),
        'n_total': len(uncertainties),
    }


def compute_all_metrics_per_fold(uncertainties_per_fold, predictions, labels, predictions_per_fold=None, ensemble_uncertainties=None,
                                per_fold_correct_idx=None, per_fold_incorrect_idx=None,
                                ensemble_correct_idx=None, ensemble_incorrect_idx=None):
    """
    Compute metrics independently for each fold, then aggregate with mean±std.
    
    This is the CORRECT way to evaluate UQ methods in a cross-validation setting:
    - Each fold represents an independent model/experiment
    - Compute metrics on each fold separately
    - Report mean±std across folds to capture model variance
    
    Args:
        uncertainties_per_fold: Array of uncertainty scores [num_folds, N]
        predictions: Array of predicted labels [N] (ensemble predictions, used if predictions_per_fold=None)
        labels: Array of true labels [N]
        predictions_per_fold: Optional [num_folds, N] array of per-fold predictions (CORRECT approach)
        ensemble_uncertainties: Optional [N] array of TRUE ensemble uncertainties (for correct auroc_f)
        per_fold_correct_idx: Optional list of correct indices for each fold (avoids recomputation from predictions)
        per_fold_incorrect_idx: Optional list of incorrect indices for each fold (avoids recomputation from predictions)
        ensemble_correct_idx: Optional correct indices for ensemble (avoids recomputation from predictions)
        ensemble_incorrect_idx: Optional incorrect indices for ensemble (avoids recomputation from predictions)
    
    Returns:
        dict: Aggregated metrics with mean and std:
            - auroc_f: TRUE ensemble AUROC_f (from ensemble_uncertainties)
            - auroc_f_mean: mean of per-fold AUROC_f values
            - aurc, augrc: from ensemble
            - accuracy (from ensemble predictions)
            - per_fold_metrics: List of dicts with metrics for each fold
    """
    num_folds = uncertainties_per_fold.shape[0]
    
    # Compute correct/incorrect for ensemble predictions
    # Use pre-computed if provided, otherwise compute from predictions vs labels
    if ensemble_correct_idx is not None and ensemble_incorrect_idx is not None:
        correct_idx_ensemble = ensemble_correct_idx
        incorrect_idx_ensemble = ensemble_incorrect_idx
    else:
        errors_ensemble = (predictions != labels)
        correct_idx_ensemble = np.where(~errors_ensemble)[0]
        incorrect_idx_ensemble = np.where(errors_ensemble)[0]
    
    # Compute metrics for each fold independently
    per_fold_metrics = []
    auroc_f_list = []
    aurc_list = []
    augrc_list = []
    
    for fold_idx in range(num_folds):
        fold_uncertainties = uncertainties_per_fold[fold_idx]
    
        fold_predictions = predictions_per_fold[fold_idx]
        
        # Use pre-computed indices if provided, otherwise compute from predictions
        if per_fold_correct_idx is not None and per_fold_incorrect_idx is not None:
            fold_correct_idx = per_fold_correct_idx[fold_idx]
            fold_incorrect_idx = per_fold_incorrect_idx[fold_idx]
        else:
            fold_errors = (fold_predictions != labels)
            fold_correct_idx = np.where(~fold_errors)[0]
            fold_incorrect_idx = np.where(fold_errors)[0]

        # Compute metrics for this fold
        fold_metrics = compute_all_metrics(
            fold_uncertainties, fold_predictions, labels,
            fold_correct_idx, fold_incorrect_idx
        )
        
        per_fold_metrics.append(fold_metrics)
        auroc_f_list.append(fold_metrics['auroc_f'])
        aurc_list.append(fold_metrics['aurc'])
        augrc_list.append(fold_metrics['augrc'])
    
    # Aggregate: mean and std across folds
    auroc_f_array = np.array(auroc_f_list)
    aurc_array = np.array(aurc_list)
    augrc_array = np.array(augrc_list)
    
    # Filter out NaN values for statistics (edge cases)
    auroc_f_valid = auroc_f_array[~np.isnan(auroc_f_array)]
    aurc_valid = aurc_array[~np.isnan(aurc_array)]
    augrc_valid = augrc_array[~np.isnan(augrc_array)]
    
    # Compute TRUE ensemble metrics (from ensemble_uncertainties if provided)
    # Otherwise fall back to averaged per-fold (legacy behavior)
    if ensemble_uncertainties is not None:
        # CORRECT: Use TRUE ensemble uncertainties
        ensemble_metrics = compute_all_metrics(
            ensemble_uncertainties, predictions, labels,
            correct_idx_ensemble, incorrect_idx_ensemble
        )
    else:
        # LEGACY: Fall back to averaged per-fold
        averaged_uncertainties = np.mean(uncertainties_per_fold, axis=0)  # [N]
        ensemble_metrics = compute_all_metrics(
            averaged_uncertainties, predictions, labels,
            correct_idx_ensemble, incorrect_idx_ensemble
        )
    
    return {
        # PRIMARY: TRUE ensemble metrics (from ensemble_uncertainties)
        # auroc_f uses TRUE ensemble, auroc_f_mean uses per-fold average
        'auroc_f': ensemble_metrics['auroc_f'],
        'aurc': ensemble_metrics['aurc'],
        'augrc': ensemble_metrics['augrc'],
        'accuracy': ensemble_metrics['accuracy'],
        'error_rate': ensemble_metrics['error_rate'],
        'n_correct': ensemble_metrics['n_correct'],
        'n_incorrect': ensemble_metrics['n_incorrect'],
        'n_total': ensemble_metrics['n_total'],
        
        # SECONDARY: Per-fold statistics (for understanding model variance)
        'auroc_f_mean': float(np.mean(auroc_f_valid)) if len(auroc_f_valid) > 0 else np.nan,
        'auroc_f_std': float(np.std(auroc_f_valid)) if len(auroc_f_valid) > 0 else np.nan,
        'aurc_mean': float(np.mean(aurc_valid)) if len(aurc_valid) > 0 else np.nan,
        'aurc_std': float(np.std(aurc_valid)) if len(aurc_valid) > 0 else np.nan,
        'augrc_mean': float(np.mean(augrc_valid)) if len(augrc_valid) > 0 else np.nan,
        'augrc_std': float(np.std(augrc_valid)) if len(augrc_valid) > 0 else np.nan,
        
        # Metadata
        'num_folds': num_folds,
        'per_fold_metrics': per_fold_metrics,  # Full details per fold
    }


def plot_risk_coverage_curve(uncertainties, predictions, labels, ax=None, 
                              show_optimal=True, save_path=None, 
                              uncertainties_per_fold=None, ensemble_uncertainties=None,
                              predictions_per_fold=None, correct_idx=None, incorrect_idx=None,
                              per_fold_correct_idx=None, per_fold_incorrect_idx=None):
    """
    Plot Risk-Coverage curve (Selective Risk) and Generalized Risk.
    
    Baselines:
    - Oracle: Perfect UQ that ranks ALL errors as most uncertain (AUROC_f=1.0)
    
    Args:
        uncertainties: Array of uncertainty scores [N] (main uncertainties to plot)
        predictions: Array of predicted labels [N] (ensemble predictions)
        labels: Array of true labels [N]
        ax: Matplotlib axis (creates new figure if None)
        show_optimal: Whether to show oracle baseline (perfect ranking)
        save_path: Path to save figure (optional)
        uncertainties_per_fold: Optional [num_folds, N] array for per-fold visualization
        ensemble_uncertainties: Optional [N] array for ensemble reference (when per-fold provided)
        predictions_per_fold: Optional [num_folds, N] array of per-fold predictions
        correct_idx: Optional pre-computed indices of correct predictions
        incorrect_idx: Optional pre-computed indices of incorrect predictions
        per_fold_correct_idx: Optional list of per-fold correct indices
        per_fold_incorrect_idx: Optional list of per-fold incorrect indices
    
    Returns:
        matplotlib figure and axis
    """
    
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig = ax.get_figure()
        ax1 = ax
        ax2 = None
    
    # ===== Plot 1: Selective Risk (standard risk-coverage curve) =====
    
    # Plot per-fold curves if provided (semi-transparent)
    fold_aurc_list = []
    fold_augrc_list = []
    if uncertainties_per_fold is not None:
        num_folds = uncertainties_per_fold.shape[0]
        for fold_idx in range(num_folds):
            fold_uncertainties = uncertainties_per_fold[fold_idx]
            
            # Priority 1: Use pre-computed per-fold indices (for new_class_shift)
            if per_fold_correct_idx is not None and per_fold_incorrect_idx is not None:
                fold_correct_idx = per_fold_correct_idx[fold_idx]
                fold_incorrect_idx = per_fold_incorrect_idx[fold_idx]
                # For new_class_shift, predictions_per_fold is available but shouldn't be used for comparison
                fold_predictions = predictions_per_fold[fold_idx] if predictions_per_fold is not None else predictions
            else:
                fold_correct_idx = None
                fold_incorrect_idx = None
                # Priority 2: Use per-fold predictions if available, otherwise use ensemble predictions
                fold_predictions = predictions_per_fold[fold_idx] if predictions_per_fold is not None else predictions
            
            aurc_fold, metrics_fold = compute_aurc(fold_uncertainties, fold_predictions, labels, 
                                                   correct_idx=fold_correct_idx, incorrect_idx=fold_incorrect_idx)
            augrc_fold, _ = compute_augrc(fold_uncertainties, fold_predictions, labels,
                                         correct_idx=fold_correct_idx, incorrect_idx=fold_incorrect_idx)
            fold_aurc_list.append(aurc_fold)
            fold_augrc_list.append(augrc_fold)
            
            coverages_fold = metrics_fold['coverages']
            risks_fold = metrics_fold['risks']
            ax1.plot(coverages_fold, risks_fold, '-', alpha=0.3, linewidth=1.5, 
                    color='cornflowerblue')
    
    # Compute curves for main uncertainties (ensemble)
    aurc, metrics = compute_aurc(uncertainties, predictions, labels, correct_idx=correct_idx, incorrect_idx=incorrect_idx)
    augrc, augrc_metrics = compute_augrc(uncertainties, predictions, labels, correct_idx=correct_idx, incorrect_idx=incorrect_idx)
    
    coverages = metrics['coverages']
    risks = metrics['risks']
    error_rate = augrc_metrics['error_rate']
    auroc_f = augrc_metrics['auroc_f']
    acc = augrc_metrics['acc']
    n_samples = len(uncertainties)
    n_errors = int(augrc_metrics['n_errors'])
    
    # Compute generalized risk: P(fail, accept) = P(fail | accept) * P(accept)
    # P(fail | accept) = risks, P(accept) = coverages
    generalized_risks = risks * coverages
    
    # Plot ensemble curve with per-fold statistics in legend
    if uncertainties_per_fold is not None and len(fold_aurc_list) > 0:
        mean_aurc = np.mean(fold_aurc_list)
        std_aurc = np.std(fold_aurc_list)
        label_main = f'Ensemble (AURC={aurc:.6f})\nPer-fold: {mean_aurc:.6f}±{std_aurc:.6f}'
    else:
        label_main = f'UQ Method (AURC={aurc:.6f})'
    
    ax1.plot(coverages, risks, 'g-', linewidth=2.5, label=label_main)
    # Shade the area under the curve (this is AURC)
    ax1.fill_between(coverages, 0, risks, alpha=0.2, color='green')
    
    # Plot oracle baseline (perfect ranking of failures)
    if show_optimal and n_errors > 0 and n_errors < n_samples:
        # Oracle: accept correct predictions first (low risk), then errors (risk increases)
        # At coverage (1 - error_rate): only correct predictions, risk = 0
        # At coverage 1.0: all predictions accepted, risk = error_rate
        cov_correct = (n_samples - n_errors) / n_samples
        oracle_cov = np.array([0.001, cov_correct, 1.0])  # Small epsilon to avoid division issues
        oracle_risk = np.array([0, 0, error_rate])
        
        # Compute Oracle AURC using trapezoidal integration
        oracle_aurc = np.trapezoid(oracle_risk, oracle_cov)
        
        ax1.plot(oracle_cov, oracle_risk, 'g--', linewidth=2, 
                label=f'Oracle (AURC={oracle_aurc:.6f})')
    
    ax1.set_xlabel('Coverage (fraction of predictions accepted)', fontsize=13)
    ax1.set_ylabel('Selective Risk: P(fail | accept)', fontsize=13)
    ax1.set_title('Risk-Coverage Curve\n(Error rate among accepted predictions)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, max(max(risks) * 1.1, error_rate * 1.1) if len(risks) > 0 and max(risks) > 0 else 1])
    
    # ===== Plot 2: Generalized Risk (rate of silent failures) =====
    if ax2 is not None:
        # Plot per-fold curves if provided (semi-transparent)
        if uncertainties_per_fold is not None:
            num_folds = uncertainties_per_fold.shape[0]
            for fold_idx in range(num_folds):
                fold_uncertainties = uncertainties_per_fold[fold_idx]
                
                # Use per-fold predictions if available
                if predictions_per_fold is not None:
                    fold_predictions = predictions_per_fold[fold_idx]
                else:
                    fold_predictions = predictions
                
                aurc_fold, metrics_fold = compute_aurc(fold_uncertainties, fold_predictions, labels)
                coverages_fold = metrics_fold['coverages']
                risks_fold = metrics_fold['risks']
                gen_risks_fold = risks_fold * coverages_fold
                ax2.plot(coverages_fold, gen_risks_fold, '-', alpha=0.3, linewidth=1.5,
                        color='cornflowerblue')
        
        # Plot ensemble curve with per-fold statistics
        if uncertainties_per_fold is not None and len(fold_augrc_list) > 0:
            mean_augrc = np.mean(fold_augrc_list)
            std_augrc = np.std(fold_augrc_list)
            label_augrc = f'Ensemble (AUGRC={augrc:.6f})\nPer-fold: {mean_augrc:.6f}±{std_augrc:.6f}'
        else:
            label_augrc = f'UQ Method (AUGRC={augrc:.6f})'
        
        ax2.plot(coverages, generalized_risks, 'g-', linewidth=2.5, label=label_augrc)
        
        # Shade the area under the curve (this is AUGRC)
        ax2.fill_between(coverages, 0, generalized_risks, alpha=0.2, color='green')
        
        # Oracle baseline for generalized risk
        if show_optimal and n_errors > 0 and n_errors < n_samples:
            # Oracle rejects all errors first → no silent failures until we accept them
            # Generalized risk = P(fail, accept) = risk * coverage
            # For oracle: risk=0 until coverage exceeds fraction of correct predictions
            cov_correct = (n_samples - n_errors) / n_samples
            oracle_gen_cov = np.array([0, cov_correct, 1.0])
            oracle_gen_risk = np.array([0, 0, error_rate * 1.0])  # Only errors accepted at end
            oracle_augrc = 0.5 * (1.0 - acc) ** 2
            ax2.plot(oracle_gen_cov, oracle_gen_risk, 'g--', linewidth=2, 
                    label=f'Oracle (AUGRC={oracle_augrc:.6f})')
        
        ax2.set_xlabel('Coverage (fraction of predictions accepted)', fontsize=13)
        ax2.set_ylabel('Generalized Risk: P(fail, accept)', fontsize=13)
        ax2.set_title('Generalized Risk-Coverage Curve\n(Absolute rate of silent failures across dataset)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        max_gen_risk = max(generalized_risks) if len(generalized_risks) > 0 else error_rate
        ax2.set_ylim([0, max(max_gen_risk * 1.1, error_rate * 0.5)])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved risk-coverage curves to {save_path}")
    
    return fig, (ax1, ax2)


def plot_roc_curve_failure_prediction(uncertainties, predictions, labels,
                                       method_name='UQ Method', save_path=None,
                                       uncertainties_per_fold=None, ensemble_uncertainties=None,
                                       predictions_per_fold=None, correct_idx=None, incorrect_idx=None,
                                       per_fold_correct_idx=None, per_fold_incorrect_idx=None):
    """
    Plot ROC curve for failure prediction (AUROC_f).
    
    This shows how well the uncertainty scores separate correct from incorrect predictions.
    
    Baselines:
    - Oracle (perfect): AUROC_f = 1.0, separates all failures perfectly
    - Random: AUROC_f = 0.5, diagonal line (no discrimination)
    
    Args:
        uncertainties: Array of uncertainty scores [N] (higher = more uncertain)
        predictions: Array of predicted labels [N] (ensemble predictions)
        labels: Array of true labels [N]
        method_name: Name of the UQ method for plot title
        save_path: Path to save figure (optional)
        uncertainties_per_fold: Optional [num_folds, N] array for per-fold visualization
        ensemble_uncertainties: Optional [N] array for ensemble reference
        predictions_per_fold: Optional [num_folds, N] array of per-fold predictions
        correct_idx: Optional pre-computed indices of correct predictions
        incorrect_idx: Optional pre-computed indices of incorrect predictions
        per_fold_correct_idx: Optional list of per-fold correct indices
        per_fold_incorrect_idx: Optional list of per-fold incorrect indices
    
    Returns:
        matplotlib figure
    """
    # Compute ROC for ensemble predictions (reference)
    # Use pre-computed indices if provided (for new_class_shift)
    if correct_idx is not None and incorrect_idx is not None:
        pass  # Already have indices
    else:
        errors = (predictions != labels)
        correct_idx = np.where(~errors)[0]
        incorrect_idx = np.where(errors)[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Compute average ROC curve if per-fold data is available
    # NOTE: This computes the EXPECTED ROC performance if you randomly pick one fold
    #       Each fold ROC uses that fold's uncertainties + predictions
    #       The averaged curve interpolates TPR values at common FPR points
    #       This is different from the Ensemble curve (uses ensemble predictions)
    if uncertainties_per_fold is not None:
        num_folds = uncertainties_per_fold.shape[0]
        
        # Collect all fold ROC curves
        fold_fprs = []
        fold_tprs = []
        fold_aurocs = []
        
        # If per-fold predictions available, use them (CORRECT approach)
        # Otherwise fall back to ensemble predictions (legacy, less accurate)
        for fold_idx in range(num_folds):
            fold_uncertainties = uncertainties_per_fold[fold_idx]
            
            # Priority 1: Use pre-computed per-fold indices (for new_class_shift)
            if per_fold_correct_idx is not None and per_fold_incorrect_idx is not None:
                fold_correct_idx = per_fold_correct_idx[fold_idx]
                fold_incorrect_idx = per_fold_incorrect_idx[fold_idx]
            elif predictions_per_fold is not None:
                # Priority 2: Compute from fold's own predictions
                fold_predictions = predictions_per_fold[fold_idx]
                fold_errors = (fold_predictions != labels)
                fold_correct_idx = np.where(~fold_errors)[0]
                fold_incorrect_idx = np.where(fold_errors)[0]
            else:
                # Priority 3: LEGACY - Use ensemble predictions (less accurate)
                fold_correct_idx = correct_idx
                fold_incorrect_idx = incorrect_idx
            
            fpr_fold, tpr_fold, _, auroc_fold = compute_roc_curve(
                fold_uncertainties, fold_correct_idx, fold_incorrect_idx
            )
            
            # Store for averaging
            fold_fprs.append(fpr_fold)
            fold_tprs.append(tpr_fold)
            fold_aurocs.append(auroc_fold)
            
            # Plot individual fold (semi-transparent)
            ax.plot(fpr_fold, tpr_fold, '-', alpha=0.6, linewidth=1.5,
                   color='cornflowerblue')
        
        # Compute average ROC curve by interpolating to common FPR points
        # This is the CORRECT way to average ROC curves
        mean_fpr = np.linspace(0, 1, 100)
        interpolated_tprs = []
        
        for fpr_fold, tpr_fold in zip(fold_fprs, fold_tprs):
            # Interpolate this fold's TPR to common FPR points
            interp_tpr = np.interp(mean_fpr, fpr_fold, tpr_fold)
            interpolated_tprs.append(interp_tpr)
        
        # Compute summary statistics
        mean_auroc = np.mean(fold_aurocs)
        std_auroc = np.std(fold_aurocs)
        
        # Plot TRUE ensemble curve (this matches JSON's auroc_f)
        # Use ensemble_uncertainties if provided, otherwise fall back to uncertainties
        if ensemble_uncertainties is not None:
            # TRUE ensemble (matches JSON auroc_f)
            fpr_ens, tpr_ens, _, auroc_ens = compute_roc_curve(
                ensemble_uncertainties, correct_idx, incorrect_idx
            )
        else:
            # Fall back to averaged per-fold (legacy)
            fpr_ens, tpr_ens, _, auroc_ens = compute_roc_curve(
                uncertainties, correct_idx, incorrect_idx
            )
        ax.plot(fpr_ens, tpr_ens, 'g-', linewidth=2.5, label='Ensemble', alpha=0.8)
        
        # Add text annotation with mean±std and ensemble
        text_lines = [f'Per-fold mean: {mean_auroc:.4f}±{std_auroc:.4f}']
        text_lines.append(f'Ensemble: {auroc_ens:.4f}')
        text_lines.append(f'({num_folds} folds)')
        
        ax.text(0.98, 0.12, '\n'.join(text_lines),
                transform=ax.transAxes, fontsize=11, verticalalignment='bottom', 
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Single curve (no folds)
        fpr, tpr, thresholds, auroc_f = compute_roc_curve(uncertainties, correct_idx, incorrect_idx)
        ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'AUROC_f = {auroc_f:.4f}')
    
    # Plot diagonal (random baseline)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.7, label='Random')
    
    # Styling
    ax.set_xlabel('False Positive Rate\n(Fraction of correct predictions flagged as uncertain)', fontsize=12)
    ax.set_ylabel('True Positive Rate\n(Fraction of incorrect predictions flagged as uncertain)', fontsize=12)
    ax.set_title(f'ROC Curve for Failure Prediction\n{method_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    
    return fig


def plot_uncertainty_distributions(uncertainties, predictions, labels, 
                                   method_name='UQ Method', save_path=None,
                                   uncertainties_per_fold=None, ensemble_uncertainties=None,
                                   predictions_per_fold=None, correct_idx=None, incorrect_idx=None,
                                   per_fold_correct_idx=None, per_fold_incorrect_idx=None):
    """
    Plot uncertainty score distributions for correct vs incorrect predictions.
    
    Creates:
    1. Boxplot comparing distributions (with per-fold overlays if provided)
    2. Histogram overlays (with per-fold overlays if provided)
    3. Summary statistics
    
    Args:
        uncertainties: Array of uncertainty scores [N]
        predictions: Array of predicted labels [N] (ensemble predictions, used if predictions_per_fold=None)
        labels: Array of true labels [N]
        method_name: Name of the UQ method for plot title
        save_path: Path to save figure (optional)
        uncertainties_per_fold: Optional [num_folds, N] array for per-fold visualization
        ensemble_uncertainties: Optional [N] array for ensemble reference
        predictions_per_fold: Optional [num_folds, N] array of per-fold predictions (CORRECT for per-fold evaluation)
        correct_idx: Optional pre-computed indices of correct predictions
        incorrect_idx: Optional pre-computed indices of incorrect predictions
        per_fold_correct_idx: Optional list of per-fold correct indices
        per_fold_incorrect_idx: Optional list of per-fold incorrect indices
    
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # ===== Plot 1: Boxplot =====
    ax1 = axes[0]
    
    if uncertainties_per_fold is not None and predictions_per_fold is not None:
        # Per-fold boxplots: show separate boxplot for each fold
        num_folds = uncertainties_per_fold.shape[0]
        
        # Compute per-fold AUROC_f values
        fold_auroc_list = []
        boxplot_data = []
        boxplot_labels = []
        colors_list = []
        
        for fold_idx in range(num_folds):
            fold_unc = uncertainties_per_fold[fold_idx]
            
            # Priority 1: Use pre-computed per-fold indices (for new_class_shift)
            if per_fold_correct_idx is not None and per_fold_incorrect_idx is not None:
                fold_correct_idx = np.asarray(per_fold_correct_idx[fold_idx], dtype=int)
                fold_incorrect_idx = np.asarray(per_fold_incorrect_idx[fold_idx], dtype=int)
            elif predictions_per_fold is not None:
                # Priority 2: Compute from fold's own predictions
                fold_predictions = predictions_per_fold[fold_idx]
                fold_errors = (fold_predictions != labels)
                fold_correct_idx = np.where(~fold_errors)[0]
                fold_incorrect_idx = np.where(fold_errors)[0]
            else:
                # Priority 3: Fallback (shouldn't happen if per-fold data provided)
                raise ValueError("Per-fold uncertainties provided but no way to determine correct/incorrect")
            
            fold_auroc = compute_auroc(fold_unc, fold_correct_idx, fold_incorrect_idx)
            fold_auroc_list.append(fold_auroc)
            
            # Add data for boxplot (correct and incorrect for this fold)
            boxplot_data.extend([fold_unc[fold_correct_idx], fold_unc[fold_incorrect_idx]])
            boxplot_labels.extend([f'Fold {fold_idx+1}\nCorrect', f'Fold {fold_idx+1}\nIncorrect'])
            colors_list.extend(['lightgreen', 'lightcoral'])
        
        # Create boxplots
        positions = list(range(1, len(boxplot_data) + 1))
        bp = ax1.boxplot(boxplot_data,
                        positions=positions,
                        labels=boxplot_labels,
                        widths=0.6,
                        patch_artist=True,
                        medianprops=dict(color='darkred', linewidth=2),
                        showfliers=False)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Compute mean AUROC_f across folds
        mean_auroc = np.mean(fold_auroc_list)
        std_auroc = np.std(fold_auroc_list)
        auroc_f = mean_auroc  # For display
        
        # Add vertical separators between folds
        for i in range(1, num_folds):
            ax1.axvline(x=i*2 + 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Compute overall statistics for text display
        # Use pre-computed indices if provided
        if correct_idx is not None and incorrect_idx is not None:
            # Ensure indices are integer arrays (might be object dtype from cache)
            correct_idx = np.asarray(correct_idx, dtype=int)
            incorrect_idx = np.asarray(incorrect_idx, dtype=int)
        else:
            errors = (predictions != labels)
            correct_idx = ~errors
            incorrect_idx = errors
        unc_correct = uncertainties[correct_idx]
        unc_incorrect = uncertainties[incorrect_idx]
    else:
        # Single boxplot (no per-fold data)
        # Use pre-computed indices if provided
        if correct_idx is not None and incorrect_idx is not None:
            # Ensure indices are integer arrays (might be object dtype from cache)
            correct_idx = np.asarray(correct_idx, dtype=int)
            incorrect_idx = np.asarray(incorrect_idx, dtype=int)
        else:
            errors = (predictions != labels)
            correct_idx = ~errors
            incorrect_idx = errors
        
        unc_correct = uncertainties[correct_idx]
        unc_incorrect = uncertainties[incorrect_idx]
        
        auroc_f = compute_auroc(uncertainties, correct_idx, incorrect_idx) if len(unc_incorrect) > 0 and len(unc_correct) > 0 else np.nan
        
        box_data = [unc_correct, unc_incorrect]
        box_labels = [f'Correct\n(n={len(unc_correct)})', 
                      f'Incorrect\n(n={len(unc_incorrect)})']
        
        bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True,
                         notch=True, showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                         medianprops=dict(linewidth=2, color='black'))
        
        # Color the boxes
        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax1.set_ylabel('Uncertainty Score', fontsize=13)
    if uncertainties_per_fold is not None and predictions_per_fold is not None:
        title_suffix = f' (Per-Fold: Mean AUROC_f = {mean_auroc:.4f}±{std_auroc:.4f})'
    else:
        title_suffix = ''
    ax1.set_title(f'{method_name}\nUncertainty Distribution by Prediction Outcome{title_suffix}', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add statistics text
    if uncertainties_per_fold is not None and predictions_per_fold is not None:
        stats_text = f'Per-fold AUROC_f:\n  Mean: {mean_auroc:.4f}\n  Std: {std_auroc:.4f}\n\n'
        stats_text += f'Overall (ensemble):\n'
        stats_text += f'  Correct mean: {unc_correct.mean():.3f}\n'
        stats_text += f'  Incorrect mean: {unc_incorrect.mean():.3f}'
    else:
        stats_text = f'AUROC_f = {auroc_f:.4f}\n\n'
        stats_text += f'Correct:\n  Mean: {unc_correct.mean():.3f}\n  Median: {np.median(unc_correct):.3f}\n'
        stats_text += f'  Std: {unc_correct.std():.3f}\n\n'
        stats_text += f'Incorrect:\n  Mean: {unc_incorrect.mean():.3f}\n  Median: {np.median(unc_incorrect):.3f}\n'
        stats_text += f'  Std: {unc_incorrect.std():.3f}'
    
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== Plot 2: Histogram overlay =====
    ax2 = axes[1]
    
    # Determine bins
    bins = np.linspace(min(uncertainties.min(), 0), 
                      uncertainties.max(), 50)
    
    # Plot per-fold histograms if provided (very transparent)
    if uncertainties_per_fold is not None:
        num_folds = uncertainties_per_fold.shape[0]
        for fold_idx in range(num_folds):
            fold_unc = uncertainties_per_fold[fold_idx]
            
            # Use per-fold predictions if available (CORRECT approach)
            if predictions_per_fold is not None:
                fold_predictions = predictions_per_fold[fold_idx]
                fold_errors = (fold_predictions != labels)
                fold_correct_idx = ~fold_errors
                fold_incorrect_idx = fold_errors
            else:
                # Fallback to averaged correct/incorrect indices
                fold_correct_idx = correct_idx
                fold_incorrect_idx = incorrect_idx
            
            fold_correct = fold_unc[fold_correct_idx]
            fold_incorrect = fold_unc[fold_incorrect_idx]
            
            ax2.hist(fold_correct, bins=bins, alpha=0.15, color='green', 
                    density=True, edgecolor=None, linewidth=0)
            ax2.hist(fold_incorrect, bins=bins, alpha=0.15, color='red', 
                    density=True, edgecolor=None, linewidth=0)
    
    # Plot main histograms
    label_correct = f'Correct (n={len(unc_correct)}, avg)' if uncertainties_per_fold is not None else f'Correct (n={len(unc_correct)})'
    label_incorrect = f'Incorrect (n={len(unc_incorrect)}, avg)' if uncertainties_per_fold is not None else f'Incorrect (n={len(unc_incorrect)})'
    
    ax2.hist(unc_correct, bins=bins, alpha=0.6, color='green', 
            label=label_correct, density=True, edgecolor='black')
    ax2.hist(unc_incorrect, bins=bins, alpha=0.6, color='red', 
            label=label_incorrect, density=True, edgecolor='black')
    
    # Add vertical lines for means
    ax2.axvline(unc_correct.mean(), color='darkgreen', linestyle='--', 
               linewidth=2, label=f'Correct mean: {unc_correct.mean():.3f}')
    ax2.axvline(unc_incorrect.mean(), color='darkred', linestyle='--', 
               linewidth=2, label=f'Incorrect mean: {unc_incorrect.mean():.3f}')
    
    ax2.set_xlabel('Uncertainty Score', fontsize=13)
    ax2.set_ylabel('Density', fontsize=13)
    ax2.set_title(f'{method_name}\nUncertainty Distribution Overlap', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved uncertainty distributions to {save_path}")
    
    return fig


def save_all_evaluation_plots(uncertainties, predictions, labels, 
                               method_name='UQ Method', output_dir='./figures',
                               uncertainties_per_fold=None, ensemble_uncertainties=None,
                               predictions_per_fold=None, model_backbone=None, setup=None,
                               corruption_info=None, correct_idx=None, incorrect_idx=None,
                               per_fold_correct_idx=None, per_fold_incorrect_idx=None):
    """
    Generate and save all evaluation plots for a UQ method.
    
    Creates:
    1. ROC curve for failure prediction (AUROC_f)
    2. Risk-coverage curves (selective + generalized risk)
    3. Uncertainty distribution plots (boxplot + histogram)
    
    Args:
        uncertainties: Array of uncertainty scores [N]
        predictions: Array of predicted labels [N]
        labels: Array of true labels [N]
        method_name: Name of the UQ method
        output_dir: Directory to save figures
        uncertainties_per_fold: Optional [num_folds, N] array for per-fold visualization
        ensemble_uncertainties: Optional [N] array for ensemble reference
        predictions_per_fold: Optional [num_folds, N] array of per-fold predictions
        correct_idx: Optional pre-computed indices of correct predictions (for new_class_shift)
        incorrect_idx: Optional pre-computed indices of incorrect predictions (for new_class_shift)
        per_fold_correct_idx: Optional list of per-fold correct indices
        per_fold_incorrect_idx: Optional list of per-fold incorrect indices
    
    Returns:
        dict: Paths to saved figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename prefix with model configuration
    prefix = ''
    if model_backbone:
        prefix += f'{model_backbone}_'
    if setup:
        prefix += f'{setup}_'
    if corruption_info:
        prefix += f'corrupt_{corruption_info}_'
    
    # Sanitize method name for filename
    safe_name = method_name.replace(' ', '_').replace('/', '_')
    
    # Save ROC curve for failure prediction
    roc_path = output_dir / f'{prefix}{safe_name}_roc_curve.png'
    fig_roc = plot_roc_curve_failure_prediction(uncertainties, predictions, labels,
                                                method_name=method_name,
                                                save_path=roc_path,
                                                uncertainties_per_fold=uncertainties_per_fold,
                                                ensemble_uncertainties=ensemble_uncertainties,
                                                predictions_per_fold=predictions_per_fold,
                                                correct_idx=correct_idx,
                                                incorrect_idx=incorrect_idx,
                                                per_fold_correct_idx=per_fold_correct_idx,
                                                per_fold_incorrect_idx=per_fold_incorrect_idx)
    plt.close(fig_roc)
    
    # Save risk-coverage curves
    rc_path = output_dir / f'{prefix}{safe_name}_risk_coverage.png'
    fig_rc, _ = plot_risk_coverage_curve(uncertainties, predictions, labels, 
                                         save_path=rc_path,
                                         uncertainties_per_fold=uncertainties_per_fold,
                                         ensemble_uncertainties=ensemble_uncertainties,
                                         predictions_per_fold=predictions_per_fold,
                                         correct_idx=correct_idx,
                                         incorrect_idx=incorrect_idx,
                                         per_fold_correct_idx=per_fold_correct_idx,
                                         per_fold_incorrect_idx=per_fold_incorrect_idx)
    plt.close(fig_rc)
    
    # Save uncertainty distributions
    dist_path = output_dir / f'{prefix}{safe_name}_distributions.png'
    fig_dist = plot_uncertainty_distributions(uncertainties, predictions, labels,
                                              method_name=method_name,
                                              save_path=dist_path,
                                              uncertainties_per_fold=uncertainties_per_fold,
                                              ensemble_uncertainties=ensemble_uncertainties,
                                              predictions_per_fold=predictions_per_fold,
                                              correct_idx=correct_idx,
                                              incorrect_idx=incorrect_idx,
                                              per_fold_correct_idx=per_fold_correct_idx,
                                              per_fold_incorrect_idx=per_fold_incorrect_idx)
    plt.close(fig_dist)
    
    return {
        'roc_curve': str(roc_path),
        'risk_coverage': str(rc_path),
        'distributions': str(dist_path)
    }
