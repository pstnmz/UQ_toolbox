"""
Self-check for the KNN_SHAP normalization fix (methods/latent.py).

The fix does NOT change per-class AUROC: z = (x - mean) / std is a positive-scale
affine transform, which preserves rank order, so per-class AUROC is identical
regardless of which mean/std is used. What it fixes is CROSS-CLASS comparability
of the pooled/global AUROC: classes with a higher local misclassification rate
have their own contaminated std inflated, which shrinks their outliers' z-scores
relative to a low-error-rate class with the same intrinsic correct/incorrect gap.
Normalizing with a train-derived (uncontaminated) reference removes that bias.

Ceiling / upgrade path: synthetic 2-class demo, not an integration test of the
full fit()/compute() pipeline (needs real models + data loaders).
Run directly: python ToolBox/tests/test_shap_knn_masking_fix.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def test_pooled_auroc_improves_with_ref_stats():
    rng = np.random.default_rng(0)

    def make_class(n_correct, n_incorrect, correct_loc, correct_scale, incorrect_loc, incorrect_scale):
        correct = rng.normal(correct_loc, correct_scale, n_correct)
        incorrect = rng.normal(incorrect_loc, incorrect_scale, n_incorrect)
        return correct, incorrect

    # Same intrinsic correct/incorrect gap, but different local error rates.
    A_correct, A_incorrect = make_class(950, 50, 1.0, 0.2, 3.0, 0.5)   # 5% error rate
    B_correct, B_incorrect = make_class(700, 300, 1.0, 0.2, 3.0, 0.5)  # 30% error rate

    ref_mean, ref_std = 1.0, 0.2  # train-derived reference, same for both classes

    def zscore(correct, incorrect, mean, std):
        return (correct - mean) / std, (incorrect - mean) / std

    # OLD: normalize using each class's own contaminated test-time mixture.
    mixA = np.concatenate([A_correct, A_incorrect])
    mixB = np.concatenate([B_correct, B_incorrect])
    oldA_c, oldA_i = zscore(A_correct, A_incorrect, mixA.mean(), mixA.std())
    oldB_c, oldB_i = zscore(B_correct, B_incorrect, mixB.mean(), mixB.std())

    # NEW: normalize using the clean train-derived reference.
    newA_c, newA_i = zscore(A_correct, A_incorrect, ref_mean, ref_std)
    newB_c, newB_i = zscore(B_correct, B_incorrect, ref_mean, ref_std)

    labels = np.concatenate([np.zeros(len(A_correct)), np.ones(len(A_incorrect)),
                             np.zeros(len(B_correct)), np.ones(len(B_incorrect))])
    old_scores = np.concatenate([oldA_c, oldA_i, oldB_c, oldB_i])
    new_scores = np.concatenate([newA_c, newA_i, newB_c, newB_i])

    old_auc = roc_auc_score(labels, old_scores)
    new_auc = roc_auc_score(labels, new_scores)

    # Per-class AUROC must be unaffected (rank-preservation sanity check).
    labels_A = np.concatenate([np.zeros(len(A_correct)), np.ones(len(A_incorrect))])
    auc_A_old = roc_auc_score(labels_A, np.concatenate([oldA_c, oldA_i]))
    auc_A_new = roc_auc_score(labels_A, np.concatenate([newA_c, newA_i]))
    assert abs(auc_A_old - auc_A_new) < 1e-9, "Per-class AUROC must be invariant to the normalization reference"

    assert new_auc >= old_auc, (
        f"Expected train-ref normalization to match or improve pooled AUROC across "
        f"classes with differing error rates, got new={new_auc:.4f} < old={old_auc:.4f}"
    )
    print(f"[OK] per-class AUROC unaffected (A: old={auc_A_old:.4f} new={auc_A_new:.4f}); "
          f"pooled AUROC old={old_auc:.4f} <= new={new_auc:.4f}")


if __name__ == "__main__":
    test_pooled_auroc_improves_with_ref_stats()

