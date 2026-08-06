import os
import numpy as np
import pickle
from scipy.linalg import sqrtm
from pathlib import Path
from typing import List

ALL_TRAIN_FLAGS: List[str] = [
    "organamnist",
    "pneumoniamnist",
    "octmnist",
    "pathmnist",
    "bloodmnist",
    "tissuemnist",
    "breastmnist",
    "dermamnist-e-id",   # train split provided via dermamnist-e
]

ALL_TEST_FLAGS_POPULATION_SHIFT: List[str] = [
    "amos22",           # AMOS-2022 abdominal CT → OrganaMNIST label space (6 organs)
    "hmu-crc",          # HMU-CRC-Hist550K histology slides → PathMNIST label space
    "dermamnist-e-ext", # dermamnist-e external test center (test-only split)
]
# New-class shift: test set contains OOD classes absent from training.
# Saved NPZ includes a ``binary_gt`` array (0=known, 1=OOD).
ALL_TEST_FLAGS_NEW_CLASS: List[str] = [
    "amos22_new_classes",  # AMOS-2022 unmapped organs (OOD) + mapped organs
    "midog",               # MIDOG++ canine tumours (OOD) + PathMNIST test samples
]

def load_embeddings(embedding_dir):
    """Load all embeddings from directory."""
    embeddings = {}
    for filename in os.listdir(embedding_dir):
        if filename.endswith('.npz'):
            dataset_name = filename.replace('.npz', '')
            filepath = os.path.join(embedding_dir, filename)
            try:
                npz_file = np.load(filepath)
                # Extract 'embeddings' array from NPZ file (saved from dinov3_projection.py)
                embeddings[dataset_name] = npz_file['embeddings']
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
    print(f"Loaded {len(embeddings)} embeddings")
    return embeddings


def compute_fid(embeddings_real, embeddings_generated):
    """
    Compute Frechet Inception Distance (FID) between two sets of embeddings.
    
    Args:
        embeddings_real: (N, D) array of real embeddings
        embeddings_generated: (M, D) array of generated/test embeddings
    
    Returns:
        FID distance value
    """
    mu1 = np.mean(embeddings_real, axis=0)
    mu2 = np.mean(embeddings_generated, axis=0)
    sigma1 = np.cov(embeddings_real.T)
    sigma2 = np.cov(embeddings_generated.T)
    
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def compute_kid(emb_real, emb_test, num_subsets=100, subset_size=1000, degree=3, seed=42):
    """
    Kernel Inception Distance using polynomial kernel MMD (unbiased estimator).
    Returns mean KID over num_subsets bootstrapped subsets.
    k(x, y) = (x·y/D + 1)^degree
    """
    rng = np.random.default_rng(seed)
    n_real, n_test = len(emb_real), len(emb_test)
    n = min(n_real, n_test, subset_size)
    D = emb_real.shape[1]

    kid_vals = []
    for _ in range(num_subsets):
        x = emb_real[rng.choice(n_real, n, replace=True)].astype(np.float64)
        y = emb_test[rng.choice(n_test, n, replace=True)].astype(np.float64)

        kxx = (x @ x.T / D + 1.0) ** degree
        kyy = (y @ y.T / D + 1.0) ** degree
        kxy = (x @ y.T / D + 1.0) ** degree
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)

        mmd2 = (kxx.sum() / (n * (n - 1))
                + kyy.sum() / (n * (n - 1))
                - 2.0 * kxy.mean())
        kid_vals.append(mmd2)

    return float(np.mean(kid_vals))


def compute_mahalanobis(emb_train, emb_test, reg=1e-5):
    """
    Mahalanobis distance between test distribution mean and train distribution.
    D_M = sqrt((mu_test - mu_train)^T * Sigma_train^{-1} * (mu_test - mu_train))
    Uses adaptive regularization and least-squares solve for numerical stability.
    """
    mu_train = np.mean(emb_train, axis=0, dtype=np.float64)
    mu_test  = np.mean(emb_test,  axis=0, dtype=np.float64)
    sigma = np.cov(emb_train.astype(np.float64).T)
    # Adaptive regularization scaled to eigenvalue magnitude
    sigma += np.eye(sigma.shape[0]) * reg * np.trace(sigma) / sigma.shape[0]
    diff = mu_test - mu_train
    x, _, _, _ = np.linalg.lstsq(sigma, diff, rcond=None)
    return float(np.sqrt(max(0.0, float(np.dot(diff, x)))))


def _dist_entry(train_emb, test_emb, id_entry=None):
    """
    Compute FID, KID, Mahalanobis between train and test embeddings.
    If id_entry is provided, also compute normalized values relative to it.
    """
    fid  = compute_fid(train_emb, test_emb)
    kid  = compute_kid(train_emb, test_emb)
    maha = compute_mahalanobis(train_emb, test_emb)

    if id_entry is None:  # This IS the ID reference
        return {
            'fid_distance': fid,  'normalized_fid': 1.0,
            'kid': kid,           'normalized_kid': 1.0,
            'mahalanobis': maha,  'normalized_mahalanobis': 1.0,
        }

    return {
        'fid_distance': fid,
        'normalized_fid': fid / id_entry['fid_distance'],
        'kid': kid,
        'normalized_kid': kid / max(abs(id_entry['kid']), 1e-8),
        'mahalanobis': maha,
        'normalized_mahalanobis': maha / max(id_entry['mahalanobis'], 1e-8),
    }


def main():
    embedding_dir = '/mnt/data/psteinmetz/computer_vision_code/code/FailCatcher/Benchmarks/medMNIST/results/dinov3_embeddings'
    embeddings = load_embeddings(embedding_dir)
    results = {}

    for dataset in ALL_TRAIN_FLAGS:
        results[dataset] = {}
        train   = embeddings[dataset + '_train_']
        id_test = embeddings[dataset + '_test_']
        cs_test = embeddings[dataset + '_test__random_s3']

        print(f"\n── {dataset} ──")

        print("  ID test …", flush=True)
        id_entry = _dist_entry(train, id_test)
        results[dataset]['id'] = id_entry
        print(f"    Done: FID={id_entry['fid_distance']:.4f}")

        print("  Corruption shift (random_s3) …", flush=True)
        results[dataset]['random_s3'] = _dist_entry(train, cs_test, id_entry)
        print(f"    Done: FID={results[dataset]['random_s3']['fid_distance']:.4f}")

        if 'organamnist' in dataset:
            print("  Population shift (AMOS22) …", flush=True)
            results[dataset]['population_shift'] = _dist_entry(
                train, embeddings['amos22_population_shift_'], id_entry)
            print(f"    Done: FID={results[dataset]['population_shift']['fid_distance']:.4f}")
            print("  New-class shift (AMOS22 unmapped) …", flush=True)
            results[dataset]['new_classes'] = _dist_entry(
                train, embeddings['amos22_new_classes_'], id_entry)
            print(f"    Done: FID={results[dataset]['new_classes']['fid_distance']:.4f}")

        elif 'pathmnist' in dataset:
            print("  Population shift (HMU-CRC) …", flush=True)
            results[dataset]['population_shift'] = _dist_entry(
                train, embeddings['hmu-crc_population_shift'], id_entry)
            print(f"    Done: FID={results[dataset]['population_shift']['fid_distance']:.4f}")
            print("  New-class shift (MIDOG) …", flush=True)
            results[dataset]['new_classes'] = _dist_entry(
                train, embeddings['midog_new_classes_'], id_entry)
            print(f"    Done: FID={results[dataset]['new_classes']['fid_distance']:.4f}")

        elif 'dermamnist-e' in dataset:
            print("  Population shift (dermamnist-e-ext) …", flush=True)
            results[dataset]['population_shift'] = _dist_entry(
                train, embeddings['dermamnist-e-ext_population_shift_'], id_entry)
            print(f"    Done: FID={results[dataset]['population_shift']['fid_distance']:.4f}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset, shifts in results.items():
        print(f"\n{dataset}:")
        for shift, m in shifts.items():
            print(f"  {shift}:")
            print(f"    FID={m['fid_distance']:.4f}  (norm {m['normalized_fid']:.4f})")
            print(f"    KID={m['kid']:.6f}  (norm {m['normalized_kid']:.4f})")
            print(f"    Mahalanobis={m['mahalanobis']:.4f}  (norm {m['normalized_mahalanobis']:.4f})")

    results_path = '/mnt/data/psteinmetz/computer_vision_code/code/FailCatcher/Benchmarks/medMNIST/results/dinov3_embeddings/fid_results.pkl'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == '__main__':
    results = main()
