"""
Latent space-based uncertainty quantification methods.
Includes SHAP importance, KNN distances, feature engineering, and hyperplane distance analysis.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as torch_mp

# Default 'file_descriptor' strategy opens an fd per shared-memory tensor transfer between
# processes (model weights, DataLoader batches); with several concurrent GPU worker processes
# each running their own DataLoader workers, this exhausts the OS open-file limit ("Too many
# open files"). 'file_system' avoids fd passing entirely (uses named /dev/shm segments).
torch_mp.set_sharing_strategy('file_system')

from ..core.base import UQMethod


# ============================================================================
# HELPER FUNCTION FOR LAYER EXTRACTION
# ============================================================================

def get_layer_from_model(model, layer_name='avgpool'):
    """
    Extract a layer from a model by name.
    
    Args:
        model: PyTorch model
        layer_name: Name or pattern of layer to extract
    
    Returns:
        torch.nn.Module: The layer module
    
    Supported patterns:
        - 'avgpool': Global average pooling
        - 'layer4': Last conv layer (ResNet)
        - 'fc': Fully connected layer
        - 'head': Classifier head (ViT)
        - Custom: Direct attribute access (e.g., 'features.denseblock4')
    
    Example:
        >>> layer = get_layer_from_model(model, 'avgpool')
        >>> layer = get_layer_from_model(model, 'layer4')
    """
    # Try direct attribute access first
    if hasattr(model, layer_name):
        return getattr(model, layer_name)
    
    # Pattern-based search
    if layer_name == 'avgpool':
        # Try common pooling layer names
        for attr in ['avgpool', 'global_pool', 'avg_pool']:
            if hasattr(model, attr):
                return getattr(model, attr)
        
        # For ViT models: use encoder as the feature extraction layer
        # ViT models have: conv_proj -> encoder -> heads
        if hasattr(model, 'encoder'):
            return model.encoder
        
        # Fallback: search for adaptive pooling in modules
        for name, module in model.named_modules():
            if 'pool' in name.lower() and 'adaptive' in str(type(module)).lower():
                return module
    
    elif layer_name == 'layer4':
        if hasattr(model, 'layer4'):
            return model.layer4
        # For DenseNet: last dense block
        if hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
            return model.features.denseblock4
    
    elif layer_name == 'fc':
        for attr in ['fc', 'classifier', 'head']:
            if hasattr(model, attr):
                return getattr(model, attr)
    
    # Nested attribute access (e.g., 'features.denseblock4')
    if '.' in layer_name:
        obj = model
        for attr in layer_name.split('.'):
            obj = getattr(obj, attr)
        return obj
    
    raise ValueError(
        f"Could not find layer '{layer_name}' in model. "
        f"Available top-level attributes: {[name for name, _ in model.named_children()]}"
    )
    
    
# ============================================================================
# PARALLEL WORKER FUNCTIONS (must be at module level for pickling)
# ============================================================================

# Populated once per worker process by the ProcessPoolExecutor `initializer` (see
# _fit_parallel/_compute_parallel) so the (potentially multi-GB) dataset objects are
# pickled once per worker instead of once per submitted fold task.
_shap_worker_datasets = {}


def _init_shap_pool_worker(train_dataset, calib_dataset, test_dataset=None):
    """ProcessPoolExecutor initializer: stashes shared dataset objects as process globals."""
    _shap_worker_datasets['train'] = train_dataset
    _shap_worker_datasets['calib'] = calib_dataset
    _shap_worker_datasets['test'] = test_dataset


def _fit_fold_worker_multigpu(fold_idx, model, train_loader_meta, calib_loader_meta, device_str, flag, gpu_id, cache_dir=None, shap_only=False, num_workers=0):
    """
    Worker function for parallel fold fitting with explicit GPU assignment.
    
    Args:
        fold_idx: Fold index
        model: PyTorch model (on CPU)
        train_loader_meta: Dict with keys: 'indices', 'batch_size', 'shuffle' (dataset itself
            comes from _shap_worker_datasets, set once via the pool initializer)
        calib_loader_meta: Dict with keys: 'indices', 'batch_size', 'shuffle'
        device_str: Device string (e.g., 'cuda:1')
        flag: Dataset name for caching
        gpu_id: GPU ID for this worker
        cache_dir: Directory for SHAP cache (None to disable)
        num_workers: DataLoader workers for THIS process's train/calib loaders (this worker's
            share of the total budget - not 0, otherwise image decode/transform serializes on
            one CPU thread per GPU process while the GPU sits idle)
    
    Returns:
        List of per-class KNN dicts
    """
    import torch
    import gc
    import os
    import random
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    # Ensure reproducible cache generation in spawned worker processes.
    worker_seed = 42 + int(fold_idx)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Bind this worker to the assigned physical GPU.
    # Note: setting CUDA_VISIBLE_DEVICES inside a spawned process is too late
    # once CUDA is initialized; direct set_device avoids accidental GPU 0 usage.
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    # Reconstruct DataLoaders from the shared dataset objects (set once by the pool
    # initializer) + this fold's indices. Each parallel GPU worker gets its own slice of
    # DataLoader workers (num_workers) so image decode/transform isn't serialized on one
    # thread while the GPU idles.
    train_subset = Subset(_shap_worker_datasets['train'], train_loader_meta['indices'])
    train_loader = DataLoader(
        train_subset,
        batch_size=train_loader_meta['batch_size'],
        shuffle=train_loader_meta.get('shuffle', False),
        num_workers=num_workers,
        pin_memory=True
        # Not persistent_workers - this loader is iterated at most once per worker call, so
        # keeping the pool alive afterward would only hold fds/shared memory longer than needed.
    )
    
    calib_subset = Subset(_shap_worker_datasets['calib'], calib_loader_meta['indices'])
    calib_loader = DataLoader(
        calib_subset,
        batch_size=calib_loader_meta['batch_size'],
        shuffle=calib_loader_meta.get('shuffle', False),
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f" Worker {fold_idx}: Using GPU {gpu_id}")
    
    # Move model to assigned GPU
    model = model.to(device)
    model.eval()
    
    # Background sample count for the analytic SHAP mean (cheap regardless of size;
    # kept modest to match the cache filename's `bg{N}` naming convention).
    max_bg_samples = 1000
    
    # Create a temporary method instance for this fold
    temp_method = KNNLatentSHAPMethod(
        layer_name='avgpool',
        k=5,
        n_shap_features=50,
        cache_dir=cache_dir,
        max_background_samples=max_bg_samples,
        shap_only=shap_only
    )
    
    # Fit this fold
    try:
        model_knns = temp_method._fit_single_fold(
            fold_idx, model, train_loader, calib_loader, device, flag
        )
    finally:
        # Always free GPU memory, even on failure (e.g. OOM), so a failed fold doesn't
        # leave memory pinned for the next fold that reuses this worker process.
        model = model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f" Worker {fold_idx}: Done on GPU {gpu_id}")
    
    return model_knns


def _compute_fold_worker_multigpu(fold_idx, model, model_knns, test_loader_meta, gpu_id, layer_name, num_classes, num_workers=0):
    """
    Worker function for parallel per-fold KNN-SHAP distance computation (inference/compute step).

    Args:
        fold_idx: Fold index
        model: PyTorch model (on CPU)
        model_knns: Fitted per-class KNN dicts for this fold (from _fit_single_fold)
        test_loader_meta: Dict with keys: 'indices', 'batch_size' (dataset comes from
            _shap_worker_datasets, set once via the pool initializer)
        gpu_id: GPU ID for this worker
        layer_name: Layer name for feature extraction
        num_classes: Number of classes
        num_workers: DataLoader workers for THIS process's test loader (this worker's share
            of the total budget)

    Returns:
        (fold_idx, distances_per_sample) tuple
    """
    import gc
    from torch.utils.data import DataLoader, Subset

    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    test_subset = Subset(_shap_worker_datasets['test'], test_loader_meta['indices'])
    test_loader = DataLoader(
        test_subset,
        batch_size=test_loader_meta['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
        # Not persistent_workers - iterated once per worker call, see _fit_fold_worker_multigpu.
    )

    model = model.to(device)
    model.eval()
    layer = get_layer_from_model(model, layer_name)

    distances_per_sample = KNNLatentSHAPMethod._distances_from_fitted(
        model, model_knns, test_loader, device, layer, num_classes, fold_idx
    )

    model = model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f" Worker {fold_idx}: compute done on GPU {gpu_id}")

    return fold_idx, distances_per_sample


# ============================================================================
# CLASS-BASED API
# ============================================================================

class ClassifierHeadWrapper(nn.Module):
    """
    Exposes a model's classifier head as a standalone affine function of its
    penultimate-layer features (avgpool output / CLS token -> logits).

    Requires the head to be a single nn.Linear: dropout-wrapped heads (nn.Sequential
    of Dropout+Linear, as used by the MC-Dropout model variants) are not supported,
    since analytic SHAP needs one fixed weight matrix.
    """
    def __init__(self, model, layer_name='avgpool'):
        """
        Args:
            model: Full PyTorch model
            layer_name: Name of layer to hook (e.g., 'avgpool', 'layer4')
        """
        super().__init__()
        self.model = model
        self.layer = get_layer_from_model(model, layer_name)

    def _get_head(self):
        """Locate the model's classifier head module."""
        # torchvision ViT's `encoder` already applies its final LayerNorm internally, so
        # the hooked feature (CLS token) is already post-LN - call heads/head directly.
        for attr in ('fc', 'heads', 'head', 'classifier'):
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
        for _, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Linear):
                return module
        raise RuntimeError(
            "Could not identify classifier head. "
            "Please manually specify the classifier in ClassifierHeadWrapper."
        )

    def get_linear(self):
        """Return the head as an nn.Linear (weight/bias) for analytic SHAP."""
        head = self._get_head()
        
        # If head is Sequential (e.g., ViT's [Dropout, Linear]), extract the Linear layer
        if isinstance(head, nn.Sequential):
            linear_layer = None
            for module in head:
                if isinstance(module, nn.Linear):
                    linear_layer = module
            if linear_layer is None:
                raise TypeError(
                    f"Sequential head contains no nn.Linear layer: {head}"
                )
            print(f" Extracted nn.Linear from Sequential head: {linear_layer}")
            return linear_layer
        
        # Direct nn.Linear (standard ResNet case)
        if isinstance(head, nn.Linear):
            return head
        
        # Unsupported head type
        raise TypeError(
            f"Analytic SHAP requires an nn.Linear classifier head, got {type(head).__name__}. "
            "ViT Sequential heads are supported (Dropout+Linear extracted). "
            "Other wrapped/composite heads are not."
        )

    def forward(self, x):
        """Forward pass from latent features (hooked at self.layer) to logits."""
        return self._get_head()(x)


class KNNLatentMethod(UQMethod):
    """
    Uncertainty quantification via KNN distance in RAW latent space.
    Supports CV ensembles with per-fold training data.
    """
    def __init__(self, layer_name='avgpool', k=5, pca_variance=0.9):
        super().__init__("KNN-Latent-Raw")
        self.layer_name = layer_name
        self.k = k
        self.pca_variance = pca_variance
        self.fitted_models = []
    
    def fit(self, models, train_loaders, device):
        """
        Fit KNN on training data latent space for each model.
        
        Args:
            models: List of models (one per fold)
            train_loaders: List of train DataLoaders (one per fold) OR single loader
            device: torch.device
        """
        if not isinstance(models, list):
            models = [models]
        
        # Handle both single loader and list of loaders
        if not isinstance(train_loaders, list):
            train_loaders = [train_loaders] * len(models)
        
        if len(models) != len(train_loaders):
            raise ValueError(
                f"Number of models ({len(models)}) must match "
                f"number of train loaders ({len(train_loaders)})"
            )
        
        self.fitted_models = []
        
        for idx, (model, train_loader) in enumerate(zip(models, train_loaders)):
            print(f" Fold {idx}: Fitting KNN on its training set...")
            
            # Extract layer from this model
            layer = get_layer_from_model(model, self.layer_name)
            
            # Extract features from THIS fold's training data
            features, labels, _, _ = extract_latent_space_and_compute_shap_importance(
                model, train_loader, device, layer, importance=False
            )
            
            # Standardize and apply PCA
            scaler = StandardScaler()
            features_std = scaler.fit_transform(features.numpy())
            
            pca = PCA(n_components=self.pca_variance)
            features_pca = pca.fit_transform(features_std)
            
            # Fit KNN
            knn = NearestNeighbors(n_neighbors=self.k)
            knn.fit(features_pca)
            
            self.fitted_models.append({
                'knn': knn,
                'scaler': scaler,
                'pca': pca,
                'layer': layer,
                # Cached so callers (e.g. k grid search) can refit KNN without re-extracting features.
                'train_features_pca': features_pca
            })
            
            print(f" {len(features)} train samples, {features.shape[1]} → {features_pca.shape[1]} PCA dims")
            
            # Clear GPU cache between folds to prevent memory buildup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f" Fitted {len(models)} fold(s)")
        return self
    
    def compute(self, models, data_loader, device, return_per_fold=True):
        """
        Compute KNN distances per fold (or averaged for backward compatibility).
        
        Args:
            models: List of models (one per fold)
            data_loader: Test data loader
            device: torch device
            return_per_fold: If True, return [num_folds, N]. If False, return [N] (averaged, legacy)
        
        Returns:
            np.ndarray: 
                - If return_per_fold=True: [num_folds, N] distances per fold
                - If return_per_fold=False: [N] averaged distances (backward compatible)
        """
        if not self.fitted_models:
            raise RuntimeError("Call fit() before compute()")
        
        if not isinstance(models, list):
            models = [models]
        
        if len(models) != len(self.fitted_models):
            raise ValueError(f"Expected {len(self.fitted_models)} models, got {len(models)}")
        
        all_distances = []
        
        for idx, (model, fitted) in enumerate(zip(models, self.fitted_models)):
            # Extract test features using the saved layer
            features, _, _, _ = extract_latent_space_and_compute_shap_importance(
                model, data_loader, device, fitted['layer'], importance=False
            )
            
            # Transform test features
            features_std = fitted['scaler'].transform(features.numpy())
            features_pca = fitted['pca'].transform(features_std)
            
            # Compute distances
            distances, _ = fitted['knn'].kneighbors(features_pca)
            avg_distances = distances.mean(axis=1)
            
            # Z-score normalization per fold (critical for fair ensembling)
            # Each fold gets its own mean/std, preventing one fold from dominating
            mean_dist = np.mean(avg_distances)
            std_dist = np.std(avg_distances)
            
            if std_dist < 1e-10:
                # Edge case: constant distances (shouldn't happen in practice)
                zscore_distances = np.zeros_like(avg_distances)
            else:
                zscore_distances = (avg_distances - mean_dist) / std_dist
            
            all_distances.append(zscore_distances)
        
        all_distances = np.array(all_distances)  # [num_folds, N] - now z-scored per fold
        
        if return_per_fold:
            print(f" Computed KNN distances for {len(models)} folds (z-scored per fold)")
            return all_distances  # [num_folds, N]
        else:
            # Legacy mode: average across folds
            final_distances = np.mean(all_distances, axis=0)
            print(f" Computed KNN distances (z-scored per fold, then averaged over {len(models)} folds)")
            return final_distances


class KNNLatentSHAPMethod(UQMethod):
    """
    Uncertainty quantification via KNN distance in SHAP-SELECTED latent space.
    
    **Critical Process:**
    For EACH model with its CV training split:
      1. Compute SHAP on TRAINING samples (background = held-out calibration data)
         → select top-k features per class
      2. Fit per-class standardization + PCA + KNN, all on TRAINING data
      3. For each test sample:
         - Get predicted class
         - Transform latent features into that class's SHAP-selected subspace
         - Compute (unnormalized) KNN distance to training samples of that class
      4. Average distances across all models
    """
    def __init__(self, layer_name='avgpool', k=5, n_shap_features=50, 
                 max_background_samples=1000, cache_dir=None, parallel=False, n_jobs=None,
                 shap_only=False, shap_batch_size=5000):
        super().__init__("KNN-Latent-SHAP")
        self.layer_name = layer_name
        self.k = k
        self.n_shap_features = n_shap_features
        self.max_background = max_background_samples
        self.cache_dir = cache_dir
        self.parallel = parallel 
        self.shap_batch_size = shap_batch_size  # Process SHAP in chunks to avoid OOM
        self.fitted_models = []
        self.num_classes = None
        # shap_only: cache calibration features and stop, skipping the train-feature
        # extraction + KNN fitting steps (used to pre-warm feature caches cheaply).
        self.shap_only = shap_only
        
        # Auto-detect number of GPUs if parallel and n_jobs not specified
        if parallel and n_jobs is None:
            import torch
            self.n_jobs = torch.cuda.device_count()
            print(f" Auto-detected {self.n_jobs} GPUs for parallel processing")
        else:
            self.n_jobs = n_jobs or 2
    
    def _get_cache_path(self, flag, fold_idx):
        """Generate cache file path for a specific fold."""
        if self.cache_dir is None:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(
            self.cache_dir, 
            f'shap_cache_{flag}_fold{fold_idx}_bg{self.max_background}.npz'
        )
    
    def _extract_loader_params(self, data_loader):
        """Extract parameters from a DataLoader for reconstruction in workers."""
        from torch.utils.data import Subset
        
        # Get dataset and indices
        if isinstance(data_loader.dataset, Subset):
            dataset = data_loader.dataset.dataset
            indices = list(data_loader.dataset.indices)
        else:
            dataset = data_loader.dataset
            indices = list(range(len(dataset)))
        
        return {
            'dataset': dataset,
            'indices': indices,
            'batch_size': data_loader.batch_size,
            'shuffle': hasattr(data_loader, 'shuffle') and data_loader.shuffle,
            'num_workers': data_loader.num_workers
        }
    
    def fit(self, models, train_loaders, calib_loader, device, flag='unknown'):
        """
        Fit KNN PER MODEL using SHAP caching (sequential or parallel).
        """
        if not isinstance(models, list):
            models = [models]
        
        if not isinstance(train_loaders, list):
            train_loaders = [train_loaders] * len(models)
        
        if len(models) != len(train_loaders):
            raise ValueError(f"Models ({len(models)}) != train_loaders ({len(train_loaders)})")
        
        if self.parallel and len(models) > 1:
            print(f"\n   Parallel mode: {self.n_jobs} workers across {len(models)} folds")
            self.fitted_models = self._fit_parallel(models, train_loaders, calib_loader, device, flag)
            
            # Infer num_classes from the fitted models (workers don't share self.num_classes)
            if self.num_classes is None:
                # Count non-None entries in first model's KNN list
                for model_knns in self.fitted_models:
                    if model_knns is not None:
                        self.num_classes = len(model_knns)
                        print(f" Inferred {self.num_classes} classes from fitted models")
                        break
            
            # Move models back to original device after parallel processing
            print(f" Moving models back to {device}...")
            for model in models:
                model.to(device)
        else:
            print(f"\n  Sequential mode: {len(models)} folds")
            self.fitted_models = self._fit_sequential(models, train_loaders, calib_loader, device, flag)
        
        print(f"\n  Fitted {len(models)} models (each with its own SHAP + KNN)")
        return self
    
    def _fit_sequential(self, models, train_loaders, calib_loader, device, flag):
        """Sequential fitting (existing code)."""
        fitted_models = []
        
        for fold_idx, (model, train_loader) in enumerate(zip(models, train_loaders)):
            print(f"\n  Model {fold_idx+1}/{len(models)}: Computing SHAP + fitting KNN...")
            
            try:
                model_knns = self._fit_single_fold(
                    fold_idx, model, train_loader, calib_loader, device, flag
                )
            except Exception as e:
                print(f" Fold {fold_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                # See _fit_parallel: don't let one fold's OOM abort a shap_only cache-warming run.
                if self.shap_only:
                    print(f" shap_only=True: skipping fold {fold_idx}, continuing with remaining folds")
                    model_knns = None
                else:
                    raise
            fitted_models.append(model_knns)
        
        return fitted_models
    
    def _fit_parallel(self, models, train_loaders, calib_loader, device, flag):
        """Parallel fitting across folds with multi-GPU support."""
        import torch
        
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPUs available for parallel processing")
        
        print(f" Detected {n_gpus} GPUs, distributing {len(models)} folds...")
        
        try:
            torch_mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # Extract calibration loader parameters once (shared across all folds)
        calib_loader_params = self._extract_loader_params(calib_loader)
        
        # Each concurrently-running GPU worker gets its own share of the original loader's
        # DataLoader workers - hardcoding 0 here serializes image decode/transform on one CPU
        # thread per GPU process while the GPU idles waiting for batches.
        max_concurrent = min(self.n_jobs, n_gpus)
        per_worker_num_workers = max(1, (calib_loader_params.get('num_workers') or 0) // max_concurrent)
        print(f" Per-worker DataLoader workers: {per_worker_num_workers} (x{max_concurrent} concurrent workers)")
        
        # All folds share the same underlying train/calib dataset objects (only the per-fold
        # indices differ). Ship those (potentially multi-GB) objects to each worker process
        # ONCE via the pool initializer, instead of re-pickling them on every one of the
        # len(models) submitted fold tasks.
        train_dataset = self._extract_loader_params(train_loaders[0])['dataset']
        calib_dataset = calib_loader_params['dataset']
        
        fold_args = []
        for fold_idx, (model, train_loader) in enumerate(zip(models, train_loaders)):
            gpu_id = fold_idx % n_gpus
            device_str = f'cuda:{gpu_id}'
            model_cpu = model.cpu()
            
            # Extract train loader parameters for this fold (indices/batch_size/shuffle only -
            # the dataset itself is shipped once via the pool initializer, not per fold)
            train_loader_meta = {k: v for k, v in self._extract_loader_params(train_loader).items() if k != 'dataset'}
            calib_loader_meta = {k: v for k, v in calib_loader_params.items() if k != 'dataset'}
            
            fold_args.append((fold_idx, model_cpu, train_loader_meta, calib_loader_meta, device_str, flag, gpu_id, self.cache_dir, self.shap_only, per_worker_num_workers))
            print(f" Fold {fold_idx} → GPU {gpu_id}")
        
        fitted_models = [None] * len(models)
        
        with ProcessPoolExecutor(
            max_workers=max_concurrent,
            initializer=_init_shap_pool_worker,
            initargs=(train_dataset, calib_dataset)
        ) as executor:
            future_to_fold = {
                executor.submit(_fit_fold_worker_multigpu, *args): args[0]
                for args in fold_args
            }
            
            for future in as_completed(future_to_fold):
                fold_idx = future_to_fold[future]
                try:
                    model_knns = future.result()
                    fitted_models[fold_idx] = model_knns
                    print(f" Fold {fold_idx} complete")
                except Exception as e:
                    print(f" Fold {fold_idx} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # shap_only is a cache-warming pass: one fold OOM-ing (e.g. a large dataset
                    # + ViT) shouldn't take down the other folds/datasets in the same launcher
                    # run. Leave fitted_models[fold_idx]=None and continue; rerun later to retry
                    # just that fold (cache already covers the folds that succeeded).
                    if self.shap_only:
                        print(f" shap_only=True: skipping fold {fold_idx}, continuing with remaining folds")
                        continue
                    raise
        
        return fitted_models
    
    def _fit_single_fold(self, fold_idx, model, train_loader, calib_loader, device, flag):
        """
        Fit a single fold (called by both sequential and parallel modes).
        Returns: List of per-class KNN dicts
        """
        import gc
    
        layer = get_layer_from_model(model, self.layer_name)
        classifierhead = ClassifierHeadWrapper(model, self.layer_name)
        
        # =====================================================================
        # Load cached features. SHAP importance is analytic (cheap), so it and the
        # feature selection are always recomputed fresh below - only the forward-pass
        # feature extraction itself is worth caching.
        # =====================================================================
        cache_path = self._get_cache_path(flag, fold_idx)
        features_calib, labels_calib, features_train, labels_train = None, None, None, None
        cache_dirty = False
        
        if cache_path and os.path.exists(cache_path):
            print(f" Loading cached features from {os.path.basename(cache_path)}")
            try:
                cache = np.load(cache_path, allow_pickle=True)
                features_calib_raw = cache['features']
                labels_calib_raw = cache['labels']
                
                # Validate cache integrity
                expected_calib_size = len(calib_loader.dataset)
                if features_calib_raw.shape[0] != expected_calib_size:
                    print(f" [WARNING] Cache size mismatch: {features_calib_raw.shape[0]} != {expected_calib_size}, recomputing...")
                    features_calib = None
                elif np.isnan(features_calib_raw).any() or np.isinf(features_calib_raw).any():
                    print(f" [WARNING] Cache contains NaNs/infs, recomputing...")
                    features_calib = None
                else:
                    features_calib = torch.from_numpy(features_calib_raw)
                    labels_calib = labels_calib_raw
                    print(f" Cache valid: {features_calib.shape}")
                    
                    if 'features_train' in cache and 'labels_train' in cache:
                        features_train_raw = cache['features_train']
                        labels_train_raw = cache['labels_train']
                        
                        expected_train_size = len(train_loader.dataset)
                        if features_train_raw.shape[0] != expected_train_size:
                            print(f" [WARNING] Train cache size mismatch: {features_train_raw.shape[0]} != {expected_train_size}, recomputing train features...")
                            features_train = None
                        elif np.isnan(features_train_raw).any() or np.isinf(features_train_raw).any():
                            print(f" [WARNING] Train cache contains NaNs/infs, recomputing...")
                            features_train = None
                        else:
                            features_train = torch.from_numpy(features_train_raw)
                            labels_train = labels_train_raw
                            print(f" Reusing cached training features: {features_train.shape}")
            except Exception as e:
                print(f" [WARNING] Cache load failed: {e}, recomputing...")
        
        # =====================================================================
        # Step 1: Calibration features (used only as the SHAP background distribution -
        # held out from training, so it doesn't self-reference the samples being explained)
        # =====================================================================
        if features_calib is None:
            print(f" Step 1: Extracting calibration features (SHAP background)...")
            features_calib, labels_calib, _, _ = extract_latent_space_and_compute_shap_importance(
                model, calib_loader, device, layer, importance=False
            )
            cache_dirty = True
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        # =====================================================================
        # Step 2: Training features
        # =====================================================================
        if features_train is None or labels_train is None:
            print(f" Step 2: Extracting training features...")
            features_train, labels_train, _, _ = extract_latent_space_and_compute_shap_importance(
                model, train_loader, device, layer, importance=False
            )
            cache_dirty = True
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        else:
            print(f" Step 2: Reusing cached training features")

        # =====================================================================
        # Step 3: Analytic SHAP importance on TRAINING samples, using CALIBRATION
        # data as the background distribution (disjoint from the explained samples,
        # so no self-reference leakage) + top-k feature selection per class
        # =====================================================================
        print(f" Step 3: Computing analytic SHAP (train samples, calib background) + selecting top features per class...")
        shap_values = compute_analytic_shap_values(
            features_train, classifierhead, self.max_background,
            batch_size=self.shap_batch_size, background=features_calib
        )
        del classifierhead

        # Binary output is [N, F, 1], multi-class is [N, F, C]
        if shap_values.ndim == 3 and len(np.unique(labels_train)) == 2:
            shap_values = shap_values.squeeze(-1)

        if shap_values.ndim == 3:
            _, num_features, num_classes = shap_values.shape
        elif shap_values.ndim == 2:
            _, num_features = shap_values.shape
            num_classes = len(np.unique(labels_train))
        else:
            raise ValueError("Expected 2D or 3D SHAP values array")

        selected_features_per_class = []
        for class_idx in range(num_classes):
            if shap_values.ndim == 3:
                class_shap_values = shap_values[:, :, class_idx]
            else:
                class_shap_values = shap_values[labels_train == class_idx, :]

            mean_abs = np.mean(np.abs(class_shap_values), axis=0)
            top_k = min(self.n_shap_features, len(mean_abs))
            top_idx = np.argpartition(mean_abs, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(mean_abs[top_idx])[::-1]]
            selected_features_per_class.append([f"Feature_{i}" for i in top_idx.tolist()])

        if self.num_classes is None:
            self.num_classes = num_classes
            print(f" Detected {self.num_classes} classes")

        if self.shap_only:
            if cache_path and cache_dirty:
                print(" Updating feature cache")
                np.savez(
                    cache_path,
                    features=features_calib.numpy(),
                    labels=labels_calib,
                    features_train=features_train.numpy(),
                    labels_train=labels_train
                )
            print(f" shap_only=True: features cached, skipping KNN fit")
            return None

        train_df = pd.DataFrame(
            features_train.numpy(),
            columns=[f"Feature_{i}" for i in range(features_train.shape[1])]
        )

        # =====================================================================
        # Step 4: Fit per-class standardization + PCA + KNN, all on TRAIN data
        # =====================================================================
        print(f" Step 4: Fitting KNN per class...")
        model_knns = []
        
        for class_idx in range(self.num_classes):
            class_mask = (labels_train == class_idx)
            train_class_df = train_df[class_mask]
            
            if len(train_class_df) == 0:
                model_knns.append(None)
                continue
            
            selected_features = selected_features_per_class[class_idx]
            train_selected = train_class_df[selected_features].values
            
            scaler = StandardScaler()
            train_std = scaler.fit_transform(train_selected)
            
            pca = PCA(n_components=0.9)
            train_pca = pca.fit_transform(train_std)
            
            knn = NearestNeighbors(n_neighbors=min(self.k, len(train_pca)))
            knn.fit(train_pca)

            model_knns.append({
                'knn': knn,
                'scaler': scaler,
                'pca': pca,
                'selected_features': selected_features,
                'n_samples': len(train_pca),
            })

        # Cache only the expensive-to-recompute feature extractions; SHAP importance
        # and feature selection are recomputed fresh (cheap) on every load.
        if cache_path and cache_dirty:
            print(" Updating feature cache")
            np.savez(
                cache_path,
                features=features_calib.numpy(),
                labels=labels_calib,
                features_train=features_train.numpy(),
                labels_train=labels_train
            )
        
        return model_knns
    
    @staticmethod
    def _distances_from_fitted(model, model_knns, data_loader, device, layer, num_classes, model_idx):
        """
        Core per-fold distance computation, shared by sequential and parallel compute paths.
        Runs feature extraction + per-class KNN lookup for a single fold's model.
        """
        # Extract test features + predictions
        features_test, labels_test, _, predicted_classes = extract_latent_space_and_compute_shap_importance(
            model, data_loader, device, layer, importance=False
        )

        # Ensure proper types
        labels_test = np.array(labels_test, dtype=int)
        predicted_classes = np.round(predicted_classes).astype(int)

        print(f"\n  Model {model_idx+1}: Processing {len(predicted_classes)} test samples")
        if predicted_classes.ndim > 1:
            predicted_classes = np.argmax(predicted_classes, axis=1)
        print(f" Predicted classes: {np.bincount(predicted_classes)}")
        print(f" True classes: {np.bincount(labels_test)}")

        test_df = pd.DataFrame(
            features_test.numpy(),
            columns=[f"Feature_{i}" for i in range(features_test.shape[1])]
        )

        distances_per_sample = np.zeros(len(test_df))

        # Compute distances per PREDICTED class
        for class_idx in range(num_classes):
            if model_knns[class_idx] is None:
                continue

            fitted = model_knns[class_idx]

            # Test samples PREDICTED as this class
            class_mask = (predicted_classes == class_idx)
            n_samples_class = class_mask.sum()

            if n_samples_class == 0:
                continue

            test_class_df = test_df[class_mask]

            # Use SHAP features for this class
            test_selected = test_class_df[fitted['selected_features']].values

            # Transform
            test_std = fitted['scaler'].transform(test_selected)
            test_pca = fitted['pca'].transform(test_std)

            # KNN distances
            distances, _ = fitted['knn'].kneighbors(test_pca)
            avg_distances = distances.mean(axis=1)

            # Debug
            class_labels = labels_test[class_mask]
            n_correct = (class_labels == class_idx).sum()
            n_incorrect = (class_labels != class_idx).sum()

            print(f" Class {class_idx}: {n_samples_class} pred ({n_correct} , {n_incorrect} )")
            if n_correct > 0:
                correct_dists = avg_distances[class_labels == class_idx]
                print(f" Correct: {correct_dists.mean():.3f}±{correct_dists.std():.3f}")
            if n_incorrect > 0:
                incorrect_dists = avg_distances[class_labels != class_idx]
                print(f" Incorrect: {incorrect_dists.mean():.3f}±{incorrect_dists.std():.3f}")

            indices = np.where(class_mask)[0]
            distances_per_sample[indices] = avg_distances

        return distances_per_sample

    def _compute_sequential(self, models, data_loader, device):
        """Sequential per-fold distance computation (single GPU)."""
        all_distances = []
        for model_idx, (model, model_knns) in enumerate(zip(models, self.fitted_models)):
            model.eval()
            layer = get_layer_from_model(model, self.layer_name)
            distances_per_sample = self._distances_from_fitted(
                model, model_knns, data_loader, device, layer, self.num_classes, model_idx
            )
            all_distances.append(distances_per_sample)
        return all_distances

    def _compute_parallel(self, models, data_loader):
        """Parallel per-fold distance computation, one fold per GPU worker process."""
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPUs available for parallel processing")

        print(f" Detected {n_gpus} GPUs, distributing {len(models)} folds for compute()...")

        try:
            torch_mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        # Test loader is shared across all folds - extract its params once
        test_loader_params = self._extract_loader_params(data_loader)
        test_dataset = test_loader_params['dataset']

        max_concurrent = min(self.n_jobs, n_gpus)
        per_worker_num_workers = max(1, (test_loader_params.get('num_workers') or 0) // max_concurrent)
        # Dataset is identical across all folds - drop it from the per-task args (shipped once
        # via the pool initializer instead of re-pickled on every submitted fold task).
        test_loader_meta = {k: v for k, v in test_loader_params.items() if k != 'dataset'}

        fold_args = []
        for fold_idx, model in enumerate(models):
            gpu_id = fold_idx % n_gpus
            model_cpu = model.cpu()
            fold_args.append((
                fold_idx, model_cpu, self.fitted_models[fold_idx],
                test_loader_meta, gpu_id, self.layer_name, self.num_classes, per_worker_num_workers
            ))
            print(f" Fold {fold_idx} → GPU {gpu_id}")

        all_distances = [None] * len(models)

        with ProcessPoolExecutor(
            max_workers=max_concurrent,
            initializer=_init_shap_pool_worker,
            initargs=(None, None, test_dataset)
        ) as executor:
            future_to_fold = {
                executor.submit(_compute_fold_worker_multigpu, *args): args[0]
                for args in fold_args
            }

            for future in as_completed(future_to_fold):
                fold_idx = future_to_fold[future]
                try:
                    idx, distances_per_sample = future.result()
                    all_distances[idx] = distances_per_sample
                    print(f" Fold {idx} compute complete")
                except Exception as e:
                    print(f" Fold {fold_idx} compute failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

        return all_distances

    def compute(self, models, data_loader, device, return_per_fold=True):
        """
        Compute KNN-SHAP distances per fold (or averaged for backward compatibility).
        
        Args:
            models: List of models (one per fold)
            data_loader: Test data loader
            device: torch device
            return_per_fold: If True, return [num_folds, N]. If False, return [N] (averaged, legacy)
        
        Returns:
            np.ndarray: 
                - If return_per_fold=True: [num_folds, N] distances per fold
                - If return_per_fold=False: [N] averaged distances (backward compatible)
        """
        if not self.fitted_models:
            raise RuntimeError("Call fit() before compute()")
        
        if not isinstance(models, list):
            models = [models]
        
        if len(models) != len(self.fitted_models):
            raise ValueError(f"Expected {len(self.fitted_models)} models, got {len(models)}")

        if self.parallel and len(models) > 1:
            print(f"\n  Parallel compute: {self.n_jobs} workers across {len(models)} folds")
            all_distances = self._compute_parallel(models, data_loader)
            # Workers move models to their own GPU; restore to the caller's device.
            print(f" Moving models back to {device}...")
            for model in models:
                model.to(device)
        else:
            all_distances = self._compute_sequential(models, data_loader, device)
        
        all_distances = np.array(all_distances)  # [num_folds, N] - z-scored per fold
        
        if return_per_fold:
            print(f"\n  Computed KNN-SHAP distances for {len(self.fitted_models)} folds (z-scored per fold)")
            return all_distances  # [num_folds, N]
        else:
            # Legacy mode: average across models
            final_distances = np.mean(all_distances, axis=0)
            print(f"\n  Computed KNN-SHAP distances (z-scored per fold, then averaged over {len(self.fitted_models)} folds)")
            return final_distances

class HyperplaneDistanceMethod(UQMethod):
    """
    Uncertainty quantification via distance to SVM hyperplane in latent space.
    """
    def __init__(self, layer_to_hook):
        super().__init__("Hyperplane-Distance")
        self.layer = layer_to_hook
        self.svm = None
        self.scaler = None
    
    def fit(self, model, train_loader, device):
        """
        Train SVM on training latent space.
        """
        features, labels, _, _ = extract_latent_space_and_compute_shap_importance(
            model, train_loader, device, self.layer, importance=False
        )
        
        self.scaler = StandardScaler()
        features_std = self.scaler.fit_transform(features.numpy())
        
        self.svm = SVC(kernel="linear")
        self.svm.fit(features_std, labels)
        
        return self
    
    def compute(self, model, data_loader, device):
        """
        Compute signed distances to hyperplane.
        
        Returns:
            np.ndarray: Distances (N,) - absolute value = uncertainty
        """
        if self.svm is None:
            raise RuntimeError("Call fit() before compute()")
        
        features, labels, success, _ = extract_latent_space_and_compute_shap_importance(
            model, data_loader, device, self.layer, importance=False
        )
        
        features_std = self.scaler.transform(features.numpy())
        distances = self.svm.decision_function(features_std)
        
        # Absolute distance = uncertainty
        return np.abs(distances)


def extract_latent_space_and_compute_shap_importance(
    model, data_loader, device, layer_to_be_hooked,
    importance=True, classifierheadwrapper=None, max_background_samples=1000
):
    """
    Extract latent features and optionally compute SHAP values.

    SHAP is computed analytically: the classifier head is a single affine nn.Linear,
    so DeepSHAP reduces exactly to phi_i,c(x) = W[c, i] * (x_i - mean_B[x_i]) - see
    `compute_analytic_shap_values`. No gradient/explainer library involved.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation
        device: torch.device
        layer_to_be_hooked: Layer to hook (e.g., model.avgpool)
        importance: Whether to compute SHAP values
        classifierheadwrapper: Wrapped classifier head for SHAP
        max_background_samples: Max samples used for the SHAP background mean
        
    Returns:
        If importance=True: (shap_values, features, labels, success_flags)
        If importance=False: (features, labels, success_flags, predictions)
    
    Example:
        >>> # Extract features only
        >>> features, labels, success, preds = extract_latent_space_and_compute_shap_importance(
        ...     model, test_loader, device, model.avgpool, importance=False
        ... )
        
        >>> # Compute SHAP values
        >>> shap_vals, features, labels, success = extract_latent_space_and_compute_shap_importance(
        ...     model, test_loader, device, model.avgpool,
        ...     importance=True, classifierheadwrapper=classifier_head
        ... )
    """
    model.eval()

    penultimate_features = []
    all_labels = []
    success_flags = []
    predictions = []
    
    def hook(module, input, output):
        # Handle ViT encoder output: [B, num_patches+1, hidden_dim]
        # Extract CLS token (first token) for ViT models
        if output.dim() == 3 and hasattr(model, 'encoder'):
            # ViT encoder output: take CLS token [:, 0, :]
            # Move to CPU immediately to prevent GPU memory buildup
            penultimate_features.append(output[:, 0, :].detach().cpu())
        else:
            # Standard CNNs: flatten spatial dimensions
            # Move to CPU immediately to prevent GPU memory buildup
            penultimate_features.append(output.detach().flatten(1).cpu())

    hook_handle = layer_to_be_hooked.register_forward_hook(hook)

    with torch.no_grad():
        is_binary = None
        for batch_idx, batch in enumerate(data_loader):
            if isinstance(batch, dict):
                batch = (batch['image'], batch['label'])

            images = batch[0].to(device, non_blocking=True)
            labels_t = batch[1].to(device, non_blocking=True)
            
            labels_flat = labels_t.view(-1).long()
            all_labels.extend(labels_flat.cpu().numpy().tolist())

            logits = model(images)
            
            if is_binary is None:
                is_binary = (logits.shape[1] == 1)

            if is_binary:
                probs = torch.sigmoid(logits).squeeze(1)
                preds_cls = (probs > 0.5).long()
                success_flags.extend((preds_cls == labels_flat).cpu().numpy().astype(int).tolist())
                predictions.extend(probs.cpu().numpy().tolist())
            else:
                probs = torch.softmax(logits, dim=1)
                preds_cls = probs.argmax(dim=1)
                success_flags.extend((preds_cls == labels_flat).cpu().numpy().astype(int).tolist())
                predictions.extend(probs.cpu().numpy().tolist())
            
            # Drop references so activations aren't held past the batch's scope.
            # ponytail: no periodic empty_cache() here - forces a CUDA sync every N
            # batches which stalls throughput; a single cache clear after the loop
            # (below) is enough on 48GB GPUs. Re-add per-batch clearing only if OOM
            # resurfaces on smaller GPUs.
            del images, labels_t, logits, probs

    hook_handle.remove()

    # Concatenate features (already on CPU from hook)
    features = torch.cat(penultimate_features)
    labels = np.array(all_labels)
    success_flags = np.array(success_flags)
    
    # Clear GPU cache to free memory for next fold
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if importance:
        if classifierheadwrapper is None:
            raise ValueError("classifierheadwrapper required when importance=True")
        shap_values = compute_analytic_shap_values(features, classifierheadwrapper, max_background_samples, batch_size=5000)
        return shap_values, features, labels, success_flags
    else:
        # No SHAP computation - just return features (already on CPU)
        return features, labels, success_flags, predictions


def compute_analytic_shap_values(features, classifierheadwrapper, max_background_samples=1000, batch_size=5000, background=None):
    """
    SHAP importance for an affine classifier head using ``shap.LinearExplainer``.

    The explainer is initialized once from a background subset and evaluates samples
    in chunks to avoid OOM on large datasets.

    Args:
        features: [N, F] CPU tensor of penultimate-layer features to explain
        classifierheadwrapper: ClassifierHeadWrapper exposing the model's nn.Linear head
        max_background_samples: Max samples used for the SHAP background
        batch_size: Process this many features at a time (default 5000, reduce if OOM)
        background: Optional [M, F] tensor/array used as the SHAP background distribution.
            If None (default), a subset of `features` itself is used as background
            (legacy behavior). Pass a disjoint set (e.g. held-out calibration features)
            to avoid self-referential background when `features` = samples being explained.

    Returns:
        np.ndarray: [N, F, C] SHAP values (C=1 for a single-logit binary head)
    """
    import shap

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(features) == 0:
        raise ValueError("features must contain at least one sample")

    if background is not None:
        background = background.numpy() if torch.is_tensor(background) else np.asarray(background)
    else:
        background = features.numpy()
    if len(background) > max_background_samples:
        indices = np.random.choice(len(background), max_background_samples, replace=False)
        background = background[indices]

    linear_head = classifierheadwrapper.get_linear()
    model_coefficients = linear_head.weight.detach().cpu().numpy()
    model_intercept = linear_head.bias.detach().cpu().numpy()
    explainer = shap.LinearExplainer(
        (model_coefficients, model_intercept),
        background
    )
    
    # Process features in batches to avoid OOM on large datasets
    shap_batches = []
    num_batches = (len(features) + batch_size - 1) // batch_size
    
    if num_batches > 1:
        print(f"  SHAP computation: processing {len(features)} samples in {num_batches} batches (size={batch_size})...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(features))
        
        features_batch = features[start_idx:end_idx]  # [B, F]
        shap_batch = explainer(features_batch.numpy())
        shap_batch = np.asarray(getattr(shap_batch, "values", shap_batch))
        if shap_batch.ndim == 2:
            shap_batch = shap_batch[:, :, np.newaxis]
        elif shap_batch.ndim == 3 and shap_batch.shape[0] == len(features_batch):
            pass
        elif isinstance(shap_batch, list):
            shap_batch = np.stack(shap_batch, axis=-1)
        else:
            raise ValueError(f"Unexpected LinearExplainer output shape: {shap_batch.shape}")
        shap_batches.append(shap_batch)
        
        if num_batches > 1 and (batch_idx + 1) % max(1, num_batches // 5) == 0:
            print(f"   Batch {batch_idx + 1}/{num_batches} complete")
    
    # Concatenate batches along the sample axis
    return np.concatenate(shap_batches, axis=0)

def compute_mean_shap_values(shap_values, fold, true_labels=None, nb_features=50):
    """
    Compute mean absolute SHAP values per class.

    Args:
        shap_values: SHAP values array (2D or 3D)
        fold: Fold index for labeling
        true_labels: True labels for binary classification class filtering
        nb_features: Number of top features to keep

    Returns:
        list: [(fold, class_idx, shap_importance_series), ...]
    
    Example:
        >>> mean_shap = compute_mean_shap_values(shap_vals, fold=0, true_labels=labels)
        >>> # [(0, 0, Series([0.5, 0.3, ...])), (0, 1, Series([0.4, 0.2, ...]))]
    """
    mean_shap_fold = []
    print(f"SHAP Feature Importances Computation (Fold {fold})")

    if shap_values.ndim == 3 and true_labels is not None and len(np.unique(true_labels)) == 2:
        shap_values = shap_values.squeeze(-1)

    if shap_values.ndim == 3:
        num_samples, num_features, num_classes = shap_values.shape
    elif shap_values.ndim == 2:
        num_samples, num_features = shap_values.shape
        num_classes = 2
    else:
        raise ValueError("Expected 2D or 3D SHAP values array")

    for class_idx in range(num_classes):
        print(f" Class {class_idx}: Computing SHAP importances")

        if shap_values.ndim == 3:
            class_shap_values = shap_values[:, :, class_idx]
        else:
            class_shap_values = shap_values[true_labels == class_idx, :]

        shap_df = pd.DataFrame(
            class_shap_values,
            columns=[f"Feature_{i}" for i in range(num_features)]
        )

        mean_abs_shap = shap_df.abs().mean(axis=0)
        top_n_features = mean_abs_shap.nlargest(nb_features).index
        shap_df_top_n = shap_df[top_n_features]

        shap_importance = display_shap_values(shap_df_top_n)
        mean_shap_fold.append((fold, class_idx, shap_importance))

    return mean_shap_fold


def display_shap_values(shap_df):
    """
    Compute mean absolute SHAP values for display.

    Args:
        shap_df: DataFrame of SHAP values (samples x features)

    Returns:
        pd.Series: Mean absolute SHAP values per feature (sorted descending)
    """
    shap_importance = shap_df.abs().mean().sort_values(ascending=False)
    return shap_importance

def feature_engineering_pipeline(mean_shap_df, latent_space, shap_threshold=0.05, corr_threshold=0.8):
    """
    DEPRECATED: Use n_shap_features parameter in KNNLatentSHAPMethod instead.
    
    Simplified version: just select top features by SHAP importance.
    PCA will handle redundancy automatically.
    """
    import warnings
    warnings.warn(
        "feature_engineering_pipeline is deprecated. "
        "Use KNNLatentSHAPMethod(n_shap_features=N) instead.",
        DeprecationWarning
    )
    
    # Just select features above threshold
    retained_features = mean_shap_df[mean_shap_df > shap_threshold].index
    retained_features = retained_features.intersection(latent_space.columns)
    
    retained_latent_space = latent_space[retained_features]
    final_features = retained_features.tolist()
    
    print(f"Selected {len(final_features)} features above SHAP threshold {shap_threshold}")
    
    return retained_latent_space, final_features


def analyze_hyperplane_distance(train_latent, train_labels, eval_latent, eval_success, display_distrib=False):
    """
    Train SVM and compute distances to hyperplane.

    Args:
        train_latent: Training features (samples x features)
        train_labels: Training labels
        eval_latent: Evaluation features
        eval_success: Success flags for evaluation set
        display_distrib: Whether to plot distributions

    Returns:
        np.ndarray: Signed distances for evaluation set
    """
    svm = SVC(kernel="linear")
    svm.fit(train_latent, train_labels)

    eval_distances = svm.decision_function(eval_latent)

    success_distances = eval_distances[eval_success == 1]
    failure_distances = eval_distances[eval_success == 0]
    
    scaler = StandardScaler()
    normalized_distances = scaler.fit_transform(eval_distances.reshape(-1, 1)).flatten()
    
    success_distances_norm = normalized_distances[eval_success == 1]
    failure_distances_norm = normalized_distances[eval_success == 0]
    
    if display_distrib:
        plt.figure(figsize=(8, 6))
        sns.histplot(success_distances_norm, color='green', label='Success', kde=True, stat="count")
        sns.histplot(failure_distances_norm, color='red', label='Failure', kde=True, stat="count")
        plt.axvline(0, color='black', linestyle='dashed', label='Decision Boundary')
        plt.xlabel("Normalized Distance to Hyperplane")
        plt.ylabel("Count")
        plt.title("Distance to Hyperplane (Success vs Failure)")
        plt.legend()
        plt.show()
    
    return eval_distances


def compute_knn_distances_to_train_data(
    model, train_loader, test_loader, layer, device,
    latent_spaces, mean_shap_importances, num_classes
):
    """
    Compute KNN distances to training data per class in latent space.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        layer: Layer to hook for features
        device: torch.device
        latent_spaces: DataFrame of latent features
        mean_shap_importances: List of (fold, class, importance_series)
        num_classes: Number of classes

    Returns:
        tuple: (knn_distances, success_flags) both as np.ndarray (N,)
    
    Example:
        >>> distances, success = compute_knn_distances_to_train_data(
        ...     model, train_loader, test_loader, model.avgpool, device,
        ...     latent_df, mean_shap, num_classes=2
        ... )
    """
    # Extract train features
    latent_space_training, labels_training, _, _ = extract_latent_space_and_compute_shap_importance(
        model, train_loader, device, layer, importance=False
    )
    
    # Extract test features
    latent_space_test, labels_test, success_test, _ = extract_latent_space_and_compute_shap_importance(
        model, test_loader, device, layer, importance=False
    )
    
    train_latent_space = pd.DataFrame(latent_space_training.numpy(), columns=latent_spaces.columns)
    test_latent_space = pd.DataFrame(latent_space_test.numpy(), columns=latent_spaces.columns)
    
    knn_distances_all = np.zeros(len(test_latent_space))
    successes_all = np.zeros(len(test_latent_space))
    
    for i in range(num_classes):
        print(f'Processing class {i}')
        
        # Get SHAP-selected features for this class
        important_features = mean_shap_importances[i][2].keys()
        train_latent_class = train_latent_space[important_features]
        
        # Filter by class
        mask_train = labels_training == i
        train_filtered = train_latent_class[mask_train]
        print(f'  Train samples: {len(train_filtered)}')
        
        test_latent_class = test_latent_space[important_features]
        mask_test = labels_test == i
        test_filtered = test_latent_class[mask_test]
        print(f'  Test samples: {len(test_filtered)}')
        
        success_filtered = success_test[mask_test.flatten()]
        indices_filtered = np.where(mask_test.flatten())[0]
        
        # Standardize and PCA
        scaler = StandardScaler()
        train_std = scaler.fit_transform(train_filtered)
        
        pca = PCA(n_components=0.9)
        train_pca = pca.fit_transform(train_std)
        
        test_std = scaler.transform(test_filtered)
        test_pca = pca.transform(test_std)
        
        # KNN
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(train_pca)
        distances, _ = knn.kneighbors(test_pca)
        avg_distances = distances.mean(axis=1)
        
        knn_distances_all[indices_filtered] = avg_distances
        successes_all[indices_filtered] = success_filtered

    return knn_distances_all, successes_all