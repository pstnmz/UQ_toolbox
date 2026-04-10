"""
Latent space-based uncertainty quantification methods.
Includes SHAP importance, KNN distances, feature engineering, and hyperplane distance analysis.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as torch_mp

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

def _fit_fold_worker_multigpu(fold_idx, model, train_loader_params, calib_loader_params, device_str, flag, gpu_id, cache_dir=None):
    """
    Worker function for parallel fold fitting with explicit GPU assignment.
    
    Args:
        fold_idx: Fold index
        model: PyTorch model (on CPU)
        train_loader_params: Dict with keys: 'dataset', 'indices', 'batch_size', 'shuffle', 'num_workers'
        calib_loader_params: Dict with keys: 'dataset', 'indices', 'batch_size', 'shuffle', 'num_workers'
        device_str: Device string (e.g., 'cuda:1')
        flag: Dataset name for caching
        gpu_id: GPU ID for this worker
        cache_dir: Directory for SHAP cache (None to disable)
    
    Returns:
        List of per-class KNN dicts
    """
    import torch
    import gc
    import os
    from torch.utils.data import DataLoader, Subset
    
    # Set CUDA device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)  # After setting visible devices, use device 0
    
    # Reconstruct device (now points to the assigned GPU)
    device = torch.device('cuda:0')
    
    # Reconstruct DataLoaders from parameters
    train_subset = Subset(train_loader_params['dataset'], train_loader_params['indices'])
    train_loader = DataLoader(
        train_subset,
        batch_size=train_loader_params['batch_size'],
        shuffle=train_loader_params.get('shuffle', False),
        num_workers=0,  # Always 0 within worker to avoid nested multiprocessing
        pin_memory=True
    )
    
    calib_subset = Subset(calib_loader_params['dataset'], calib_loader_params['indices'])
    calib_loader = DataLoader(
        calib_subset,
        batch_size=calib_loader_params['batch_size'],
        shuffle=calib_loader_params.get('shuffle', False),
        num_workers=0,
        pin_memory=True
    )
    
    print(f" Worker {fold_idx}: Using GPU {gpu_id}")
    
    # Move model to assigned GPU
    model = model.to(device)
    model.eval()
    
    # Reduce background samples for parallel mode (saves GPU memory)
    max_bg_samples = 1000
    
    # Create a temporary method instance for this fold
    temp_method = KNNLatentSHAPMethod(
        layer_name='avgpool',
        k=5,
        n_shap_features=50,
        cache_dir=cache_dir,
        max_background_samples=max_bg_samples
    )
    
    # Fit this fold
    model_knns = temp_method._fit_single_fold(
        fold_idx, model, train_loader, calib_loader, device, flag
    )
    
    # Cleanup GPU memory
    model = model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f" Worker {fold_idx}: Done on GPU {gpu_id}")
    
    return model_knns


# ============================================================================
# CLASS-BASED API
# ============================================================================

class ClassifierHeadWrapper(nn.Module):
    """
    Generic wrapper for classifier head extraction.
    Works with ResNets, ViTs, EfficientNets, etc.
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
        
    def forward(self, x):
        """
        Forward pass from latent features to predictions.
        
        Args:
            x: Latent features (B, feature_dim) - already flattened/CLS token extracted
        
        Returns:
            Predictions (logits or probabilities)
        """
        # For ResNets: avgpool -> flatten -> fc
        if hasattr(self.model, 'fc'):
            return self.model.fc(x)
        
        # For ViTs: encoder CLS token -> ln (layer norm) -> heads
        elif hasattr(self.model, 'heads'):
            # Apply layer norm if present (ViT architecture)
            if hasattr(self.model.encoder, 'ln'):
                x = self.model.encoder.ln(x)
            return self.model.heads(x)
        
        elif hasattr(self.model, 'head'):
            # Apply layer norm if present
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'ln'):
                x = self.model.encoder.ln(x)
            return self.model.head(x)
        
        # For EfficientNet/MobileNet: usually 'classifier'
        elif hasattr(self.model, 'classifier'):
            return self.model.classifier(x)
        
        # Fallback: try to find any Linear layer at the end
        else:
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Linear):
                    return module(x)
            
            raise RuntimeError(
                "Could not identify classifier head. "
                "Please manually specify the classifier in ClassifierHeadWrapper."
            )


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
                'layer': layer
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
      1. Compute SHAP on calibration → select top-k features per class
      2. For each test sample:
         - Get predicted class
         - Extract latent features, keep only top-k for that class
         - Compute KNN distance to training samples (true label = predicted class)
      3. Average distances across all models
    """
    def __init__(self, layer_name='avgpool', k=5, n_shap_features=50, 
                 max_background_samples=1000, cache_dir=None, parallel=False, n_jobs=None):
        super().__init__("KNN-Latent-SHAP")
        self.layer_name = layer_name
        self.k = k
        self.n_shap_features = n_shap_features
        self.max_background = max_background_samples
        self.cache_dir = cache_dir
        self.parallel = parallel 
        self.fitted_models = []
        self.num_classes = None
        
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
            
            model_knns = self._fit_single_fold(
                fold_idx, model, train_loader, calib_loader, device, flag
            )
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
        
        fold_args = []
        for fold_idx, (model, train_loader) in enumerate(zip(models, train_loaders)):
            gpu_id = fold_idx % n_gpus
            device_str = f'cuda:{gpu_id}'
            model_cpu = model.cpu()
            
            # Extract train loader parameters for this fold
            train_loader_params = self._extract_loader_params(train_loader)
            
            fold_args.append((fold_idx, model_cpu, train_loader_params, calib_loader_params, device_str, flag, gpu_id, self.cache_dir))
            print(f" Fold {fold_idx} → GPU {gpu_id}")
        
        fitted_models = [None] * len(models)
        
        with ProcessPoolExecutor(max_workers=min(self.n_jobs, n_gpus)) as executor:
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
        # Check cache first
        # =====================================================================
        cache_path = self._get_cache_path(flag, fold_idx)
        shap_values, features_calib, labels_calib = None, None, None
        
        if cache_path and os.path.exists(cache_path):
            print(f" Loading cached SHAP from {os.path.basename(cache_path)}")
            try:
                cache = np.load(cache_path, allow_pickle=True)
                shap_values = cache['shap_values']
                features_calib = torch.from_numpy(cache['features'])
                labels_calib = cache['labels']
                
                # FIX: Infer num_classes from labels, not SHAP shape
                # For binary classification, SHAP may be 2D (N, features) but we still have 2 classes
                inferred_num_classes = len(np.unique(labels_calib))
                
                if self.num_classes is None:
                    self.num_classes = inferred_num_classes
                    print(f" Inferred {self.num_classes} classes from cached labels")
                
            except Exception as e:
                print(f" [WARNING] Cache load failed: {e}, recomputing...")
                shap_values = None
        
        # Compute SHAP if not cached
        if shap_values is None:
            print(f" Step 1: Computing SHAP on calibration...")
            shap_values, features_calib, labels_calib, _ = \
                extract_latent_space_and_compute_shap_importance(
                    model, calib_loader, device, layer,
                    importance=True,
                    classifierheadwrapper=classifierhead,
                    max_background_samples=self.max_background,
                    shap_batch_size=5000
                )
                
            # Cleanup GPU memory after SHAP
            del classifierhead
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Save to cache
            if cache_path:
                print(f" Saving SHAP to cache")
                np.savez_compressed(
                    cache_path,
                    shap_values=shap_values,
                    features=features_calib.numpy(),
                    labels=labels_calib
                )
        
        # =====================================================================
        # Compute mean SHAP per class
        # =====================================================================
        print(f" Step 2: Computing mean SHAP per class...")
        mean_shap_results = compute_mean_shap_values(
            shap_values, fold=fold_idx, true_labels=labels_calib, 
            nb_features=self.n_shap_features
        )
        
        # FIX: Always set num_classes (even if cached)
        if self.num_classes is None:
            self.num_classes = len(mean_shap_results)
            print(f" Detected {self.num_classes} classes")
        
        # Extract top features per class
        selected_features_per_class = []
        for class_idx in range(self.num_classes):
            mean_shap_series = mean_shap_results[class_idx][2]
            top_features = mean_shap_series.nlargest(self.n_shap_features).index.tolist()
            selected_features_per_class.append(top_features)
        
        # =====================================================================
        # Extract training features
        # =====================================================================
        print(f" Step 3: Extracting training features...")
        features_train, labels_train, _, _ = extract_latent_space_and_compute_shap_importance(
            model, train_loader, device, layer, importance=False
        )
        
        # Cleanup after feature extraction
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        train_df = pd.DataFrame(
            features_train.numpy(),
            columns=[f"Feature_{i}" for i in range(features_train.shape[1])]
        )
        
        # =====================================================================
        # Fit KNN per class
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
                'n_samples': len(train_pca)
            })
        
        return model_knns
    
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
        
        all_distances = []
        
        for model_idx, (model, model_knns) in enumerate(zip(models, self.fitted_models)):
            model.eval()
            
            # Get fresh layer reference from the MAIN process model
            layer = get_layer_from_model(model, self.layer_name)
            
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
            for class_idx in range(self.num_classes):
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
                
                # Debug (before normalization)
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
                
                # Standardize distances within this class
                # Each class uses different SHAP features + PCA space, so distances have different scales
                if len(avg_distances) > 1:
                    class_mean = avg_distances.mean()
                    class_std = avg_distances.std()
                    if class_std > 1e-10:
                        avg_distances = (avg_distances - class_mean) / class_std
                    else:
                        avg_distances = np.zeros_like(avg_distances)
                indices = np.where(class_mask)[0]
                distances_per_sample[indices] = avg_distances
            
            # Additional z-score normalization per fold (after per-class normalization)
            # This ensures fair averaging across folds
            fold_mean = np.mean(distances_per_sample)
            fold_std = np.std(distances_per_sample)
            if fold_std > 1e-10:
                distances_per_sample = (distances_per_sample - fold_mean) / fold_std
            
            all_distances.append(distances_per_sample)
        
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
    importance=True, classifierheadwrapper=None, max_background_samples=1000,
    shap_batch_size=500
):
    """
    Extract latent features and optionally compute SHAP values.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation
        device: torch.device
        layer_to_be_hooked: Layer to hook (e.g., model.avgpool)
        importance: Whether to compute SHAP values
        classifierheadwrapper: Wrapped classifier head for SHAP
        max_background_samples: Max samples for SHAP background
        shap_batch_size: Batch size for SHAP computation (smaller = faster but less accurate)
        
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
            
            # Free memory after each batch to prevent OOM with ViT models
            del images, labels_t, logits, probs
            if device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()

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
        
        # ====================================================================
        # Subsample BEFORE moving to GPU
        # ====================================================================
        print(f" Selecting {max_background_samples} background samples from {len(features)}...")
        
        if len(features) > max_background_samples:
            # Random sampling
            indices = np.random.choice(
                len(features), max_background_samples, replace=False
            )
            background_features = features[indices]
        else:
            background_features = features
        
        # NOW move subsampled features to GPU
        background_features = background_features.to(device)
        
        print(f" Computing SHAP with {len(background_features)} background samples (GPU mem: ~{background_features.element_size() * background_features.nelement() / 1e6:.1f}MB)...")
        
        # DeepExplainer with subsampled background
        explainer = shap.DeepExplainer(classifierheadwrapper, background_features.clone())
        
        # Process SHAP in batches to reduce memory
        all_shap_values = []
        n_samples = len(features)
        
        for start_idx in range(0, n_samples, shap_batch_size):
            end_idx = min(start_idx + shap_batch_size, n_samples)
            batch_features = features[start_idx:end_idx].to(device)
            
            batch_shap = explainer.shap_values(batch_features)
            all_shap_values.append(batch_shap)
            
            # Free GPU memory immediately after batch
            del batch_features
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if (start_idx // shap_batch_size) % 10 == 0:
                print(f" SHAP progress: {end_idx}/{n_samples} samples")
        
        # Cleanup explainer
        del explainer
        del background_features
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Concatenate batches (on CPU)
        if isinstance(all_shap_values[0], list):  # Multi-class
            shap_values = [
                np.concatenate([batch[i] for batch in all_shap_values], axis=0)
                for i in range(len(all_shap_values[0]))
            ]
        else:  # Binary
            shap_values = np.concatenate(all_shap_values, axis=0)
        
        return shap_values, features, labels, success_flags
    else:
        # No SHAP computation - just return features (already on CPU)
        return features, labels, success_flags, predictions

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