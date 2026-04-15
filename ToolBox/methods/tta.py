"""
Test-Time Augmentation (TTA) methods for uncertainty quantification.
Includes both functional API (TTA, apply_augmentations) and class-based API (TTAMethod, GPSMethod).
"""
import os
import re
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from .gps_augment.utils.randaugment import BetterRandAugment

from ..core.base import UQMethod
from ..core.utils import (
    get_batch_predictions,
    average_predictions,
    compute_stds,
    build_monai_cache_dataset,
    EnsurePIL,
    _CachedRandAugDataset,
    _dl_worker_init,
    to_1_channel
)


# ============================================================================
# CLASS-BASED API (new, recommended)
# ============================================================================

class TTAMethod(UQMethod):
    """
    TTA uncertainty quantification using fixed or random augmentations.
    """
    def __init__(self, transformations=None, n=2, m=45, nb_augmentations=10, **kwargs):
        """
        Args:
            transformations: List of augmentation policies or None (random if None)
            n: Number of ops per augmentation (BetterRandAugment)
            m: Magnitude of ops (BetterRandAugment)
            nb_augmentations: Number of augmentations (if transformations=None)
            **kwargs: Additional TTA parameters (image_size, mean, std, etc.)
        """
        super().__init__("TTA")
        self.transformations = transformations
        self.n = n
        self.m = m
        self.nb_augmentations = nb_augmentations
        self.kwargs = kwargs
    
    def compute(self, models, dataset, device, ensemble_mode=False, return_per_fold=False, seed=None, **run_kwargs):
        """
        Compute per-sample std across augmentations.
        
        Args:
            models: List of models or single model
            dataset: Dataset to evaluate
            device: torch.device
            ensemble_mode: If True, compute per-model uncertainty then average (for per-fold evaluation)
            return_per_fold: If True, return per-fold uncertainties [M, N] instead of averaging
            seed: Random seed for augmentations (None = use current RNG state)
            **run_kwargs: Additional arguments
        
        Returns:
            tuple or np.ndarray: 
                - If return_per_fold=True: (stds, per_fold_stds, per_fold_mean_preds) where
                  stds/per_fold_stds are [M, N] per-model uncertainties and
                  per_fold_mean_preds [M, N] are argmax of mean softmax across augmentations
                - Otherwise: stds [N]
        """
        stds, _, per_fold_stds, per_fold_mean_preds = TTA(
            self.transformations, models, dataset, device,
            nb_augmentations=self.nb_augmentations,
            usingBetterRandAugment=False,  # Use torchvision.RandAugment for TTA
            n=self.n, m=self.m,
            ensemble_mode=ensemble_mode,
            return_per_fold=return_per_fold,
            seed=seed,
            **{**self.kwargs, **run_kwargs}
        )
        
        if return_per_fold:
            return stds, per_fold_stds, per_fold_mean_preds
        return np.array(stds)


class GPSMethod(UQMethod):
    """
    Greedy Policy Search (GPS) for TTA-based uncertainty quantification.
    
    Discovers optimal augmentation policies on calibration set, then applies to test set.
    """
    def __init__(self, aug_folder, correct_calib, incorrect_calib, max_iter=50):
        """
        Args:
            aug_folder: Directory with pre-computed augmentation predictions (.npz)
            correct_calib: Indices of correct calibration predictions
            incorrect_calib: Indices of incorrect calibration predictions
            max_iter: Max policies to select per greedy search
        """
        super().__init__("GPS")
        self.aug_folder = aug_folder
        self.correct_calib = correct_calib
        self.incorrect_calib = incorrect_calib
        self.max_iter = max_iter
        self.policies = None
    
    def search_policies(self, **kwargs):
        """
        Search for optimal augmentation policies on calibration set.
        
        Args:
            **kwargs: Arguments for perform_greedy_policy_search
                (num_workers, top_k, num_searches, seed, etc.)
        
        Returns:
            list: Selected policy filenames (groups)
        """
        from ..search.greedy import perform_greedy_policy_search
        self.policies = perform_greedy_policy_search(
            self.aug_folder, self.correct_calib, self.incorrect_calib,
            max_iterations=self.max_iter, **kwargs
        )
        return self.policies
    
    def compute(self, models, dataset, device, n=2, m=45, ensemble_mode=False, return_per_fold=False, **kwargs):
        """
        Compute uncertainty using discovered policies.
        
        Args:
            models: List of models or single model
            dataset: Dataset to evaluate
            device: torch.device
            n: Number of ops per augmentation
            m: Magnitude parameter
            ensemble_mode: If True, compute per-model uncertainty then average (for per-fold evaluation)
            return_per_fold: If True, return per-fold uncertainties
            **kwargs: Additional arguments
        
        Returns:
            tuple or np.ndarray: 
                - If return_per_fold=True: (stds, per_fold_stds)
                - Otherwise: stds [N]
        """
        if self.policies is None:
            raise RuntimeError("Call search_policies() before compute()")
        
        print(f" Extracting {len(self.policies)} policy groups...")
        
        # Extract policies from search results
        transformation_pipeline = []
        extracted_m = None
        for group in self.policies:
            n_group, m_group, transformations_group = extract_gps_augmentations_info(group)
            transformation_pipeline.append(transformations_group)
            # Use M value from policy filenames (they encode the max_magnitude used during caching)
            if m_group is not None:
                extracted_m = m_group
        
        # Override m parameter with extracted M from filenames if available
        # This ensures max_magnitude matches the scale used when policies were generated
        if extracted_m is not None:
            print(f" Using M={extracted_m} extracted from policy filenames (overriding m={m})")
            m = extracted_m
        
        print(f" Extracted {len(transformation_pipeline)} transformation pipelines")
        
        if not transformation_pipeline:
            raise RuntimeError("No valid policies extracted from search results")
        
        # Call TTA in GPS mode
        stds, _, per_fold_stds, per_fold_mean_preds = TTA(
            transformations=transformation_pipeline,
            models=models,
            dataset=dataset,
            device=device,
            usingBetterRandAugment=True,
            n=n, m=m,
            is_gps_mode=True,
            average_groups=True,
            ensemble_mode=ensemble_mode,
            return_per_fold=return_per_fold,
            **kwargs
        )
        
        if return_per_fold:
            return stds, per_fold_stds, per_fold_mean_preds
        return np.array(stds)


# ============================================================================
# FUNCTIONAL API (existing, for backward compatibility)
# ============================================================================

def _batched_augmentation_inference(augmented_inputs, models, device, batch_size, ensemble_mode=False):
    """
    Process multiple augmentations in a single forward pass (faster).
    
    Args:
        augmented_inputs: [K, N, C, H, W] where K=num_augmentations, N=num_samples
        models: List of models
        device: torch.device
        batch_size: Max batch size
        ensemble_mode: If True, return per-model predictions [M, K, N, num_classes]
                       If False, average across models first [K, N, num_classes]
    
    Returns:
        torch.Tensor: 
            - If ensemble_mode=False: [K, N, num_classes] (averaged across models)
            - If ensemble_mode=True: [M, K, N, num_classes] (per-model predictions)
    """
    K, N, C, H, W = augmented_inputs.shape
    
    # Ensure models is a list
    if not isinstance(models, list):
        models = [models]
    M = len(models)
    
    # Check if we can fit all augmentations in memory
    total_samples = K * N
    
    # Use batch_size as the limit (respects user's --batch-size flag)
    if total_samples <= batch_size:
        # Fast path: batch all augmentations together
        print(f" Using batched inference ({K} augmentations × {N} samples = {total_samples})")
        
        # Reshape: [K, N, C, H, W] → [K*N, C, H, W]
        imgs_batched = augmented_inputs.reshape(total_samples, C, H, W)
        
        # Single forward pass - returns [K*N, M, num_classes]
        batch_preds = get_batch_predictions(models, imgs_batched, device)  # [K*N, M, num_classes]
        
        if ensemble_mode:
            # Reshape to [M, K, N, num_classes]
            num_classes = batch_preds.shape[2]
            predictions = batch_preds.reshape(K, N, M, num_classes).permute(2, 0, 1, 3)  # [M, K, N, num_classes]
        else:
            # Average across models first
            avg_preds = average_predictions(batch_preds)  # [K*N, num_classes]
            # Reshape back: [K*N, num_classes] → [K, N, num_classes]
            predictions = avg_preds.reshape(K, N, -1)
        
        return predictions
    else:
        # Fallback: process each augmentation separately with proper batching
        print(f" Using sequential inference (total samples {total_samples} > batch_size {batch_size})")
        
        if ensemble_mode:
            # Store per-model predictions: [M, K, N, num_classes]
            preds_per_model = [[] for _ in range(M)]
            
            for k in range(K):
                dataset_aug = TensorDataset(augmented_inputs[k])
                loader = DataLoader(dataset_aug, batch_size=batch_size, pin_memory=True)
                
                all_preds = []
                for batch in loader:
                    imgs = batch[0].to(device)
                    batch_predictions = get_batch_predictions(models, imgs, device)  # [B, M, C]
                    all_preds.append(batch_predictions)
                
                aug_preds = torch.cat(all_preds, dim=0)  # [N, M, num_classes]
                
                # Split by model
                for model_idx in range(M):
                    preds_per_model[model_idx].append(aug_preds[:, model_idx, :])  # [N, num_classes]
            
            # Stack: M x [K, N, num_classes] → [M, K, N, num_classes]
            predictions = torch.stack([torch.stack(model_preds, dim=0) for model_preds in preds_per_model], dim=0)
        else:
            # Average across models during inference
            preds_list = []
            
            for k in range(K):
                dataset_aug = TensorDataset(augmented_inputs[k])
                loader = DataLoader(dataset_aug, batch_size=batch_size, pin_memory=True)
                
                all_preds = []
                for batch in loader:
                    imgs = batch[0].to(device)
                    batch_predictions = get_batch_predictions(models, imgs, device)
                    avg_preds = average_predictions(batch_predictions)
                    all_preds.append(avg_preds)
                
                preds_list.append(torch.cat(all_preds, dim=0))  # [N, num_classes]
            
            predictions = torch.stack(preds_list, dim=0)  # [K, N, num_classes]
        
        return predictions
    
def extract_gps_augmentations_info(policies):
    """
    Parse N, M and policy ops from filenames of form:
      N{N}M{M}__{A}_np.float64_{magA}__{B}_np.float64_{magB}__.npz
    Returns: N (int), M (int), [ [ (op_index, magnitude), ... ], ... ]
    """
    if not policies:
        return None, None, []
    if isinstance(policies, str):
        policies = [policies]

    # Determine valid op index range from BetterRandAugment
    try:
        _probe = BetterRandAugment(n=2, m=45, rand_m=True, resample=False, image_size=51)
        max_valid_index = len(_probe.augment_list) - 1
    except Exception:
        max_valid_index = None  # fall back to accepting provided indices

    # Regex components for both styles (with or without underscore between N and M)
    nm_regex = re.compile(r'N(?P<N>\d+)_?M(?P<M>\d+)')
    pair_regex = re.compile(r'__(?P<idx>\d+)_np\.float64_(?P<mag>-?\d+(?:\.\d+)?)')

    formatted_policies = []
    N_global = None
    M_global = None

    for fname in policies:
        base = os.path.basename(fname)
        stem = base[:-4] if base.endswith('.npz') else base

        # Extract N and M
        nm_match = nm_regex.search(stem)
        if nm_match:
            N_val = int(nm_match.group('N'))
            M_val = int(nm_match.group('M'))
            if N_global is None:  # record from first file
                N_global, M_global = N_val, M_val
        else:
            # Default if missing
            if N_global is None:
                N_global, M_global = 2, 45

        # Extract all (idx, magnitude) pairs
        pairs = []
        for m in pair_regex.finditer(stem):
            idx_raw = int(m.group('idx'))
            mag_raw = float(m.group('mag'))
            mag_cast = int(mag_raw) if abs(mag_raw - int(mag_raw)) < 1e-6 else mag_raw

            if max_valid_index is not None and (idx_raw < 0 or idx_raw > max_valid_index):
                # Strict: raise or clamp. Here we clamp and warn.
                idx_clamped = idx_raw % (max_valid_index + 1)
                print(f"Policy index {idx_raw} out of range [0,{max_valid_index}]; remapped to {idx_clamped}")
                idx_raw = idx_clamped

            pairs.append((idx_raw, mag_cast))

        if not pairs:
            print(f"Warning: could not parse policy ops from '{base}'. Expected pattern N*M*__A_np.float64_mag__B_np.float64_mag__")
        formatted_policies.append(pairs)

    # Ensure global N/M defaults
    if N_global is None:
        N_global = 2
    if M_global is None:
        M_global = 45
    return N_global, M_global, formatted_policies


def TTA(transformations, models, dataset, device, nb_augmentations=10, 
        usingBetterRandAugment=False, n=2, m=45, image_normalization=False, 
        nb_channels=1, mean=None, std=None, image_size=51, batch_size=None, 
        use_monai_cache=False, cache_rate=1.0, cache_num_workers=0, 
        dataloader_workers=None, is_gps_mode=False, average_groups=True,
        ensemble_mode=False, return_per_fold=False, seed=None):
    """
    Perform Test-Time Augmentation (TTA) on a batch of images.
    
    Args:
        ...existing args...
        is_gps_mode: If True, expect transformations = [[group1], [group2], [group3]]
        average_groups: If True (default), average std across groups in GPS mode
        ensemble_mode: If True, compute uncertainty per-model then average (for per-fold evaluation)
                       If False (default), average models first then compute uncertainty (ensemble evaluation)
        return_per_fold: If True, return per-fold uncertainties [M, N] instead of averaging
        seed: Random seed for augmentations. If None (default), uses current PyTorch RNG state.
              Set to an integer for reproducibility, or use time.time_ns() for randomness.
    
    Returns:
        tuple: (stds, averaged_predictions, per_fold_stds)
            - If return_per_fold=True: stds is [M, N] per-model uncertainties
            - If ensemble_mode=True and return_per_fold=False: stds is [N] (averaged across models)
            - If GPS mode with average_groups=True: stds is [N] averaged across groups
            - If GPS mode with average_groups=False: stds is [G, N] per-group stds
            - Otherwise: stds is [N] for standard TTA
            - per_fold_stds: [M, N] array of per-model uncertainties (only if ensemble_mode=True)
    """
    # Set seed for augmentation reproducibility/randomness
    # CRITICAL: Augmentation operations use random.random() internally!
    if seed is not None:
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if usingBetterRandAugment and transformations is not None and not isinstance(transformations, list):
        raise ValueError("Transformations must be a list (or None for random policies) when usingBetterRandAugment.")
    if usingBetterRandAugment and transformations is not None and not isinstance(transformations, list):
        raise ValueError("Transformations must be a list (or None for random policies) when usingBetterRandAugment.")
    
    # Determine number of augmentations
    if usingBetterRandAugment and transformations is not None:
        nb_augmentations = len(transformations)
    
    # Build cached dataset if requested
    cached_dataset = None
    if use_monai_cache:
        cached_dataset = build_monai_cache_dataset(dataset, cache_rate=cache_rate, num_workers=cache_num_workers)
    
    with torch.no_grad():
        predictions = []
        if is_gps_mode:
            # GPS mode: transformations is [[group1_policies], [group2_policies], [group3_policies]]
            all_groups_stds = []
            all_groups_model_stds = [] if ensemble_mode else None
            all_groups_model_mean_probs = [] if ensemble_mode else None  # [G, M, N, C]
            
            # Memory-efficient mode: process one policy at a time if group is large
            # Heuristic: if K policies * N samples * batch_size might exceed RAM, go sequential
            memory_efficient_threshold = 5  # Process sequentially if > 5 policies in a group (typical with top_k=3)
            
            for group_idx, policy_group in enumerate(transformations):
                print(f" Applying policy group {group_idx + 1}/{len(transformations)} ({len(policy_group)} policies)...")
                
                K = len(policy_group)
                use_sequential = K > memory_efficient_threshold
                
                if use_sequential:
                    print(f" Using memory-efficient mode (processing {K} policies one at a time)")
                    # Process one policy at a time to avoid RAM buildup
                    if ensemble_mode:
                        # Track predictions per model: [M] → list of [K, N, num_classes]
                        M = len(models) if isinstance(models, list) else 1
                        model_policy_preds = [[] for _ in range(M)]
                        
                        for policy_idx, single_policy in enumerate(policy_group):
                            if policy_idx % 10 == 0:
                                print(f" Policy {policy_idx+1}/{K}...")
                            
                            # Apply single policy: [1, N, C, H, W]
                            augmented_inputs, _ = apply_augmentations(
                                dataset, 1, usingBetterRandAugment, n, m, 
                                image_normalization, nb_channels, mean, std, image_size, 
                                [single_policy], batch_size=batch_size
                            )
                            
                            # Inference: [M, 1, N, num_classes]
                            policy_preds = _batched_augmentation_inference(
                                augmented_inputs, models, device, batch_size, ensemble_mode=True
                            )
                            
                            # Store per model: [1, N, num_classes]
                            for model_idx in range(M):
                                model_policy_preds[model_idx].append(policy_preds[model_idx, 0])  # [N, num_classes]
                        
                        # Stack all policies per model: [K, N, num_classes]
                        model_stds = []
                        for model_idx in range(M):
                            model_preds_stacked = torch.stack(model_policy_preds[model_idx], dim=0)  # [K, N, num_classes]
                            # Compute std across K policies
                            std_per_class = torch.std(model_preds_stacked, dim=0)  # [N, num_classes]
                            
                            if std_per_class.shape[1] == 1:
                                model_std = std_per_class.squeeze(1).cpu().numpy()
                            else:
                                model_std = torch.mean(std_per_class, dim=1).cpu().numpy()
                            
                            model_stds.append(model_std)
                        
                        all_groups_model_stds.append(np.array(model_stds))  # [M, N]
                        group_stds = np.mean(model_stds, axis=0)  # [N]
                        # Per-model mean softmax probs for this group: [M, N, C]
                        group_model_mean_probs = np.array([
                            torch.stack(model_policy_preds[mi], dim=0).mean(dim=0).cpu().numpy()
                            for mi in range(M)
                        ])
                        all_groups_model_mean_probs.append(group_model_mean_probs)  # [M, N, C]
                    else:
                        # Ensemble averaging: track [K, N, num_classes]
                        all_policy_preds = []
                        
                        for policy_idx, single_policy in enumerate(policy_group):
                            if policy_idx % 10 == 0:
                                print(f" Policy {policy_idx+1}/{K}...")
                            
                            augmented_inputs, _ = apply_augmentations(
                                dataset, 1, usingBetterRandAugment, n, m, 
                                image_normalization, nb_channels, mean, std, image_size, 
                                [single_policy], batch_size=batch_size
                            )
                            
                            # Inference: [1, N, num_classes]
                            policy_preds = _batched_augmentation_inference(
                                augmented_inputs, models, device, batch_size, ensemble_mode=False
                            )
                            all_policy_preds.append(policy_preds[0])  # [N, num_classes]
                        
                        # Stack and compute std: [K, N, num_classes]
                        all_preds_stacked = torch.stack(all_policy_preds, dim=0)
                        std_per_class = torch.std(all_preds_stacked, dim=0)  # [N, num_classes]
                        
                        if std_per_class.shape[1] == 1:
                            group_stds = std_per_class.squeeze(1).cpu().numpy()
                        else:
                            group_stds = torch.mean(std_per_class, dim=1).cpu().numpy()
                else:
                    # Standard batched mode (fast but uses more RAM)
                    # Get augmented inputs: [K, N, C, H, W]
                    augmented_inputs, _ = apply_augmentations(
                        dataset, len(policy_group), usingBetterRandAugment, n, m, 
                        image_normalization, nb_channels, mean, std, image_size, 
                        policy_group, batch_size=batch_size
                    )
                    
                    # Use batched inference
                    predictions = _batched_augmentation_inference(
                        augmented_inputs, models, device, batch_size, ensemble_mode=ensemble_mode
                    )  # [K, N, num_classes] or [M, K, N, num_classes] if ensemble_mode
                    
                    if ensemble_mode:
                        # Compute std per-model, then average across models
                        # predictions: [M, K, N, num_classes]
                        M, K, N, num_classes = predictions.shape
                        
                        model_stds = []
                        for model_idx in range(M):
                            # Std across K augmentations for this model
                            std_per_class = torch.std(predictions[model_idx], dim=0)  # [N, num_classes]
                            
                            if num_classes == 1:
                                model_std = std_per_class.squeeze(1)
                            else:
                                model_std = torch.mean(std_per_class, dim=1)  # [N]
                            
                            model_stds.append(model_std.cpu().numpy())
                        
                        # Store per-model stds for this group
                        all_groups_model_stds.append(np.array(model_stds))  # [M, N]
                        # Per-model mean softmax probs for this group: [M, N, C]
                        group_model_mean_probs = np.array([
                            predictions[mi].mean(dim=0).cpu().numpy()
                            for mi in range(M)
                        ])
                        all_groups_model_mean_probs.append(group_model_mean_probs)  # [M, N, C]
                        
                        # Average across models for group-level uncertainty
                        group_stds = np.mean(model_stds, axis=0)  # [N]
                    else:
                        # Standard: compute std across K augmentations (models already averaged)
                        std_per_class = torch.std(predictions, dim=0)  # [N, num_classes]
                        
                        if std_per_class.shape[1] == 1:
                            group_stds = std_per_class.squeeze(1).cpu().numpy()
                        else:
                            group_stds = torch.mean(std_per_class, dim=1).cpu().numpy()
                
                all_groups_stds.append(group_stds)
            
            # Average across groups if requested
            if average_groups:
                stds = np.mean(all_groups_stds, axis=0)  # [N]
            else:
                stds = all_groups_stds  # [G, N] - return per-group stds
            
            averaged_predictions = None # not computed in GPS mode
            
            # Handle per-fold mode for GPS
            if ensemble_mode and return_per_fold:
                # Average per-model stds across groups: [G, M, N] → [M, N]
                per_fold_stds = np.mean(all_groups_model_stds, axis=0)  # [M, N]
                # Average per-model mean probs across groups: [G, M, N, C] → [M, N, C] → argmax [M, N]
                mean_probs_across_groups = np.mean(all_groups_model_mean_probs, axis=0)  # [M, N, C]
                per_fold_mean_predictions = np.argmax(mean_probs_across_groups, axis=2)  # [M, N]
            else:
                per_fold_stds = None
                per_fold_mean_predictions = None
        else:
            # Single policy or standard TTA: process each augmentation one by one
            if ensemble_mode:
                # Ensemble mode: store per-model predictions
                if not isinstance(models, list):
                    models = [models]
                M = len(models)
                
                model_predictions = [[] for _ in range(M)]
                
                for aug_idx in range(nb_augmentations):
                    augmented_inputs = apply_augmentations(
                        dataset, 1, usingBetterRandAugment, n, m, image_normalization, nb_channels, mean, std, image_size, transformations, batch_size=batch_size, cached_dataset=cached_dataset, dataloader_workers=dataloader_workers
                    )
                    # augmented_inputs shape: [1, batch_size, C, H, W]
                    dataset_aug = TensorDataset(augmented_inputs[0])
                    loader = DataLoader(dataset_aug, batch_size=batch_size, pin_memory=True)
                    all_preds = []
                    for batch in loader:
                        batch_predictions = get_batch_predictions(models, batch[0], device)  # [B, M, C]
                        all_preds.append(batch_predictions)
                    
                    aug_preds = torch.cat(all_preds, dim=0)  # [N, M, num_classes]
                    
                    # Split by model
                    for model_idx in range(M):
                        model_predictions[model_idx].append(aug_preds[:, model_idx, :])  # [N, num_classes]
                
                # Compute std per model, then average
                model_stds = []
                for model_idx in range(M):
                    # Stack augmentations for this model: [nb_augmentations, N, num_classes]
                    model_aug_preds = torch.stack(model_predictions[model_idx], dim=0)
                    # Permute to [N, nb_augmentations, num_classes]
                    model_aug_preds = model_aug_preds.permute(1, 0, 2)
                    # Compute std for this model
                    model_std = compute_stds(model_aug_preds)  # Returns numpy array
                    model_stds.append(model_std)
                
                # Convert to numpy array: [M, N]
                model_stds_array = np.array(model_stds)
                
                # Per-fold mean predictions: argmax of mean softmax over augmentations per fold
                model_mean_preds_accum = []
                for model_idx in range(M):
                    model_aug_preds_stacked = torch.stack(model_predictions[model_idx], dim=0)  # [nb_aug, N, C]
                    mean_probs = model_aug_preds_stacked.mean(dim=0)  # [N, C]
                    model_mean_preds_accum.append(torch.argmax(mean_probs, dim=1).cpu().numpy())  # [N]
                per_fold_mean_predictions = np.array(model_mean_preds_accum)  # [M, N]
                
                # Return per-fold or averaged
                if return_per_fold:
                    stds = model_stds_array  # [M, N]
                else:
                    stds = np.mean(model_stds_array, axis=0)  # [N]
                
                per_fold_stds = model_stds_array  # Always keep for reference
                averaged_predictions = None  # Not needed in ensemble mode
            else:
                # Standard mode: average models first
                for aug_idx in range(nb_augmentations):
                    augmented_inputs = apply_augmentations(
                        dataset, 1, usingBetterRandAugment, n, m, image_normalization, nb_channels, mean, std, image_size, transformations, batch_size=batch_size, cached_dataset=cached_dataset, dataloader_workers=dataloader_workers
                    )
                    # augmented_inputs shape: [1, batch_size, C, H, W]
                    dataset_aug = TensorDataset(augmented_inputs[0])
                    loader = DataLoader(dataset_aug, batch_size=batch_size, pin_memory=True)
                    all_preds = []
                    for batch in loader:
                        batch_predictions = get_batch_predictions(models, batch[0], device)
                        avg_preds = average_predictions(batch_predictions)
                        all_preds.append(avg_preds)
                    predictions.append(torch.cat(all_preds, dim=0))
                # Stack predictions: [nb_augmentations, batch_size, num_classes]
                averaged_predictions = torch.stack(predictions, dim=0).permute(1, 0, 2)  # [batch_size, nb_augmentations, num_classes]
                stds = compute_stds(averaged_predictions)
                per_fold_stds = None  # No per-fold data in standard mode
                per_fold_mean_predictions = None
    
    # Ensure stds is always a numpy array (compute_stds returns list)
    if isinstance(stds, list):
        stds = np.array(stds)
    
    return stds, averaged_predictions, per_fold_stds, per_fold_mean_predictions


def apply_augmentations(dataset, nb_augmentations, usingBetterRandAugment, n, m, image_normalization, nb_channels, mean, std, image_size, transformations=None, batch_size=None, cached_dataset=None, dataloader_workers=None):
    """
    Apply augmentations to the images.

    Args:
        images (torch.Tensor): Batch of images.
        transformations (callable or list): Transformations to apply to each image.
        nb_augmentations (int): Number of augmentations to apply per image.
        usingBetterRandAugment (bool): If True, use BetterRandAugment with provided policies.
        n (int): Number of augmentation transformations to apply when using BetterRandAugment.
        m (int): Magnitude of the augmentation transformations when using BetterRandAugment.
        batch_norm (bool): Whether to use batch normalization.
        nb_channels (int): Number of channels in the input images.
        mean (list or None): Mean for normalization.
        std (list or None): Standard deviation for normalization.
        image_size (int): Size of the input images.
        cached_dataset (CacheDataset or None): Pre-cached dataset to speed up augmentation.
        dataloader_workers (int or None): Number of workers for the augmentation DataLoader.

    Returns:
        torch.Tensor: Augmented images.
    """
    augmented_inputs = []
    worker_count = 4 if dataloader_workers is None else dataloader_workers
    
    def _get_loader(augmentation):
        if cached_dataset is not None:
            aug_dataset = _CachedRandAugDataset(cached_dataset, augmentation)
            return DataLoader(
                dataset=aug_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=worker_count,
                pin_memory=True,
            )
        
        # Save original transform before modifying
        saved_transforms = None
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'datasets'):
            saved_transforms = [subds.transform for subds in dataset.dataset.datasets]
            for subds in dataset.dataset.datasets:
                subds.transform = augmentation
        else:
            saved_transforms = dataset.transform
            dataset.transform = augmentation
        
        # Create loader and consume all data
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=True,
        )
        
        # Store saved transforms to restore later
        loader._saved_transforms = saved_transforms
        loader._target_dataset = dataset
        
        return loader
    
    if usingBetterRandAugment:
        if isinstance(transformations, list):
            # Case 1: Explicit policies provided
            rand_aug_policies = [BetterRandAugment(n=n, m=m, resample=False, transform=policy, verbose=False, randomize_sign=False, image_size=image_size) for policy in transformations]
        
        # Case 2: Random policies (transformations is None or False)
        else:
            rand_aug_policies = [BetterRandAugment(n, m, True, False, randomize_sign=False, image_size=image_size) for _ in range(nb_augmentations)] 

        # Ensure input is PIL, run randaugment (expects PIL), then convert to tensor and normalize.
        augmentations = [transforms.Compose([
                    EnsurePIL(),                                 # convert tensor/ndarray -> PIL if needed
                    transforms.Lambda(lambda img: img.convert("RGB")),  # ensure RGB for randaug
                    rand_aug,                                            # BetterRandAugment expects PIL Image
                    *([to_1_channel] if nb_channels == 1 else []),       # convert back to single channel if needed
                    transforms.PILToTensor(),                            # PIL -> Tensor (uint8 -> [0,255] -> Tensor)
                    transforms.ConvertImageDtype(torch.float),          # ensure float Tensor
                    *([transforms.Normalize(mean=mean, std=std)] if image_normalization else [])
                ]) for rand_aug in rand_aug_policies]
        

        for i, augmentation in enumerate(augmentations):
            augmented_inputs_batch = []
            print(f" Loading augmentation {i+1}/{len(augmentations)}...", end='', flush=True)
            data_loader = _get_loader(augmentation)
            for batch in data_loader:
                augmented_images = batch[0]
                augmented_inputs_batch.append(augmented_images)
            augmented_inputs.append(torch.cat(augmented_inputs_batch, dim=0))
            
            # Restore original transform after consuming data
            if hasattr(data_loader, '_saved_transforms') and data_loader._saved_transforms is not None:
                target_dataset = data_loader._target_dataset
                if hasattr(target_dataset, 'dataset') and hasattr(target_dataset.dataset, 'datasets') and isinstance(data_loader._saved_transforms, list):
                    for subds, saved_transform in zip(target_dataset.dataset.datasets, data_loader._saved_transforms):
                        subds.transform = saved_transform
                else:
                    target_dataset.transform = data_loader._saved_transforms
            
            print(" done")
        augmented_inputs = torch.stack(augmented_inputs, dim=0)  # Shape: [ num_augmentations, batch_size, C, H, W]
    
    else:
        # Standard TTA with torchvision RandAugment
        # Build complete augmentation pipeline that outputs tensors
        base_augmentations = [
            transforms.Compose([
                EnsurePIL(),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.RandAugment(num_ops=n, magnitude=m),  # torchvision RandAugment
                *([to_1_channel] if nb_channels == 1 else []),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                *([transforms.Normalize(mean=mean, std=std)] if image_normalization else [])
            ]) for _ in range(nb_augmentations)
        ]
        for i, augmentation in enumerate(base_augmentations):
            augmented_inputs_batch = []
            print(f"Applying augmentation {i+1}/{nb_augmentations}")
            data_loader = _get_loader(augmentation)
            for batch in data_loader:
                augmented_images = batch[0]
                augmented_inputs_batch.append(augmented_images)
            augmented_inputs.append(torch.cat(augmented_inputs_batch, dim=0))
            
            # Restore original transform after consuming data
            if hasattr(data_loader, '_saved_transforms') and data_loader._saved_transforms is not None:
                target_dataset = data_loader._target_dataset
                if hasattr(target_dataset, 'dataset') and hasattr(target_dataset.dataset, 'datasets') and isinstance(data_loader._saved_transforms, list):
                    for subds, saved_transform in zip(target_dataset.dataset.datasets, data_loader._saved_transforms):
                        subds.transform = saved_transform
                else:
                    target_dataset.transform = data_loader._saved_transforms
        augmented_inputs = torch.stack(augmented_inputs, dim=0)  # Shape: [num_augmentations, batch_size, C, H, W]
    
    if usingBetterRandAugment : 
        return augmented_inputs, augmentations
    else:
        return augmented_inputs


def apply_randaugment_and_store_results(
    dataset, models, N, M, num_policies, device, folder_name='savedpolicies',
    image_normalization=False, mean=False, std=False, nb_channels=1, image_size=51,
    batch_size=None, use_monai_cache=False, cache_rate=1.0, cache_num_workers=0,
    dataloader_workers=None, dataloader_prefetch=8
):
    """
    Apply random augmentation policies and save predictions to disk (for GPS pre-computation).
    
    Generates `num_policies` random augmentation policies, applies each to the dataset,
    computes model predictions, and saves to .npz files for later greedy policy search.
    
    Args:
        dataset: PyTorch dataset
        models: Single model or list of models
        N: Number of ops per augmentation (BetterRandAugment)
        M: Magnitude parameter (BetterRandAugment)
        num_policies: Number of random policies to generate
        device: torch.device
        folder_name: Output directory for .npz files
        image_normalization: Whether to normalize images
        mean: Normalization mean
        std: Normalization std
        nb_channels: Number of channels (1 or 3)
        image_size: Input image size
        batch_size: Batch size for DataLoader
        use_monai_cache: Whether to use MONAI CacheDataset for faster loading
        cache_rate: Fraction of dataset to cache (0-1)
        cache_num_workers: Workers for cache building
        dataloader_workers: Workers for DataLoader
        dataloader_prefetch: Prefetch factor for DataLoader
    
    Returns:
        None (saves .npz files to disk)
    
    Example:
        >>> apply_randaugment_and_store_results(
        ...     calibration_dataset, models, N=2, M=45, num_policies=500,
        ...     device=device, folder_name='gps_augment/breastmnist_calib',
        ...     use_monai_cache=True, batch_size=128
        ... )
    """
    os.makedirs(folder_name, exist_ok=True)
    cached_dataset = None
    
    if use_monai_cache:
        cached_dataset = build_monai_cache_dataset(
            dataset, cache_rate=cache_rate, num_workers=cache_num_workers
        )

    # Streaming mode with cached dataset (memory-efficient)
    if cached_dataset is not None:
        print(f"Streaming {num_policies} policies in chunks to avoid RAM overload")
        policy_chunk_size = min(15, num_policies)  # Conservative chunk size for memory safety
        num_samples = len(cached_dataset)

        def build_augmentations_chunk(k_chunk):
            """Build k_chunk random BetterRandAugment policies."""
            rand_aug_policies = [
                BetterRandAugment(N, M, True, False, randomize_sign=False, image_size=image_size)
                for _ in range(k_chunk)
            ]
            return [
                transforms.Compose([
                    EnsurePIL(),
                    transforms.Lambda(lambda img: img.convert("RGB")),
                    rand_aug,
                    *([to_1_channel] if nb_channels == 1 else []),
                    transforms.ToTensor(),
                    *([transforms.Normalize(mean=mean, std=std)] if image_normalization else [])
                ]) for rand_aug in rand_aug_policies
            ]

        # Infer num_classes once
        with torch.no_grad():
            dummy = torch.zeros(1, nb_channels, image_size, image_size, device=device)
            try:
                dp = get_batch_predictions(models, dummy, device)
                nc = average_predictions(dp).shape[1]
            except Exception:
                nc = 1

        # Process policies in chunks
        for start in range(0, num_policies, policy_chunk_size):
            end = min(start + policy_chunk_size, num_policies)
            K_chunk = end - start
            print(f"Processing policy chunk {start}-{end-1} (K={K_chunk})")
            
            # Build augmentations and DataLoader
            augmentations = build_augmentations_chunk(K_chunk)
            worker_count = dataloader_workers or 0
            aug_dataset = _CachedRandAugDataset(cached_dataset, augmentations)
            
            # Conservative prefetch to avoid OOM - with K policies, memory multiplies fast
            # prefetch_factor * num_workers * batch_size * K_chunk can easily exceed GPU memory
            safe_prefetch = min(2, dataloader_prefetch) if worker_count > 0 else None
            
            loader = DataLoader(
                dataset=aug_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=worker_count,
                pin_memory=True,  # Disable to save GPU memory
                worker_init_fn=_dl_worker_init,
                persistent_workers=(worker_count > 0),  # Keep workers alive for speed
                prefetch_factor=safe_prefetch,
            )

            # Get first batch to determine actual K
            it = iter(loader)
            try:
                first_batch = next(it)
            except StopIteration:
                raise RuntimeError("Augmentation loader yielded no data")
            
            imgs_stacked = first_batch[0]  # [B, K, C, H, W]
            B0, K_actual, C, H, W = imgs_stacked.shape
            
            if K_actual != K_chunk:
                print(f"Warning: K_actual ({K_actual}) != K_chunk ({K_chunk}); using K_actual")

            # Create memory-mapped files for each policy
            memmaps, memmap_paths = [], []
            for idx in range(K_actual):
                mmap_path = os.path.join(folder_name, f"tmp_policy_{start+idx}.mmap")
                # Pre-create the file with correct size
                with open(mmap_path, 'wb') as f:
                    f.write(b'\0' * (num_samples * nc * 4))  # 4 bytes per float32
                mem = np.memmap(mmap_path, dtype='float32', mode='r+', shape=(num_samples, nc))
                memmaps.append(mem)
                memmap_paths.append(mmap_path)

            # Stream batches and fill memmaps
            sample_ptr = 0
            
            def _process_batch(imgs_stacked_local):
                """Process a batch and return predictions."""
                B, K, C, H, W = imgs_stacked_local.shape
                imgs_flat = imgs_stacked_local.reshape(B * K, C, H, W).float().to(device)
                batch_predictions = get_batch_predictions(models, imgs_flat, device)
                avg_preds = average_predictions(batch_predictions).view(B, K, -1).cpu().numpy()
                # Explicitly free GPU tensors
                del imgs_flat, batch_predictions
                return B, K, avg_preds

            # Process first batch
            B, K, avg_preds = _process_batch(imgs_stacked)
            for local_k in range(K):
                memmaps[local_k][sample_ptr:sample_ptr + B, :] = avg_preds[:, local_k, :]
            sample_ptr += B
            del imgs_stacked, avg_preds  # Free after use
            
            # Process remaining batches
            for batch in it:
                imgs_stacked = batch[0]
                B, K, avg_preds = _process_batch(imgs_stacked)
                for local_k in range(K):
                    memmaps[local_k][sample_ptr:sample_ptr + B, :] = avg_preds[:, local_k, :]
                sample_ptr += B
                del imgs_stacked, avg_preds  # Free after each batch

            if sample_ptr != num_samples:
                raise RuntimeError(f"Expected {num_samples} samples but wrote {sample_ptr}")

            # Cleanup memmaps and free memory
            for local_k, (mem, mempath) in enumerate(zip(memmaps, memmap_paths)):
                # Extract policy key for filename
                rand_transform = None
                try:
                    for t in augmentations[local_k].transforms:
                        if isinstance(t, BetterRandAugment):
                            rand_transform = t
                            break
                except Exception:
                    pass

                if rand_transform is not None and hasattr(rand_transform, 'get_transform_str'):
                    policy_key = rand_transform.get_transform_str()
                else:
                    policy_key = f"policy_{start + local_k}"

                safe_key = re.sub(r'[^A-Za-z0-9_.-]', '_', policy_key)
                out_fname = os.path.join(folder_name, f'N{N}_M{M}_{safe_key}.npz')
                
                arr = np.asarray(mem)
                np.savez_compressed(out_fname, predictions=arr)
                print(f"Saved policy {start + local_k} -> {out_fname}")
                del arr, mem  # Free memory immediately after saving

            # Cleanup memmap files
            try:
                for mempath in memmap_paths:
                    os.remove(mempath)
            except Exception as e:
                print(f"Warning: Failed to remove memmap files: {e}")
            
            # Shutdown persistent workers explicitly
            if hasattr(loader, '_iterator'):
                loader._iterator = None
            
            # Aggressive cleanup to free GPU memory
            del memmaps, memmap_paths, augmentations, loader, aug_dataset
            
            # Clear GPU cache and synchronize to ensure memory is freed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete

    else:
        # Fallback: no cache (process one policy at a time)
        print("Processing policies sequentially (no cache)")
        for i in range(num_policies):
            print(f"Applying policy {i+1}/{num_policies}")
            
            augmented_inputs, augmentations = apply_augmentations(
                dataset, 1, True, N, M, image_normalization, nb_channels,
                mean, std, image_size, batch_size=batch_size
            )
            
            augmented_input = augmented_inputs[0]  # [N, C, H, W]
            dataset_aug = TensorDataset(augmented_input)
            loader = DataLoader(dataset_aug, batch_size=batch_size, pin_memory=True)
            
            all_preds = []
            for batch in loader:
                batch_predictions = get_batch_predictions(models, batch[0], device)
                avg_preds = average_predictions(batch_predictions)
                all_preds.append(avg_preds)
            
            averaged_predictions = torch.cat(all_preds, dim=0)
            
            # Extract policy key
            rand_transform = None
            for t in augmentations[0].transforms:
                if isinstance(t, BetterRandAugment):
                    rand_transform = t
                    break

            if rand_transform is not None and hasattr(rand_transform, 'get_transform_str'):
                policy_key = rand_transform.get_transform_str()
            else:
                policy_key = f"policy_{i}"

            safe_key = re.sub(r'[^A-Za-z0-9_.-]', '_', policy_key)
            filename = os.path.join(folder_name, f'N{N}_M{M}_{safe_key}.npz')
            
            np.savez_compressed(filename, predictions=averaged_predictions.numpy())
            print(f"Saved predictions to {filename}")