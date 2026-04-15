"""
Ensemble-based uncertainty quantification methods.
Compute uncertainty from prediction variance across multiple models.
"""
import numpy as np
import torch

from ..core.base import UQMethod
from ..core.utils import evaluate_models_on_loader, get_prediction


# ============================================================================
# CLASS-BASED API (new, recommended)
# ============================================================================

class EnsembleSTDMethod(UQMethod):
    """
    Uncertainty quantification via standard deviation across ensemble models.
    
    Higher std indicates higher disagreement between models → higher uncertainty.
    """
    def __init__(self):
        super().__init__("Ensemble-STD")
    
    def compute(self, models, data_loader, device):
        """
        Compute per-sample uncertainty as std across ensemble predictions.
        
        Args:
            models: List of PyTorch models
            data_loader: DataLoader for evaluation
            device: torch.device
        
        Returns:
            np.ndarray: Per-sample uncertainty scores (N,)
        """
        _, _, _, correct_idx, incorrect_idx, indiv_scores = evaluate_models_on_loader(
            models, data_loader, device
        )
        stds = ensembling_stds_computation(indiv_scores)
        return stds


class MCDropoutMethod(UQMethod):
    """
    Monte Carlo Dropout for uncertainty quantification.
    
    Performs multiple stochastic forward passes with dropout enabled at test time.
    Uncertainty is measured as standard deviation across predictions.
    
    Requirements:
    - Models must have dropout layers
    - Dropout is enabled during inference (not eval mode)
    
    Args:
        num_samples: Number of MC dropout samples per model (default: 5)
    """
    def __init__(self, num_samples=5):
        super().__init__("MC-Dropout")
        self.num_samples = num_samples
    
    def _has_dropout(self, model):
        """Check if model has any dropout layers."""
        for module in model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                return True
        return False
    
    def _enable_dropout(self, model):
        """Enable dropout layers for MC sampling."""
        for module in model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                module.train()  # Enable dropout
    
    def compute(self, models, data_loader, device, ensemble_mode=False, return_per_fold=False):
        """
        Compute MC Dropout uncertainty.
        
        Args:
            models: List of PyTorch models or single model
            data_loader: DataLoader for evaluation
            device: torch.device
            ensemble_mode: If True, compute per-model uncertainty then average (for per-fold evaluation)
            return_per_fold: If True, return per-fold uncertainties [M, N]
        
        Returns:
            np.ndarray: Per-sample uncertainty scores
                - If return_per_fold=True: tuple ([M, N] per-model uncertainties, [M, N] per-fold mean predictions)
                - Otherwise: [N] averaged uncertainties
        """
        if not isinstance(models, list):
            models = [models]
        
        # Check if models have dropout
        models_with_dropout = [self._has_dropout(m) for m in models]
        if not any(models_with_dropout):
            raise ValueError("No dropout layers found in any model. MC Dropout requires models with dropout layers.")
        
        print(f" Models with dropout: {sum(models_with_dropout)}/{len(models)}")
        print(f" Performing {self.num_samples} MC samples per model...")
        
        all_model_uncertainties = []
        per_fold_mean_pred_list = []  # [M, N] argmax of mean MC probs per fold
        
        for model_idx, model in enumerate(models):
            if not models_with_dropout[model_idx]:
                print(f" Warning: Model {model_idx} has no dropout, skipping")
                continue
            
            # Enable dropout for this model
            self._enable_dropout(model)
            
            # Collect predictions from multiple forward passes
            mc_predictions = []
            
            for sample_idx in range(self.num_samples):
                sample_preds = []
                
                with torch.no_grad():
                    for batch in data_loader:
                        if isinstance(batch, dict):
                            images = batch["image"].to(device)
                        else:
                            images, _ = batch
                            images = images.to(device)
                        
                        logits = model(images)
                        
                        # Convert to probabilities
                        if logits.shape[1] == 1:
                            probs = torch.sigmoid(logits)
                            probs = torch.cat([1 - probs, probs], dim=1)
                        else:
                            probs = torch.softmax(logits, dim=1)
                        
                        sample_preds.append(probs.cpu().numpy())
                
                mc_predictions.append(np.concatenate(sample_preds, axis=0))  # [N, C]
            
            # Stack MC samples: [num_samples, N, C]
            mc_predictions = np.stack(mc_predictions, axis=0)
            
            # Mean prediction for this fold: argmax of mean softmax across MC samples
            mean_probs = np.mean(mc_predictions, axis=0)  # [N, C]
            per_fold_mean_pred_list.append(np.argmax(mean_probs, axis=1))  # [N]

            # Compute std across MC samples
            std_per_class = np.std(mc_predictions, axis=0)  # [N, C]
            model_uncertainty = np.mean(std_per_class, axis=1)  # [N]
            
            all_model_uncertainties.append(model_uncertainty)
            
            # Return model to eval mode (good practice, though MCDropout now runs last)
            model.eval()
        
        # Stack uncertainties: [M, N]
        all_model_uncertainties = np.array(all_model_uncertainties)
        per_fold_mean_preds_arr = np.array(per_fold_mean_pred_list)  # [M, N]
        
        if return_per_fold:
            return all_model_uncertainties, per_fold_mean_preds_arr  # [M, N], [M, N]
        else:
            # Average across models
            return np.mean(all_model_uncertainties, axis=0)  # [N]


# ============================================================================
# FUNCTIONAL API (existing, for backward compatibility)
# ============================================================================

def ensembling_stds_computation(individual_scores):
    """
    Compute standard deviation across ensemble predictions.
    
    Args:
        individual_scores: numpy array of shape [N, K, C] where:
            N = number of samples
            K = number of models
            C = number of classes
    
    Returns:
        list: Per-sample standard deviations (length N)
    
    Example:
        >>> # 100 samples, 5 models, 2 classes
        >>> scores = np.random.rand(100, 5, 2)
        >>> stds = ensembling_stds_computation(scores)
        >>> len(stds)  # 100
    """
    if not isinstance(individual_scores, np.ndarray):
        individual_scores = np.array(individual_scores)
    
    # Ensure shape is [N, K, C]
    if individual_scores.ndim != 3:
        raise ValueError(
            f"individual_scores must be 3D [N, K, C], got shape {individual_scores.shape}"
        )
    
    N, K, C = individual_scores.shape
    
    # Compute std across models (axis=1), then average across classes (axis=1 after reduction)
    stds_per_class = np.std(individual_scores, axis=1)  # [N, C]
    stds = np.mean(stds_per_class, axis=1)  # [N]
    
    return stds  # Return numpy array, not list


def ensembling_predictions(individual_scores):
    """
    Average predictions across ensemble models.
    
    Args:
        individual_scores: numpy array of shape [N, K, C]
    
    Returns:
        numpy.ndarray: Averaged predictions of shape [N, C]
    """
    if not isinstance(individual_scores, np.ndarray):
        individual_scores = np.array(individual_scores)
    
    # Average across models (axis=1)
    return np.mean(individual_scores, axis=1)  # [N, C]


def ensembling_variance_computation(individual_scores):
    """
    Compute variance across ensemble predictions.
    
    Args:
        individual_scores: numpy array of shape [N, K, C]
    
    Returns:
        list: Per-sample variances (length N)
    """
    if not isinstance(individual_scores, np.ndarray):
        individual_scores = np.array(individual_scores)
    
    # Compute variance across models (axis=1), then average across classes
    vars_per_class = np.var(individual_scores, axis=1)  # [N, C]
    vars = np.mean(vars_per_class, axis=1)  # [N]
    
    return vars.tolist()