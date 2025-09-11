import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, log_loss, brier_score_loss
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
import numpy as np
import pandas as pd
import seaborn as sns
import os
from concurrent.futures import ThreadPoolExecutor
import torch
import re
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from gps_augment.utils.randaugment import BetterRandAugment
import shap
import torch.multiprocessing as mp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler

class AddBatchDimension:
    def __call__(self, image):
        # Ensure the image is a tensor and add batch dimension
        if isinstance(image, torch.Tensor):
            return image.unsqueeze(0).float()
        raise TypeError("Input should be a torch Tensor")


def get_prediction(model, image, device):
    """
    Generates a prediction from a given model and image.

    Args:
        model (torch.nn.Module): The model used for prediction.
        image (torch.Tensor): The input image tensor.
        device (torch.device): The device to run the model on (e.g., 'cpu' or 'cuda').
        softmax_application (bool, optional): If True, applies softmax to the prediction. Defaults to False.

    Returns:
        torch.Tensor: The prediction output from the model.
    """
    model.to(device)
    image = image.to(device, non_blocking=True)
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():  # Disable gradient computation
        prediction = model(image).detach().cpu()

    return prediction


def extract_gps_augmentations_info(policies):
    """
    Extracts N, M values and the list of policies from a list of policy filenames.

    Args:
    - policies (list of str): List of filenames in the format 'N2_M45_[(op, magnitude), (op, magnitude)].npz'.

    Returns:
    - N (int): The value of N (same for all policies).
    - M (int): The value of M (same for all policies).
    - formatted_policies (list of str): List of policies as strings.
    """
    # Extract N and M from the first policy string
    if not policies:
        return None, None, []

    first_policy = policies[0]
    match = re.search(r'N(\d+)_M(\d+)', first_policy)
    if match:
        N = int(match.group(1))
        M = int(match.group(2))
    
    formatted_policies = []

    # Extract the policy part from each filename and keep it as a string
    for policy in policies:
        policy_match = re.search(r'\[(.*?)\]', policy)
        if policy_match:
            policy_str = f"[{policy_match.group(1)}]"  # Add brackets back around the tuple
            formatted_policies.append(policy_str)

    return N, M, formatted_policies


def TTA(transformations, models, dataset, device, nb_augmentations=10, usingBetterRandAugment=False, n=2, m=45, image_normalization=False, nb_channels=1, mean=None, std=None, image_size=51, output_activation=None, batch_size=None):
    """
    Perform Test-Time Augmentation (TTA) on a batch of images using specified transformations and models.

    Args:
        transformations (callable or list): Transformations to apply to each image. Must be a list when usingBetterRandAugment is True.
        models (torch.nn.Module or list): Model or list of models to use for predictions.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the images.
        device (torch.device): Device to run the models on (e.g., 'cpu' or 'cuda').
        nb_augmentations (int, optional): Number of augmentations to apply per image. Defaults to 10.
        usingBetterRandAugment (bool, optional): If True, use BetterRandAugment with provided policies. Defaults to False.
        n (int, optional): Number of augmentation transformations to apply when using BetterRandAugment. Defaults to 2.
        m (int, optional): Magnitude of the augmentation transformations when using BetterRandAugment. Defaults to 45.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
        nb_channels (int, optional): Number of channels in the input images. Defaults to 1.
        mean (list or None, optional): Mean for normalization. Defaults to None.
        std (list or None, optional): Standard deviation for normalization. Defaults to None.
        image_size (int, optional): Size of the input images. Defaults to 51.
        softmax_application (bool, optional): If True, applies softmax to the prediction. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - stds (list): List of standard deviations for each sample.
            - averaged_predictions (torch.Tensor): Averaged predictions for each sample.
    """
    if usingBetterRandAugment and not isinstance(transformations, list):
        raise ValueError("Transformations must be a list when usingBetterRandAugment.")
    if usingBetterRandAugment:
        nb_augmentations = len(transformations)
    with torch.no_grad():
        predictions = []
        if usingBetterRandAugment and isinstance(transformations, list) and all(isinstance(t, list) for t in transformations):
            # Multiple policies: process each policy one by one
            for transformation in transformations:
                # Apply augmentations for this policy, get DataLoader
                augmented_inputs, _ = apply_augmentations(
                    dataset, 1, usingBetterRandAugment, n, m, image_normalization, nb_channels, mean, std, image_size, transformation, batch_size=batch_size
                )
                # augmented_inputs shape: [1, batch_size, C, H, W]
                dataset_aug = TensorDataset(augmented_inputs[0])
                loader = DataLoader(dataset_aug, batch_size=batch_size, pin_memory=True)
                all_preds = []
                for batch in loader:
                    batch_predictions = get_batch_predictions(models, batch[0], device)
                    avg_preds = average_predictions(batch_predictions, output_activation)
                    all_preds.append(avg_preds)
                predictions.append(torch.cat(all_preds, dim=0))
            # Stack predictions: [num_policies, batch_size, num_classes]
            averaged_predictions = torch.stack(predictions, dim=0).permute(1, 0, 2)  # [batch_size, num_policies, num_classes]
            stds = compute_stds(averaged_predictions)
        else:
            # Single policy or standard TTA: process each augmentation one by one
            for aug_idx in range(nb_augmentations):
                # Apply augmentation for this index
                augmented_inputs = apply_augmentations(
                    dataset, 1, usingBetterRandAugment, n, m, image_normalization, nb_channels, mean, std, image_size, transformations, batch_size=batch_size
                )
                # augmented_inputs shape: [1, batch_size, C, H, W]
                dataset_aug = TensorDataset(augmented_inputs[0])
                loader = DataLoader(dataset_aug, batch_size=batch_size, pin_memory=True)
                all_preds = []
                for batch in loader:
                    batch_predictions = get_batch_predictions(models, batch[0], device)
                    avg_preds = average_predictions(batch_predictions, output_activation)
                    all_preds.append(avg_preds)
                predictions.append(torch.cat(all_preds, dim=0))
            # Stack predictions: [nb_augmentations, batch_size, num_classes]
            averaged_predictions = torch.stack(predictions, dim=0).permute(1, 0, 2)  # [batch_size, nb_augmentations, num_classes]
            stds = compute_stds(averaged_predictions)
    
    return stds, averaged_predictions


def apply_randaugment_and_store_results(
    dataset, models, N, M, num_policies, device, folder_name='savedpolicies',
    image_normalization=False, mean=False, std=False, nb_channels=1, image_size=51,
    output_activation=None, batch_size=None
):
    """
    Apply RandAugment transformations to the data and store the results, one augmentation at a time.
    """
    
    os.makedirs(folder_name, exist_ok=True)

    for i in range(num_policies):
    
        print(f"Applying augmentation policy {i+1}/{num_policies}")
        # Apply augmentation and get augmented images
        augmented_inputs, augmentations = apply_augmentations(
            dataset, 1, True, N, M, image_normalization, nb_channels, mean, std, image_size, batch_size=batch_size
        )
        # augmented_inputs shape: [1, batch_size, C, H, W]
        augmented_input = augmented_inputs[0]  # shape: [batch_size, C, H, W]
        dataset_aug = TensorDataset(augmented_input)
        loader = DataLoader(dataset_aug, batch_size=batch_size, pin_memory=True)
        all_preds = []
        for batch in loader:
            batch_predictions = get_batch_predictions(models, batch[0], device)
            averaged_predictions = [average_predictions(pred, output_activation) for pred in batch_predictions.permute(1, 0, 2)]
            all_preds.extend(averaged_predictions)
        averaged_predictions = torch.stack(all_preds)
        # Save predictions
        policy_key = str(augmentations[0].transforms[3].get_transform())
        filename = f'{folder_name}/N{N}_M{M}_{policy_key}.npz'
        np.savez_compressed(filename, predictions=averaged_predictions.numpy())


def apply_augmentations(dataset, nb_augmentations, usingBetterRandAugment, n, m, image_normalization, nb_channels, mean, std, image_size, transformations=False, batch_size=None):
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

    Returns:
        torch.Tensor: Augmented images.
    """
    augmented_inputs = []
    if usingBetterRandAugment:
        if isinstance(transformations, list):
            rand_aug_policies = [BetterRandAugment(n=n, m=m, resample=False, transform=policy, verbose=True, randomize_sign=False, image_size=image_size) for policy in transformations]
            
        elif transformations is False:
            rand_aug_policies = [BetterRandAugment(n, m, True, False, randomize_sign=False, image_size=image_size) for _ in range(nb_augmentations)] 

        augmentations = [transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(),
                    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure image is in RGB format
                    *([to_3_channels] if nb_channels == 1 else []),  # Conditionally add to_3_channels
                    rand_aug,
                    *([to_1_channel] if nb_channels == 1 else []),  # Conditionally add to_1_channel
                    transforms.PILToTensor(),
                    transforms.Lambda(lambda x: x.float()) if nb_channels == 1 else transforms.ConvertImageDtype(torch.float),
                    *([transforms.Normalize(mean=mean, std=std)] if image_normalization else [])
                ]) for rand_aug in rand_aug_policies]
        
        for i, augmentation in enumerate(augmentations):
            augmented_inputs_batch = []
            print(f"Applying augmentation n : {i}")
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'datasets'):
                for subds in dataset.dataset.datasets:
                    subds.transform = augmentation
            else:
                dataset.transform = augmentation
            
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=72, pin_memory=True)

            for batch in data_loader:
                augmented_images = batch[0]
                augmented_inputs_batch.append(augmented_images)
                
            augmented_inputs.append(torch.cat(augmented_inputs_batch, dim=0))
        augmented_inputs = torch.stack(augmented_inputs, dim=0)  # Shape: [ num_augmentations, batch_size, C, H, W]

    else:
        for i in range(nb_augmentations):
            augmented_inputs_batch = []
            print(f"Applying augmentation n : {i}")
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'datasets'):
                for subds in dataset.dataset.datasets:
                    subds.transform = transformations
            else:
                dataset.transform = transformations
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=72, pin_memory=True)

            for batch in data_loader:
                augmented_images = batch[0]
                augmented_inputs_batch.append(augmented_images)
            augmented_inputs.append(torch.cat(augmented_inputs_batch, dim=0))
        augmented_inputs = torch.stack(augmented_inputs, dim=0)  # Shape: [ num_augmentations, batch_size, C, H, W]
    
    if usingBetterRandAugment : 
        return augmented_inputs, augmentations
    else:
        return augmented_inputs


def get_batch_predictions(models, augmented_inputs, device):
    """
    Get predictions for the augmented inputs.

    Args:
        models (torch.nn.Module or list): Model or list of models to use for predictions.
        augmented_inputs (torch.Tensor): Augmented images.
        device (torch.device): Device to run the models on.
        softmax_application (bool): If True, applies softmax to the prediction.

    Returns:
        torch.Tensor: Batch predictions.
    """
    if isinstance(models, list):
        batch_predictions = []
        for model in models:
            prediction = get_prediction(model, augmented_inputs, device)
            batch_predictions.append(prediction)
    else:
        prediction = get_prediction(models, augmented_inputs, device)
        batch_predictions = [prediction]
    
    batch_predictions = torch.stack(batch_predictions, dim=0)  # Shape: [num_models, batch_size * num_augmentations, num_classes]
    return batch_predictions


def average_predictions(batch_predictions, output_activation=None):
    """
    Average predictions across models and group augmentations back with their respective images.

    Args:
        batch_predictions (torch.Tensor): Batch predictions. Shape: [num_models, batch_size * num_augmentations, num_classes].
        output_activation (str, optional): Activation function to apply to the predictions. 
                                           Options: 'softmax', 'sigmoid', or None. Defaults to None.

    Returns:
        torch.Tensor: Averaged predictions. Shape: [batch_size * num_augmentations, num_classes].
    """
    # Average predictions across models
    averaged_predictions = torch.mean(batch_predictions, dim=0)  # Shape: [batch_size * num_augmentations, num_classes]

    # Apply the specified activation function
    if output_activation == 'softmax':
        averaged_predictions = torch.nn.functional.softmax(averaged_predictions, dim=-1)
    elif output_activation == 'sigmoid':
        averaged_predictions = torch.sigmoid(averaged_predictions)

    return averaged_predictions


def compute_stds(averaged_predictions):
    """
    Compute standard deviations for the predictions.

    Args:
        averaged_predictions (torch.Tensor): Averaged predictions.

    Returns:
        list: List of standard deviations for each sample.
    """
    if averaged_predictions.ndim == 2 or averaged_predictions.shape[2] == 1:
        stds = torch.std(averaged_predictions, dim=1).squeeze().tolist()  # Binary classification: shape (num_models, num_samples)
    elif averaged_predictions.ndim == 3:
        stds_per_class = torch.std(averaged_predictions, dim=1).squeeze()  # Multiclass classification: shape (num_models, num_samples, num_classes)
        stds = torch.mean(stds_per_class, dim=1).tolist()
    return stds


def ensembling_predictions(models, image):
    ensembling_predictions = [get_prediction(model, image) for model in models]
    
    return ensembling_predictions
    
def distance_to_hard_labels_computation(predictions):
    """
    Compute the distance to hard labels for binary or multiclass predictions.

    Args:
    - predictions (numpy.ndarray): Array of predictions. Shape (num_samples,) for binary or (num_samples, num_classes) for multiclass.

    Returns:
    - list: List of distances to the hard labels.
    """
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        # Binary classification
        distances = 0.5 - np.abs(predictions - 0.5)
    else:
        # Multiclass classification
        distances = 1.0 - np.max(predictions, axis=1)

    return distances

def ensembling_stds_computation(models_predictions):
    """
    Compute the standard deviations of model predictions for ensembling.
    Parameters:
    models_predictions (array-like): An array of shape (num_models, num_samples) for binary classification
                                     or (num_models, num_samples, num_classes) for multiclass classification
                                     containing the predictions from different models.
    Returns:
    np.ndarray: An array of standard deviations for each sample in binary classification or
                an array of mean standard deviations across classes for each sample in multiclass classification.
    Raises:
    ValueError: If the shape of models_predictions array is not 2D or 3D.
    """
    models_predictions = np.asarray(models_predictions)
    
    if models_predictions.ndim == 2:
        # Binary classification: shape (num_models, num_samples)
        stds = np.std(models_predictions, axis=0)
    elif models_predictions.ndim == 3:
        # Multiclass classification: shape (num_models, num_samples, num_classes)
        class_wise_stds = np.std(models_predictions, axis=0)
        stds = np.mean(class_wise_stds, axis=1)
    else:
        raise ValueError("Unexpected shape of models_predictions array. Expected 2D or 3D array.")
    
    return stds
    
def plot_calibration_curve(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    
    return prob_true, prob_pred

def compute_class_weights(labels_np):
    classes, counts = np.unique(labels_np, return_counts=True)
    total = sum(counts)
    weights = total / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float)


class TemperatureScaler(nn.Module):
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor([init_temp])))

    def forward(self, logits):
        temperature = torch.exp(self.log_temperature)
        temperature = torch.clamp(temperature, min=0.5, max=5.0)
        return logits / temperature

def fit_temperature_scaling(logits, labels, max_iter=1000):
    logits = torch.from_numpy(logits).float()
    if logits.ndim == 1:
       logits = logits.unsqueeze(1)

    labels_np = labels.copy()
    labels = torch.from_numpy(labels).float()
    model = TemperatureScaler()        # uses model.log_temperature internally

    if logits.shape[1] == 1:
        # Binary classification
        class_weights = compute_class_weights(labels_np)
        # BCEWithLogitsLoss expects weights for each sample, so map from label
        sample_weights = torch.where(labels == 1, class_weights[1], class_weights[0])
        criterion = nn.BCEWithLogitsLoss(weight=sample_weights)
    else:
        labels = labels.long()
        class_weights = compute_class_weights(labels_np)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimize the log-temperature parameter
    optimizer = optim.LBFGS([model.log_temperature], lr=0.001, max_iter=max_iter)
    print(f"Initial log-temperature: {model.log_temperature.item():.4f}")
    print(f"Initial temperature: {torch.exp(model.log_temperature).item():.4f}")
    def closure():
        optimizer.zero_grad()
        loss = criterion(model(logits).squeeze(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    print(f"Optimized log-temperature: {model.log_temperature.item():.4f}")
    print(f"Optimized temperature: {torch.exp(model.log_temperature).item():.4f}")
    return model


def posthoc_calibration(y_scores, y_true, method_calibration='platt'):
    """
    Perform posthoc calibration using Platt scaling, isotonic regression, or temperature scaling.

    Args:
        y_scores (numpy.ndarray): Predicted probabilities (or logits if method_calibration='temperature').
        y_true (numpy.ndarray): True labels.
        method_calibration (str): Calibration method ('platt', 'isotonic', 'temperature').
        logits (numpy.ndarray): Raw logits (required for temperature scaling).
        multiclass (bool): Whether the problem is multiclass.

    Returns:
        calibrated_probs (numpy.ndarray): Calibrated probabilities.
        model: The calibration model used.
    """
    if method_calibration == 'temperature':
        model = fit_temperature_scaling(y_scores, y_true)
        logits_tensor = torch.from_numpy(y_scores).float()
        if logits_tensor.ndim == 1:
            logits_tensor = logits_tensor.unsqueeze(1)
        calibrated_logits = model(logits_tensor).detach()

        # Binary or multiclass?
        if calibrated_logits.ndim == 1 or calibrated_logits.shape[1] == 1:
            calibrated_probs = torch.sigmoid(calibrated_logits).numpy()
            y_prob_true = calibrated_probs.squeeze()
                  
        else:
            calibrated_probs = torch.softmax(calibrated_logits, dim=1).numpy()
            y_prob_true = calibrated_probs[np.arange(len(y_true)), y_true]

    elif method_calibration == 'platt':
        model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000)
        model.fit(y_scores.reshape(-1, 1), y_true)
        calibrated_probs = model.predict_proba(y_scores.reshape(-1, 1))[:, 1]
        y_prob_true = calibrated_probs
    elif method_calibration == 'isotonic':
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(y_scores, y_true)
        calibrated_probs = model.predict(y_scores)
        y_prob_true = calibrated_probs
    else:
        raise ValueError("Invalid method. Choose 'platt', 'isotonic', or 'temperature'.")

    brier = brier_score_loss((y_true == np.argmax(calibrated_probs, axis=1)) if y_scores.ndim > 1 else y_true, y_prob_true)
    print(f"Brier Score Loss ({method_calibration}): {brier:.4f}")

    return y_prob_true, model
        
def model_calibration_plot(true_labels, predictions, n_bins=20):
    """
    Plot a calibration curve (reliability diagram).
    
    - For binary classification: regular calibration curve.
    - For multiclass: top-1 reliability diagram (based on confidence of predicted class).

    Args:
        true_labels (np.ndarray): True class labels.
        predictions (np.ndarray): Probabilities from the model.
        n_bins (int): Number of bins for calibration.
    """

    if predictions.ndim == 1 or predictions.shape[1] == 1:
        # Binary case
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        prob_true, prob_pred = plot_calibration_curve(np.array(true_labels), np.array(predictions))
        plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration Curve')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve (Binary)')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        # Multiclass Top-1 reliability diagram
        y_true = np.array(true_labels)
        y_prob = np.array(predictions)

        # Top-1 predicted class and its confidence
        top1_preds = np.argmax(y_prob, axis=1)
        top1_confs = np.max(y_prob, axis=1)
        top1_correct = (top1_preds == y_true).astype(int)

        # Compute calibration curve for top-1 predictions
        prob_true, prob_pred = calibration_curve(top1_correct, top1_confs, n_bins=n_bins, strategy='uniform')

        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.plot(prob_pred, prob_true, marker='o', label='Top-1 Reliability')
        plt.xlabel('Confidence (Top-1 Prediction)')
        plt.ylabel('Accuracy (Fraction Correct)')
        plt.title('Top-1 Reliability Diagram (Multiclass)')
        plt.legend()
        plt.grid()
        plt.show()


def UQ_method_plot(correct_predictions, incorrect_predictions, y_title, title, flag, swarmplot=False):
    """
    Plot a boxplot (and optionally a swarmplot) for uncertainty quantification (UQ) methods.

    Args:
        correct_predictions (list): List of predictions for correct results.
        incorrect_predictions (list): List of predictions for incorrect results.
        y_title (str): Label for the y-axis.
        title (str): Title of the plot.
        swarmplot (bool, optional): If True, adds a swarmplot overlay. Defaults to True.

        sns.swarmplot(x='Category', y=y_title, data=df, color='black', alpha=0.3)
        None
    """
    df = pd.DataFrame({
        y_title: correct_predictions + incorrect_predictions,
        'Category': ['Success'] * len(correct_predictions) + ['Failures'] * len(incorrect_predictions)
        })
    
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(x='Category', y=y_title, data=df, palette='muted')
    if swarmplot:
        sns.swarmplot(x='Category', y=y_title, data=df, color='k', alpha=0.3)
    
    # Show the plot
    plt.title(title, fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel(y_title, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    plt.savefig(f"/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/medMNIST_UQ_results/medMNIST_augmented{flag}_{title}.png")  # or any filename you want
    plt.close()

def roc_curve_UQ_method_computation(correct_predictions, incorrect_predictions):
    """
    Compute the ROC curve and AUC score for uncertainty quantification methods.

    Args:
        correct_predictions (list): List of predictions for correct results.
        incorrect_predictions (list): List of predictions for incorrect results.

    Returns:
        tuple: A tuple containing:
            - fpr (numpy.ndarray): False positive rates.
            - tpr (numpy.ndarray): True positive rates.
            - auc_score (float): Area under the ROC curve.
    """
    failures_gstd = np.ones(len(incorrect_predictions))
    success_gstd = np.zeros(len(correct_predictions))
    
    # Concatenate arrays once
    gstd = np.concatenate((failures_gstd, success_gstd))
    predictions = np.concatenate((incorrect_predictions, correct_predictions))
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(gstd, predictions)

    # Calculate and print AUC
    auc_score = roc_auc_score(gstd, predictions)

    # Sign the AUC: ensure bad_idx (failures) have higher preds_std
    if np.mean(incorrect_predictions) < np.mean(correct_predictions):
        auc_score = -auc_score  # or 1 - auc_score, depending on your convention
    
    return fpr, tpr, auc_score

def roc_curve_UQ_methods_plot(method_names, fprs, tprs, auc_scores):
    """
    Plot the ROC curve for different UQ methods.

    Args:
        method_names (list): List of method names.
        fprs (list): List of false positive rates for each method.
        tprs (list): List of true positive rates for each method.
        auc_scores (list): List of AUC scores for each method.
    """
    # Plot the ROC curve
    plt.figure()
    for fpr, tpr, auc_score, method_name in zip(fprs, tprs, auc_scores, method_names):
        plt.plot(fpr, tpr, lw=2, label=f'AUC {method_name}: {auc_score:.2f}')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for UQ methods')
    plt.legend(loc="lower right")
    plt.show()

def standardize_and_ensemble(distributions, metric):
    """
    Standardizes any number of distributions using a global mean and standard deviation.
    Returns a single column array with the mean value for each instance across the standardized distributions.
    
    Parameters:
    distributions: 2D numpy array, where each column represents a different distribution (uncertainty method).
    
    Returns:
    mean_values: 1D numpy array containing the mean standardized value for each row.
    """
    
    # Compute global mean and standard deviation
    scaler = StandardScaler()
    standardized_distributions = scaler.fit_transform(distributions)
    if metric == 'mean':
        # Compute the mean standardized value for each instance (row)
        ensembled_values = np.mean(standardized_distributions, axis=1)
    elif metric == 'max':
        # Compute the mean standardized value for each instance (row)
        ensembled_values = np.max(standardized_distributions, axis=1)
    elif metric == 'min':
        # Compute the mean standardized value for each instance (row)
        ensembled_values = np.min(standardized_distributions, axis=1)
    
    return ensembled_values
    
def to_3_channels(img):
    if img.mode == 'L':  # Grayscale image
        img = img.convert('RGB')  # Convert to 3 channels by duplicating
    return img

def to_1_channel(img):
    img = img.convert('L')  # Convert back to grayscale
    return img


def greedy_search(initial_aug_idx, val_preds, good_idx, bad_idx, select_only, min_improvement=0.005, patience=5):
    """
    A single greedy search instance that starts from a random initial augmentation (initial_aug_idx).
    Returns the best augmentations based on the maximum ROC AUC achieved.
    """
    group_indices = [initial_aug_idx]  # Initialize with the given augmentation
    best_metric = -np.inf
    best_group_indices = list(group_indices)  # Track the augmentations that give the best ROC AUC
    all_aucs = []
    all_roc_aucs = []  # Store the AUCs for plotting
    no_improvement_count = 0  # Track consecutive iterations with insufficient improvement
    for new_member_i in range(select_only):
        print(f"Evaluating policy {new_member_i+1}/{select_only}...", flush=True)
        best_iteration_metric = -np.inf
        best_s = None

        # Evaluate standard deviation of augmentations for success vs failure classification
        for new_i in range(val_preds.shape[0]):
            if new_i in group_indices:
                continue

            current_augmentations = group_indices + [np.int64(new_i)]  # Add new augmentation to the selected group
            if val_preds[np.array(current_augmentations), :, :].shape[2] == 1:
                # Binary classification: shape (num_models, num_samples)
                preds_std = np.std(val_preds[current_augmentations, :, :], axis=0)
            elif val_preds[np.array(current_augmentations), :, :].shape[2] != 1 or val_preds[np.array(current_augmentations), :, :].ndim == 3:
                # Multiclass classification: shape (num_models, num_samples, num_classes)
                stds_per_class = np.std(val_preds[current_augmentations, :, :], axis=0)
                preds_std = np.mean(stds_per_class, axis=1)

            # Compute ROC AUC for the current set of augmentations
            roc_auc = roc_curve_UQ_method_computation(
                [preds_std[k] for k in good_idx], 
                [preds_std[j] for j in bad_idx]
            )[2]
            all_aucs.append(roc_auc)  # Store the AUC for plotting
            if roc_auc > 0.5 and roc_auc > best_iteration_metric:
                best_s = new_i
                best_iteration_metric = roc_auc
        if best_s is None:
            print(f"No valid policy found for iteration {new_member_i + 1}. Stopping search.")
            break
        if len(all_roc_aucs) > 0:
            # Calculate improvement and check early stopping
            improvement = best_iteration_metric - all_roc_aucs[-1]
            if improvement > min_improvement:
                no_improvement_count = 0  # Reset the counter
            else:
                no_improvement_count += 1

        # Stop if there is no significant improvement for `patience` consecutive iterations
        if no_improvement_count >= patience:
            print(f"Early stopping at iteration {new_member_i + 1} due to no improvement > {min_improvement} in last {patience} iterations.")
            break
    
        # Track the best augmentations and metric so far
        if best_iteration_metric > best_metric:
            best_metric = best_iteration_metric
            best_group_indices = list(group_indices)  # Copy the best augmentations so far
        
        # Update group indices and store the AUC for the current iteration
        group_indices.append(best_s)
        all_roc_aucs.append(best_iteration_metric)
        print(f"Selected Policy {best_s}: roc_auc={best_iteration_metric:.4f}")

    # If only one policy was selected, try to add the next best policy
    if len(best_group_indices) == 1:
        print("Only one policy selected, searching for the next best policy to add...")
        best_second_metric = -np.inf
        best_second = None
        for new_i in range(val_preds.shape[0]):
            if new_i == best_group_indices[0]:
                continue
            current_augmentations = best_group_indices + [new_i]
            if val_preds[current_augmentations, :, :].ndim == 2:
                preds_std = np.std(val_preds[current_augmentations, :, :], axis=0)
            elif val_preds[current_augmentations, :, :].ndim == 3:
                stds_per_class = np.std(val_preds[current_augmentations, :, :], axis=0)
                preds_std = np.mean(stds_per_class, axis=1)
            roc_auc = roc_curve_UQ_method_computation(
                [preds_std[k] for k in good_idx], 
                [preds_std[j] for j in bad_idx]
            )[2]
            if roc_auc > 0.5 and roc_auc > best_second_metric:
                best_second = new_i
                best_second_metric = roc_auc
        if best_second is not None:
            best_group_indices.append(best_second)
            print(f"Added next best policy {best_second} with roc_auc={best_second_metric:.4f}")

    return best_metric, best_group_indices, all_roc_aucs
    
def plot_auc_curves(results):
    """
    Plot ROC AUC curves for all greedy searches over the iterations.
    
    Parameters:
    - all_roc_aucs: List of lists where each sublist contains the AUCs for a particular search.
    - num_searches: Total number of parallel greedy searches.
    """
    plt.figure(figsize=(10, 6))

    for idx, res in enumerate(results):
        plt.plot(res[2], label=f"Search {idx + 1}")

    plt.xlabel("Iterations")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC Progress Over Iterations for Each Greedy Search")
    plt.grid(True)
    plt.show()
    
def select_greedily_on_ens(
    all_preds, good_idx, bad_idx, keys, search_set_len, select_only=50,
    num_workers=1, num_searches=10, top_k=5, method='top_policies'
):
    val_preds = np.copy(all_preds[:, :search_set_len, :])
    with mp.Pool(processes=num_workers) as pool:
        initial_augmentations = [
            int(np.random.choice(range(val_preds.shape[0]))) for _ in range(num_searches)
        ]
        try:
            results = pool.starmap(
                greedy_search,
                [(initial_aug, val_preds, good_idx, bad_idx, select_only) for initial_aug in initial_augmentations]
            )
        except IndexError as e:
            print("Debugging IndexError...")
            print(f"val_preds shape: {val_preds.shape}")
            pool.close()
            pool.join()
            
        finally:
            pool.close()
            pool.join()

    # Select the best result based on the ROC AUC metric
    best_result = max(results, key=lambda x: x[0])
    best_metric, best_group_indices, _ = best_result

    print("\nParallel greedy search complete. Best metric:", best_metric)

    if method == 'top_k_policies':
        # Get the top_k results by ROC AUC
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        print(results)
        top_k_group_indices = []
        for i in range(top_k):
            top_k_group_indices.append(sorted_results[i][1])
        
        policies = top_k_group_indices
    else:
        policies = np.array(best_group_indices)

    return policies, results

def load_npz_files_for_greedy_search(npz_dir):
    """
    Load all .npz files from the specified directory.
    Return the predictions stacked for each policy and corresponding filenames.
    """
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    all_preds = []
    all_keys = []

    # Load the predictions from each .npz file
    for npz_file in npz_files:
        file_path = os.path.join(npz_dir, npz_file)
        try:
            data = np.load(file_path)
            preds = data['predictions']
            all_preds.append(preds)
            all_keys.append(npz_file)  # Use the filename (or another key) as the identifier for the policy
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")

    all_preds = np.array(all_preds)  # Shape: [num_policies, num_samples, num_classes]
    return all_preds, all_keys

def perform_greedy_policy_search(
    npz_dir, good_idx, bad_idx, max_iterations=50, num_workers=1, num_searches=10, top_k=5, plot=True, method='top_k_policies'
):
    print('Loading predictions...')
    all_preds, all_keys = load_npz_files_for_greedy_search(npz_dir)
    search_set_len = all_preds[0].size

    selected_policies, results = select_greedily_on_ens(
        all_preds, good_idx, bad_idx, all_keys,
        search_set_len=search_set_len,
        select_only=max_iterations,
        num_workers=num_workers,
        num_searches=num_searches,
        top_k=top_k,
        method=method
    )
    if isinstance(selected_policies, list) and all(isinstance(policy, list) for policy in selected_policies):
        selected_policy_names = [[all_keys[i] for i in selected_policy] for selected_policy in selected_policies]
    else:
        selected_policy_names = [all_keys[i] for i in selected_policies]

    if plot:
        plot_auc_curves(results)

    return selected_policy_names


def visualize_input_shap_overlayed_multimodel(
    models, eval_dataloader, device, success_indices, failure_indices, sample_size=5, max_background_samples=1000
):
    """
    Visualize SHAP values overlayed on the original image for input pixels across multiple models.

    Args:
        models (list): List of trained PyTorch models.
        eval_dataloader (DataLoader): DataLoader for evaluation data.
        device (str): Device to run computations ('cuda' or 'cpu').
        success_indices (list): Indices of success cases.
        failure_indices (list): Indices of failure cases.
        sample_size (int): Number of random success and failure cases to visualize.
        max_background_samples (int): Maximum number of background samples to use for SHAP computation. Defaults to 1000.
    """
    for model in models:
        model.eval()

    # Combine indices and select random samples
    np.random.seed(433)
    success_sample = np.random.choice(success_indices, sample_size, replace=False)
    failure_sample = np.random.choice(failure_indices, sample_size, replace=False)
    selected_indices = np.concatenate([success_sample, failure_sample])

    # Extract corresponding images
    images_to_explain = []
    labels_to_explain = []
    cases = []
    background_images = []
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            images = batch["image"].to(device)
            labels = batch["shape"]  # Adjust key for labels based on your dataset
            background_images.append(images)
            for idx, img, lbl in zip(range(len(images)), images, labels):
                global_idx = i * eval_dataloader.batch_size + idx
                if global_idx in selected_indices:
                    images_to_explain.append(img)
                    labels_to_explain.append(lbl.item())
                    cases.append(
                        "Success" if global_idx in success_sample else "Failure"
                    )

            if len(images_to_explain) >= len(selected_indices):
                break

    # Convert to Tensor and preprocess
    images_to_explain = torch.stack(images_to_explain).to(device)  # Shape: (sample_size, 1, H, W)
    labels_to_explain = np.array(labels_to_explain)
    background_images = torch.cat(background_images).to(device)  # Combine all background images

    # Limit the number of background samples
    if len(background_images) > max_background_samples:
        background_images = background_images[:max_background_samples]

    # Create subplots
    num_images = len(selected_indices)
    num_models = len(models)
    fig, axes = plt.subplots(
        num_images, num_models + 1, figsize=(4 * (num_models + 1), 4 * num_images)
    )

    # Iterate over cases
    for i in range(num_images):
        # Display original image in the first column
        original_image = images_to_explain[i].cpu().numpy().squeeze()
        axes[i, 0].imshow(original_image, cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Original Image (Case {i + 1})", fontsize=12)

        # Iterate over models
        for j, model in enumerate(models):
            # Use GradientExplainer to compute SHAP values
            explainer = shap.GradientExplainer(model, background_images)
            shap_values = explainer.shap_values(images_to_explain[i : i + 1])

            # Extract SHAP values and model prediction
            shap_value = shap_values[0].squeeze()  # Single image SHAP values
            prediction = model(images_to_explain[i : i + 1]).item()  # Sigmoid output

            # Overlay SHAP values on the original image
            axes[i, j + 1].imshow(original_image, cmap="gray")
            axes[i, j + 1].imshow(
                shap_value,
                cmap="coolwarm",
                alpha=0.6,
                interpolation="nearest",
                extent=(0, original_image.shape[1], original_image.shape[0], 0),
            )
            axes[i, j + 1].axis("off")

            # Set title with label, prediction, and case type
            label_text = "Round" if labels_to_explain[i] == 0 else "Irregular"
            axes[i, j + 1].set_title(
                f"Model {j} (Case {i + 1}): {label_text}\nPrediction: {prediction:.2f} ({cases[i]})",
                fontsize=10,
            )

    plt.tight_layout()
    plt.show()
    
def extract_latent_space_and_compute_shap_importance(model, data_loader, device, layer_to_be_hooked, importance=True, classifierheadwrapper=None, max_background_samples=1000):
    """
    Compute SHAP values for the penultimate layer of the model and track success/failure.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the test set.
        device (str): The device to run computations on ('cuda' or 'cpu').
        importance (bool): Whether to compute SHAP values or only return features.
        max_background_samples (int): Maximum number of background samples to use for SHAP computation. Defaults to 1000.

    Returns:
        tuple: 
            - If `importance=True`: (shap_values, features, labels, success_flags)
            - If `importance=False`: (features, labels, success_flags)
    """
    model.eval()

    # Hook to extract features from the penultimate layer
    penultimate_features = []
    all_labels = []
    success_flags = []  # List to track success/failure per sample
    predictions = []
    
    def hook(module, input, output):
        # The output of the average pooling layer has the shape (nb_images, size of final avg_pool op, 1, 1)
        # because it reduces the spatial dimensions (height and width) to 1x1.
        # Flatten the output to (nb_images, size of final avg_pool op)
        penultimate_features.append(output.detach().flatten(1))

    hook_handle = layer_to_be_hooked.register_forward_hook(hook)  # Attach hook

    # Collect features, labels, and predictions
    with torch.no_grad():
        is_binary = None
        for batch in data_loader:
            if isinstance(batch, dict):
                batch = (batch['image'], batch['label'])  # Convert to tuple

            images = batch[0].to(device, non_blocking=True)     # tensors on GPU
            labels_t = batch[1].to(device, non_blocking=True)   # keep on GPU for compare
            
            # Flatten labels to [B]
            labels_flat = labels_t.view(-1).long()
            all_labels.extend(labels_flat.detach().cpu().numpy().tolist())

            # Forward once
            logits = model(images)
            
            # Decide binary vs multiclass once (first batch)
            if is_binary is None:
                is_binary = (logits.shape[1] == 1)

            if is_binary:
                probs = torch.sigmoid(logits).squeeze(1)        # [B]
                preds_cls = (probs > 0.5).long()                # [B]
                success_flags.extend((preds_cls == labels_flat).detach().cpu().numpy().astype(int).tolist())
                predictions.extend(probs.detach().cpu().numpy())  # store probs
            else:
                probs = torch.softmax(logits, dim=1)            # [B, C]
                preds_cls = probs.argmax(dim=1)                 # [B]
                # labels may be shape [B,1]; squeeze for compare
                success_flags.extend((preds_cls == labels_flat).detach().cpu().numpy().astype(int).tolist())
                predictions.extend(probs.detach().cpu().numpy())

    # Remove hook
    hook_handle.remove()

    # Prepare features and labels
    features = torch.cat(penultimate_features).cpu().detach()
    labels = np.array(all_labels)
    success_flags = np.array(success_flags)  # Convert to numpy array for easier manipulation
    background_features = features.to(device)

    # Limit the number of background samples
    if len(background_features) > max_background_samples:
        background_features = background_features[:max_background_samples]

    if importance:
        # Wrap the classifier head
        classifier_head = classifierheadwrapper

        # SHAP Explainer for the classifier head
        explainer = shap.DeepExplainer(classifier_head, background_features.clone().detach())

        # Compute SHAP values
        shap_values = explainer.shap_values(features.clone().detach())

        return shap_values, features, labels, success_flags
    else:
        return features, labels, success_flags, predictions
    
    
def display_shap_values(shap_df):
    """
    Display SHAP values with feature indices for a single fold.

    Returns:
        pd.Series: Mean absolute SHAP values for each feature.
    """

    # Compute the mean absolute SHAP values for each feature
    shap_importance = shap_df.abs().mean().sort_values(ascending=False)

    return shap_importance


def plot_shap_importance(shap_importance, fold, feature_names=None):
    """
    Plot SHAP feature importance as a bar chart.

    Args:
        shap_importance (pd.Series): Mean absolute SHAP values for each feature.
        fold (int): Fold index for labeling the plot.
        feature_names (list, optional): List of feature names to include in the plot. If provided,
                                         only these features will be plotted.
    """

    if feature_names is not None:
        # Filter shap_importance for the specified features
        shap_importance = shap_importance[shap_importance.index.isin(feature_names)]

    plt.figure(figsize=(15, 8))
    shap_importance.plot(kind="bar")
    plt.title(f"SHAP Feature Importance (Fold {fold})")
    plt.xlabel("Features")
    plt.ylabel("Mean |SHAP Value|")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    
    
def plot_clustered_feature_heatmap(features, fold):
    """
    Plot a clustered heatmap of feature correlations.

    Args:
        features (numpy.ndarray): Features for a fold with shape (num_samples, num_features).
        fold (int): The fold index for labeling the plot.
    """
    # Compute feature correlation matrix
    correlation_matrix = np.corrcoef(features, rowvar=False)  # Correlation between features
    abs_correlation_matrix = np.abs(correlation_matrix)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(abs_correlation_matrix, method='ward')
    clustered_order = leaves_list(linkage_matrix)  # Order of features after clustering

    # Reorder the correlation matrix based on clustering
    clustered_corr_matrix = abs_correlation_matrix[clustered_order][:, clustered_order]

    # Reorder feature labels
    clustered_labels = [f"Feature_{i}" for i in clustered_order]

    # Plot the clustered heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        clustered_corr_matrix,
        xticklabels=clustered_labels,
        yticklabels=clustered_labels,
        cmap="coolwarm",
        annot=False,
        fmt=".2f",
        cbar=True
    )
    plt.title(f"Clustered Feature Correlation Heatmap (Fold {fold})")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


def feature_engineering_pipeline(mean_shap_df, latent_space, shap_threshold=0.05, corr_threshold=0.8):
    """
    Feature engineering pipeline based on SHAP values and feature correlations.

    Args:
        mean_shap_df (pd.Series): Mean absolute SHAP values indexed by feature names.
        latent_space (pd.DataFrame): Latent space values for all features (samples x features).
        shap_threshold (float): Threshold for mean absolute SHAP values.
        corr_threshold (float): Threshold for absolute correlation coefficients.

    Returns:
        pd.DataFrame: Final latent space values of retained features (samples x features).
        list: Final list of retained feature names.
    """
    # Step 1: Filter features based on SHAP threshold
    retained_features = mean_shap_df[mean_shap_df > shap_threshold].index
    retained_features = retained_features.intersection(latent_space.columns)  # Align with latent_space
    print(f"Retained {len(retained_features)} features after SHAP filtering.")

    # Step 2: Compute absolute correlation matrix for retained features
    retained_latent_space = latent_space[retained_features]  # Latent space for retained features
    correlation_matrix = retained_latent_space.corr()
    abs_correlation_matrix = np.abs(correlation_matrix)

    # Visualize correlation heatmap with dendrogram after SHAP filtering


    linkage_matrix = linkage(squareform(1 - abs_correlation_matrix), method="ward")

    # Clustered heatmap
    sns.clustermap(
        abs_correlation_matrix,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        figsize=(12, 12),
        annot=True
    )
    plt.title("Clustered Correlation Heatmap After SHAP Filtering")
    plt.show()

    # Step 3: Identify clusters of correlated features
    clusters = fcluster(linkage_matrix, t=1 - corr_threshold, criterion="distance")
    cluster_groups = {cluster: [] for cluster in np.unique(clusters)}

    for feature, cluster in zip(abs_correlation_matrix.columns, clusters):
        cluster_groups[cluster].append(feature)

    # Step 4: Visualize correlation heatmap for each cluster
    print(f"Identified {len(cluster_groups)} clusters.")

    # Step 5: Keep the most important feature from each cluster
    final_features = []
    for cluster, features in cluster_groups.items():
        if len(features) > 1:
            # Keep only the feature with the highest mean SHAP value
            most_important_feature = max(features, key=lambda f: mean_shap_df[f])
            final_features.append(most_important_feature)
        else:
            final_features.extend(features)

    # # Step 6: Resolve any remaining pairs with correlation > threshold
    retained_latent_space = latent_space[final_features]
    correlation_matrix = retained_latent_space.corr()
    abs_correlation_matrix = np.abs(correlation_matrix)

    while True:
        correlated_pairs = [
            (i, j)
            for i in abs_correlation_matrix.columns
            for j in abs_correlation_matrix.columns
            if i != j and abs_correlation_matrix.loc[i, j] > corr_threshold
        ]
        if not correlated_pairs:
            break

        # Remove the less important feature from each correlated pair
        features_to_remove = set()
        for i, j in correlated_pairs:
            less_important = i if mean_shap_df[i] < mean_shap_df[j] else j
            features_to_remove.add(less_important)

        final_features = [f for f in final_features if f not in features_to_remove]
        retained_latent_space = latent_space[final_features]
        correlation_matrix = retained_latent_space.corr()
        abs_correlation_matrix = np.abs(correlation_matrix)

    print(f"Retained {len(final_features)} features after correlation filtering.")

    # Step 7: Plot final heatmap of retained features correlation
    final_corr_matrix = abs_correlation_matrix.loc[final_features, final_features]
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        final_corr_matrix,
        xticklabels=final_features,
        yticklabels=final_features,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        annot=True,
        cbar=True
    )
    plt.title("Final Retained Features Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

    return retained_latent_space, final_features

def visualize_umap_with_labels(umap_train, umap_test, success, labels, fold=0):
    """
    Improved UMAP visualization with better readability.

    Args:
        umap_train (numpy.ndarray): UMAP-reduced train data (samples x 2).
        umap_test (numpy.ndarray): UMAP-reduced test data (samples x 2).
        success (list): List of success (1) or failure (0) cases in the test set.
        labels (list): List of train labels (0 = Round, 1 = Irregular).
        fold (int): Fold index to visualize.
    """
    # Extract train labels
    round_indices = np.where(np.array(labels) == 0)[0]
    irregular_indices = np.where(np.array(labels) == 1)[0]

    # Extract test labels (success/failure)
    success_indices = np.where(np.array(success) == 1)[0]
    failure_indices = np.where(np.array(success) == 0)[0]

    plt.figure(figsize=(12, 8))

    # Use KDE density estimation for train data
    sns.kdeplot(
        x=umap_train[:, 0], 
        y=umap_train[:, 1], 
        cmap="Blues", 
        fill=True, 
        alpha=0.3, 
        levels=20
    )

    # Plot round cases in train data (blue, circles)
    plt.scatter(
        umap_train[round_indices, 0],
        umap_train[round_indices, 1],
        label="Malignant (Train)",
        alpha=0.4,
        color="blue",
        marker="o",
        s=50  # Slightly larger
    )

    # Plot irregular cases in train data (black, stars)
    plt.scatter(
        umap_train[irregular_indices, 0],
        umap_train[irregular_indices, 1],
        label="Benign/normal (Train)",
        alpha=0.4,
        color="black",
        marker="*",
        s=50
    )

    plt.scatter(
        umap_test[success_indices, 0],
        umap_test[success_indices, 1],
        label="Success (Test)",
        alpha=1.0,
        color="green",
        marker="x",
        s=70  # Larger size for visibility
    )

    # Plot failure cases in test data (red, outlined x)
    plt.scatter(
        umap_test[failure_indices, 0],
        umap_test[failure_indices, 1],
        label="Failure (Test)",
        alpha=1.0,
        color="red",
        marker="x",
        s=70
    )

    # Add plot details
    plt.title(f"UMAP Visualization of Latent Space (Fold {fold})", fontsize=14, fontweight="bold")
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(fontsize=10, loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()
    
    
def analyze_hyperplane_distance(train_latent, train_labels, eval_latent, eval_success, display_distrib=False):
    """
    Train an SVM hyperplane on the train latent space and compute distances for evaluation cases.

    Args:
        train_latent (np.ndarray): Training latent space (samples x features).
        train_labels (np.ndarray): Labels for training data (0 = round, 1 = irregular).
        eval_latent (np.ndarray): Evaluation latent space (samples x features).
        eval_labels (np.ndarray): Labels for evaluation data (0 = round, 1 = irregular).
        eval_success (np.ndarray): Success (1) or Failure (0) for evaluation cases.

    Returns:
        None (Plots distributions and computes AUROC).
    """

    # Train a linear SVM on the latent space
    svm = SVC(kernel="linear")
    svm.fit(train_latent, train_labels)

    # Compute signed distances to the decision hyperplane
    eval_distances = svm.decision_function(eval_latent)

    # Separate success and failure cases based on distances
    success_distances = eval_distances[eval_success == 1]
    failure_distances = eval_distances[eval_success == 0]
    
    scaler = StandardScaler()
    normalized_distances = scaler.fit_transform(eval_distances.reshape(-1, 1)).flatten()
        # Separate success and failure cases based on distances
    success_distances_normalized = normalized_distances[eval_success == 1]
    failure_distances_normalized = normalized_distances[eval_success == 0]
    
    if display_distrib:
        plt.figure(figsize=(8, 6))
        sns.histplot(success_distances_normalized, color='green', label='Success', kde=True, stat="count")
        sns.histplot(failure_distances_normalized, color='red', label='Failure', kde=True, stat="count")
        plt.axvline(0, color='black', linestyle='dashed', label='Decision Boundary (Normalized)')
        plt.xlabel("Distance to Hyperplane")
        plt.ylabel("Count")
        plt.title("Normalized distance to Hyperplane (Success vs Failure)")
        plt.legend()
        plt.show()
    
    return eval_distances

def compute_mean_shap_values(shap_values, fold, true_labels=None, nb_features=50):
    mean_shap_fold = []
    print(f"SHAP Feature Importances Computation")

    # Ensure shap_values is a 2D array for binary classification
    if shap_values.ndim == 3 and true_labels is not None and len(np.unique(true_labels)) == 2:
        shap_values = shap_values.squeeze(-1)  # Remove the last dimension to make it 2D

    # Ensure shap_values is a 3D array for multiclass or 2D for binary classification
    if shap_values.ndim == 3:
        num_samples, num_features, num_classes = shap_values.shape
    elif shap_values.ndim == 2:
        num_samples, num_features = shap_values.shape
        num_classes = 2  # Binary classification treated as two classes (0 and 1)
    else:
        raise ValueError("Unexpected shape of shap_values. Expected 2D or 3D array.")

    for class_idx in range(num_classes):
        print(f"Class {class_idx}: SHAP Feature Importances")

        if shap_values.ndim == 3:  # Multiclass classification
            # Extract SHAP values for the current class
            class_shap_values = shap_values[:, :, class_idx]
        else:  # Binary classification
            # Isolate cases with true label matching the current class
            class_shap_values = shap_values[true_labels == class_idx, :]

        # Create a DataFrame for SHAP values of the current class
        shap_df = pd.DataFrame(
            class_shap_values,
            columns=[f"Feature_{i}" for i in range(num_features)]
        )

        # Compute mean absolute SHAP values
        mean_abs_shap = shap_df.abs().mean(axis=0)

        # Select top 50 features
        top_n_features = mean_abs_shap.nlargest(nb_features).index

        # Keep only the top 50 features
        shap_df_top_n = shap_df[top_n_features]

        shap_importance = display_shap_values(shap_df_top_n)
        print(shap_importance)
        mean_shap_fold.append((fold, class_idx, shap_importance))

    return mean_shap_fold
        
def compute_knn_distances_to_train_data(model, train_loader, test_loader, layer, device, latent_spaces, mean_shap_importances, num_classes):
    
    latent_space_training, labels_training, _, _ = extract_latent_space_and_compute_shap_importance(
        model=model,
        data_loader=train_loader,
        device=device,
        layer_to_be_hooked=layer,
        importance=False
    )
    
    latent_space_test, labels_test, success_test, _ = extract_latent_space_and_compute_shap_importance(
        model=model,
        data_loader=test_loader,
        device=device,
        layer_to_be_hooked=layer,
        importance=False
    )
    
    train_latent_space = pd.DataFrame(latent_space_training, columns=latent_spaces.columns)
    test_latent_space = pd.DataFrame(latent_space_test, columns=latent_spaces.columns)
    
    knn_distances_all = np.zeros(len(test_latent_space))
    successes_all = np.zeros(len(test_latent_space))
    
    for i in range(num_classes):
        print('class' + str(i))
        train_latent_space_class = train_latent_space[mean_shap_importances[i][2].keys()]
        
        mask_training = labels_training == i
        train_latent_space_filtered = train_latent_space_class[mask_training]
        
        print(f'Number of samples with true label {i}: {len(train_latent_space_filtered)}')
        
        test_latent_space_class = test_latent_space[mean_shap_importances[i][2].keys()]
        
        mask_test = labels_test == i
        test_latent_space_filtered = test_latent_space_class[mask_test]
        
        print(f'Number of samples with true label {i}: {len(test_latent_space_filtered)}')
        
        success_test_filtered = success_test[mask_test.flatten()]
        indices_test_filtered = np.where(mask_test.flatten())[0]
        
        scaler = StandardScaler()
        train_latent_space_standardized = scaler.fit_transform(train_latent_space_filtered)
        
        pca = PCA(n_components=0.9)
        train_latent_space_pca = pca.fit_transform(train_latent_space_standardized)
        
        test_latent_space_standardized = scaler.transform(test_latent_space_filtered)
        test_latent_space_pca = pca.transform(test_latent_space_standardized)
        
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(train_latent_space_pca)
        distances, _ = knn.kneighbors(test_latent_space_pca)
        
        average_distances = distances.mean(axis=1)
        
        knn_distances_all[indices_test_filtered] = average_distances
        if num_classes == 2:
            successes_all[indices_test_filtered] = success_test_filtered#.squeeze(-1)
        else:
            successes_all[indices_test_filtered] = success_test_filtered

    return knn_distances_all, successes_all