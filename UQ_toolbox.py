import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, log_loss
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
import numpy as np
import pandas as pd
import seaborn as sns
import os
import multiprocessing
import torch
import re
from torchvision import transforms
import torch.nn.functional as F
from collections import defaultdict
from gps_augment.utils.randaugment import BetterRandAugment
import shap
import torch.multiprocessing as mp

class AddBatchDimension:
    def __call__(self, image):
        # Ensure the image is a tensor and add batch dimension
        if isinstance(image, torch.Tensor):
            return image.unsqueeze(0).float()
        raise TypeError("Input should be a torch Tensor")


def get_prediction(model, image, device, softmax_application=False):
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
    image = image.to(device)
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():  # Disable gradient computation
        prediction = model(image).detach().cpu()
    if softmax_application:
        prediction = F.softmax(prediction, dim=1)
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


def TTA(transformations, models, data_loader, device, nb_augmentations=10, usingBetterRandAugment=False, n=2, m=45, batch_norm=False, nb_channels=1, mean=None, std=None, image_size=51, softmax_application=False):
    """
    Perform Test-Time Augmentation (TTA) on a batch of images using specified transformations and models.

    Args:
        transformations (callable or list): Transformations to apply to each image.
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
    """
    if usingBetterRandAugment and not isinstance(transformations, list):
        raise ValueError("Transformations must be a list when usingBetterRandAugment.")
    
    tta_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                batch = (batch['image'], batch['label'])  # Convert to tuple

            images = batch[0]  # Access the images using positional indexing
            
            augmented_inputs = apply_augmentations(images, transformations, nb_augmentations, usingBetterRandAugment, n, m, batch_norm, nb_channels, mean, std, image_size)
            batch_predictions = get_batch_predictions(models, augmented_inputs, device, softmax_application)
            averaged_predictions = average_predictions(batch_predictions, images.size(0), nb_augmentations, usingBetterRandAugment, transformations)
            
            tta_predictions.append(averaged_predictions)

    global_preds = torch.cat(tta_predictions, dim=0)  # Shape: [total_images, num_augmentations, num_classes]
    stds = compute_stds(global_preds)
    
    return stds, global_preds


def apply_augmentations(images, transformations, nb_augmentations, usingBetterRandAugment, n, m, batch_norm, nb_channels, mean, std, image_size):
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
    if usingBetterRandAugment and isinstance(transformations, list):
        augmented_inputs = torch.stack(
            [torch.stack(
                [transforms.Compose([
                    transforms.ToPILImage(),
                    *([to_3_channels] if nb_channels == 1 else []),  # Conditionally add to_3_channels
                    BetterRandAugment(n=n, m=m, resample=False, transform=policy, verbose=True, randomize_sign=False, image_size=image_size),
                    *([to_1_channel] if nb_channels == 1 else []),  # Conditionally add to_1_channel
                    transforms.PILToTensor(),
                    transforms.Lambda(lambda x: x.float()) if nb_channels == 1 else transforms.ConvertImageDtype(torch.float),
                    *([transforms.Normalize(mean=mean, std=std)] if batch_norm is False else [])
                ])(image) for policy in transformations]  # Use only up to `nb_augmentations` policies
            ) for image in images], dim=0
        )  # Shape: [batch_size, num_augmentations, C, H, W]
    else:
        augmented_inputs = torch.stack(
            [torch.stack([transformations(image) for _ in range(nb_augmentations)]) for image in images], 
            dim=0
        )  # Shape: [batch_size, num_augmentations, C, H, W]
    
    augmented_inputs = augmented_inputs.view(-1, *augmented_inputs.shape[2:])  # Shape: [batch_size * num_augmentations, C, H, W]
    return augmented_inputs


def get_batch_predictions(models, augmented_inputs, device, softmax_application):
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
        batch_predictions = [
            get_prediction(model, augmented_inputs, device, softmax_application) for model in models
        ]
    else:
        batch_predictions = [get_prediction(models, augmented_inputs, device, softmax_application)]
    
    batch_predictions = torch.stack(batch_predictions, dim=0)  # Shape: [num_models, batch_size * num_augmentations, num_classes]
    return batch_predictions


def average_predictions(batch_predictions, batch_size, nb_augmentations, usingBetterRandAugment, transformations):
    """
    Average predictions across models and group augmentations back with their respective images.

    Args:
        batch_predictions (torch.Tensor): Batch predictions.
        batch_size (int): Size of the batch.
        nb_augmentations (int): Number of augmentations to apply per image.
        usingBetterRandAugment (bool): If True, use BetterRandAugment with provided policies.
        transformations (callable or list): Transformations to apply to each image.

    Returns:
        torch.Tensor: Averaged predictions.
    """
    averaged_predictions = torch.mean(batch_predictions, dim=0)  # Shape: [batch_size * num_augmentations, num_classes]
    if usingBetterRandAugment:
        averaged_predictions = averaged_predictions.view(batch_size, len(transformations), -1)  # Shape: [batch_size, num_augmentations, num_classes]
    else:
        averaged_predictions = averaged_predictions.view(batch_size, nb_augmentations, -1)  # Shape: [batch_size, num_augmentations, num_classes]
    return averaged_predictions


def compute_stds(global_preds):
    """
    Compute standard deviations for the predictions.

    Args:
        global_preds (torch.Tensor): Global predictions.

    Returns:
        list: List of standard deviations for each sample.
    """
    if global_preds.ndim == 2:
        stds = torch.std(global_preds, dim=1).squeeze().tolist()  # Binary classification: shape (num_models, num_samples)
    elif global_preds.ndim == 3:
        stds_per_class = torch.std(global_preds, dim=1).squeeze()  # Multiclass classification: shape (num_models, num_samples, num_classes)
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
    if predictions.ndim == 1:
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
        
def model_calibration_plot(true_labels, predictions):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    prob_true, prob_pred = plot_calibration_curve(np.array(true_labels), np.array(predictions))
    plt.plot(prob_pred, prob_true, marker='o', label=f'Model Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid()
    plt.show()

def UQ_method_plot(correct_predictions, incorrect_predictions, y_title, title, swarmplot=True):
    df = pd.DataFrame({
        y_title: correct_predictions + incorrect_predictions,
        'Category': ['Correct Results'] * len(correct_predictions) + ['Incorrect Results'] * len(incorrect_predictions)
        })
    
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(x='Category', y=y_title, data=df, palette='muted')
    if swarmplot:
        sns.swarmplot(x='Category', y=y_title, data=df, color='k', alpha=0.3)
    
    # Show the plot
    plt.title(title)
    plt.show()

def roc_curve_UQ_method_computation(correct_predictions, incorrect_predictions):
    failures_gstd = np.ones(len(incorrect_predictions))
    success_gstd = np.zeros(len(correct_predictions))
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(np.concatenate((failures_gstd, success_gstd)), np.concatenate((incorrect_predictions, correct_predictions)))

    # Calculate and print AUC
    auc_score = roc_auc_score(np.concatenate((failures_gstd, success_gstd)), np.concatenate((incorrect_predictions, correct_predictions)))
    
    return fpr, tpr, auc_score

def roc_curve_UQ_methods_plot(method_names, fprs, tprs, auc_scores): 
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

def standardize_and_mean_ensembling(distributions):
    """
    Standardizes any number of distributions using a global mean and standard deviation.
    Returns a single column array with the maximum value for each instance across the standardized distributions.
    
    Parameters:
    distributions: 2D numpy array, where each column represents a different distribution (uncertainty method).
    
    Returns:
    max_values: 1D numpy array containing the maximum standardized value for each row.
    """
    # Flatten the array to compute the global mean and standard deviation
    combined = distributions.flatten()
    
    # Compute global mean and standard deviation
    global_mean = np.mean(combined)
    global_std_dev = np.std(combined)
    
    # Apply z-score standardization to each distribution (column)
    standardized_distributions = (distributions - global_mean) / global_std_dev
    
    # Find the maximum standardized value for each instance (row)
    mean_values = np.mean(standardized_distributions, axis=1)
    
    return mean_values
    
def to_3_channels(img):
    if img.mode == 'L':  # Grayscale image
        img = img.convert('RGB')  # Convert to 3 channels by duplicating
    return img

def to_1_channel(img):
    img = img.convert('L')  # Convert back to grayscale
    return img

def apply_randaugment_and_store_results(data_loader, models, N, M, num_policies, device, folder_name='savedpolicies', batch_norm=False, mean=False, std=False, nb_channels=1, image_size=51, softmax_application=False):
    """
    Apply RandAugment transformations to the data and store the results.
    Parameters:
    data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
    models (list): List of models to use for predictions.
    N (int): Number of augmentation transformations to apply.
    M (int): Magnitude of the augmentation transformations.
    num_policies (int): Number of random augmentation policies to generate.
    device (torch.device): Device to run the models on (e.g., 'cpu' or 'cuda').
    batch_norm (bool, optional): Whether to use batch normalization. Default is False.
    mean (bool or list, optional): Mean for normalization. Default is False.
    std (bool or list, optional): Standard deviation for normalization. Default is False.
    bw (bool, optional): Whether to convert images to black and white. Default is True.
    Returns:
    tuple: A tuple containing:
        - results_dict (dict): Dictionary with augmentation policies as keys and predictions as values.
        - dict_name (str): Name of the results dictionary.
    """
    results_dict = {}
    
    # Create folder for saving policies if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    for i in range(num_policies):
        print(f'augmentation n:{i}')
        augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            *([to_3_channels] if nb_channels == 1 else []),  # Conditionally add to_3_channels
            BetterRandAugment(N, M, True, False, randomize_sign=False, image_size=image_size),
            *([to_1_channel] if nb_channels == 1 else []),  # Conditionally add to_1_channel
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float()) if nb_channels == 1 else transforms.ConvertImageDtype(torch.float),
            *([transforms.Normalize(mean=mean, std=std)] if not batch_norm else [])
        ])
        # Apply the policy and get the predictions
        predictions = apply_policy_and_get_predictions(data_loader, models, augment_transform, device, softmax_application=softmax_application)
        # Store the results in the dictionary with a unique key for each random policy
        policy_key = str(augment_transform.transforms[2].get_transform()) if nb_channels == 1 else str(augment_transform.transforms[1].get_transform())
        # Extract the policy key based on the number of channels
        if nb_channels == 1:
            policy_key = str(augment_transform.transforms[2].get_transform())
        else:
            policy_key = str(augment_transform.transforms[1].get_transform())
        
        # Store the predictions in the results dictionary with the policy key
        results_dict[policy_key] = predictions
        # Saving results in a .npz file in the 'savedpolicies' folder
        filename = f'{folder_name}/N{N}_M{M}_{policy_key}.npz'
        np.savez_compressed(filename, predictions=predictions)

    dict_name = f'results_randaugment_{num_policies}TTA_N{N}_M{M}'
    
    return results_dict, dict_name

def group_and_merge_similar_augmentations(policies, keys):
    """
    Group similar augmentations based on type and magnitude closeness,
    and merge them by averaging the magnitudes.

    Parameters:
    - policies: List of tuples [(type, magnitude), ...] representing augmentations.
    - tolerance: Fraction of the max magnitude to define closeness in magnitudes.

    Returns:
    - merged_augmentations: List of tuples [(type, avg_magnitude), ...] with merged augmentations.
    """
    
    similar_augmentations = [keys[i] for i in policies]
    # Step 1: Clean and parse the data
    cleaned_data = [re.sub(r'N2_M45_|\.npz', '', item) for item in similar_augmentations]
    parsed_data = [eval(item) for item in cleaned_data]

    # Step 2: Normalize the lists (sort tuples within the list)
    normalized_data = [tuple(sorted(augmentation)) for augmentation in parsed_data]
    
    # Step 3: Remove exact duplicates in the dataset
    unique_data = list(set(normalized_data))

    # Step 3: Define a similarity function
    def are_similar(list1, list2, tolerance=25.0):
        """
        Check if two lists are similar based on augmentation type and magnitude tolerance.
        """
        if len(list1) != len(list2):
            return False
        
        for (type1, mag1), (type2, mag2) in zip(list1, list2):
            if type1 != type2 or abs(mag1 - mag2) > tolerance:
                return False
        return True

    # Step 5: Group similar lists
    groups = []
    visited = [False] * len(unique_data)

    for i, aug1 in enumerate(unique_data):
        if visited[i]:
            continue
        
        group = [aug1]
        visited[i] = True
        
        for j, aug2 in enumerate(unique_data):
            if not visited[j] and are_similar(aug1, aug2):
                group.append(aug2)
                visited[j] = True
        
        groups.append(group)

    # Step 6: Keep groups with multiple augmentations and average magnitudes
    result = []

    for group in groups:
        if len(group) > 1:  # Keep only groups with multiple augmentations
            # Combine all augmentations in the group
            combined = defaultdict(list)
            for aug_list in group:
                for i, (aug_type, magnitude) in enumerate(aug_list):
                    combined[(aug_type, i)].append(magnitude)
            
            # Calculate average magnitude for each augmentation type/position pair
            averaged_group = [
                (aug_type, sum(magnitudes) / len(magnitudes)) for (aug_type, _), magnitudes in combined.items()
            ]
            
            # Format as strings and append to result
            result.append(
                "[" + ", ".join(f"({t}, {m})" for t, m in sorted(averaged_group, key=lambda x: x[0])) + "]"
            )

    return result


def prioritize_and_merge_with_similarity(results, keys, top_k=5):
    """
    Prioritize augmentations from top-performing searches considering similarity,
    and merge close augmentations by averaging magnitudes.

    Parameters:
    - results: List of tuples (best_metric, best_group_indices, all_roc_aucs) from parallel searches.
    - top_k: Number of top-performing searches to consider.
    - threshold: Minimum number of top searches in which an augmentation must appear to be retained.
    - tolerance: Fraction of the max magnitude to define closeness in magnitudes.

    Returns:
    - prioritized_augmentations: List of merged augmentations [(type, avg_magnitude), ...].
    """
    # Sort results by the best_metric (ROC AUC) in descending order
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)

    # Select the top `k` performing searches
    top_results = sorted_results[:top_k]
    print(top_results)
    # Extract augmentations from the top searches
    all_policies = []
    for _, group_indices, _ in top_results:
        all_policies.extend(group_indices)

    # Group and merge similar augmentations
    merged_augmentations = group_and_merge_similar_augmentations(all_policies, keys)

    return merged_augmentations

def apply_policy_and_get_predictions(data_loader, models, augment_transform, device, softmax_application=False):
    results = []
    
    # Predict for each sample in the test set
    for batch in data_loader:
        if isinstance(batch, dict):
            batch = (batch['image'], batch['label'])  # Convert to tuple

        images = batch[0]  # Access the images using positional indexing
        augmented_inputs = torch.stack([augment_transform(image) for image in images])

        with torch.no_grad():
            if isinstance(models, list):
                batch_predictions = []
                for model in models:
                    model.to(device)
                    model_preds = get_prediction(model, augmented_inputs, device, softmax_application)
                    batch_predictions.append(model_preds)
                
                stacked_preds = np.stack(batch_predictions, axis=0)
                # Average predictions for each sample in the batch across models
                predictions = np.mean(stacked_preds, axis=0) 
            else: 
                models.to(device)
                predictions = get_prediction(models, augmented_inputs, device, softmax_application)
            results.extend(predictions)
    
    # Return the results as a numpy array (shape: [num_samples, 1] for binary, [num_samples, num_classes] for multi-class)
    return np.array(results)

def greedy_search(initial_aug_idx, val_preds, good_idx, bad_idx, select_only, min_improvement=0.005, patience=5):
    """
    A single greedy search instance that starts from a random initial augmentation (initial_aug_idx).
    Returns the best augmentations based on the maximum ROC AUC achieved.
    """
    group_indices = [initial_aug_idx]  # Initialize with the given augmentation
    best_metric = -np.inf
    best_group_indices = list(group_indices)  # Track the augmentations that give the best ROC AUC

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

            current_augmentations = group_indices + [new_i]  # Add new augmentation to the selected group
            if val_preds[current_augmentations, :, :].ndim == 2:
                # Binary classification: shape (num_models, num_samples)
                preds_std = np.std(val_preds[current_augmentations, :, :], axis=0)
            elif val_preds[current_augmentations, :, :].ndim == 3:
                # Multiclass classification: shape (num_models, num_samples, num_classes)
                stds_per_class = np.std(val_preds[current_augmentations, :, :], axis=0)
                preds_std = np.mean(stds_per_class, axis=1)

            # Compute ROC AUC for the current set of augmentations
            roc_auc = roc_curve_UQ_method_computation(
                [preds_std[k] for k in good_idx], 
                [preds_std[j] for j in bad_idx]
            )[2]

            if roc_auc > 0.5 and roc_auc > best_iteration_metric:
                best_s = new_i
                best_iteration_metric = roc_auc
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
    
def select_greedily_on_ens(all_preds, good_idx, bad_idx, keys, search_set_len, select_only=50, num_workers=1, num_searches=10, top_k=5, method='top_policies'):

    val_preds = np.copy(all_preds[:, :search_set_len, :])
    # Initialize a multiprocessing pool for parallel execution
    
    with mp.Pool(processes=num_workers) as pool:
        # Initialize random starting augmentations for each search
        initial_augmentations = [
            np.random.choice(range(val_preds.shape[0])) for _ in range(num_searches)
        ]
        try:
            results = pool.starmap(greedy_search, [(initial_aug, val_preds, good_idx, bad_idx, select_only) for initial_aug in initial_augmentations])
        except IndexError as e:
            print("Debugging IndexError...")
            print(f"val_preds shape: {val_preds.shape}")
            pool.close()
            pool.join()
            raise e
        finally:
            pool.close()
            pool.join()

    # Select the best result based on the ROC AUC metric
    best_result = max(results, key=lambda x: x[0])  # Select based on the best metric (ROC AUC)
    best_metric, best_group_indices, _ = best_result

    print("\nParallel greedy search complete. Best metric:", best_metric)
    
    if method == 'mutualized_policies':
        # Prioritize policies from top-performing searches
        policies = prioritize_and_merge_with_similarity(results, keys, top_k=top_k)
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

def perform_greedy_policy_search(npz_dir, good_idx, bad_idx, max_iterations=50, num_workers=1, num_searches=10, top_k=5, plot=True, method='top_policies'):
    """
    Perform parallel greedy policy search using the select_greedily_on_ens function.
    Loads .npz files, aggregates predictions, and performs policy search.
    """

    print('Loading predictions...')
    # Step 1: Load the .npz files containing predictions
    all_preds, all_keys = load_npz_files_for_greedy_search(npz_dir)
    search_set_len = all_preds[0].size
    
    if method == 'top_policies':
        # Step 3: Call the `select_greedily_on_ens` function with loaded predictions
        selected_policies, results = select_greedily_on_ens(
            all_preds,  # Predictions from npz files
            good_idx,
            bad_idx,
            all_keys,
            search_set_len=search_set_len,
            select_only=max_iterations,  # Size of the dataset to be used for searching
            num_workers=num_workers,  # Number of workers for parallel processing
            num_searches=num_searches,  # Number of parallel greedy search processes
            top_k=top_k,
            method=method
        )
        # Return the selected policies and their corresponding names
        selected_policy_names = [all_keys[i] for i in selected_policies]
        
    elif method == 'mutualized_policies':
                # Step 3: Call the `select_greedily_on_ens` function with loaded predictions
        selected_policies, results= select_greedily_on_ens(
            all_preds,  # Predictions from npz files
            good_idx,
            bad_idx,
            all_keys,
            search_set_len=search_set_len,
            select_only=max_iterations,  # Size of the dataset to be used for searching
            num_workers=num_workers,  # Number of workers for parallel processing
            num_searches=num_searches,  # Number of parallel greedy search processes
            top_k=top_k,
            method=method
        )
        selected_policy_names = selected_policies
    if plot:
        # Plot the ROC AUC curves for each greedy search
        plot_auc_curves(results)

    
    
    return selected_policy_names


def visualize_input_shap_overlayed_multimodel(
    models, eval_dataloader, device, success_indices, failure_indices, sample_size=5
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
    
def extract_latent_space_and_compute_shap_importance(model, data_loader, device, importance=True, classifierheadwrapper=None):
    """
    Compute SHAP values for the penultimate layer of the model and track success/failure.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the test set.
        device (str): The device to run computations on ('cuda' or 'cpu').
        importance (bool): Whether to compute SHAP values or only return features.

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
        penultimate_features.append(output.detach())

    hook_handle = model.fc1.register_forward_hook(hook)  # Attach hook

    # Collect features, labels, and predictions
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['shape'].cpu().numpy()
            all_labels.extend(labels)

            # Compute model predictions
            preds = model(images).cpu().numpy()
            predicted_classes = (preds > 0.5).astype(int)  # Convert to binary classification
            
            # Track success (1) / failure (0)
            success_flags.extend((predicted_classes.flatten() == labels).astype(int))
            predictions.extend(preds)

    # Remove hook
    hook_handle.remove()

    # Prepare features and labels
    features = torch.cat(penultimate_features).cpu().detach().numpy()
    labels = np.array(all_labels)
    success_flags = np.array(success_flags)  # Convert to numpy array for easier manipulation

    if importance:
        # Wrap the classifier head
        classifier_head = classifierheadwrapper

        # SHAP Explainer for the classifier head
        explainer = shap.DeepExplainer(classifier_head, torch.tensor(features, dtype=torch.float32, device=device))

        # Compute SHAP values
        shap_values = explainer.shap_values(torch.tensor(features, dtype=torch.float32, device=device))
        shap_values = shap_values.squeeze(axis=-1)

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
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

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
        label="Round (Train)",
        alpha=0.4,
        color="blue",
        marker="o",
        s=50  # Slightly larger
    )

    # Plot irregular cases in train data (black, stars)
    plt.scatter(
        umap_train[irregular_indices, 0],
        umap_train[irregular_indices, 1],
        label="Irregular (Train)",
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

