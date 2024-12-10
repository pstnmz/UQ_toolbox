import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, log_loss
import numpy as np
import pandas as pd
import seaborn as sns
import os
import multiprocessing
import torch
import re
import random
from torchvision import transforms
import torch.nn.functional as F
from gps_augment.utils.randaugment import BetterRandAugment

class AddBatchDimension:
    def __call__(self, image):
        # Ensure the image is a tensor and add batch dimension
        if isinstance(image, torch.Tensor):
            return image.unsqueeze(0).float()
        raise TypeError("Input should be a torch Tensor")


def get_prediction(model, image, device):
    image = image.to(device)
    prediction = model(image).cpu().detach().numpy()
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


def TTA(transformations, models, data_loader, device, nb_augmentations=10, usingBetterRandAugment=False, n=None, m=None, batch_norm=False, bw=False, mean=None, std=None):
    tta_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['image']  # Get the batch of images
            
            # If using BetterRandAugment, apply the provided policies
            if usingBetterRandAugment and isinstance(transformations, list):
                # If using BetterRandAugment, apply the provided policies
                augmented_inputs = torch.stack(
                    [torch.stack(
                        [transforms.Compose([
                            transforms.ToPILImage(),
                            to_3_channels if bw else None,  # Use * to unpack only if `bw=True`
                            BetterRandAugment(n=n, m=m, resample=False, transform=policy, verbose=True, randomize_sign=False),
                            to_1_channel if bw else None,
                            transforms.PILToTensor(),
                            transforms.Lambda(lambda x: x.float()) if bw else transforms.ConvertImageDtype(torch.float),
                            transforms.Normalize(mean=mean, std=std)
                        ])(image) for policy in transformations]  # Use only up to `nb_augmentations` policies
                    ) for image in inputs], dim=0
                )  # Shape: [batch_size, num_augmentations, C, H, W]
            else:
                
                # Apply the specified number of augmentations per image and stack them
                augmented_inputs = torch.stack(
                    [torch.stack([transformations(image) for _ in range(nb_augmentations)]) for image in inputs], 
                    dim=0
                )  # Shape: [batch_size, num_augmentations, C, H, W]
            
            # Reshape augmented_inputs to combine the batch and augmentation dimensions
            augmented_inputs = augmented_inputs.view(-1, *augmented_inputs.shape[2:])  # Shape: [batch_size * num_augmentations, C, H, W]
            
            if isinstance(models, list):
                # Perform predictions for the augmented inputs (batch-wise)
                batch_predictions = [
                    torch.tensor(get_prediction(model, augmented_inputs, device)) for model in models
                ]
            else:
                batch_predictions = [torch.tensor(get_prediction(models, augmented_inputs, device))]
            
            # Stack predictions from different models
            batch_predictions = torch.stack(batch_predictions, dim=0)  # Shape: [num_models, batch_size * num_augmentations, num_classes]
            
            # Average predictions across models (keeping augmentations separate)
            averaged_predictions = torch.mean(batch_predictions, dim=0)  # Shape: [batch_size * num_augmentations, num_classes]
            

            # Reshape averaged_predictions to group augmentations back with their respective images
            averaged_predictions = averaged_predictions.view(inputs.size(0), len(transformations), -1) if usingBetterRandAugment else averaged_predictions.view(inputs.size(0), nb_augmentations, -1) # Shape: [batch_size, num_augmentations, num_classes]

            # Collect predictions
            tta_predictions.append(averaged_predictions)

    # Stack all predictions together
    global_preds = torch.cat(tta_predictions, dim=0)  # Shape: [total_images, num_augmentations, num_classes]

    # Compute standard deviation across the augmentations for each image
    stds = torch.std(global_preds, dim=1).squeeze().tolist()  # Shape: [total_images, num_classes]
    
    return stds, global_preds


def ensembling_predictions(models, image):
    ensembling_predictions = [get_prediction(model, image) for model in models]
    
    return ensembling_predictions
    
def distance_to_hard_labels_computation(predictions):
    distances = [0.5 - abs(pred - 0.5) for pred in predictions]
    
    return distances

def ensembling_stds_computation(models_predictions):
    stds = [np.std(row) for row in zip(*models_predictions)]
    
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

def UQ_method_plot(correct_predictions, incorrect_predictions, y_title, title):
    df = pd.DataFrame({
        y_title: correct_predictions + incorrect_predictions,
        'Category': ['Correct Results'] * len(correct_predictions) + ['Incorrect Results'] * len(incorrect_predictions)
        })
    
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(x='Category', y=y_title, data=df, palette='muted')
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

def apply_randaugment_and_store_results(data_loader, models, N, M, num_policies, device, binary_classification=False, batch_norm=False, mean=False, std=False, bw=True):
    results_dict = {}
    
    # Create folder for saving policies if it doesn't exist
    os.makedirs('savedpolicies', exist_ok=True)
    for i in range(num_policies):
        print(f'augmentation n:{i}')
        
        augment_transform = transforms.Compose([
                                transforms.ToPILImage(),
                                to_3_channels if bw else None,  # Use * to unpack only if `bw=True`
                                BetterRandAugment(N, M, True, False, randomize_sign=False),
                                to_1_channel if bw else None,
                                transforms.PILToTensor(),
                                transforms.Lambda(lambda x: x.float()) if bw else transforms.ConvertImageDtype(torch.float),
                                transforms.Normalize(mean=mean, std=std)])
        
        # Apply the policy and get the predictions
        predictions = apply_policy_and_get_predictions(data_loader, models, augment_transform, device, binary_classification=binary_classification)
        
        # Store the results in the dictionary with a unique key for each random policy
        policy_key = str(augment_transform.transforms[2].get_transform())
        results_dict[policy_key] = predictions
        
        # Saving results in a .npz file in the 'savedpolicies' folder
        filename = f'savedpolicies/N{N}_M{M}_{policy_key}.npz'
        np.savez_compressed(filename, predictions=predictions)

    dict_name = f'results_randaugment_{num_policies}TTA_N{N}_M{M}'
    
    return results_dict, dict_name

def apply_policy_and_get_predictions(data_loader, models, augment_transform, device, binary_classification=False):
    results = []
    
    # Predict for each sample in the test set
    for batch in data_loader:
        inputs = batch['image']
        augmented_inputs = torch.stack([augment_transform(image) for image in inputs])

        with torch.no_grad():
            if isinstance(models, list):
                batch_predictions = []
                for model in models:
                    model_preds = get_prediction(model, augmented_inputs, device)
                    batch_predictions.append(model_preds)
                
                stacked_preds = np.stack(batch_predictions, axis=0)
                # Average predictions for each sample in the batch across models
                predictions = np.mean(stacked_preds, axis=0) 
            else: 
                predictions = get_prediction(models, augmented_inputs, device)
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
            preds_std = np.std(val_preds[current_augmentations, :, :], axis=0)

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
    plt.legend()
    plt.grid(True)
    plt.show()
    
def select_greedily_on_ens(all_preds, good_idx, bad_idx, search_set_len, select_only=50, num_workers=1, num_searches=10):
    """
    Run multiple greedy search processes in parallel and select the best result based on the metric.
    """
    val_preds = all_preds[:, :search_set_len, :]

    # Initialize a multiprocessing pool for parallel execution
    pool = multiprocessing.Pool(processes=num_workers)

    # Run multiple greedy searches with different initializations in parallel
    initial_augmentations = [np.random.choice(range(val_preds.shape[0])) for _ in range(num_searches)]
    results = pool.starmap(greedy_search, [(initial_aug, val_preds, good_idx, bad_idx, select_only) for initial_aug in initial_augmentations])

    pool.close()
    pool.join()

    # Select the best result based on the ROC AUC metric
    best_result = max(results, key=lambda x: x[0])  # Select based on the best metric (ROC AUC)
    best_metric, best_group_indices, all_roc_aucs = best_result

    print("\nParallel greedy search complete. Best metric:", best_metric)
    return np.array(best_group_indices), all_roc_aucs, results

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

def perform_greedy_policy_search(npz_dir, good_idx, bad_idx, select_by='roc_auc', max_iterations=50, num_workers=1, num_searches=10):
    """
    Perform parallel greedy policy search using the select_greedily_on_ens function.
    Loads .npz files, aggregates predictions, and performs policy search.
    """
    print('Loading predictions...')
    # Step 1: Load the .npz files containing predictions
    all_preds, all_keys = load_npz_files_for_greedy_search(npz_dir)
    search_set_len = all_preds[0].size

    # Step 3: Call the `select_greedily_on_ens` function with loaded predictions
    selected_policies, all_roc_aucs, results = select_greedily_on_ens(
        all_preds,  # Predictions from npz files
        good_idx,
        bad_idx,
        search_set_len=search_set_len,
        select_only=max_iterations,  # Size of the dataset to be used for searching
        num_workers=num_workers,  # Number of workers for parallel processing
        num_searches=num_searches  # Number of parallel greedy search processes
    )

    # Plot the ROC AUC curves for each greedy search
    plot_auc_curves(results)

    # Return the selected policies and their corresponding names
    selected_policy_names = [all_keys[i] for i in selected_policies]
    return selected_policy_names