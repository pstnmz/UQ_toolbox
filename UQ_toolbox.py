import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, log_loss
import numpy as np
import pandas as pd
import seaborn as sns
import os
import multiprocessing
import torch
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

def TTA_test(transforms, models, image, device, nb_augmentations=10):
    plt.close('all')
    if isinstance(models, list):
        tta_predictions = [np.mean([get_prediction(model, transforms(image), device) for model in models]) for _ in range(nb_augmentations)]
    else: 
        tta_predictions = [get_prediction(models, transforms(image), device) for _ in range(nb_augmentations)]
    
    std = np.std(tta_predictions)  
    
    return tta_predictions, std


def TTA(transforms, models, data_loader, device, nb_augmentations=10, usingBetterRandAugment=False, policies=None):
    all_augmentations = []
    
    for _ in range(nb_augmentations):
        print(_)
        tta_predictions = []
        # Predict for each sample in the test set
        for i, batch in enumerate(data_loader):
            inputs = batch['image']
            augmented_inputs = torch.stack([transforms(image) for image in inputs])

            with torch.no_grad():
                if isinstance(models, list):
                    batch_predictions = []
                    for model in models:
                        model_preds = get_prediction(model, augmented_inputs, device)
                        if isinstance(model_preds, np.ndarray):
                            model_preds = torch.tensor(model_preds)
                        batch_predictions.append(model_preds)
                    
                    stacked_preds = torch.stack(batch_predictions, dim=0)
                    # Average predictions for each sample in the batch across models
                    predictions = torch.mean(stacked_preds, dim=0) 
                else: 
                    predictions = get_prediction(models, augmented_inputs, device)
            tta_predictions.extend(predictions)
        all_augmentations.append(torch.tensor(tta_predictions))
    global_preds = torch.stack(all_augmentations, dim=1)
    std = torch.std(global_preds, dim=1).tolist() 
    
    return global_preds, std

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
    max_values = np.mean(standardized_distributions, axis=1)
    
    return max_values
    
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
        if batch_norm is False:
            if bw is True:
                augment_transform = transforms.Compose([
                    AddBatchDimension(),
                    transforms.ToPILImage(),
                    to_3_channels,
                    BetterRandAugment(2, 45, True, False, verbose=True),
                    to_1_channel,
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),  # Convert to float
                    transforms.Lambda(lambda x: x * 255.0),
                    transforms.Normalize(mean=mean, std=std)
                ])
            else:
                augment_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    BetterRandAugment(N, M, True, False, verbose=True),
                    transforms.PILToTensor(),
                    AddBatchDimension(),
                    transforms.Normalize(mean=mean, std=std)
                ])
        else:
            # Initialize BetterRandAugment with given N and M (random augmentations applied)
            augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                BetterRandAugment(n=N, m=M)
            ])
        
        # Apply the policy and get the predictions
        predictions = apply_policy_and_get_predictions(data_loader, models, augment_transform, device, binary_classification=binary_classification)
        
        # Store the results in the dictionary with a unique key for each random policy
        policy_key = str(augment_transform.transforms[3].get_transform())
        results_dict[policy_key] = predictions
        
        # Saving results in a .npz file in the 'savedpolicies' folder
        filename = f'savedpolicies/N{N}_M{M}_{policy_key}.npz'
        np.savez_compressed(filename, predictions=predictions)

    dict_name = f'results_randaugment_{num_policies}TTA_N{N}_M{M}'
    
    return results_dict, dict_name

def apply_policy_and_get_predictions(data_loader, models, augment_transform, device, binary_classification=False):
    results = []
    
    # Predict for each sample in the test set
    for i, batch in enumerate(data_loader):
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

def greedy_search(initial_aug_idx, val_preds, good_idx, bad_idx, select_only):
    """
    A single greedy search instance that starts from a random initial augmentation (initial_aug_idx).
    """
    group_indices = [initial_aug_idx]  # Initialize with the given augmentation

    for new_member_i in range(select_only):
        print(f"Evaluating policy {new_member_i+1}/{select_only}...", flush=True)
        best_metric = -np.inf
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

            if roc_auc > 0.5 and roc_auc > best_metric:
                best_s = new_i
                best_metric = roc_auc

        group_indices.append(best_s)
        print(f"Selected Policy {best_s}: roc_auc={best_metric:.4f}")

    return best_metric, group_indices

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
    best_metric, best_group_indices = best_result

    print("\nParallel greedy search complete. Best metric:", best_metric)
    return np.array(best_group_indices)

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

def perform_greedy_policy_search(npz_dir, good_idx, bad_idx, select_by='roc_auc', select_only=50, num_workers=1, num_searches=10):
    """
    Perform parallel greedy policy search using the select_greedily_on_ens function.
    Loads .npz files, aggregates predictions, and performs policy search.
    """
    print('load predictions')
    # Step 1: Load the .npz files containing predictions
    all_preds, all_keys = load_npz_files_for_greedy_search(npz_dir)
    search_set_len = all_preds[0].size

    # Step 3: Call the `select_greedily_on_ens` function with loaded predictions
    selected_policies = select_greedily_on_ens(
        all_preds,  # Predictions from npz files
        good_idx,
        bad_idx,
        search_set_len=search_set_len,
        select_only=select_only,  # Size of the dataset to be used for searching
        num_workers=num_workers,  # Number of workers for parallel processing
        num_searches=num_searches  # Number of parallel greedy search processes
    )

    # Return the selected policies and their corresponding names
    selected_policy_names = [all_keys[i] for i in selected_policies]
    return selected_policy_names
