import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from gps_augment.utils.randaugment import BetterRandAugment


def get_prediction(model, image, device):
    
    # plt.figure(figsize=(10, 8))
    # plt.imshow(image.squeeze(), cmap='Greys')
    # plt.show()
    image = image.to(device)
    prediction = model(image).cpu().detach().numpy()
    return prediction

def TTA(transforms, models, image, device, nb_augmentations=10):
    plt.close('all')
    if isinstance(models, list):
        tta_predictions = [np.mean([get_prediction(model, transforms(image), device) for model in models]) for _ in range(nb_augmentations)]
    else: 
        tta_predictions = [get_prediction(models, transforms(image), device) for _ in range(nb_augmentations)]
    
    std = np.std(tta_predictions)  
    
    return tta_predictions, std

def ensembling_predictions(models, image):
    ensembling_predictions = [get_prediction(model, image) for model in models]
    
    return ensembling_predictions
    
def distance_to_gold_standard_computation(predictions):
    distances = [0.5 - abs(pred - 0.5) for pred in predictions]
    
    return distances

def ensembling_stds_computation(models_predictions):
    stds = [np.std(row) for row in zip(*models_predictions)]
    
    return stds
    

def plot_calibration_curve(y_true, y_prob):
    # Calculate the calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    
    return prob_true, prob_pred
        
def model_calibration_plot(true_labels, predictions):
    plt.figure(figsize=(10, 8))

    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    i = 0

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
    
def GPS(model, test_set):
    gps_augment.get_predictions_randaugment.main(test_set, model)

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
    
def apply_randaugment(data_loader, model, N=2, M=5, num_tta=10, binary_classification=False):
    model.eval()
    all_predictions = []
    
    for _ in range(num_tta):
        augmented_loader = apply_better_randaugment(data_loader, N, M)
        preds = []
        
        for inputs, targets in augmented_loader:
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = model(inputs)
                if binary_classification:
                    preds.append(torch.sigmoid(outputs).cpu().numpy())
                else:
                    preds.append(torch.nn.functional.log_softmax(outputs, dim=1).cpu().numpy())
        
        all_predictions.append(np.vstack(preds))

    return np.mean(all_predictions, axis=0)  # average over TTA

def evaluate_policy(predictions, targets, binary_classification=False):
    """
    Evaluate policy based on calibrated log-likelihood, accuracy, and ROC AUC.
    
    Parameters:
    - predictions: np.array of shape (num_samples, num_classes or 1) after averaging TTA
    - targets: Ground truth labels.
    - binary_classification: True for binary classification, False for multi-class.

    Returns:
    - dict with 'calibrated_log_likelihood', 'accuracy', and 'roc_auc' scores.
    """
    
    # Calibrated Log-Likelihood
    if binary_classification:
        log_likelihood = -np.mean(np.log(np.clip(predictions, 1e-7, 1 - 1e-7)))  # Binary case
    else:
        log_likelihood = -np.mean(np.sum(targets * np.log(np.clip(predictions, 1e-7, 1 - 1e-7)), axis=1))  # Multi-class case

    # Accuracy
    if binary_classification:
        preds_labels = (predictions > 0.5).astype(int)
    else:
        preds_labels = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(targets, preds_labels)
    
    # ROC AUC based on SD of TTA predictions
    # Assuming you have multiple predictions from TTA stored, calculate SD for uncertainty measure
    stds = np.std(predictions, axis=0)  # SD of the predictions from multiple augmentations
    success_failure = np.abs(targets - preds_labels)  # Success = 0, Failure = 1

    if binary_classification:
        roc_auc = roc_auc_score(success_failure, stds)
    else:
        roc_auc = roc_auc_score(success_failure, stds, multi_class='ovr')  # One-vs-Rest for multi-class

    return {
        'calibrated_log_likelihood': log_likelihood,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    
def apply_better_randaugment(data_loader, N, M):
    # Apply BetterRandAugment to the dataset
    augment_transform = BetterRandAugment(N=N, M=M)
    data_loader.dataset.transform = augment_transform  # Modify the data loader with augmentation
    return data_loader

def gps_policy_search(model, data_loader, num_policies=5, binary_classification=False,  N=3, M=9):
    best_policy = None
    best_score = -np.inf

    for _ in range(num_policies):
        current_predictions = apply_randaugment(data_loader, model, N=N, M=M, binary_classification=binary_classification)
        
        # Evaluate policy performance using validation set, select best policy based on log-likelihood, accuracy, etc.
        # For simplicity, you can compare the performance using your metric of choice
        score = evaluate_policy(current_predictions, data_loader)  # Custom function for evaluation
        
        if score > best_score:
            best_score = score
            best_policy = (N, M)  # Store the best policy parameters

    return best_policy