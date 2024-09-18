import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import seaborn as sns


def get_prediction(model, image, device):
    image = image.to(device)
    prediction = model(image).cpu().detach().numpy()
    
    return prediction

def TTA(transforms, model, image, device, nb_augmentations=10):
    tta_predictions = [get_prediction(model, transforms(image), device) for _ in range(nb_augmentations)]
    
    return tta_predictions

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
    
def distance_to_gold_std_plot(correct_predictions_distances, incorrect_predictions_distances):
    df = pd.DataFrame({
        'Distance': correct_predictions_distances + incorrect_predictions_distances,
        'Category': ['Correct Results'] * len(correct_predictions_distances) + ['Incorrect Results'] * len(incorrect_predictions_distances)
        })
    
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(x='Category', y='Distance', data=df, palette='muted')
    sns.swarmplot(x='Category', y='Distance', data=df, color='k', alpha=0.3)
    # Show the plot
    plt.title('Distance with gold standard')
    plt.show()
    
def ensembling_std_plot(correct_predictions_stds, incorrect_predictions_stds):
    df = pd.DataFrame({
        'Stds': correct_predictions_stds + incorrect_predictions_stds,
        'Category': ['Correct Results'] * len(correct_predictions_stds) + ['Incorrect Results'] * len(incorrect_predictions_stds)
        })
    
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    sns.boxplot(x='Category', y='Stds', data=df, palette='muted')
    sns.swarmplot(x='Category', y='Stds', data=df, color='k', alpha=0.3)
    # Show the plot
    plt.title('Standard deviations ensembling')
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
        