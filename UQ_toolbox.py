import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import seaborn as sns


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
        