import sys
import os

# Get the absolute path to the root directory where UQ_toolbox.py is located
root_dir = os.path.abspath(os.path.join(os.path.dirname('medMNIST'), '..'))
sys.path.append(root_dir)
import medmnist
from adjustText import adjust_text
from medmnist import INFO, Evaluator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
import UQ_toolbox as uq
import pickle as pkl
import pandas as pd
from PIL import Image
import umap.umap_ as umap
from medMNIST.utils import train_load_datasets_resnet as tr

def test_eval(test_loader, device, models, data_flag):
    info = INFO[data_flag]
    task_type = info['task']  # Determine the task type (binary-class or multi-class)
    num_classes = len(info['label'])  # Number of classes

    # Perform inference on the test set
    y_true = []
    y_scores = []
    y_raw_digits = []
    indiv_scores = [[] for _ in range(len(models))]  # Store individual model scores

    for m in models:
        m.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = [model(data) for model in models]

            # Store individual model scores
            for i, output in enumerate(outputs):
                if task_type == 'binary-class':
                    indiv_scores[i].extend(F.sigmoid(output).cpu().numpy().flatten())
                else:
                    indiv_scores[i].extend(F.softmax(output, dim=1).cpu().numpy())

            # Average the outputs for ensemble prediction
            avg_output = torch.mean(torch.stack(outputs), dim=0)
            y_true.extend(target.cpu().numpy().flatten())
            if task_type == 'binary-class':
                avg_output_sig = F.sigmoid(avg_output)
                y_scores.extend(avg_output_sig.cpu().numpy().flatten())
                y_raw_digits.extend(avg_output.cpu().numpy().flatten())
            else:
                y_scores.extend(avg_output.cpu().numpy())
                y_raw_digits.extend(avg_output.cpu().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_scores_raw_digits = np.array(y_raw_digits)

    if task_type == 'binary-class':
        y_pred = (y_scores > 0.5).astype(int)
        auc = roc_auc_score(y_true, y_scores)
    else:
        y_pred = np.argmax(y_scores, axis=1)
        # Calculate metrics
        auc = roc_auc_score(y_true, apply_softmax(y_scores), multi_class='ovr')
    acc = accuracy_score(y_true, y_pred)
    print(f'Ensemble AUC: {auc:.3f}, Ensemble Accuracy: {acc:.3f}')

    # Generate the confusion matrix
    if task_type == 'binary-class':
        cm = confusion_matrix(y_true, y_pred)
    else:
        cm = confusion_matrix(y_true, y_scores.argmax(axis=1))

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=info['label'].values(), yticklabels=info['label'].values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Compute balanced accuracy
    if task_type == 'binary-class':
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
    else:
        balanced_acc = balanced_accuracy_score(y_true, y_scores.argmax(axis=1))

    # Compute sensitivity (recall)
    if task_type == 'binary-class':
        sensitivity = recall_score(y_true, y_pred, average='binary')
    else:
        sensitivity = recall_score(y_true, y_scores.argmax(axis=1), average='macro')
    

    # Compute specificity
    specificities = []
    for i in range(num_classes):
        if task_type == 'binary-class':
            cm = confusion_matrix(y_true, y_pred)
        else:
            cm = confusion_matrix(y_true, y_scores.argmax(axis=1))
        FP = np.sum(cm[:, i]) - cm[i, i]  # False Positives
        TN = np.sum(cm) - (np.sum(cm[i, :]) + FP)  # True Negatives
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 1.0
        specificities.append(specificity)

    macro_specificity = np.mean(specificities)

    perfs = {
        'auc': auc,
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': macro_specificity,
        'confusion_matrix': cm
    }
    print(perfs)
    return y_true, y_scores, y_scores_raw_digits, indiv_scores, perfs

def apply_softmax(y):
    y_scores = np.array(F.softmax(torch.tensor(y), dim=1))
    return y_scores

def train_val_loaders(train_dataset, batch_size):
    # Create stratified K-fold cross-validator
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Get the labels for stratification
    labels = [label for _, label in train_dataset]

    # Create a list to store the new dataloaders
    train_loaders = []
    val_loaders = []

    for train_index, val_index in skf.split(np.zeros(len(labels)), labels):
        train_subset = torch.utils.data.Subset(train_dataset, train_index)
        val_subset = torch.utils.data.Subset(train_dataset, val_index)
        
        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    return train_loaders, val_loaders

def find_best_threshold_and_compute_metrics(values, correct_predictions, optimization_metric='balanced_accuracy'):
    """
    Find the best threshold and compute metrics.

    Args:
        values (numpy.ndarray): UQ values.
        correct_predictions (list): Indices of correct predictions.
        optimization_metric (str): Metric to optimize ('balanced_accuracy', 'sensitivity', 'specificity').

    Returns:
        None
    """
    # Function to compute metrics
    def compute_metrics(uq_values, labels, threshold):
        predictions = (uq_values <= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        balanced_acc = balanced_accuracy_score(labels, predictions)
        return tn, fp, fn, tp, specificity, sensitivity, balanced_acc

    # Update the function to handle ties in specificity or sensitivity
    def find_optimal_threshold(uq_values, labels, metric):
        thresholds = np.linspace(min(uq_values), max(uq_values), 1000)
        best_threshold = thresholds[0]
        best_metric_value = 0
        best_secondary_metric_value = 0

        for threshold in thresholds:
            _, _, _, _, specificity, sensitivity, balanced_acc = compute_metrics(uq_values, labels, threshold)

            if metric == 'balanced_accuracy' and balanced_acc > best_metric_value:
                best_metric_value = balanced_acc
                best_threshold = threshold
            elif metric == 'sensitivity':
                if sensitivity > best_metric_value or (sensitivity == best_metric_value and specificity > best_secondary_metric_value):
                    best_metric_value = sensitivity
                    best_secondary_metric_value = specificity
                    best_threshold = threshold
            elif metric == 'specificity':
                if specificity > best_metric_value or (specificity == best_metric_value and sensitivity > best_secondary_metric_value):
                    best_metric_value = specificity
                    best_secondary_metric_value = sensitivity
                    best_threshold = threshold

        return best_threshold

    # Find the optimal threshold
    labels = np.array([1 if i in correct_predictions else 0 for i in range(len(values))])
    optimal_threshold = find_optimal_threshold(values, labels, optimization_metric)

    # Compute the confusion matrix using the optimal threshold
    predictions = (values <= optimal_threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    
    # Display the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 36}, xticklabels=['Failure', 'Success'], yticklabels=['Failure', 'Success'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Print the optimal threshold and metrics
    _, _, _, _, specificity, sensitivity, balanced_acc = compute_metrics(values, labels, optimal_threshold)
    
    print(f"Optimal Threshold: {optimal_threshold}")
    print(f"Balanced Accuracy: {balanced_acc}")
    print(f"Specificity: {specificity}")
    print(f"Sensitivity: {sensitivity}")

    return balanced_acc

def computeMSR(y_prob, y_true, task_type, calibration_needed=False, method_calibration=None, display_calibration_curve=False, y_scores_calibration=None, y_true_calibration=None):
    if calibration_needed:
            # Perform post-hoc calibration using the calibration dataset
            calibrated_scores, calibration_model = uq.posthoc_calibration(y_scores_calibration, y_true_calibration, method_calibration)
            logits_tensor = torch.from_numpy(y_prob).float()
            if method_calibration == 'temperature':
                if logits_tensor.ndim == 1:
                    logits_tensor = logits_tensor.unsqueeze(1)
                logits_test_scaled = calibration_model(logits_tensor).squeeze()

                if logits_test_scaled.ndim == 1 or logits_test_scaled.shape[1] == 1:
                    # Binary case → use sigmoid
                    calibrated_test_scores = torch.sigmoid(logits_test_scaled).detach().numpy().squeeze().squeeze()
                else:
                    # Multiclass → use softmax
                    calibrated_test_scores = torch.nn.functional.softmax(logits_test_scaled, dim=1).detach().numpy()

            else:
                if logits_tensor.ndim == 1:
                    logits_tensor = logits_tensor.unsqueeze(1)
                calibrated_test_scores = calibration_model.predict_proba(logits_tensor.reshape(-1, 1))[:, 1]

            metric = uq.distance_to_hard_labels_computation(calibrated_test_scores)
            if display_calibration_curve:
                uq.model_calibration_plot(y_true, calibrated_test_scores)
    else:
        metric = uq.distance_to_hard_labels_computation(y_prob)
        if display_calibration_curve:
            uq.model_calibration_plot(y_true, y_prob)
    
    return metric

def compute_ensembling(indiv_scores):
    metric = uq.ensembling_stds_computation(indiv_scores)
    return metric
    
def computeKNNshap(models, train_loaders, test_loader, device, num_classes=None, latent_spaces=None, shap_values_folds=None, labels_folds=None, shap=True):
    if shap:
        mean_shap_importances = []
        if num_classes == 2:
            for fold, (shap_values, labels) in enumerate(zip(shap_values_folds, labels_folds)):
                mean_shap_fold = uq.compute_mean_shap_values(shap_values, fold, labels, 50)
                mean_shap_importances.append(mean_shap_fold)
        else:
            for fold, shap_values in enumerate(shap_values_folds):
                mean_shap_fold = uq.compute_mean_shap_values(shap_values, fold, true_labels=None, nb_features=50)
                mean_shap_importances.append(mean_shap_fold)

        latent_spaces_df = []
        for fold, latent_space in enumerate(latent_spaces):
            num_samples, num_features = latent_space.shape
            latent_space_df = pd.DataFrame(
                latent_space,
                columns=[f"Feature_{i}" for i in range(num_features)]
            )
            latent_spaces_df.append(latent_space_df)

        plt.close('all')
        successes_folds = []
        knn_distances_folds = []

        for fold in range(5):
            knn_distances_all, successes_all = uq.compute_knn_distances_to_train_data(models[fold], train_loaders[fold], test_loader, models[fold].avgpool, device, latent_spaces_df[fold], mean_shap_importances[fold], num_classes)
            knn_distances_folds.append(knn_distances_all)
            successes_folds.append(successes_all)
        
        # Calculate the mean of the lists inside knn_distances_folds
        metric = np.mean(knn_distances_folds, axis=0)

    else:
        knn_distances_all = []
        for fold in range(5):
            latent_space_training, _, _, _ = uq.extract_latent_space_and_compute_shap_importance(
                model=models[fold],
                data_loader=train_loaders[fold],
                device=device,
                layer_to_be_hooked=models[fold].avgpool,
                importance=False
            )
            
            latent_space_test, _, _, _ = uq.extract_latent_space_and_compute_shap_importance(
                model=models[fold],
                data_loader=test_loader,
                device=device,
                layer_to_be_hooked=models[fold].avgpool,
                importance=False
            )
            
            train_latent_space = pd.DataFrame(latent_space_training)
            test_latent_space = pd.DataFrame(latent_space_test)
                
            # Print the initial number of dimensions
            print(f"Initial number of dimensions: {train_latent_space.shape[1]}")
                
            scaler = StandardScaler()
            train_latent_space_standardized = scaler.fit_transform(train_latent_space)
            
            pca = PCA(n_components=0.8)
            train_latent_space_pca = pca.fit_transform(train_latent_space_standardized)
            
            # Print the number of dimensions after PCA
            print(f"Number of dimensions after PCA: {train_latent_space_pca.shape[1]}")
            
            test_latent_space_standardized = scaler.transform(test_latent_space)
            test_latent_space_pca = pca.transform(test_latent_space_standardized)
            
            knn = NearestNeighbors(n_neighbors=5)
            knn.fit(train_latent_space_pca)
            distances, _ = knn.kneighbors(test_latent_space_pca)
            
            average_distances = distances.mean(axis=1)
            
            knn_distances_all.append(average_distances)
            # Calculate the mean of the lists inside knn_distances_folds
            metric = np.mean(knn_distances_all, axis=0)

    return metric

def remove_black_borders(pil_img, padding=2):
    """Removes black borders from a PIL image and resizes back to original size."""
    img_np = np.array(pil_img)
    
    # Handle grayscale (2D) or RGB (3D)
    if img_np.ndim == 2:
        mask = img_np != 0
    else:
        mask = np.any(img_np != 0, axis=2)

    if not np.any(mask):
        return pil_img  # Totally black

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Pad the crop box
    y0 = max(y0 - padding, 0)
    x0 = max(x0 - padding, 0)
    y1 = min(y1 + padding, pil_img.height)
    x1 = min(x1 + padding, pil_img.width)

    cropped = pil_img.crop((x0, y0, x1, y1))
    return cropped.resize((pil_img.width, pil_img.height), resample=Image.BILINEAR)

def computeTTA(aug_type, models, test_dataset, device, num_classes=2, correct_predictions_calibration=None, incorrect_predictions_calibration=None, image_normalization=False, aug_folder=None, mean=[0.5], std=[0.5], max_iterations=10, gps_augment=None, batch_size=None, color=False, im_size=None):
    if num_classes == 2:
        output_activation='sigmoid'
    else:
        output_activation='softmax'
    if aug_type == 'randaugment':
        # Original RandAugment (may include geometric transforms)
        transformation_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandAugment(num_ops=2, magnitude=9, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            *([transforms.Normalize(mean=mean, std=std)] if image_normalization else []),
            *([transforms.Lambda(lambda x: x.repeat(3, 1, 1))] if color is False else [])
        ])
        metric, global_preds = uq.TTA(transformation_pipeline, models, test_dataset, device, nb_augmentations=5, nb_channels=3, output_activation=output_activation, usingBetterRandAugment=False, mean=mean, std=std, batch_size=batch_size)
    
    elif aug_type == 'randaugment_without_geometric_transforms':
        
        transformation_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandAugment(num_ops=2, magnitude=9, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Lambda(remove_black_borders),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            *([transforms.Normalize(mean=mean, std=std)] if image_normalization else []),
            *([transforms.Lambda(lambda x: x.repeat(3, 1, 1))] if color is False else [])
        ])
        metric, global_preds = uq.TTA(transformation_pipeline, models, test_dataset, device, nb_augmentations=5, nb_channels=3, output_activation=output_activation, usingBetterRandAugment=False, mean=mean, std=std, batch_size=batch_size)
        
    elif aug_type == 'crops_flips':
        transformation_pipeline = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=20, scale=(0.8, 1.0)),  # Random crop with resizing
            transforms.RandomHorizontalFlip(p=0.5),                   # Random horizontal flip
            transforms.RandomRotation(degrees=180),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
        metric, global_preds = uq.TTA(transformation_pipeline, models, test_dataset, device, nb_augmentations=5, nb_channels=3, output_activation=output_activation, usingBetterRandAugment=False, mean=mean, std=std, batch_size=batch_size)
        
    elif aug_type == 'GPS':
        if gps_augment is None:
            best_aug = uq.perform_greedy_policy_search(aug_folder, correct_predictions_calibration, incorrect_predictions_calibration, num_workers=90, max_iterations=max_iterations, num_searches=30, top_k=5, plot=True)
            if isinstance(best_aug, list) and all(isinstance(policy, list) for policy in best_aug):
                transformation_pipeline = []
                for aug in best_aug:
                    n, m, transformations = uq.extract_gps_augmentations_info(aug)
                    transformation_pipeline.append(transformations)
        else:
            n = gps_augment[0]
            m = gps_augment[1]
            transformation_pipeline = gps_augment[2]
        metric, global_preds_GPS = uq.TTA(transformation_pipeline, models, test_dataset, device, usingBetterRandAugment=True, n=n, m=m, nb_channels=3, image_size=im_size, image_normalization=image_normalization, output_activation=output_activation, mean=mean, std=std, batch_size=batch_size)
    
    return metric

def display_UQ_results(metric, correct_predictions, incorrect_predictions, y_axis_title, title, optim_metric, swarmplot=False):
    balanced_acc = find_best_threshold_and_compute_metrics(metric, correct_predictions, optim_metric)
    fpr, tpr, auc = uq.roc_curve_UQ_method_computation([metric[k] for k in correct_predictions], [metric[j] for j in incorrect_predictions])
    uq.UQ_method_plot([metric[k] for k in correct_predictions], [metric[j] for j in incorrect_predictions], y_axis_title, title, swarmplot)
    print(auc)
    return auc, balanced_acc

def call_UQ_methods(
    methods,
    models=None,
    y_prob=None,
    digits=None,
    y_true=None,
    digits_calib=None,
    y_true_calibration=None,
    indiv_scores=None,
    task_type=None,
    correct_predictions=None,
    incorrect_predictions=None,
    test_loader=None,
    device=None,
    optim_metric=None,
    train_loaders=None,
    test_dataset_tta=None,
    num_classes=None,
    latent_spaces=None,
    shap_values_folds=None,
    labels_fold=None,
    correct_predictions_calibration=None,
    incorrect_predictions_calibration=None,
    image_normalization=False,
    color=False,
    aug_folder=None,
    gps_augment=None,
    max_iteration=10,
    swarmplot=True,
    calib_method='temperature',
    batch_size=None,
    image_size=None
):
    metrics = []
    methods = methods or []
    aucs = []
    balanced_acc = []

    for method in methods:
        if task_type == 'binary-class':
            y_axis = 'min(y_pred, 1-y_pred)'
            title = 'DHL'
        else:
            y_axis = '1-y_pred'
            title = 'MSR'
        if method == 'MSR':
            if y_prob is not None and y_true is not None and task_type is not None:
                metric = computeMSR(y_prob, y_true, task_type, calibration_needed=False, display_calibration_curve=True)
                auc, b_acc = display_UQ_results(metric, correct_predictions, incorrect_predictions, y_axis, title, optim_metric=optim_metric, swarmplot=swarmplot)
                aucs.append((method, auc))
                balanced_acc.append((method, b_acc))
                metrics.append((method, metric))
        elif method == 'MSR_temp_scale':
            if digits is not None and y_true is not None and task_type is not None and digits_calib is not None and y_true_calibration is not None:
                metric = computeMSR(digits, y_true, task_type, calibration_needed=True, display_calibration_curve=True, method_calibration=calib_method, y_scores_calibration=digits_calib, y_true_calibration=y_true_calibration)
                auc, b_acc = display_UQ_results(metric, correct_predictions, incorrect_predictions, y_axis, title + ' after temperature scaling', optim_metric=optim_metric, swarmplot=swarmplot)
                aucs.append((method, auc))
                balanced_acc.append((method, b_acc))
                metrics.append((method, metric))
        elif method == 'Ensembling':
            if indiv_scores is not None:
                metric = compute_ensembling(indiv_scores)
                auc, b_acc = display_UQ_results(metric, correct_predictions, incorrect_predictions, 'std', 'Ensembling Results', optim_metric=optim_metric, swarmplot=swarmplot)
                aucs.append((method, auc))
                balanced_acc.append((method, b_acc))
                metrics.append((method, metric))
        elif method == 'TTA':
            if models is not None and test_dataset_tta is not None and device is not None:
                metric = computeTTA('randaugment', models, test_dataset_tta, device, num_classes=num_classes, image_normalization=image_normalization, batch_size=batch_size, color=color)
                auc, b_acc = display_UQ_results(metric, correct_predictions, incorrect_predictions, 'std', 'TTA', optim_metric=optim_metric, swarmplot=swarmplot)
                aucs.append((method, auc))
                balanced_acc.append((method, b_acc))
                metrics.append((method, metric))
        elif method == 'TTA_without_geometric_transforms':
            if models is not None and test_dataset_tta is not None and device is not None:
                metric = computeTTA('randaugment_without_geometric_transforms', models, test_dataset_tta, device, num_classes=num_classes, image_normalization=image_normalization, batch_size=batch_size, color=color)
                auc, b_acc = display_UQ_results(metric, correct_predictions, incorrect_predictions, 'std', 'TTA_no_geom_transforms', optim_metric=optim_metric, swarmplot=swarmplot)
                aucs.append((method, auc))
                balanced_acc.append((method, b_acc))
                metrics.append((method, metric))
        elif method == 'GPS':
            if models is not None and test_dataset_tta is not None and device is not None:
                metric = computeTTA('GPS', models, test_dataset_tta, device, num_classes=num_classes, correct_predictions_calibration=correct_predictions_calibration, incorrect_predictions_calibration=incorrect_predictions_calibration, image_normalization=image_normalization, aug_folder=aug_folder, max_iterations=max_iteration, gps_augment=gps_augment, im_size=image_size, batch_size=batch_size)
                auc, b_acc = display_UQ_results(metric, correct_predictions, incorrect_predictions, 'std', 'GPS', optim_metric=optim_metric, swarmplot=swarmplot)
                aucs.append((method, auc))
                balanced_acc.append((method, b_acc))
                metrics.append((method, metric))
        elif method == 'KNNshap':
            if models is not None and train_loaders is not None and test_loader is not None and device is not None and num_classes is not None and latent_spaces is not None and shap_values_folds is not None:
                metric = computeKNNshap(models, train_loaders, test_loader, device, num_classes, latent_spaces, shap_values_folds, labels_folds=labels_fold, shap=True)
                auc, b_acc = display_UQ_results(metric, correct_predictions, incorrect_predictions, 'KNN distances after features selection', 'KNNshap', optim_metric=optim_metric, swarmplot=swarmplot)
                aucs.append((method, auc))
                balanced_acc.append((method, b_acc))
                metrics.append((method, metric))
        elif method == 'KNNall':
            if models is not None and train_loaders is not None and test_loader is not None and device is not None:
                metric = computeKNNshap(models, train_loaders, test_loader, device, shap=False)
                auc, b_acc = display_UQ_results(metric, correct_predictions, incorrect_predictions, 'KNN distances', 'KNNall', optim_metric=optim_metric, swarmplot=swarmplot)
                aucs.append((method, auc))
                balanced_acc.append((method, b_acc))
                metrics.append((method, metric))
    return metrics, aucs, balanced_acc

flags = ['breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist']
calib_method = ['platt', 'temperature', 'platt', 'temperature', 'temperature', 'temperature', 'temperature', 'temperature']
colors = [False, False, False, True, False, True, True, False]  # Colors for the flags
activations = ['sigmoid', 'softmax', 'sigmoid', 'softmax', 'softmax', 'softmax', 'softmax', 'softmax']  # Output activations for each flag
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
uq_methods = ['GPS']#, 'KNNshap', 'KNNall']  
size = 224  # Image size for the models
batch_size = 4000  # Batch size for the DataLoader
model_global_perfs = {}

for flag, color, activation, calib_method in zip(flags, colors, activations, calib_method):
    print(f"Processing {flag} with color={color} and activation={activation}")
    if color is True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        
        transform_tta = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        # For grayscale images, repeat the single channel to make it compatible with ResNet
        # ResNet expects 3 channels, so we repeat the single channel image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        
        transform_tta = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
    models = tr.load_models(flag, device=device)
    [train_dataset, calibration_dataset, test_dataset], [train_loader, calibration_loader, test_loader], info = tr.load_datasets(flag, color, size, transform, batch_size)
    train_loaders, val_loaders = train_val_loaders(train_dataset, batch_size=batch_size)
    task_type = info['task']  # Determine the task type (binary-class or multi-class)
    num_classes = len(info['label'])  # Number of classes
    [_, calibration_dataset_tta, test_dataset_tta], [_, calibration_loader_tta, test_loader_tta], _ = tr.load_datasets(flag, color, size, transform_tta, batch_size)

    y_true, y_scores, digits, indiv_scores, performances = test_eval(test_loader, device=device, models=models, data_flag=flag)
    y_true_calibration, y_scores_calibration, digits_calib, indiv_scores_calib, performances_calib = test_eval(calibration_loader, device=device, models=models, data_flag=flag)

    if task_type == 'binary-class':
        y_prob = y_scores
        y_prob_calibration = y_scores_calibration
    else:
        y_prob = apply_softmax(y_scores)
        y_prob_calibration = apply_softmax(y_scores_calibration)

    if task_type == 'binary-class':
        correct_predictions = [i for i in range(len(y_true)) if (y_true[i] == 1 and y_scores[i] > 0.5) or (y_true[i] == 0 and y_scores[i] <= 0.5)]
        incorrect_predictions = [i for i in range(len(y_true)) if (y_true[i] == 1 and y_scores[i] <= 0.5) or (y_true[i] == 0 and y_scores[i] > 0.5)]

        correct_predictions_calibration = [i for i in range(len(y_true_calibration)) if (y_true_calibration[i] == 1 and y_scores_calibration[i] > 0.5) or (y_true_calibration[i] == 0 and y_scores_calibration[i] <= 0.5)]
        incorrect_predictions_calibration = [i for i in range(len(y_true_calibration)) if (y_true_calibration[i] == 1 and y_scores_calibration[i] <= 0.5) or (y_true_calibration[i] == 0 and y_scores_calibration[i] > 0.5)]
    else:
        correct_predictions = [i for i in range(len(y_true)) if y_true[i] == np.argmax(y_scores[i])]
        incorrect_predictions = [i for i in range(len(y_true)) if y_true[i] != np.argmax(y_scores[i])]

        correct_predictions_calibration = [i for i in range(len(y_true_calibration)) if y_true_calibration[i] == np.argmax(y_scores_calibration[i])]
        incorrect_predictions_calibration = [i for i in range(len(y_true_calibration)) if y_true_calibration[i] != np.argmax(y_scores_calibration[i])]
    break

uq_metrics, aucs, balanced_acc = call_UQ_methods(uq_methods, models, y_prob, digits, y_true, digits_calib, y_true_calibration, indiv_scores, task_type, correct_predictions, incorrect_predictions, test_loader, device, optim_metric='balanced_accuracy', train_loaders=train_loaders, test_dataset_tta=test_dataset_tta, num_classes=num_classes, image_normalization=True, swarmplot=False, calib_method=calib_method, batch_size=batch_size, color=color, aug_folder=f'/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/gps_augment/{size}*{size}/{flag}_calibration_set', correct_predictions_calibration=correct_predictions_calibration, incorrect_predictions_calibration=incorrect_predictions_calibration, image_size=size)