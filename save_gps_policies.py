import sys
import os
import medmnist
from medmnist import INFO, Evaluator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, accuracy_score, recall_score
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import UQ_toolbox as uq

def load_models(flag):

    # Load organAMNIST dataset
    data_flag = flag
    info = INFO[data_flag]
    num_classes = len(info['label'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load saved models
    models = []
    for i in range(5):
        # Initialize the model
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        if num_classes == 2:
            model.fc = nn.Linear(model.fc.in_features, 1)  # Output 1 value for binary classification
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)  # Output logits for each class
        
        # Load the state dictionary
        state_dict = torch.load(f'/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/resnet18_{flag}{i}.pt')

        # Remove the 'model.' prefix from the state_dict keys if necessary
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # Load the modified state dictionary into the model
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
    return models

def load_datasets(flag, transform):
    transform=transform
    
    # Load organAMNIST dataset
    data_flag = flag
    download = True
    info = INFO[data_flag]
    task_type = info['task']  # Determine the task type (binary-class or multi-class)
    DataClass = getattr(medmnist, info['python_class'])


    test_dataset = DataClass(split='test', download=download, transform=transform)
    train_dataset = DataClass(split='train', download=download, transform=transform)
    val_dataset = DataClass(split='val', download=download, transform=transform)

    # Combine train_dataset and val_dataset
    combined_train_dataset = ConcatDataset([train_dataset, val_dataset])

    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Calculate the sizes for training and calibration datasets
    train_size = int(0.8 * len(combined_train_dataset))
    calibration_size = len(combined_train_dataset) - train_size

    # Split the combined_train_dataset into training and calibration datasets
    train_dataset, calibration_dataset = random_split(combined_train_dataset, [train_size, calibration_size])

    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Calibration dataset size: {len(calibration_dataset)}')

    return train_dataset, calibration_dataset, test_dataset, task_type


# Load organAMNIST dataset
flag = 'breastmnist'
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
models = load_models(flag)
train_dataset, calibration_dataset, test_dataset, task_type = load_datasets(flag, transform)
test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False)
calibration_loader=DataLoader(calibration_dataset, batch_size=32, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

uq.apply_randaugment_and_store_results(test_loader, models, 2, 45, 500, device, folder_name=f'/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/gps_augment_breastmnist', batch_norm=True, image_size=28, nb_channels=3, softmax_application=True)