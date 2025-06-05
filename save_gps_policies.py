import sys
import os
import medmnist
from medmnist import INFO, Evaluator
from medMNIST.utils import train_resnet as tr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, accuracy_score, recall_score
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import UQ_toolbox as uq

def load_models(flag, size=224):

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
        state_dict = torch.load(f'/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/models/{size}x{size}/resnet18_{flag}_{size}_{i}.pt')

        # Remove the 'model.' prefix from the state_dict keys if necessary
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # Load the modified state dictionary into the model
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
    return models

def load_datasets(dataflag, color, batch_size, im_size):
    
    if color:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])
            
            transform_for_tta = transforms.Compose([
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
        
        transform_for_tta = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])
        
    _, datasets, info = tr.get_data_loaders(dataflag, return_datasets=True, im_size=im_size, color=color, transform=transform_for_tta)
    # Combine train_dataset and val_dataset
    combined_train_dataset = ConcatDataset([datasets[0], datasets[1]])

    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Calculate the sizes for training and calibration datasets
    train_size = int(0.8 * len(combined_train_dataset))
    calibration_size = len(combined_train_dataset) - train_size

    # Split the combined_train_dataset into training and calibration datasets
    train_dataset, calibration_dataset = random_split(combined_train_dataset, [train_size, calibration_size])

    # Create DataLoaders for the new training and calibration datasets
    calibration_loader_for_tta = DataLoader(dataset=calibration_dataset, batch_size=batch_size, shuffle=False)

    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Calibration dataset size: {len(calibration_dataset)}')
    
    return calibration_loader_for_tta, calibration_dataset

dataflag = 'octmnist'
color = False # True for color, False for grayscale
activation = 'softmax'
batch_size = 4000
im_size = 224
models = load_models(dataflag)
calibration_loader_for_tta, calibration_dataset=load_datasets(dataflag, color, batch_size, im_size)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  

uq.apply_randaugment_and_store_results(calibration_loader_for_tta, models, 2, 45, 500, device, folder_name=f'/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/gps_augment/{im_size}*{im_size}/{dataflag}_calibration_set', image_normalization=True, mean=[.5], std=[.5], image_size=im_size, nb_channels=3, output_activation=activation, calibration_dataset=calibration_dataset, batch_size=batch_size)