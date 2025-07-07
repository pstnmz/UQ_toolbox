import UQ_toolbox as uq
from medMNIST.utils import train_load_datasets_resnet as tr
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torchvision.transforms as transforms


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

dataflag = 'bloodmnist'
color = True # True for color, False for grayscale
activation = 'softmax'
batch_size = 4000
im_size = 224
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
size = 224  # Image size for the models
batch_size = 4000  # Batch size for the DataLoader

print(f"Processing {dataflag} with color={color} and activation={activation}")
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
models = tr.load_models(dataflag, device=device)
[train_dataset, calibration_dataset, test_dataset], [train_loader, calibration_loader, test_loader], info = tr.load_datasets(dataflag, color, size, transform, batch_size)
train_loaders, val_loaders = train_val_loaders(train_dataset, batch_size=batch_size)
task_type = info['task']  # Determine the task type (binary-class or multi-class)
num_classes = len(info['label'])  # Number of classes
[_, calibration_dataset_tta, test_dataset_tta], [_, calibration_loader_tta, test_loader_tta], _ = tr.load_datasets(dataflag, color, size, transform_tta, batch_size)
uq.apply_randaugment_and_store_results(calibration_dataset, models, 2, 45, 500, device, folder_name=f'/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/gps_augment/{im_size}*{im_size}/{dataflag}_calibration_set', image_normalization=True, mean=[.5], std=[.5], image_size=im_size, nb_channels=3, output_activation=activation, batch_size=batch_size)