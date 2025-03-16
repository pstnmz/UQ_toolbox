import sys
import os

import medmnist
from medmnist import INFO
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18
import UQ_toolbox as uq
import pickle

# Define ResNet18 model
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class ClassifierHeadWrapper(nn.Module):
    def __init__(self, model):
        super(ClassifierHeadWrapper, self).__init__()
        self.fc = model.fc  # Replace 'fc2' with the appropriate layer name

    def forward(self, x):
        return self.fc(x)

def repeat_channels(x):
    return x.repeat(3, 1, 1)

def compute_shap_for_fold(fold, model, test_loader, device, results):
    print(f'fold n{fold}')
    classifier_head = ClassifierHeadWrapper(model.model).to(device)
    shap_values, shap_features, labels, success = uq.extract_latent_space_and_compute_shap_importance(
        model=model.to(device),
        data_loader=test_loader,
        device=device,
        layer_to_be_hooked=model.model.avgpool,
        classifierheadwrapper=classifier_head,
        max_background_samples=1000
    )
    results[fold] = (shap_features, shap_values, success.squeeze() if success.ndim > 1 else success)

if __name__ == '__main__':
    mp.set_start_method('spawn')

    # Load organAMNIST dataset
    data_flag = 'organamnist'
    download = True
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    num_classes = len(info['label'])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(repeat_channels)  # Use named function instead of lambda
    ])

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

    # Create DataLoaders for the new training and calibration datasets
    calibration_loader = DataLoader(dataset=calibration_dataset, batch_size=128, shuffle=False)

    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Calibration dataset size: {len(calibration_dataset)}')
    
    # Load saved models
    models = []
    for i in range(5):
        model = ResNet18(num_classes=num_classes)
        model.load_state_dict(torch.load(f'/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/resnet18_organamnist{i}.pt'))
        model.eval()
        models.append(model)
    latent_spaces = []
    shap_values_folds = []
    success_folds = []

    # Create a manager to store results
    manager = mp.Manager()
    results = manager.dict()

    # Define the devices for each fold
    devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:0', 'cuda:1']

    # Create processes for folds 0, 1, 2
    processes = []
    for fold in range(3):
        p = mp.Process(target=compute_shap_for_fold, args=(fold, models[fold], calibration_loader, devices[fold], results))
        p.start()
        processes.append(p)

    # Wait for processes to finish
    for p in processes:
        p.join()

    # Process folds 3 and 4 sequentially
    for fold in range(3, 5):
        compute_shap_for_fold(fold, models[fold], calibration_loader, devices[fold], results)

    # Collect results
    for fold in range(5):
        shap_features, shap_values, success = results[fold]
        latent_spaces.append(shap_features)
        shap_values_folds.append(shap_values)
        success_folds.append(success)

    # Save results to a file
    with open('/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/shap_results_calibration.pkl', 'wb') as f:
        pickle.dump({
            'latent_spaces': latent_spaces,
            'shap_values_folds': shap_values_folds,
            'success_folds': success_folds,
            'results': dict(results)  # Save the results dictionary as well
        }, f)