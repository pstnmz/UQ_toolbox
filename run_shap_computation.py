import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle
import numpy as np
import UQ_toolbox as uq
from medMNIST.utils import train_load_datasets_resnet as tr
import torch
import os

# Top-level callable instead of lambda (picklable)
class RepeatGrayToRGB:
    def __call__(self, x):
        return x.repeat(3, 1, 1)
class ClassifierHeadWrapper(nn.Module):
    def __init__(self, model):
        super(ClassifierHeadWrapper, self).__init__()
        self.fc = model.fc  # Replace 'fc2' with the appropriate layer name

    def forward(self, x):
        return self.fc(x)

def compute_shap_for_fold(fold, device_str, results, color, dataflag, size, batch_size):
    print(f'fold n{fold}')
    device = torch.device(device_str)
    if color is True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        
    else:
        # For grayscale images, repeat the single channel to make it compatible with ResNet
        # ResNet expects 3 channels, so we repeat the single channel image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
            RepeatGrayToRGB(),  # instead of transforms.Lambda(lambda x: x.repeat(3,1,1))
        ])
        
    model = tr.load_models(dataflag, device=device)[fold]
    [train_dataset, calibration_dataset, test_dataset], [train_loader, calibration_loader, test_loader], _ = tr.load_datasets(dataflag, color, size, transform, batch_size)
    
    model = model.to(device).eval()

    classifier_head = ClassifierHeadWrapper(model).to(device)
    shap_values, shap_features, labels, success = uq.extract_latent_space_and_compute_shap_importance(
        model=model,
        data_loader=calibration_loader,
        device=device,
        layer_to_be_hooked=model.avgpool,
        classifierheadwrapper=classifier_head,
        max_background_samples=1000
    )
    results[fold] = (shap_features, shap_values, success.squeeze() if success.ndim > 1 else success)


if __name__ == '__main__':
    mp.set_start_method('spawn')

    dataflag = 'organamnist'
    color = False # True for color, False for grayscale
    activation = 'softmax'
    batch_size = 4000
    im_size = 224
    size = 224  # Image size for the models
    batch_size = 4000  # Batch size for the DataLoader

    print(f"Processing {dataflag} with color={color} and activation={activation}")
    
    latent_spaces = []
    shap_values_folds = []
    success_folds = []

    # Create a manager to store results
    manager = mp.Manager()
    results = manager.dict()

    # Define the devices for each fold
    devices = ['cuda:0', 'cuda:1', 'cuda:0', 'cuda:0', 'cuda:1']

    # Create processes for folds 0, 1, 2
    processes = []
    for fold in range(3):
        p = mp.Process(target=compute_shap_for_fold, args=(fold, devices[fold], results, color, dataflag, size, batch_size))
        p.start()
        processes.append(p)

    # Wait for processes to finish
    for p in processes:
        p.join()

    # Process folds 3 and 4 sequentially
    for fold in range(3, 5):
        compute_shap_for_fold(fold, devices[fold], results, color, dataflag, size, batch_size)

    # Collect results
    for fold in range(5):
        shap_features, shap_values, success = results[fold]
        latent_spaces.append(shap_features)
        shap_values_folds.append(shap_values)
        success_folds.append(success)
    folder_name = f'/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/shap/{im_size}*{im_size}'
    os.makedirs(folder_name, exist_ok=True)
    # Save results to a file
    with open(f'/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/shap/{im_size}*{im_size}/shap_results_calibration_{dataflag}.pkl', 'wb') as f:
        pickle.dump({
            'latent_spaces': latent_spaces,
            'shap_values_folds': shap_values_folds,
            'success_folds': success_folds,
            'results': dict(results)  # Save the results dictionary as well
        }, f)