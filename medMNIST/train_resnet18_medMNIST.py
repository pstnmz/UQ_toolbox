from utils import train_load_datasets_resnet as tr
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import os, json, time

flags = ['breastmnist', 'organamnist', 'pneumoniamnist', 'dermamnist', 'octmnist', 'pathmnist', 'bloodmnist', 'tissuemnist']
colors = [False, False, False, True, False, True, True, False]  # Colors for the flags
batch_sizes = [32, 128, 128, 128, 640, 640, 640, 640]  # Batch sizes for the flags

for flag, color, batch_size in zip(flags, colors, batch_sizes):
    print(f"Training on {flag} with color={color} and batch_size={batch_size}")

    use_randaugment = True         # <- enable/disable RandAugment here
    randaugment_ops = 2            # number of ops per image
    randaugment_mag = 9            # magnitude (0-10 typical)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    size = 224  # Image size for the models

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join("/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/medMNIST/runs", flag, f"resnet18_{size}_{timestamp}_randaug{int(use_randaugment)}")
    os.makedirs(os.path.join(exp_dir, "figs"), exist_ok=True)

    if color is True:
        train_tfms = []
        if use_randaugment:
            train_tfms.append(transforms.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_mag))
        train_tfms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ]
        transform_train = transforms.Compose(train_tfms)

        transform_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        
    else:
        train_tfms = []
        if use_randaugment:
            train_tfms.append(transforms.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_mag))
        train_tfms += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]
        transform_train = transforms.Compose(train_tfms)

        transform_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

    # Load plain datasets/loaders (no augmentation)
    [train_dataset_plain, calibration_dataset, test_dataset], [_, calibration_loader, test_loader], info = tr.load_datasets(flag, color, size, transform_eval, batch_size)

    if use_randaugment:
        print(f'Using RandAugment with {randaugment_ops} ops and magnitude {randaugment_mag}')
        # Build an augmented view aligned to the same 80% train indices
        # 1) Recreate train+val with augmented transform
        aug_triplet, _ = tr.get_datasets(flag, im_size=size, color=color, transform=transform_train)
        combined_aug = ConcatDataset([aug_triplet[0], aug_triplet[1]])

        # 2) Wrap with the exact same indices as the 80% train subset
        train_dataset_aug = torch.utils.data.Subset(combined_aug, train_dataset_plain.indices)

        train_loaders, val_loaders = tr.CV_train_val_loaders(train_dataset_aug, train_dataset_plain, batch_size=batch_size)
    else:
        print('Not using RandAugment')
        train_loaders, val_loaders = tr.CV_train_val_loaders(None, train_dataset_plain, batch_size=batch_size)

    models = []
    results = []
    for i in range(5):
        print('MODEL ' + str(i))
        model, res = tr.train_resnet18(
            flag,
            train_loader=train_loaders[i],
            val_loader=val_loaders[i],
            test_loader=test_loader,
            num_epochs=2,
            learning_rate=0.0001,
            device='cuda:0',
            random_seed=42,
            output_dir=exp_dir,
            run_name=f"fold_{i}"
        )
        models.append(model)
        results.append(res)

    # Save per-fold results summary
    with open(os.path.join(exp_dir, "results_folds.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Evaluate ensemble and save
    ensemble_res = tr.evaluate_model(model=models, test_loader=test_loader, data_flag=flag,
                                    device='cuda:0', output_dir=exp_dir, prefix="ensemble")
    with open(os.path.join(exp_dir, "results_ensemble.json"), "w") as f:
        json.dump(ensemble_res, f, indent=2)

    # Save models
    for i, model in enumerate(models):
        path = os.path.join(exp_dir, f'resnet18_augmented_{flag}_224_{i}.pt')
        tr.save_model(model, path=path)
        print(f"Saved: {path}")