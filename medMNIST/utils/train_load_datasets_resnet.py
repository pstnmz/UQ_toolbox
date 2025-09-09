import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.nn.functional import sigmoid, softmax
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchvision.models import resnet18, ResNet18_Weights
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet18_Weights
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import random
import numpy as np
import os, json, time


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def _append_log(path, text):
    with open(path, 'a') as f:
        f.write(text.rstrip() + '\n')


def get_datasets(data_flag, download=True, random_seed=None, im_size=28, color=False, transform=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    if transform is None:
        if color:
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
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))
            ])

    
    train_dataset = DataClass(split='train', transform=transform, size=im_size, download=download)
    val_dataset = DataClass(split='val', transform=transform, size=im_size, download=download)
    test_dataset = DataClass(split='test', transform=transform, size=im_size, download=download)

    return [train_dataset, val_dataset, test_dataset], info

def get_dataloaders(datasets, batch_size=32, num_workers=20):
    train_dataset, calib_dataset, test_dataset = datasets
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    calib_loader = DataLoader(dataset=calib_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, calib_loader, test_loader


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Check the criterion type and adjust the target size accordingly
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            # Ensure both output and target are (N,1)
            target_t = target.float().view(-1, 1)
            loss = criterion(output, target_t)
        else:
            # CrossEntropyLoss: targets shape (N,)
            target_t = target.view(-1).long()
            loss = criterion(output, target_t)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return epoch_loss / len(train_loader)


def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                target_t = target.float().view(-1, 1)
                val_loss += criterion(output, target_t).item()
                pred = (output > 0).float()
                correct += pred.eq(target_t).sum().item()
            else:
                target_t = target.view(-1).long()
                val_loss += criterion(output, target_t).item()
                pred = output.argmax(dim=1)
                correct += (pred == target_t).sum().item()

    val_loss /= len(val_loader)  # Average loss per batch instead of per dataset
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n')
    return val_loss


def train_resnet18(data_flag, num_epochs=10, batch_size=32, learning_rate=0.001, device=None,
                   train_loader=None, val_loader=None, test_loader=None, color=False, im_size=224,
                   transform=None, random_seed=None, output_dir=None, run_name="run"):
        # Optional seeding
    if random_seed is not None:
        import random
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If loaders are not given, fall back to default dataset loading
    datasets_and_loaders = load_datasets(data_flag, color, im_size, transform, batch_size)
    if (train_loader is None) or (val_loader is None) or (test_loader is None):
        _, (train_loader_fallback, calib_loader, test_loader_fallback), info = datasets_and_loaders
        # Use fallback only if missing
        train_loader = train_loader or train_loader_fallback
        val_loader = val_loader or calib_loader
        test_loader = test_loader or test_loader_fallback
    else:
        _, _, info = datasets_and_loaders  # keep info

    num_classes = len(info['label'])

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features

    if num_classes == 2:
            model.fc = torch.nn.Linear(in_features, 1)
            criterion = BCEWithLogitsLoss()
    else:
        model.fc = torch.nn.Linear(in_features, num_classes)
        criterion = CrossEntropyLoss()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_losses = []
    val_losses = []
    epoch_times = []
    if output_dir:
        figs_dir = os.path.join(output_dir, "figs")
        _ensure_dir(figs_dir)
        log_path = os.path.join(output_dir, "metrics.log")
        _append_log(log_path, f"=== {run_name} start: epochs={num_epochs}, lr={learning_rate} ===")

    run_t0 = time.time()
    for epoch in range(num_epochs):
        ep_t0 = time.time()
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss = validate(model, device, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        ep_dur = time.time() - ep_t0
        epoch_times.append(ep_dur)
        if output_dir:
            _append_log(log_path, f"{run_name} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} epoch_time_s={ep_dur:.2f}")

        print(f"{run_name} | epoch {epoch}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f}")
    
    total_train_time = time.time() - run_t0
    # Save loss curve + history
    if output_dir:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Losses - {run_name}')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figs", f"loss_curve_{run_name}.png"), dpi=200)
        plt.close()

        history = {"run_name": run_name, "train_losses": train_losses, "val_losses": val_losses, "epoch_times_sec": epoch_times, "total_train_sec": total_train_time}
        _save_json(history, os.path.join(output_dir, f"history_{run_name}.json"))
        _append_log(log_path, f"{run_name} total_train_sec={total_train_time:.2f}")


    # Final test evaluation
    eval_result = evaluate_model(model, test_loader, data_flag, device=device,
                                 output_dir=output_dir, prefix=f"{run_name}_test")

    return model, {
        "run_name": run_name,
        "history": {"train_losses": train_losses, "val_losses": val_losses, "epoch_times_sec": epoch_times},
        "timing": {"total_train_sec": total_train_time},
        "test": eval_result["metrics"],
        "confusion_matrix": eval_result["confusion_matrix"]
    }


def evaluate_model(model, test_loader, data_flag, device=None, output_dir=None, prefix="test"):
    info = INFO[data_flag]
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    class_names = list(info['label'].values())
    num_classes = len(class_names)
    is_binary = (num_classes == 2)

    if output_dir:
        figs_dir = os.path.join(output_dir, "figs")
        _ensure_dir(figs_dir)
        log_path = os.path.join(output_dir, "metrics.log")

    # Normalize to list for ensemble averaging
    models = model if isinstance(model, list) else [model]
    for m in models:
        m.eval()

    y_true = []
    y_probs = []  # shape (N, C) for multiclass; (N, 1) for binary
    t0_eval = time.time()  # timing start
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_true.append(y.numpy())

            # collect per-model probabilities then average
            probs_accum = []
            for m in models:
                logits = m(x)
                if is_binary:
                    p = sigmoid(logits).view(-1, 1)  # (B, 1)
                else:
                    p = softmax(logits, dim=1)       # (B, C)
                probs_accum.append(p.detach().cpu().numpy())

            probs_avg = np.mean(np.stack(probs_accum, axis=0), axis=0)  # (B, C) or (B, 1)
            y_probs.append(probs_avg)

    y_true = np.concatenate(y_true, axis=0)
    y_probs = np.concatenate(y_probs, axis=0)

    if is_binary:
        y_score = y_probs.ravel()                         # (N,)
        y_pred = (y_score >= 0.5).astype(int)
    else:
        y_score = y_probs                                 # (N, C)
        y_pred = np.argmax(y_score, axis=1)

    eval_wall = time.time() - t0_eval
    n_samples = int(len(y_true))

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        if is_binary:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    result = {
        "data_flag": data_flag,
        "num_classes": num_classes,
        "class_names": class_names,
        "is_ensemble": isinstance(model, list),
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "auc": auc
        },
        "confusion_matrix": cm.tolist(),
        "counts": {
            "n_samples": int(len(y_true))
        },
        "timing": {
            "eval_wall_sec": float(eval_wall),
            "throughput_img_per_s": float(n_samples / eval_wall) if eval_wall > 0 else float('inf'),
            "latency_ms_per_img": float(1000.0 * eval_wall / n_samples) if n_samples > 0 else float('nan')
        }
    }

    # Save confusion matrix figure
    if output_dir:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix ({prefix})")
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "figs", f"confusion_matrix_{prefix}.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()

        # Save metrics JSON and append log
        _save_json(result, os.path.join(output_dir, f"metrics_{prefix}.json"))
        _append_log(log_path, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {prefix} "
                              f"acc={acc:.4f} bal_acc={bal_acc:.4f} auc={auc:.4f}")

    # Minimal print
    print(f"[{prefix}] acc={acc:.3f} bal_acc={bal_acc:.3f} auc={auc:.3f}")
    return result

def save_model(model, path):
    """
    Save the PyTorch model to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_models(flag, device, size=224):

    # Load organAMNIST dataset
    data_flag = flag
    info = INFO[data_flag]
    num_classes = len(info['label'])
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

def load_datasets(dataflag, color, im_size, transform, batch_size):
        
    datasets, info = get_datasets(dataflag, im_size=im_size, color=color, transform=transform)
    # Combine train_dataset and val_dataset
    combined_train_dataset = ConcatDataset([datasets[0], datasets[1]])

    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Calculate the sizes for training and calibration datasets
    train_size = int(0.8 * len(combined_train_dataset))
    calibration_size = len(combined_train_dataset) - train_size

    # Split the combined_train_dataset into training and calibration datasets
    train_dataset, calibration_dataset = random_split(combined_train_dataset, [train_size, calibration_size])
    test_dataset = datasets[2]  # Use the test dataset as is

    dataloaders = get_dataloaders([train_dataset, calibration_dataset, test_dataset], batch_size=batch_size)

    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Calibration dataset size: {len(calibration_dataset)}')
    
    return [train_dataset, calibration_dataset, test_dataset], dataloaders, info

def CV_train_val_loaders(train_dataset_aug, train_dataset_plain, batch_size, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Labels from the plain (non-augmented) view
    labels = [label for _, label in train_dataset_plain]

    train_loaders = []
    val_loaders = []

    for train_index, val_index in skf.split(np.zeros(len(labels)), labels):
        if train_dataset_aug is not None:
            # Augmented subset for training fold
            train_subset = torch.utils.data.Subset(train_dataset_aug, train_index)
        else:
            # If no augmentation dataset is provided, use the plain dataset for training
            train_subset = torch.utils.data.Subset(train_dataset_plain, train_index)
        # Plain subset for validation fold
        val_subset = torch.utils.data.Subset(train_dataset_plain, val_index)

        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False, drop_last=False)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
    return train_loaders, val_loaders