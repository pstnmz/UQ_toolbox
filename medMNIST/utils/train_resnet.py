import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.nn.functional import sigmoid, softmax
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet18_Weights
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

def get_data_loaders(data_flag, batch_size=32, download=True, return_datasets=False, random_seed=None, im_size=28, color=False):
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

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if return_datasets:
        return [train_loader, val_loader, test_loader], [train_dataset, val_dataset, test_dataset], info
    else:
        return [train_loader, val_loader, test_loader], info


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Check the criterion type and adjust the target size accordingly
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            loss = criterion(output, target.float())  # Ensure target is float for BCEWithLogitsLoss
        else:
            loss = criterion(output, target.squeeze().long())  # Keep squeeze for other loss types
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
            # Check the criterion type and adjust the target size accordingly
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                val_loss += criterion(output, target.float()).item()  # sum up batch loss
            else:
                val_loss += criterion(output, target.squeeze().long()).item()  # sum up batch loss

            # Adjust prediction logic based on the task type
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                pred = (output > 0).float()  # Binary classification: threshold at 0 for logits
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                pred = output.argmax(dim=1, keepdim=True)  # Multiclass classification: argmax for class index
                correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader)  # Average loss per batch instead of per dataset
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n')
    return val_loss


def train_resnet18(data_flag, num_epochs=10, batch_size=32, learning_rate=0.001, device=None, train_loader=None, val_loader=None, test_loader=None, random_seed=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dataloaders, info = get_data_loaders(data_flag, batch_size, random_seed=random_seed)
    if train_loader is None or val_loader is None or test_loader is None:
        train_loader, val_loader, test_loader = dataloaders[0], dataloaders[1], dataloaders[2]

    num_classes = len(info['label'])
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Modify the final fully connected layer to handle both binary and multiclass classification
    if num_classes == 2:
        model.fc = nn.Linear(model.fc.in_features, 1)  # Output 1 value for binary classification
        criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss for binary classification
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Output logits for each class
        criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss for multiclass classification
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss = validate(model, device, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

    evaluate_model(model, test_loader, data_flag, device)
    return model

def evaluate_model(model, test_loader, data_flag, device=None):
    info = INFO[data_flag]
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if the model is a list (for ensembling)
    is_ensemble = isinstance(model, list)

    y_true = []
    y_score = []
    if is_ensemble:
        for m in model:
            m.eval()
    else:
        model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if is_ensemble:
                outputs = [m(data) for m in model]  # Collect predictions from all models
                output = torch.mean(torch.stack(outputs), dim=0)  # Average predictions
            else:
                output = model(data)
            
            if len(np.unique(target.cpu().numpy())) == 2:  # Binary classification
                output = sigmoid(output)  # Apply sigmoid for binary classification
                y_score.extend(output.cpu().numpy().flatten())
            else:
                output = softmax(output, dim=1)
                y_score.extend(output.cpu().numpy())
            y_true.extend(target.cpu().numpy().flatten())
            
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Calculate metrics
    if len(np.unique(y_true)) == 2:  # Binary classification
        auc = roc_auc_score(y_true, y_score)
        y_pred = (y_score > 0.5).astype(int)
    else:  # Multiclass classification
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        y_pred = np.argmax(y_score, axis=1)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(f'Accuracy: {acc:.3f}')
    print(f'Balanced Accuracy: {bal_acc:.3f}')
    print(f'AUC: {auc:.3f}')

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=info['label'].values(), yticklabels=info['label'].values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def save_model(model, path):
    """
    Save the PyTorch model to the specified path.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")