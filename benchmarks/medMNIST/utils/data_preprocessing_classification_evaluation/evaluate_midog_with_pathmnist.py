"""
Evaluate MIDOG++ canine patches using PathMNIST trained model.
Creates confusion matrix with MIDOG tumor types vs PathMNIST predictions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import sys
import os

# Add benchmarks/medMNIST to path for proper imports
benchmark_dir = Path(__file__).parent
sys.path.insert(0, str(benchmark_dir))
os.chdir(str(benchmark_dir))

# Import from utils module
from utils import train_models_load_datasets

# Import medmnist for PathMNIST label info
from medmnist import INFO


class MIDOGPatchDataset(Dataset):
    """Dataset wrapper for MIDOG++ patches from .npz file."""
    
    def __init__(self, npz_path, transform=None):
        """
        Args:
            npz_path: Path to .npz file containing images, labels, ids
            transform: Optional transform to apply to images
        """
        data = np.load(npz_path, allow_pickle=True)
        
        self.images = data['images']  # (N, 224, 224, 3) uint8
        self.labels = data['labels']  # (N,) int32
        self.ids = data['ids']        # (N,) object (strings)
        self.tumor_types = list(data['tumor_types'])  # List of tumor type names
        
        self.transform = transform
        
        print(f"Loaded MIDOG++ dataset:")
        print(f"  Images: {self.images.shape}")
        print(f"  Labels: {self.labels.shape}")
        print(f"  Tumor types: {self.tumor_types}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]  # (224, 224, 3) uint8
        label = self.labels[idx]
        
        if self.transform:
            # Transform expects PIL Image or needs ToTensor
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize to [0, 1]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label


def get_pathmnist_class_names():
    """Get PathMNIST class names."""
    info = INFO['pathmnist']
    return list(info['label'].values())


def run_inference(model, dataloader, device):
    """
    Run inference on dataset and return predictions and true labels.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for the dataset
        device: Device to run inference on
    
    Returns:
        y_true: True labels (MIDOG tumor types)
        y_pred: Predicted labels (PathMNIST classes)
        y_probs: Prediction probabilities
    """
    model.eval()
    
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # PathMNIST has 9 classes, so outputs shape is (batch_size, 9)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def plot_confusion_matrix(y_true, y_pred, true_labels, pred_labels, title="Confusion Matrix"):
    """
    Plot confusion matrix with custom labels.
    
    Args:
        y_true: True label indices
        y_pred: Predicted label indices
        true_labels: Names for true labels (rows)
        pred_labels: Names for predicted labels (columns)
        title: Plot title
    """
    # Get unique labels that actually appear in true labels
    unique_true = np.unique(y_true)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only keep rows corresponding to labels that exist in y_true (remove zero rows)
    cm = cm[:len(unique_true), :]
    actual_true_labels = [true_labels[i] for i in unique_true]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=pred_labels, yticklabels=actual_true_labels,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted PathMNIST Class', fontsize=12)
    plt.ylabel('True MIDOG++ Tumor Type', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return cm


def main():
    # Configuration
    npz_path = '/home/lito/Documents/Codes/UQ_toolbox/benchmarks/medMNIST/Data/MIDOG++/midog_canine_patches.npz'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    
    print(f"Using device: {device}")
    print(f"Loading data from: {npz_path}\n")
    
    # Define preprocessing transform (same as PathMNIST training)
    # PathMNIST uses RGB images with normalization to [-1, 1] with mean=std=0.5
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # Normalize to [-1, 1]
    ])
    
    # Load MIDOG++ dataset
    midog_dataset = MIDOGPatchDataset(npz_path, transform=transform)
    midog_loader = DataLoader(midog_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=4, pin_memory=True)
    
    # Get label names
    midog_tumor_types = midog_dataset.tumor_types
    pathmnist_classes = get_pathmnist_class_names()
    
    print(f"\nMIDOG++ Tumor Types ({len(midog_tumor_types)}):")
    for i, tumor_type in enumerate(midog_tumor_types):
        count = np.sum(midog_dataset.labels == i)
        print(f"  {i}: {tumor_type} ({count} patches)")
    
    print(f"\nPathMNIST Classes ({len(pathmnist_classes)}):")
    for i, class_name in enumerate(pathmnist_classes):
        print(f"  {i}: {class_name}")
    
    # Load PathMNIST ResNet18 model (fold 0, standard training - no augmentation, no dropout)
    print(f"\nLoading PathMNIST ResNet18 models (standard training)...")
    pathmnist_models = train_models_load_datasets.load_models(
        flag='pathmnist',
        device=device,
        waugmentation=False,  # Standard training (no data augmentation)
        size=224,
        model_backbone='resnet18',
        setup=''  # Standard training
    )
    
    # Use only the first fold model
    model = pathmnist_models[0]
    print(f"Using model: PathMNIST ResNet18 fold 0 (standard training)")
    
    # Run inference
    print(f"\nRunning inference on {len(midog_dataset)} MIDOG++ patches...")
    y_true, y_pred, y_probs = run_inference(model, midog_loader, device)
    
    print(f"\nInference complete!")
    print(f"  True labels shape: {y_true.shape}")
    print(f"  Predictions shape: {y_pred.shape}")
    print(f"  Probabilities shape: {y_probs.shape}")
    
    # Plot confusion matrix
    print(f"\nGenerating confusion matrix...")
    cm = plot_confusion_matrix(
        y_true, y_pred,
        true_labels=midog_tumor_types,
        pred_labels=pathmnist_classes,
        title="MIDOG++ Canine Tumors vs PathMNIST Predictions\n(Cross-Domain Evaluation)"
    )
    
    # Save figure
    output_dir = Path('/home/lito/Documents/Codes/UQ_toolbox/benchmarks/medMNIST/Data/MIDOG++')
    output_path = output_dir / 'confusion_matrix_midog_vs_pathmnist.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to: {output_path}")
    
    # Print summary statistics
    print(f"\nConfusion Matrix Shape: {cm.shape}")
    print(f"(Rows: {len(midog_tumor_types)} MIDOG tumor types, Columns: {len(pathmnist_classes)} PathMNIST classes)")
    
    # Print most common predictions per tumor type
    print(f"\nMost common PathMNIST prediction per MIDOG tumor type:")
    for i, tumor_type in enumerate(midog_tumor_types):
        row = cm[i]
        top_pred_idx = np.argmax(row)
        top_pred_count = row[top_pred_idx]
        total_count = row.sum()
        percentage = (top_pred_count / total_count * 100) if total_count > 0 else 0
        print(f"  {tumor_type}:")
        print(f"    → {pathmnist_classes[top_pred_idx]} ({top_pred_count}/{total_count} = {percentage:.1f}%)")


if __name__ == '__main__':
    main()
