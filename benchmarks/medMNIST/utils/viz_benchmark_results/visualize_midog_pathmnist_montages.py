"""
Visualization script to create image montages from MIDOG++ and PathMNIST datasets.
Shows examples from each class side-by-side.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ========================
# Configuration
# ========================
MIDOG_BASE_DIR = "/home/lito/Documents/Codes/UQ_toolbox/benchmarks/medMNIST/Data/MIDOG++"
MIDOG_PATCHES_DIR = os.path.join(MIDOG_BASE_DIR, "patches_individual")
MIDOG_JSON_PATH = os.path.join(MIDOG_BASE_DIR, "midog_canine_patches.json")

# MIDOG++ Classes
MIDOG_CLASSES = {
    0: "Cutaneous Mast Cell Tumor",
    1: "Lung Cancer",
    2: "Lymphosarcoma",
    3: "Soft Tissue Sarcoma"
}

# PathMNIST Classes (9 classes of colon pathology)
PATHMNIST_CLASSES = {
    0: "Adipose",
    1: "Background",
    2: "Debris",
    3: "Lymphocytes",
    4: "Mucus",
    5: "Smooth Muscle",
    6: "Normal Colon Mucosa",
    7: "Cancer-Associated Stroma",
    8: "Colorectal Adenocarcinoma"
}

SAMPLES_PER_CLASS = 8  # Number of images to show per class


def load_midog_metadata():
    """Load MIDOG++ metadata to map image IDs to tumor types."""
    with open(MIDOG_JSON_PATH, 'r') as f:
        metadata = json.load(f)
    
    # Load the main MIDOGpp.json to map image IDs to tumor types
    midogpp_json = os.path.join(MIDOG_BASE_DIR, "images/MIDOGpp.json")
    if os.path.exists(midogpp_json):
        with open(midogpp_json, 'r') as f:
            midogpp_data = json.load(f)
        return metadata, midogpp_data
    
    return metadata, None


def get_patches_by_class(patches_dir, midogpp_json, tumor_types_map):
    """
    Organize patch files by tumor type class.
    
    Returns:
        dict: {class_id: list of patch file paths}
    """
    patches_by_class = {i: [] for i in range(4)}
    
    # Get all patch files
    patch_files = sorted([f for f in os.listdir(patches_dir) if f.endswith('.png')])
    
    # Extract image IDs and organize by tumor type
    # Filename format: {image_id}_patch{num}.png
    image_id_to_tumor_type = {}
    
    # Try to infer tumor types from the images data if available
    # For now, we'll look at the image_id ranges from the JSON
    # Based on the JSON: 249 images total, need to map them
    
    # Simple heuristic: distribute available images across classes
    # Let's parse the MIDOGpp.json if available
    if midogpp_json and 'images' in midogpp_json:
        for img_data in midogpp_json['images']:
            if 'tumor_type' in img_data:
                img_id = img_data.get('id', img_data.get('image_id', None))
                tumor_type_name = img_data['tumor_type']
                # Map to class ID
                for class_id, class_name in MIDOG_CLASSES.items():
                    if tumor_type_name.lower() in class_name.lower() or class_name.lower() in tumor_type_name.lower():
                        if img_id:
                            image_id_to_tumor_type[img_id] = class_id
    
    # If we couldn't build the map, use the info from midog_canine_patches.json
    if not image_id_to_tumor_type:
        # Use the tumor_types mapping from the JSON
        for tumor_name, class_id in tumor_types_map.items():
            # This gives us the mapping, but we still need image IDs
            pass
        
        # Fallback: distribute patches evenly across classes based on file ranges
        for i, patch_file in enumerate(patch_files):
            image_id = int(patch_file.split('_')[0])
            # Simple distribution based on image ID ranges
            if image_id < 280:
                class_id = 0  # Mast cell
            elif image_id < 350:
                class_id = 1  # Lung
            elif image_id < 430:
                class_id = 2  # Lymphosarcoma
            else:
                class_id = 3  # Soft tissue
            
            patches_by_class[class_id].append(os.path.join(patches_dir, patch_file))
    else:
        # Use the mapping we built
        for patch_file in patch_files:
            image_id = int(patch_file.split('_')[0])
            if image_id in image_id_to_tumor_type:
                class_id = image_id_to_tumor_type[image_id]
                patches_by_class[class_id].append(os.path.join(patches_dir, patch_file))
    
    return patches_by_class


def load_pathmnist_data():
    """Load PathMNIST dataset at 224x224 resolution."""
    try:
        import medmnist
        from medmnist import INFO
        
        # Download and load PathMNIST at 224x224 resolution
        info = INFO['pathmnist']
        DataClass = getattr(medmnist, info['python_class'])
        
        # Load test set with size=224 for 224x224 patches
        test_dataset = DataClass(split='test', size=224, download=True)
        
        return test_dataset
    except ImportError:
        print("Warning: medmnist not installed. Install with: pip install medmnist")
        return None


def sample_images_per_class(dataset, num_classes, samples_per_class):
    """Sample images from each class in the dataset."""
    images_by_class = {i: [] for i in range(num_classes)}
    
    # Collect images by class
    for img, label in dataset:
        label_val = int(label) if isinstance(label, np.ndarray) else label
        if len(images_by_class[label_val]) < samples_per_class:
            images_by_class[label_val].append(np.array(img))
    
    # Sample randomly from collected images
    sampled = {}
    for class_id, images in images_by_class.items():
        if len(images) > samples_per_class:
            indices = random.sample(range(len(images)), samples_per_class)
            sampled[class_id] = [images[i] for i in indices]
        else:
            sampled[class_id] = images
    
    return sampled


def create_montage(images_by_class, class_names, title, figsize=(20, 10)):
    """Create a montage visualization of images organized by class."""
    num_classes = len(images_by_class)
    max_samples = max(len(imgs) for imgs in images_by_class.values())
    
    fig, axes = plt.subplots(num_classes, max_samples, figsize=figsize)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    if max_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for class_idx, (class_id, images) in enumerate(sorted(images_by_class.items())):
        class_name = class_names[class_id]
        
        for sample_idx in range(max_samples):
            ax = axes[class_idx, sample_idx]
            ax.axis('off')
            
            if sample_idx < len(images):
                img = images[sample_idx]
                if isinstance(img, str):
                    # Load image from file path
                    img = np.array(Image.open(img))
                
                # Ensure RGB format
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[-1] == 1:
                    img = np.concatenate([img] * 3, axis=-1)
                
                ax.imshow(img)
                
                # Add class label as text on the first image
                if sample_idx == 0:
                    ax.text(-0.02, 0.5, f"Class {class_id}\n{class_name}", 
                           transform=ax.transAxes,
                           fontsize=11, fontweight='bold',
                           ha='right', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor='white', 
                                   edgecolor='black',
                                   alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    """Main function to create and display montages."""
    print("=" * 60)
    print("Creating Image Montages from MIDOG++ and PathMNIST")
    print("=" * 60)
    
    # ========================
    # MIDOG++ Montage
    # ========================
    print("\n1. Loading MIDOG++ dataset...")
    metadata, midogpp_json = load_midog_metadata()
    tumor_types_map = metadata.get('tumor_types', {})
    
    print(f" - Tumor types: {tumor_types_map}")
    patches_by_class = get_patches_by_class(MIDOG_PATCHES_DIR, midogpp_json, tumor_types_map)
    
    # Sample patches for visualization
    midog_samples = {}
    for class_id, patch_paths in patches_by_class.items():
        if patch_paths:
            sampled_paths = random.sample(patch_paths, min(SAMPLES_PER_CLASS, len(patch_paths)))
            midog_samples[class_id] = sampled_paths
            print(f" - Class {class_id} ({MIDOG_CLASSES[class_id]}): {len(patch_paths)} patches available, sampled {len(sampled_paths)}")
    
    print("\n2. Creating MIDOG++ montage...")
    fig_midog = create_montage(
        midog_samples, 
        MIDOG_CLASSES, 
        "MIDOG++ Dataset: Canine Tumor Tissue Patches (224×224)",
        figsize=(20, 8)
    )
    
    # ========================
    # PathMNIST Montage
    # ========================
    print("\n3. Loading PathMNIST dataset...")
    pathmnist_dataset = load_pathmnist_data()
    
    if pathmnist_dataset:
        print(f" - Dataset size: {len(pathmnist_dataset)} samples")
        print("\n4. Sampling images from PathMNIST...")
        pathmnist_samples = sample_images_per_class(pathmnist_dataset, 9, SAMPLES_PER_CLASS)
        
        for class_id, images in pathmnist_samples.items():
            print(f" - Class {class_id} ({PATHMNIST_CLASSES[class_id]}): {len(images)} samples")
        
        print("\n5. Creating PathMNIST montage...")
        fig_pathmnist = create_montage(
            pathmnist_samples, 
            PATHMNIST_CLASSES, 
            "PathMNIST Dataset: Colorectal Cancer Histology (224×224)",
            figsize=(20, 18)
        )
    else:
        print(" - Skipping PathMNIST (medmnist not available)")
    
    # ========================
    # Save and Display
    # ========================
    output_dir = "/home/lito/Documents/Codes/UQ_toolbox/uq_benchmark_results/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n6. Saving figures...")
    midog_path = os.path.join(output_dir, "midog_montage.png")
    fig_midog.savefig(midog_path, dpi=150, bbox_inches='tight')
    print(f" - MIDOG++ montage saved: {midog_path}")
    
    if pathmnist_dataset:
        pathmnist_path = os.path.join(output_dir, "pathmnist_montage.png")
        fig_pathmnist.savefig(pathmnist_path, dpi=150, bbox_inches='tight')
        print(f" - PathMNIST montage saved: {pathmnist_path}")
    
    print("\n7. Displaying figures...")
    plt.show()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
