"""
Create 3x3 montages for all benchmark datasets and distribution shifts.

Usage:
    python create_dataset_montages.py
    
Or import and use the function:
    from create_dataset_montages import display_montage
    fig = display_montage('organamnist', 'id')
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
from pathlib import Path
import sys

# Add workspace root to path for imports
workspace_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(workspace_root))
from Benchmarks.medMNIST.utils.data_preprocessing_classification_evaluation.dataset_utils import apply_random_corruptions
from Benchmarks.medMNIST.utils.data_preprocessing_classification_evaluation.local_dermamnist_e import DermaMNIST_E


def display_montage(dataset_name, shift='id', save_path=None):
    """
    Display a 3x1 montage of 224x224 images.
    
    Parameters:
    -----------
    dataset_name : str
        One of: 'organamnist', 'amos2022', 'dermamnist-e-id', 'dermamnist-e-ext', 
                'pathmnist', 'tissuemnist', 'breastmnist', 'pneumoniamnist', 
                'octmnist', 'bloodmnist', 'midog'
    shift : str
        One of: 'id' (in-distribution), 'cs' (corruption shift), 
                'ps' (population shift), 'ncs' (new class shift)
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Handle special cases first
    if dataset_name == 'midog':
        # Load MIDOG++ canine patches (OOD for PathMNIST)
        midog_path = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/benchmarks/medMNIST/Data/MIDOG++/midog_canine_patches.npz')
        
        if not midog_path.exists():
            raise FileNotFoundError(f"MIDOG++ patches not found at {midog_path}")
        
        print(f" Loading MIDOG++ from: {midog_path}")
        data = np.load(str(midog_path), allow_pickle=True)
        images = data['images']  # (N, 224, 224, 3) uint8 RGB
        
        # Randomly select 3 patches
        indices = np.random.choice(len(images), min(3, len(images)), replace=False)
        selected_images = images[indices]
        
        # Convert HWC to CHW format
        if selected_images.ndim == 4 and selected_images.shape[-1] == 3:
            selected_images = np.transpose(selected_images, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
        
        # Normalize to [0, 1] range for display
        if selected_images.dtype == np.uint8 or selected_images.max() > 1.0:
            selected_images = selected_images.astype(np.float32) / 255.0
    
    elif dataset_name == 'hmu-crc':
        # Load HMU-CRC dataset (H&E stained colorectal cancer histology)
        # Use memory-mapped .npy files for fast random access without loading into RAM
        hmu_path = Path('/workspace/Benchmarks/medMNIST/data/HMU-CRC-Hist550K/hmu_crc_224_images.npy')
        
        if not hmu_path.exists():
            raise FileNotFoundError(f"HMU-CRC dataset not found at {hmu_path}")
        
        num_images = 3
        print(f" Loading HMU-CRC from: {hmu_path} (memory-mapped, selecting {num_images} random images)")
        
        # Use memory-mapping for fast random access without loading into RAM
        images_mmap = np.load(str(hmu_path), mmap_mode='r')  # (N, 224, 224, 3) uint8 RGB
        num_total = len(images_mmap)
        
        # Randomly select indices
        indices = np.random.choice(num_total, min(num_images, num_total), replace=False)
        
        # Load only the selected images
        selected_images = np.array([images_mmap[i] for i in indices])
        
        # Convert HWC to CHW format
        if selected_images.ndim == 4 and selected_images.shape[-1] == 3:
            selected_images = np.transpose(selected_images, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
        
        # Normalize to [0, 1] range for display
        if selected_images.dtype == np.uint8 or selected_images.max() > 1.0:
            selected_images = selected_images.astype(np.float32) / 255.0

    elif dataset_name == 'amos2022':
        # Load AMOS dataset - try multiple possible paths
        possible_paths = [
            Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/amos_external_test_224.npz'),
            Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/benchmarks/medMNIST/Data/AMOS_2022/amos_external_test_224.npz'),
        ]
        
        amos_path = None
        for path in possible_paths:
            if path.exists() and path.stat().st_size > 1000:  # Real file should be ~133MB, not a pointer
                amos_path = path
                break
        
        if amos_path is None:
            raise FileNotFoundError(
                " AMOS dataset not found or is a Git LFS pointer.\n"
                "   Please run: git lfs pull\n"
                f"   Checked paths: {[str(p) for p in possible_paths]}"
            )
        
        print(f" Loading AMOS from: {amos_path}")
        data = np.load(str(amos_path), allow_pickle=True)
        images = data['test_images']
        labels = data['test_labels']  # (N, 15) one-hot encoded
        amos_organ_ids = np.argmax(labels, axis=1)  # Get organ ID for each sample
        
        if shift == 'ps':
            # Filter for mapped classes ONLY (6 organs that match OrganAMNIST)
            # AMOS classes IN OrganAMNIST:
            # 0=spleen, 1=right kidney, 2=left kidney, 5=liver, 9=pancreas, 13=bladder
            mapped_classes = [0, 1, 2, 5, 9, 13]
            mask = np.isin(amos_organ_ids, mapped_classes)
            
            if mask.sum() > 0:
                images = images[mask]
                print(f" PS: Filtered to {mask.sum()} mapped class samples (6 organs in OrganAMNIST)")
            else:
                print(f" No mapped class samples found!")
        
        elif shift == 'ncs':
            # Filter for new/unseen classes (unmapped AMOS organs)
            # AMOS classes NOT in OrganAMNIST (9 novel organs):
            # 3=gall bladder, 4=esophagus, 6=stomach, 7=aorta, 8=postcava,
            # 10=right adrenal gland, 11=left adrenal gland, 12=duodenum, 14=prostate/uterus
            unmapped_classes = [3, 4, 6, 7, 8, 10, 11, 12, 14]
            mask = np.isin(amos_organ_ids, unmapped_classes)
            
            if mask.sum() > 0:
                images = images[mask]
                print(f" NCS: Filtered to {mask.sum()} new/unseen class samples (9 novel organs)")
            else:
                print(f" No new class samples found!")
        
        # Select 3 random images
        indices = np.random.choice(len(images), min(3, len(images)), replace=False)
        selected_images = images[indices]
        
    elif dataset_name == 'dermamnist-e-id':
        # ID test centers only (rosendahl, vidir_modern, vidir_molemax, vienna_dias)
        dataset = DermaMNIST_E(split='test', test_subset='id', transform=transforms.ToTensor(), size=224)
        
        # Apply corruption if needed
        if shift == 'cs':
            print(f" Applying medMNIST-C corruptions to dermamnist (ID centers)...")
            dataset = apply_random_corruptions(dataset, 'dermamnist', severity=4, cache=False, seed=None, return_pil=False)
        
        indices = np.random.choice(len(dataset), 3, replace=False)
        selected_images = []
        for idx in indices:
            img, _ = dataset[idx]
            selected_images.append(img.numpy())
        selected_images = np.array(selected_images)
        
    elif dataset_name == 'dermamnist-e-ext':
        # External test center only
        dataset = DermaMNIST_E(split='test', test_subset='external', transform=transforms.ToTensor(), size=224)
        
        indices = np.random.choice(len(dataset), 3, replace=False)
        selected_images = []
        for idx in indices:
            img, _ = dataset[idx]
            selected_images.append(img.numpy())
        selected_images = np.array(selected_images)
        
    else:
        # Regular medMNIST dataset
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', transform=transforms.ToTensor(), download=True, size=224)
        
        # Apply medMNIST-C corruption if needed
        if shift == 'cs':
            print(f" Applying medMNIST-C corruptions to {dataset_name}...")
            # Use severity=4 (max supported by medmnistc) and remove seed for varied corruptions each run
            dataset = apply_random_corruptions(dataset, dataset_name, severity=4, cache=False, seed=None, return_pil=False)
        
        indices = np.random.choice(len(dataset), 3, replace=False)
        selected_images = []
        for idx in indices:
            img, _ = dataset[idx]
            # Ensure consistent tensor format
            if not isinstance(img, np.ndarray):
                img = img.numpy() if hasattr(img, 'numpy') else np.array(img)
            # Ensure consistent shape [C, H, W]
            if img.ndim == 2:  # [H, W]
                img = img[np.newaxis, :, :]  # [1, H, W]
            elif img.ndim == 3:
                # Ensure CHW format
                if img.shape[0] not in [1, 3]:  # Not in CHW format
                    img = np.transpose(img, (2, 0, 1))
            
            # Normalize channel count: convert all to grayscale if original is grayscale
            if img.shape[0] == 3 and info['n_channels'] == 1:
                # Convert RGB to grayscale using standard weights
                img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                img = img[np.newaxis, :, :]  # Add channel dimension back
            elif img.shape[0] == 1 and info['n_channels'] == 3:
                # Convert grayscale to RGB by repeating
                img = np.repeat(img, 3, axis=0)
            
            selected_images.append(img)
        
        # Stack with explicit shape checking
        selected_images = np.stack(selected_images, axis=0)
    
    # Create 3x1 montage - exact size for 224x224*3 = 224x672 pixels at 100 DPI
    fig, axes = plt.subplots(3, 1, figsize=(2.24, 6.72), dpi=100)
    
    for idx in range(3):
        # 3 rows, 1 column
        ax = axes[idx]
        
        img = selected_images[idx]
        
        # Handle different image formats
        if img.shape[0] in [1, 3]:  # CHW format
            img = np.transpose(img, (1, 2, 0))
        
        if img.shape[-1] == 1:  # Grayscale
            ax.imshow(img.squeeze(), cmap='gray')
        else:
            ax.imshow(img)
        
        ax.axis('off')
        # Remove all padding and margins
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    
    # Remove all spacing and padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    if save_path:
        # Save with exact dimensions - 224x672 pixels (224*1 x 224*3)
        fig.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0)
        print(f"Saved to {save_path}")
    
    return fig


def display_square_montage(dataset_name, shift='id', grid_size=4, save_path=None):
    """
    Display an NxN square montage of 224x224 images.
    
    Parameters:
    -----------
    dataset_name : str
        One of: 'organamnist', 'amos2022', 'dermamnist-e-id', 'dermamnist-e-ext', 
                'pathmnist', 'tissuemnist', 'breastmnist', 'pneumoniamnist', 
                'octmnist', 'bloodmnist', 'midog'
    shift : str
        One of: 'id' (in-distribution), 'cs' (corruption shift), 
                'ps' (population shift), 'ncs' (new class shift)
    save_path : str, optional
        If provided, save the figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Handle special cases first
    if dataset_name == 'midog':
        # Load MIDOG++ canine patches (OOD for PathMNIST)
        midog_path = Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/benchmarks/medMNIST/Data/MIDOG++/midog_canine_patches.npz')
        
        if not midog_path.exists():
            raise FileNotFoundError(f"MIDOG++ patches not found at {midog_path}")
        
        data = np.load(str(midog_path), allow_pickle=True)
        images = data['images']  # (N, 224, 224, 3) uint8 RGB
        
        # Randomly select images for montage
        num_images = grid_size * grid_size
        indices = np.random.choice(len(images), min(num_images, len(images)), replace=False)
        selected_images = images[indices]
        
        # Convert HWC to CHW format
        if selected_images.ndim == 4 and selected_images.shape[-1] == 3:
            selected_images = np.transpose(selected_images, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
        
        # Normalize to [0, 1] range for display
        if selected_images.dtype == np.uint8 or selected_images.max() > 1.0:
            selected_images = selected_images.astype(np.float32) / 255.0
    
    elif dataset_name == 'amos2022':
        # Load AMOS dataset
        possible_paths = [
            Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/amos_external_test_224.npz'),
            Path('/mnt/data/psteinmetz/computer_vision_code/code/UQ_Toolbox/benchmarks/medMNIST/Data/AMOS_2022/amos_external_test_224.npz'),
        ]
        
        amos_path = None
        for path in possible_paths:
            if path.exists() and path.stat().st_size > 1000:
                amos_path = path
                break
        
        if amos_path is None:
            raise FileNotFoundError("AMOS dataset not found or is a Git LFS pointer.")
        
        data = np.load(str(amos_path), allow_pickle=True)
        images = data['test_images']
        labels = data['test_labels']
        amos_organ_ids = np.argmax(labels, axis=1)
        
        if shift == 'ps':
            mapped_classes = [0, 1, 2, 5, 9, 13]
            mask = np.isin(amos_organ_ids, mapped_classes)
            images = images[mask]
        elif shift == 'ncs':
            unmapped_classes = [3, 4, 6, 7, 8, 10, 11, 12, 14]
            mask = np.isin(amos_organ_ids, unmapped_classes)
            images = images[mask]
        
        num_images = grid_size * grid_size
        indices = np.random.choice(len(images), min(num_images, len(images)), replace=False)
        selected_images = images[indices]
        
    elif dataset_name == 'hmu-crc':
        # Load HMU-CRC dataset (H&E stained colorectal cancer histology)
        # Use memory-mapped .npy files for fast random access without loading into RAM
        hmu_path = Path('/workspace/Benchmarks/medMNIST/data/HMU-CRC-Hist550K/hmu_crc_224_images.npy')
        
        if not hmu_path.exists():
            raise FileNotFoundError(f"HMU-CRC dataset not found at {hmu_path}")
        
        num_images = grid_size * grid_size
        print(f" Loading HMU-CRC from: {hmu_path} (memory-mapped, selecting {num_images} random images)")
        
        # Use memory-mapping for fast random access without loading into RAM
        images_mmap = np.load(str(hmu_path), mmap_mode='r')  # (N, 224, 224, 3) uint8 RGB
        num_total = len(images_mmap)
        
        # Randomly select indices
        indices = np.random.choice(num_total, min(num_images, num_total), replace=False)
        
        # Load only the selected images
        selected_images = np.array([images_mmap[i] for i in indices])
        
        # Convert HWC to CHW format
        if selected_images.ndim == 4 and selected_images.shape[-1] == 3:
            selected_images = np.transpose(selected_images, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
        
        # Normalize to [0, 1] range for display
        if selected_images.dtype == np.uint8 or selected_images.max() > 1.0:
            selected_images = selected_images.astype(np.float32) / 255.0
        
    elif dataset_name == 'dermamnist-e-id':
        # ID test centers only
        dataset = DermaMNIST_E(split='test', test_subset='id', transform=transforms.ToTensor(), size=224)
        
        if shift == 'cs':
            dataset = apply_random_corruptions(dataset, 'dermamnist', severity=4, cache=False, seed=None, return_pil=False)
        
        num_images = grid_size * grid_size
        indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
        selected_images = []
        for idx in indices:
            img, _ = dataset[idx]
            selected_images.append(img.numpy())
        selected_images = np.array(selected_images)
        
    elif dataset_name == 'dermamnist-e-ext':
        # External test center only
        dataset = DermaMNIST_E(split='test', test_subset='external', transform=transforms.ToTensor(), size=224)
        
        num_images = grid_size * grid_size
        indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
        selected_images = []
        for idx in indices:
            img, _ = dataset[idx]
            selected_images.append(img.numpy())
        selected_images = np.array(selected_images)
        
    else:
        # Regular medMNIST dataset
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])
        is_grayscale = (info['n_channels'] == 1)
        
        # Special handling for grayscale datasets with corruption
        if shift == 'cs' and is_grayscale:
            # Load dataset without transform first
            dataset = DataClass(split='test', transform=None, download=True, size=224)
            
            num_images = grid_size * grid_size
            indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
            selected_images = []
            
            # Apply corruption per-image with RGB conversion
            from PIL import Image as PILImage
            from medmnistc.corruptions.registry import CORRUPTIONS_DS
            
            # Get available corruptions for this dataset
            if dataset_name in CORRUPTIONS_DS:
                corruption_funcs = list(CORRUPTIONS_DS[dataset_name].values())
            else:
                corruption_funcs = []
            
            # Set random seed for reproducible corruption selection
            np.random.seed(42)
            
            for idx in indices:
                img, _ = dataset[idx]  # PIL Image or numpy array
                
                # Convert to PIL if needed
                if isinstance(img, np.ndarray):
                    if img.ndim == 3 and img.shape[-1] == 1:
                        img = img.squeeze(-1)
                    img_pil = PILImage.fromarray(img, mode='L')
                else:
                    img_pil = img
                
                # Convert grayscale to RGB for corruption
                img_rgb = img_pil.convert('RGB')
                
                # Apply random corruption if available
                if corruption_funcs:
                    corruption_func = np.random.choice(corruption_funcs)
                    # Ensure corruption has proper random state
                    if not hasattr(corruption_func, 'rng'):
                        corruption_func.rng = np.random.default_rng()
                    img_rgb = corruption_func.apply(img_rgb, severity=4)
                
                # Convert back to grayscale (handle both PIL and numpy)
                if isinstance(img_rgb, np.ndarray):
                    # Convert numpy array RGB to grayscale
                    if img_rgb.ndim == 3 and img_rgb.shape[-1] == 3:
                        img_gray_np = (0.299 * img_rgb[:, :, 0] + 
                                      0.587 * img_rgb[:, :, 1] + 
                                      0.114 * img_rgb[:, :, 2]).astype(np.uint8)
                    else:
                        img_gray_np = img_rgb
                    img_gray = PILImage.fromarray(img_gray_np, mode='L')
                else:
                    img_gray = img_rgb.convert('L')
                
                # Convert to tensor
                img_tensor = transforms.ToTensor()(img_gray)
                selected_images.append(img_tensor.numpy())
            
            selected_images = np.array(selected_images)
        else:
            # Normal loading with or without corruption (for RGB or non-corrupted grayscale)
            dataset = DataClass(split='test', transform=transforms.ToTensor(), download=True, size=224)
            
            if shift == 'cs':
                dataset = apply_random_corruptions(dataset, dataset_name, severity=4, cache=False, seed=None, return_pil=False)
            
            num_images = grid_size * grid_size
            indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
            selected_images = []
            for idx in indices:
                img, _ = dataset[idx]
                if not isinstance(img, np.ndarray):
                    img = img.numpy() if hasattr(img, 'numpy') else np.array(img)
                
                # Ensure CHW format
                if img.ndim == 2:
                    img = img[np.newaxis, :, :]
                elif img.ndim == 3 and img.shape[0] not in [1, 3]:
                    img = np.transpose(img, (2, 0, 1))
                
                # Normalize channel count
                if img.shape[0] == 3 and info['n_channels'] == 1:
                    img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                    img = img[np.newaxis, :, :]
                elif img.shape[0] == 1 and info['n_channels'] == 3:
                    img = np.repeat(img, 3, axis=0)
                
                selected_images.append(img)
            
            selected_images = np.stack(selected_images, axis=0)
    
    # Create NxN montage - 224*N pixels per side at 100 DPI
    total_pixels = 224 * grid_size
    figsize = total_pixels / 100.0  # Convert to inches for 100 DPI
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(figsize, figsize), dpi=100)
    
    # Handle single image case (1x1 grid)
    if grid_size == 1:
        axes = np.array([[axes]])
    elif grid_size > 1 and len(axes.shape) == 1:
        axes = axes.reshape(grid_size, 1)
    
    num_images = grid_size * grid_size
    num_available = len(selected_images)
    
    for idx in range(num_images):
        i = idx // grid_size
        j = idx % grid_size
        ax = axes[i, j]
        
        if idx < num_available:
            img = selected_images[idx]
            
            # Handle different image formats
            if img.shape[0] in [1, 3]:  # CHW format
                img = np.transpose(img, (1, 2, 0))
            
            if img.shape[-1] == 1:  # Grayscale
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)
        
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    
    # Remove all spacing and padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    if save_path:
        # Save with exact dimensions - 224*N x 224*N pixels
        fig.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0)
        print(f"Saved to {save_path} ({total_pixels}x{total_pixels} pixels)")
    
    return fig


def display_4x4_montage(dataset_name, shift='id', save_path=None):
    """Legacy wrapper for backward compatibility."""
    return display_square_montage(dataset_name, shift, grid_size=4, save_path=save_path)


def generate_all_montages(output_dir=None, montage_type='both', square_size=None):
    """
    Generate all montages for the benchmark.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory to save all generated montages. If None, defaults to workspace/uq_benchmark_results/...
    montage_type : str
        Type of montages to generate: '3x1', 'NxN', or 'both'
    square_size : int, optional
        Size of square grid for NxN montages (e.g., 8 for 8x8)
    """
    if output_dir is None:
        output_dir = workspace_root / 'uq_benchmark_results' / 'figures' / 'dataset_montages'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define all datasets and their applicable shifts
    datasets_shifts = {
        # ID datasets
        'dermamnist-e-id': ['id', 'cs'],
        'organamnist': ['id', 'cs'],
        'tissuemnist': ['id', 'cs'],
        'octmnist': ['id', 'cs'],
        'pneumoniamnist': ['id', 'cs'],
        'breastmnist': ['id', 'cs'],
        'bloodmnist': ['id', 'cs'],
        'pathmnist': ['id', 'cs'],
        
        # Population shifts
        'dermamnist-e-ext': ['ps'],
        'amos2022': ['ps', 'ncs'],
        'midog': ['ncs'],  # MIDOG++ canine patches (OOD for PathMNIST)
        'hmu-crc': ['ps'],  # HMU-CRC dataset (H&E stained colorectal cancer histology)
    }
    
    # Determine what to generate
    generate_3x1 = (montage_type in ['3x1', 'both'])
    generate_square = (square_size is not None or montage_type in ['4x4', 'both'])
    
    if generate_square and square_size is None:
        square_size = 4  # Default to 4x4 for 'both' or '4x4'
    
    montage_desc = []
    if generate_3x1:
        montage_desc.append('3x1')
    if generate_square:
        montage_desc.append(f'{square_size}x{square_size}')
    
    print(f"Generating {' and '.join(montage_desc)} montages...")
    for dataset_name, shifts in datasets_shifts.items():
        for shift in shifts:
            print(f" {dataset_name} - {shift}")
            
            # Generate 3x1 montage
            if generate_3x1:
                fig = display_montage(dataset_name, shift)
                filename = f"{dataset_name}_{shift}_3x1.png"
                fig.savefig(output_path / filename, dpi=100, bbox_inches=None, pad_inches=0)
                plt.close(fig)
            
            # Generate NxN square montage
            if generate_square:
                fig = display_square_montage(dataset_name, shift, grid_size=square_size)
                filename = f"{dataset_name}_{shift}_{square_size}x{square_size}.png"
                fig.savefig(output_path / filename, dpi=100, bbox_inches=None, pad_inches=0)
                plt.close(fig)
    
    print(f"\nAll montages saved to: {output_path}")


def parse_montage_type(arg):
    """Parse montage type argument, supporting NxN format.
    
    Returns:
        tuple: (montage_type, square_size) where square_size is None for non-square types
    """
    if arg in ['3x1', 'both']:
        return arg, None
    elif 'x' in arg:
        # Parse NxN format (e.g., '4x4', '8x8', '16x16')
        parts = arg.split('x')
        if len(parts) == 2 and parts[0] == parts[1] and parts[0].isdigit():
            size = int(parts[0])
            if size > 0:
                return 'square', size
        print(f"Error: Invalid square montage format '{arg}'. Use NxN (e.g., '4x4', '8x8')")
        sys.exit(1)
    else:
        print(f"Error: Invalid montage type '{arg}'. Use '3x1', 'NxN' (e.g., '4x4', '8x8'), or 'both'")
        sys.exit(1)


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Define all datasets for validation
    all_datasets = {
        'dermamnist-e-id': ['id', 'cs'],
        'organamnist': ['id', 'cs'],
        'tissuemnist': ['id', 'cs'],
        'octmnist': ['id', 'cs'],
        'pneumoniamnist': ['id', 'cs'],
        'breastmnist': ['id', 'cs'],
        'bloodmnist': ['id', 'cs'],
        'pathmnist': ['id', 'cs'],
        'dermamnist-e-ext': ['ps'],
        'amos2022': ['ps', 'ncs'],
        'midog': ['ncs'],
        'hmu-crc': ['ps'],
    }
    
    if len(sys.argv) == 3 or len(sys.argv) == 4:
        # Command line usage: python create_dataset_montages.py organamnist id [montage_type]
        dataset = sys.argv[1]
        shift = sys.argv[2]
        montage_arg = sys.argv[3] if len(sys.argv) == 4 else 'both'
        
        montage_type, square_size = parse_montage_type(montage_arg)
        
        output_dir = workspace_root / 'uq_benchmark_results' / 'figures' / 'dataset_montages'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine what to generate
        generate_3x1 = (montage_type in ['3x1', 'both'])
        generate_square = (montage_type == 'square' or montage_type == 'both')
        
        if generate_square and square_size is None:
            square_size = 4  # Default for 'both'
        
        montage_desc = []
        if generate_3x1:
            montage_desc.append('3x1')
        if generate_square:
            montage_desc.append(f'{square_size}x{square_size}')
        
        print(f"\nGenerating {' and '.join(montage_desc)} montage(s) for {dataset} - {shift}...")
        
        # Generate 3x1 montage
        if generate_3x1:
            fig = display_montage(dataset, shift)
            output_file = output_dir / f'{dataset}_{shift}_3x1.png'
            fig.savefig(output_file, dpi=100, bbox_inches=None, pad_inches=0)
            print(f" Saved 3x1 montage to: {output_file} (224x672 pixels)")
            plt.close(fig)
        
        # Generate square montage
        if generate_square:
            fig = display_square_montage(dataset, shift, grid_size=square_size)
            total_pixels = 224 * square_size
            output_file = output_dir / f'{dataset}_{shift}_{square_size}x{square_size}.png'
            fig.savefig(output_file, dpi=100, bbox_inches=None, pad_inches=0)
            print(f" Saved {square_size}x{square_size} montage to: {output_file} ({total_pixels}x{total_pixels} pixels)")
            plt.close(fig)
            
    elif len(sys.argv) == 2:
        # Single argument: could be dataset name, montage type, or shorthand
        arg = sys.argv[1]
        
        # Check if it's a known dataset name
        if arg in all_datasets:
            # Generate montages for this dataset with its default shifts
            output_dir = workspace_root / 'uq_benchmark_results' / 'figures' / 'dataset_montages'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            shifts = all_datasets[arg]
            print(f"\nGenerating montages for {arg} with shifts: {', '.join(shifts)}...")
            
            for shift in shifts:
                fig = display_montage(arg, shift)
                output_file = output_dir / f'{arg}_{shift}_3x1.png'
                fig.savefig(output_file, dpi=100, bbox_inches=None, pad_inches=0)
                print(f" Saved 3x1 montage: {output_file} (224x672 pixels)")
                plt.close(fig)
                
                fig = display_square_montage(arg, shift, grid_size=4)
                output_file = output_dir / f'{arg}_{shift}_4x4.png'
                fig.savefig(output_file, dpi=100, bbox_inches=None, pad_inches=0)
                print(f" Saved 4x4 montage: {output_file} (896x896 pixels)")
                plt.close(fig)
        else:
            # Treat as montage type for all datasets
            montage_type, square_size = parse_montage_type(arg)
            if montage_type == 'both':
                square_size = 4  # Default for 'both'
            generate_all_montages(montage_type=montage_type, square_size=square_size)
            
    elif len(sys.argv) == 1:
        # No arguments: generate all montages
        generate_all_montages(montage_type='both', square_size=4)
    else:
        print("Usage:")
        print(" python create_dataset_montages.py                           # Generate all datasets")
        print(" python create_dataset_montages.py <dataset>                 # Generate one dataset")
        print(" python create_dataset_montages.py <montage_type>            # Generate all with montage type")
        print(" python create_dataset_montages.py <dataset> <shift>         # Generate specific dataset+shift")
        print(" python create_dataset_montages.py <dataset> <shift> <type>  # Full control")
        print("")
        print("Montage types: 3x1, NxN (e.g., 4x4, 8x8, 16x16), both")
        print("")
        print("Examples:")
        print(" python create_dataset_montages.py hmu-crc                  # Generate all shifts for hmu-crc")
        print(" python create_dataset_montages.py 8x8                      # Generate 8x8 for all datasets")
        print(" python create_dataset_montages.py organamnist id 16x16     # Generate 16x16 for organamnist-id")
        print("")
        print("Datasets: organamnist, amos2022, dermamnist-e-id, dermamnist-e-ext,")
        print(" pathmnist, tissuemnist, breastmnist, pneumoniamnist, octmnist, bloodmnist, midog, hmu-crc")
        print("Shifts: id, cs, ps, ncs")
        sys.exit(1)
