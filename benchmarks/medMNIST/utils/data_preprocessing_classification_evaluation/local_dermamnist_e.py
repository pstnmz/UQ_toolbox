import medmnist
from medmnist.dataset import MedMNIST
from pathlib import Path
import numpy as np
import torch
from PIL import Image

DERMAMNIST_E_INFO = {
    'python_class': 'DermaMNIST_E',
    'description': (
        'The DermaMNIST-E dataset is an enhanced and corrected version of DermaMNIST, '
        'proposed by Abhishek et al. (2025) to address data leakage and duplication issues in the '
        'original MedMNIST DermaMNIST benchmark. Like its predecessor, it is based on the '
        'HAM10000 dataset of 10,015 dermatoscopic images from two clinical sites in Austria and '
        'Australia, representing 7 diagnostic categories of common pigmented skin lesions. '
        'DermaMNIST-E ensures strict partitioning by lesion identity (no overlap of lesion IDs '
        'across train, validation, and test sets) and higher-quality image resizing for improved '
        'visual fidelity. Images are resized to MNIST-like dimensions (3×28×28) for lightweight '
        'benchmarking, with corrected splits and updated metadata. It is part of the DermaMNIST-C/E '
        'revisions introduced to improve data integrity and benchmark reliability.'
    ),
    'url_224': 'https://zenodo.org/records/12739457/files/dermamnist_e_224.npz?download=1',
    'MD5_224': '2b778311816a55b580227fa7ea8c9ce9',
    'task': 'multi-class',
    'label': {
        '0': 'actinic keratoses and intraepithelial carcinoma',
        '1': 'basal cell carcinoma',
        '2': 'benign keratosis-like lesions',
        '3': 'dermatofibroma',
        '4': 'melanoma',
        '5': 'melanocytic nevi',
        '6': 'vascular lesions'
    },
    'n_channels': 3,
    'n_samples': {
        'train': 10015,
        'val': 193,
        'test': 1511,
        'test_id': 1195,  # ID test centers (rosendahl, vidir_modern, vidir_molemax, vienna_dias)
        'test_external': 316  # External test center
    },
    'license': 'CC BY-NC 4.0'
    }

class DermaMNIST_E(MedMNIST):
    INFO = DERMAMNIST_E_INFO
    flag = 'dermamnist-e'  # User-facing flag
    available_sizes = (224,)  # only 224 supported
    
    def __init__(self, split='train', transform=None, target_transform=None,
                 download=False, as_rgb=True, size=224, root=None, 
                 test_subset='all', **kwargs):
        """
        Args:
            split: 'train', 'val', or 'test'
            transform: Image transformations
            target_transform: Label transformations
            download: Unused (data is local)
            as_rgb: Convert to RGB
            size: Image size (only 224 supported)
            root: Root directory for data
            test_subset: For split='test' only. Options:
                - 'all': Use all test samples (default)
                - 'id': Use only ID test centers (rosendahl, vidir_modern, vidir_molemax, vienna_dias)
                - 'external': Use only external test center
        """
        # Use custom filename for local file
        # File is located at: benchmarks/medMNIST/Data/ISIC_2018/dermamnist_extended_224_wsitesources.npz
        if root is None:
            # Default to the local data directory
            import os
            script_dir = Path(__file__).parent.parent  # Go up to medMNIST/
            root = script_dir / "Data" / "ISIC_2018"
        
        self.root = Path(root)
        
        # Load the NPZ file directly (bypass download)
        npz_filename = f"dermamnist_extended_{size}_wsitesources.npz"
        npz_path = self.root / npz_filename
        
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Could not find {npz_filename} at {self.root}\n"
                f"Expected path: {npz_path}"
            )
        
        # Load data directly instead of calling parent init (which tries to download)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb
        self.size = size
        self.test_subset = test_subset
        
        # Load the NPZ file
        npz_data = np.load(npz_path, allow_pickle=True)
        
        # Load images and labels for the specified split
        imgs = npz_data[f'{split}_images']
        labels = npz_data[f'{split}_labels']
        
        # Load extra metadata if present (test centers, etc.)
        if self.split == 'test' and 'test_centers' in npz_data.files:
            self.test_centers = npz_data['test_centers']
            
            # Filter by test_subset if specified
            if test_subset == 'id':
                # ID centers: all except 'external'
                id_mask = self.test_centers != 'external'
                imgs = imgs[id_mask]
                labels = labels[id_mask]
                self.test_centers = self.test_centers[id_mask]
                print(f"  Using ID test centers: {len(imgs)} samples")
            elif test_subset == 'external':
                # External center only
                ext_mask = self.test_centers == 'external'
                imgs = imgs[ext_mask]
                labels = labels[ext_mask]
                self.test_centers = self.test_centers[ext_mask]
                print(f"  Using external test center: {len(imgs)} samples")
            # else: test_subset == 'all', use all samples
        else:
            self.test_centers = None
        
        # Store as standard attributes
        self.imgs = imgs
        self.labels = labels
        self.data = self.imgs
        self.targets = self.labels
    def __len__(self):
        # Prefer explicit length attributes if present
        if hasattr(self, "data"):
            return len(self.data)
        for name in ("imgs", "images", "x", "X"):
            if hasattr(self, name):
                return len(getattr(self, name))
        raise AttributeError("No image array attribute found (expected data/imgs/images/x).")

    def _get_arrays(self):
        # Figure out image array
        if hasattr(self, "data"):
            imgs = self.data
        elif hasattr(self, "imgs"):
            imgs = self.imgs
        elif hasattr(self, "images"):
            imgs = self.images
        elif hasattr(self, "x"):
            imgs = self.x
        else:
            raise AttributeError("Could not find images (data/imgs/images/x).")

        # Figure out label array
        if hasattr(self, "targets"):
            labels = self.targets
        elif hasattr(self, "labels"):
            labels = self.labels
        elif hasattr(self, "y"):
            labels = self.y
        else:
            raise AttributeError("Could not find labels (targets/labels/y).")
        return imgs, labels

    def __getitem__(self, index):
        imgs, labels = self._get_arrays()
        img = imgs[index]
        label = labels[index]

        # Convert label to plain int
        if isinstance(label, (np.ndarray, np.generic)):
            label = int(label)
        elif torch.is_tensor(label):
            label = int(label.item())

        # Ensure numpy array then to PIL if needed for transforms
        if torch.is_tensor(img):
            img = img.cpu().numpy()

        if isinstance(img, np.ndarray):
            # If shape (H,W) make it 3-channel
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=2)
            # If shape (C,H,W) transpose to (H,W,C)
            if img.ndim == 3 and img.shape[0] in (1,3) and img.shape[-1] != 3:
                # assume CHW
                img = np.transpose(img, (1,2,0))
            img = Image.fromarray(img.astype(np.uint8))

        if hasattr(self, "transform") and self.transform:
            img = self.transform(img)
        if hasattr(self, "target_transform") and self.target_transform:
            label = self.target_transform(label)

        return img, label

# Register into medmnist module
setattr(medmnist, 'DermaMNIST_E', DermaMNIST_E)
try:
    # Register with both hyphen and underscore for compatibility
    medmnist.INFO['dermamnist-e'] = DERMAMNIST_E_INFO
    medmnist.INFO['dermamnist_e'] = DERMAMNIST_E_INFO
except Exception:
    from medmnist.info import INFO as _INFO
    _INFO['dermamnist-e'] = DERMAMNIST_E_INFO
    _INFO['dermamnist_e'] = DERMAMNIST_E_INFO
