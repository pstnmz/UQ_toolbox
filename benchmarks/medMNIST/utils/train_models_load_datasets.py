"""
Shared utilities for medMNIST dataset loading, model training, and CV fold generation.

Public API
----------
Dataset / DataLoader helpers:
    get_datasets          -- load a medMNIST (or custom) dataset
    load_datasets         -- convenience wrapper returning train/val/test datasets
    get_dataloaders       -- wrap datasets into DataLoaders (with optional caching)

Model training:
    train_resnet18        -- train a ResNet-18 for one fold (returns model + metrics)
    train_vit             -- train a ViT-B/16 for one fold
    evaluate_model        -- evaluate a model (or ensemble) on a test loader
    save_model / load_models -- checkpoint I/O

Cross-validation:
    CV_fold_generator     -- generator yielding (fold_idx, train_loader, val_loader)
    CV_train_val_loaders  -- return all folds at once (kept for backward compat.)
    get_single_CV_fold    -- materialize one specific fold

Note: CV parameters (n_splits=5, seed=42, StratifiedKFold) are shared with
the benchmark inference pipeline (run_medmnist_benchmark.py).  Never change
them without retraining the models.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.nn.functional import sigmoid, softmax
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, ViT_B_16_Weights
import medmnist
from medmnist import INFO
from .data_preprocessing_classification_evaluation.local_dermamnist_e import DERMAMNIST_E_INFO
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torchvision import transforms as T
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import random
import gc
import numpy as np
import os, json, time
from pathlib import Path
from monai.data import CacheDataset as MONAI_CacheDataset
MONAI_AVAILABLE = True

torch.backends.cudnn.benchmark=True

def _get_project_root():
    """Get the UQ_toolbox project root directory."""
    current = Path(__file__).resolve()
    # Walk up until we find the directory containing 'FailCatcher' and 'benchmarks'
    for parent in current.parents:
        if (parent / 'FailCatcher').exists() and (parent / 'benchmarks').exists():
            return parent
    # Fallback: assume we're in benchmarks/medMNIST/utils
    return current.parent.parent.parent

# Custom RandAugment without flip operations for anatomical consistency
class RandAugmentNoFlip(T.RandAugment):
    """
    RandAugment without horizontal/vertical flip operations.
    Use this for datasets where left/right orientation matters (e.g., OrganaMNIST).
    
    Note: Standard torchvision RandAugment doesn't include flips by default,
    but this class explicitly ensures no flip operations are ever added.
    """
    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=T.InterpolationMode.BILINEAR, fill=None):
        super().__init__(num_ops=num_ops, magnitude=magnitude, num_magnitude_bins=num_magnitude_bins, 
                         interpolation=interpolation, fill=fill)
        
        # Standard RandAugment operations (no flips by design):
        # Identity, ShearX, ShearY, TranslateX, TranslateY, Rotate, 
        # Brightness, Color, Contrast, Sharpness, Posterize, Solarize, 
        # AutoContrast, Equalize
        # 
        # This class exists to be explicit about no flips and allow future customization

def _clear_cache_dataset(ds):
    if ds is None:
        return
    if hasattr(ds, "clear_cache"):
        try:
            ds.clear_cache()
        except Exception:
            pass
    for attr in ("_cache", "_cached", "cache", "data"):
        if hasattr(ds, attr):
            setattr(ds, attr, None)

# --- ResNet18 with Dropout for MC Dropout ---
class ResNet18WithDropout(nn.Module):
    """
    ResNet18 with dropout layers added after each residual block and before FC layer.
    For Monte Carlo Dropout uncertainty quantification.
    """
    def __init__(self, num_classes, dropout_rate=0.5, pretrained=True):
        super(ResNet18WithDropout, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            base_model = models.resnet18(weights=None)
        
        # Extract layers (exclude final FC layer and avgpool)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.layer2 = base_model.layer2
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.layer3 = base_model.layer3
        self.dropout3 = nn.Dropout(p=dropout_rate)
        
        self.layer4 = base_model.layer4
        self.dropout4 = nn.Dropout(p=dropout_rate)
        
        self.avgpool = base_model.avgpool
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        
        # Replace FC layer based on num_classes
        in_features = base_model.fc.in_features
        if num_classes == 2:
            self.fc = nn.Linear(in_features, 1)  # Binary classification
        else:
            self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        x = self.dropout3(x)
        
        x = self.layer4(x)
        x = self.dropout4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        x = self.fc(x)
        
        return x

# --- Vision Transformer with Dropout for MC Dropout ---
class ViTWithDropout(nn.Module):
    """
    Vision Transformer (ViT-B/16) with dropout layers for Monte Carlo Dropout.
    Adds dropout to attention and MLP layers in each transformer block.
    """
    def __init__(self, num_classes, dropout_rate=0.1, pretrained=True):
        super(ViTWithDropout, self).__init__()
        
        # Load pretrained ViT-B/16
        if pretrained:
            base_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        else:
            base_model = vit_b_16(weights=None)
        
        # Copy all layers except the final head
        self.conv_proj = base_model.conv_proj
        self.encoder = base_model.encoder
        self.class_token = base_model.class_token
        
        # Add dropout after encoder
        self.dropout_encoder = nn.Dropout(p=dropout_rate)
        
        # Inject dropout into each transformer block's attention and MLP
        for block in self.encoder.layers:
            # Add dropout after attention
            if hasattr(block, 'self_attention'):
                # Increase dropout in attention
                block.self_attention.dropout = dropout_rate
            # Add dropout in MLP
            if hasattr(block, 'mlp'):
                # Replace the MLP with dropout-enabled version
                old_mlp = block.mlp
                mlp_layers = []
                for layer in old_mlp:
                    mlp_layers.append(layer)
                    if isinstance(layer, nn.Linear):
                        mlp_layers.append(nn.Dropout(p=dropout_rate))
                block.mlp = nn.Sequential(*mlp_layers)
        
        # Replace head
        hidden_dim = base_model.hidden_dim
        if num_classes == 2:
            self.heads = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.heads = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )
    
    def forward(self, x):
        # Reshape and permute input
        n, c, h, w = x.shape
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Encoder
        x = self.encoder(x)
        x = self.dropout_encoder(x)
        
        # Classifier head (use class token)
        x = x[:, 0]
        x = self.heads(x)
        
        return x

# --- New: simple patience-based early stopper ---
class EarlyStopper:
    def __init__(self, mode='min', patience=10, min_delta=0.0):
        """
        mode: 'min' for loss, 'max' for accuracy
        patience: epochs to wait without improvement before stopping
        min_delta: minimum change to qualify as improvement
        """
        assert mode in ('min', 'max')
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad_epochs = 0

    def _improved(self, current):
        if self.best is None:
            return True
        if self.mode == 'min':
            return current < (self.best - self.min_delta)
        else:
            return current > (self.best + self.min_delta)

    def step(self, current):
        improved = self._improved(current)
        if improved:
            self.best = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    def should_stop(self):
        return self.bad_epochs >= self.patience
    
# Prefetcher: moves batches to device asynchronously to overlap CPU/GPU work
class PrefetchLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream() if self.device.type == 'cuda' else None
        # expose common attributes for compatibility (e.g. `.dataset`, `.batch_size`, ...)
        self.dataset = getattr(loader, "dataset", None)
        self.batch_size = getattr(loader, "batch_size", None)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._iter = iter(self.loader)
        return self

    def __next__(self):
        batch = next(self._iter)  # may raise StopIteration
        if self.stream is None:
            x, y = batch
            return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        # async copy on separate stream
        with torch.cuda.stream(self.stream):
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        # ensure main stream waits for the prefetch stream
        torch.cuda.current_stream().wait_stream(self.stream)
        return x, y

    def __getattr__(self, name):
        # Delegate unknown attributes to the underlying loader (keeps compatibility)
        return getattr(self.loader, name)

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def _save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def _append_log(path, text):
    with open(path, 'a') as f:
        f.write(text.rstrip() + '\n')

def get_datasets(data_flag, download=True, random_seed=None, im_size=28, color=False, transform=None, transform_test=None, test_subset='all'):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if data_flag == 'dermamnist-e':
        # use the local INFO we registered above
        info = DERMAMNIST_E_INFO
        DataClass = getattr(medmnist, info['python_class'])
    else : 
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
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(mean=[.5], std=[.5])
            ])

    train_dataset = DataClass(split='train', transform=transform, size=im_size, download=download)
    val_dataset = DataClass(split='val', transform=transform, size=im_size, download=download)
    if transform_test is None:
        transform_test = transform
    
    # Pass test_subset for dermamnist-e
    if data_flag == 'dermamnist-e':
        test_dataset = DataClass(split='test', transform=transform_test, size=im_size, download=download, test_subset=test_subset)
    else:
        test_dataset = DataClass(split='test', transform=transform_test, size=im_size, download=download)

    return [train_dataset, val_dataset, test_dataset], info


def get_dataloaders(datasets, batch_size=32, num_workers=None, use_cache_test=False, cache_backend='monai', cache_rate=1.0):
    """
    Build dataloaders for (train, calibration, test) datasets.

    Args:
      datasets: (train_dataset, calib_dataset, test_dataset)
      batch_size: int
      num_workers: int or None (auto heuristic)
      use_monai: bool - if True and MONAI available, use MONAI ThreadDataLoader
      monai_params: optional dict passed to MONAI ThreadDataLoader
      use_cache: bool - if True and MONAI available, build MONAI CacheDataset (cache_backend='monai')
      cache_rate: fraction (0..1) to cache in MONAI CacheDataset
      train_augment_transform: callable applied on-the-fly to training items (keeps RandAugment random)
    Returns:
      train_loader, calib_loader, test_loader
    """
    # default: conservative shared‑machine heuristic (allow override via NUM_WORKERS)
    if num_workers is None:
        try:
            env_n = os.environ.get("NUM_WORKERS")
            if env_n is not None:
                num_workers = int(env_n)
            else:
                n_cpu = os.cpu_count() or 1
                n_users = int(os.environ.get("SHARED_USERS", "4"))
                per_user = max(2, n_cpu // (n_users * 8))
                num_workers = int(min(max(per_user, 2), 16))
        except Exception:
            num_workers = 4

    train_dataset, calib_dataset, test_dataset = datasets

    # MONAI CacheDataset backend (preferred when use_cache=True)
    if use_cache_test and cache_backend == 'monai' and MONAI_AVAILABLE:
        try:
            # build MONAI-style data list (keep tensors as tensors to avoid needless numpy<->tensor roundtrip)
            def _build_data_list(ds):
                data_list = []
                if isinstance(ds, torch.utils.data.Subset):
                    base = ds.dataset
                    indices = ds.indices
                else:
                    base = ds
                    indices = range(len(ds))
                for i in indices:
                    item = base[i]
                    if isinstance(item, dict) and 'image' in item and 'label' in item:
                        img, lbl = item['image'], item['label']
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        img, lbl = item[0], item[1]
                    else:
                        raise RuntimeError("Unsupported dataset item format for MONAI caching.")
                    if torch.is_tensor(img):
                        # keep tensor (detached on CPU) to avoid round-trip
                        data_list.append({'image_tensor': img.detach().cpu(), 'label': int(lbl)})
                    else:
                        # keep numpy array
                        data_list.append({'image_numpy': np.asarray(img), 'label': int(lbl)})
                return data_list
            
            test_list = _build_data_list(test_dataset)

            # transform that returns (tensor, label) and handles both stored tensor or numpy
            def _to_tensor_tuple(d):
                if 'image_tensor' in d:
                    img = d['image_tensor']
                    if not isinstance(img, torch.Tensor):
                        img = torch.as_tensor(img)
                    img = img.float()
                else:
                    img = torch.from_numpy(d['image_numpy']).float()
                lbl = torch.tensor(int(d['label']), dtype=torch.long)
                return img, lbl

            test_cache_ds  = MONAI_CacheDataset(data=test_list,  transform=_to_tensor_tuple, cache_rate=float(cache_rate))

            persistent = True if (num_workers and num_workers>0) else False

            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            calib_loader = DataLoader(dataset=calib_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(dataset=test_cache_ds, batch_size=batch_size, shuffle=False, prefetch_factor=3, num_workers=num_workers,
                                    pin_memory=True, persistent_workers=persistent)
            print(f"Using MONAI CacheDataset (cache_rate={cache_rate}) for test set with {len(test_cache_ds)} items.")
            return train_loader, calib_loader, test_loader
        except Exception as e:
            print("MONAI CacheDataset construction failed, falling back to non-cached loaders:", e)

    else:
        # Default: standard torch DataLoader
        persistent = True if (num_workers and num_workers > 0) else False
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                prefetch_factor=3, num_workers=num_workers,
                                pin_memory=True, persistent_workers=persistent)
        calib_loader = DataLoader(dataset=calib_dataset, batch_size=batch_size, shuffle=False,
                                prefetch_factor=3, num_workers=num_workers,
                                pin_memory=True, persistent_workers=persistent)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                prefetch_factor=3, num_workers=num_workers,
                                pin_memory=True, persistent_workers=persistent)

        return train_loader, calib_loader, test_loader
    

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
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
    val_loss = 0.0
    correct = 0
    n_samples = len(val_loader.dataset)
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
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

    val_loss /= len(val_loader)  # average per batch
    val_acc = correct / float(n_samples)
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{n_samples} ({100. * val_acc:.0f}%)\n')
    return val_loss, val_acc

def _compute_class_weights_from_loader(train_loader, num_classes):
    """
    Returns:
      class_weight (torch.Tensor or None) for CE
      pos_weight (torch.Tensor or None) for BCEWithLogits
    """
    counts = np.zeros(int(num_classes), dtype=np.int64)
    with torch.no_grad():
        for _, target in train_loader:
            t = target.detach().cpu().numpy().reshape(-1)
            # targets come as ints [0..C-1] (binary: 0/1)
            for c in t:
                counts[int(c)] += 1

    total = counts.sum()
    # Avoid div-by-zero
    counts = np.clip(counts, 1, None)

    if num_classes == 2:
        neg, pos = counts[0], counts[1]
        # pos_weight = neg/pos
        pos_weight = torch.tensor([float(neg) / float(pos)], dtype=torch.float32)
        return None, pos_weight
    else:
        # CE class weights: inverse frequency, normalized to mean 1.0
        w = 1.0 / counts.astype(np.float64)
        w = w / (w.mean() + 1e-12)
        class_weight = torch.tensor(w, dtype=torch.float32)
        return class_weight, None

def train_resnet18(data_flag, info, num_epochs=10, learning_rate=0.001, device=None,
                   train_loader=None, val_loader=None, test_loader=None, random_seed=None, 
                   output_dir=None, run_name="run", scheduler=False, early_stop=True, 
                   monitor='val_loss', patience=10, min_delta=0.001, restore_best=True, 
                   checkpoint_best=False, class_weighting=True, use_dropout=False, dropout_rate=0.5):
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
    num_classes = len(info['label'])
    
    # Compute weights BEFORE wrapping loaders for CUDA prefetch
    ce_weight_cpu, bce_pos_weight_cpu = (None, None)
    if class_weighting:
        try:
            ce_weight_cpu, bce_pos_weight_cpu = _compute_class_weights_from_loader(train_loader, num_classes)
            if ce_weight_cpu is not None:
                print(f"Using CE class weights (mean=1): {ce_weight_cpu.tolist()}")
            if bce_pos_weight_cpu is not None:
                print(f"Using BCE pos_weight: {bce_pos_weight_cpu.item():.4f}")
        except Exception as e:
            print("Warning: failed to compute class weights, continuing unweighted:", e)

    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Wrap loaders with PrefetchLoader to overlap copies (only for CUDA)
    if device and 'cuda' in str(device).lower():
        train_loader = PrefetchLoader(train_loader, device)
        val_loader = PrefetchLoader(val_loader, device)
        test_loader = PrefetchLoader(test_loader, device)

    # Create model with or without dropout
    if use_dropout:
        print(f"Using ResNet18 with Dropout (dropout_rate={dropout_rate})")
        model = ResNet18WithDropout(num_classes=num_classes, dropout_rate=dropout_rate, pretrained=True)
    else:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        if num_classes == 2:
            model.fc = torch.nn.Linear(in_features, 1)
        else:
            model.fc = torch.nn.Linear(in_features, num_classes)
    
    # Setup loss criterion with class weights
    if num_classes == 2:
        # Move pos_weight to device if available
        if bce_pos_weight_cpu is not None:
            criterion = BCEWithLogitsLoss(pos_weight=bce_pos_weight_cpu.to(device))
        else:
            criterion = BCEWithLogitsLoss()
    else:
        # Move CE weights to device if available
        if ce_weight_cpu is not None:
            criterion = CrossEntropyLoss(weight=ce_weight_cpu.to(device))
        else:
            criterion = CrossEntropyLoss()
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if scheduler is True:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    train_losses = []
    val_losses = []
    val_accs = []  # new: keep val_acc history
    epoch_times = []
    if output_dir:
        figs_dir = os.path.join(output_dir, "figs")
        _ensure_dir(figs_dir)
        log_path = os.path.join(output_dir, "metrics.log")
        _append_log(log_path, f"=== {run_name} start: epochs={num_epochs}, lr={learning_rate} ===")

    # New: setup early stopper
    stopper = None
    best_state = None
    best_epoch = -1
    best_metric_val = None
    metric_mode = 'min' if monitor == 'val_loss' else 'max'
    if early_stop:
        stopper = EarlyStopper(mode=metric_mode, patience=patience, min_delta=min_delta)
    best_ckpt_path = os.path.join(output_dir, f"best_{run_name}.pt") if (output_dir and checkpoint_best) else None

    run_t0 = time.time()
    for epoch in range(num_epochs):
        ep_t0 = time.time()
        t0_train = time.perf_counter()
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        if torch.cuda.is_available() and ('cuda' in str(device).lower()):
            torch.cuda.synchronize()
        t1_train = time.perf_counter()

        if torch.cuda.is_available() and ('cuda' in str(device).lower()):
            torch.cuda.synchronize()
        t_before_val = time.perf_counter()

        val_loss, val_acc = validate(model, device, val_loader, criterion)
        if torch.cuda.is_available() and ('cuda' in str(device).lower()):
            torch.cuda.synchronize()
        t_after_val = time.perf_counter()

        print(f"[timing] epoch={epoch} train_exec_s={(t1_train - t0_train):.3f} transition_s={(t_before_val - t1_train):.3f} val_exec_s={(t_after_val - t_before_val):.3f}")
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        ep_dur = time.time() - ep_t0
        epoch_times.append(ep_dur)
        if output_dir:
            _append_log(log_path, f"{run_name} epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.4f} lr={current_lr:.6e} epoch_time_s={ep_dur:.2f}")
        
        if scheduler is not None:
            scheduler.step()

        print(f"{run_name} | epoch {epoch}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f} | val_acc {val_acc:.4f}")

        # --- Early stopping check ---
        if stopper is not None:
            metric_value = val_loss if monitor == 'val_loss' else val_acc
            improved = stopper.step(metric_value)
            if improved:
                best_metric_val = metric_value
                best_epoch = epoch
                # keep an in-memory copy and optionally checkpoint
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if best_ckpt_path is not None:
                    torch.save(model.state_dict(), best_ckpt_path)
            if stopper.should_stop():
                print(f"Early stopping at epoch {epoch} (best {monitor}={best_metric_val:.4f} at epoch {best_epoch})")
                if output_dir:
                    _append_log(log_path, f"{run_name} early_stop epoch={epoch} best_epoch={best_epoch} best_{monitor}={best_metric_val:.6f}")
                break
    
    # Restore best weights if requested
    if restore_best and best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            # fallback to disk if needed
            if best_ckpt_path and os.path.isfile(best_ckpt_path):
                model.load_state_dict(torch.load(best_ckpt_path, map_location='cpu'))
        print(f"Restored best model from epoch {best_epoch} ({monitor}={best_metric_val:.4f})")

    total_train_time = time.time() - run_t0
    # Save loss curve + history
    if output_dir:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Losses - {run_name}')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figs", f"loss_curve_{run_name}.png"), dpi=200)
        plt.close()

        # also save val_acc curve
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(f'Val Acc - {run_name}')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figs", f"val_acc_{run_name}.png"), dpi=200)
        plt.close()

        history = {
            "run_name": run_name,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "epoch_times_sec": epoch_times,
            "total_train_sec": total_train_time,
            "early_stop": {
                "enabled": bool(early_stop),
                "monitor": monitor,
                "best_epoch": int(best_epoch),
                "best_value": float(best_metric_val) if best_metric_val is not None else None
            }
        }
        _save_json(history, os.path.join(output_dir, f"history_{run_name}.json"))
        _append_log(log_path, f"{run_name} total_train_sec={total_train_time:.2f}")

    # Final test evaluation
    eval_result = evaluate_model(model, test_loader, data_flag, device=device,
                                 output_dir=output_dir, prefix=f"{run_name}_test")

    return model, {
        "run_name": run_name,
        "history": {"train_losses": train_losses, "val_losses": val_losses, "val_accs": val_accs, "epoch_times_sec": epoch_times},
        "timing": {"total_train_sec": total_train_time},
        "test": eval_result["metrics"],
        "confusion_matrix": eval_result["confusion_matrix"],
        "early_stop": {
            "enabled": bool(early_stop),
            "monitor": monitor,
            "best_epoch": int(best_epoch),
            "best_value": float(best_metric_val) if best_metric_val is not None else None
        }
    }


def train_vit(data_flag, info, num_epochs=10, learning_rate=0.001, device=None,
              train_loader=None, val_loader=None, test_loader=None, random_seed=None, 
              output_dir=None, run_name="run", scheduler=False, early_stop=True, 
              monitor='val_loss', patience=10, min_delta=0.001, restore_best=True, 
              checkpoint_best=False, class_weighting=True, use_dropout=False, dropout_rate=0.1):
    """
    Train Vision Transformer (ViT-B/16) on medMNIST dataset.
    Same training loop as train_resnet18 but for ViT architecture.
    """
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
    num_classes = len(info['label'])
    
    # Compute weights BEFORE wrapping loaders for CUDA prefetch
    ce_weight_cpu, bce_pos_weight_cpu = (None, None)
    if class_weighting:
        try:
            ce_weight_cpu, bce_pos_weight_cpu = _compute_class_weights_from_loader(train_loader, num_classes)
            if ce_weight_cpu is not None:
                print(f"Using CE class weights (mean=1): {ce_weight_cpu.tolist()}")
            if bce_pos_weight_cpu is not None:
                print(f"Using BCE pos_weight: {bce_pos_weight_cpu.item():.4f}")
        except Exception as e:
            print("Warning: failed to compute class weights, continuing unweighted:", e)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Wrap loaders with PrefetchLoader to overlap copies (only for CUDA)
    if device and 'cuda' in str(device).lower():
        train_loader = PrefetchLoader(train_loader, device)
        val_loader = PrefetchLoader(val_loader, device)
        test_loader = PrefetchLoader(test_loader, device)

    # Create ViT model with or without dropout
    if use_dropout:
        print(f"Using ViT-B/16 with Dropout (dropout_rate={dropout_rate})")
        model = ViTWithDropout(num_classes=num_classes, dropout_rate=dropout_rate, pretrained=True)
    else:
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        hidden_dim = model.hidden_dim
        if num_classes == 2:
            model.heads = nn.Linear(hidden_dim, 1)
        else:
            model.heads = nn.Linear(hidden_dim, num_classes)
    
    # Setup loss criterion with class weights
    if num_classes == 2:
        if bce_pos_weight_cpu is not None:
            criterion = BCEWithLogitsLoss(pos_weight=bce_pos_weight_cpu.to(device))
        else:
            criterion = BCEWithLogitsLoss()
    else:
        if ce_weight_cpu is not None:
            criterion = CrossEntropyLoss(weight=ce_weight_cpu.to(device))
        else:
            criterion = CrossEntropyLoss()
    
    model = model.to(device)

    # ViT typically benefits from lower learning rate than ResNet
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler is True:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses = []
    val_losses = []
    val_accs = []
    epoch_times = []
    if output_dir:
        figs_dir = os.path.join(output_dir, "figs")
        _ensure_dir(figs_dir)
        log_path = os.path.join(output_dir, "metrics.log")
        _append_log(log_path, f"=== {run_name} start: epochs={num_epochs}, lr={learning_rate}, model=ViT-B/16 ===")

    # Setup early stopper
    stopper = None
    best_state = None
    best_epoch = -1
    best_metric_val = None
    metric_mode = 'min' if monitor == 'val_loss' else 'max'
    if early_stop:
        if data_flag == 'tissuemnist':
            patience = 5
        stopper = EarlyStopper(mode=metric_mode, patience=patience, min_delta=min_delta)
    best_ckpt_path = os.path.join(output_dir, f"best_{run_name}.pt") if (output_dir and checkpoint_best) else None

    # Training loop (same as ResNet)
    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, dict):
                images, labels = batch['image'], batch['label']
            else:
                images, labels = batch[0], batch[1]
            
            if not isinstance(images, torch.Tensor):
                images = images.to(device)
            if not isinstance(labels, torch.Tensor):
                labels = labels.to(device)

            # Squeeze but preserve at least 1D (avoid 0-d tensors)
            labels = labels.squeeze()
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
            
            # Skip empty batches (edge case with certain dataset sizes)
            if images.size(0) == 0 or labels.size(0) == 0:
                continue
            
            # Skip mismatched batches
            if images.size(0) != labels.size(0):
                continue

            optimizer.zero_grad()
            outputs = model(images)
            if num_classes == 2:
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    images, labels = batch['image'], batch['label']
                else:
                    images, labels = batch[0], batch[1]
                
                if not isinstance(images, torch.Tensor):
                    images = images.to(device)
                if not isinstance(labels, torch.Tensor):
                    labels = labels.to(device)

                # Squeeze but preserve at least 1D (avoid 0-d tensors)
                labels = labels.squeeze()
                if labels.ndim == 0:
                    labels = labels.unsqueeze(0)
                
                # Skip empty batches (edge case with certain dataset sizes)
                if images.size(0) == 0 or labels.size(0) == 0:
                    continue
                
                # Skip mismatched batches
                if images.size(0) != labels.size(0):
                    continue
                
                outputs = model(images)
                
                if num_classes == 2:
                    outputs_sq = outputs.squeeze()
                    loss = criterion(outputs_sq, labels.float())
                    preds = (torch.sigmoid(outputs_sq) > 0.5).long()
                else:
                    loss = criterion(outputs, labels.long())
                    preds = torch.argmax(outputs, dim=1)
                
                val_loss += loss.item() * images.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if scheduler:
            scheduler.step()

        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {t_epoch:.2f}s")
        
        if output_dir:
            _append_log(log_path, f"Epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.4f} | "
                       f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | time={t_epoch:.2f}s")

        # Early stopping
        if stopper is not None:
            current_metric = val_loss if monitor == 'val_loss' else val_acc
            improved = stopper.step(current_metric)
            if improved:
                best_epoch = epoch
                best_metric_val = current_metric
                if restore_best:
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if best_ckpt_path:
                    torch.save(model.state_dict(), best_ckpt_path)
                    print(f" → Best model saved (epoch {epoch+1})")
            
            if stopper.should_stop():
                print(f"Early stopping triggered at epoch {epoch+1}")
                if output_dir:
                    _append_log(log_path, f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model if requested
    if restore_best and best_state is not None:
        # Move state dict back to model's device before loading
        best_state_device = {k: v.to(device) for k, v in best_state.items()}
        model.load_state_dict(best_state_device)
        print(f"Restored best model from epoch {best_epoch+1} (metric={best_metric_val:.4f})")

    # Save loss curves and training history
    if output_dir:
        # Loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Losses - {run_name}')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figs", f"loss_curve_{run_name}.png"), dpi=200)
        plt.close()

        # Val accuracy curve
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(f'Val Acc - {run_name}')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figs", f"val_acc_{run_name}.png"), dpi=200)
        plt.close()

        # Save training history
        history = {
            "run_name": run_name,
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses],
            "val_accs": [float(x) for x in val_accs],
            "epoch_times_sec": [float(x) for x in epoch_times],
            "early_stop": {
                "enabled": bool(early_stop),
                "monitor": monitor,
                "best_epoch": int(best_epoch),
                "best_value": float(best_metric_val) if best_metric_val is not None else None
            }
        }
        _save_json(history, os.path.join(output_dir, f"history_{run_name}.json"))

    # Final test evaluation with confusion matrix
    test_results = evaluate_model(model, test_loader, data_flag, device, output_dir, prefix=run_name, display_cm=True)
    
    if output_dir:
        _append_log(log_path, f"=== {run_name} end: test_acc={test_results['metrics']['accuracy']:.4f} ===")

    return model, {
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "val_accs": [float(x) for x in val_accs],
        "epoch_times": [float(x) for x in epoch_times],
        "test_results": test_results,
        "early_stopping": {
            "enabled": bool(early_stop),
            "monitor": monitor,
            "best_epoch": int(best_epoch),
            "best_value": float(best_metric_val) if best_metric_val is not None else None
        }
    }


def evaluate_model(model, test_loader, data_flag, device=None, output_dir=None, prefix="test", display_cm=True):
    if data_flag == 'dermamnist-e':
        info = DERMAMNIST_E_INFO
    else:
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
            y_device = y.to(device) if isinstance(y, torch.Tensor) else y

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

            # move labels to host safely and append
            y_true.append(y_device.detach().cpu().numpy())
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
            # Handle case where model predicts more classes than exist in test set
            # (e.g., AMOS test set only has 6 of 11 OrganaMNIST classes)
            present_classes = np.unique(y_true)
            if len(present_classes) < num_classes:
                # Filter y_score to only include present classes and renormalize
                y_score_filtered = y_score[:, present_classes]
                # Renormalize probabilities to sum to 1.0
                y_score_filtered = y_score_filtered / y_score_filtered.sum(axis=1, keepdims=True)
                auc = roc_auc_score(y_true, y_score_filtered, multi_class='ovr', average='macro', labels=present_classes)
            else:
                auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
    except Exception as e:
        print(f" AUC computation failed: {e}")
        print(f" y_true shape: {y_true.shape}, unique classes: {np.unique(y_true)}")
        print(f" y_score shape: {y_score.shape}, num_classes: {num_classes}")
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
    if output_dir and display_cm:
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
    
    elif output_dir is None and display_cm:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix ({prefix})")
        plt.tight_layout()

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


def load_models(flag, device, waugmentation=False, size=224, model_backbone='resnet18', setup=''):
    """
    Load trained models for a dataset.
    
    Args:
        flag: Dataset name (e.g., 'breastmnist')
        device: Device to load models on
        waugmentation: Legacy parameter (use setup='DA' instead)
        size: Image size (default: 224)
        model_backbone: 'resnet18' or 'vit_b_16'
        setup: Training setup - '' (standard), 'DA' (data augmentation), 'DO' (dropout), 'DADO' (both)
    
    Returns:
        List of 5 trained models (one per CV fold)
    """
    # Load dataset info
    data_flag = flag
    info = INFO[data_flag]
    num_classes = len(info['label'])
    
    # Determine filename pattern based on setup
    # Pattern: {flag}_{backbone}_224_randaug{0|1}_dropout{rate}_fold_{i}.pt
    if waugmentation and not setup:  # Legacy support
        setup = 'DA'
    
    if setup == 'DA':
        randaug = 1
        dropout_suffix = ''
    elif setup == 'DO':
        randaug = 0
        dropout_suffix = '_dropout03' if model_backbone == 'resnet18' else '_dropout01'
    elif setup == 'DADO':
        randaug = 1
        dropout_suffix = '_dropout03' if model_backbone == 'resnet18' else '_dropout01'
    else:  # Standard training
        randaug = 0
        dropout_suffix = ''
    
    # Load saved models
    models = []
    for i in range(5):
        # Initialize the model architecture
        if model_backbone == 'resnet18':
            if setup in ['DO', 'DADO']:
                # ResNet with dropout
                dropout_rate = 0.3
                model = ResNet18WithDropout(num_classes=num_classes, dropout_rate=dropout_rate, pretrained=True)
            else:
                # Standard ResNet
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
                if num_classes == 2:
                    model.fc = nn.Linear(model.fc.in_features, 1)
                else:
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        elif model_backbone == 'vit_b_16':
            if setup in ['DO', 'DADO']:
                # ViT with dropout
                dropout_rate = 0.1
                model = ViTWithDropout(num_classes=num_classes, dropout_rate=dropout_rate, pretrained=True)
            else:
                # Standard ViT
                model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
                hidden_dim = model.hidden_dim
                if num_classes == 2:
                    model.heads = nn.Linear(hidden_dim, 1)
                else:
                    model.heads = nn.Linear(hidden_dim, num_classes)
        else:
            raise ValueError(f"Unknown model_backbone: {model_backbone}")
        
        # Construct filename
        model_filename = f"{flag}_{model_backbone}_{size}_randaug{randaug}{dropout_suffix}_fold_{i}.pt"
        
        # Allow environment variable override for models directory
        models_base_dir = os.environ.get('MEDMNIST_MODELS_DIR')
        if models_base_dir:
            model_path = Path(models_base_dir) / f'{size}*{size}' / model_filename
        else:
            project_root = _get_project_root()
            model_path = project_root / 'medMNIST' / 'models' / f'{size}*{size}' / model_filename
        model_path = str(model_path)
        
        # Load the state dictionary – fall back to HuggingFace Hub when missing
        if not os.path.exists(model_path):
            try:
                import importlib.util as _ilu
                _hub_path = Path(__file__).resolve().parent / "hub.py"
                _spec = _ilu.spec_from_file_location("_failcatcher_hub", _hub_path)
                _hub = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_hub)
                model_path = str(_hub.ensure_model_file(model_filename, local_dir=Path(model_path).parent))
            except Exception as hub_err:
                raise FileNotFoundError(
                    f"Model file not found: {model_path}\n"
                    f"Expected pattern: {flag}_{model_backbone}_{size}_randaug{randaug}{dropout_suffix}_fold_{{0-4}}.pt\n"
                    f"Attempted Hub download but failed: {hub_err}\n"
                    f"To download all models at once run:  python scripts/setup_from_hub.py"
                ) from hub_err
        
        state_dict = torch.load(model_path, map_location=device)
        
        # Remove the 'model.' prefix from the state_dict keys if necessary
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        # Load the modified state dictionary into the model
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
    
    print(f"Loaded {len(models)} models: {model_backbone} with setup '{setup or 'standard'}' from {model_filename.rsplit('_', 1)[0]}_*")
    return models

def load_datasets(dataflag, color, im_size, transform, batch_size, cache_test=False, transform_test=None, test_subset='all'):
    """
    Load and prepare datasets for training and evaluation.
    
    Data Split Strategy:
    --------------------
    For organamnist:
        - Training set: Original medMNIST train split (no mixing)
        - Calibration set: Original medMNIST val split (no mixing)
        - Rationale: OrganAMNIST images are slices from 3D volumes. Using the original
          splits prevents data leakage from correlated slices appearing in both train and calib.
    
    For all other datasets:
        - Combines medMNIST train + val splits
        - Random 80/20 split: 80% for training (used in CV), 20% for calibration
        - Rationale: Other datasets don't have the same volumetric correlation issue
    
    Args:
        dataflag: Dataset name (e.g., 'breastmnist', 'organamnist')
        color: Whether dataset uses color images
        im_size: Image size
        transform: Transform to apply
        batch_size: Batch size for dataloaders
        cache_test: Whether to cache test set
        transform_test: Optional separate transform for test set
        test_subset: For dermamnist-e only. 'all' (default), 'id', or 'external'
    
    Returns:
        tuple: ([train_dataset, calibration_dataset, test_dataset], 
                [train_loader, calib_loader, test_loader], 
                info)
    """
    datasets, info = get_datasets(dataflag, im_size=im_size, color=color, transform=transform, transform_test=transform_test, test_subset=test_subset)
    # Combine train_dataset and val_dataset
    combined_train_dataset = ConcatDataset([datasets[0], datasets[1]])

    # Set the random seed for reproducibility
    torch.manual_seed(42)
    if dataflag != 'organamnist':
        # Calculate the sizes for training and calibration datasets
        train_size = int(0.8 * len(combined_train_dataset))
        calibration_size = len(combined_train_dataset) - train_size

        # Split the combined_train_dataset into training and calibration datasets
        train_dataset, calibration_dataset = random_split(combined_train_dataset, [train_size, calibration_size])
    else:
        # OrganAMNIST: Use original medMNIST splits to prevent volumetric data leakage
        train_dataset, calibration_dataset = datasets[0], datasets[1]
    test_dataset = datasets[2]  # Use the test dataset as is

    dataloaders = get_dataloaders([train_dataset, calibration_dataset, test_dataset], batch_size=batch_size, use_cache_test=cache_test)

    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Calibration dataset size: {len(calibration_dataset)}')
    
    return [train_dataset, calibration_dataset, test_dataset], dataloaders, info

def CV_train_val_loaders(study_dataset_aug, study_dataset_plain, batch_size,
                         n_splits=5, seed=42, use_monai=False, cache_rate=1.0, train_augment_transform=None, num_workers=None, pin_memory=True, prewarm_cache=False):
    """
    Create CV train/val DataLoaders with optional MONAI ThreadDataLoader and CacheDataset support.
    - use_monai: prefer MONAI ThreadDataLoader (if available)
    - use_cache: build MONAI CacheDataset per-fold (only when MONAI available)
    - cache_rate: fraction to cache in MONAI CacheDataset
    - train_augment_transform: callable applied on-the-fly to training items (RandAugment)
    
    NOTE: When using MONAI CacheDataset, persistent_workers is automatically set to False
          to avoid multiprocessing conflicts (CacheDataset has internal multiprocessing).
    """
    # decide num_workers if not provided
    if num_workers is None:
        try:
            env_n = os.environ.get("NUM_WORKERS")
            if env_n is not None:
                num_workers = int(env_n)
            else:
                n_cpu = os.cpu_count() or 1
                n_users = int(os.environ.get("SHARED_USERS", "4"))
                per_user = max(2, n_cpu // (n_users * 8))
                num_workers = int(min(max(per_user, 2), 16))
        except Exception:
            num_workers = 4
        print(f"CV loaders: using num_workers={num_workers}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Labels from the plain (non-augmented) view
    labels = [label for _, label in study_dataset_plain]

    train_loaders = []
    val_loaders = []
    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    use_monai_local = bool(use_monai) and MONAI_AVAILABLE

    def _build_data_list_from_subset(ds):
        data_list = []
        if isinstance(ds, torch.utils.data.Subset):
            base = ds.dataset
            indices = ds.indices
        else:
            base = ds
            indices = range(len(ds))
        for i in indices:
            item = base[i]
            if isinstance(item, dict) and 'image' in item and 'label' in item:
                img, lbl = item['image'], item['label']
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                img, lbl = item[0], item[1]
            else:
                raise RuntimeError("Unsupported dataset item format for caching.")
            if torch.is_tensor(img):
                data_list.append({'image_tensor': img.detach().cpu(), 'label': int(lbl)})
            else:
                data_list.append({'image_numpy': np.asarray(img), 'label': int(lbl)})
        return data_list

    for train_index, val_index in skf.split(np.zeros(len(labels)), labels):
        # Build subsets (indices are relative to the combined train set)
        if study_dataset_aug is not None:
            train_subset = torch.utils.data.Subset(study_dataset_aug, train_index)
        else:
            train_subset = torch.utils.data.Subset(study_dataset_plain, train_index)
        val_subset = torch.utils.data.Subset(study_dataset_plain, val_index)

        # If caching with MONAI is requested and available, build CacheDataset per-fold
        if use_monai_local:
            try:
                train_list = _build_data_list_from_subset(train_subset)
                val_list = _build_data_list_from_subset(val_subset)

                def _to_tensor_tuple(d):
                    # handle cached tensor or numpy entry without unnecessary roundtrip
                    if 'image_tensor' in d:
                        img = d['image_tensor']
                        if not isinstance(img, torch.Tensor):
                            img = torch.as_tensor(img)
                        img = img.float()
                    else:
                        img = torch.from_numpy(d['image_numpy']).float()
                    # when training uses RandAugment we want val cached as normalized tensors
                    if train_augment_transform is not None:
                        img = normalize(img)
                    lbl = torch.tensor(int(d['label']), dtype=torch.long)
                    return img, lbl
                def _to_train_cached(d):
                    # produce cached item for training: prefer to keep tensor when possible,
                    # but convert to PIL if we want to cache PIL for faster augment use
                    if 'image_tensor' in d:
                        img = d['image_tensor']
                        if not isinstance(img, torch.Tensor):
                            img = torch.as_tensor(img)
                        img = img.float()
                    else:
                        img = torch.from_numpy(d['image_numpy']).float()

                    if train_augment_transform is not None:
                        # cache as PIL to avoid repeated to_pil_image at runtime (optional)
                        img_pil = to_pil_image(torch.clamp(img, 0., 1.))
                        lbl = torch.tensor(int(d['label']), dtype=torch.long)
                        return img_pil, lbl
                    else:
                        lbl = torch.tensor(int(d['label']), dtype=torch.long)
                        return img, lbl
                train_cache_ds = MONAI_CacheDataset(data=train_list, transform=_to_train_cached, cache_rate=float(cache_rate))
                val_cache_ds = MONAI_CacheDataset(data=val_list, transform=_to_tensor_tuple, cache_rate=float(cache_rate))

                # wrap training cached dataset with augment transform if requested (augment should include Normalize)
                if train_augment_transform is not None:
                    class AugmentCachedDataset(torch.utils.data.Dataset):
                        """
                        Wrap a cached dataset and apply a (PIL-based) augment transform at runtime.
                        
                        wrapper will augment -> tensor -> normalize.
                        """
                        def __init__(self, cache_ds, augment):
                            self.cache_ds = cache_ds
                            self.augment = augment

                        def __len__(self):
                            return len(self.cache_ds)

                        def __getitem__(self, idx):
                            x, y = self.cache_ds[idx]
                            if train_augment_transform is not None and isinstance(x, Image.Image):
                                aug = self.augment(x)
                                x_aug = aug if torch.is_tensor(aug) else T.ToTensor()(aug)
                            else:
                                x = x.detach().cpu().float()
                                try:
                                    print("")
                                    x_aug = self.augment(x)
                                    if not torch.is_tensor(x_aug):
                                        x_aug = T.ToTensor()(x_aug)
                                except Exception:
                                    x_aug = self.augment(to_pil_image(torch.clamp(x, 0., 1.)))
                                    if not torch.is_tensor(x_aug):
                                        x_aug = T.ToTensor()(x_aug)
                            if train_augment_transform is not None:
                                x_aug = x_aug.float()
                            return x_aug, y
                    train_ds_wrapped = AugmentCachedDataset(train_cache_ds, train_augment_transform)
                else:    
                    train_ds_wrapped = train_cache_ds

                
                val_ds_wrapped = val_cache_ds
                # CRITICAL: persistent_workers=False when using MONAI CacheDataset
                # CacheDataset uses internal multiprocessing - persistent workers cause conflicts
                train_loader = DataLoader(dataset=train_ds_wrapped, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory,
                                          persistent_workers=False, prefetch_factor=2, drop_last=True)
                
                val_loader = DataLoader(val_ds_wrapped, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, persistent_workers=False, prefetch_factor=3)
                print('train/val loaders using torch DataLoader for cached dataset (persistent_workers=False)')

                if prewarm_cache and use_monai_local:
                    try:
                        t0 = time.time()
                        print("Pre-warming MONAI cache for this fold (this may take some time)...")
                        for _ in train_loader:
                            pass
                        print(f"Cache pre-warm done ({time.time()-t0:.1f}s)")
                    except Exception as e:
                        print("Pre-warm failed or was interrupted:", e)
            except Exception as e:
                print("MONAI CacheDataset failed for fold:", e, "- falling back to torch DataLoader.")
                persistent = True if (num_workers and num_workers > 0) else False
                train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent)
                val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False,
                                        num_workers=max(0, num_workers), pin_memory=pin_memory, persistent_workers=persistent)
        else:
            # no caching: optionally wrap train subset with runtime augment transform
            if train_augment_transform is not None:
                class AugmentDataset(torch.utils.data.Dataset):
                    def __init__(self, ds, augment):
                        self.ds = ds
                        self.augment = augment
                    def __len__(self):
                        return len(self.ds)
                    def __getitem__(self, idx):
                        item = self.ds[idx]
                        if isinstance(item, dict) and 'image' in item and 'label' in item:
                            x, y = item['image'], item['label']
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            x, y = item[0], item[1]
                        else:
                            raise RuntimeError("Unsupported dataset item format in AugmentDataset.")
                        try:
                            x_aug = self.augment(x)
                        except Exception:
                            from torchvision.transforms.functional import to_pil_image, to_tensor
                            if torch.is_tensor(x):
                                x_pil = to_pil_image(x)
                            else:
                                x_pil = x
                            x_aug = self.augment(x_pil)
                            if not torch.is_tensor(x_aug):
                                x_aug = to_tensor(x_aug)
                        return x_aug, y
                train_ds_wrapped = AugmentDataset(train_subset, train_augment_transform)
            else:
                train_ds_wrapped = train_subset

            # DataLoader for normal (non-cache) case
            persistent = True if (num_workers and num_workers > 0) else False
            train_loader = DataLoader(dataset=train_ds_wrapped, batch_size=batch_size, shuffle=True,
                                        num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent)
            val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False,
                                    num_workers=max(0, num_workers), pin_memory=pin_memory, persistent_workers=persistent)


        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    loader_type = "DataLoader w persistent workers + MONAI CacheDataset" if use_monai_local else "torch DataLoader"
    print(f"CV loaders created: {n_splits} folds using {loader_type} (num_workers={num_workers}, cache_rate={cache_rate})")
    return train_loaders, val_loaders

def get_single_CV_fold(study_dataset_aug, study_dataset_plain, batch_size, fold_index,
                       n_splits=5, seed=42, use_monai=False, cache_rate=1.0,
                       train_augment_transform=None, num_workers=None, pin_memory=True, prewarm_cache=False):
    """
    Get loaders for a single CV fold without iterating through all folds.
    This avoids caching all folds when you only need one.
    
    NOTE: When using MONAI CacheDataset, persistent_workers is automatically set to False
          to avoid multiprocessing conflicts (CacheDataset has internal multiprocessing).
    
    Returns: (train_loader, val_loader) for the specified fold_index
    """
    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError(f"fold_index must be in [0, {n_splits-1}], got {fold_index}")
    
    if num_workers is None:
        try:
            env_n = os.environ.get("NUM_WORKERS")
            if env_n is not None:
                num_workers = int(env_n)
            else:
                n_cpu = os.cpu_count() or 1
                n_users = int(os.environ.get("SHARED_USERS", "4"))
                per_user = max(2, n_cpu // (n_users * 8))
                num_workers = int(min(max(per_user, 2), 16))
        except Exception:
            num_workers = 4

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = [label for _, label in study_dataset_plain]
    use_monai_local = bool(use_monai) and MONAI_AVAILABLE
    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])

    def _build_data_list_from_subset(ds):
        data_list = []
        if isinstance(ds, torch.utils.data.Subset):
            base = ds.dataset
            indices = ds.indices
        else:
            base = ds
            indices = range(len(ds))
        for i in indices:
            item = base[i]
            if isinstance(item, dict) and 'image' in item and 'label' in item:
                img, lbl = item['image'], item['label']
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                img, lbl = item[0], item[1]
            else:
                raise RuntimeError("Unsupported dataset item format for caching.")
            if torch.is_tensor(img):
                data_list.append({'image_tensor': img.detach().cpu(), 'label': int(lbl)})
            else:
                data_list.append({'image_numpy': np.asarray(img), 'label': int(lbl)})
        return data_list

    # Find the target fold's indices
    for current_fold_idx, (train_index, val_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if current_fold_idx != fold_index:
            continue
        
        # Build subsets for this fold only
        if study_dataset_aug is not None:
            train_subset = torch.utils.data.Subset(study_dataset_aug, train_index)
        else:
            train_subset = torch.utils.data.Subset(study_dataset_plain, train_index)
        val_subset = torch.utils.data.Subset(study_dataset_plain, val_index)

        # Build loaders (same logic as CV_fold_generator)
        if use_monai_local:
            try:
                train_list = _build_data_list_from_subset(train_subset)
                val_list = _build_data_list_from_subset(val_subset)

                def _to_tensor_tuple(d):
                    if 'image_tensor' in d:
                        img = d['image_tensor']
                        if not isinstance(img, torch.Tensor):
                            img = torch.as_tensor(img)
                        img = img.float()
                    else:
                        img = torch.from_numpy(d['image_numpy']).float()
                    if train_augment_transform is not None:
                        img = normalize(img)
                    lbl = torch.tensor(int(d['label']), dtype=torch.long)
                    return img, lbl

                def _to_train_cached(d):
                    if 'image_tensor' in d:
                        img = d['image_tensor']
                        if not isinstance(img, torch.Tensor):
                            img = torch.as_tensor(img)
                        img = img.float()
                    else:
                        img = torch.from_numpy(d['image_numpy']).float()

                    if train_augment_transform is not None:
                        img_pil = to_pil_image(torch.clamp(img, 0., 1.))
                        lbl = torch.tensor(int(d['label']), dtype=torch.long)
                        return img_pil, lbl
                    else:
                        lbl = torch.tensor(int(d['label']), dtype=torch.long)
                        return img, lbl

                train_cache_ds = MONAI_CacheDataset(data=train_list, transform=_to_train_cached, cache_rate=float(cache_rate))
                val_cache_ds = MONAI_CacheDataset(data=val_list, transform=_to_tensor_tuple, cache_rate=float(cache_rate))

                # runtime augment wrapper if requested
                if train_augment_transform is not None:
                    class AugmentCachedDataset(torch.utils.data.Dataset):
                        def __init__(self, cache_ds, augment):
                            self.cache_ds = cache_ds
                            self.augment = augment

                        def __len__(self):
                            return len(self.cache_ds)

                        def __getitem__(self, idx):
                            x, y = self.cache_ds[idx]
                            if train_augment_transform is not None and isinstance(x, Image.Image):
                                aug = self.augment(x)
                                x_aug = aug if torch.is_tensor(aug) else T.ToTensor()(aug)
                            else:
                                x = x.detach().cpu().float()
                                try:
                                    x_aug = self.augment(x)
                                    if not torch.is_tensor(x_aug):
                                        x_aug = T.ToTensor()(x_aug)
                                except Exception:
                                    x_aug = self.augment(to_pil_image(torch.clamp(x, 0., 1.)))
                                    if not torch.is_tensor(x_aug):
                                        x_aug = T.ToTensor()(x_aug)
                            if train_augment_transform is not None:
                                x_aug = x_aug.float()
                            return x_aug, y
                    train_ds_wrapped = AugmentCachedDataset(train_cache_ds, train_augment_transform)
                else:
                    train_ds_wrapped = train_cache_ds

                val_ds_wrapped = val_cache_ds
                # CRITICAL: persistent_workers=False when using MONAI CacheDataset
                # CacheDataset uses internal multiprocessing - persistent workers cause conflicts
                train_loader = DataLoader(dataset=train_ds_wrapped, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory,
                                          persistent_workers=False, prefetch_factor=2, drop_last=True)
                val_loader = DataLoader(dataset=val_ds_wrapped, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False, prefetch_factor=3)

                if prewarm_cache:
                    print(f"Prewarming cache for fold {fold_index}...")
                    try:
                        for _ in train_loader:
                            pass
                    except Exception:
                        pass

            except Exception as e:
                print(f"Warning: MONAI caching failed for fold {fold_index}, falling back to plain DataLoader: {e}")
                # fallback to plain DataLoader
                persistent = True if (num_workers and num_workers > 0) else False
                train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent)
                val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False,
                                        num_workers=max(0, num_workers), pin_memory=pin_memory, persistent_workers=persistent)
        else:
            if train_augment_transform is not None:
                class AugmentDataset(torch.utils.data.Dataset):
                    def __init__(self, ds, augment):
                        self.ds = ds
                        self.augment = augment
                    def __len__(self):
                        return len(self.ds)
                    def __getitem__(self, idx):
                        item = self.ds[idx]
                        if isinstance(item, dict) and 'image' in item and 'label' in item:
                            x, y = item['image'], item['label']
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            x, y = item[0], item[1]
                        else:
                            raise RuntimeError("Unsupported dataset item format in AugmentDataset.")
                        try:
                            x_aug = self.augment(x)
                        except Exception:
                            from torchvision.transforms.functional import to_pil_image, to_tensor
                            if torch.is_tensor(x):
                                x_pil = to_pil_image(x)
                            else:
                                x_pil = x
                            x_aug = self.augment(x_pil)
                            if not torch.is_tensor(x_aug):
                                x_aug = to_tensor(x_aug)
                        return x_aug, y
                train_ds_wrapped = AugmentDataset(train_subset, train_augment_transform)
            else:
                train_ds_wrapped = train_subset

            persistent = True if (num_workers and num_workers > 0) else False
            train_loader = DataLoader(dataset=train_ds_wrapped, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent)
            val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False,
                                    num_workers=max(0, num_workers), pin_memory=pin_memory, persistent_workers=persistent)

        loader_type = "MONAI CacheDataset" if use_monai_local else "torch DataLoader"
        print(f"Created fold {fold_index}/{n_splits-1} loaders: {loader_type} (num_workers={num_workers}, cache_rate={cache_rate})")
        return train_loader, val_loader

    raise RuntimeError(f"Fold {fold_index} not found (should not happen)")

def CV_fold_generator(study_dataset_aug, study_dataset_plain, batch_size,
                      n_splits=5, seed=42, use_monai=False, cache_rate=1.0,
                      train_augment_transform=None, num_workers=None, pin_memory=True, prewarm_cache=False):
    """
    Generator that yields (train_loader, val_loader, fold_index) for each CV fold.
    Build and return one fold at a time so caller can free memory after training that fold.
    Same parameters/behavior as CV_train_val_loaders but lazily constructs per-fold loaders.
    
    NOTE: When using MONAI CacheDataset, persistent_workers is automatically set to False
          to avoid multiprocessing conflicts (CacheDataset has internal multiprocessing).
    """
    if num_workers is None:
        try:
            env_n = os.environ.get("NUM_WORKERS")
            if env_n is not None:
                num_workers = int(env_n)
            else:
                n_cpu = os.cpu_count() or 1
                n_users = int(os.environ.get("SHARED_USERS", "4"))
                per_user = max(2, n_cpu // (n_users * 8))
                num_workers = int(min(max(per_user, 2), 16))
        except Exception:
            num_workers = 4

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = [label for _, label in study_dataset_plain]
    use_monai_local = bool(use_monai) and MONAI_AVAILABLE
    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])

    def _build_data_list_from_subset(ds):
        data_list = []
        if isinstance(ds, torch.utils.data.Subset):
            base = ds.dataset
            indices = ds.indices
        else:
            base = ds
            indices = range(len(ds))
        for i in indices:
            item = base[i]
            if isinstance(item, dict) and 'image' in item and 'label' in item:
                img, lbl = item['image'], item['label']
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                img, lbl = item[0], item[1]
            else:
                raise RuntimeError("Unsupported dataset item format for caching.")
            if torch.is_tensor(img):
                data_list.append({'image_tensor': img.detach().cpu(), 'label': int(lbl)})
            else:
                data_list.append({'image_numpy': np.asarray(img), 'label': int(lbl)})
        return data_list

    for fold_idx, (train_index, val_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        train_loader = None
        val_loader = None
        train_cache_ds = None
        val_cache_ds = None
        train_list = None
        val_list = None
        try:
            # build subsets
            if study_dataset_aug is not None:
                train_subset = torch.utils.data.Subset(study_dataset_aug, train_index)
            else:
                train_subset = torch.utils.data.Subset(study_dataset_plain, train_index)
            val_subset = torch.utils.data.Subset(study_dataset_plain, val_index)

            # Build loaders for this fold (reuse the same logic as CV_train_val_loaders)
            if use_monai_local:
                try:
                    train_list = _build_data_list_from_subset(train_subset)
                    val_list = _build_data_list_from_subset(val_subset)

                    def _to_tensor_tuple(d):
                        if 'image_tensor' in d:
                            img = d['image_tensor']
                            if not isinstance(img, torch.Tensor):
                                img = torch.as_tensor(img)
                            img = img.float()
                        else:
                            img = torch.from_numpy(d['image_numpy']).float()
                        if train_augment_transform is not None:
                            img = normalize(img)
                        lbl = torch.tensor(int(d['label']), dtype=torch.long)
                        return img, lbl

                    def _to_train_cached(d):
                        if 'image_tensor' in d:
                            img = d['image_tensor']
                            if not isinstance(img, torch.Tensor):
                                img = torch.as_tensor(img)
                            img = img.float()
                        else:
                            img = torch.from_numpy(d['image_numpy']).float()

                        if train_augment_transform is not None:
                            img_pil = to_pil_image(torch.clamp(img, 0., 1.))
                            lbl = torch.tensor(int(d['label']), dtype=torch.long)
                            return img_pil, lbl
                        else:
                            lbl = torch.tensor(int(d['label']), dtype=torch.long)
                            return img, lbl

                    train_cache_ds = MONAI_CacheDataset(data=train_list, transform=_to_train_cached, cache_rate=float(cache_rate))
                    val_cache_ds = MONAI_CacheDataset(data=val_list, transform=_to_tensor_tuple, cache_rate=float(cache_rate))

                    # runtime augment wrapper if requested
                    if train_augment_transform is not None:
                        class AugmentCachedDataset(torch.utils.data.Dataset):
                            def __init__(self, cache_ds, augment):
                                self.cache_ds = cache_ds
                                self.augment = augment

                            def __len__(self):
                                return len(self.cache_ds)

                            def __getitem__(self, idx):
                                x, y = self.cache_ds[idx]
                                if train_augment_transform is not None and isinstance(x, Image.Image):
                                    aug = self.augment(x)
                                    x_aug = aug if torch.is_tensor(aug) else T.ToTensor()(aug)
                                else:
                                    x = x.detach().cpu().float()
                                    try:
                                        x_aug = self.augment(x)
                                        if not torch.is_tensor(x_aug):
                                            x_aug = T.ToTensor()(x_aug)
                                    except Exception:
                                        x_aug = self.augment(to_pil_image(torch.clamp(x, 0., 1.)))
                                        if not torch.is_tensor(x_aug):
                                            x_aug = T.ToTensor()(x_aug)
                                if train_augment_transform is not None:
                                    x_aug = x_aug.float()
                                return x_aug, y
                        train_ds_wrapped = AugmentCachedDataset(train_cache_ds, train_augment_transform)
                    else:
                        train_ds_wrapped = train_cache_ds

                    val_ds_wrapped = val_cache_ds
                    # CRITICAL: persistent_workers=False when using MONAI CacheDataset
                    # CacheDataset uses internal multiprocessing - persistent workers cause conflicts
                    train_loader = DataLoader(dataset=train_ds_wrapped, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=pin_memory,
                                              persistent_workers=False, prefetch_factor=2, drop_last=True)
                    val_loader = DataLoader(dataset=val_ds_wrapped, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False, prefetch_factor=3)

                    if prewarm_cache:
                        try:
                            for _ in train_loader:
                                pass
                        except Exception:
                            pass

                except Exception:
                    # fallback to plain DataLoader
                    persistent = True if (num_workers and num_workers > 0) else False
                    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent)
                    val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False,
                                            num_workers=max(0, num_workers), pin_memory=pin_memory, persistent_workers=persistent)
                    train_cache_ds = None
                    val_cache_ds = None
            else:
                if train_augment_transform is not None:
                    class AugmentDataset(torch.utils.data.Dataset):
                        def __init__(self, ds, augment):
                            self.ds = ds
                            self.augment = augment
                        def __len__(self):
                            return len(self.ds)
                        def __getitem__(self, idx):
                            item = self.ds[idx]
                            if isinstance(item, dict) and 'image' in item and 'label' in item:
                                x, y = item['image'], item['label']
                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                x, y = item[0], item[1]
                            else:
                                raise RuntimeError("Unsupported dataset item format in AugmentDataset.")
                            try:
                                x_aug = self.augment(x)
                            except Exception:
                                from torchvision.transforms.functional import to_pil_image, to_tensor
                                if torch.is_tensor(x):
                                    x_pil = to_pil_image(x)
                                else:
                                    x_pil = x
                                x_aug = self.augment(x_pil)
                                if not torch.is_tensor(x_aug):
                                    x_aug = to_tensor(x_aug)
                            return x_aug, y
                    train_ds_wrapped = AugmentDataset(train_subset, train_augment_transform)
                else:
                    train_ds_wrapped = train_subset

                persistent = True if (num_workers and num_workers > 0) else False
                train_loader = DataLoader(dataset=train_ds_wrapped, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin_memory, drop_last=True, persistent_workers=persistent)
                val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False,
                                        num_workers=max(0, num_workers), pin_memory=pin_memory, persistent_workers=persistent)

            yield fold_idx, train_loader, val_loader
        finally:
            for dl in (train_loader, val_loader):
                if dl is None:
                    continue
                it = getattr(dl, "_iterator", None)
                if it is not None:
                    try:
                        it._shutdown_workers()
                    except Exception:
                        pass
                _clear_cache_dataset(getattr(dl, "dataset", dl))

            for ds in (train_cache_ds, val_cache_ds):
                _clear_cache_dataset(ds)

            del train_loader, val_loader, train_cache_ds, val_cache_ds, train_list, val_list
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        