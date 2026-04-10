"""Train a ResNet-18 with 5-fold cross-validation on a medMNIST dataset.

For each CV fold the script:
  1. Builds MONAI-cached train/val DataLoaders (optional RandAugment at runtime).
  2. Trains a ResNet-18 (with optional MC-Dropout head) using cross-entropy / BCE loss,
     a cosine-annealing LR scheduler, and early stopping on validation accuracy.
  3. Saves the fold checkpoint and, after all folds, writes JSON results summaries.

Run via the batch launcher (``launcher_resnet_training.py``) or directly::

    python train_resnet18_medMNIST.py \\
        --flag breastmnist --color False \\
        --batch_size 128 --num_epochs 100 \\
        --use_randaugment True --use_dropout False
"""
import sys
import os
# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import train_models_load_datasets as tr
from torchvision import transforms
import torch
import json, time
import argparse
import gc

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

# CLI args
parser = argparse.ArgumentParser(description="Train ResNet18 on medMNIST dataset")
parser.add_argument("--flag", type=str, default=None, help="Dataset flag to run (e.g. pneumoniamnist). If omitted runs default list)")
parser.add_argument("--color", type=str2bool, nargs='?', const=True, default=False, help="Use color images (True/False)")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--use_randaugment", type=str2bool, nargs='?', const=True, default=False, help="Use RandAugment (True/False)")
parser.add_argument("--use_dropout", type=str2bool, nargs='?', const=True, default=False, help="Use Dropout layers for MC Dropout (True/False)")
parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate (default: 0.5)")
parser.add_argument("--cuda", type=str, default="cuda:2", help="CUDA device string")
parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
args = parser.parse_args()

default_flags = ['pneumoniamnist', 'breastmnist']
default_colors = [False, False]
default_batch_sizes = [128, 128]
default_use_randaugments = [False, False]
default_num_epochs = 100
default_cuda = "cuda:2"

# override defaults with CLI args
cuda = args.cuda
num_epochs = args.num_epochs
batch_size_arg = args.batch_size
use_randaugment_arg = args.use_randaugment
use_dropout_arg = args.use_dropout
dropout_rate_arg = args.dropout_rate
color_arg = args.color

def _clear_monai_cache(obj):
    """Best-effort clear for MONAI CacheDataset internals and dataloader wrappers."""
    # unwrap dataloader
    ds = getattr(obj, "dataset", obj)
    # handle lists/tuples of datasets
    if isinstance(ds, (list, tuple)):
        for d in ds:
            _clear_monai_cache(d)
        return
    # call clear_cache if provided
    if hasattr(ds, "clear_cache"):
        try:
            ds.clear_cache()
        except Exception:
            pass
    # wipe common internal attributes (best-effort)
    for attr in ("_cache", "_cached", "cache"):
        if hasattr(ds, attr):
            try:
                val = getattr(ds, attr)
                if isinstance(val, dict):
                    val.clear()
                else:
                    setattr(ds, attr, None)
            except Exception:
                pass

def shutdown_dataloader_workers_and_clear_MONAI_cache(dl):
    # Remove references for this fold's loaders/datasets so GC can free memory
    try:
        # best-effort clear MONAI caches
        _clear_monai_cache(dl)
    except Exception:
        print("Warning: failed to clear MONAI cache for dataloader")
        pass

    # try to shutdown data loader workers so they don't keep references
    try:
        it = getattr(dl, "_iterator", None)
        if it is not None:
            try:
                it._shutdown_workers()
            except Exception:
                print("Warning: failed to shutdown dataloader workers")
                pass
    except Exception:
        print("Warning: failed to access dataloader iterator")
        pass

    # best-effort clear MONAI caches and delete refs
    try:
        _clear_monai_cache(dl)
    except Exception:
        pass

    dl = None

    # force GC and free CUDA pinned memory
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # drop any other large references you don't need per-fold
    # e.g., if you stored fold-specific dataset objects, del them here

    # force python GC and release CUDA memory
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        print("Warning: failed to empty CUDA cache for dataloader")
        pass


# if --flag passed, run only that dataset in this process
if args.flag:
    flags = [args.flag]
    colors = [color_arg]
    batch_sizes = [batch_size_arg]
    use_randaugments = [use_randaugment_arg]
    num_epochs = num_epochs
else:
    flags = default_flags
    colors = default_colors
    batch_sizes = default_batch_sizes
    use_randaugments = default_use_randaugments
    num_epochs = default_num_epochs
    cuda = default_cuda

for flag, color, batch_size, use_randaugment in zip(flags, colors, batch_sizes, use_randaugments):
    print(f"Training on {flag} with color={color}, batch_size={batch_size}, use_dropout={use_dropout_arg}, dropout_rate={dropout_rate_arg}")
    
    randaugment_ops = 2            # number of ops per image
    randaugment_mag = 9            # magnitude (0-10 typical)
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
    size = 224  # Image size for the models

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dropout_suffix = f"_dropout{dropout_rate_arg}" if use_dropout_arg else ""
    # Get the absolute path to benchmarks/medMNIST/runs
    script_dir = os.path.dirname(os.path.abspath(__file__))  # trainings/
    benchmarks_dir = os.path.dirname(script_dir)  # benchmarks/medMNIST/
    runs_dir = os.path.join(benchmarks_dir, "runs")
    exp_dir = os.path.join(runs_dir, flag, f"resnet18_{size}_{timestamp}_randaug{int(use_randaugment)}_numepochs{num_epochs}_bs{batch_size}{dropout_suffix}")
    os.makedirs(os.path.join(exp_dir, "figs"), exist_ok=True)

    if color is True:
        normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        if use_randaugment:
            transform_base = transforms.Compose([transforms.ToTensor()])
            transform_train = transforms.Compose([
                transforms.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_mag),
                transforms.ToTensor(),
                normalize,
            ])
            transform_eval = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            # cache normalized tensors (normalization performed once at cache time)
            transform_base = transforms.Compose([transforms.ToTensor(), normalize])
            transform_train = None
            transform_eval = transform_base

    else:
        normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        if use_randaugment:
            transform_base = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3,1,1))
            ])
            # runtime training augment only (no Normalize here)
            transform_train = transforms.Compose([
                transforms.RandAugment(num_ops=randaugment_ops, magnitude=randaugment_mag),
                transforms.ToTensor(),
                normalize
            ])
            transform_eval = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                normalize
            ])
        else:
            # cache normalized tensors (repeat then normalize once)
            transform_base = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1)), normalize])
            transform_train = None
            transform_eval = transform_base


    # Build datasets using transform_base (cached transform). When use_randaugment == False
    # transform_base already includes Normalize so cache will be normalized once.
    [study_dataset_plain, _, test_dataset], [_, _, test_loader], info = tr.load_datasets(flag, color, size, transform_base, batch_size, cache_test=True, transform_test=transform_eval)

    # Decide loader/cache strategy
    use_monai = True       # set False to disable MONAI CacheDataset
    cache_rate = 1.0              # fraction of items to cache (0..1)
    num_workers = num_workers_val= 8            # or set explicit int / use NUM_WORKERS env var

    models = []
    results = []
    fold_gen = tr.CV_fold_generator(
        None,
        study_dataset_plain,
        batch_size=batch_size,
        n_splits=5,
        seed=42,
        use_monai=use_monai,
        cache_rate=cache_rate,
        train_augment_transform=transform_train if use_randaugment else None,
        num_workers=num_workers,
        pin_memory=True,
        prewarm_cache=True
    )


    for fold_idx, train_loader, val_loader in fold_gen:
        print('MODEL ' + str(fold_idx))
        model, res = tr.train_resnet18(
            flag,
            info,
            num_epochs=num_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=0.001,
            device=cuda,
            random_seed=42,
            output_dir=exp_dir,
            run_name=f"fold_{fold_idx}",
            scheduler=True,
            use_dropout=use_dropout_arg,
            dropout_rate=dropout_rate_arg
        )
        models.append(model)
        results.append(res)
        
        # Save model immediately after training this fold
        model_path = os.path.join(exp_dir, f'resnet18_{flag}_224_{fold_idx}.pt')
        tr.save_model(model, path=model_path)
        print(f"Saved fold {fold_idx}: {model_path}")

        # Shutdown dataloader workers and clear MONAI cache to free memory
        shutdown_dataloader_workers_and_clear_MONAI_cache(train_loader)
        shutdown_dataloader_workers_and_clear_MONAI_cache(val_loader)
        

    # Save per-fold results summary
    with open(os.path.join(exp_dir, "results_folds.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Evaluate ensemble and save
    ensemble_res = tr.evaluate_model(model=models, test_loader=test_loader, data_flag=flag, device=cuda, output_dir=exp_dir, prefix="ensemble")
    with open(os.path.join(exp_dir, "results_ensemble.json"), "w") as f:
        json.dump(ensemble_res, f, indent=2)

    print(f"All models saved in: {exp_dir}")