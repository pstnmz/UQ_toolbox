"""Batch launcher for ViT-B/16 training runs.

Edit the lists below to define which training configurations to run.
Each entry in the four lists corresponds to a single training job:
  - flags:           dataset name (medMNIST flag)
  - colors:          True if the dataset has 3-channel colour images
  - use_randaugment: enable RandAugment data augmentation
  - use_dropouts:    enable MC-Dropout (adds a dropout layer before the classifier)

The four benchmark setups used in the paper are:
  (DA=False, DO=False) -> standard
  (DA=True,  DO=False) -> DA
  (DA=False, DO=True)  -> DO
  (DA=True,  DO=True)  -> DADO
"""
import subprocess, shlex
from pathlib import Path

# --- Datasets to train (one entry per job) ---
flags = ['organamnist', 'organamnist', 'organamnist', 'organamnist']
colors = [False, False, False, False]  # True for colour datasets (dermamnist, pathmnist, bloodmnist)
use_randaugment = [False, True, False, True]  # enable/disable RandAugment
use_dropouts = [False, False, True, True]      # enable/disable MC-Dropout

# --- Shared hyperparameters ---
dropout_rate = 0.1      # dropout rate (0.1 for ViT; lower than ResNet to avoid under-fitting)
learning_rate = 0.0001  # ViT benefits from a lower LR than ResNet-18
num_epochs = 100
batch_size = 128        # ViT uses a smaller batch size due to memory footprint
cuda = "cuda:2"        # target CUDA device

python = "/home/psteinmetz/venvs/venv_medMNIST/bin/python3.12"  # or path to your venv python
script_path = Path(__file__).parent / 'train_vit_medMNIST.py'

for f, c, r, d in zip(flags, colors, use_randaugment, use_dropouts):
    cmd = f"{python} {script_path} --flag {shlex.quote(f)} --color {str(c)} --batch_size {str(batch_size)} --use_randaugment {str(r)} --use_dropout {str(d)} --dropout_rate {str(dropout_rate)} --learning_rate {str(learning_rate)} --num_epochs {str(num_epochs)} --cuda {shlex.quote(cuda)}"
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)
