import UQ_toolbox as uq
from medMNIST.utils import train_load_datasets_resnet as tr
import torch


dataflag = 'octmnist'
color = False # True for color, False for grayscale
activation = 'softmax'
batch_size = 4000
im_size = 224
models = tr.load_models(dataflag)
calibration_loader_for_tta, calibration_dataset=tr.load_datasets(dataflag, color, batch_size, im_size)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  

uq.apply_randaugment_and_store_results(calibration_dataset, models, 2, 45, 500, device, folder_name=f'/mnt/data/psteinmetz/archive_notebooks/Documents/medMNIST/gps_augment/{im_size}*{im_size}/{dataflag}_calibration_set', image_normalization=True, mean=[.5], std=[.5], image_size=im_size, nb_channels=3, output_activation=activation, batch_size=batch_size)