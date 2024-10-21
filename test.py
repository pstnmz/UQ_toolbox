import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import datatable as dt
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import UQ_toolbox as uq
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import math


try:
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs available:", num_gpus)
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print("Error while checking GPUs:", e)
    
print(torch.cuda.is_available())
device = torch.device("cuda:1")

# Define your CNN model
class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*12*12, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x= self.dropout_conv(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class ToTensor(object):

    def __call__(self, sample):
        image, shape, img_name = sample['image'], sample['shape'], sample['name']
        if shape=='Round':
            shape=0
        elif shape=='Irregular':
            shape=1
        elif shape=='Ambiguous':
            shape=2
        return {'image': torch.from_numpy(image).unsqueeze(0),
                'shape': torch.from_numpy(np.asarray(shape)),
                'name': img_name}
        
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        image, shape, img_name = sample['image'], sample['shape'], sample['name']
        norm = transforms.Normalize(mean=self.mean, std=self.std)
        return {'image': norm(image.float()),
                'shape': shape,            
                'name': img_name}
        
# Load the models
model_paths = ['model_shape_0_augmented.pt', 'model_shape_1_augmented.pt', 'model_shape_2_augmented.pt', 'model_shape_3_augmented.pt', 'model_shape_4_augmented.pt']
models_list = []
path_to_models = './models/'
for path in model_paths:
    model = simpleNet()
    model.load_state_dict(torch.load(path_to_models + path))
    model.eval()  # Set the model to evaluation mode
    models_list.append(model)
    
    
class AxialCutsDataset(Dataset):

    def __init__(self, data_shape, transform=None, for_trainning=False, mean=False, std=False, downsample=False):
        self.data = data_shape
        self.transform=transform
        self.for_trainning = for_trainning
        self.mean= mean
        self.std = std
        self.downsample = downsample

        if self.downsample:
            df_majority = self.data[self.data.iloc[:, 1]=='Irregular']
            df_majority_downsampled = resample(df_majority, replace=False, n_samples=1200, random_state=125)
            self.data = pd.concat((self.data[self.data['Shape'] != 'Irregular'], df_majority_downsampled))

    def __len__(self):
        return len(self.data.iloc[:, 0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data.iloc[idx, 0]
        image = io.imread(img_name)
        shape = self.data.iloc[idx, 1]
        sample = {'image': image, 'shape': shape, 'name':img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
# Define functions for evaluation metrics
def accuracy(outputs, labels):
    preds = outputs > 0.5 
    return accuracy_score(labels, preds)

def f1(outputs, labels):
    preds = outputs > 0.5
    return f1_score(labels, preds, average='binary')

def calculate_sensitivity(outputs, labels):
    preds = outputs > 0.5
    return recall_score(labels, preds, average='binary')

def roc_auc(outputs, labels):
    probs = outputs
    return roc_auc_score(labels, probs)

def compute_confusion_matrix(outputs, labels):
    preds = outputs > 0.5
    return confusion_matrix(labels, preds)

def calculate_specificity(cm):
    TN = cm[0, 0]  # True negatives
    FP = cm[0, 1]  # False positives
    return TN / (TN + FP)

def display_confusion_matrix(cm):
    # Define class names
    class_names = ['Round', 'Irregular']
    
    # Create a DataFrame for better visualization
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 20})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
def calculate_val_metrics(all_preds, all_labels):

    # Compute evaluation metrics
    acc = accuracy(all_preds, all_labels)
    f1_result = f1(all_preds, all_labels)
    roc_auc_result = roc_auc(all_preds, all_labels)
    cm = compute_confusion_matrix(all_preds, all_labels)
    display_confusion_matrix(cm)
    sensitivity_value = calculate_sensitivity(all_preds, all_labels)
    specificity_value = calculate_specificity(cm)

    print('Accuracy: {:.6f} \tF1 Score: {:.6f} \tROC AUC: {:.6f} \tSpecificity: {:.6f} \tSensitivity: {:.6f}'.format(
        acc, f1_result, roc_auc_result, specificity_value, sensitivity_value))
    
    
# Function to make predictions
def predict(models, image):
    image = image.to(device)
    predictions = [model(image).cpu().detach().numpy() for model in models]
    return predictions

mean = 87.42158495776914
std = 29.82248099334633

images_path = '/mnt/data/psteinmetz/neotex/data_CNN/model_evaluation/evaluation_IRM_villes/'
data = pd.concat(
    (
        pd.DataFrame(glob.glob(f'{images_path}*/*png')),
        pd.DataFrame(
            [x.split('/')[-1][:-4] for x in glob.glob(f'{images_path}*/*png')]
        ),
        pd.DataFrame(
            [k.split('/')[-2] for k in glob.glob(f'{images_path}*/*png')]
        ),
    ),
    axis=1,
)
data.columns = ['Path', 'ID', 'Shape']
data.set_index('ID', inplace=True)


axialcuts_dataset_eval = AxialCutsDataset(data_shape=data, downsample=False)
data_without_amb = axialcuts_dataset_eval.data[axialcuts_dataset_eval.data['Shape']!='Ambiguous']
data_amb = axialcuts_dataset_eval.data[axialcuts_dataset_eval.data['Shape']=='Ambiguous']

data_transforms = transforms.Compose([
    ToTensor(),
    Normalize(mean=mean, std=std)
])

data_without_amb = AxialCutsDataset(data_shape=data_without_amb, downsample=False, transform=data_transforms)
data_amb = AxialCutsDataset(data_shape=data_amb, downsample=False, transform=data_transforms)
eval_data = DataLoader(data_without_amb, num_workers=12, batch_size=128, shuffle=False)
eval_data_amb = DataLoader(data_amb, num_workers=12, batch_size=128, shuffle=False)
axialcuts_dataset_eval_all = AxialCutsDataset(data_shape=data, downsample=False, transform=data_transforms)
eval_all_data = DataLoader(axialcuts_dataset_eval_all, num_workers=12, batch_size=128, shuffle=False)
axialcuts_dataset_eval_all_gps = AxialCutsDataset(data_shape=data, downsample=False, transform=False)
eval_all_data_gps = DataLoader(axialcuts_dataset_eval_all_gps, num_workers=12, batch_size=128, shuffle=False)
models = [model.to(device) for model in models_list]

# Store results
all_results = []
mean_pred = []
true_labels = []
models = [model.to(device) for model in models_list]
# Inference
with torch.no_grad():
    for batch in eval_data:
        images = batch['image']
        labels = batch['shape']

        pred_probs = predict(models, images)
        mean_probs = np.mean(pred_probs, axis=0)
        std_probs = np.std(pred_probs, axis=0) 
        
        if eval_data.batch_size == 1:
            # Collect the results
            mean_pred.append(mean_probs.item())
            true_labels.append(labels.item())
            all_results.append({
                'true_label': labels.item(),
                'predicted_probabilities': pred_probs,
                'predicted_class': int(mean_probs > 0.5),
                'std': std_probs,
                'mean': mean_probs
            })
        else:
            for i in range(len(labels)):
                mean_pred.append(mean_probs[i])
                true_labels.append(labels[i].item())
                all_results.append({
                    'true_label': labels[i].item(),
                    'predicted_probabilities': [pred_probs[k][i] for k in range(len(models))],  # Pred probs for the i-th sample over 5 models
                    'predicted_class': int(mean_probs[i] > 0.5),
                    'std': float(std_probs[i]),
                    'mean': float(mean_probs[i])
                })
                
good_idx = [k for k in range(len(all_results)) if all_results[k]['true_label'] == all_results[k]['predicted_class']]
bad_idx = [k for k in range(len(all_results)) if all_results[k]['true_label'] != all_results[k]['predicted_class']]

tta_transform = transforms.Compose([
                    #AddBatchDimension(),
                    transforms.ToPILImage(),
                    transforms.RandAugment(2, 9),
                    transforms.PILToTensor(),
                    transforms.Lambda(lambda x: x.float()), 
                    transforms.Normalize(mean=mean, std=std)
                ])

data_without_amb_TTA = axialcuts_dataset_eval.data[axialcuts_dataset_eval.data['Shape']!='Ambiguous']
data_without_amb_TTA = AxialCutsDataset(data_shape=data_without_amb_TTA, downsample=False)
eval_data_TTA = DataLoader(data_without_amb_TTA, num_workers=12, batch_size=1, shuffle=False)

ll_results = []
mean_pred = []
true_labels = []
stds_tta = []
models = [model.to(device) for model in models_list]
tta_pred, stds_tta = uq.TTA(tta_transform, models, eval_data_TTA, device, nb_augmentations=5)
#uq.UQ_method_plot([stds_tta[k] for k in good_idx], [stds_tta[j] for j in bad_idx], 'Stds', 'Test Time Augmentation (n=50)')

fpr_std_tta, tpr_std_tta, auc_std_tta = uq.roc_curve_UQ_method_computation([stds_tta[k] for k in good_idx], [stds_tta[j] for j in bad_idx])
print(auc_std_tta)