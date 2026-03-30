import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np

# Dataset statistics extracted from the output
dataset_info = {
    'breastmnist': {'train_size': 399, 'IR': 2.70, 'n_classes': 2, 'color': False},
    'organAmnist': {'train_size': 26273, 'IR': 4.44, 'n_classes': 11, 'color': False},
    'pneumoniamnist': {'train_size': 3348, 'IR': 2.86, 'n_classes': 2, 'color': False},
    'dermamnist-e': {'train_size': 6533, 'IR': 61.63, 'n_classes': 7, 'color': True},
    'octmnist': {'train_size': 69318, 'IR': 5.91, 'n_classes': 4, 'color': False},
    'pathmnist': {'train_size': 64000, 'IR': 1.62, 'n_classes': 9, 'color': True},
    'bloodmnist': {'train_size': 8749, 'IR': 2.68, 'n_classes': 8, 'color': True},
    'tissuemnist': {'train_size': 121027, 'IR': 9.07, 'n_classes': 8, 'color': False}
}

# Function to format dataset names with capital first letter and MNIST
def format_dataset_name(name):
    # Handle special cases
    if name.startswith('oct'):
        parts = name.split('mnist')
        return 'OCT' + 'MNIST' + parts[1].upper() if len(parts) == 2 else 'OCT'
    if name.startswith('organA'):
        parts = name.split('mnist')
        return 'OrganA' + 'MNIST' + parts[1].upper() if len(parts) == 2 else 'OrganA'
    
    parts = name.split('mnist')
    if len(parts) == 2:
        return parts[0].capitalize() + 'MNIST' + parts[1].upper()
    return name.capitalize()

# Prepare data for plotting
train_sizes = []
imbalance_ratios = []
num_classes = []
colors = []
labels = []

for dataset_name, stats in dataset_info.items():
    train_sizes.append(stats['train_size'])
    imbalance_ratios.append(stats['IR'])
    num_classes.append(stats['n_classes'])
    colors.append('#E6B0E6' if stats['color'] else '#C0C0C0')  # Light pink/purple for color, lighter grey for grayscale
    labels.append(format_dataset_name(dataset_name))

# Create the scatter plot with thinner figure
plt.figure(figsize=(7.5, 8))
scatter = plt.scatter(train_sizes, imbalance_ratios, 
                     s=[n**2 * 10 for n in num_classes],  # Size proportional to number of classes
                     c=colors, 
                     alpha=0.6, 
                     edgecolors='black', 
                     linewidth=1.5)

# Add dataset labels with custom positioning
labels_left = ['TissueMNIST', 'OCTMNIST', 'PathMNIST']  # Labels to display on the left
for i, label in enumerate(labels):
    # Determine horizontal alignment and offset
    if label in labels_left:
        ha = 'right'
        xytext = (-5, 0)
    elif label == 'PneumoniaMNIST':
        ha = 'left'
        xytext = (5, 10)  # Move up slightly
    else:
        ha = 'left'
        xytext = (5, 5)
    
    plt.annotate(label, (train_sizes[i], imbalance_ratios[i]), 
                xytext=xytext, textcoords='offset points', 
                fontsize=12, fontweight='bold', ha=ha)

plt.xscale('log')
plt.yscale('log')

# Set custom y-axis ticks
ax = plt.gca()
ax.yaxis.set_major_locator(FixedLocator([10, 50, 80]))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

plt.xlabel('Training Set Size (log scale)', fontsize=12, fontweight='bold')
plt.ylabel('Imbalance Ratio (IR = $p_{max}/p_{min}$, log scale)', fontsize=12, fontweight='bold')
plt.title('MedMNIST Datasets: Training Size vs Class Imbalance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')

# Create merged legend with image type and number of classes
legend_elements = [
    Patch(facecolor='#C0C0C0', edgecolor='black', label='Grayscale', alpha=0.6),
    Patch(facecolor='#E6B0E6', edgecolor='black', label='Color RGB', alpha=0.6),
]

# Add size legend for number of classes (show representative sizes)
class_sizes = [2, 4, 7, 11]  # Representative class counts from the datasets
for n in class_sizes:
    size = n**2 * 10
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='gray', markersize=np.sqrt(size)/2,
                                  markeredgecolor='black', markeredgewidth=1,
                                  label=f'{n} classes', linestyle='None'))

# Create single merged legend
plt.legend(handles=legend_elements, loc='upper left', 
          fontsize=10, title='Image Type & Number of Classes', framealpha=0.9, ncol=1)

plt.tight_layout()

# Save the figure
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'comprehensive_evaluation_results/figures/datasets_info.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

plt.show()
