import json
from pathlib import Path
import numpy as np

results_dir = Path('/mnt/data/psteinmetz/computer_vision_code/code/FailCatcher/Benchmarks/medMNIST/results/jsons_results/corruption_shifts')

if not results_dir.exists():
    print(f"Error: Directory {results_dir} does not exist.")
    exit(1)

tta_values = []
for json_file in sorted(results_dir.glob('uq_benchmark_*.json')):
    with open(json_file) as f:
        data = json.load(f)
    if data.get('model_backbone') != 'resnet18':
        continue
    methods = data.get('methods', {})
    if 'TTA' not in methods:
        continue
    tta = methods['TTA']
    augrc = tta.get('augrc_mean', tta.get('augrc', None))
    if augrc is not None:
        flag = data.get('flag', '')
        setup = data.get('setup', 'standard')
        print(f"{flag}_{setup}: TTA AUGRC = {augrc:.4f}")
        tta_values.append(augrc)

if tta_values:
    print(f"\nMean TTA AUGRC (ResNet18 CS): {np.mean(tta_values):.4f}")
    print(f"Values range: {min(tta_values):.4f} - {max(tta_values):.4f}")
else:
    print("No TTA AUGRC values found for ResNet18.")
