#!/home/psteinmetz/venvs/venv_medMNIST/bin/python3.12
"""
Generate radar plot for OrganaMNIST + AMOS (resnet18 only) for ISBI abstract.
Uses parse_results_directory and create_radar_plot_on_axis from generate_radar_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import from generate_radar_plots
sys.path.insert(0, str(Path(__file__).parent))
from benchmarks.medMNIST.utils.viz_benchmark_results.generate_radar_plots import parse_results_directory, create_radar_plot_on_axis, compute_mean_aggregation_metric


def main():
    """Main execution."""
    print("=" * 80)
    print("OrganaMNIST + AMOS Radar Plot Generator (resnet18)")
    print("=" * 80)
    
    # Paths
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent.parent
    results_dir = workspace_root / 'uq_benchmark_results'
    output_dir = script_dir / 'organamnist_amos_analysis'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data from all shift types using parse_results_directory
    print("\nLoading data from ID results...")
    id_results = parse_results_directory(results_dir / 'id_results' / 'log_results', metric='auroc_f')
    
    print("\nLoading data from corruption shifts...")
    cs_results = parse_results_directory(results_dir / 'corruption_shifts', metric='auroc_f')
    
    print("\nLoading data from population shifts...")
    ps_results = parse_results_directory(results_dir / 'population_shifts', metric='auroc_f')
    
    print("\nLoading data from new class shifts...")
    ncs_results = parse_results_directory(results_dir / 'new_class_shifts', metric='auroc_f')
    
    # Merge all results for resnet18, filtering for organamnist and amos2022
    # Add numbered prefixes to control sort order: 1_ID (top), 2_CS (right), 3_PS (bottom), 4_NCS (left)
    model_results = {}
    
    # Add ID results (organamnist only) - prefix with "1_ID_"
    if 'resnet18' in id_results:
        for dataset_key, methods in id_results['resnet18'].items():
            if 'organamnist' in dataset_key:
                new_key = f"1_ID_{dataset_key}"
                model_results[new_key] = methods
                print(f" Added from ID: {new_key} ({len(methods)} methods)")
    
    # Add corruption shift results (organamnist only) - prefix with "2_CS_"
    if 'resnet18' in cs_results:
        for dataset_key, methods in cs_results['resnet18'].items():
            if 'organamnist' in dataset_key:
                new_key = f"2_CS_{dataset_key}"
                model_results[new_key] = methods
                print(f" Added from CS: {new_key} ({len(methods)} methods)")
    
    # Add population shift results (amos2022 only) - prefix with "3_PS_"
    if 'resnet18' in ps_results:
        for dataset_key, methods in ps_results['resnet18'].items():
            if 'amos' in dataset_key.lower():
                new_key = f"3_PS_{dataset_key}"
                model_results[new_key] = methods
                print(f" Added from PS: {new_key} ({len(methods)} methods)")
    
    # Add new class shift results (amos2022 only) - prefix with "4_NCS_"
    if 'resnet18' in ncs_results:
        for dataset_key, methods in ncs_results['resnet18'].items():
            if 'amos' in dataset_key.lower():
                new_key = f"4_NCS_{dataset_key}"
                model_results[new_key] = methods
                print(f" Added from NCS: {new_key} ({len(methods)} methods)")
    
    print(f"\nTotal configurations loaded: {len(model_results)}")
    print(f"Dataset keys: {sorted(model_results.keys())}")
    
    if not model_results:
        print("ERROR: No data loaded!")
        return
    
    # Manually compute Mean_Aggregation for each dataset using original keys
    print("\nComputing Mean_Aggregation markers...")
    shift_map = {
        '1_ID_': 'in_distribution',
        '2_CS_': 'corruption_shifts',
        '3_PS_': 'population_shift',
        '4_NCS_': 'new_class_shift'
    }
    
    agg_success = 0
    agg_ens_success = 0
    
    for prefixed_key in sorted(model_results.keys()):
        # Extract original key and shift type
        original_key = prefixed_key
        shift_type = 'in_distribution'
        for prefix, shift in shift_map.items():
            if prefixed_key.startswith(prefix):
                original_key = prefixed_key.replace(prefix, '')
                shift_type = shift
                # CRITICAL: For new_class_shift, the compute function expects 'new_class_' prefix
                if shift == 'new_class_shift':
                    original_key = f"new_class_{original_key}"
                break
        
        try:
            # Per-fold Mean Aggregation
            mean_agg = compute_mean_aggregation_metric(
                results_dir, original_key, 'resnet18',
                metric='auroc_f', aggregation='mean',
                shift=shift_type, use_ensemble=False
            )
            if not np.isnan(mean_agg):
                model_results[prefixed_key]['Mean_Aggregation'] = mean_agg
                agg_success += 1
                print(f" {prefixed_key}: Mean_Agg = {mean_agg:.3f}")
            else:
                print(f" {prefixed_key}: Mean_Agg = NaN (shift={shift_type})")
            
            # Ensemble Mean Aggregation
            mean_agg_ens = compute_mean_aggregation_metric(
                results_dir, original_key, 'resnet18',
                metric='auroc_f', aggregation='mean',
                shift=shift_type, use_ensemble=True
            )
            if not np.isnan(mean_agg_ens):
                model_results[prefixed_key]['Mean_Aggregation_Ensemble'] = mean_agg_ens
                agg_ens_success += 1
        except Exception as e:
            print(f" {prefixed_key}: Error - {e}")
    
    print(f"\n  Total: {agg_success}/16 Mean_Aggregation, {agg_ens_success}/16 Mean_Aggregation_Ensemble")
    
    # Create radar plot using the standard function
    print("\nGenerating radar plot...")
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='polar')
    
    # Use create_radar_plot_on_axis with the prefixed data to maintain order
    # The prefixes ensure proper ordering
    # IMPORTANT: Set results_dir=None to prevent re-computing Mean_Aggregation
    # (we already computed it manually above with correct shift types)
    handles, labels = create_radar_plot_on_axis(
        ax, model_results, 'resnet18',
        results_dir=None, runs_dir=None,
        metric='auroc_f', aggregation='mean', shift='in_distribution'
    )
    
    # Clean up tick labels - remove the prefixes
    tick_labels = ax.get_xticklabels()
    for label in tick_labels:
        current_text = label.get_text()
        # The tick labels show setup names (S, DA, DO, DADO) which are correct
        # But we need to ensure dataset family labels are clean
        # This is already handled by create_radar_plot_on_axis
        pass
    
    # Remove the default title
    ax.set_title('')
    
    # Hide outer ring family name labels completely (organa, amos2022)
    # We already have ID/CS/PS/NCS labels, so dataset family names are redundant
    for text_obj in ax.texts:
        current_text = text_obj.get_text()
        # Check if this is a family name label (contains our prefixes or just dataset names)
        for prefix in ['1_ID_', '2_CS_', '3_PS_', '4_NCS_']:
            if prefix in current_text:
                # Hide the label by making it invisible
                text_obj.set_visible(False)
                print(f" Hidden label: '{current_text}'")
                break
        # Also hide if it's just the cleaned dataset name
        if current_text.lower() in ['organa', 'organamnist', 'amos2022', 'amos']:
            text_obj.set_visible(False)
            print(f" Hidden label: '{current_text}'")
    
    # Add family labels (ID, CS, PS, NCS) to show which shift type each group belongs to
    # Calculate the center angle for each family based on the dataset grouping
    dataset_keys_sorted = sorted(model_results.keys())
    
    # Group datasets by family prefix
    families = {
        '1_ID_': [],
        '2_CS_': [],
        '3_PS_': [],
        '4_NCS_': []
    }
    
    for i, key in enumerate(dataset_keys_sorted):
        for prefix in families.keys():
            if key.startswith(prefix):
                families[prefix].append(i)
                break
    
    # Get the tick positions (angles)
    tick_angles = ax.get_xticks()
    
    # Add label at the center of each family group
    shift_labels = {'1_ID_': 'ID', '2_CS_': 'CS', '3_PS_': 'PS', '4_NCS_': 'NCS'}
    for prefix, indices in families.items():
        if indices:
            # Calculate center angle for this family
            center_idx = (indices[0] + indices[-1]) / 2
            center_angle = tick_angles[int(center_idx)] + 0.12 if int(center_idx) < len(tick_angles) else 0
            
            # Add text label at family center (closer to radar, larger font)
            ax.text(center_angle, 1.10, shift_labels[prefix],
                   ha='center', va='center',
                   fontsize=20, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
    
    # Rename legend labels
    handles, labels = ax.get_legend_handles_labels()

    if handles and labels:
        labels = [
            label.replace('KNN_Raw', 'KNN latent distance to train')
                .replace('MSR', 'Max Softmax Response (MSR)')
                .replace('Max Softmax Response (MSR)-S', 'MSR-Scaled')
                .replace('MLS', 'Max Logit Score')
                .replace('Ensembling', 'Deep Ensembles')
                .replace('MCDropout', 'Monte Carlo Dropout')
                .replace('TTA', 'Test Time Augmentation (TTA)')
                .replace('GPS', 'Optimized TTA')
                .replace('Mean_Aggregation', 'Mean Agg')
            for label in labels
        ]

        order = [0, 2, 3, 4, 5, 6, 9, 1, 7, 8]

        ax.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc='upper left',
            bbox_to_anchor=(0.72, 0.25),
            fontsize=16,
            frameon=True,
            ncol=1
        )
    
    # Save
    output_path = output_dir / 'organamnist_amos_radar_resnet18.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Saved: {output_path}")
    plt.close()
    
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
