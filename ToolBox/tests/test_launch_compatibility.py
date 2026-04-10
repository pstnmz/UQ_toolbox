"""
Verify launch_uq_methods.py will work with new structure.
"""

def test_launch_imports():
    """Test all imports used in launch_uq_methods.py."""
    print("Checking launch_uq_methods.py compatibility...")
    
    try:
        # Old style (what launch_uq_methods.py uses)
        import UQ_Toolbox.UQ_toolbox as uq
        
        # Check all required functions exist
        required = [
            'TTA',
            'apply_randaugment_and_store_results',
            'perform_greedy_policy_search',
            'extract_gps_augmentations_info',
            'ensembling_stds_computation',
            'distance_to_hard_labels_computation',
            'roc_curve_UQ_method_computation',
            'UQ_method_plot',
            'build_monai_cache_dataset',
        ]
        
        missing = []
        for func in required:
            if not hasattr(uq, func):
                missing.append(func)
        
        if missing:
            print(f"[ERROR] Missing functions: {missing}")
            return False
        
        print("[OK] All launch_uq_methods.py imports available!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Import error: {e}")
        return False

if __name__ == "__main__":
    if test_launch_imports():
        print("\nlaunch_uq_methods.py should work without changes!")
    else:
        print("\n[WARNING] Need to fix imports before running launch_uq_methods.py")