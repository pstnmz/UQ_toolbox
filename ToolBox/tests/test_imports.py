"""Test that all imports work correctly."""

def test_backward_compatible_imports():
    """Test old import style still works."""
    import UQ_Toolbox.UQ_toolbox as uq
    
    # Functions
    assert hasattr(uq, 'TTA')
    assert hasattr(uq, 'perform_greedy_policy_search')
    assert hasattr(uq, 'ensembling_stds_computation')
    
    # Classes
    assert hasattr(uq, 'TTAMethod')
    assert hasattr(uq, 'GPSMethod')
    assert hasattr(uq, 'EnsembleSTDMethod')
    
    print("[OK] Backward compatible imports working!")

def test_new_modular_imports():
    """Test new import style works."""
    from UQ_Toolbox.methods import TTAMethod, EnsembleSTDMethod
    from UQ_Toolbox.search import perform_greedy_policy_search
    from UQ_Toolbox.visualization import compare_uq_methods
    
    assert TTAMethod is not None
    assert perform_greedy_policy_search is not None
    
    print("[OK] Modular imports working!")

def test_package_level_imports():
    """Test importing from package level."""
    import UQ_Toolbox as uq
    
    # Should have access via __init__.py
    assert hasattr(uq, 'UQMethod')
    assert hasattr(uq, 'UQResult')
    
    print("[OK] Package-level imports working!")

if __name__ == "__main__":
    test_backward_compatible_imports()
    test_new_modular_imports()
    test_package_level_imports()
    print("\nAll import tests passed!")