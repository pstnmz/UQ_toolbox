"""
Basic functionality tests with dummy data.
Run this before using real datasets.
"""
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def test_tta_basic():
    """Test TTA with dummy data."""
    print("Testing TTA...")
    from UQ_Toolbox.methods.tta import TTAMethod
    
    # Dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, 2)  # Binary classification
    
    model = DummyModel()
    model.eval()
    
    dataset = TensorDataset(torch.randn(10, 3, 224, 224))
    device = torch.device('cpu')
    
    tta = TTAMethod(transformations=None, n=2, m=45)
    stds = tta.compute(
        model, dataset, device, 
        nb_augmentations=3, 
        batch_size=2,
        image_size=224,
        nb_channels=3
    )
    
    # stds is a list, not a numpy array
    assert len(stds) == 10, f"Expected 10 samples, got {len(stds)}"
    assert all(isinstance(s, (float, int)) for s in stds), "Stds should be numeric"
    print(f" Computed stds for 10 samples: mean={np.mean(stds):.4f}")
    print("[OK] TTA works!")

def test_ensemble_basic():
    """Test ensemble with dummy data."""
    print("\nTesting Ensemble...")
    from UQ_Toolbox.methods.ensemble import EnsembleSTDMethod
    
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 2)
    
    models = [DummyModel() for _ in range(3)]
    dataset = TensorDataset(torch.randn(10, 3, 224, 224), torch.randint(0, 2, (10,)))
    loader = DataLoader(dataset, batch_size=2)
    device = torch.device('cpu')
    
    ensemble = EnsembleSTDMethod()
    stds = ensemble.compute(models, loader, device)
    
    assert stds.shape == (10,), f"Expected shape (10,), got {stds.shape}"
    print("[OK] Ensemble works!")

def test_distance_basic():
    """Test distance method with dummy data."""
    print("\nTesting Distance method...")
    from UQ_Toolbox.methods.distance import DistanceToHardLabelsMethod
    
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 3)  # 3 classes
    
    model = DummyModel()
    dataset = TensorDataset(torch.randn(10, 3, 224, 224), torch.randint(0, 3, (10,)))
    loader = DataLoader(dataset, batch_size=2)
    device = torch.device('cpu')
    
    distance_method = DistanceToHardLabelsMethod()
    dists = distance_method.compute([model], loader, device)
    
    assert dists.shape == (10,), f"Expected shape (10,), got {dists.shape}"
    print("[OK] Distance method works!")

def test_visualization_basic():
    """Test visualization functions don't crash."""
    print("\nTesting Visualization...")
    from UQ_Toolbox.visualization.plots import roc_curve_UQ_method_computation, UQ_method_plot
    
    correct = np.random.rand(50)
    incorrect = np.random.rand(30)
    
    fpr, tpr, auc = roc_curve_UQ_method_computation(correct, incorrect)
    assert 0 <= auc <= 1, f"AUC should be in [0,1], got {auc}"
    print(f" AUC computed: {auc:.3f}")
    
    # Test plotting (don't show, just ensure it runs)
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    UQ_method_plot(correct, incorrect, "Test Metric", "Test Plot", swarmplot=False)
    
    print("[OK] Visualization works!")

if __name__ == "__main__":
    test_tta_basic()
    test_ensemble_basic()
    test_distance_basic()
    test_visualization_basic()
    print("\nAll basic tests passed! Ready for real data.")