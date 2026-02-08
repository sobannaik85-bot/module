"""
test_models.py
==============
Unit tests for Module 3 Pixel Selector models.

Run with: pytest -v stegotool/modules/module3_pixel_selector/test_models.py
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import torch


class TestModels:
    """Test neural network models."""
    
    def test_unet_forward(self):
        """Test U-Net forward pass."""
        from stegotool.modules.module3_pixel_selector.models import PixelSelectorUNet
        
        model = PixelSelectorUNet()
        x = torch.randn(2, 1, 128, 128)
        y = model(x)
        
        assert y.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {y.shape}"
        assert y.min() >= 0 and y.max() <= 1, "Output should be in [0, 1]"
        print("✓ U-Net forward pass test passed")
    
    def test_resnet_forward(self):
        """Test ResNet forward pass."""
        from stegotool.modules.module3_pixel_selector.models import PixelSelectorResNet
        
        model = PixelSelectorResNet(patch_size=32)
        x = torch.randn(8, 1, 32, 32)
        y = model(x)
        
        assert y.shape == (8,), f"Expected (8,), got {y.shape}"
        assert y.min() >= 0 and y.max() <= 1, "Output should be in [0, 1]"
        print("✓ ResNet forward pass test passed")
    
    def test_lightweight_forward(self):
        """Test lightweight model forward pass."""
        from stegotool.modules.module3_pixel_selector.models import StegoSuitabilityNet
        
        model = StegoSuitabilityNet()
        x = torch.randn(2, 1, 128, 128)
        y = model(x)
        
        assert y.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {y.shape}"
        assert y.min() >= 0 and y.max() <= 1, "Output should be in [0, 1]"
        print("✓ Lightweight model forward pass test passed")
    
    def test_get_model_factory(self):
        """Test model factory function."""
        from stegotool.modules.module3_pixel_selector.models import get_model
        
        for name in ['unet', 'resnet', 'lightweight']:
            model = get_model(name)
            assert model is not None, f"Failed to create {name} model"
        
        print("✓ Model factory test passed")
    
    def test_srm_filters(self):
        """Test SRM filter layer."""
        from stegotool.modules.module3_pixel_selector.models import SRMFilterLayer
        
        srm = SRMFilterLayer()
        x = torch.randn(2, 1, 64, 64)
        y = srm(x)
        
        assert y.shape == (2, 4, 64, 64), f"Expected (2, 4, 64, 64), got {y.shape}"
        print("✓ SRM filter test passed")


class TestLossFunctions:
    """Test custom loss functions."""
    
    def test_combined_loss(self):
        """Test combined loss function."""
        from stegotool.modules.module3_pixel_selector.models import CombinedLoss
        
        loss_fn = CombinedLoss()
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        loss = loss_fn(pred, target)
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss >= 0, "Loss should be non-negative"
        print("✓ Combined loss test passed")
    
    def test_robustness_loss(self):
        """Test robustness loss function."""
        from stegotool.modules.module3_pixel_selector.models import RobustnessLoss
        
        loss_fn = RobustnessLoss()
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        
        loss = loss_fn(pred, target)
        assert loss.dim() == 0, "Loss should be scalar"
        print("✓ Robustness loss test passed")


class TestDataset:
    """Test dataset generation and loading."""
    
    def test_embedding_extraction(self):
        """Test payload embedding and extraction."""
        from stegotool.modules.module3_pixel_selector.dataset import (
            embed_payload, extract_payload
        )
        
        # Create test image
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        payload = b"Test message 123"
        coords = [(x, y) for y in range(10) for x in range(10)]
        
        # Embed
        stego = embed_payload(image, payload, coords)
        
        # Extract
        extracted = extract_payload(stego, len(payload), coords)
        
        assert extracted == payload, "Extracted payload doesn't match"
        print("✓ Embedding/extraction test passed")
    
    def test_jpeg_corruption(self):
        """Test JPEG corruption simulation."""
        from stegotool.modules.module3_pixel_selector.dataset import apply_jpeg_compression
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        for quality in [95, 85, 75]:
            corrupted = apply_jpeg_compression(image, quality)
            assert corrupted.shape == image.shape, "Shape should be preserved"
            assert corrupted.dtype == np.uint8, "Type should be preserved"
        
        print("✓ JPEG corruption test passed")
    
    def test_dataset_class(self):
        """Test PixelSuitabilityDataset class."""
        from stegotool.modules.module3_pixel_selector.dataset import PixelSuitabilityDataset
        import tempfile
        
        # Create temporary dataset
        patches = np.random.rand(100, 32, 32).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, patches=patches, labels=labels)
            
            # Load dataset
            dataset = PixelSuitabilityDataset(f.name)
            
            assert len(dataset) == 100
            x, y = dataset[0]
            assert x.shape == (1, 32, 32)
            assert y.shape == ()
        
        print("✓ Dataset class test passed")


class TestInference:
    """Test inference module."""
    
    def test_pixel_selector_baseline(self):
        """Test pixel selector with baseline fallback."""
        from stegotool.modules.module3_pixel_selector.inference import PixelSelector
        
        # Create test image
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        
        # Create selector (will use baseline if no model)
        selector = PixelSelector()
        
        # Select pixels
        coords = selector.select_pixels(image, num_pixels=100)
        
        assert len(coords) == 100, f"Expected 100 coords, got {len(coords)}"
        assert all(0 <= x < 200 and 0 <= y < 200 for x, y in coords)
        print("✓ Pixel selector baseline test passed")
    
    def test_suitability_map(self):
        """Test suitability map generation."""
        from stegotool.modules.module3_pixel_selector.inference import PixelSelector
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        selector = PixelSelector()
        
        smap = selector.get_suitability_map(image)
        
        assert smap.shape == (100, 100), f"Expected (100, 100), got {smap.shape}"
        assert smap.min() >= 0 and smap.max() <= 1, "Map should be normalized"
        print("✓ Suitability map test passed")
    
    def test_selection_strategies(self):
        """Test different selection strategies."""
        from stegotool.modules.module3_pixel_selector.inference import PixelSelector
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        selector = PixelSelector()
        
        for strategy in ['top_k', 'weighted', 'adaptive']:
            coords = selector.select_pixels(image, num_pixels=50, strategy=strategy)
            assert len(coords) == 50, f"Strategy {strategy} returned wrong count"
        
        print("✓ Selection strategies test passed")


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_baseline(self):
        """Test end-to-end pipeline with baseline."""
        from stegotool.modules.module3_pixel_selector.inference import PixelSelector
        from stegotool.modules.module3_pixel_selector.dataset import (
            embed_payload, extract_payload
        )
        
        # Create image
        image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        
        # Select pixels
        selector = PixelSelector()
        coords = selector.select_pixels(image, payload_bits=128)
        
        # Embed
        payload = b"Secret!"
        stego = embed_payload(image, payload, coords)
        
        # Extract
        extracted = extract_payload(stego, len(payload), coords)
        
        assert extracted == payload, "End-to-end extraction failed"
        print("✓ End-to-end baseline test passed")
    
    def test_backward_compatibility(self):
        """Test backward compatibility with old API."""
        from stegotool.modules.module3_pixel_selector import select_pixels
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        coords = select_pixels(image, payload_bits=100)
        
        assert len(coords) > 0, "Should return coordinates"
        print("✓ Backward compatibility test passed")


def run_all_tests():
    """Run all tests manually."""
    print("\n" + "="*60)
    print("MODULE 3 PIXEL SELECTOR - TEST SUITE")
    print("="*60 + "\n")
    
    test_classes = [
        TestModels(),
        TestLossFunctions(),
        TestDataset(),
        TestInference(),
        TestIntegration()
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n--- {class_name} ---")
        
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                try:
                    getattr(test_class, method_name)()
                    total_passed += 1
                except Exception as e:
                    print(f"✗ {method_name}: {e}")
                    total_failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("="*60)
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
