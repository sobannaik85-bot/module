"""
Module 3: Pixel Selector
========================

This module provides ML-based pixel selection for steganography.

Main Components:
- PixelSelector: Production inference class for selecting robust pixels
- PixelSelectorUNet: U-Net model for dense suitability prediction
- PixelSelectorResNet: ResNet model for patch classification
- StegoSuitabilityNet: Lightweight model for fast inference

Usage:
    from stegotool.modules.module3_pixel_selector import PixelSelector
    
    selector = PixelSelector()
    coords = selector.select_pixels(image, num_pixels=1000)

For training:
    from stegotool.modules.module3_pixel_selector import train
    # See train.py --help for options

Legacy API (backward compatible):
    from stegotool.modules.module3_pixel_selector import select_pixels
"""

# Baseline heuristic selector (always available)
from .selector_baseline import select_pixels

# Legacy model (for backward compatibility)
from .selector_model import TinyPatchNet, infer_scores

# New ML Models
from .models import (
    PixelSelectorUNet,
    PixelSelectorResNet,
    StegoSuitabilityNet,
    get_model,
    CombinedLoss,
    RobustnessLoss
)

# Production Inference
from .inference import (
    PixelSelector,
    select_pixels_ml,
    model_select_pixels
)

# Dataset and Training
from .dataset import (
    PixelSuitabilityDataset,
    PixelMapDataset,
    DatasetConfig,
    generate_patch_dataset,
    generate_map_dataset
)

# Utilities
from .selector_utils import (
    get_gray,
    extract_patch,
    compute_entropy,
    patch_variance
)

__all__ = [
    # Main API
    "PixelSelector",
    "select_pixels",
    "select_pixels_ml",
    
    # Models
    "PixelSelectorUNet",
    "PixelSelectorResNet", 
    "StegoSuitabilityNet",
    "get_model",
    
    # Loss functions
    "CombinedLoss",
    "RobustnessLoss",
    
    # Dataset
    "PixelSuitabilityDataset",
    "PixelMapDataset",
    "DatasetConfig",
    "generate_patch_dataset",
    "generate_map_dataset",
    
    # Legacy
    "TinyPatchNet",
    "infer_scores",
    "model_select_pixels",
    
    # Utilities
    "get_gray",
    "extract_patch",
    "compute_entropy",
    "patch_variance"
]
