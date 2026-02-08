# Module 3: Pixel Selector

## Overview

This module provides **supervised ML-based pixel selection** for steganographic embedding. It uses a neural network trained on real robustness labels to identify which pixels are most suitable for hiding data, ensuring messages survive compression and other corruptions.

**Learning Type**: Supervised Learning  
**Labels**: Binary (robust=1, weak=0) generated from actual embed/compress/extract tests

## Architecture

The module uses a **StegoSuitabilityNet** - a lightweight neural network that predicts pixel-level robustness scores:

```
Input Image (Grayscale)
        │
        ▼
┌─────────────────────┐
│  Initial Conv Layers │  (1→16→32 channels)
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│     Multi-Scale Dilated Branches     │
│  ┌─────┐  ┌─────┐  ┌─────┐          │
│  │ d=1 │  │ d=2 │  │ d=4 │          │
│  └─────┘  └─────┘  └─────┘          │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Feature Fusion    │  (Concatenate + Conv)
│   + Residual Blocks │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Attention Modules  │
│  - Spatial Attention│
│  - Channel Attention│
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Output Sigmoid    │  → Suitability Map [0,1]
└─────────────────────┘
```

## Features

- **3 Model Architectures**: U-Net, ResNet, Lightweight
- **SRM Filters**: Steganalysis-aware preprocessing
- **Attention Mechanisms**: Focus on textured regions
- **Multiple Selection Strategies**: top_k, weighted, adaptive
- **Robustness-Based Training**: Labels from real embed/compress/extract tests

## Installation

Dependencies are in the main `requirements.txt`:
```
torch
numpy
opencv-python
pillow
```

## Quick Start

### 1. Basic Usage

```python
from stegotool.modules.module3_pixel_selector import PixelSelector
import numpy as np
from PIL import Image

# Load image
image = np.array(Image.open('image.png').convert('RGB'))

# Create selector (auto-loads trained model)
selector = PixelSelector()

# Select pixels for embedding
coords = selector.select_pixels(image, num_pixels=1000)
# Returns: [(x1, y1), (x2, y2), ...] sorted by robustness

# Or specify payload size
coords = selector.select_pixels(image, payload_bits=8000, lsb_bits=1)
```

### 2. Get Suitability Map

```python
# Get full suitability map (for visualization)
suitability_map = selector.get_suitability_map(image)
# Returns: HxW numpy array with values in [0, 1]
```

### 3. Different Selection Strategies

```python
# Top-k: Select highest scoring pixels
coords = selector.select_pixels(image, num_pixels=500, strategy='top_k')

# Weighted: Probabilistic selection based on scores
coords = selector.select_pixels(image, num_pixels=500, strategy='weighted')

# Adaptive: Spatially distributed selection
coords = selector.select_pixels(image, num_pixels=500, strategy='adaptive')
```

### 4. Visualization

```python
# Visualize selected pixels on image
vis = selector.visualize_selection(image, coords, output_path='selection.png')
```

## Training Your Own Model

### Step 1: Generate Training Data

```bash
# Requires images in stegotool/data/dev_images/
python -m stegotool.modules.module3_pixel_selector.quick_start generate
```

Options:
- `--images PATH`: Input image directory
- `--output PATH`: Output .npz file
- `--samples N`: Samples per image (default: 200)
- `--max-images N`: Max images to process

### Step 2: Train Model

```bash
python -m stegotool.modules.module3_pixel_selector.quick_start train
```

Options:
- `--model {unet,resnet,lightweight}`: Architecture (default: lightweight)
- `--epochs N`: Training epochs (default: 50)
- `--batch-size N`: Batch size (default: 32)
- `--lr FLOAT`: Learning rate (default: 0.001)

### Step 3: Test Model

```bash
python -m stegotool.modules.module3_pixel_selector.quick_start test --visualize
```

## API Reference

### PixelSelector

```python
class PixelSelector:
    def __init__(
        self,
        model_path: str = None,      # Path to .pth file
        model_type: str = 'auto',    # 'unet', 'resnet', 'lightweight', 'auto'
        device: str = None,          # 'cuda', 'cpu', or None for auto
        use_gpu: bool = True
    )
    
    def select_pixels(
        self,
        image: np.ndarray,           # RGB image (H, W, 3)
        num_pixels: int = None,      # Number of pixels to select
        payload_bits: int = None,    # Alternative: bits to embed
        lsb_bits: int = 1,           # LSBs per channel
        strategy: str = 'top_k',     # Selection strategy
        seed: int = 0                # Random seed
    ) -> List[Tuple[int, int]]       # Returns (x, y) coordinates
    
    def get_suitability_map(
        self,
        image: np.ndarray            # RGB or grayscale image
    ) -> np.ndarray                  # Suitability scores (H, W)
```

### Available Models

| Model | Description | Speed | Accuracy |
|-------|-------------|-------|----------|
| `lightweight` | Multi-scale dilated convolutions | Fast | Good |
| `unet` | Encoder-decoder with skip connections | Medium | Best |
| `resnet` | Patch-based classifier | Slow | Good |

## File Structure

```
module3_pixel_selector/
├── models.py           # Neural network architectures
├── dataset.py          # Training data generation
├── train.py            # Training pipeline
├── inference.py        # PixelSelector class
├── quick_start.py      # CLI interface
├── test_models.py      # Unit tests
├── selector_baseline.py # Heuristic fallback
├── selector_utils.py   # Utility functions
├── __init__.py         # Module exports
└── README.md           # This file
```

## How It Works

### Training Data Generation

1. Load images from directory
2. For each pixel location:
   - Embed test payload in surrounding pixels
   - Apply JPEG compression (multiple quality levels)
   - Extract payload
   - Label as **robust** (1) if payload recovers, **weak** (0) otherwise
3. Extract 32x32 patches centered on each pixel
4. Save patches + labels as .npz dataset

### Model Training

1. Load patch dataset
2. Train neural network to predict robustness from patches
3. Use combined loss: BCE + Dice + Smoothness regularization
4. Early stopping based on validation loss
5. Save best model weights

### Inference

1. Load trained model
2. Run image through network → suitability map
3. Select top-scoring pixels based on strategy
4. Return coordinates in priority order

## Integration Example

Using with Module 6 (Redundancy) and embedding:

```python
from stegotool.modules.module3_pixel_selector import PixelSelector
from stegotool.modules.module6_redundancy.rs_wrapper import add_redundancy

# Prepare payload with ECC
message = b"Secret message"
encoded = add_redundancy(message, nsym=32)

# Select robust pixels
selector = PixelSelector()
coords = selector.select_pixels(image, payload_bits=len(encoded) * 8)

# Embed using selected coordinates
# (pass coords to your embedding function)
```

## Troubleshooting

### "Model not found" warning
The module falls back to baseline heuristics. Train a model using quick_start.

### CUDA out of memory
Use `--batch-size 16` or `device='cpu'` in PixelSelector.

### Low accuracy after training
- Use more training images (at least 20-50)
- Increase samples per image
- Try different model architecture

## Authors

Module 3 - Pixel Selector  
AI-Integrated Steganography Toolbox

## License

Part of the StegoTool project.
