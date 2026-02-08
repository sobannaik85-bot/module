"""
inference.py
=============
Production Inference Module for Pixel Selection

This module provides:
1. Efficient model loading and caching
2. GPU/CPU inference with batching
3. Pixel ranking and selection APIs
4. Integration with steganography pipeline

Usage:
    from stegotool.modules.module3_pixel_selector.inference import PixelSelector
    
    selector = PixelSelector()
    coords = selector.select_pixels(image, num_pixels=1000)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F

# Local imports
try:
    from .models import (
        PixelSelectorUNet,
        PixelSelectorResNet,
        StegoSuitabilityNet,
        get_model
    )
    from .selector_baseline import select_pixels as baseline_select
    from .selector_utils import get_gray
except ImportError:
    from models import (
        PixelSelectorUNet,
        PixelSelectorResNet,
        StegoSuitabilityNet,
        get_model
    )
    from selector_baseline import select_pixels as baseline_select
    from selector_utils import get_gray


# =============================================================================
# Model Paths
# =============================================================================

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "module3_pixel_selector"
DEFAULT_MODEL_PATH = MODEL_DIR / "best.pth"


# =============================================================================
# Pixel Selector Class
# =============================================================================

class PixelSelector:
    """
    Production-ready pixel selector using trained ML models.
    
    This class provides:
    - Automatic model loading with fallback to baseline
    - Efficient GPU inference
    - Batch processing for large images
    - Multiple selection strategies
    
    Example:
        >>> selector = PixelSelector()
        >>> image = np.array(Image.open('image.png'))
        >>> coords = selector.select_pixels(image, num_pixels=1000)
        >>> print(f"Selected {len(coords)} pixels")
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_type: str = 'auto',
        device: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the pixel selector.
        
        Args:
            model_path: Path to trained model weights. If None, uses default.
            model_type: Model architecture ('unet', 'resnet', 'lightweight', 'auto')
            device: Device to run inference ('cuda', 'cpu', or None for auto)
            use_gpu: Whether to use GPU if available
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model_type = model_type
        self.use_baseline = False
        
        # Set device
        if device:
            self.device = torch.device(device)
        elif use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model or fall back to baseline."""
        if not self.model_path.exists():
            print(f"⚠️ Model not found at {self.model_path}")
            print("   Using baseline heuristic selector")
            self.use_baseline = True
            self.model = None
            return
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Determine model type from checkpoint or config
            if self.model_type == 'auto':
                if 'config' in checkpoint and 'model_type' in checkpoint['config']:
                    self.model_type = checkpoint['config']['model_type']
                else:
                    # Try to infer from state dict keys
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                    if any('enc1' in k for k in state_dict.keys()):
                        self.model_type = 'unet'
                    elif any('features' in k for k in state_dict.keys()):
                        self.model_type = 'resnet'
                    else:
                        self.model_type = 'lightweight'
            
            # Create model
            self.model = get_model(self.model_type)
            
            # Load weights
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            
            # Set to eval mode and move to device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Loaded {self.model_type} model from {self.model_path}")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            print("   Using baseline heuristic selector")
            self.use_baseline = True
            self.model = None
    
    @torch.no_grad()
    def get_suitability_map(
        self,
        image: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute pixel suitability map for an image.
        
        Args:
            image: RGB image as numpy array (H x W x 3) or grayscale (H x W)
            normalize: Whether to normalize output to [0, 1]
        
        Returns:
            Suitability map (H x W) with values in [0, 1]
        """
        if self.use_baseline:
            return self._baseline_suitability_map(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Prepare input tensor
        img_tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W
        img_tensor = img_tensor.to(self.device)
        
        # Handle different model types
        if self.model_type == 'resnet':
            # For patch-based model, use sliding window
            return self._patch_based_inference(gray, normalize)
        
        # For dense prediction models (unet, lightweight)
        # Pad to multiple of 8 for network compatibility
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Inference
        output = self.model(img_tensor)
        
        # Remove padding
        output = output[:, :, :h, :w]
        
        # Convert to numpy
        suitability = output.squeeze().cpu().numpy()
        
        if normalize:
            suitability = (suitability - suitability.min()) / (suitability.max() - suitability.min() + 1e-8)
        
        return suitability
    
    def _patch_based_inference(
        self,
        gray: np.ndarray,
        normalize: bool = True,
        patch_size: int = 32,
        stride: int = 8
    ) -> np.ndarray:
        """Inference for patch-based models using sliding window."""
        h, w = gray.shape
        suitability = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        
        half = patch_size // 2
        padded = np.pad(gray, half, mode='reflect')
        
        # Collect patches
        patches = []
        positions = []
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                py, px = y + half, x + half
                patch = padded[py - half:py + half, px - half:px + half]
                patches.append(patch)
                positions.append((y, x))
        
        # Batch inference
        batch_size = 256
        for i in range(0, len(patches), batch_size):
            batch_patches = np.stack(patches[i:i + batch_size])
            batch_patches = batch_patches.astype(np.float32) / 255.0
            batch_tensor = torch.from_numpy(batch_patches).unsqueeze(1)
            batch_tensor = batch_tensor.to(self.device)
            
            scores = self.model(batch_tensor).cpu().numpy()
            
            for j, (y, x) in enumerate(positions[i:i + batch_size]):
                # Spread score over patch area
                y_end = min(y + stride, h)
                x_end = min(x + stride, w)
                suitability[y:y_end, x:x_end] += scores[j]
                counts[y:y_end, x:x_end] += 1
        
        # Average overlapping regions
        suitability = suitability / (counts + 1e-8)
        
        if normalize:
            suitability = (suitability - suitability.min()) / (suitability.max() - suitability.min() + 1e-8)
        
        return suitability
    
    def _baseline_suitability_map(self, image: np.ndarray) -> np.ndarray:
        """Generate suitability map using baseline heuristics."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Compute texture features
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.abs(laplacian)
        
        # Local variance
        kernel_size = 5
        mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        sqr_mean = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
        variance = sqr_mean - mean ** 2
        variance = np.maximum(variance, 0)
        
        # Combine features
        suitability = 0.6 * lap_var / (lap_var.max() + 1e-8) + 0.4 * variance / (variance.max() + 1e-8)
        
        return suitability.astype(np.float32)
    
    def select_pixels(
        self,
        image: np.ndarray,
        num_pixels: Optional[int] = None,
        payload_bits: Optional[int] = None,
        lsb_bits: int = 1,
        threshold: Optional[float] = None,
        strategy: str = 'top_k',
        seed: int = 0
    ) -> List[Tuple[int, int]]:
        """
        Select optimal pixels for steganographic embedding.
        
        Args:
            image: RGB image as numpy array (H x W x 3)
            num_pixels: Number of pixels to select (overrides payload_bits)
            payload_bits: Number of bits to embed (used if num_pixels not given)
            lsb_bits: LSBs used per channel
            threshold: Minimum suitability threshold (0-1)
            strategy: Selection strategy ('top_k', 'weighted', 'adaptive')
            seed: Random seed for reproducibility
        
        Returns:
            List of (x, y) pixel coordinates in priority order
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB (H x W x 3)")
        
        h, w, _ = image.shape
        
        # Calculate pixels needed
        if num_pixels is None:
            if payload_bits is None:
                raise ValueError("Must specify num_pixels or payload_bits")
            capacity_per_pixel = 3 * lsb_bits
            num_pixels = int(np.ceil(payload_bits / capacity_per_pixel))
        
        # Get suitability map
        suitability = self.get_suitability_map(image)
        
        # Apply threshold if specified
        if threshold is not None:
            suitability[suitability < threshold] = 0
        
        # Select pixels based on strategy
        if strategy == 'top_k':
            coords = self._select_top_k(suitability, num_pixels)
        elif strategy == 'weighted':
            coords = self._select_weighted(suitability, num_pixels, seed)
        elif strategy == 'adaptive':
            coords = self._select_adaptive(suitability, num_pixels, image)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return coords
    
    def _select_top_k(
        self,
        suitability: np.ndarray,
        num_pixels: int
    ) -> List[Tuple[int, int]]:
        """Select top-k pixels by suitability score."""
        h, w = suitability.shape
        
        # Flatten and get top indices
        flat_indices = np.argsort(suitability.flatten())[::-1]
        
        coords = []
        for idx in flat_indices[:num_pixels]:
            y, x = divmod(idx, w)
            coords.append((int(x), int(y)))
        
        return coords
    
    def _select_weighted(
        self,
        suitability: np.ndarray,
        num_pixels: int,
        seed: int
    ) -> List[Tuple[int, int]]:
        """Select pixels with probability proportional to suitability."""
        np.random.seed(seed)
        h, w = suitability.shape
        
        # Normalize to probabilities
        probs = suitability.flatten()
        probs = probs / (probs.sum() + 1e-8)
        
        # Sample without replacement
        indices = np.random.choice(
            len(probs),
            size=min(num_pixels, len(probs)),
            replace=False,
            p=probs
        )
        
        coords = []
        for idx in indices:
            y, x = divmod(idx, w)
            coords.append((int(x), int(y)))
        
        return coords
    
    def _select_adaptive(
        self,
        suitability: np.ndarray,
        num_pixels: int,
        image: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Adaptive selection considering spatial distribution.
        
        Uses a grid-based approach to ensure pixels are spread across the image.
        """
        h, w = suitability.shape
        
        # Determine grid size
        grid_size = max(4, int(np.sqrt(num_pixels / 10)))
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        pixels_per_cell = num_pixels // (grid_size * grid_size) + 1
        
        coords = []
        
        for gy in range(grid_size):
            for gx in range(grid_size):
                # Get cell bounds
                y1, y2 = gy * cell_h, (gy + 1) * cell_h if gy < grid_size - 1 else h
                x1, x2 = gx * cell_w, (gx + 1) * cell_w if gx < grid_size - 1 else w
                
                # Get cell suitability
                cell = suitability[y1:y2, x1:x2]
                
                # Select top pixels from this cell
                flat = cell.flatten()
                top_indices = np.argsort(flat)[::-1][:pixels_per_cell]
                
                cell_w_actual = x2 - x1
                for idx in top_indices:
                    cy, cx = divmod(idx, cell_w_actual)
                    coords.append((int(x1 + cx), int(y1 + cy)))
        
        # Sort by global suitability and return top num_pixels
        coords_with_scores = [
            (suitability[y, x], x, y) for (x, y) in coords
        ]
        coords_with_scores.sort(reverse=True)
        
        return [(x, y) for (_, x, y) in coords_with_scores[:num_pixels]]
    
    def visualize_selection(
        self,
        image: np.ndarray,
        coords: List[Tuple[int, int]],
        output_path: Optional[Union[str, Path]] = None,
        show_map: bool = True
    ) -> np.ndarray:
        """
        Visualize selected pixels on the image.
        
        Args:
            image: Original RGB image
            coords: Selected pixel coordinates
            output_path: Path to save visualization
            show_map: Whether to overlay suitability heatmap
        
        Returns:
            Visualization image
        """
        vis = image.copy()
        h, w = image.shape[:2]
        
        if show_map:
            # Get suitability map
            suitability = self.get_suitability_map(image)
            
            # Create heatmap overlay
            heatmap = cv2.applyColorMap(
                (suitability * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Blend with original
            vis = cv2.addWeighted(vis, 0.7, heatmap, 0.3, 0)
        
        # Draw selected pixels
        for i, (x, y) in enumerate(coords[:min(len(coords), 500)]):
            # Color gradient from green (first) to red (last)
            ratio = i / max(len(coords) - 1, 1)
            color = (int(255 * ratio), int(255 * (1 - ratio)), 0)
            cv2.circle(vis, (x, y), 2, color, -1)
        
        if output_path:
            Image.fromarray(vis).save(output_path)
        
        return vis


# =============================================================================
# Convenience Functions
# =============================================================================

def select_pixels_ml(
    image: np.ndarray,
    payload_bits: int,
    lsb_bits: int = 1,
    model_path: Optional[str] = None,
    strategy: str = 'top_k',
    seed: int = 0
) -> List[Tuple[int, int]]:
    """
    Convenience function for pixel selection.
    
    This is a drop-in replacement for the baseline selector with ML support.
    
    Args:
        image: RGB image (H x W x 3)
        payload_bits: Number of bits to embed
        lsb_bits: LSBs per channel
        model_path: Optional model path
        strategy: Selection strategy
        seed: Random seed
    
    Returns:
        List of (x, y) coordinates
    """
    selector = PixelSelector(model_path=model_path)
    return selector.select_pixels(
        image,
        payload_bits=payload_bits,
        lsb_bits=lsb_bits,
        strategy=strategy,
        seed=seed
    )


# Legacy API compatibility
def model_select_pixels(
    image_np: np.ndarray,
    payload_bits: int,
    patch_size: int = 5,
    lsb_bits: int = 1,
    device: str = 'cpu'
) -> List[Tuple[int, int]]:
    """
    Legacy API for backward compatibility.
    
    This function maintains compatibility with the old selector_model_infer API.
    """
    selector = PixelSelector(device=device)
    return selector.select_pixels(
        image_np,
        payload_bits=payload_bits,
        lsb_bits=lsb_bits,
        strategy='top_k'
    )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pixel selection inference')
    parser.add_argument('image', type=str, help='Input image path')
    parser.add_argument('--num-pixels', type=int, default=1000,
                        help='Number of pixels to select')
    parser.add_argument('--output', type=str, default=None,
                        help='Output visualization path')
    parser.add_argument('--model', type=str, default=None,
                        help='Model weights path')
    parser.add_argument('--strategy', type=str, default='top_k',
                        choices=['top_k', 'weighted', 'adaptive'],
                        help='Selection strategy')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load image
    image = np.array(Image.open(args.image).convert('RGB'))
    print(f"Loaded image: {image.shape}")
    
    # Create selector
    selector = PixelSelector(model_path=args.model, device=args.device)
    
    # Select pixels
    coords = selector.select_pixels(
        image,
        num_pixels=args.num_pixels,
        strategy=args.strategy
    )
    print(f"Selected {len(coords)} pixels")
    
    # Visualize
    output_path = args.output or f"{Path(args.image).stem}_selection.png"
    vis = selector.visualize_selection(image, coords, output_path)
    print(f"Saved visualization to {output_path}")


if __name__ == '__main__':
    main()
