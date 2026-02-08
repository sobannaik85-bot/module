"""
dataset.py
===========
Training Dataset Generation for Pixel Selection Model

This module creates proper training data by:
1. Embedding test payloads at specific pixel locations
2. Applying realistic corruptions (JPEG compression, noise)
3. Attempting to extract and verify the payload
4. Labeling pixels as robust (1) or weak (0) based on recovery success

The dataset provides ground truth for supervised learning of pixel suitability.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Optional, Generator
import io
import random
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    patch_size: int = 32
    payload_size: int = 32  # bytes
    ecc_symbols: int = 16
    lsb_bits: int = 1
    jpeg_qualities: List[int] = None  # Will be set in __post_init__
    noise_levels: List[float] = None
    num_samples_per_image: int = 500
    positive_ratio: float = 0.5  # Target ratio of positive samples
    seed: int = 42
    
    def __post_init__(self):
        if self.jpeg_qualities is None:
            self.jpeg_qualities = [95, 90, 85, 80, 75, 70]
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0]


# =============================================================================
# LSB Embedding/Extraction (Ground Truth Generator)
# =============================================================================

def _bytes_to_bits(data: bytes) -> List[int]:
    """Convert bytes to list of bits."""
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits


def _bits_to_bytes(bits: List[int]) -> bytes:
    """Convert list of bits to bytes."""
    while len(bits) % 8 != 0:
        bits.append(0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (bits[i + j] & 1)
        out.append(byte)
    return bytes(out)


def embed_payload(image: np.ndarray, payload: bytes, 
                  coords: List[Tuple[int, int]], lsb_bits: int = 1) -> np.ndarray:
    """
    Embed payload into image at specified coordinates.
    
    Args:
        image: HxWx3 RGB uint8 array
        payload: bytes to embed
        coords: list of (x, y) coordinates
        lsb_bits: number of LSBs to use per channel
    
    Returns:
        Modified image array
    """
    arr = image.copy().astype(np.int32)  # Use int32 to avoid overflow issues
    h, w, c = arr.shape
    
    capacity_bits = len(coords) * c * lsb_bits
    payload_bits = _bytes_to_bits(payload)
    
    if len(payload_bits) > capacity_bits:
        raise ValueError(f"Payload too large: {len(payload_bits)} bits > {capacity_bits} capacity")
    
    # Pad payload
    payload_bits.extend([0] * (capacity_bits - len(payload_bits)))
    
    bit_idx = 0
    for (x, y) in coords:
        for ch in range(c):
            for b in range(lsb_bits):
                if bit_idx >= len(payload_bits):
                    break
                bit = payload_bits[bit_idx]
                mask = 1 << b
                arr[y, x, ch] = (arr[y, x, ch] & (255 - mask)) | (bit << b)
                bit_idx += 1
    
    return arr.astype(np.uint8)


def extract_payload(image: np.ndarray, num_bytes: int,
                    coords: List[Tuple[int, int]], lsb_bits: int = 1) -> bytes:
    """
    Extract payload from image at specified coordinates.
    
    Args:
        image: HxWx3 RGB uint8 array
        num_bytes: number of bytes to extract
        coords: list of (x, y) coordinates
        lsb_bits: number of LSBs used per channel
    
    Returns:
        Extracted bytes
    """
    h, w, c = image.shape
    needed_bits = num_bytes * 8
    capacity_bits = len(coords) * c * lsb_bits
    
    if needed_bits > capacity_bits:
        raise ValueError("Not enough coordinates for requested bytes")
    
    bits = []
    for (x, y) in coords:
        for ch in range(c):
            for b in range(lsb_bits):
                if len(bits) >= needed_bits:
                    break
                val = (image[y, x, ch] >> b) & 1
                bits.append(val)
            if len(bits) >= needed_bits:
                break
        if len(bits) >= needed_bits:
            break
    
    return _bits_to_bytes(bits)


# =============================================================================
# Error Correction (Reed-Solomon)
# =============================================================================

def add_ecc(data: bytes, nsym: int = 16) -> bytes:
    """Add Reed-Solomon error correction."""
    try:
        from reedsolo import RSCodec
        rs = RSCodec(nsym)
        return bytes(rs.encode(data))
    except ImportError:
        # Fallback: simple repetition code
        return data + data + data


def recover_ecc(data: bytes, nsym: int = 16) -> Optional[bytes]:
    """Attempt to recover data using Reed-Solomon."""
    try:
        from reedsolo import RSCodec, ReedSolomonError
        rs = RSCodec(nsym)
        try:
            decoded = rs.decode(data)
            return bytes(decoded)
        except ReedSolomonError:
            return None
    except ImportError:
        # Fallback: majority voting for repetition code
        third = len(data) // 3
        d1, d2, d3 = data[:third], data[third:2*third], data[2*third:3*third]
        # Return most common
        from collections import Counter
        votes = Counter([d1, d2, d3])
        return votes.most_common(1)[0][0] if votes else None


# =============================================================================
# Corruption Simulation
# =============================================================================

def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression and return decompressed image."""
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer).convert('RGB'))


def apply_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to image."""
    if sigma == 0:
        return image
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def apply_corruption(image: np.ndarray, corruption_type: str, 
                     param: float) -> np.ndarray:
    """Apply a corruption to the image."""
    if corruption_type == 'jpeg':
        return apply_jpeg_compression(image, int(param))
    elif corruption_type == 'noise':
        return apply_gaussian_noise(image, param)
    elif corruption_type == 'none':
        return image.copy()
    else:
        return image.copy()


# =============================================================================
# Patch Extraction and Feature Computation
# =============================================================================

def extract_patch(image: np.ndarray, x: int, y: int, 
                  patch_size: int = 32) -> np.ndarray:
    """Extract a patch centered at (x, y)."""
    h, w = image.shape[:2]
    half = patch_size // 2
    
    # Handle boundaries with reflection padding
    if len(image.shape) == 2:
        padded = np.pad(image, ((half, half), (half, half)), mode='reflect')
    else:
        padded = np.pad(image, ((half, half), (half, half), (0, 0)), mode='reflect')
    
    y_pad, x_pad = y + half, x + half
    return padded[y_pad - half:y_pad + half, x_pad - half:x_pad + half]


def compute_texture_score(patch: np.ndarray) -> float:
    """Compute texture complexity score for a patch."""
    import cv2
    
    if len(patch.shape) == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = patch
    
    # Laplacian variance (edge density)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = laplacian.var()
    
    # Local variance
    local_var = gray.astype(np.float32).var()
    
    # Entropy
    hist = np.histogram(gray, bins=256, range=(0, 256))[0]
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    # Combined score
    return 0.4 * (lap_var / 1000) + 0.3 * (local_var / 1000) + 0.3 * (entropy / 8)


def rank_pixels_by_texture(image: np.ndarray, patch_size: int = 5) -> List[Tuple[int, int]]:
    """Rank all pixels by texture complexity."""
    import cv2
    
    h, w = image.shape[:2]
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    scores = []
    half = patch_size // 2
    padded = np.pad(gray, half, mode='reflect')
    
    for y in range(h):
        for x in range(w):
            patch = padded[y:y + patch_size, y:y + patch_size]
            
            # Quick scoring
            lap = cv2.Laplacian(patch, cv2.CV_64F).var()
            var = patch.astype(np.float32).var()
            score = lap + var
            
            scores.append((score, x, y))
    
    scores.sort(reverse=True, key=lambda t: t[0])
    return [(x, y) for (_, x, y) in scores]


# =============================================================================
# Robustness Testing
# =============================================================================

def test_pixel_robustness(image: np.ndarray, coords: List[Tuple[int, int]],
                          config: DatasetConfig) -> bool:
    """
    Test if a set of pixels can robustly store a payload.
    
    Returns True if payload survives corruption and can be recovered.
    """
    # Generate deterministic payload based on coordinates
    coord_hash = hashlib.md5(str(coords[:10]).encode()).digest()
    payload = coord_hash[:config.payload_size]
    
    # Add ECC
    encoded = add_ecc(payload, config.ecc_symbols)
    
    # Check capacity
    capacity = len(coords) * 3 * config.lsb_bits
    if len(encoded) * 8 > capacity:
        return False
    
    try:
        # Embed
        stego = embed_payload(image, encoded, coords, config.lsb_bits)
        
        # Test multiple corruptions
        success_count = 0
        total_tests = 0
        
        for quality in config.jpeg_qualities:
            corrupted = apply_jpeg_compression(stego, quality)
            extracted = extract_payload(corrupted, len(encoded), coords, config.lsb_bits)
            recovered = recover_ecc(extracted, config.ecc_symbols)
            
            if recovered == payload:
                success_count += 1
            total_tests += 1
        
        # Consider robust if survives majority of tests
        return success_count >= (total_tests // 2)
    
    except Exception:
        return False


def test_single_pixel_robustness(image: np.ndarray, x: int, y: int,
                                  config: DatasetConfig,
                                  neighborhood_size: int = 8) -> float:
    """
    Test robustness of a single pixel's neighborhood.
    
    Returns a score between 0 and 1 indicating robustness.
    """
    h, w = image.shape[:2]
    
    # Get neighborhood coordinates
    half = int(np.sqrt(neighborhood_size))
    coords = []
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                coords.append((nx, ny))
    
    if len(coords) < 4:
        return 0.0
    
    # Test with minimal payload
    payload = b'\x55\xAA\x55\xAA'  # Alternating pattern
    
    try:
        # Embed in neighborhood
        stego = embed_payload(image, payload, coords[:len(payload) * 8 // 3 + 1], 1)
        
        # Test against corruptions
        scores = []
        for quality in [85, 75, 65]:
            corrupted = apply_jpeg_compression(stego, quality)
            
            # Measure bit accuracy
            original_bits = _bytes_to_bits(payload)
            extracted = extract_payload(corrupted, len(payload), 
                                        coords[:len(payload) * 8 // 3 + 1], 1)
            extracted_bits = _bytes_to_bits(extracted)
            
            # Calculate bit error rate
            min_len = min(len(original_bits), len(extracted_bits))
            errors = sum(a != b for a, b in zip(original_bits[:min_len], 
                                                 extracted_bits[:min_len]))
            accuracy = 1.0 - (errors / min_len if min_len > 0 else 1.0)
            scores.append(accuracy)
        
        return np.mean(scores)
    
    except Exception:
        return 0.0


# =============================================================================
# Dataset Classes
# =============================================================================

class PixelSuitabilityDataset(Dataset):
    """
    PyTorch Dataset for pixel suitability training.
    
    Provides (patch, label) pairs where:
    - patch: 32x32 grayscale image patch
    - label: 0 (weak) or 1 (robust) for classification
             or float robustness score for regression
    """
    
    def __init__(self, data_path: str, transform=None, mode: str = 'classification'):
        """
        Args:
            data_path: Path to .npz file with 'patches' and 'labels'
            transform: Optional torchvision transforms
            mode: 'classification' or 'regression'
        """
        self.transform = transform
        self.mode = mode
        
        data = np.load(data_path)
        self.patches = data['patches']  # N x H x W
        self.labels = data['labels']    # N
        
        # Ensure patches are float32 in [0, 1]
        if self.patches.max() > 1.0:
            self.patches = self.patches.astype(np.float32) / 255.0
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[idx]
        label = self.labels[idx]
        
        # Add channel dimension
        patch = torch.from_numpy(patch).unsqueeze(0).float()
        
        if self.transform:
            patch = self.transform(patch)
        
        if self.mode == 'classification':
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(label, dtype=torch.float32)
        
        return patch, label


class PixelMapDataset(Dataset):
    """
    Dataset for full-image pixel suitability maps.
    
    Provides (image, suitability_map) pairs for U-Net style training.
    """
    
    def __init__(self, data_path: str, transform=None):
        """
        Args:
            data_path: Path to .npz file with 'images' and 'maps'
            transform: Optional transforms
        """
        self.transform = transform
        
        data = np.load(data_path)
        self.images = data['images']  # N x H x W
        self.maps = data['maps']      # N x H x W
        
        if self.images.max() > 1.0:
            self.images = self.images.astype(np.float32) / 255.0
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        smap = self.maps[idx]
        
        image = torch.from_numpy(image).unsqueeze(0).float()
        smap = torch.from_numpy(smap).unsqueeze(0).float()
        
        if self.transform:
            image = self.transform(image)
        
        return image, smap


# =============================================================================
# Data Generation Functions
# =============================================================================

def generate_patch_dataset(image_dir: Path, output_path: Path,
                           config: DatasetConfig,
                           max_images: int = 100,
                           verbose: bool = True) -> Dict[str, int]:
    """
    Generate patch-based training dataset.
    
    Args:
        image_dir: Directory containing training images
        output_path: Path to save .npz dataset
        config: Dataset configuration
        max_images: Maximum number of images to process
        verbose: Print progress
    
    Returns:
        Statistics dictionary
    """
    import cv2
    
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Find images
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_paths.extend(list(image_dir.glob(ext)))
        image_paths.extend(list(image_dir.glob(ext.upper())))
    
    image_paths = image_paths[:max_images]
    
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    all_patches = []
    all_labels = []
    
    for img_idx, img_path in enumerate(image_paths):
        if verbose:
            print(f"Processing {img_idx + 1}/{len(image_paths)}: {img_path.name}")
        
        # Load image
        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Rank pixels by texture
        ranked = rank_pixels_by_texture(img, patch_size=5)
        
        # Sample from different regions
        samples_per_region = config.num_samples_per_image // 3
        
        # Top pixels (likely robust)
        top_coords = ranked[:samples_per_region]
        
        # Middle pixels (uncertain)
        mid_start = len(ranked) // 3
        mid_coords = ranked[mid_start:mid_start + samples_per_region]
        
        # Bottom pixels (likely weak)
        bottom_coords = ranked[-samples_per_region:]
        
        for coords_set, region_name in [(top_coords, 'top'), 
                                         (mid_coords, 'mid'),
                                         (bottom_coords, 'bottom')]:
            for (x, y) in coords_set:
                # Extract patch
                patch = extract_patch(gray, x, y, config.patch_size)
                
                if patch.shape != (config.patch_size, config.patch_size):
                    continue
                
                # Test robustness
                robustness = test_single_pixel_robustness(img, x, y, config)
                
                # Binary label
                label = 1 if robustness > 0.7 else 0
                
                all_patches.append(patch.astype(np.float32) / 255.0)
                all_labels.append(label)
    
    # Convert to arrays
    patches = np.stack(all_patches)
    labels = np.array(all_labels, dtype=np.float32)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, patches=patches, labels=labels)
    
    # Statistics
    stats = {
        'total_samples': len(labels),
        'positive_samples': int(labels.sum()),
        'negative_samples': int(len(labels) - labels.sum()),
        'images_processed': len(image_paths)
    }
    
    if verbose:
        print(f"\n✅ Dataset saved to {output_path}")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Positive (robust): {stats['positive_samples']}")
        print(f"   Negative (weak): {stats['negative_samples']}")
    
    return stats


def generate_map_dataset(image_dir: Path, output_path: Path,
                         config: DatasetConfig,
                         target_size: Tuple[int, int] = (256, 256),
                         max_images: int = 50,
                         verbose: bool = True) -> Dict[str, int]:
    """
    Generate full-image suitability map dataset for U-Net training.
    
    Args:
        image_dir: Directory containing training images
        output_path: Path to save .npz dataset
        config: Dataset configuration
        target_size: Resize images to this size
        max_images: Maximum number of images to process
        verbose: Print progress
    
    Returns:
        Statistics dictionary
    """
    import cv2
    
    np.random.seed(config.seed)
    
    # Find images
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_paths.extend(list(image_dir.glob(ext)))
    
    image_paths = image_paths[:max_images]
    
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    all_images = []
    all_maps = []
    
    for img_idx, img_path in enumerate(image_paths):
        if verbose:
            print(f"Processing {img_idx + 1}/{len(image_paths)}: {img_path.name}")
        
        # Load and resize
        img = np.array(Image.open(img_path).convert('RGB'))
        img = cv2.resize(img, target_size)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Compute suitability map
        suitability_map = np.zeros((h, w), dtype=np.float32)
        
        # Use sliding window for efficiency
        step = 4  # Compute every 4th pixel, interpolate rest
        for y in range(0, h, step):
            for x in range(0, w, step):
                score = test_single_pixel_robustness(img, x, y, config)
                
                # Fill local region
                y_end = min(y + step, h)
                x_end = min(x + step, w)
                suitability_map[y:y_end, x:x_end] = score
        
        # Smooth the map
        suitability_map = cv2.GaussianBlur(suitability_map, (5, 5), 0)
        
        all_images.append(gray.astype(np.float32) / 255.0)
        all_maps.append(suitability_map)
    
    # Convert to arrays
    images = np.stack(all_images)
    maps = np.stack(all_maps)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, images=images, maps=maps)
    
    stats = {
        'total_images': len(images),
        'image_size': target_size
    }
    
    if verbose:
        print(f"\n✅ Map dataset saved to {output_path}")
        print(f"   Total images: {stats['total_images']}")
    
    return stats


# =============================================================================
# Data Augmentation
# =============================================================================

class StegoAugmentation:
    """Data augmentation for steganography training."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Random horizontal flip
        if random.random() < self.p:
            x = torch.flip(x, dims=[-1])
        
        # Random vertical flip
        if random.random() < self.p:
            x = torch.flip(x, dims=[-2])
        
        # Random rotation (90 degree increments)
        if random.random() < self.p:
            k = random.randint(1, 3)
            x = torch.rot90(x, k, dims=[-2, -1])
        
        # Random brightness adjustment
        if random.random() < self.p:
            factor = random.uniform(0.8, 1.2)
            x = torch.clamp(x * factor, 0, 1)
        
        # Random contrast adjustment
        if random.random() < self.p:
            factor = random.uniform(0.8, 1.2)
            mean = x.mean()
            x = torch.clamp((x - mean) * factor + mean, 0, 1)
        
        return x


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate training data for pixel selector')
    parser.add_argument('--mode', type=str, choices=['patches', 'maps'], default='patches',
                        help='Dataset type to generate')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .npz file path')
    parser.add_argument('--max-images', type=int, default=100,
                        help='Maximum images to process')
    parser.add_argument('--patch-size', type=int, default=32,
                        help='Patch size for patch dataset')
    parser.add_argument('--samples-per-image', type=int, default=500,
                        help='Samples per image for patch dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    config = DatasetConfig(
        patch_size=args.patch_size,
        num_samples_per_image=args.samples_per_image,
        seed=args.seed
    )
    
    if args.mode == 'patches':
        generate_patch_dataset(
            Path(args.input),
            Path(args.output),
            config,
            max_images=args.max_images
        )
    else:
        generate_map_dataset(
            Path(args.input),
            Path(args.output),
            config,
            max_images=args.max_images
        )


if __name__ == '__main__':
    main()
