"""
selector_utils.py
Utility functions for Module 3: entropy, patch extraction, scoring helpers.
"""

from typing import Tuple
import numpy as np
import cv2
from math import log2

def compute_entropy(patch: np.ndarray) -> float:
    """Shannon entropy for a grayscale patch (0-255)."""
    if patch.size == 0:
        return 0.0
    hist = np.bincount(patch.flatten(), minlength=256).astype(np.float32)
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -float((prob * np.log2(prob)).sum())

def patch_variance(patch: np.ndarray) -> float:
    return float(np.var(patch.astype(np.float32)))

def get_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB numpy image to grayscale (uint8)."""
    if image_rgb.ndim == 2:
        return image_rgb
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

def extract_patch(gray: np.ndarray, cx: int, cy: int, size: int = 3) -> np.ndarray:
    h, w = gray.shape
    half = size // 2
    y1 = max(0, cy - half)
    y2 = min(h, cy + half + 1)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half + 1)
    return gray[y1:y2, x1:x2]
