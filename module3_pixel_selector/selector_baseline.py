"""
selector_baseline.py
Heuristic baseline pixel selector.

API:
    select_pixels(image_np, payload_bits, patch_size=3, top_k_strategy='count')
Returns:
    List[(x,y)] ordered by priority (length == payload_bits)
Notes:
    - image_np: HxWx3 RGB uint8 numpy array
    - payload_bits: total number of bits you need to embed
    - patch_size: neighborhood size for local features
"""

from typing import List, Tuple
import numpy as np
import cv2

from .selector_utils import get_gray, extract_patch, compute_entropy, patch_variance

def _compute_score_for_pixel(gray: np.ndarray, x: int, y: int, patch_size: int) -> float:
    patch = extract_patch(gray, x, y, patch_size)
    lap = float(cv2.Laplacian(patch, cv2.CV_64F).var())
    ent = compute_entropy(patch)
    var = patch_variance(patch)
    # weighted sum (tunable)
    return (lap * 1.0) + (ent * 0.8) + (var * 0.2)

def select_pixels(image_np: np.ndarray, payload_bits: int, patch_size: int = 3, lsb_bits: int = 1, seed: int = 0) -> List[Tuple[int,int]]:
    """
    Select top pixels for embedding by score.
    payload_bits = number of bits to embed. Each pixel has channels * lsb_bits capacity.
    For RGB and embedding per channel, capacity_per_pixel = 3 * lsb_bits.
    """
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("image_np must be HxWx3 RGB numpy array")
    h, w, _ = image_np.shape
    capacity_per_pixel = 3 * lsb_bits
    pixels_needed = int(np.ceil(payload_bits / capacity_per_pixel))
    gray = get_gray(image_np)

    scores = []
    # compute score per pixel (can be optimized to sample or patch-grid)
    for y in range(h):
        for x in range(w):
            s = _compute_score_for_pixel(gray, x, y, patch_size)
            scores.append((s, x, y))
    # sort descending by score
    scores.sort(reverse=True, key=lambda t: t[0])

    # deterministic tie-breaker by seed
    np.random.seed(seed)
    # pick top pixels_needed
    selected = [(int(x), int(y)) for (_, x, y) in scores[:pixels_needed]]
    return selected
