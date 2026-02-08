"""
selector_model.py
Lightweight ML model skeleton for pixel selection.

This file provides:
- a PyTorch model class (tiny CNN) to score patches
- helper to run inference over an image and return pixel scores

You can train later with labeled data (robust=1/weak=0) produced by baseline + augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from .selector_utils import get_gray, extract_patch

class TinyPatchNet(nn.Module):
    def __init__(self):
        super().__init__()
        # small network for 3x3 or 5x5 patches (input 1 channel)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x: Bx1xHxW
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze(1)

def infer_scores(model: TinyPatchNet, image_np: np.ndarray, patch_size: int = 5, device='cpu', batch_size=1024) -> List[Tuple[float, int, int]]:
    """
    Compute score for each pixel using the model.
    Returns list of (score, x, y) sorted descending.
    """
    model.to(device)
    model.eval()
    gray = get_gray(image_np)
    h, w = gray.shape
    patches = []
    coords = []
    half = patch_size // 2
    # build patches, pad edges
    padded = np.pad(gray, pad_width=half, mode='reflect')
    for y in range(h):
        for x in range(w):
            py = y + half
            px = x + half
            patch = padded[py-half:py+half+1, px-half:px+half+1]
            patches.append(patch.astype(np.float32) / 255.0)
            coords.append((x, y))
    # batch inference
    scores = []
    with torch.no_grad():
        import torch
        arr = np.stack(patches)  # N x H x W
        N, H, W = arr.shape
        arr = arr.reshape(N, 1, H, W)
        tensor = torch.from_numpy(arr).float().to(device)
        for i in range(0, N, batch_size):
            batch = tensor[i:i+batch_size]
            out = model(batch).cpu().numpy()
            for j, s in enumerate(out):
                idx = i + j
                x, y = coords[idx]
                scores.append((float(s), x, y))
    scores.sort(reverse=True, key=lambda t: t[0])
    return scores
