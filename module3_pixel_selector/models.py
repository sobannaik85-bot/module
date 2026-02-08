"""
models.py
==========
Proper Deep Learning Models for Steganographic Pixel Selection

This module implements research-backed neural network architectures for
identifying robust embedding locations in images. The models predict
pixel-level suitability scores based on local texture complexity,
edge characteristics, and noise tolerance.

Architectures Implemented:
1. PixelSelectorUNet - U-Net based dense prediction network
2. PixelSelectorResNet - ResNet-based patch classifier
3. StegoSuitabilityNet - Custom architecture optimized for steganography

References:
- SRNet (Boroumand et al., 2018)
- XuNet (Xu et al., 2016)
- U-Net (Ronneberger et al., 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBNReLU(nn.Module):
    """Convolution + Batch Normalization + ReLU block."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class SRMFilterLayer(nn.Module):
    """
    Spatial Rich Model (SRM) filter layer.
    
    Applies high-pass filters used in steganalysis to detect 
    subtle pixel modifications. This helps the model learn 
    which areas are more susceptible to detection.
    """
    
    def __init__(self):
        super().__init__()
        # Define SRM filters (3 basic high-pass filters)
        # Filter 1: Basic 3x3 high-pass
        f1 = np.array([
            [-1, 2, -1],
            [2, -4, 2],
            [-1, 2, -1]
        ], dtype=np.float32) / 4.0
        
        # Filter 2: Edge detector
        f2 = np.array([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32) / 4.0
        
        # Filter 3: SPAM filter
        f3 = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32) / 12.0
        
        # Register as fixed (non-trainable) convolution weights
        self.register_buffer('filter1', torch.from_numpy(f1).unsqueeze(0).unsqueeze(0))
        self.register_buffer('filter2', torch.from_numpy(f2).unsqueeze(0).unsqueeze(0))
        self.register_buffer('filter3', torch.from_numpy(f3).unsqueeze(0).unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SRM filters and concatenate results."""
        # x: B x 1 x H x W (grayscale)
        out1 = F.conv2d(x, self.filter1, padding=1)
        out2 = F.conv2d(x, self.filter2, padding=2)
        out3 = F.conv2d(x, self.filter3, padding=2)
        return torch.cat([x, out1, out2, out3], dim=1)  # B x 4 x H x W


class AttentionGate(nn.Module):
    """Attention gate for focusing on important regions."""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# =============================================================================
# Main Model: PixelSelectorUNet
# =============================================================================

class PixelSelectorUNet(nn.Module):
    """
    U-Net based pixel suitability predictor.
    
    This model predicts a per-pixel suitability score (0-1) indicating
    how robust each pixel location is for steganographic embedding.
    
    Features:
    - Multi-scale feature extraction (encoder-decoder)
    - Skip connections for preserving spatial information
    - SRM filters for steganalysis-aware features
    - Attention gates for focusing on textured regions
    
    Input: Grayscale image (B x 1 x H x W)
    Output: Suitability map (B x 1 x H x W), values in [0, 1]
    """
    
    def __init__(self, in_channels: int = 1, base_filters: int = 32):
        super().__init__()
        
        # SRM preprocessing
        self.srm = SRMFilterLayer()
        in_ch = 4  # 1 original + 3 SRM filters
        
        # Encoder
        self.enc1 = nn.Sequential(
            ConvBNReLU(in_ch, base_filters),
            ConvBNReLU(base_filters, base_filters),
            ResidualBlock(base_filters)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            ConvBNReLU(base_filters, base_filters * 2),
            ConvBNReLU(base_filters * 2, base_filters * 2),
            ResidualBlock(base_filters * 2)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            ConvBNReLU(base_filters * 2, base_filters * 4),
            ConvBNReLU(base_filters * 4, base_filters * 4),
            ResidualBlock(base_filters * 4)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNReLU(base_filters * 4, base_filters * 8),
            ResidualBlock(base_filters * 8),
            ResidualBlock(base_filters * 8),
            ConvBNReLU(base_filters * 8, base_filters * 4)
        )
        
        # Decoder with attention
        self.up3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 4, 2, stride=2)
        self.att3 = AttentionGate(base_filters * 4, base_filters * 4, base_filters * 2)
        self.dec3 = nn.Sequential(
            ConvBNReLU(base_filters * 8, base_filters * 4),
            ConvBNReLU(base_filters * 4, base_filters * 2)
        )
        
        self.up2 = nn.ConvTranspose2d(base_filters * 2, base_filters * 2, 2, stride=2)
        self.att2 = AttentionGate(base_filters * 2, base_filters * 2, base_filters)
        self.dec2 = nn.Sequential(
            ConvBNReLU(base_filters * 4, base_filters * 2),
            ConvBNReLU(base_filters * 2, base_filters)
        )
        
        self.up1 = nn.ConvTranspose2d(base_filters, base_filters, 2, stride=2)
        self.att1 = AttentionGate(base_filters, base_filters, base_filters // 2)
        self.dec1 = nn.Sequential(
            ConvBNReLU(base_filters * 2, base_filters),
            ConvBNReLU(base_filters, base_filters)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(base_filters, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply SRM filters
        x = self.srm(x)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder
        d3 = self.up3(b)
        # Handle size mismatch
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        
        return self.output(d1)


# =============================================================================
# Alternative Model: PixelSelectorResNet (Patch-based)
# =============================================================================

class PixelSelectorResNet(nn.Module):
    """
    ResNet-based patch classifier for pixel selection.
    
    This model classifies individual patches as suitable/unsuitable
    for embedding. More efficient for inference on specific regions.
    
    Input: Patches (B x 1 x 32 x 32)
    Output: Suitability score (B,), values in [0, 1]
    """
    
    def __init__(self, patch_size: int = 32):
        super().__init__()
        
        self.srm = SRMFilterLayer()
        
        # Feature extractor
        self.features = nn.Sequential(
            ConvBNReLU(4, 32),  # 4 channels from SRM
            ResidualBlock(32),
            nn.MaxPool2d(2),  # 16x16
            
            ConvBNReLU(32, 64),
            ResidualBlock(64),
            nn.MaxPool2d(2),  # 8x8
            
            ConvBNReLU(64, 128),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.MaxPool2d(2),  # 4x4
            
            ConvBNReLU(128, 256),
            ResidualBlock(256),
            nn.AdaptiveAvgPool2d(1)  # 1x1
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.srm(x)
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)


# =============================================================================
# Lightweight Model: StegoSuitabilityNet
# =============================================================================

class StegoSuitabilityNet(nn.Module):
    """
    Lightweight model optimized for steganography pixel selection.
    
    This model balances accuracy and speed, suitable for real-time
    applications. Uses depthwise separable convolutions and efficient
    attention mechanisms.
    
    Input: Grayscale image (B x 1 x H x W)
    Output: Suitability map (B x 1 x H x W), values in [0, 1]
    """
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        # Initial feature extraction with SRM-inspired filters
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, dilation=1, groups=32, bias=False),
            nn.Conv2d(32, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=2, dilation=2, groups=32, bias=False),
            nn.Conv2d(32, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=4, dilation=4, groups=32, bias=False),
            nn.Conv2d(32, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(48, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial features
        x = self.initial(x)
        
        # Multi-scale branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Concatenate multi-scale features
        x = torch.cat([b1, b2, b3], dim=1)
        x = self.fusion(x)
        
        # Apply attention
        spatial = self.spatial_att(x)
        channel = self.channel_att(x).unsqueeze(-1).unsqueeze(-1)
        x = x * spatial * channel
        
        return self.output(x)


# =============================================================================
# Model Factory
# =============================================================================

def get_model(model_name: str = 'unet', **kwargs) -> nn.Module:
    """
    Factory function to get a model by name.
    
    Args:
        model_name: One of 'unet', 'resnet', 'lightweight'
        **kwargs: Model-specific arguments
    
    Returns:
        Initialized model
    """
    models = {
        'unet': PixelSelectorUNet,
        'resnet': PixelSelectorResNet,
        'lightweight': StegoSuitabilityNet
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](**kwargs)


# =============================================================================
# Loss Functions
# =============================================================================

class RobustnessLoss(nn.Module):
    """
    Custom loss function for robustness-aware pixel selection.
    
    Combines:
    1. Binary cross-entropy for classification
    2. Focal loss to handle class imbalance
    3. Smoothness regularization
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, 
                 smoothness_weight: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothness_weight = smoothness_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Focal loss
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = (self.alpha * target + (1 - self.alpha) * (1 - target)) * focal_weight * bce
        
        # Smoothness regularization (encourage spatially coherent predictions)
        if pred.dim() == 4:
            dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
            dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
            smoothness = (dx.mean() + dy.mean()) / 2
        else:
            smoothness = 0.0
        
        return focal_loss.mean() + self.smoothness_weight * smoothness


class DiceLoss(nn.Module):
    """Dice loss for better handling of class imbalance."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined loss: BCE + Dice + Smoothness."""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.3,
                 smoothness_weight: float = 0.2):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smoothness_weight = smoothness_weight
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE loss
        bce = F.binary_cross_entropy(pred, target)
        
        # Dice loss
        dice = self.dice(pred, target)
        
        # Smoothness (TV regularization)
        if pred.dim() == 4:
            dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean()
            dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
            smoothness = dx + dy
        else:
            smoothness = 0.0
        
        return self.bce_weight * bce + self.dice_weight * dice + self.smoothness_weight * smoothness


if __name__ == "__main__":
    # Test models
    print("Testing PixelSelectorUNet...")
    model = PixelSelectorUNet()
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"  Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 1, 256, 256), "UNet output shape mismatch"
    print("  ✓ UNet test passed")
    
    print("\nTesting PixelSelectorResNet...")
    model = PixelSelectorResNet(patch_size=32)
    x = torch.randn(8, 1, 32, 32)
    y = model(x)
    print(f"  Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (8,), "ResNet output shape mismatch"
    print("  ✓ ResNet test passed")
    
    print("\nTesting StegoSuitabilityNet...")
    model = StegoSuitabilityNet()
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"  Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 1, 256, 256), "Lightweight output shape mismatch"
    print("  ✓ Lightweight test passed")
    
    print("\n✅ All model tests passed!")
