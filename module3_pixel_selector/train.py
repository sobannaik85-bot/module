"""
train.py
=========
Comprehensive Training Pipeline for Pixel Selection Models

This module provides:
1. Full training loop with validation
2. Learning rate scheduling
3. Early stopping
4. Model checkpointing
5. Training visualization and logging
6. Support for multiple model architectures

Usage:
    python -m stegotool.modules.module3_pixel_selector.train --help
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    ReduceLROnPlateau,
    OneCycleLR
)

# Local imports
try:
    from .models import (
        PixelSelectorUNet, 
        PixelSelectorResNet, 
        StegoSuitabilityNet,
        get_model,
        CombinedLoss,
        RobustnessLoss
    )
    from .dataset import (
        PixelSuitabilityDataset, 
        PixelMapDataset,
        StegoAugmentation,
        DatasetConfig,
        generate_patch_dataset
    )
except ImportError:
    from models import (
        PixelSelectorUNet, 
        PixelSelectorResNet, 
        StegoSuitabilityNet,
        get_model,
        CombinedLoss,
        RobustnessLoss
    )
    from dataset import (
        PixelSuitabilityDataset, 
        PixelMapDataset,
        StegoAugmentation,
        DatasetConfig,
        generate_patch_dataset
    )


# =============================================================================
# Configuration
# =============================================================================

class TrainingConfig:
    """Training configuration."""
    
    def __init__(
        self,
        model_type: str = 'lightweight',
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        min_delta: float = 1e-4,
        scheduler: str = 'cosine',
        loss_type: str = 'combined',
        val_split: float = 0.15,
        test_split: float = 0.1,
        seed: int = 42,
        num_workers: int = 4,
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        save_frequency: int = 5,
    ):
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.scheduler = scheduler
        self.loss_type = loss_type
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        self.save_frequency = save_frequency
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingConfig':
        return cls(**d)


# =============================================================================
# Training Utilities
# =============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MetricsTracker:
    """Track training metrics."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def update(self, train_loss: float, val_loss: float,
               train_acc: float, val_acc: float, lr: float):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['learning_rate'].append(lr)
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot(self, save_path: Optional[Path] = None):
        """Plot training curves."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Loss
            axes[0].plot(self.history['train_loss'], label='Train')
            axes[0].plot(self.history['val_loss'], label='Validation')
            axes[0].set_title('Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].legend()
            
            # Accuracy
            axes[1].plot(self.history['train_acc'], label='Train')
            axes[1].plot(self.history['val_acc'], label='Validation')
            axes[1].set_title('Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
            
            # Learning rate
            axes[2].plot(self.history['learning_rate'])
            axes[2].set_title('Learning Rate')
            axes[2].set_xlabel('Epoch')
            axes[2].set_yscale('log')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150)
                plt.close()
            else:
                plt.show()
        except ImportError:
            print("matplotlib not available for plotting")


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Compute classification metrics."""
    with torch.no_grad():
        preds = (outputs > threshold).float()
        
        correct = (preds == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total if total > 0 else 0.0
        
        # Precision, Recall, F1
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device,
                scaler: Optional[torch.cuda.amp.GradScaler] = None,
                gradient_clip: float = 1.0) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                # Flatten outputs for patch models
                if outputs.dim() > 1:
                    outputs = outputs.view(outputs.size(0), -1).mean(dim=1)
                targets_flat = targets.view(-1)
                loss = criterion(outputs, targets_flat)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            # Flatten outputs for patch models
            if outputs.dim() > 1:
                outputs = outputs.view(outputs.size(0), -1).mean(dim=1)
            targets_flat = targets.view(-1)
            loss = criterion(outputs, targets_flat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        all_outputs.append(outputs.detach())
        all_targets.append(targets_flat.detach())
    
    avg_loss = total_loss / len(loader.dataset)
    
    # Compute accuracy
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_outputs, all_targets)
    
    return avg_loss, metrics['accuracy']


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, Dict[str, float]]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)
        # Flatten outputs for patch models
        if outputs.dim() > 1:
            outputs = outputs.view(outputs.size(0), -1).mean(dim=1)
        targets_flat = targets.view(-1)
        
        loss = criterion(outputs, targets_flat)
        
        total_loss += loss.item() * inputs.size(0)
        all_outputs.append(outputs)
        all_targets.append(targets_flat)
    
    avg_loss = total_loss / len(loader.dataset)
    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_outputs, all_targets)
    
    return avg_loss, metrics['accuracy'], metrics


# =============================================================================
# Main Training Class
# =============================================================================

class Trainer:
    """Main trainer class."""
    
    def __init__(self, config: TrainingConfig, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Loss
        self.criterion = self._create_loss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler (will be initialized with data)
        self.scheduler = None
        
        # Mixed precision
        self.scaler = None
        if config.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Tracking
        self.metrics = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
    
    def _create_model(self) -> nn.Module:
        """Create model based on config."""
        if self.config.model_type == 'unet':
            return PixelSelectorUNet()
        elif self.config.model_type == 'resnet':
            return PixelSelectorResNet(patch_size=32)
        else:
            return StegoSuitabilityNet()
    
    def _create_loss(self) -> nn.Module:
        """Create loss function based on config."""
        if self.config.loss_type == 'combined':
            return CombinedLoss()
        elif self.config.loss_type == 'robustness':
            return RobustnessLoss()
        else:
            return nn.BCELoss()
    
    def _create_scheduler(self, num_batches: int):
        """Create learning rate scheduler."""
        if self.config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        elif self.config.scheduler == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=num_batches
            )
    
    def prepare_data(self, dataset_path: Path) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data loaders."""
        # Load dataset
        if self.config.model_type in ['unet', 'lightweight']:
            # Check if we have map dataset, otherwise use patch dataset
            try:
                full_dataset = PixelMapDataset(str(dataset_path))
            except:
                full_dataset = PixelSuitabilityDataset(
                    str(dataset_path),
                    transform=StegoAugmentation(p=0.3)
                )
        else:
            full_dataset = PixelSuitabilityDataset(
                str(dataset_path),
                transform=StegoAugmentation(p=0.3)
            )
        
        # Split dataset
        total = len(full_dataset)
        test_size = int(total * self.config.test_split)
        val_size = int(total * self.config.val_split)
        train_size = total - val_size - test_size
        
        train_ds, val_ds, test_ds = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f"Dataset: {total} samples")
        print(f"  Train: {len(train_ds)}")
        print(f"  Val: {len(val_ds)}")
        print(f"  Test: {len(test_ds)}")
        
        return train_loader, val_loader, test_loader
    
    def train(self, dataset_path: Path) -> Dict:
        """Main training loop."""
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(dataset_path)
        
        # Create scheduler
        self._create_scheduler(len(train_loader))
        
        # Save config
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"\nStarting training for {self.config.epochs} epochs...")
        print(f"Model: {self.config.model_type}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = train_epoch(
                self.model, train_loader, self.optimizer,
                self.criterion, self.device, self.scaler,
                self.config.gradient_clip
            )
            
            # Validate
            val_loss, val_acc, val_metrics = validate(
                self.model, val_loader, self.criterion, self.device
            )
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update metrics
            self.metrics.update(train_loss, val_loss, train_acc, val_acc, current_lr)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif not isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self._save_checkpoint('best.pth', epoch, val_loss, val_acc)
            
            # Periodic save
            if epoch % self.config.save_frequency == 0:
                self._save_checkpoint(f'epoch_{epoch}.pth', epoch, val_loss, val_acc)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:3d}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save final model
        self._save_checkpoint('final.pth', epoch, val_loss, val_acc)
        
        # Save metrics
        self.metrics.save(self.output_dir / 'metrics.json')
        self.metrics.plot(self.output_dir / 'training_curves.png')
        
        # Test evaluation
        print("\nEvaluating on test set...")
        self._load_checkpoint('best.pth')
        test_loss, test_acc, test_metrics = validate(
            self.model, test_loader, self.criterion, self.device
        )
        
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        
        # Save test results
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'training_time_minutes': total_time / 60,
            'epochs_trained': epoch
        }
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _save_checkpoint(self, filename: str, epoch: int, 
                         val_loss: float, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config.to_dict()
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.output_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train pixel selection model')
    
    # Data
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training dataset (.npz)')
    parser.add_argument('--output', type=str, default='./training_output',
                        help='Output directory')
    
    # Model
    parser.add_argument('--model', type=str, default='lightweight',
                        choices=['unet', 'resnet', 'lightweight'],
                        help='Model architecture')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'onecycle'],
                        help='Learning rate scheduler')
    parser.add_argument('--loss', type=str, default='combined',
                        choices=['bce', 'combined', 'robustness'],
                        help='Loss function')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_type=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        scheduler=args.scheduler,
        loss_type=args.loss,
        seed=args.seed,
        num_workers=args.workers,
        mixed_precision=not args.no_mixed_precision
    )
    
    # Create trainer
    output_dir = Path(args.output) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(config, output_dir)
    
    # Train
    results = trainer.train(Path(args.data))
    
    print(f"\n✅ Training complete! Output saved to {output_dir}")


if __name__ == '__main__':
    main()
