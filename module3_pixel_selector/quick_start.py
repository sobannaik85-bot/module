"""
quick_start.py
==============
Quick Start Script for Module 3 Pixel Selector

This script provides easy-to-use commands for:
1. Generating training data
2. Training a model
3. Running inference

Usage:
    python -m stegotool.modules.module3_pixel_selector.quick_start --help
"""

import argparse
import sys
from pathlib import Path


def cmd_generate(args):
    """Generate training dataset."""
    from .dataset import generate_patch_dataset, DatasetConfig
    
    # Find images
    image_dir = Path(args.images)
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        # Try to create demo images
        dev_images = Path(__file__).parent.parent.parent / "data" / "dev_images"
        if dev_images.exists():
            image_dir = dev_images
            print(f"   Using existing dev_images: {dev_images}")
        else:
            print("   Run make_demo_images.py first to create sample images")
            return 1
    
    # Configure
    config = DatasetConfig(
        patch_size=args.patch_size,
        num_samples_per_image=args.samples,
        seed=args.seed
    )
    
    # Output path
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Generating dataset...")
    print(f"   Images: {image_dir}")
    print(f"   Output: {output}")
    print(f"   Max images: {args.max_images}")
    print(f"   Samples/image: {args.samples}")
    
    stats = generate_patch_dataset(
        image_dir, output, config, 
        max_images=args.max_images,
        verbose=True
    )
    
    print(f"\n✅ Dataset generated successfully!")
    return 0


def cmd_train(args):
    """Train the model."""
    from .train import Trainer, TrainingConfig
    from datetime import datetime
    
    # Check data exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        print("   Run 'quick_start.py generate' first")
        return 1
    
    # Create config
    config = TrainingConfig(
        model_type=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed
    )
    
    # Output directory
    output_dir = Path(args.output) / f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🚀 Starting training...")
    print(f"   Model: {args.model}")
    print(f"   Data: {data_path}")
    print(f"   Output: {output_dir}")
    
    trainer = Trainer(config, output_dir)
    results = trainer.train(data_path)
    
    # Copy best model to standard location
    model_dir = Path(__file__).parent.parent.parent / "models" / "module3_pixel_selector"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    best_model = output_dir / "best.pth"
    if best_model.exists():
        shutil.copy(best_model, model_dir / "best.pth")
        print(f"\n✅ Model copied to {model_dir / 'best.pth'}")
    
    return 0


def cmd_test(args):
    """Test the trained model."""
    from .inference import PixelSelector
    import numpy as np
    from PIL import Image
    
    # Load image
    if args.image:
        image_path = Path(args.image)
    else:
        # Find a demo image
        dev_images = Path(__file__).parent.parent.parent / "data" / "dev_images"
        images = list(dev_images.glob("*.png")) + list(dev_images.glob("*.jpg"))
        if not images:
            print("❌ No test image found. Specify --image")
            return 1
        image_path = images[0]
    
    print(f"🔍 Testing model...")
    print(f"   Image: {image_path}")
    
    image = np.array(Image.open(image_path).convert('RGB'))
    print(f"   Size: {image.shape}")
    
    # Create selector
    selector = PixelSelector()
    
    # Get suitability map
    suitability = selector.get_suitability_map(image)
    print(f"   Suitability range: [{suitability.min():.3f}, {suitability.max():.3f}]")
    
    # Select pixels
    coords = selector.select_pixels(image, num_pixels=args.num_pixels)
    print(f"   Selected {len(coords)} pixels")
    
    # Visualize if requested
    if args.visualize:
        output_path = args.output or f"{image_path.stem}_selection.png"
        vis = selector.visualize_selection(image, coords, output_path)
        print(f"   Saved visualization: {output_path}")
    
    # Show some stats
    scores = [suitability[y, x] for (x, y) in coords]
    print(f"\n   Selected pixel scores:")
    print(f"     Min: {min(scores):.3f}")
    print(f"     Max: {max(scores):.3f}")
    print(f"     Mean: {np.mean(scores):.3f}")
    
    print("\n✅ Test completed!")
    return 0


def cmd_benchmark(args):
    """Benchmark model performance."""
    from .inference import PixelSelector
    from .selector_baseline import select_pixels as baseline_select
    import numpy as np
    from PIL import Image
    import time
    
    # Find images
    dev_images = Path(__file__).parent.parent.parent / "data" / "dev_images"
    images = list(dev_images.glob("*.png")) + list(dev_images.glob("*.jpg"))
    
    if not images:
        print("❌ No images found for benchmarking")
        return 1
    
    print(f"⏱️  Benchmarking on {len(images)} images...")
    
    # Initialize selector
    selector = PixelSelector()
    
    baseline_times = []
    ml_times = []
    
    for img_path in images[:args.num_images]:
        image = np.array(Image.open(img_path).convert('RGB'))
        payload_bits = 10000
        
        # Baseline
        start = time.time()
        _ = baseline_select(image, payload_bits)
        baseline_times.append(time.time() - start)
        
        # ML
        start = time.time()
        _ = selector.select_pixels(image, payload_bits=payload_bits)
        ml_times.append(time.time() - start)
    
    print(f"\nResults:")
    print(f"  Baseline: {np.mean(baseline_times)*1000:.1f}ms avg")
    print(f"  ML Model: {np.mean(ml_times)*1000:.1f}ms avg")
    print(f"  Speedup:  {np.mean(baseline_times)/np.mean(ml_times):.2f}x")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Module 3 Pixel Selector - Quick Start',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data
  python -m stegotool.modules.module3_pixel_selector.quick_start generate

  # Train a model
  python -m stegotool.modules.module3_pixel_selector.quick_start train
  
  # Test the model
  python -m stegotool.modules.module3_pixel_selector.quick_start test --visualize
  
  # Benchmark performance
  python -m stegotool.modules.module3_pixel_selector.quick_start benchmark
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate training data')
    gen_parser.add_argument('--images', type=str, 
                           default=str(Path(__file__).parent.parent.parent / "data" / "dev_images"),
                           help='Directory containing training images')
    gen_parser.add_argument('--output', type=str,
                           default=str(Path(__file__).parent.parent.parent / "data" / "module3" / "training_data.npz"),
                           help='Output dataset path')
    gen_parser.add_argument('--max-images', type=int, default=50, help='Max images to process')
    gen_parser.add_argument('--samples', type=int, default=200, help='Samples per image')
    gen_parser.add_argument('--patch-size', type=int, default=32, help='Patch size')
    gen_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data', type=str,
                             default=str(Path(__file__).parent.parent.parent / "data" / "module3" / "training_data.npz"),
                             help='Training dataset path')
    train_parser.add_argument('--output', type=str,
                             default=str(Path(__file__).parent.parent.parent / "models" / "module3_pixel_selector" / "training_runs"),
                             help='Output directory for training')
    train_parser.add_argument('--model', type=str, default='lightweight',
                             choices=['unet', 'resnet', 'lightweight'],
                             help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test model')
    test_parser.add_argument('--image', type=str, default=None, help='Test image path')
    test_parser.add_argument('--num-pixels', type=int, default=1000, help='Pixels to select')
    test_parser.add_argument('--visualize', action='store_true', help='Save visualization')
    test_parser.add_argument('--output', type=str, default=None, help='Visualization output path')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark performance')
    bench_parser.add_argument('--num-images', type=int, default=10, help='Images to benchmark')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    commands = {
        'generate': cmd_generate,
        'train': cmd_train,
        'test': cmd_test,
        'benchmark': cmd_benchmark
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
