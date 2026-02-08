# steg otool/modules/module3_pixel_selector/generate_training_data.py
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple
# ---- begin import-compat shim ----
import sys
from pathlib import Path

try:
    # preferred: package-relative import (works when run with -m)
    from .selector_baseline import select_pixels
    from .selector_utils import get_gray
except Exception:
    # fallback: running as a plain script. Add project root to sys.path and import absolutely.
    # project_root = parents[3] because file is at .../stegotool/modules/module3_pixel_selector/...
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from stegotool.modules.module3_pixel_selector.selector_baseline import select_pixels
    from stegotool.modules.module3_pixel_selector.selector_utils import get_gray
# ---- end import-compat shim ----


from .selector_baseline import select_pixels
from .selector_utils import get_gray

# default small sample path (in-repo)
REPO_SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "module3"
REPO_SAMPLE_PATH.mkdir(parents=True, exist_ok=True)

def extract_patches_and_labels(image_np: np.ndarray, payload_bits=1024,
                               patch_size=5, lsb_bits=1, corrupt_step=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heuristic self-supervised labeling:
    - get baseline top pixels (good)
    - create patches centered on pixels (positive)
    - create negative patches from low-score pixels (neg)
    - Optionally apply a corruption step (JPEG recompress) and prefer pixels robust to corruption
    """
    from io import BytesIO
    from PIL import Image
    import cv2

    h, w, _ = image_np.shape
    gray = get_gray(image_np)
    # get baseline ranking for a generous payload size
    bits = payload_bits
    ranked = select_pixels(image_np, payload_bits=bits, patch_size=patch_size, lsb_bits=lsb_bits, seed=0)
    # choose top N and bottom N
    capacity_per_pixel = 3 * lsb_bits
    pixels_needed = int(np.ceil(bits / capacity_per_pixel))
    top_k = max(128, pixels_needed)        # number of positives to collect (capable)
    bottom_k = top_k                       # negatives

    top_coords = ranked[:top_k]
    bottom_coords = ranked[-bottom_k:]

    patches = []
    labels = []

    half = patch_size // 2
    padded = np.pad(gray, pad_width=half, mode='reflect')

    def get_patch_xy(x,y):
        py = y + half
        px = x + half
        patch = padded[py-half:py+half+1, px-half:px+half+1]
        return patch

    # positives
    for (x,y) in top_coords:
        p = get_patch_xy(x,y)
        patches.append(p.astype(np.float32) / 255.0)
        labels.append(1)

    # negatives
    for (x,y) in bottom_coords:
        p = get_patch_xy(x,y)
        patches.append(p.astype(np.float32) / 255.0)
        labels.append(0)

    patches = np.stack(patches)  # N x H x W
    labels = np.array(labels).astype(np.int64)
    return patches, labels

def build_sample_dataset(input_images, out_path: Path, per_image_limit=1):
    """
    input_images: list of image file paths
    Saves a small .npz containing arrays 'patches' and 'labels'
    """
    all_patches = []
    all_labels = []
    for i, imgp in enumerate(input_images):
        if i >= per_image_limit:
            break
        im = Image.open(imgp).convert("RGB")
        arr = np.array(im)
        patches, labels = extract_patches_and_labels(arr)
        all_patches.append(patches)
        all_labels.append(labels)
    patches = np.concatenate(all_patches, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, patches=patches, labels=labels)
    print(f"Saved sample dataset to {out_path}  (patches={patches.shape}, labels={labels.shape})")

def find_dev_images(repo_limit=5):
    # looks for data/dev_images in repo root; fallback to sample images in module
    root = Path(__file__).resolve().parents[3]
    candidate = root / "data" / "dev_images"
    if candidate.exists():
        imgs = list(candidate.glob("*.*"))
        return imgs[:repo_limit]
    # fallback: try to find any png in repo
    imgs = list(root.glob("**/*.png"))
    return imgs[:repo_limit]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(REPO_SAMPLE_PATH / "sample.npz"),
                        help="Output .npz path for small sample (default in-repo)")
    parser.add_argument("--full_out", type=str, default=None,
                        help="Optional path to write full dataset (outside repo)")
    parser.add_argument("--limit", type=int, default=1, help="How many images to use for sample")
    args = parser.parse_args()
    dev = find_dev_images(repo_limit=args.limit)
    if not dev:
        print("No dev images found in repo. Place some in data/dev_images/ or pass --out a path.")
        return
    out_path = Path(args.out)
    build_sample_dataset(dev, out_path, per_image_limit=args.limit)
    if args.full_out:
        # generate a larger dataset from more images — same format but potentially large
        full_out = Path(args.full_out)
        all_imgs = list((Path(__file__).resolve().parents[3] / "data" / "dev_images").glob("*.*"))
        # naive: process all images
        all_patches = []
        all_labels = []
        for imgp in all_imgs:
            im = Image.open(imgp).convert("RGB")
            arr = np.array(im)
            patches, labels = extract_patches_and_labels(arr)
            all_patches.append(patches)
            all_labels.append(labels)
        patches = np.concatenate(all_patches, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        np.savez_compressed(full_out, patches=patches, labels=labels)
        print(f"Saved full dataset to {full_out} (patches={patches.shape})")

if __name__ == "__main__":
    main()
