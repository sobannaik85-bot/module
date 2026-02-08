"""
generate_labels_robust.py

Advanced labeler (embed -> compress -> extract -> ECC recover)
Produces dataset: patches (N x H x W) and labels (N,)

Usage (from project root):
python -m stegotool.modules.module3_pixel_selector.generate_labels_robust --out .\stegotool\data\module3\robust_labels.npz --limit 8

Notes:
- This script embeds the same test payload into groups of pixels (group_size).
- For each group, it attempts JPEG recompress and runs ECC recovery.
- If recovered payload == original payload => label = 1 (robust), else 0.
- Patches are centered on the group's central pixel.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from typing import List, Tuple
import io

# imports from module
from .selector_baseline import select_pixels
from .selector_utils import get_gray, extract_patch
from stegotool.modules.module6_redundancy.rs_wrapper import add_redundancy, recover_redundancy
from stegotool.modules.module6_redundancy.corruption_simulator import recompress_and_reload

# Minimal LSB embedding/extraction that writes bytes into given pixel coords (RGB order)
def _bytes_to_bits(b: bytes) -> List[int]:
    bits = []
    for byte in b:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits

def _bits_to_bytes(bits: List[int]) -> bytes:
    # pad to multiple of 8
    while len(bits) % 8 != 0:
        bits.append(0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (bits[i + j] & 1)
        out.append(byte)
    return bytes(out)

def embed_bytes_into_pixels(image_pil: Image.Image, payload: bytes, coords: List[Tuple[int,int]], lsb_bits: int = 1) -> Image.Image:
    """
    Embed payload bits into the specified pixel coords (RGB channels per pixel),
    using lsb_bits per channel.
    """
    arr = np.array(image_pil).copy()  # H x W x 3
    h, w, c = arr.shape
    assert c == 3
    capacity_bits = len(coords) * 3 * lsb_bits
    payload_bits = _bytes_to_bits(payload)
    if len(payload_bits) > capacity_bits:
        raise ValueError(f"Payload too large for coords: {len(payload_bits)} bits vs capacity {capacity_bits}")
    # pad payload bits
    payload_bits += [0] * (capacity_bits - len(payload_bits))
    bit_idx = 0
    for (x, y) in coords:
        # x is column, y is row as per our selector output
        for ch in range(3):
            for b in range(lsb_bits):
                bit = payload_bits[bit_idx]
                # set LSB b of arr[y,x,ch] (we treat b=0 as least significant)
                # To set the specific LSB (b-th), we first clear that bit then OR it
                mask = 1 << b
                arr[y, x, ch] = (arr[y, x, ch] & (~mask)) | (bit << b)
                bit_idx += 1
    return Image.fromarray(arr)

def extract_bytes_from_pixels(image_pil: Image.Image, num_payload_bytes: int, coords: List[Tuple[int,int]], lsb_bits: int = 1) -> bytes:
    arr = np.array(image_pil)
    h, w, c = arr.shape
    capacity_bits = len(coords) * 3 * lsb_bits
    needed_bits = num_payload_bytes * 8
    if needed_bits > capacity_bits:
        raise ValueError("Requested more bytes than capacity of coords")
    bits = []
    bit_idx = 0
    for (x, y) in coords:
        for ch in range(3):
            for b in range(lsb_bits):
                if bit_idx >= needed_bits:
                    break
                val = (arr[y, x, ch] >> b) & 1
                bits.append(int(val))
                bit_idx += 1
            if bit_idx >= needed_bits:
                break
        if bit_idx >= needed_bits:
            break
    return _bits_to_bytes(bits)

def build_label_dataset_for_image(image_path: Path,
                                  out_patches: List[np.ndarray],
                                  out_labels: List[int],
                                  payload_len: int = 16,
                                  group_size: int = 32,
                                  nsym: int = 32,
                                  lsb_bits: int = 1,
                                  patch_size: int = 5,
                                  top_groups: int = 128):
    """
    For the given image, produce labels by testing groups sampled from top and bottom ranked pixels.
    - group_size: number of pixels used to embed the payload (must match capacity)
    - top_groups: number of groups to sample from top and bottom each (controls dataset size)
    """
    im = Image.open(image_path).convert("RGB")
    arr = np.array(im)
    h, w, _ = arr.shape
    # baseline ranking
    payload_bits = 256  # used to get ranking; not the test payload size
    ranked = select_pixels(arr, payload_bits=payload_bits, patch_size=patch_size, lsb_bits=lsb_bits, seed=0)
    # prepare groups: consecutive chunks of group_size
    def make_groups(coords_list):
        groups = []
        for i in range(0, len(coords_list), group_size):
            grp = coords_list[i:i+group_size]
            if len(grp) == group_size:
                groups.append(grp)
            if len(groups) >= top_groups:
                break
        return groups
    top_groups_list = make_groups(ranked[:top_groups*group_size])
    bottom_groups_list = make_groups(ranked[-(top_groups*group_size):])

    # create payload and ECC
    base_payload = (b"LAB" + b"\x00") * ((payload_len // 4) + 1)
    base_payload = base_payload[:payload_len]
    # ECC encode
    enc_payload = add_redundancy(base_payload, nsym=nsym)
    encoded_len = len(enc_payload)
    # we'll embed enc_payload across group_size pixels (check capacity)
    cap_bits = group_size * 3 * lsb_bits
    if encoded_len * 8 > cap_bits:
        raise ValueError(f"Group too small: encoded payload {encoded_len*8} bits but capacity {cap_bits} bits. Increase group_size or reduce nsym/payload_len.")

    # helper to test a group
    def test_group(coords):
        try:
            stego = embed_bytes_into_pixels(im, enc_payload, coords, lsb_bits=lsb_bits)
            # simulate recompression/noise
            recompr = recompress_and_reload(stego, quality=75)
            # extract encoded bytes back
            extracted = extract_bytes_from_pixels(recompr, num_payload_bytes=encoded_len, coords=coords, lsb_bits=lsb_bits)
            # try ECC recover
            try:
                rec = recover_redundancy(extracted, nsym=nsym)
            except Exception:
                return 0
            return 1 if rec == base_payload else 0
        except Exception:
            return 0

    half = patch_size // 2
    # collect patches + labels for top groups
    for grp in top_groups_list:
        label = test_group(grp)
        # central pixel = middle of group
        cx, cy = grp[len(grp)//2]
        patch = extract_patch(get_gray(arr), cx, cy, size=patch_size)
        out_patches.append(patch.astype(np.float32) / 255.0)
        out_labels.append(label)
    # bottom groups
    for grp in bottom_groups_list:
        label = test_group(grp)
        cx, cy = grp[len(grp)//2]
        patch = extract_patch(get_gray(arr), cx, cy, size=patch_size)
        out_patches.append(patch.astype(np.float32) / 255.0)
        out_labels.append(label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "module3" / "robust_labels.npz"))
    parser.add_argument("--limit", type=int, default=8, help="How many images to use")
    parser.add_argument("--payload", type=int, default=16, help="payload bytes before ECC")
    parser.add_argument("--nsym", type=int, default=32, help="RS parity bytes")
    parser.add_argument("--group_size", type=int, default=32, help="pixels per group")
    parser.add_argument("--top_groups", type=int, default=64, help="how many groups from top/bottom")
    parser.add_argument("--lsb_bits", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=5)
    args = parser.parse_args()

    dev_list = Path(__file__).resolve().parents[2] / "data" / "dev_images"
    imgs = list(dev_list.glob("*.*"))[:args.limit]
    if not imgs:
        print("No dev images found in data/dev_images/. Place images and try again.")
        return
    patches = []
    labels = []
    for img in imgs:
        print("Processing", img)
        build_label_dataset_for_image(img, patches, labels,
                                      payload_len=args.payload,
                                      group_size=args.group_size,
                                      nsym=args.nsym,
                                      lsb_bits=args.lsb_bits,
                                      patch_size=args.patch_size,
                                      top_groups=args.top_groups)
    patches = np.stack(patches)
    labels = np.array(labels).astype(np.int64)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, patches=patches, labels=labels)
    print(f"Saved robust label dataset to {outp} (patches={patches.shape}, labels={labels.shape})")

if __name__ == "__main__":
    main()
