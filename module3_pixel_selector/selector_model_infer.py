from pathlib import Path
import torch
from .selector_model import TinyPatchNet, infer_scores
from .selector_baseline import select_pixels as baseline_select

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "module3_pixel_selector" / "best.pth"

def model_select_pixels(image_np, payload_bits, patch_size=5, lsb_bits=1, device='cpu'):
    """
    Try to use trained model. If model file missing, fall back to baseline.
    Returns List[(x,y)]
    """
    if not MODEL_PATH.exists():
        return baseline_select(image_np, payload_bits, patch_size=3, lsb_bits=lsb_bits, seed=0)

    device = torch.device(device)
    model = TinyPatchNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    scores = infer_scores(model, image_np, patch_size=patch_size, device=device.type)
    capacity_per_pixel = 3 * lsb_bits
    pixels_needed = int((payload_bits + capacity_per_pixel - 1) // capacity_per_pixel)
    selected = [(x,y) for (_,x,y) in scores[:pixels_needed]]
    return selected
