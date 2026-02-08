import numpy as np
from stegotool.modules.module3_pixel_selector.selector_baseline import select_pixels
from stegotool.modules.module3_pixel_selector.selector_utils import get_gray

def gen_dummy_rgb(w=64, h=48):
    # create a gradient + noise image
    x = np.linspace(0,255,w, dtype=np.uint8)
    y = np.linspace(0,255,h, dtype=np.uint8)[:,None]
    grad = (x + y) // 2
    rgb = np.stack([grad, grad, grad], axis=2)
    # add noise
    noise = (np.random.RandomState(0).randint(0,30,(h,w))).astype(np.uint8)
    rgb[:,:,0] = np.clip(rgb[:,:,0] + noise, 0, 255)
    return rgb

def test_select_pixels_basic():
    img = gen_dummy_rgb(32,24)
    # want to embed 100 bits, 1 lsb -> capacity per pixel = 3*1 =3
    coords = select_pixels(img, payload_bits=100, patch_size=3, lsb_bits=1, seed=0)
    assert isinstance(coords, list)
    assert len(coords) >= 1
    # coords within bounds
    h,w,_ = img.shape
    x,y = coords[0]
    assert 0 <= x < w and 0 <= y < h
