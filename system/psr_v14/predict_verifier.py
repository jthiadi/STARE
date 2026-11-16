"""Run the FINAL verifier (trained on train+val) on a split's localizer crops -> cache/verif_<split>/<video>.npz.
Inputs only: crops come from extract_crops.py (same localizer, every 2nd frame); no labels are read."""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

from psr.core import list_videos
from psr.vision import CACHE
from train_verifier import RES, TAG, Verifier, predict

split, weights = sys.argv[1], sys.argv[2]
m = Verifier().cuda()
m.load_state_dict(torch.load(weights, map_location="cuda"))
m.eval()
out = CACHE / f"{TAG}_{split}"
out.mkdir(exist_ok=True)
for v in list_videos(split):
    if RES:                                   # same high-res JPEG crops as in training
        z = dict(np.load(CACHE / "crops_hr" / split / f"{v}.npz", allow_pickle=True)); imgs = z["jpg"]
    else:
        z = np.load(CACHE / "crops" / split / f"{v}.npz"); imgs = z["img"]
    p = predict(m, imgs)
    np.savez(out / f"{v}.npz", t=z["t"], p=p.astype(np.float16), q=z["q"])
    print(v, len(p), flush=True)
