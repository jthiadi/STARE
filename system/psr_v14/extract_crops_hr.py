"""High-resolution crops (default 336 px, JPEG) from the SAME runtime-localizer boxes as extract_crops.py
(cache/crops/<split>/<video>.npz holds t and box).  Inputs only; no labels are read.
-> cache/crops_hr/<split>/<video>.npz  (t, jpg (object array of bytes), q)"""
import sys
import time

import cv2
import numpy as np

from psr.core import frame_paths, list_videos
from psr.vision import CACHE, crop

S = int(sys.argv[2]) if len(sys.argv) > 2 else 336
for split in sys.argv[1].split(","):
    out = CACHE / "crops_hr" / split
    out.mkdir(parents=True, exist_ok=True)
    for v in list_videos(split):
        f = out / f"{v}.npz"
        if f.exists():
            continue
        t0 = time.time()
        z = np.load(CACHE / "crops" / split / f"{v}.npz")
        paths = frame_paths(split, v)
        jpgs = []
        for t, box in zip(z["t"], z["box"]):
            bgr = cv2.imread(str(paths[int(t)]))
            roi = cv2.resize(crop(bgr, box, pad=0.1), (S, S), interpolation=cv2.INTER_AREA)
            jpgs.append(cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
        arr = np.empty(len(jpgs), dtype=object)
        arr[:] = jpgs
        np.savez(f, t=z["t"], jpg=arr, q=z["q"])
        print(split, v, len(jpgs), round(time.time() - t0), flush=True)
