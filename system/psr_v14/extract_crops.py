"""Runtime localizer crops for every 2nd frame (inputs only; no labels read). -> cache/crops/<split>/<video>.npz"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np
from psr.core import frame_paths, list_videos
from psr.vision import CACHE, VisionVerifier, read_rgb, crop
S = 160
vv = VisionVerifier(bank="exemplar")
for split in sys.argv[1].split(","):
    out = CACHE / "crops" / split; out.mkdir(parents=True, exist_ok=True)
    for v in list_videos(split):
        f = out / f"{v}.npz"
        if f.exists(): continue
        t0 = time.time(); paths = frame_paths(split, v); vv.prev_box = None
        T, IM, Q, B = [], [], [], []
        for t in range(0, len(paths), 2):
            rgb = read_rgb(paths[t]); _, pt = vv.enc.encode(rgb, vv.FRAME_SIZE)
            box, _ = vv.localize(pt)
            if box is None: continue
            roi = crop(rgb, box, pad=0.1)
            q, _ = vv.view_quality(roi)
            T.append(t); IM.append(cv2.resize(roi, (S, S), interpolation=cv2.INTER_AREA)); Q.append(q); B.append(box)
        np.savez(f, t=np.array(T), img=np.stack(IM).astype(np.uint8), q=np.array(Q, np.float32), box=np.array(B, np.float32))
        print(split, v, len(T), round(time.time() - t0), flush=True)
