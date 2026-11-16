"""Precompute progress per GPU shard: recordings done, current stage, frames left.   python -m live.progress"""
import json
import time
from pathlib import Path

import cv2

from assist.industreal import SPLITS

V, O = Path("data/generated/industreal/videos"), Path("output/assist")
for k in (0, 1):
    rids = SPLITS["test"][k::2]
    fr = {r: int(cv2.VideoCapture(str(V / f"{r}.mp4")).get(7)) for r in rids}
    done = [r for r in rids if (O / r / "engine_pred_vjepap_asdv14.json").exists()]
    cur = [r for r in rids if r not in done and (O / r / "live_status.json").exists()]
    s = json.loads((O / cur[0] / "live_status.json").read_text()) if cur else {}
    mins = round((time.time() - s.get("started", time.time())) / 60, 1)
    print(f"GPU{k}: {len(done)}/{len(rids)} done; now {cur[:1]} stage {s.get('stage')} (finished {s.get('done')}, "
          f"{mins} min in, {fr[cur[0]] if cur else 0} frames); frames left incl. current {sum(f for r, f in fr.items() if r not in done)}")
