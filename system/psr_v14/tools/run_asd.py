"""Run the official IndustReal ASD model on every frame of a split (causal: one frame at a time).
Writes cache/asd_<split>/<video>.npz: rows (frame, class, conf, x, y, w, h) of the top-2 boxes with conf >= 0.05.

    python tools/run_asd.py <split> --weights <asd_best_IndustRealandSynthetic.pt>

Weights: assembly_state_detection_model_weights.zip from the IndustReal dataset page (data.4tu.nl, Apache-2.0).
Needs `pip install ultralytics==8.3.40` (installed with --no-deps here; torch / numpy unchanged).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from psr.core import frame_paths, list_videos
from psr.vision import CACHE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", choices=["train", "val", "test"])
    ap.add_argument("--weights", default=str(HERE / "weights" / "asd_best_IndustRealandSynthetic.pt"))
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.weights)
    out = CACHE / f"asd_{a.split}"; out.mkdir(parents=True, exist_ok=True)
    for v in list_videos(a.split):
        if (out / f"{v}.npz").exists():
            continue
        P = [str(p) for p in frame_paths(a.split, v)]; rows = []
        for j in range(0, len(P), 32):
            for k, res in enumerate(m.predict(P[j:j + 32], conf=0.05, imgsz=640, verbose=False)):
                b = sorted(zip(res.boxes.conf.tolist(), res.boxes.cls.tolist(), res.boxes.xywh.tolist()), reverse=True)[:2]
                rows += [[j + k, int(cl), c, *xywh] for c, cl, xywh in b]
        np.savez_compressed(out / f"{v}.npz", rows=np.array(rows, np.float32).reshape(-1, 7), n=len(P))
        print(v, len(P), flush=True)


if __name__ == "__main__":
    main()
