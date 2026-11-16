"""Dump the lower-centre (hands) crop of every frame as JPEG for backbone fine-tuning."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2

from monitor.settings import ROOT

HANDS_BOX = (0.2, 0.3, 0.8, 1.0)
CROP_SIZE = (405, 270)  # (w, h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/ar_hand_crops")
    args = parser.parse_args()
    try:
        import psutil

        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    cv2.setNumThreads(3)
    recordings = sorted(
        p.name for p in (ROOT / "data/evaluation/official").iterdir()
        if (p / "AR_labels.csv").is_file()
    )
    for rid in recordings:
        target = args.output / rid
        done = target / "done.txt"
        if done.is_file():
            continue
        target.mkdir(parents=True, exist_ok=True)
        start = time.time()
        cap = cv2.VideoCapture(str(ROOT / "data/generated/industreal/videos" / f"{rid}.mp4"))
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (int(HANDS_BOX[0] * w), int(HANDS_BOX[1] * h),
                              int(HANDS_BOX[2] * w), int(HANDS_BOX[3] * h))
            crop = cv2.resize(frame[y1:y2, x1:x2], CROP_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(target / f"{index:05d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            index += 1
        cap.release()
        done.write_text(str(index), encoding="utf-8")
        print(json.dumps({"recording": rid, "frames": index,
                          "seconds": round(time.time() - start, 1)}), flush=True)


if __name__ == "__main__":
    main()
