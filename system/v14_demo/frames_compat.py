"""When the IndustReal dataset folder has no JPEG frames (rgb/), read the frames from the recording's .mp4 instead.
psr.core.frame_paths then returns '<video>.mp4#<i>' entries (same count as the JPEG frames: checked on 3 recordings) and
cv2.imread reads such an entry from the video. With rgb/ present nothing changes. Display / demo only; the scored
results use the cached verifier and ASD outputs, which do not depend on this."""
import os
from pathlib import Path

import cv2
import psr.core as core

VIDEOS = Path(os.environ.get("INDUSTREAL_VIDEOS", Path(__file__).resolve().parents[1] / "assistant/data/generated/industreal/videos"))
_orig = core.frame_paths
_n = {}


def frame_paths(split, video):
    if (core.video_dir(split, video) / "rgb").is_dir():
        return _orig(split, video)
    if video not in _n:
        cap = cv2.VideoCapture(str(VIDEOS / f"{video}.mp4")); _n[video] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    return [f"{VIDEOS / video}.mp4#{i}" for i in range(_n[video])]


core.frame_paths = frame_paths
_imread, _cap = cv2.imread, {}


def imread(path, *a):
    p = str(path)
    if "#" not in p:
        return _imread(path, *a)
    f, i = p.rsplit("#", 1); i = int(i)
    c = _cap.get(f)
    if c is None or c[1] != i:                    # sequential reads are the fast path
        if c:
            c[0].release()
        cap = cv2.VideoCapture(f); cap.set(cv2.CAP_PROP_POS_FRAMES, i); c = [cap, i]; _cap[f] = c
    ok, img = c[0].read(); c[1] = i + 1
    return img if ok else None


cv2.imread = imread
