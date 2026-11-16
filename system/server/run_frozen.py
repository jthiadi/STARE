"""Run a script of the frozen v14 package (PSR_v14/) on this server, without changing it.
The server has the IndustReal videos but not the JPEG frames, so psr.core.frame_paths is replaced by one that
returns '<video>.mp4#<i>' entries (same count as the JPEG frames: checked on 3 recordings, identical counts) and
cv2.imread is taught to read such an entry from the video. Everything else is the frozen code.
    python run_frozen.py <script relative to PSR_v14> [args...]"""
import os, runpy, sys
from pathlib import Path
import cv2

SYSTEM = Path(__file__).resolve().parents[1]          # system/
PSR = Path(os.environ.get("PSR_V14", SYSTEM / "psr_v14"))
VIDEOS = Path(os.environ.get("INDUSTREAL_VIDEOS", SYSTEM / "assistant/data/generated/industreal/videos"))
os.environ.setdefault("INDUSTREAL_DATA", str(SYSTEM / "dataset"))
os.environ.setdefault("TEAMMATE_AR", str(SYSTEM / "assistant"))
sys.path.insert(0, str(PSR))
import psr.core as core

_n = {}
def frame_paths(split, video):
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
    if c is None or c[1] != i:                     # sequential reads are the fast path
        if c: c[0].release()
        cap = cv2.VideoCapture(f); cap.set(cv2.CAP_PROP_POS_FRAMES, i); c = [cap, i]; _cap[f] = c
    ok, img = c[0].read(); c[1] = i + 1
    return img if ok else None
cv2.imread = imread

script = PSR / sys.argv[1]
sys.argv = [str(script)] + sys.argv[2:]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
