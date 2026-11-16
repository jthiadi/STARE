"""IndustReal readers: ground-truth actions and assembly states, released ASD predictions, MediaPipe hands.

Frame numbers are 10 fps video frames (the ``000123.jpg`` names in the label files).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.segmentation.run_official_split_eval import TEST, TRAIN, VAL

SPLITS = {"train": TRAIN, "val": VAL, "test": TEST}


def task_of(rid: str) -> str:
    """``assy`` (assembly) or ``main`` (maintenance), from the recording name."""
    return rid.split("_")[1]


def _frame(name: str) -> int:
    return int(Path(name).stem)


def gt_actions(rid: str, labels: Path, fps: float) -> list[dict]:
    """Annotated actions, sorted by start: ``{action, start, end}`` in seconds."""
    rows = []
    for r in csv.reader(open(labels / rid / "AR_labels.csv", encoding="utf-8")):
        rows.append({"action": r[2], "start": _frame(r[3]) / fps, "end": (_frame(r[4]) + 1) / fps})
    return sorted(rows, key=lambda r: (r["start"], r["end"]))


def psr_states(rid: str, labels: Path) -> list[tuple[int, list[int]]]:
    """``(frame, state)`` at every annotated state change; state has one value per component (1 / 0 / -1)."""
    with open(labels / rid / "PSR_labels_raw.csv", encoding="utf-8") as f:
        return [(_frame(r[0]), [int(v) for v in r[1:]]) for r in csv.reader(f)]


def n_frames(rid: str, labels: Path, videos: Path) -> int:
    """Video length in frames (from the video if present, else the last annotated frame)."""
    video = videos / f"{rid}.mp4"
    if video.is_file():
        import cv2
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    last = max(_frame(r[4]) for r in csv.reader(open(labels / rid / "AR_labels.csv", encoding="utf-8")))
    return max(last, psr_states(rid, labels)[-1][0]) + 1


def asd_release(rid: str, folder: Path, categories: list[str], min_conf: float = 0.0) -> list[dict]:
    """Released IndustReal ASD detections: ``{frame, state, conf, box}``, the best box per frame (>= min_conf).

    ``state`` is an 11-character string such as ``"11110111111"``, or ``"error_state"``.
    """
    best: dict[int, dict] = {}
    path = folder / f"{rid}_results_pred.csv"
    for r in csv.DictReader(open(path, encoding="utf-8")):
        frame, conf = int(r["framenr"]), float(r["bb_conf"])
        state = categories[int(r["bb_class"])]
        if state == "background" or conf < min_conf or (frame in best and best[frame]["conf"] >= conf):
            continue
        cx, cy, w, h = (float(r[k]) for k in ("bb_x", "bb_y", "bb_w", "bb_h"))   # YOLO: box centre + size
        best[frame] = {"frame": frame, "state": state, "conf": round(conf, 3),
                       "box": [round(cx - w / 2, 1), round(cy - h / 2, 1), round(w, 1), round(h, 1)]}
    return [best[f] for f in sorted(best)]


def hands_cache(rid: str, folder: Path) -> np.ndarray:
    """MediaPipe landmarks ``(T, 2, 21, 3)``: x, y normalised to the image, missing hands are all 0."""
    return np.load(folder / f"{rid}.npz")["landmarks"].astype(np.float32)
