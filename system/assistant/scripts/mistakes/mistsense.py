"""MistSense-style mistake head on top of our AR windows (no VLM, no Q-Former clone).

We keep action recognition. For each GT action window (placeholder for our AR onset):
  MediaPipe hands (21 landmarks x 2) + DINOv2 appearance + SOP procedure flag
  -> logistic P(mistake), threshold chosen on val.

    python -m scripts.mistakes.mistsense --extract
    python -m scripts.mistakes.mistsense --train
    python -m scripts.mistakes.mistsense --overlay --video 03_assy_1_3
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from monitor.settings import ROOT
from scripts.mistakes.order_check import has_mistake, learn
from scripts.mistakes.detect import _fit_logistic, _metrics, _pick_threshold, _predict_logistic
from scripts.mistakes.warn import (
    COMPONENTS, OUT, action_events, infer_targets, procedure_warn, _dino_mean,
)
from scripts.segmentation.run_official_split_eval import TEST, TRAIN, VAL

HANDS = OUT / "hands"
SCORES = OUT / "mistsense.json"
DETECT_WIDTH = 640
EARLY = 12  # frames from action start used for the execution head


def _video(vid: str) -> Path:
    found = sorted((ROOT / "data/generated/industreal/videos").glob(f"{vid}.*"))
    assert found, vid
    return found[0]


def extract_video(vid: str, video: Path | None = None, out: Path = HANDS) -> Path:
    """Save (T, 2, 21, 3) float16 normalized landmarks; missing hands are 0."""
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{vid}.npz"
    if dest.is_file():
        return dest
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video or _video(vid)))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    out = np.zeros((n, 2, 21, 3), np.float16)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=0,
        min_detection_confidence=0.4, min_tracking_confidence=0.4)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (DETECT_WIDTH, max(1, int(h * DETECT_WIDTH / w))))
        res = hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            continue
        for k, lm in enumerate(res.multi_hand_landmarks[:2]):
            out[i, k] = [(p.x, p.y, p.z) for p in lm.landmark]
    hands.close()
    cap.release()
    np.savez_compressed(dest, landmarks=out)
    print(json.dumps({"extracted": vid, "frames": n}), flush=True)
    return dest


def extract_all(vids: list[str]) -> None:
    for vid in vids:
        extract_video(vid)


def _hand_vec(path: Path, start: int, end: int) -> np.ndarray:
    x = np.load(path)["landmarks"].astype(np.float32)
    a = int(np.clip(start, 0, len(x) - 1))
    b = int(np.clip(max(end, start + 1), a + 1, len(x)))
    win = x[a:b]
    present = (np.abs(win).sum(axis=(2, 3)) > 0)
    mean = np.zeros((2, 21, 3), np.float32)
    for h in range(2):
        rows = win[:, h][present[:, h]]
        if len(rows):
            mean[h] = rows.mean(0)
    wrist = win[:, :, 0, :2]
    speed = np.linalg.norm(np.diff(wrist, axis=0), axis=-1)
    speed = speed[present[1:]] if len(speed) else np.zeros((0, 2))
    stats = np.array([
        present.mean(), present[:, 0].mean(), present[:, 1].mean(),
        float(speed.mean()) if speed.size else 0.0,
        float(speed.max()) if speed.size else 0.0,
    ], np.float32)
    return np.concatenate([mean.reshape(-1), stats])


def _event_features(events: list[dict], rules, dino_cache: dict) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    n_comp = len(COMPONENTS)
    for e in events:
        path = HANDS / f"{e['vid']}.npz"
        if not path.is_file():
            continue
        hand = _hand_vec(path, e["start_f"], e["start_f"] + EARLY)
        vis = _dino_mean(e["vid"], e["start_f"], e["end_f"], dino_cache)
        if vis is None:
            continue
        p_warn, _ = procedure_warn(e["action"], e["state"], rules)
        one = np.zeros(n_comp, np.float32)
        for c in infer_targets(e["action"], e["state"]):
            one[c] = 1.0
        xs.append(np.concatenate([hand, vis, one, [1.0 if p_warn else 0.0]]))
        ys.append(1.0 if e["mistake"] else 0.0)
        if e["vid"] in dino_cache and len(dino_cache) > 3:
            dino_cache.clear()
    return np.stack(xs), np.asarray(ys, np.float32)


def train() -> dict:
    rules = learn([v for v in TRAIN if not has_mistake(v)])
    cache: dict = {}
    x_tr, y_tr = _event_features(action_events(TRAIN), rules, cache)
    cache.clear()
    x_va, y_va = _event_features(action_events(VAL), rules, cache)
    cache.clear()
    x_te, y_te = _event_features(action_events(TEST), rules, cache)
    pack = _fit_logistic(x_tr, y_tr, seed=0)
    s_va = _predict_logistic(pack, x_va)
    s_te = _predict_logistic(pack, x_te)
    thr = _pick_threshold(y_va, s_va)
    np.savez(OUT / "mistsense_head.npz", pack=pack, threshold=np.array([thr]))
    report = {
        "inputs": "MediaPipe hands + DINOv2 window + SOP procedure flag, on GT AR action windows",
        "keeps_our_ar": True,
        "val_threshold": thr,
        "val": _metrics(y_va, s_va, thr),
        "test": _metrics(y_te, s_te, thr),
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te)),
    }
    (OUT / "mistsense.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--video", default="03_assy_1_3")
    parser.add_argument("--limit", type=int, default=0, help="Extract only the first N videos (debug).")
    args = parser.parse_args()
    vids = TRAIN + VAL + TEST
    if args.limit:
        vids = vids[: args.limit]
    if args.extract:
        extract_all(vids)
    if args.train:
        train()
    if args.overlay:
        from scripts.mistakes.render_hands_overlay import render
        out = OUT / "overlays" / f"{args.video}_mistsense.mp4"
        print(json.dumps(render(args.video, out)), flush=True)
    if not (args.extract or args.train or args.overlay):
        parser.error("pass --extract and/or --train and/or --overlay")


if __name__ == "__main__":
    main()
