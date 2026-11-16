"""MediaPipe hands -> per-frame 21 keypoints per hand (``output/assist/<rid>/hands.jsonl``).

Rows: ``{f, t, hands: [[[x, y] * 21], ...]}`` with x, y normalised to the image (0..1); frames without
hands have an empty list. By default the landmarks already extracted for all 84 videos
(``output/mistakes/hands``) are converted; ``--compute`` runs MediaPipe on the video (CPU, frame by frame,
so it never holds more than one frame in memory).

    python -m assist.extract.run_hands --video all
    python -m assist.extract.run_hands --video 18_assy_2_5 --compute --force
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from assist.common import load_config, write_jsonl
from assist.industreal import SPLITS, hands_cache

DETECT_WIDTH = 640


def compute(video, n_hands: int = 2) -> np.ndarray:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    out = np.zeros((n, n_hands, 21, 3), np.float32)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=n_hands, model_complexity=0,
                                     min_detection_confidence=0.4, min_tracking_confidence=0.4)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (DETECT_WIDTH, max(1, int(h * DETECT_WIDTH / w))))
        res = hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        for k, lm in enumerate((res.multi_hand_landmarks or [])[:n_hands]):
            out[i, k] = [(p.x, p.y, p.z) for p in lm.landmark]
    hands.close()
    cap.release()
    return out


def rows(landmarks: np.ndarray, fps: float) -> list[dict]:
    out = []
    for i, frame in enumerate(landmarks):
        present = [h[:, :2] for h in frame if np.abs(h).sum() > 0]
        out.append({"f": i, "t": round(i / fps, 2), "hands": [np.round(h, 4).tolist() for h in present]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="all", help='Recording id, or "all" (the test split).')
    parser.add_argument("--split", default="test", choices=list(SPLITS))
    parser.add_argument("--compute", action="store_true", help="Run MediaPipe instead of reading the cache.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    rids = SPLITS[args.split] if args.video == "all" else [args.video]
    for rid in rids:
        dest = cfg["paths"]["out"] / rid / "hands.jsonl"
        if dest.is_file() and not args.force:
            continue
        lm = compute(cfg["paths"]["videos"] / f"{rid}.mp4") if args.compute \
            else hands_cache(rid, cfg["paths"]["hands_cache"])
        write_jsonl(dest, rows(lm, cfg["fps"]))
        print(json.dumps({"video": rid, "frames": len(lm),
                          "frames_with_hands": int((np.abs(lm).sum((1, 2, 3)) > 0).sum())}), flush=True)


if __name__ == "__main__":
    main()
