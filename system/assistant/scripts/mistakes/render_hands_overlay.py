"""Overlay one IndustReal video with MediaPipe hands + SOP procedure warnings (no VLM).

Uses GT action onsets and GT PSR only to decide WARN vs OK (same rules as ``warn.py``).
Hands are drawn every frame. Header shows current action and whether the procedure branch
would warn at this action's start.

    python -m scripts.mistakes.render_hands_overlay --video 03_assy_1_3
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from monitor.settings import ROOT
from scripts.mistakes.order_check import has_mistake, learn, names, steps
from scripts.mistakes.warn import action_events, procedure_warn
from scripts.segmentation.render_action_overlay import GREEN, GREY, RED, WHITE, text
from scripts.segmentation.run_official_split_eval import TRAIN

VIDEOS = ROOT / "data/generated/industreal/videos"
OUT = ROOT / "output/mistakes/overlays"
FPS = 10.0
HEADER = 128
DETECT_WIDTH = 640


def _hands():
    import mediapipe as mp

    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ), mp.solutions.drawing_utils, mp.solutions.hands.HAND_CONNECTIONS


def _video(vid: str) -> Path:
    found = sorted(VIDEOS.glob(f"{vid}.*"))
    assert found, f"no video {vid} under {VIDEOS}"
    return found[0]


def render(vid: str, target: Path) -> dict:
    events = action_events([vid])
    rules = learn([v for v in TRAIN if not has_mistake(v)])
    for e in events:
        warn, why = procedure_warn(e["action"], e["state"], rules)
        e["procedure_warn"], e["procedure_why"] = warn, why
        e["ms_score"] = None
        e["ms_warn"] = False
    head_path = ROOT / "output/mistakes/mistsense_head.npz"
    if head_path.is_file() and (ROOT / "output/mistakes/hands" / f"{vid}.npz").is_file():
        from scripts.mistakes.detect import _predict_logistic
        from scripts.mistakes.mistsense import _event_features
        pack_f = np.load(head_path)
        pack, thr = pack_f["pack"], float(pack_f["threshold"][0])
        x, _ = _event_features(events, rules, {})
        if len(x):
            scores = _predict_logistic(pack, x)
            for e, s in zip(events, scores):
                e["ms_score"], e["ms_warn"] = float(s), bool(s >= thr)
    mistakes = [s for s in steps(vid) if s["mistake"]]
    label = names(vid)

    video = _video(vid)
    cap = cv2.VideoCapture(str(video))
    width, height, n = int(cap.get(3)), int(cap.get(4)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hands, draw, conn = _hands()
    target.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height + HEADER}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(target),
    ], stdin=subprocess.PIPE)

    seen_hands = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (DETECT_WIDTH, int(height * DETECT_WIDTH / width)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            seen_hands += 1
            for lm in res.multi_hand_landmarks:
                draw.draw_landmarks(
                    frame, lm, conn,
                    draw.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=2),
                    draw.DrawingSpec(color=(0, 180, 255), thickness=2),
                )

        current = next((e for e in events if e["start_f"] <= i <= e["end_f"]), None)
        head = np.full((HEADER, width, 3), 24, np.uint8)
        if current is None:
            text(head, f"{vid}  {i / FPS:6.1f}s   no action", (12, 36), 0.7, GREY)
            text(head, "procedure: -     hands+RGB: -", (12, 72), 0.55, GREY)
            text(head, "GT: -", (12, 108), 0.55, GREY)
        else:
            act = current["action"].replace("_", " ")
            proc_s = "WARN" if current["procedure_warn"] else "OK"
            gt_bad = bool(current["mistake"])
            if current["ms_score"] is None:
                ms_s, q_tag, vis_w = "-", "", None
            else:
                vis_w = bool(current["ms_warn"])
                ms_s = f"{'WARN' if vis_w else 'OK'} {current['ms_score']:.2f}"
                q_tag = f"  ({'HIT' if vis_w == gt_bad else 'MISS'} vs GT)"
            text(head, f"{vid}  {i / FPS:6.1f}s   AR action: {act}", (12, 36), 0.6, WHITE)
            text(head, f"procedure: {proc_s}     hands+RGB: {ms_s}{q_tag}",
                 (12, 72), 0.5, RED if vis_w else GREEN)
            if gt_bad:
                why = current.get("mistake_name") or current.get("procedure_why") or ""
                text(head, f"GT: MISTAKE  {why}"[:90], (12, 108), 0.5, RED)
            else:
                text(head, "GT: OK  (PSR install was correct)", (12, 108), 0.5, GREEN)

        proc.stdin.write(np.vstack([head, frame]).tobytes())
        if i % 200 == 0:
            print(json.dumps({"frame": i, "of": n, "hands_frames": seen_hands}), flush=True)

    hands.close()
    cap.release()
    proc.stdin.close()
    proc.wait()
    return {"video": str(target), "frames": n, "frames_with_hands": seen_hands,
            "actions": len(events), "procedure_warns": sum(e["procedure_warn"] for e in events),
            "gt_mistakes": len(mistakes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="03_assy_1_3")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or OUT / f"{args.video}_hands_procedure.mp4"
    print(json.dumps(render(args.video, out)), flush=True)


if __name__ == "__main__":
    main()
