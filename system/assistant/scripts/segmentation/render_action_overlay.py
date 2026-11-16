"""Overlay video of frame-wise action predictions (verb_noun), optionally vs IndustReal GT and with SAM 3.1.

Header: predicted action + confidence (and GT, per-frame status, running accuracy when GT is given).
Footer: whole-video segment strips with a playhead.
With --sam: SAM 3.1 hand outlines plus the mask of the in-hand instance of the predicted noun.

    python -m scripts.segmentation.render_action_overlay --predictions output/predict/<video>/predictions.npz \
        --video <video>.mp4 [--recording <IndustReal id, adds GT>] [--sam output/kit_card_sam31_<id>]
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from scripts.segmentation.evaluate_ar_object_segments import gt_noun_from_action

FPS = 10.0
FOOTER = 76
BG_NAME = "background"
BG_COLOR = (70, 70, 70)
GREEN, RED, AMBER, WHITE, GREY = (80, 200, 80), (60, 60, 230), (40, 170, 240), (235, 235, 235), (150, 150, 150)


def color(name: str) -> tuple[int, int, int]:
    if name == BG_NAME:
        return BG_COLOR
    hue = int(hashlib.md5(name.encode()).hexdigest()[:4], 16) / 0xFFFF
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def text(img, s, org, scale, tone, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, tone, thick, cv2.LINE_AA)


def strip(rows: list[tuple[str, list[str]]], width: int) -> np.ndarray:
    bar = np.full((FOOTER, width, 3), 24, np.uint8)
    h = (FOOTER - 8) // len(rows)
    for r, (title, labels) in enumerate(rows):
        n = len(labels)
        y = 6 + r * h
        for x in range(width):
            bar[y:y + h - 6, x] = color(labels[min(n - 1, int(x * n / width))])
        text(bar, title, (4, y + h - 12), 0.45, WHITE, 1)
    return bar


def render(video: Path, pred: list[str], conf: np.ndarray, target: Path, name: str,
           gt: list[str] | None = None, sam_dir: Path | None = None) -> Path:
    sam = None
    if sam_dir:
        from scripts.kit_sop.render_model_overlay import SamLayer

        sam = SamLayer(sam_dir)
    header = 120 if gt else 82
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(3)), int(cap.get(4))
    rows = ([("GT", gt)] if gt else []) + [(name.split()[0], pred)]
    bar = strip(rows, width)
    target.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height + header + FOOTER}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target),
    ], stdin=subprocess.PIPE)

    correct = 0
    for i in range(len(pred)):
        ok, frame = cap.read()
        if not ok:
            break
        p = pred[i]
        g = gt[i] if gt else None
        status, tone = "", WHITE
        if gt:
            correct += p == g
            if p == g:
                status, tone = ("MATCH", GREEN) if g != BG_NAME else ("MATCH (no action)", GREY)
            elif g == BG_NAME:
                status, tone = "FALSE ALARM - GT has no action", AMBER
            elif p == BG_NAME:
                status, tone = f"MISSED - {name} says no action", AMBER
            else:
                status, tone = "MISS - wrong action", RED
        if sam:
            sam.draw(frame, i / FPS, "" if p == BG_NAME else gt_noun_from_action(p), float(conf[i]),
                     ("MATCH" if p == g else ("MISS" if g != BG_NAME else "")) if gt else "",
                     label=p.replace("_", " "))

        head = np.full((header, width, 3), 24, np.uint8)
        cv2.rectangle(head, (12, 14), (34, 36), color(p), -1)
        text(head, f"{name}: {p.replace('_', ' ')}  {conf[i]:.2f}", (44, 34), 0.8, WHITE)
        info = f"{video.stem}   {i / FPS:6.1f}s"
        if gt:
            cv2.rectangle(head, (12, 52), (34, 74), color(g), -1)
            text(head, f"GT:   {g.replace('_', ' ')}", (44, 72), 0.8, WHITE)
            text(head, status, (12, 108), 0.7, tone)
            info += f"   frame accuracy so far: {100 * correct / (i + 1):.1f}%"
        text(head, info, (max(12, width - 12 - 10 * len(info)), header - 12), 0.5, WHITE, 1)

        foot = bar.copy()
        x = int(i * width / len(pred))
        cv2.line(foot, (x, 2), (x, FOOTER - 2), WHITE, 2)
        proc.stdin.write(np.vstack([head, frame, foot]).tobytes())
    cap.release()
    proc.stdin.close()
    proc.wait()
    return target


def main() -> None:
    from monitor.settings import ROOT
    from scripts.data.tas_labels import frame_labels

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="predictions.npz from predict_actions.py, or test_predictions.npz from the evaluation.")
    parser.add_argument("--video", type=Path, default=None, help="Defaults to the IndustReal video of --recording.")
    parser.add_argument("--recording", default=None, help="IndustReal recording id: key into test_predictions.npz + GT.")
    parser.add_argument("--sam", type=Path, default=None, help="SAM 3.1 detection folder (detections.json + masks/).")
    parser.add_argument("--name", default="Verb+Noun TCN")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = np.load(args.predictions, allow_pickle=False)
    actions = [str(a) for a in data["actions"]]
    prefix = f"{args.recording}/" if args.recording and f"{args.recording}/pred" in data.files else ""
    pred_idx, conf = data[f"{prefix}pred"], data[f"{prefix}conf"]
    pred = [actions[int(i)] for i in pred_idx]
    gt = None
    if args.recording:
        gt = [actions[int(i)] for i in frame_labels(args.recording, len(pred), actions)]
    video = args.video or ROOT / "data/generated/industreal/videos" / f"{args.recording}.mp4"
    tag = "_sam" if args.sam else ""
    target = args.output or args.predictions.parent / f"{video.stem}_actions{tag}.mp4"
    print(render(video, pred, conf, target, args.name, gt, args.sam), flush=True)


if __name__ == "__main__":
    main()
