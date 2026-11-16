"""Overlay video of the official-split DINOv2 + TCN noun model vs the official AR ground truth.

Header: model noun + confidence, GT action, per-frame status, running accuracy.
Footer: whole-video strip (GT row above, model row below) with a playhead.
With --sam: SAM 3.1 hand outlines plus the box of the in-hand instance of the model's noun.
GT is used only for display/scoring, never by the model.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from monitor.kit_catalog import HAND_LABEL, KIT_PART_LABELS, ar_object_for_kit
from monitor.settings import ROOT
from scripts.segmentation.evaluate_ar_object_segments import gt_noun_from_action, gt_object_frame_labels
from scripts.data.ar_labels import load_ar_intervals
from scripts.kit_sop.run_automatic_assembly import latest_at
from scripts.kit_sop.run_kit_sop_compare import _draw_hand, _draw_part, _past_boxes, choose_box

FPS = 10.0
HEADER, FOOTER = 112, 64
IDLE = (90, 90, 90)
GREEN, RED, AMBER, WHITE = (80, 200, 80), (60, 60, 230), (40, 170, 240), (235, 235, 235)


def noun_color(noun: str) -> tuple[int, int, int]:
    if not noun:
        return IDLE
    hue = int(hashlib.md5(noun.encode()).hexdigest()[:4], 16) / 0xFFFF
    r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _text(img, text, org, scale, color, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _strip(gt: list[str], pred: list[str], width: int) -> np.ndarray:
    bar = np.full((FOOTER, width, 3), 24, np.uint8)
    n = len(gt)
    for x in range(width):
        i = min(n - 1, int(x * n / width))
        bar[8:30, x] = noun_color(gt[i])
        bar[36:58, x] = noun_color(pred[i])
    _text(bar, "GT", (4, 25), 0.45, WHITE, 1)
    _text(bar, "MODEL", (4, 54), 0.45, WHITE, 1)
    return bar


class SamLayer:
    def __init__(self, source: Path):
        detection = json.loads((source / "detections.json").read_text(encoding="utf-8"))
        self.records = detection["records"]
        self.by_index = {int(r["index"]): r for r in self.records}
        self.masks = source / "masks"
        self.kit_from_ar = {ar_object_for_kit(k): k for k in KIT_PART_LABELS if ar_object_for_kit(k)}
        self.locked_id: str | None = None
        self.locked_kit = ""

    def draw(self, frame: np.ndarray, t: float, noun: str, conf: float | None, status: str,
             label: str | None = None) -> None:
        record = latest_at(self.records, t)
        if record is None:
            return
        try:
            with np.load(self.masks / f"{int(record['index']):05d}.npz") as archive:
                mask_map = {name: archive[name] for name in archive.files}
        except (OSError, ValueError):
            mask_map = {}
        objects = list(record.get("objects") or [])
        width = frame.shape[1]
        for row in objects:
            if str(row.get("class") or "") == HAND_LABEL:
                mask = mask_map.get(str(row.get("id")))
                _draw_hand(frame, row, None if mask is None else np.asarray(mask, bool), width)
        kit = self.kit_from_ar.get(noun, noun.replace("_", " ")) if noun else ""
        chosen, _ = choose_box(
            "contact", objects, mask_map, kit, frame.shape[:2],
            past_boxes=_past_boxes(self.by_index, int(record["index"])),
            locked_id=self.locked_id if self.locked_kit == kit else None,
        )
        if chosen is None:
            return
        self.locked_id, self.locked_kit = str(chosen.get("id")), kit
        chosen = dict(chosen, model_label=label or noun.replace("_", " "), model_conf=conf)
        mask = mask_map.get(str(chosen.get("id")))
        _draw_part(frame, chosen, None if mask is None else np.asarray(mask, bool), width,
                   match_status=status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--result", type=Path,
                        default=ROOT / "output/official_split_eval/result_tcn_v7_tcn_v8")
    parser.add_argument("--output", type=Path, default=ROOT / "output/overlays")
    parser.add_argument("--sam", type=Path, default=None,
                        help="SAM 3.1 detection folder (detections.json + masks/) to draw boxes from.")
    args = parser.parse_args()
    rid = args.recording
    sam = SamLayer(args.sam) if args.sam else None

    rows = list(csv.DictReader((args.result / rid / "ar_object_timeline.csv").open(encoding="utf-8")))
    pred = [r["ar_object"] for r in rows]
    conf = [float(r["score"]) if r["score"] else None for r in rows]
    intervals = load_ar_intervals(ROOT / "data/evaluation/official" / rid / "AR_labels.csv", source_fps=FPS)
    gt, _ = gt_object_frame_labels(intervals, n_frames=len(pred), fps=FPS, full_nouns=True)
    gt_action = [""] * len(pred)
    for row in intervals:
        noun = gt_noun_from_action(row["action"])
        for i in range(max(0, int(round(row["start_s"] * FPS))), min(len(pred), int(round(row["end_s"] * FPS)) + 1)):
            if not gt_action[i] and gt[i] == noun:
                gt_action[i] = str(row["action"])

    cap = cv2.VideoCapture(str(ROOT / "data/generated/industreal/videos" / f"{rid}.mp4"))
    width, height = int(cap.get(3)), int(cap.get(4))
    strip = _strip(gt, pred, width)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"{rid}_{'sam_' if sam else ''}model_vs_gt.mp4"
    proc = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height + HEADER + FOOTER}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target),
    ], stdin=subprocess.PIPE)

    ok_frames = act_frames = 0
    for i in range(len(pred)):
        ok, frame = cap.read()
        if not ok:
            break
        p, g = pred[i], gt[i]
        if g:
            act_frames += 1
            ok_frames += int(p == g)
        if g and p:
            status, tone = ("MATCH", GREEN) if p == g else ("MISS", RED)
        elif g:
            status, tone = "MISSED - model says idle", AMBER
        elif p:
            status, tone = "FALSE ALARM - GT idle", AMBER
        else:
            status, tone = "IDLE - correct", IDLE
        if sam:
            sam.draw(frame, i / FPS, p, conf[i], status if status in ("MATCH", "MISS") else "")

        head = np.full((HEADER, width, 3), 24, np.uint8)
        label = p.replace("_", " ") if p else "idle"
        if p and conf[i] is not None:
            label += f"  {conf[i]:.2f}" + ("  (unsure)" if conf[i] < 0.7 else "")
        cv2.rectangle(head, (12, 14), (34, 36), noun_color(p), -1)
        _text(head, f"MODEL: {label}", (44, 34), 0.8, WHITE)
        cv2.rectangle(head, (12, 52), (34, 74), noun_color(g), -1)
        _text(head, f"GT: {gt_action[i].replace('_', ' ') if g else 'idle'}", (44, 72), 0.7, WHITE)
        _text(head, status, (12, 102), 0.7, tone)
        acc = f"{rid}   {i / FPS:6.1f}s   running accuracy (action frames): {100 * ok_frames / max(1, act_frames):.1f}%"
        _text(head, acc, (max(12, width - 12 - 10 * len(acc)), 102), 0.5, WHITE, 1)

        foot = strip.copy()
        x = int(i * width / len(pred))
        cv2.line(foot, (x, 2), (x, FOOTER - 2), WHITE, 2)
        proc.stdin.write(np.vstack([head, frame, foot]).tobytes())
    cap.release()
    proc.stdin.close()
    proc.wait()
    print(target, flush=True)


if __name__ == "__main__":
    main()
