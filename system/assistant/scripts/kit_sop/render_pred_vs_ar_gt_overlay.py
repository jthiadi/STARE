"""Compare kit AR-object predictions to official AR ground truth on one video.

EVAL visualization only. Reads held-out ``AR_labels.csv`` for display/scoring;
does not feed GT into detection. Shows GT action/object vs primary prediction
and colors the pred box green (match) / red (mismatch) / grey (no GT object).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from monitor.kit_catalog import HAND_LABEL, KIT_PART_LABELS, kit_class_color
from monitor.overlay_hud import fit_text
from monitor.settings import ROOT
from scripts.data.ar_labels import active_ar, gt_object_from_action, load_ar_intervals
from scripts.kit_sop.run_automatic_assembly import latest_at, read_frame
from scripts.segmentation.evaluate_ar_object_segments import load_pred_timeline


def primary_gt(active: list[dict]) -> tuple[str, str | None]:
    """Pick one GT action for the frame; prefer actions with a kit object noun."""
    if not active:
        return "", None
    ranked = sorted(
        active,
        key=lambda row: (
            0 if gt_object_from_action(row["action"]) else 1,
            row["start_s"],
            row["action"],
        ),
    )
    action = str(ranked[0]["action"])
    return action, gt_object_from_action(action)


def _draw_hand(frame: np.ndarray, obj: dict, mask: np.ndarray | None, width: int) -> None:
    tone = kit_class_color(HAND_LABEL)
    box = obj.get("box")
    if mask is not None and mask.shape[:2] == frame.shape[:2] and mask.any():
        contours = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        cv2.drawContours(frame, contours, -1, tone, 1)
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if len(xs):
            box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return
    x1, y1, x2, y2 = map(int, box)
    if (x2 - x1) * (y2 - y1) > 0.25 * width * frame.shape[0]:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), tone, 1)


def _draw_pred(
    frame: np.ndarray,
    obj: dict,
    mask: np.ndarray | None,
    width: int,
    tint: tuple[int, int, int],
    label: str,
) -> None:
    if mask is not None and mask.shape[:2] == frame.shape[:2] and mask.any():
        frame[mask] = (frame[mask] * 0.62 + np.array(tint) * 0.38).astype(np.uint8)
        contours = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        cv2.drawContours(frame, contours, -1, tint, 2)
    box = obj.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), tint, 2)
    y = max(20, y1 - 6)
    text_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
    cv2.rectangle(
        frame, (x1, max(0, y - 20)), (min(width, x1 + text_w + 8), y + 4),
        (16, 18, 22), -1,
    )
    cv2.putText(
        frame, label, (x1 + 3, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, tint, 2, cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "output/kit_card_sam31_13_assy_0_1",
        help="Kit detect folder (detections.json + masks/).",
    )
    parser.add_argument(
        "--timeline",
        type=Path,
        default=None,
        help="Pred AR-object CSV (default: <source>/overlay/ar_object_timeline.csv).",
    )
    parser.add_argument(
        "--ar-labels",
        type=Path,
        default=ROOT / "data/evaluation/official/13_assy_0_1/AR_labels.csv",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "data/generated/industreal/videos/13_assy_0_1.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/kit_card_sam31_13_assy_0_1/overlay_vs_gt",
    )
    parser.add_argument("--playback-fps", type=float, default=10.0)
    parser.add_argument("--source-fps", type=float, default=10.0)
    args = parser.parse_args()

    source = args.source.resolve()
    timeline_path = (args.timeline or (source / "overlay" / "ar_object_timeline.csv")).resolve()
    output = args.output.resolve()
    if ROOT.resolve() not in output.parents and output != ROOT.resolve():
        raise SystemExit("Output must stay inside this project.")
    output.mkdir(parents=True, exist_ok=True)

    detection = json.loads((source / "detections.json").read_text(encoding="utf-8"))
    if not detection.get("records"):
        raise SystemExit("Missing detections.")
    timeline = load_pred_timeline(timeline_path)
    if not timeline:
        raise SystemExit(f"Empty timeline: {timeline_path}")
    by_time = {round(row["timestamp_s"], 3): row for row in timeline}
    intervals = load_ar_intervals(args.ar_labels, source_fps=float(args.source_fps))

    meta = {
        "title": "Kit AR-object prediction vs official AR ground truth",
        "source_detections": str(source / "detections.json"),
        "timeline": str(timeline_path),
        "ar_labels": str(args.ar_labels),
        "video": str(args.video),
        "note": (
            "GT is held-out eval only. Object-level compare: pred ar_object vs "
            "noun extracted from AR action (take_nut→nut). Instruction/align/"
            "objects frames are unscored."
        ),
    }
    (output / "overlay_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cap = cv2.VideoCapture(str(args.video))
    width, height = int(cap.get(3)), int(cap.get(4))
    footer = 150
    playback_fps = float(args.playback_fps)
    start_s = float(detection["start_s"])
    end_s = float(detection["end_s"])
    destination = output / "pred_vs_ar_gt_overlay.mp4"
    temp_destination = output / "pred_vs_ar_gt_overlay.rendering.mp4"
    if temp_destination.exists():
        temp_destination.unlink()
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height + footer}", "-r", str(playback_fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp_destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    started = time.perf_counter()
    frame_count = 0
    scored = match = miss = 0
    unscored = 0
    compare_rows: list[dict] = []
    mask_dir = source / "masks"

    try:
        total = max(1, int(math.floor((end_s - start_s) * playback_fps)))
        for frame_index in range(total):
            timestamp = start_s + frame_index / playback_fps
            if timestamp >= end_s - 1e-9:
                timestamp = max(start_s, end_s - 1e-3)
            record = latest_at(detection["records"], timestamp)
            if record is None:
                continue
            pred = by_time.get(round(timestamp, 3))
            if pred is None and timeline:
                pred = min(timeline, key=lambda row: abs(row["timestamp_s"] - timestamp))

            frame = read_frame(cap, timestamp)
            objects = list(record.get("objects") or [])
            archive = None
            mask_map: dict[str, np.ndarray] = {}
            try:
                archive = np.load(mask_dir / f"{int(record['index']):05d}.npz")
                mask_map = {name: archive[name] for name in archive.files}
            except (OSError, ValueError):
                archive = None

            gt_actions = active_ar(intervals, timestamp)
            gt_action, gt_object = primary_gt(gt_actions)
            pred_kit = str((pred or {}).get("kit_label") or "")
            pred_ar = str((pred or {}).get("ar_object") or "")

            if gt_object:
                scored += 1
                hit = bool(pred_ar) and pred_ar == gt_object
                if hit:
                    match += 1
                    status = "MATCH"
                    tint = (60, 220, 90)
                else:
                    miss += 1
                    status = "MISS"
                    tint = (60, 60, 255)
            else:
                unscored += 1
                status = "n/a"
                tint = (180, 180, 180)

            try:
                for row in objects:
                    if str(row.get("class") or "") != HAND_LABEL:
                        continue
                    key = str(row.get("id"))
                    mask = mask_map.get(key)
                    if mask is not None:
                        mask = np.asarray(mask, dtype=bool)
                    _draw_hand(frame, row, mask, width)

                pred_obj = None
                if pred_kit:
                    for row in objects:
                        if str(row.get("class") or "") == pred_kit:
                            pred_obj = row
                            break
                if pred_obj is not None:
                    key = str(pred_obj.get("id"))
                    mask = mask_map.get(key)
                    if mask is not None:
                        mask = np.asarray(mask, dtype=bool)
                    _draw_pred(
                        frame, pred_obj, mask, width, tint,
                        label=pred_kit or pred_ar or "pred",
                    )
            finally:
                if archive is not None:
                    archive.close()

            # Top strip: GT vs PRED
            cv2.rectangle(frame, (0, 0), (width, 56), (18, 20, 24), -1)
            gt_line = (
                f"GT: {gt_action or '(none)'}"
                + (f"  ->  {gt_object}" if gt_object else "  ->  (no object)")
            )
            pred_line = (
                f"PRED: {pred_kit or '(none)'}"
                + (f"  ({pred_ar})" if pred_ar else "")
                + f"   [{status}]"
            )
            cv2.putText(
                frame, fit_text(gt_line, 95), (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 210, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, fit_text(pred_line, 95), (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, tint, 2, cv2.LINE_AA,
            )

            acc = (match / scored) if scored else None
            canvas = np.zeros((height + footer, width, 3), np.uint8)
            canvas[:height] = frame
            lines = [
                fit_text(
                    f"t={timestamp:.1f}s   AR object compare   "
                    f"scored={scored} match={match} miss={miss}"
                    + (f"  acc={acc:.3f}" if acc is not None else ""),
                    110,
                ),
                fit_text(
                    "GT action->object vs kit top-1. Green=match, red=miss, "
                    "grey=no GT object (instruction/align/objects).",
                    110,
                ),
                fit_text(
                    "Kit classes: " + ", ".join(KIT_PART_LABELS[:8]) + ", …",
                    110,
                ),
                fit_text(
                    f"Active GT: {', '.join(row['action'] for row in gt_actions) or '(none)'}",
                    110,
                ),
            ]
            tones = [(210, 220, 230), (160, 175, 190), (160, 175, 190), (120, 210, 255)]
            for index, line in enumerate(lines):
                cv2.putText(
                    canvas, line, (12, height + 28 + index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, tones[index], 1, cv2.LINE_AA,
                )
            process.stdin.write(canvas.tobytes())
            frame_count += 1
            compare_rows.append({
                "frame_index": frame_index,
                "timestamp_s": round(timestamp, 3),
                "gt_action": gt_action,
                "gt_object": gt_object or "",
                "pred_kit": pred_kit,
                "pred_ar": pred_ar,
                "status": status,
            })
            if frame_index % max(1, int(playback_fps * 30)) == 0:
                cv2.imwrite(str(output / f"preview_{int(timestamp):04d}.jpg"), canvas)
                print(json.dumps({
                    "frame": frame_index, "of": total, "time": round(timestamp, 1),
                    "gt": gt_action or None, "gt_object": gt_object,
                    "pred": pred_ar or None, "status": status,
                    "acc": round(acc, 3) if acc is not None else None,
                }), flush=True)
    finally:
        cap.release()
        if process.stdin:
            process.stdin.close()
    if process.wait() != 0:
        if temp_destination.exists():
            temp_destination.unlink()
        raise RuntimeError("Video encoder failed")
    if destination.exists():
        destination.unlink()
    temp_destination.replace(destination)

    compare_csv = output / "pred_vs_ar_gt.csv"
    with compare_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index", "timestamp_s", "gt_action", "gt_object",
                "pred_kit", "pred_ar", "status",
            ],
        )
        writer.writeheader()
        writer.writerows(compare_rows)

    accuracy = (match / scored) if scored else None
    result = {
        "video": str(destination),
        "compare_csv": str(compare_csv),
        "frames": frame_count,
        "scored_frames": scored,
        "unscored_frames": unscored,
        "matches": match,
        "misses": miss,
        "object_accuracy": round(accuracy, 4) if accuracy is not None else None,
        "render_seconds": round(time.perf_counter() - started, 1),
        "mode": "pred_vs_ar_gt_object",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
