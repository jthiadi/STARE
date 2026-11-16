"""Clear overlay: one AR-style primary kit object per frame + hand.

Reuses an existing SAM 3.1 detections folder. Does not re-run the detector or
read AR/PSR ground truth as model input. Labels on the video are the official
kit card strings word-for-word. At most one touched part is shown (like AR GT).
Also writes ``ar_object_timeline.csv`` with the predicted AR object noun.
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

from monitor.kit_catalog import (
    HAND_LABEL,
    KIT_PART_LABELS,
    ar_object_for_kit,
    hold_touched_parts,
    kit_class_color,
    select_hand_touched_parts,
    stabilize_detection_records,
)
from monitor.overlay_hud import fit_text
from monitor.settings import ROOT
from scripts.kit_sop.run_automatic_assembly import latest_at, read_frame


def _wrap(text: str, width: int = 108, maximum_lines: int = 2) -> list[str]:
    words = " ".join(str(text or "").split()).split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if len(trial) <= width:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= maximum_lines:
            break
    if current and len(lines) < maximum_lines:
        lines.append(current)
    return lines[:maximum_lines] or [""]


def _draw_part(frame: np.ndarray, obj: dict, mask: np.ndarray | None, width: int) -> None:
    label = str(obj.get("kit_label") or obj.get("class") or "part")
    tint = kit_class_color(label)
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


def _draw_hand(frame: np.ndarray, obj: dict, mask: np.ndarray | None, width: int) -> None:
    """Labeled hand outline (no fill) so it does not hide the part underneath."""
    tone = kit_class_color(HAND_LABEL)
    box = obj.get("box")
    if mask is not None and mask.shape[:2] == frame.shape[:2] and mask.any():
        contours = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        cv2.drawContours(frame, contours, -1, tone, 2)
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if len(xs):
            box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return
    x1, y1, x2, y2 = map(int, box)
    if (x2 - x1) * (y2 - y1) > 0.25 * width * frame.shape[0]:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), tone, 2)
    y = max(20, y1 - 6)
    label = HAND_LABEL
    text_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
    cv2.rectangle(
        frame, (x1, max(0, y - 20)), (min(width, x1 + text_w + 8), y + 4),
        (16, 18, 22), -1,
    )
    cv2.putText(
        frame, label, (x1 + 3, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, tone, 2, cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "output/kit_card_sam31_13_assy_0_1",
        help="Folder with detections.json and masks/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/kit_card_sam31_13_assy_0_1/overlay",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "data/generated/industreal/videos/13_assy_0_1.mp4",
    )
    parser.add_argument("--playback-fps", type=float, default=10.0)
    parser.add_argument(
        "--max-parts", type=int, default=1,
        help="AR-style default is 1 primary object per frame.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if ROOT.resolve() not in output.parents and output != ROOT.resolve():
        raise SystemExit("Output must stay inside this project.")
    output.mkdir(parents=True, exist_ok=True)

    detection = json.loads((source / "detections.json").read_text(encoding="utf-8"))
    if not detection.get("records"):
        raise SystemExit("Source detections have no records.")
    detection["records"] = stabilize_detection_records(detection["records"])

    meta = {
        "title": "AR-style primary kit object + hand",
        "source_detections": str(source / "detections.json"),
        "source_masks": str(source / "masks"),
        "video": str(args.video),
        "kit_labels": list(KIT_PART_LABELS),
        "allowed_classes": [*KIT_PART_LABELS, HAND_LABEL],
        "discarded_classes": ["paper"],
        "max_parts": int(args.max_parts),
        "note": (
            "One primary hand-touched kit object per frame (AR GT style). "
            "Paper discarded; lookalike labels temporally locked. "
            "Writes ar_object_timeline.csv for later verb fusion."
        ),
    }
    (output / "overlay_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    timeline_path = output / "ar_object_timeline.csv"
    timeline_rows: list[dict] = []

    cap = cv2.VideoCapture(str(args.video))
    width, height = int(cap.get(3)), int(cap.get(4))
    footer = 132
    playback_fps = float(args.playback_fps)
    detect_fps = float(detection.get("inference_fps") or playback_fps)
    start_s = float(detection["start_s"])
    end_s = float(detection["end_s"])
    destination = output / "kit_hand_overlay.mp4"
    temp_destination = output / "kit_hand_overlay.rendering.mp4"
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
    touched_frames = 0
    prev_kept: list[dict] = []
    touch_ages: dict[str, int] = {}
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
            frame = read_frame(cap, timestamp)
            objects = list(record.get("objects") or [])
            kept: list[dict] = []
            archive = None
            mask_map: dict[str, np.ndarray] = {}
            mask_path = mask_dir / f"{int(record['index']):05d}.npz"
            try:
                archive = np.load(mask_path)
                mask_map = {name: archive[name] for name in archive.files}
            except (OSError, ValueError):
                archive = None
            try:
                for row in objects:
                    if str(row.get("class") or "") != HAND_LABEL:
                        continue
                    box = row.get("box")
                    if isinstance(box, (list, tuple)) and len(box) == 4:
                        area = max(0.0, float(box[2]) - float(box[0])) * max(
                            0.0, float(box[3]) - float(box[1]))
                        if area > 0.25 * width * frame.shape[0]:
                            continue
                    key = str(row.get("id"))
                    mask = mask_map.get(key)
                    if mask is not None:
                        mask = np.asarray(mask, dtype=bool)
                    _draw_hand(frame, row, mask, width)
                kept = select_hand_touched_parts(
                    objects, masks=mask_map, max_parts=int(args.max_parts),
                )
                kept, touch_ages = hold_touched_parts(
                    kept, prev_kept, touch_ages,
                    hold=4, max_parts=int(args.max_parts),
                )
                prev_kept = [dict(row) for row in kept]
                if kept:
                    touched_frames += 1
                for obj in kept:
                    if not obj.get("ar_object"):
                        obj["ar_object"] = ar_object_for_kit(
                            str(obj.get("kit_label") or ""),
                        )
                    key = str(obj.get("id"))
                    mask = mask_map.get(key)
                    if mask is not None:
                        mask = np.asarray(mask, dtype=bool)
                    _draw_part(frame, obj, mask, width)
            finally:
                if archive is not None:
                    archive.close()

            primary = kept[0] if kept else None
            kit_name = str(primary.get("kit_label") or "") if primary else ""
            ar_name = str(primary.get("ar_object") or "") if primary else ""
            timeline_rows.append({
                "frame_index": frame_index,
                "timestamp_s": round(timestamp, 3),
                "detect_index": int(record.get("index") or frame_index),
                "kit_label": kit_name,
                "ar_object": ar_name,
                "score": round(float(primary.get("score") or 0), 4) if primary else "",
                "contact": round(float(primary.get("contact") or 0), 4) if primary else "",
                "held": bool(primary.get("held")) if primary else False,
            })

            canvas = np.zeros((height + footer, width, 3), np.uint8)
            canvas[:height] = frame
            catalog = _wrap(
                "Kit classes: " + ", ".join(KIT_PART_LABELS),
                width=108, maximum_lines=2,
            )
            primary_line = (
                f"Primary now: {kit_name}  (AR object: {ar_name})"
                if kit_name else "Primary now: (none)"
            )
            lines = [
                fit_text(
                    f"t={timestamp:.1f}s   detect {detect_fps:g} FPS   "
                    "AR-STYLE TOP-1 + hand  (paper discarded, labels locked)",
                    110,
                ),
                *catalog,
                fit_text(primary_line, 110),
            ]
            tones = [(210, 220, 230), (170, 190, 210), (170, 190, 210), (80, 230, 160)]
            for index, line in enumerate(lines[:4]):
                cv2.putText(
                    canvas, line, (12, height + 26 + index * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, tones[min(index, 3)], 1, cv2.LINE_AA,
                )
            process.stdin.write(canvas.tobytes())
            frame_count += 1
            if frame_index % max(1, int(playback_fps * 30)) == 0:
                cv2.imwrite(str(output / f"preview_{int(timestamp):04d}.jpg"), canvas)
                print(
                    json.dumps({
                        "frame": frame_index,
                        "of": total,
                        "time": round(timestamp, 1),
                        "primary": kit_name or None,
                        "ar_object": ar_name or None,
                    }),
                    flush=True,
                )
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

    with timeline_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index", "timestamp_s", "detect_index",
                "kit_label", "ar_object", "score", "contact", "held",
            ],
        )
        writer.writeheader()
        writer.writerows(timeline_rows)

    result = {
        "video": str(destination),
        "timeline": str(timeline_path),
        "frames": frame_count,
        "touched_frames": touched_frames,
        "fps": playback_fps,
        "kit_labels": list(KIT_PART_LABELS),
        "source": str(source),
        "render_seconds": round(time.perf_counter() - started, 1),
        "allowed_classes": [*KIT_PART_LABELS, HAND_LABEL],
        "mode": "ar_primary_top1",
        "max_parts": int(args.max_parts),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
