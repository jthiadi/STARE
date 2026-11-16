"""Render official AR + PSR ground truth onto one IndustReal video.

EVALUATION VISUALIZATION ONLY. Writes into the GT overlay folder alongside the
AR-only video; does not touch SOP-object / step-rule outputs.
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

from monitor.readiness import OFFICIAL_STEPS
from monitor.settings import ROOT
from monitor.step_eval import completed_steps_at, load_psr_events, load_psr_raw_states, ready_steps_at


def _frame_index(name: str) -> int:
    return int(Path(str(name)).stem)


def load_ar_intervals(path: Path, *, source_fps: float = 10.0) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or len(row) < 5:
                continue
            start_f = _frame_index(row[3])
            end_f = _frame_index(row[4])
            rows.append({
                "action_id": int(row[1]),
                "action": str(row[2]),
                "start_s": start_f / source_fps,
                "end_s": end_f / source_fps,
            })
    rows.sort(key=lambda item: (item["start_s"], item["end_s"], item["action"]))
    return rows


def active_ar(intervals: list[dict], timestamp_s: float) -> list[dict]:
    return [
        row for row in intervals
        if row["start_s"] - 1e-9 <= timestamp_s <= row["end_s"] + 1e-9
    ]


def recent_psr_events(events: list[dict], timestamp_s: float, *, window_s: float = 2.5) -> list[dict]:
    return [
        row for row in events
        if abs(float(row["timestamp_s"]) - timestamp_s) <= window_s
    ]


def _fit(text: str, maximum: int = 110) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= maximum:
        return text
    return text[: max(0, maximum - 3)] + "..."


def _color(key: str) -> tuple[int, int, int]:
    digest = sum(ord(ch) * (index + 1) for index, ch in enumerate(key))
    return (
        40 + (digest * 37) % 180,
        40 + (digest * 73) % 180,
        40 + (digest * 97) % 180,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", type=Path,
        default=ROOT / "data/generated/industreal/videos/03_assy_0_1.mp4",
    )
    parser.add_argument(
        "--ar-labels", type=Path,
        default=ROOT / "data/evaluation/official/03_assy_0_1/AR_labels.csv",
    )
    parser.add_argument(
        "--psr-events", type=Path,
        default=ROOT / "data/evaluation/official/03_assy_0_1/PSR_labels.csv",
    )
    parser.add_argument(
        "--psr-raw", type=Path,
        default=ROOT / "data/evaluation/official/03_assy_0_1/PSR_labels_raw.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/ar_gt_overlay_03_assy_0_1",
    )
    parser.add_argument("--playback-fps", type=float, default=10.0)
    args = parser.parse_args()

    output = args.output.resolve()
    if ROOT.resolve() not in output.parents and output != ROOT.resolve():
        raise SystemExit("Output must stay inside this project.")
    output.mkdir(parents=True, exist_ok=True)

    ar_intervals = load_ar_intervals(args.ar_labels, source_fps=10.0)
    psr_events = load_psr_events(args.psr_events, source_fps=10.0)
    psr_raw = load_psr_raw_states(args.psr_raw, source_fps=10.0)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    duration_s = frame_count_src / source_fps if source_fps > 0 else 0.0
    playback_fps = float(args.playback_fps)
    footer = 210
    destination = output / "ar_psr_gt_overlay.mp4"

    meta = {
        "title": "Official IndustReal AR + PSR ground-truth overlay",
        "video": str(args.video),
        "ar_labels": str(args.ar_labels),
        "psr_events": str(args.psr_events),
        "psr_raw": str(args.psr_raw),
        "ar_intervals": len(ar_intervals),
        "psr_completion_events": [
            {
                "timestamp_s": row["timestamp_s"],
                "step_id": row["step_id"],
                "description": row.get("description", ""),
            }
            for row in psr_events
        ],
        "note": "Held-out GT visualization only; not a model prediction.",
    }
    (output / "ar_psr_overlay_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height + footer}", "-r", str(playback_fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    started = time.perf_counter()
    written = 0
    try:
        total = max(1, int(math.floor(duration_s * playback_fps)))
        sequential = abs(playback_fps - source_fps) < 1e-6
        source_frame = -1
        for frame_index in range(total):
            timestamp = frame_index / playback_fps
            if timestamp >= duration_s - 1e-9:
                timestamp = max(0.0, duration_s - 1e-3)
            target_frame = min(frame_count_src - 1, int(round(timestamp * source_fps)))
            if sequential:
                ok, frame = cap.read()
                source_frame += 1
                if not ok:
                    break
            else:
                if target_frame != source_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    source_frame = target_frame
                ok, frame = cap.read()
                if not ok:
                    break
                source_frame = target_frame

            active = active_ar(ar_intervals, timestamp)
            completed = sorted(completed_steps_at(psr_raw, timestamp))
            ready = sorted(ready_steps_at(set(completed)))
            flash = recent_psr_events(psr_events, timestamp, window_s=2.5)

            # Top: AR GT
            cv2.rectangle(frame, (0, 0), (width, 40), (20, 25, 30), -1)
            if active:
                ar_text = " | ".join(row["action"] for row in active[:3])
                tint = _color(active[0]["action"])
                cv2.rectangle(frame, (0, 0), (8, 40), tint, -1)
                cv2.putText(
                    frame, _fit(f"AR GT: {ar_text}", 95),
                    (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tint, 2, cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame, "AR GT: (no labeled action)",
                    (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 160, 170), 2, cv2.LINE_AA,
                )

            # Second banner: PSR completion flash / completed set
            cv2.rectangle(frame, (0, 40), (width, 80), (25, 30, 38), -1)
            if flash:
                names = ", ".join(
                    f"{row['step_id']} {OFFICIAL_STEPS.get(row['step_id'], ('', '', row.get('description','')))[2]}"
                    for row in flash
                )
                cv2.rectangle(frame, (0, 40), (8, 80), (80, 220, 120), -1)
                cv2.putText(
                    frame, _fit(f"PSR COMPLETE: {names}", 92),
                    (16, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 230, 160), 2, cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame, _fit(
                        "PSR completed: " + (", ".join(completed) if completed else "(none yet)"),
                        95,
                    ),
                    (16, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 200, 220), 2, cv2.LINE_AA,
                )

            canvas = np.zeros((height + footer, width, 3), np.uint8)
            canvas[:height] = frame
            lines = [
                f"t={timestamp:.1f}s   frame={max(0, source_frame):06d}   official GT only (AR + PSR)",
                _fit(
                    "AR active: " + (
                        ", ".join(f"{row['action']}(id={row['action_id']})" for row in active)
                        if active else "(none)"
                    ),
                    115,
                ),
                _fit(
                    "PSR completed: " + (", ".join(completed) if completed else "(none)"),
                    115,
                ),
                _fit(
                    "PSR ready now: " + (", ".join(ready) if ready else "(none)"),
                    115,
                ),
                _fit(
                    "Next PSR event: " + (
                        next(
                            (
                                f"{row['step_id']} @{row['timestamp_s']:.1f}s — "
                                f"{row.get('description') or OFFICIAL_STEPS.get(row['step_id'], ('','',''))[2]}"
                                for row in psr_events
                                if row["timestamp_s"] > timestamp + 1e-9
                            ),
                            "(all PSR completions done)",
                        )
                    ),
                    115,
                ),
            ]
            for index, line in enumerate(lines):
                cv2.putText(
                    canvas, line, (12, height + 26 + index * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 230, 240), 1, cv2.LINE_AA,
                )
            process.stdin.write(canvas.tobytes())
            written += 1
            if frame_index % 100 == 0:
                print(
                    json.dumps({
                        "frame": frame_index,
                        "of": total,
                        "time": round(timestamp, 1),
                        "ar": [row["action"] for row in active],
                        "psr_completed": completed,
                    }),
                    flush=True,
                )
    finally:
        if process.stdin:
            process.stdin.close()
        process.wait()
        cap.release()

    result = {
        "video": str(destination),
        "frames": written,
        "fps": playback_fps,
        "ar_intervals": len(ar_intervals),
        "psr_events": len(psr_events),
        "render_seconds": round(time.perf_counter() - started, 1),
        "mode": "ar_psr_ground_truth_overlay",
        "folder": str(output),
    }
    (output / "ar_psr_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
