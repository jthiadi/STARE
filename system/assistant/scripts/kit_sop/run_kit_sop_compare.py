"""SOP-guided kit overlay vs AR GT + PSR, with stable class colors.

Uses kit-card SAM detections (no GT at inference):
  1. Temporal label lock so the same blob keeps one class/color
  2. SOP step rules (ready/current/completed) like the top-down pipeline
  3. Prefer hand-touched parts listed on ready/current SOP steps (only without
     --timeline-csv; with it, part labels come solely from that model timeline)
  4. Overlay with SOP checklist + AR GT + PSR + stable kit-class colors
  5. Report AR object FairAcc + PSR step metrics
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
    remap_records_to_sop_classes,
    select_hand_touched_parts_sop,
    sop_preferred_kit_labels,
    stabilize_detection_records,
)
from monitor.overlay_hud import (
    draw_sop_step_rail,
    fit_text,
    step_labels_from_sop,
    step_order_from_sop,
)
from monitor.in_hand import STATIC_LAGS, select_in_hand
from monitor.settings import ROOT
from monitor.step_eval import score_step_observations
from monitor.step_rules import reason_over_detection_records
from scripts.segmentation.evaluate_ar_object_segments import (
    evaluate_object_stream,
    frames_to_segments,
    gt_object_frame_labels,
    majority_smooth,
    pred_object_frame_labels,
)
from scripts.data.ar_labels import active_ar, gt_object_from_action, load_ar_intervals
from scripts.kit_sop.render_ar_psr_gt_overlay import recent_psr_events
from scripts.kit_sop.render_pred_vs_ar_gt_overlay import primary_gt
from scripts.kit_sop.run_automatic_assembly import latest_at, read_frame
from monitor.step_eval import load_psr_events


def _draw_hand(frame: np.ndarray, obj: dict, mask: np.ndarray | None, width: int) -> None:
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


def _draw_part(
    frame: np.ndarray,
    obj: dict,
    mask: np.ndarray | None,
    width: int,
    *,
    match_status: str = "",
) -> None:
    label = str(obj.get("model_label") or obj.get("kit_label") or obj.get("class") or "part")
    tint = kit_class_color(label)  # always class-stable; match only accents HUD
    if mask is not None and mask.shape[:2] == frame.shape[:2] and mask.any():
        frame[mask] = (frame[mask] * 0.8 + np.array(tint) * 0.2).astype(np.uint8)
        contours = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        cv2.drawContours(frame, contours, -1, tint, 2)
    box = obj.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return
    x1, y1, x2, y2 = map(int, box)
    border = _visible_tone(tint)
    if match_status == "MATCH":
        border = (60, 220, 90)
    elif match_status == "MISS":
        border = (60, 60, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2)
    words = label
    if obj.get("model_conf") is not None:
        words = f"{label} {float(obj['model_conf']):.2f}"
    tag = _visible_tone(tint)
    ty = max(20, y1 - 6)
    text_w = cv2.getTextSize(words, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
    corner = (min(width, x1 + text_w + 10), ty + 5)
    cv2.rectangle(frame, (x1, max(0, ty - 21)), corner, tag, -1)
    cv2.rectangle(frame, (x1, max(0, ty - 21)), corner, (255, 255, 255), 1)
    cv2.putText(
        frame, words, (x1 + 5, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _text_on(tag), 2, cv2.LINE_AA,
    )


def _luma(bgr) -> float:
    b, g, r = bgr
    return 0.114 * b + 0.587 * g + 0.299 * r


def _visible_tone(bgr) -> tuple[int, int, int]:
    """Lift very dark class colours (e.g. wheel) so tags and borders stay visible."""
    return (200, 200, 200) if _luma(bgr) < 70 else tuple(int(v) for v in bgr)


def _text_on(bgr) -> tuple[int, int, int]:
    """Black or white text, whichever contrasts with the tag background."""
    return (0, 0, 0) if _luma(bgr) > 140 else (255, 255, 255)


_LOCK_SLACK_PX = 40.0
UNSURE_BELOW = 0.7


def _box_gap(a, b) -> float:
    """Pixel gap between two boxes (0 when they overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def _pick_in_hand(candidates: list[dict], hands: list, locked_id: str | None) -> dict | None:
    """Instance of the predicted class closest to a hand; keep the locked one unless clearly beaten."""
    if not candidates:
        return None
    if not hands:
        locked = [row for row in candidates if str(row.get("id")) == locked_id]
        return locked[0] if locked else max(candidates, key=lambda row: float(row.get("score") or 0))
    gaps = {id(row): min(_box_gap(row["box"], h) for h in hands) for row in candidates}
    best = min(candidates, key=lambda row: gaps[id(row)])
    for row in candidates:
        if str(row.get("id")) == locked_id and gaps[id(row)] <= gaps[id(best)] + _LOCK_SLACK_PX:
            return row
    return best


def _past_boxes(records_by_index: dict[int, dict], index: int) -> list[list[list[float]]]:
    out = []
    for lag in STATIC_LAGS:
        past = records_by_index.get(index - lag)
        if past is None:
            return []
        out.append([row["box"] for row in past.get("objects") or []
                    if row.get("box") and str(row.get("class") or "") != HAND_LABEL])
    return out


def choose_box(picker: str, objects: list[dict], mask_map: dict, pred_kit: str, shape,
               *, past_boxes, locked_id: str | None) -> tuple[dict | None, list[dict]]:
    """(highlighted object relabeled to ``pred_kit`` or None, extra same-class outlines)."""
    if not pred_kit:
        return None, []
    if picker == "contact":
        chosen, _ = select_in_hand(objects, mask_map, pred_kit, shape,
                                   past_boxes=past_boxes, locked_id=locked_id)
        return chosen, []
    hands = [row.get("box") for row in objects
             if str(row.get("class") or "") == HAND_LABEL and row.get("box")]
    candidates = [row for row in objects if str(row.get("class") or "") == pred_kit and row.get("box")]
    chosen = _pick_in_hand(candidates, hands, locked_id)
    if chosen is None:
        return None, []
    others = [row for row in candidates if str(row.get("id")) != str(chosen.get("id"))]
    chosen = dict(chosen)
    chosen["kit_label"] = pred_kit
    return chosen, others


def _draw_outline(frame: np.ndarray, obj: dict, tone) -> None:
    x1, y1, x2, y2 = map(int, obj["box"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), tone, 1)


def _obs_at(observations: list[dict], timestamp_s: float) -> dict:
    if not observations:
        return {}
    return min(observations, key=lambda row: abs(float(row["timestamp_s"]) - timestamp_s))


def build_timeline_and_observations(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    source = args.source.resolve()
    sop = json.loads(args.sop.read_text(encoding="utf-8"))
    detection = json.loads((source / "detections.json").read_text(encoding="utf-8"))
    records = stabilize_detection_records(list(detection.get("records") or []))
    mask_dir = source / "masks"

    # SOP step reasoning on remapped kit→SOP class names (GT-free).
    sop_records = remap_records_to_sop_classes(records)
    print(json.dumps({"reasoning": "sop_step_rules", "records": len(sop_records)}), flush=True)
    observations = reason_over_detection_records(sop_records, sop)

    # Index observations by nearest detect index for preferred-part bias.
    obs_by_index = {
        int(row.get("index") if row.get("index") is not None else round(float(row["timestamp_s"]) * 10)):
        row
        for row in observations
    }

    timeline: list[dict] = []
    prev_kept: list[dict] = []
    ages: dict[str, int] = {}
    for record in records:
        index = int(record["index"])
        obs = obs_by_index.get(index) or _obs_at(observations, float(record["timestamp_s"]))
        preferred_steps = set(obs.get("ready_steps") or [])
        current = obs.get("current_step") or obs.get("step")
        if current and current not in {"unknown", "None", None}:
            preferred_steps.add(str(current))
        preferred = sop_preferred_kit_labels(sop, preferred_steps)

        masks: dict[str, np.ndarray] = {}
        path = mask_dir / f"{index:05d}.npz"
        try:
            with np.load(path) as archive:
                masks = {name: archive[name] for name in archive.files}
        except OSError:
            masks = {}

        kept = select_hand_touched_parts_sop(
            list(record.get("objects") or []),
            masks=masks,
            max_parts=1,
            preferred_kit=preferred,
        )
        kept, ages = hold_touched_parts(
            kept, prev_kept, ages, hold=6, max_parts=1,
        )
        prev_kept = [dict(row) for row in kept]
        primary = kept[0] if kept else None
        kit = str(primary.get("kit_label") or "") if primary else ""
        ar = str(primary.get("ar_object") or ar_object_for_kit(kit) or "") if primary else ""
        timeline.append({
            "frame_index": index,
            "timestamp_s": round(float(record["timestamp_s"]), 3),
            "detect_index": index,
            "kit_label": kit,
            "ar_object": ar,
            "score": round(float(primary.get("score") or 0), 4) if primary else "",
            "contact": round(float(primary.get("contact") or 0), 4) if primary else "",
            "held": bool(primary.get("held")) if primary else False,
            "sop_step": str(current or ""),
            "sop_ready": ",".join(str(s) for s in (obs.get("ready_steps") or [])),
            "sop_completed": ",".join(str(s) for s in (obs.get("completed_steps") or [])),
        })

    smoothed = majority_smooth(
        [str(row.get("ar_object") or "") for row in timeline],
        window=int(args.smooth_window),
    )
    kit_from_ar = {ar_object_for_kit(name) or "": name for name in KIT_PART_LABELS}
    kit_from_ar = {k: v for k, v in kit_from_ar.items() if k}
    for row, ar in zip(timeline, smoothed):
        if ar and ar != row["ar_object"]:
            row["ar_object"] = ar
            row["kit_label"] = kit_from_ar.get(ar, row["kit_label"])

    if args.timeline_csv:
        override = _load_object_timeline(args.timeline_csv, float(args.source_fps))
        for row in timeline:
            hit = override.get(int(round(float(row["timestamp_s"]) * float(args.source_fps))))
            if hit is None:
                continue
            ar, confidence = hit
            row["ar_object"] = ar
            row["kit_label"] = kit_from_ar.get(ar, ar.replace("_", " ")) if ar else ""
            row["model_conf"] = confidence

    return timeline, observations, detection


def _load_object_timeline(path: Path, fps: float) -> dict[int, tuple[str, float | None]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {
            int(round(float(row["timestamp_s"]) * fps)): (
                str(row.get("ar_object") or ""),
                float(row["score"]) if row.get("score") else None,
            )
            for row in csv.DictReader(handle)
        }


def render_overlay(
    args: argparse.Namespace,
    timeline: list[dict],
    observations: list[dict],
    detection: dict,
    sop: dict,
) -> dict:
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    by_time = {round(row["timestamp_s"], 3): row for row in timeline}
    intervals = load_ar_intervals(args.ar_labels, source_fps=float(args.source_fps))
    psr_events = load_psr_events(args.psr_events, source_fps=float(args.source_fps))
    step_order = step_order_from_sop(sop)
    step_labels = step_labels_from_sop(sop)

    cap = cv2.VideoCapture(str(args.video))
    width, height = int(cap.get(3)), int(cap.get(4))
    footer = 168
    playback_fps = float(args.playback_fps)
    start_s = float(detection["start_s"])
    end_s = float(detection["end_s"])
    destination = output / "kit_sop_vs_gt_psr_overlay.mp4"
    temp = output / "kit_sop_vs_gt_psr_overlay.rendering.mp4"
    if temp.exists():
        temp.unlink()
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height + footer}", "-r", str(playback_fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    started = time.perf_counter()
    mask_dir = source / "masks"
    locked_id: str | None = None
    locked_class = ""
    records_by_index = {int(r["index"]): r for r in detection["records"]}
    scored = match = miss = unscored = 0
    compare_rows: list[dict] = []

    try:
        if args.max_seconds:
            end_s = min(end_s, start_s + float(args.max_seconds))
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
            obs = _obs_at(observations, timestamp)

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
                else:
                    miss += 1
                    status = "MISS"
            else:
                unscored += 1
                status = "n/a"

            localized = False
            box_class = ""
            try:
                for row in objects:
                    if str(row.get("class") or "") != HAND_LABEL:
                        continue
                    key = str(row.get("id"))
                    mask = mask_map.get(key)
                    if mask is not None:
                        mask = np.asarray(mask, dtype=bool)
                    _draw_hand(frame, row, mask, width)

                lock = locked_id if locked_class == pred_kit else None
                pred_obj, outlines = choose_box(
                    args.picker, objects, mask_map, pred_kit, frame.shape[:2],
                    past_boxes=_past_boxes(records_by_index, int(record["index"])),
                    locked_id=lock,
                )
                localized = pred_obj is not None
                if pred_obj is not None:
                    box_class = str(pred_obj.get("kit_label") or "")
                    locked_id, locked_class = str(pred_obj.get("id")), pred_kit
                    pred_obj["model_label"] = pred_kit
                    pred_obj["model_conf"] = (pred or {}).get("model_conf")
                for row in outlines:
                    _draw_outline(frame, row, kit_class_color(pred_kit))
                if pred_obj is not None:
                    key = str(pred_obj.get("id"))
                    mask = mask_map.get(key)
                    if mask is not None:
                        mask = np.asarray(mask, dtype=bool)
                    _draw_part(frame, pred_obj, mask, width, match_status=status)
            finally:
                if archive is not None:
                    archive.close()

            draw_sop_step_rail(
                frame,
                step_order,
                step_labels,
                completed=list(obs.get("completed_steps") or []),
                current_step=obs.get("current_step") or obs.get("step"),
                ready=list(obs.get("ready_steps") or []),
            )

            # Top HUD: AR GT vs PRED (match status text only — fill stays class color)
            cv2.rectangle(frame, (0, 0), (width, 52), (18, 20, 24), -1)
            gt_line = (
                f"AR GT: {gt_action or '(none)'}"
                + (f" -> {gt_object}" if gt_object else "")
            )
            conf = (pred or {}).get("model_conf")
            unsure = conf is not None and conf < UNSURE_BELOW
            pred_line = (
                f"MODEL: {pred_kit or '(none)'}"
                + (f" {conf:.2f}" if pred_kit and conf is not None else "")
                + (" (unsure)" if pred_kit and unsure else "")
                + ("  - not located in hand" if pred_kit and not localized else "")
                + f"  [{status}]"
            )
            status_tone = {
                "MATCH": (80, 230, 120),
                "MISS": (80, 80, 255),
                "n/a": (170, 170, 170),
            }.get(status, (200, 200, 200))
            cv2.putText(
                frame, fit_text(gt_line, 90), (12, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 210, 220), 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, fit_text(pred_line, 90), (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_tone, 1, cv2.LINE_AA,
            )

            recent = recent_psr_events(psr_events, timestamp, window_s=3.0)
            psr_line = "PSR: " + (
                ", ".join(
                    f"{row['step_id']}@{row['timestamp_s']:.1f}s"
                    for row in recent[:4]
                ) or "(none recent)"
            )
            sop_line = (
                f"SOP now: {obs.get('current_step') or obs.get('step') or '(none)'}   "
                f"ready={','.join(obs.get('ready_steps') or []) or '-'}   "
                f"done={len(obs.get('completed_steps') or [])}"
            )

            canvas = np.zeros((height + footer, width, 3), np.uint8)
            canvas[:height] = frame
            lines = [
                fit_text(f"t={timestamp:.1f}s  {args.title}", 110),
                fit_text(sop_line, 110),
                fit_text(psr_line, 110),
                fit_text(
                    f"Object acc so far: "
                    f"{(100.0 * match / scored) if scored else 0:.1f}%  "
                    f"({match}/{scored} labeled)",
                    110,
                ),
            ]
            tones = [(210, 220, 230), (160, 210, 180), (160, 180, 220), (200, 200, 120)]
            for i, line in enumerate(lines):
                cv2.putText(
                    canvas, line, (12, height + 28 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, tones[i], 1, cv2.LINE_AA,
                )
            process.stdin.write(canvas.tobytes())

            compare_rows.append({
                "frame_index": frame_index,
                "timestamp_s": round(timestamp, 3),
                "gt_action": gt_action,
                "gt_object": gt_object or "",
                "pred_kit": pred_kit,
                "pred_ar": pred_ar,
                "model_conf": "" if conf is None else round(float(conf), 3),
                "box_sam_class": box_class,
                "status": status,
                "sop_step": str(obs.get("current_step") or obs.get("step") or ""),
                "sop_ready": ",".join(str(s) for s in (obs.get("ready_steps") or [])),
            })
            if frame_index % max(1, int(playback_fps * 30)) == 0:
                print(json.dumps({
                    "frame": frame_index, "of": total,
                    "pred": pred_kit, "gt": gt_object, "status": status,
                    "acc": round(match / scored, 3) if scored else None,
                }), flush=True)
    finally:
        if process.stdin:
            process.stdin.close()
        process.wait()
        cap.release()

    if temp.exists():
        if destination.exists():
            destination.unlink()
        temp.rename(destination)

    object_accuracy = (match / scored) if scored else 0.0
    result = {
        "video": str(destination),
        "frames": frame_count if (frame_count := len(compare_rows)) else 0,
        "scored_frames": scored,
        "unscored_frames": unscored,
        "matches": match,
        "misses": miss,
        "object_accuracy": round(object_accuracy, 4),
        "render_seconds": round(time.perf_counter() - started, 1),
        "mode": "kit_sop_vs_ar_gt_psr",
    }
    with (output / "pred_vs_ar_gt.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compare_rows[0].keys()) if compare_rows else [])
        if compare_rows:
            writer.writeheader()
            writer.writerows(compare_rows)
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=ROOT / "output/kit_card_sam31_13_assy_0_1",
    )
    parser.add_argument(
        "--sop", type=Path,
        default=ROOT / "data/sops/13_assy_0_1.sop.json",
    )
    parser.add_argument(
        "--video", type=Path,
        default=ROOT / "data/generated/industreal/videos/13_assy_0_1.mp4",
    )
    parser.add_argument(
        "--ar-labels", type=Path,
        default=ROOT / "data/evaluation/official/13_assy_0_1/AR_labels.csv",
    )
    parser.add_argument(
        "--psr-events", type=Path,
        default=ROOT / "data/evaluation/official/13_assy_0_1/PSR_labels.csv",
    )
    parser.add_argument(
        "--psr-raw", type=Path,
        default=ROOT / "data/evaluation/official/13_assy_0_1/PSR_labels_raw.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/kit_card_sam31_13_assy_0_1/overlay_sop_vs_gt_psr",
    )
    parser.add_argument("--playback-fps", type=float, default=10.0)
    parser.add_argument("--source-fps", type=float, default=10.0)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument(
        "--timeline-csv", type=Path, default=None,
        help="Use AR objects from this timeline (e.g. TCN output) instead of hand-contact picks.",
    )
    parser.add_argument("--title", default="SOP-guided kit + stable class colors")
    parser.add_argument("--picker", choices=("legacy", "contact"), default="contact",
                        help="legacy: nearest same-label box; contact: mask touching the hand.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Render only the first N seconds.")
    args = parser.parse_args()

    sop = json.loads(args.sop.read_text(encoding="utf-8"))
    timeline, observations, detection = build_timeline_and_observations(args)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    with (out / "ar_object_timeline.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "frame_index", "timestamp_s", "detect_index", "kit_label", "ar_object",
            "score", "contact", "held", "sop_step", "sop_ready", "sop_completed", "model_conf",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(timeline)

    reasoning = {
        "schema_version": 1,
        "predictions_finalized": True,
        "ground_truth_used_for_inference": False,
        "ground_truth_used": False,
        "model": "kit_card+sop_step_rules",
        "observations": observations,
        "bootstrap_prerequisites": [],
        "completion_verified": False,
    }
    (out / "reasoning.json").write_text(json.dumps(reasoning), encoding="utf-8")

    # Paper-style AR object metrics
    n_frames = max((row["frame_index"] for row in timeline), default=-1) + 1
    gt, gt_segs = gt_object_frame_labels(
        load_ar_intervals(args.ar_labels, source_fps=float(args.source_fps)),
        n_frames=n_frames, fps=float(args.source_fps),
    )
    pred = pred_object_frame_labels(timeline, n_frames=n_frames)
    pred_segs = frames_to_segments(pred, float(args.source_fps))
    ar_metrics = evaluate_object_stream(gt, pred, gt_segs, pred_segs)
    (out / "ar_object_paper_metrics.json").write_text(
        json.dumps(ar_metrics, indent=2), encoding="utf-8",
    )
    print(json.dumps({"ar_object_metrics": {
        "covered": round(ar_metrics["covered"], 2),
        "all_labeled": round(ar_metrics["all_labeled"], 2),
        "segment_top1": round(ar_metrics["segment_top1"], 2),
    }}, indent=2), flush=True)

    # PSR metrics (held-out labels, eval only)
    psr_metrics = score_step_observations(
        observations,
        psr_events_path=args.psr_events,
        psr_raw_path=args.psr_raw,
        source_fps=float(args.source_fps),
    )
    (out / "psr_eval.json").write_text(json.dumps(psr_metrics, indent=2), encoding="utf-8")
    completed = psr_metrics.get("completed_set") or {}
    summary = {
        "completed_set_f1": completed.get("mean_f1"),
        "completed_set_precision": completed.get("mean_precision"),
        "completed_set_recall": completed.get("mean_recall"),
        "abstention_rate": psr_metrics.get("abstention_rate"),
    }
    print(json.dumps({"psr_metrics": summary}, indent=2), flush=True)

    result = render_overlay(args, timeline, observations, detection, sop)
    print(json.dumps({"overlay": result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
