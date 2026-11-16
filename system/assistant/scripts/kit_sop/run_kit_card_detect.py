"""SAM 3.1 detect using only official kit-card class names + hand (+ paper discard).

Writes plan/detections/masks under ``output/kit_card_sam31_<recording>/``, then
renders the AR-style top-1 overlay. Does not use SOP class aliases like
``hex nut`` / ``toy car wheel``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from monitor.kit_catalog import (
    KIT_PART_LABELS,
    PAPER_LABEL,
    kit_detection_plan,
    kit_label_for_detector,
    kit_prompt_variants,
)
from monitor.settings import ROOT
from scripts.kit_sop.run_automatic_assembly import (
    filter_detections,
    load_sam3,
    mask_iou,
    read_frame,
    write_json,
)


def _remap_kit_class(name: str) -> str | None:
    text = " ".join(str(name or "").strip().lower().replace("_", " ").split())
    if text == PAPER_LABEL:
        return None
    if text == "hand":
        return "hand"
    return kit_label_for_detector(name)


def detect_kit(args: argparse.Namespace) -> Path:
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    plan = kit_detection_plan(include_acorn=not bool(args.no_acorn))
    write_json(output / "plan.json", plan)
    print(json.dumps({"plan_classes": [row["name"] for row in plan["classes"]]}), flush=True)

    predictor = load_sam3(args.checkpoint)
    query_rows = []
    seen = set()
    for row in plan["classes"]:
        for prompt in kit_prompt_variants(row["name"])[:2]:
            key = (row["name"], prompt)
            if key in seen:
                continue
            seen.add(key)
            query_rows.append({**row, "prompt": prompt})

    cap = cv2.VideoCapture(str(args.video))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end = min(source_count / max(source_fps, 1e-6), args.start + args.duration)
    samples = int(math.ceil((end - args.start) * args.fps))
    mask_dir = output / "masks"
    mask_dir.mkdir(exist_ok=True)
    detections_path = output / "detections.json"

    records: list[dict] = []
    previous: list[dict] = []
    next_id = 1
    start_index = 0
    if args.resume and detections_path.is_file():
        prior = json.loads(detections_path.read_text(encoding="utf-8"))
        if prior.get("finalized"):
            print(json.dumps({"resume": "already_finalized", "records": len(prior.get("records") or [])}), flush=True)
            cap.release()
            return output
        records = list(prior.get("records") or [])
        start_index = len(records)
        if records:
            last = records[-1]
            max_id = 0
            previous = []
            mask_path = mask_dir / f"{int(last['index']):05d}.npz"
            archive = np.load(mask_path) if mask_path.is_file() else None
            try:
                for obj in last.get("objects") or []:
                    obj_id = int(obj.get("id") or 0)
                    max_id = max(max_id, obj_id)
                    row = dict(obj)
                    if archive is not None and str(obj_id) in archive.files:
                        row["mask"] = archive[str(obj_id)]
                    previous.append(row)
            finally:
                if archive is not None:
                    archive.close()
            next_id = max_id + 1
        print(json.dumps({"resume": True, "from_index": start_index, "of": samples}), flush=True)

    started = time.perf_counter()
    import torch

    for index in range(start_index, samples):
        timestamp = args.start + index / args.fps
        frame = read_frame(cap, timestamp)
        tick = time.perf_counter()
        predictor.set_image(frame)
        rows: list[dict] = []
        batch = int(getattr(args, "batch", 4) or 4)
        for first in range(0, len(query_rows), batch):
            group = query_rows[first:first + batch]
            try:
                result = predictor(text=[row["prompt"] for row in group])[0]
            except RuntimeError as exc:
                if "CUDA" not in str(exc):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(0.5)
                # Retry one prompt at a time after OOM.
                for row in group:
                    try:
                        part = predictor(text=[row["prompt"]])[0]
                    except RuntimeError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    if part.masks is None:
                        continue
                    masks = part.masks.data.cpu().numpy().astype(bool)
                    for mask, box in zip(masks, part.boxes.data.cpu().numpy()):
                        area = float(mask.mean())
                        if area < args.min_mask_area:
                            continue
                        if row["name"] not in {"hand", PAPER_LABEL} and area > 0.18:
                            continue
                        score = float(box[4])
                        if score < args.min_det_score:
                            continue
                        # Class-specific gate: acorn over-fires on pink blobs.
                        if row["name"] == "acorn nut" and score < 0.45:
                            continue
                        label = _remap_kit_class(row["name"])
                        if not label:
                            continue
                        rows.append({
                            "class": label,
                            "score": score,
                            "mask": mask,
                            "box": [float(v) for v in box[:4]],
                        })
                continue
            if result.masks is None:
                continue
            masks = result.masks.data.cpu().numpy().astype(bool)
            for mask, box in zip(masks, result.boxes.data.cpu().numpy()):
                class_info = group[int(box[5])] if 0 <= int(box[5]) < len(group) else None
                if class_info is None:
                    continue
                area = float(mask.mean())
                if area < args.min_mask_area:
                    continue
                if class_info["name"] not in {"hand", PAPER_LABEL} and area > 0.18:
                    continue
                score = float(box[4])
                if score < args.min_det_score:
                    continue
                if class_info["name"] == "acorn nut" and score < 0.45:
                    continue
                label = _remap_kit_class(class_info["name"])
                if not label:
                    continue
                rows.append({
                    "class": label,
                    "score": score,
                    "mask": mask,
                    "box": [float(v) for v in box[:4]],
                })
        predictor.reset_image()
        if torch.cuda.is_available() and index % 40 == 0:
            torch.cuda.empty_cache()
        rows = filter_detections(rows, frame_shape=frame.shape)
        cleaned = []
        for row in rows:
            label = _remap_kit_class(str(row.get("class") or ""))
            if not label:
                continue
            item = dict(row)
            item["class"] = label
            cleaned.append(item)
        rows = cleaned

        used: set[int] = set()
        for row in rows:
            matches = [
                (mask_iou(row["mask"], old["mask"]), old["id"])
                for old in previous
                if old.get("class") == row["class"]
                and old.get("id") not in used
                and old.get("mask") is not None
            ]
            score, old_id = max(matches, default=(0, None))
            if score >= 0.25:
                row["id"] = old_id
            else:
                row["id"] = next_id
                next_id += 1
            used.add(row["id"])

        if rows:
            np.savez_compressed(
                mask_dir / f"{index:05d}.npz",
                **{str(row["id"]): row["mask"] for row in rows},
            )
        else:
            np.savez_compressed(mask_dir / f"{index:05d}.npz")

        objects = [{k: v for k, v in row.items() if k != "mask"} for row in rows]
        record = {
            "index": index,
            "timestamp_s": timestamp,
            "latency_ms": round(1000 * (time.perf_counter() - tick)),
            "objects": objects,
        }
        records.append(record)
        previous = rows
        write_json(detections_path, {
            "model": "SAM3.1",
            "checkpoint": str(args.checkpoint),
            "source_video": str(args.video),
            "source_fps": source_fps,
            "inference_fps": args.fps,
            "start_s": args.start,
            "end_s": end,
            "ground_truth_used": False,
            "runtime_clicks": 0,
            "kit_card_only": True,
            "kit_labels": list(KIT_PART_LABELS),
            "discarded": [PAPER_LABEL],
            "min_det_score": args.min_det_score,
            "records": records,
            "finalized": index == samples - 1,
            "elapsed_s": time.perf_counter() - started,
        })
        print(json.dumps({
            "sample": index,
            "of": samples,
            "time": round(timestamp, 1),
            "objects": len(objects),
            "latency_ms": record["latency_ms"],
            "classes": sorted({row["class"] for row in objects}),
        }), flush=True)
    cap.release()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path,
                        default=ROOT / "data/generated/industreal/videos/13_assy_0_1.mp4")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "models/sam3.1/sam3.1_multiplex.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/kit_card_sam31_13_assy_0_1")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=100000.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--min-det-score", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=float, default=0.00004)
    parser.add_argument("--batch", type=int, default=4,
                        help="Text-prompt batch size (higher = more GPU use).")
    parser.add_argument("--no-acorn", action="store_true",
                        help="Omit acorn-nut prompt (SAM collapses pink parts onto it).")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--render", action="store_true",
                        help="After detect, render the kit hand overlay.")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}")
    if not args.video.is_file():
        raise SystemExit(f"Missing video: {args.video}")

    output = detect_kit(args)
    if args.render:
        from scripts.kit_sop.render_kit_hand_overlay import main as render_main
        import sys
        sys.argv = [
            "render_kit_hand_overlay",
            "--source", str(output),
            "--output", str(output / "overlay"),
            "--video", str(args.video),
            "--playback-fps", str(args.fps),
            "--max-parts", "1",
        ]
        render_main()
    print(json.dumps({"output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
