"""SOP-conditioned open-vocabulary proposals for SAM2.

YOLO-World proposes boxes from text copied from the active SOP.  It does not
decide whether a step is complete.  SAM2 propagates accepted proposals and the
downstream relation monitor makes a separate, deterministic temporal check.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .sam2_focus import validate_prompt_box
from .settings import ROOT, YOLO_WORLD_DIR, YOLO_WORLD_ID


def _clean_prompt(value: Any) -> str:
    value = " ".join(str(value).strip().split())
    if not value or len(value) > 100:
        raise ValueError("Every open-vocabulary prompt must contain 1-100 characters.")
    return value


def validate_part_prompts(rows: Any, sop_step: dict) -> list[dict]:
    """Validate a runtime prompt plan against the active SOP's named parts."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("The active step needs at least one part prompt.")
    sop_text = " ".join([sop_step.get("instruction", ""), *sop_step.get("parts", [])]).lower()
    output, names = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every part prompt must be an object.")
        name = _clean_prompt(row.get("name", "")).lower().replace(" ", "_")
        prompt = _clean_prompt(row.get("prompt", ""))
        role = row.get("role")
        count = row.get("count", 1)
        if name in names or role not in ("moving", "target", "context"):
            raise ValueError("Part names must be unique and roles must be moving, target, or context.")
        if type(count) is not int or not 1 <= count <= 6:
            raise ValueError("Part prompt count must be an integer from 1 to 6.")
        evidence = _clean_prompt(row.get("sop_evidence", prompt))
        # The prompt may add generic detector wording ("toy component"), but
        # each semantic identity must remain traceable to active-SOP wording.
        if evidence.lower() not in sop_text:
            raise ValueError(f"Part prompt {name} is not traceable to the active SOP.")
        aliases = [_clean_prompt(x) for x in row.get("aliases", [])]
        names.add(name)
        output.append({"name": name, "prompt": prompt, "aliases": aliases,
                       "role": role, "count": count, "sop_evidence": evidence})
    if not any(row["role"] == "moving" for row in output):
        raise ValueError("At least one moving part prompt is required.")
    if not any(row["role"] == "target" for row in output):
        raise ValueError("At least one target prompt is required.")
    return output


def _box_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if not union else intersection / union


def _deduplicate(rows: list[dict], threshold: float = .75) -> list[dict]:
    kept = []
    for row in sorted(rows, key=lambda item: item["confidence"], reverse=True):
        if all(_box_iou(row["box"], other["box"]) < threshold for other in kept):
            kept.append(row)
    return kept


class YoloWorldLocalizer:
    """Training-free-at-station text-to-box adapter using YOLO-World."""
    def __init__(self, model_path: Path = YOLO_WORLD_DIR, device: str = "cuda"):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ValueError(f"YOLO-World is missing: {model_path}")
        # Keep Ultralytics settings/runs inside this repository.  CLIP's model
        # cache follows its own standard cache location.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(str(model_path))
        self.device = device
        # set_classes builds CLIP text embeddings. Move the detector before
        # that call so those embeddings and the image tensors share a device.
        self.model.to(device)

    def locate(self, image: Image.Image, prompts: list[dict], confidence: float = .035,
               image_size: int = 960) -> dict:
        if not 0 < confidence < 1 or image_size not in (640, 768, 960, 1280):
            raise ValueError("Use a confidence in (0,1) and a supported image size.")
        classes, class_to_part = [], {}
        for row in prompts:
            for text in [row["prompt"], *row.get("aliases", [])]:
                if text not in class_to_part:
                    class_to_part[text] = row["name"]
                    classes.append(text)
        self.model.set_classes(classes)
        result = self.model.predict(np.asarray(image.convert("RGB")), conf=confidence,
                                    iou=.45, imgsz=image_size, device=self.device,
                                    verbose=False, max_det=40)[0]
        width, height = image.size
        candidates: dict[str, list[dict]] = {row["name"]: [] for row in prompts}
        for box, score, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            class_name = result.names[int(class_id)]
            semantic_name = class_to_part.get(class_name)
            if semantic_name is None:
                continue
            normalized = validate_prompt_box([
                float(box[0]) / width, float(box[1]) / height,
                float(box[2]) / width, float(box[3]) / height,
            ])
            candidates[semantic_name].append({"box": normalized, "confidence": float(score),
                                              "matched_prompt": class_name})

        objects, missing = [], []
        object_id = 1
        by_name = {row["name"]: row for row in prompts}
        for semantic_name, rows in candidates.items():
            spec = by_name[semantic_name]
            selected = _deduplicate(rows)[:spec["count"]]
            if len(selected) < spec["count"]:
                missing.append({"name": semantic_name, "expected": spec["count"],
                                "found": len(selected)})
            # Stable left-to-right suffixes make two wheels distinguishable.
            selected.sort(key=lambda item: (item["box"][0] + item["box"][2]) / 2)
            for index, item in enumerate(selected, 1):
                label = semantic_name if spec["count"] == 1 else f"{semantic_name}_{index}"
                objects.append({"object_id": object_id, "name": label,
                                "semantic_name": semantic_name, "role": spec["role"], **item})
                object_id += 1
        return {
            "model": YOLO_WORLD_ID,
            "station_training": False,
            "source": "active_sop_text_prompts",
            "objects": objects,
            "missing": missing,
            "complete": not missing,
        }


@dataclass
class RelocalizationGate:
    """Ask for a fresh detector pass after persistent SAM2 identity failure."""
    invalid_samples_before_retry: int = 3
    minimum_seconds_between_retries: float = 2.0

    def __post_init__(self):
        if self.invalid_samples_before_retry < 1 or self.minimum_seconds_between_retries < 0:
            raise ValueError("Invalid re-localization policy.")
        self.invalid_streak = 0
        self.last_retry_s = -math.inf

    def observe(self, quality_valid: bool, timestamp_s: float) -> bool:
        if quality_valid:
            self.invalid_streak = 0
            return False
        self.invalid_streak += 1
        eligible = (self.invalid_streak >= self.invalid_samples_before_retry
                    and timestamp_s - self.last_retry_s >= self.minimum_seconds_between_retries)
        if eligible:
            self.invalid_streak = 0
            self.last_retry_s = timestamp_s
        return eligible
