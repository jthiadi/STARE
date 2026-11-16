"""Pick the SAM instance that is actually in the hand for a predicted part class.

The predicted class comes from the temporal model; SAM class names are only used
to restrict candidates to the same look-alike family (SAM confuses nut / washers /
acorn nut, the pins, etc.). The returned object is never renamed, so a SAM/model
disagreement stays visible. A candidate must touch a hand mask; parts that have
not moved for a while are penalised so assembled parts next to the hand lose.
"""
from __future__ import annotations

import cv2
import numpy as np

from monitor.kit_catalog import HAND_LABEL, LOOKALIKE_FAMILIES

CONTACT_PX = 12
MIN_CONTACT_FRAC = 0.04
MIN_CONTACT_PIXELS = 25
MAX_AREA_FRAC = 0.08
STATIC_IOU = 0.7
STATIC_LAGS = (15, 30)
LOCK_BONUS = 0.15
EXACT_BONUS = 0.25
STATIC_PENALTY = 0.35


NOT_LOOKALIKE = frozenset({"pulley", "wheel"})  # pink spool vs black tyre


def family(label: str) -> frozenset[str]:
    for group in LOOKALIKE_FAMILIES:
        if label in group and group != NOT_LOOKALIKE:
            return group
    return frozenset({label})


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def hand_contact_mask(objects: list[dict], masks: dict, shape: tuple[int, int]) -> np.ndarray | None:
    union = np.zeros(shape, dtype=np.uint8)
    for row in objects:
        if str(row.get("class") or "") != HAND_LABEL:
            continue
        mask = masks.get(str(row.get("id")))
        if mask is not None and mask.shape[:2] == shape:
            union |= np.asarray(mask, dtype=np.uint8)
    if not union.any():
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CONTACT_PX + 1, 2 * CONTACT_PX + 1))
    return cv2.dilate(union, kernel).astype(bool)


def is_static(box, past_boxes: list[list[list[float]]]) -> bool:
    """True when every lagged frame had an object at (almost) the same place."""
    if not past_boxes:
        return False
    return all(any(_iou(box, b) >= STATIC_IOU for b in boxes) for boxes in past_boxes)


def select_in_hand(
    objects: list[dict],
    masks: dict,
    pred_kit: str,
    shape: tuple[int, int],
    *,
    past_boxes: list[list[list[float]]] | None = None,
    locked_id: str | None = None,
) -> tuple[dict | None, float]:
    """Return (object in hand, score) or (None, 0). The object keeps SAM's own class."""
    if not pred_kit:
        return None, 0.0
    contact = hand_contact_mask(objects, masks, shape)
    if contact is None:
        return None, 0.0
    allowed = family(pred_kit)
    frame_area = float(shape[0] * shape[1])
    best, best_score = None, 0.0
    for row in objects:
        label = str(row.get("class") or "")
        if label == HAND_LABEL or label not in allowed or not row.get("box"):
            continue
        mask = masks.get(str(row.get("id")))
        if mask is None or mask.shape[:2] != shape:
            continue
        mask = np.asarray(mask, dtype=bool)
        area = int(mask.sum())
        if area == 0 or area > MAX_AREA_FRAC * frame_area:
            continue
        touching = int((mask & contact).sum())
        frac = touching / area
        if touching < MIN_CONTACT_PIXELS or frac < MIN_CONTACT_FRAC:
            continue
        score = min(1.0, frac * 2)
        if label == pred_kit:
            score += EXACT_BONUS
        if str(row.get("id")) == locked_id:
            score += LOCK_BONUS
        if is_static(row["box"], past_boxes or []):
            score -= STATIC_PENALTY
        if score > best_score:
            best, best_score = row, score
    if best is None:
        return None, 0.0
    chosen = dict(best)
    chosen["sam_class"] = str(best.get("class") or "")
    chosen["kit_label"] = chosen["sam_class"]
    chosen["model_label"] = pred_kit
    return chosen, best_score
