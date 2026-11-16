"""IndustReal kit part names as printed on the official parts card.

Detection and overlay keep only the 15 card classes plus ``hand``.
``paper`` may be queried so it can be discarded, never shown as a kit part.
"""
from __future__ import annotations

from typing import Any

import math

import cv2
import numpy as np


# Exact labels from the kit "Parts" card (left column then right column).
KIT_PART_LABELS: tuple[str, ...] = (
    "nut",
    "wing",
    "acorn nut",
    "pulley",
    "round washer",
    "tooth washer",
    "wheel",
    "base",
    "front bracket",
    "short brace",
    "long brace",
    "pin long",
    "pin middle",
    "pin short",
    "screw pin",
)

KIT_PART_LABEL_SET = {name.casefold(): name for name in KIT_PART_LABELS}

# Detailed SAM text prompts from the official parts card (≤80 chars for the decoder).
# Video fasteners are pink/magenta plastic; pins are grey; braces/base are white.
KIT_PART_PROMPTS: dict[str, str] = {
    "nut": "tiny pink hex star nut, six points, center hole, not a washer",
    "wing": "pink rectangular panel with an angled face and a mounting tab with a hole",
    "acorn nut": "pink dome acorn cap nut, tall rounded top, not flat",
    "pulley": "pink grooved spool pulley disc, not a rubber wheel",
    "round washer": "thin flat pink circular washer ring, hollow center",
    "tooth washer": "pink lock washer with jagged outer teeth around edge",
    "wheel": "black rubber toy-car wheel with tread, not a washer",
    "base": "White plate with round holes and two raised tabs with rectangular openings",
    "front bracket": "white plastic U-shaped front bracket for the front wheel axle",
    "short brace": "short straight white brace bar with three holes",
    "long brace": "long white L-shaped perforated brace bar with holes",
    "pin long": "Dark gray pin with the longest unthreaded shaft and a round head",
    "pin middle": "Dark gray unthreaded pin, between the short and long pins in length",
    "pin short": "Dark gray pin with the shortest unthreaded shaft and a round head",
    "screw pin": "Dark gray pin with visible screw threads at the tip and a round head",
    "hand": "human hand and fingers",
    "paper": "printed paper assembly instruction sheet with diagrams",
}

KIT_PART_PROMPT_VARIANTS: dict[str, tuple[str, ...]] = {
    "nut": (
        "small pink star-shaped plastic nut with a center hole",
        "tiny pink hexagonal plastic nut fastener",
    ),
    "tooth washer": (
        "small pink toothed lock washer with outer teeth",
        "small pink lock washer with radial teeth",
    ),
    "round washer": (
        "small thin pink flat circular washer ring",
        "tiny pink flat washer disc with a hole",
    ),
    "wheel": (
        "black rubber toy-car wheel with tread, not a washer",
        "black circular toy tire wheel",
    ),
    "front bracket": (
        "white plastic steering bracket for toy car front wheel",
    ),
    "wing": (
        "white plastic wing beam block, bent corner piece with holes",
    ),
    "pin short": (
        "short grey T-handle pin, smooth unthreaded shaft",
        "small grey T-shaped pin on the table",
    ),
    "screw pin": (
        "short grey threaded screw pin with a screw head",
        "tiny grey machine screw axle, not a T-handle pin",
    ),
    "pulley": (
        "small pink spool-shaped plastic pulley with a groove",
        "tiny pink grooved pulley disc, not a rubber tire",
    ),
}

# Legacy SAM / SOP names → kit card strings. Unlisted classes are dropped.
_DETECTOR_TO_KIT: dict[str, str] = {
    "nut": "nut",
    "hex nut": "nut",
    "wing": "wing",
    "wing beam": "wing",
    "acorn nut": "acorn nut",
    "pulley": "pulley",
    "round washer": "round washer",
    "round flat washer": "round washer",
    "tooth washer": "tooth washer",
    "tooth lock washer": "tooth washer",
    "wheel": "wheel",
    "toy car wheel": "wheel",
    "base": "base",
    "chassis base": "base",
    "partial model": "base",
    "front bracket": "front bracket",
    "white front bracket": "front bracket",
    "short brace": "short brace",
    "long brace": "long brace",
    "pin long": "pin long",
    "long pin": "pin long",
    "long metal pin": "pin long",
    "pin middle": "pin middle",
    "middle pin": "pin middle",
    "middle metal pin": "pin middle",
    "pin short": "pin short",
    "short pin": "pin short",
    "short metal pin": "pin short",
    "screw pin": "screw pin",
}

PAPER_LABEL = "paper"
HAND_LABEL = "hand"


def kit_prompt_for(name: str) -> str:
    """Primary SAM prompt for a kit-card / hand / paper class."""
    text = " ".join(str(name or "").strip().lower().replace("_", " ").split())
    if text in KIT_PART_LABEL_SET:
        text = KIT_PART_LABEL_SET[text]
    return KIT_PART_PROMPTS.get(text, text)[:80]


def kit_prompt_variants(name: str) -> list[str]:
    """Primary prompt plus optional recall variants (still ≤80 chars)."""
    text = " ".join(str(name or "").strip().lower().replace("_", " ").split())
    if text == "hand" or text == PAPER_LABEL:
        primary = kit_prompt_for(text)
        return [primary] if primary else []
    if text in KIT_PART_LABEL_SET:
        text = KIT_PART_LABEL_SET[text]
    elif text in _DETECTOR_TO_KIT:
        text = _DETECTOR_TO_KIT[text]
    primary = kit_prompt_for(text)
    out: list[str] = []
    for prompt in (primary, *(KIT_PART_PROMPT_VARIANTS.get(text) or ())):
        clipped = str(prompt or "")[:80]
        if clipped and clipped not in out:
            out.append(clipped)
    return out


def kit_detection_plan(*, include_acorn: bool = True) -> dict:
    """Plan with only the 15 kit classes + hand + paper (paper for discard)."""
    classes = []
    for name in KIT_PART_LABELS:
        if name == "acorn nut" and not include_acorn:
            continue
        classes.append({
            "name": name,
            "prompt": kit_prompt_for(name),
            "entity": name.replace(" ", "_"),
            "role": "part",
            "source": "kit_parts_card",
        })
    classes.append({
        "name": HAND_LABEL,
        "prompt": kit_prompt_for(HAND_LABEL),
        "entity": "hand",
        "role": "context",
        "source": "kit_parts_card",
    })
    classes.append({
        "name": PAPER_LABEL,
        "prompt": kit_prompt_for(PAPER_LABEL),
        "entity": "paper",
        "role": "context",
        "source": "kit_parts_card_discard",
    })
    return {
        "classes": classes,
        "kit_labels": [row["name"] for row in classes if row["name"] in KIT_PART_LABELS],
        "discarded": [PAPER_LABEL],
        "source": "official_parts_card",
        "include_acorn": bool(include_acorn),
    }

# Mask: part is touched when this fraction of its pixels sit under the hand.
_MASK_COVER = 0.10
_MASK_PIXELS = 30
# Box-on-hand-mask: used when the part mask is missing or disjoint from the hand.
_BOX_ON_HAND_COVER = 0.18
_HAND_DILATE_PX = 8
# AR-style: one primary kit object per frame (matches AR GT sparsity).
DEFAULT_MAX_PARTS = 1

# Kit card → IndustReal AR object noun (verb comes from another model later).
KIT_TO_AR_OBJECT: dict[str, str] = {
    "nut": "nut",
    "wing": "wing",
    "acorn nut": "acorn_nut",
    "pulley": "pulley",
    "round washer": "round_washer",
    "tooth washer": "tooth_washer",
    "wheel": "wheel",
    "base": "partial_model",
    "front bracket": "front_bracket",
    "short brace": "short_brace",
    "long brace": "long_brace",
    "pin long": "pin_long",
    "pin middle": "pin_middle",
    "pin short": "pin_short",
    "screw pin": "screw_pin",
}

# Kit card → SOP / step-rules part vocabulary (IndustReal SOP JSON).
KIT_TO_SOP_PART: dict[str, str] = {
    "nut": "hex nut",
    "acorn nut": "acorn nut",
    "round washer": "round flat washer",
    "tooth washer": "tooth lock washer",
    "wheel": "toy car wheel",
    "base": "partial model",
    "front bracket": "white front bracket",
    "pin short": "short metal pin",
    "pin middle": "middle metal pin",
    "pin long": "long metal pin",
    "screw pin": "screw pin",
    "wing": "wing",
    "pulley": "pulley",
    "short brace": "short brace",
    "long brace": "long brace",
}

SOP_TO_KIT: dict[str, str] = {
    value: key for key, value in KIT_TO_SOP_PART.items()
}
SOP_TO_KIT.update({
    "wing beam": "wing",
    "hex nut": "nut",
    "tooth lock washer": "tooth washer",
    "round flat washer": "round washer",
    "partial model": "base",
    "toy car wheel": "wheel",
    "white front bracket": "front bracket",
    "short metal pin": "pin short",
    "middle metal pin": "pin middle",
    "long metal pin": "pin long",
})

# Prefer gripped fasteners/pins over the chassis the wrist rests on.
# Acorn is rare in AR but SAM over-fires it on every pink blob — keep weight tiny.
_PRIMARY_WEIGHT: dict[str, float] = {
    "nut": 1.70,
    "round washer": 1.55,
    "tooth washer": 1.55,
    "acorn nut": 0.22,
    "pin short": 1.35,
    "pin middle": 1.30,
    "pin long": 1.40,
    "screw pin": 1.25,
    "wheel": 1.35,
    "pulley": 1.00,
    "wing": 1.45,
    "short brace": 0.95,
    "long brace": 0.80,
    "front bracket": 0.75,
    "base": 0.35,
}

# SAM flips these when the part rotates or the hand occludes it.
LOOKALIKE_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"nut", "tooth washer", "round washer", "acorn nut"}),
    frozenset({"pin short", "pin middle", "pin long", "screw pin"}),
    frozenset({"pulley", "wheel"}),
    frozenset({"short brace", "long brace", "front bracket", "wing", "base"}),
)
_COLLAPSE_IOU = 0.40
_SAME_BLOB_IOU = 0.65
_MATCH_IOU = 0.30
_SWITCH_LOOKALIKE = 10
_SWITCH_OTHER = 7
_SWITCH_CONFIDENT = 3
_SCORE_MARGIN = 0.30
_HOLD_MISS = 6
_HOLD_TOUCH = 5


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _as_box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(x == x and abs(x) != float("inf") for x in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def kit_label_for_detector(name: str) -> str | None:
    """Map a detector class onto the kit card string, or None if not a kit part."""
    text = _clean(name)
    if not text:
        return None
    if text in KIT_PART_LABEL_SET:
        return KIT_PART_LABEL_SET[text]
    return _DETECTOR_TO_KIT.get(text)


def is_hand_class(name: str) -> bool:
    text = _clean(name)
    return text == "hand" or text.endswith(" hand")


def kit_class_color(label: str) -> tuple[int, int, int]:
    """Stable BGR tint per kit / SOP class name (never per instance id)."""
    text = kit_label_for_detector(label) or _clean(label)
    # Explicit palette so the same class never drifts hue when track ids change.
    palette: dict[str, tuple[int, int, int]] = {
        "nut": (50, 50, 240),
        "wing": (40, 160, 255),
        "acorn nut": (80, 40, 200),
        "pulley": (200, 80, 200),
        "round washer": (40, 200, 220),
        "tooth washer": (30, 180, 140),
        "wheel": (30, 30, 30),
        "base": (220, 220, 220),
        "front bracket": (180, 200, 80),
        "short brace": (160, 220, 120),
        "long brace": (100, 200, 60),
        "pin long": (200, 160, 80),
        "pin middle": (220, 140, 60),
        "pin short": (240, 120, 40),
        "screw pin": (180, 100, 40),
        "hand": (210, 210, 210),
        # SOP / legacy aliases → same hues as kit card counterparts.
        "hex nut": (50, 50, 240),
        "tooth lock washer": (30, 180, 140),
        "round flat washer": (40, 200, 220),
        "partial model": (220, 220, 220),
        "toy car wheel": (30, 30, 30),
        "white front bracket": (180, 200, 80),
        "short metal pin": (240, 120, 40),
        "middle metal pin": (220, 140, 60),
        "long metal pin": (200, 160, 80),
        "wing beam": (40, 160, 255),
        "paper": (90, 90, 140),
    }
    if text in palette:
        return palette[text]
    seed = sum((index + 1) * ord(ch) for index, ch in enumerate(text))
    return (
        40 + (seed * 73) % 200,
        40 + (seed * 43) % 200,
        40 + (seed * 97) % 200,
    )


def _box_intersection_area(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a: list[float], b: list[float]) -> float:
    inter = _box_intersection_area(a, b)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dilate_mask(mask: np.ndarray, radius: int = _HAND_DILATE_PX) -> np.ndarray:
    array = np.asarray(mask, dtype=np.uint8)
    if array.size == 0 or radius <= 0:
        return array.astype(bool)
    size = max(3, int(radius) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(array, kernel).astype(bool)


def _mask_contact(hand_mask: np.ndarray, part_mask: np.ndarray) -> float:
    if hand_mask.shape != part_mask.shape:
        return 0.0
    part = np.asarray(part_mask, dtype=bool)
    hand = np.asarray(hand_mask, dtype=bool)
    part_area = int(part.sum())
    if part_area <= 0:
        return 0.0
    inter = int((part & hand).sum())
    if inter < _MASK_PIXELS:
        return 0.0
    cover = inter / part_area
    return cover if cover >= _MASK_COVER else 0.0


def _box_on_hand_mask(hand_mask: np.ndarray, part: list[float]) -> float:
    """Fraction of the part box that sits on the (dilated) hand silhouette."""
    height, width = hand_mask.shape[:2]
    x1 = max(0, int(part[0]))
    y1 = max(0, int(part[1]))
    x2 = min(width, int(math.ceil(part[2])))
    y2 = min(height, int(math.ceil(part[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = np.asarray(hand_mask[y1:y2, x1:x2], dtype=bool)
    part_area = (x2 - x1) * (y2 - y1)
    inter = int(crop.sum())
    if part_area <= 0 or inter < _MASK_PIXELS:
        return 0.0
    cover = inter / part_area
    return cover if cover >= _BOX_ON_HAND_COVER else 0.0


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_contact(hand: list[float], part: list[float]) -> float:
    inter = _box_intersection_area(hand, part)
    part_area = _box_area(part)
    if part_area <= 0:
        return 0.0
    cover = inter / part_area
    iou = _box_iou(hand, part)
    if cover >= 0.20 or iou >= 0.04:
        return max(cover, iou)
    return 0.0


def _hand_union_mask(hands: list[np.ndarray]) -> np.ndarray | None:
    if not hands:
        return None
    merged = np.zeros(hands[0].shape, dtype=bool)
    for mask in hands:
        if mask.shape == merged.shape:
            merged |= np.asarray(mask, dtype=bool)
    return merged if merged.any() else None


def ar_object_for_kit(label: str) -> str | None:
    """Map a kit-card label onto the IndustReal AR object noun."""
    text = kit_label_for_detector(label) or _clean(label)
    return KIT_TO_AR_OBJECT.get(text)


def sop_part_for_kit(label: str) -> str | None:
    text = kit_label_for_detector(label) or _clean(label)
    return KIT_TO_SOP_PART.get(text)


def kit_label_for_sop(name: str) -> str | None:
    text = _clean(name)
    if text in SOP_TO_KIT:
        return SOP_TO_KIT[text]
    return kit_label_for_detector(text)


def remap_records_to_sop_classes(records: list[dict]) -> list[dict]:
    """Rewrite kit-card ``class`` strings to SOP vocabulary for step rules."""
    output = []
    for record in records or []:
        objects = []
        for row in record.get("objects") or []:
            item = dict(row)
            raw = str(item.get("class") or "")
            if is_hand_class(raw):
                item["class"] = HAND_LABEL
                objects.append(item)
                continue
            kit = kit_label_for_detector(raw)
            sop = KIT_TO_SOP_PART.get(kit or "", "")
            if not sop:
                continue
            item["class"] = sop
            item["kit_label"] = kit
            objects.append(item)
        output.append({**record, "objects": objects})
    return output


def sop_preferred_kit_labels(sop: dict, step_ids: list[str] | set[str]) -> set[str]:
    """Kit labels listed on the given SOP steps (for ranking bias)."""
    wanted = {str(sid) for sid in (step_ids or [])}
    out: set[str] = set()
    for step in sop.get("steps") or []:
        if str(step.get("id") or "") not in wanted:
            continue
        for part in step.get("parts") or []:
            kit = kit_label_for_sop(str(part))
            if kit:
                out.add(kit)
    return out


def select_hand_touched_parts_sop(
    objects: list[dict],
    *,
    masks: dict[str, Any] | None = None,
    max_parts: int = DEFAULT_MAX_PARTS,
    preferred_kit: set[str] | None = None,
) -> list[dict]:
    """Like ``select_hand_touched_parts`` but boosts SOP-preferred kit labels."""
    kept = select_hand_touched_parts(
        objects, masks=masks, max_parts=max(6, int(max_parts)),
    )
    if not preferred_kit or not kept:
        return kept[: max(0, int(max_parts))]
    preferred = []
    other = []
    for row in kept:
        label = str(row.get("kit_label") or "")
        if label in preferred_kit:
            preferred.append(row)
        else:
            other.append(row)
    return (preferred + other)[: max(0, int(max_parts))]


def _primary_rank(
    item: dict,
    *,
    frame_area: float | None,
) -> tuple[float, float, float]:
    """Higher is better: contact×class weight, smallness, SAM score."""
    label = str(item.get("kit_label") or "")
    contact = float(item.get("contact") or 0.0)
    score = float(item.get("score") or 0.0)
    weight = _PRIMARY_WEIGHT.get(label, 1.0)
    box = _as_box(item.get("box"))
    area_frac = 0.02
    aspect = 1.0
    if box is not None and frame_area and frame_area > 0:
        area_frac = max(1e-6, _box_area(box) / frame_area)
        width = max(1e-6, box[2] - box[0])
        height = max(1e-6, box[3] - box[1])
        aspect = max(width, height) / min(width, height)
    # Smaller gripped parts beat large chassis under the wrist.
    smallness = 1.0 / (area_frac ** 0.35)
    # Acorn is dome-shaped (taller); flat pink blobs should not win as acorn.
    if label == "acorn nut":
        if aspect < 1.45:
            weight *= 0.15
        elif aspect < 1.8:
            weight *= 0.45
        if score < 0.45:
            weight *= 0.4
    if label in {"nut", "round washer", "tooth washer"} and aspect > 1.8:
        weight *= 0.7
    # Wing is a larger pink plate — prefer over tiny false fasteners when big.
    if label == "wing" and area_frac > 0.008:
        weight *= 1.25
    return (contact * weight * smallness, contact, score)


def _relabel_false_acorn(item: dict, *, frame_area: float | None) -> dict:
    """Solo acorn detections on flat pink blobs are almost always nut/washer."""
    if str(item.get("kit_label") or "") != "acorn nut":
        return item
    box = _as_box(item.get("box"))
    aspect = 1.0
    area_frac = 0.02
    if box is not None and frame_area and frame_area > 0:
        area_frac = max(1e-6, _box_area(box) / frame_area)
        width = max(1e-6, box[2] - box[0])
        height = max(1e-6, box[3] - box[1])
        aspect = max(width, height) / min(width, height)
    score = float(item.get("score") or 0.0)
    # True acorn: tall dome + confident. Otherwise treat as common nut.
    if aspect >= 1.55 and score >= 0.50:
        return item
    out = dict(item)
    out["kit_label"] = "nut"
    out["class"] = "nut"
    out["ar_object"] = ar_object_for_kit("nut")
    out["relabeled_from"] = "acorn nut"
    return out


def _resolve_pink_lookalikes(candidates: list[dict]) -> list[dict]:
    """Acorn never wins when any other pink fastener is also hand-touched."""
    if not candidates:
        return candidates
    pink = {"nut", "round washer", "tooth washer", "acorn nut"}
    present = [row for row in candidates if row.get("kit_label") in pink]
    has_other_pink = any(row.get("kit_label") != "acorn nut" for row in present)
    if not has_other_pink:
        return candidates
    filtered = [row for row in candidates if row.get("kit_label") != "acorn nut"]
    return filtered or candidates


def select_hand_touched_parts(
    objects: list[dict],
    *,
    masks: dict[str, Any] | None = None,
    max_parts: int = DEFAULT_MAX_PARTS,
) -> list[dict]:
    """Keep kit parts whose mask/box actually overlaps a detected hand.

    Returns copies with ``kit_label`` set to the card string. At most one
    instance per kit class; ranked for AR-style primary-object selection
    (prefer small gripped fasteners over chassis).
    """
    rows = list(objects or [])
    mask_map = {str(key): value for key, value in (masks or {}).items()}
    hands = [row for row in rows if is_hand_class(str(row.get("class") or ""))]
    if not hands:
        return []

    hand_masks = []
    hand_boxes = []
    for row in hands:
        box = _as_box(row.get("box"))
        if box:
            hand_boxes.append(box)
        mask = mask_map.get(str(row.get("id")))
        if mask is not None:
            hand_masks.append(np.asarray(mask, dtype=bool))
    hand_union = _hand_union_mask(hand_masks)
    if hand_union is not None:
        hand_union = _dilate_mask(hand_union)
    frame_area = float(hand_union.size) if hand_union is not None else None
    usable_hands = []
    for box in hand_boxes:
        if frame_area and _box_area(box) > 0.22 * frame_area:
            continue
        usable_hands.append(box)

    scored: list[tuple[tuple[float, float, float], dict]] = []
    for row in rows:
        label = kit_label_for_detector(str(row.get("class") or ""))
        if not label:
            continue
        box = _as_box(row.get("box"))
        mask = mask_map.get(str(row.get("id")))
        contact = 0.0
        mask_contact = 0.0
        if hand_union is not None:
            if mask is not None:
                mask_contact = _mask_contact(
                    hand_union, np.asarray(mask, dtype=bool),
                )
                contact = max(contact, mask_contact)
            if box is not None:
                # Prefer real mask overlap; box-on-hand is a weak fallback only.
                box_c = _box_on_hand_mask(hand_union, box)
                if mask_contact <= 0:
                    contact = max(contact, box_c * 0.65)
                else:
                    contact = max(contact, box_c * 0.25)
        elif box is not None and usable_hands:
            contact = max(_box_contact(hand, box) for hand in usable_hands)
        if contact <= 0:
            continue
        item = dict(row)
        item["kit_label"] = label
        item["contact"] = round(contact, 4)
        item["ar_object"] = ar_object_for_kit(label)
        scored.append((_primary_rank(item, frame_area=frame_area), item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    ordered = [item for _, item in scored]
    # Resolve pink competition before ranking finalizes top-1.
    pink_rows = [row for row in ordered if row.get("kit_label") in {
        "nut", "round washer", "tooth washer", "acorn nut",
    }]
    if pink_rows:
        resolved_pink = _resolve_pink_lookalikes(pink_rows)
        # Solo acorn remap needs geometry; re-apply with frame_area.
        resolved_pink = [
            _relabel_false_acorn(row, frame_area=frame_area)
            if row.get("kit_label") == "acorn nut" else row
            for row in resolved_pink
        ]
        non_pink = [row for row in ordered if row.get("kit_label") not in {
            "nut", "round washer", "tooth washer", "acorn nut",
        }]
        ordered = resolved_pink + non_pink
        ordered.sort(
            key=lambda item: _primary_rank(item, frame_area=frame_area),
            reverse=True,
        )
    chosen: list[dict] = []
    seen: set[str] = set()
    limit = max(0, int(max_parts))
    for item in ordered:
        label = item["kit_label"]
        if label in seen:
            continue
        seen.add(label)
        chosen.append(item)
        if len(chosen) >= limit:
            break
    return chosen


def select_primary_ar_object(
    objects: list[dict],
    *,
    masks: dict[str, Any] | None = None,
) -> dict | None:
    """AR-style: at most one primary kit object for this frame, or None."""
    kept = select_hand_touched_parts(objects, masks=masks, max_parts=1)
    return kept[0] if kept else None


def allowed_detect_label(name: str) -> str | None:
    """15 kit card classes + hand. Paper and every other class are discarded."""
    if is_hand_class(name):
        return HAND_LABEL
    if _clean(name) == "paper":
        return None
    return kit_label_for_detector(name)


def filter_to_kit_and_hand(objects: list[dict]) -> list[dict]:
    """Keep only kit-card parts and hands; rewrite ``class`` to the card string."""
    output: list[dict] = []
    for row in objects or []:
        label = allowed_detect_label(str(row.get("class") or ""))
        if not label:
            continue
        item = dict(row)
        item["class"] = label
        item["kit_label"] = label
        output.append(item)
    return output


def _family_of(label: str) -> frozenset[str] | None:
    for group in LOOKALIKE_FAMILIES:
        if label in group:
            return group
    return None


def collapse_overlapping_lookalikes(objects: list[dict]) -> list[dict]:
    """One box per blob when SAM emits nut+washer (or similar) in the same frame."""
    ranked = sorted(
        objects or [],
        key=lambda row: float(row.get("score") or 0),
        reverse=True,
    )
    kept: list[dict] = []
    for row in ranked:
        label = str(row.get("kit_label") or row.get("class") or "")
        box = _as_box(row.get("box"))
        family = _family_of(label)
        drop = False
        for other in kept:
            other_label = str(other.get("kit_label") or other.get("class") or "")
            if label == HAND_LABEL or other_label == HAND_LABEL:
                continue
            other_box = _as_box(other.get("box"))
            if box is None or other_box is None:
                continue
            iou = _box_iou(box, other_box)
            same_family = family is not None and other_label in family
            if (same_family and iou >= _COLLAPSE_IOU) or (
                label != other_label and iou >= _SAME_BLOB_IOU
            ):
                drop = True
                break
        if not drop:
            kept.append(row)
    return kept


def _switch_delay(old: str, new: str, old_score: float, new_score: float) -> int:
    family = _family_of(old)
    lookalike = family is not None and new in family
    delay = _SWITCH_LOOKALIKE if lookalike else _SWITCH_OTHER
    if new_score >= old_score + _SCORE_MARGIN:
        delay = min(delay, _SWITCH_CONFIDENT)
    return delay


class KitLabelStabilizer:
    """IoU-track labels so a part does not blink nut→washer→nut across frames."""

    def __init__(
        self,
        *,
        match_iou: float = _MATCH_IOU,
        hold_miss: int = _HOLD_MISS,
    ) -> None:
        self.match_iou = float(match_iou)
        self.hold_miss = int(hold_miss)
        self._tracks: list[dict] = []
        self._next_id = 1

    def step(self, objects: list[dict]) -> list[dict]:
        rows = collapse_overlapping_lookalikes(filter_to_kit_and_hand(objects))
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks):
            tbox = _as_box(track["box"])
            if tbox is None:
                continue
            for di, row in enumerate(rows):
                box = _as_box(row.get("box"))
                if box is None:
                    continue
                hand_ok = (track["label"] == HAND_LABEL) == (
                    str(row.get("kit_label")) == HAND_LABEL
                )
                if not hand_ok:
                    continue
                iou = _box_iou(tbox, box)
                if iou >= self.match_iou:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        assignments: dict[int, int] = {}
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            assignments[ti] = di

        emitted: list[dict] = []
        alive: list[dict] = []
        for ti, track in enumerate(self._tracks):
            if ti in assignments:
                row = rows[assignments[ti]]
                proposed = str(row.get("kit_label") or track["label"])
                score = float(row.get("score") or 0)
                if proposed == track["label"]:
                    track["pending"] = None
                    track["pending_n"] = 0
                else:
                    if track.get("pending") != proposed:
                        track["pending"] = proposed
                        track["pending_n"] = 1
                    else:
                        track["pending_n"] = int(track["pending_n"]) + 1
                    need = _switch_delay(
                        track["label"], proposed, float(track["score"]), score,
                    )
                    if track["pending_n"] >= need:
                        track["label"] = proposed
                        track["pending"] = None
                        track["pending_n"] = 0
                track["box"] = list(row["box"])
                track["score"] = score
                track["det_id"] = row.get("id")
                track["miss"] = 0
                track["source"] = dict(row)
                alive.append(track)
                emitted.append(_emit_track(track, row))
            else:
                track["miss"] = int(track.get("miss") or 0) + 1
                if track["miss"] <= self.hold_miss:
                    held = _emit_track(track, track.get("source") or {})
                    held["held"] = True
                    emitted.append(held)
                    alive.append(track)

        for di, row in enumerate(rows):
            if di in used_dets:
                continue
            track = {
                "tid": self._next_id,
                "label": str(row.get("kit_label") or row.get("class")),
                "box": list(row.get("box") or []),
                "score": float(row.get("score") or 0),
                "det_id": row.get("id"),
                "pending": None,
                "pending_n": 0,
                "miss": 0,
                "source": dict(row),
            }
            self._next_id += 1
            alive.append(track)
            emitted.append(_emit_track(track, row))
        self._tracks = alive
        return emitted


def _emit_track(track: dict, row: dict) -> dict:
    item = dict(row)
    item["class"] = track["label"]
    item["kit_label"] = track["label"]
    item["track_id"] = track["tid"]
    item["score"] = track["score"]
    if track.get("box"):
        item["box"] = list(track["box"])
    if track.get("det_id") is not None:
        item["id"] = track["det_id"]
    return item


def stabilize_detection_records(records: list[dict]) -> list[dict]:
    """Rewrite each frame's objects to kit+hand labels with temporal lock."""
    stabilizer = KitLabelStabilizer()
    output = []
    for record in records or []:
        output.append({
            **record,
            "objects": stabilizer.step(list(record.get("objects") or [])),
        })
    return output


def hold_touched_parts(
    current: list[dict],
    previous: list[dict],
    ages: dict[str, int] | None = None,
    *,
    hold: int = _HOLD_TOUCH,
    max_parts: int = DEFAULT_MAX_PARTS,
) -> tuple[list[dict], dict[str, int]]:
    """Keep a touched label on-screen for a few frames if SAM misses once."""
    ages = dict(ages or {})
    by_label: dict[str, dict] = {}
    for row in current or []:
        label = str(row.get("kit_label") or "")
        if not label:
            continue
        by_label[label] = row
        ages[label] = 0
    for row in previous or []:
        label = str(row.get("kit_label") or "")
        if not label or label in by_label:
            continue
        ages[label] = int(ages.get(label) or 0) + 1
        if ages[label] <= hold and len(by_label) < max(0, int(max_parts)):
            held = dict(row)
            held["held"] = True
            by_label[label] = held
        elif ages[label] > hold:
            ages.pop(label, None)
    ages = {key: value for key, value in ages.items() if key in by_label}
    return list(by_label.values()), ages
