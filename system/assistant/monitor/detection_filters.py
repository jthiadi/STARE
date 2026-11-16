"""Post-SAM3 detection hygiene for IndustReal SOP classes.

Open-vocabulary models confuse lookalikes (wheel ↔ round washer, short pin ↔
screw pin, hex nut ↔ tooth washer). This module enforces physical size priors
and conflict rules so each mask gets one stable label.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _norm(name: str) -> str:
    return " ".join(str(name or "").strip().lower().replace("_", " ").split())


def _box_area_fraction(box: list[float] | None, frame_area: float) -> float | None:
    if not box or len(box) != 4 or frame_area <= 0:
        return None
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(0.0, float(box[3]) - float(box[1]))
    return (width * height) / frame_area


def _mask_area_fraction(mask: Any) -> float | None:
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.size == 0:
        return None
    return float(np.count_nonzero(array)) / float(array.size)


def class_family(name: str) -> str:
    text = _norm(name)
    if "paper" in text:
        return "paper"
    if text == "hand" or text.endswith(" hand"):
        return "hand"
    if "partial" in text and "model" in text:
        return "partial"
    if "pulley" in text:
        return "pulley"
    if "wheel" in text or "tire" in text:
        return "wheel"
    if "washer" in text:
        return "washer"
    if "nut" in text:
        return "nut"
    # Grey STEMFIE rods are pins (short pin / screw pin). Legacy "short screw"
    # labels from an earlier misname also count as pins.
    if ("screw" in text and "pin" not in text) or "pin" in text:
        return "pin"
    if "brace" in text or "beam" in text or text == "wing" or text.startswith("wing "):
        return "brace"
    if "bracket" in text:
        return "bracket"
    if "chassis" in text or "base" in text:
        return "chassis"
    if text in {"screwdriver", "wrench"}:
        return "tool"
    return "other"


# Allowed mask/box area as a fraction of the frame.
AREA_LIMITS: dict[str, tuple[float, float]] = {
    "hand": (0.001, 0.35),
    "paper": (0.01, 0.55),
    "partial": (0.004, 0.220),
    "nut": (0.00004, 0.0035),
    "washer": (0.00004, 0.0035),
    "pin": (0.00015, 0.060),
    "brace": (0.0004, 0.120),
    "bracket": (0.0003, 0.080),
    "pulley": (0.0003, 0.016),
    "wheel": (0.003, 0.220),
    "chassis": (0.002, 0.220),
    "tool": (0.0002, 0.080),
    "other": (0.00004, 0.180),
}


def passes_size_gate(row: dict, *, frame_area: float | None = None) -> bool:
    """Return False when the detection is the wrong scale for its class."""
    family = class_family(row.get("class", ""))
    minimum, maximum = AREA_LIMITS.get(family, AREA_LIMITS["other"])
    area = _mask_area_fraction(row.get("mask"))
    if area is None and frame_area:
        area = _box_area_fraction(row.get("box"), frame_area)
    if area is None:
        return True
    return minimum <= area <= maximum


def _mask_iou(a: Any, b: Any) -> float:
    left = np.asarray(a, dtype=bool)
    right = np.asarray(b, dtype=bool)
    if left.shape != right.shape:
        return 0.0
    inter = np.count_nonzero(left & right)
    if inter <= 0:
        return 0.0
    return float(inter) / float(max(1, np.count_nonzero(left | right)))


def _box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def _overlap(a: dict, b: dict) -> float:
    if a.get("mask") is not None and b.get("mask") is not None:
        return _mask_iou(a["mask"], b["mask"])
    if a.get("box") and b.get("box"):
        return _box_iou(a["box"], b["box"])
    return 0.0


def _containment(inner: dict, outer: dict) -> float:
    """Fraction of inner mask/box covered by outer."""
    if inner.get("mask") is not None and outer.get("mask") is not None:
        left = np.asarray(inner["mask"], dtype=bool)
        right = np.asarray(outer["mask"], dtype=bool)
        if left.shape != right.shape:
            return 0.0
        denom = np.count_nonzero(left)
        if denom <= 0:
            return 0.0
        return float(np.count_nonzero(left & right)) / float(denom)
    if inner.get("box") and outer.get("box"):
        a, b = inner["box"], outer["box"]
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        return inter / max(1e-9, area)
    return 0.0


def _box_extent(row: dict) -> tuple[float, float]:
    """Return (short_side, long_side) in pixels from box or mask."""
    box = row.get("box")
    if box and len(box) == 4:
        width = abs(float(box[2]) - float(box[0]))
        height = abs(float(box[3]) - float(box[1]))
        return (min(width, height), max(width, height))
    mask = row.get("mask")
    if mask is not None:
        ys, xs = np.where(np.asarray(mask, dtype=bool))
        if len(xs):
            width = float(xs.max() - xs.min() + 1)
            height = float(ys.max() - ys.min() + 1)
            return (min(width, height), max(width, height))
    return (0.0, 0.0)


def _aspect_ratio(row: dict) -> float:
    short, long = _box_extent(row)
    if short <= 0:
        return 1.0
    return long / short


def _frame_width(rows: list[dict]) -> float | None:
    for row in rows:
        mask = row.get("mask")
        if mask is not None:
            return float(np.asarray(mask).shape[1])
        box = row.get("box")
        if box:
            return max(float(box[2]), 1.0)
    return None


def pin_label_for_length(long_side: float, frame_w: float) -> str:
    """Map measured smooth-pin length to short / middle / long metal pin.

    Screw pin is a different part (threaded front-bracket axle) and must not be
    assigned from length alone — long smooth T-pins were previously mislabeled
    as screw pins.
    """
    ratio = float(long_side) / max(float(frame_w), 1e-6)
    if ratio >= 0.28:
        return "long metal pin"
    if ratio >= 0.16:
        return "middle metal pin"
    return "short metal pin"


def _is_screw_pin_label(name: str) -> bool:
    text = _norm(name)
    return text == "screw pin" or text.endswith(" screw pin")


def _smooth_pin_label(name: str, long_side: float, frame_w: float) -> str:
    """Length-bin T-pins. Keep screw pin only for a short rod SAM uniquely named."""
    # Short T-pins and the front-bracket screw are similar length; SAM's
    # "screw pin" prompt otherwise labels every grey rod. Length decides
    # short/middle/long. A unique short screw-pin hypothesis can stay.
    if _is_screw_pin_label(name) and long_side < 0.16 * frame_w:
        return "screw pin"
    return pin_label_for_length(long_side, frame_w)


def _mask_or_box_area(row: dict, frame_area: float | None) -> float | None:
    area = _mask_area_fraction(row.get("mask"))
    if area is None and frame_area:
        area = _box_area_fraction(row.get("box"), frame_area)
    return area


def reshape_lookalike_parts(
    rows: list[dict], *, frame_area: float | None = None
) -> list[dict]:
    """Fix common SAM3 lookalikes using shape priors (not class names alone).

    - Squat/fat 'pin' blobs → drop (diagrams, pile clutter, wrong family).
    - Wide elongated 'pins' → braces (perforated bars).
    - Oversized 'white front bracket' → partial model; tiny 'partial' → bracket.
    - Drop leftover generic tools that are not in the lean SOP.
    """
    if not rows:
        return []
    frame_w = _frame_width(rows)
    if not frame_w and frame_area:
        # Rough width from square-ish frame assumption when masks are absent.
        frame_w = float(frame_area) ** 0.5
    if not frame_w:
        return rows

    output: list[dict] = []
    for row in rows:
        name = _norm(row.get("class", ""))
        family = class_family(name)
        if family == "tool" or name in {"screwdriver", "wrench"}:
            continue
        short, long_side = _box_extent(row)
        aspect = long_side / max(short, 1.0)
        area = _mask_or_box_area(row, frame_area)
        updated = dict(row)
        updated.pop("ambiguous_classes", None)

        if family == "pin" or ("screw" in name and "pin" not in name):
            # Perforated brace bars are thick and long; grey pins are slender rods.
            if (
                long_side >= 0.20 * frame_w
                and short >= 0.048 * frame_w
                and aspect >= 2.0
            ):
                brace_cut = 0.25 * frame_w
                updated["class"] = "long brace" if long_side >= brace_cut else "short brace"
                output.append(updated)
                continue
            # Nearly square or chunky blobs are almost never STEMFIE pins.
            if aspect < 1.55:
                continue
            if aspect < 2.05 and short >= 0.048 * frame_w:
                continue
            updated["class"] = _smooth_pin_label(name, long_side, frame_w)
            output.append(updated)
            continue

        if family == "bracket" and area is not None and area >= 0.08:
            # Only remap huge chassis-sized blobs; keep real front brackets.
            updated["class"] = "partial model"
            output.append(updated)
            continue

        if family == "partial" and area is not None and area <= 0.012 and aspect >= 1.25:
            updated["class"] = "white front bracket"
            output.append(updated)
            continue

        if family == "pulley" and area is not None and area >= 0.012:
            # Tire-sized "pulley" hits are toy-car wheels.
            updated["class"] = "toy car wheel"
            output.append(updated)
            continue

        output.append(updated)
    return output


def disambiguate_length_pairs(rows: list[dict]) -> list[dict]:
    """Normalize brace / pin length labels; map misnamed grey 'screws' to pins."""
    if not rows:
        return []
    frame_w = _frame_width(rows)
    if not frame_w:
        return rows

    brace_cut = 0.25 * frame_w
    output = []
    for row in rows:
        name = _norm(row.get("class", ""))
        family = class_family(name)
        _, long_side = _box_extent(row)
        updated = dict(row)
        updated.pop("ambiguous_classes", None)
        if family == "brace":
            updated["class"] = "long brace" if long_side >= brace_cut else "short brace"
        elif family == "pin" or ("screw" in name and "pin" not in name):
            updated["class"] = _smooth_pin_label(name, long_side, frame_w)
        output.append(updated)
    return output


def resolve_class_conflicts(rows: list[dict]) -> list[dict]:
    """Drop fastener claims that are really wheels, pins, or brace ends."""
    if not rows:
        return []
    ranked = sorted(rows, key=lambda row: float(row.get("score") or 0), reverse=True)
    wheels = [row for row in ranked if class_family(row.get("class", "")) in {"wheel", "pulley"}]
    pins = [row for row in ranked if class_family(row.get("class", "")) == "pin"]
    braces = [row for row in ranked if class_family(row.get("class", "")) == "brace"]
    kept: list[dict] = []
    for row in ranked:
        family = class_family(row.get("class", ""))
        if family in {"washer", "nut"} and wheels:
            if any(_overlap(row, wheel) >= 0.20 or _containment(row, wheel) >= 0.50
                   for wheel in wheels):
                continue
        if family == "nut" and pins:
            if any(_overlap(row, pin) >= 0.15 or _containment(row, pin) >= 0.40
                   for pin in pins):
                continue
        if family in {"washer", "nut"} and braces:
            if any(_overlap(row, brace) >= 0.15 or _containment(row, brace) >= 0.40
                   for brace in braces):
                continue
        if family in {"washer", "nut"}:
            if any(
                class_family(other.get("class", "")) in {"wheel", "pin", "brace"}
                and (_overlap(row, other) >= 0.15 or _containment(row, other) >= 0.40)
                for other in kept
            ):
                continue
        if _norm(row.get("class", "")) == "toy car wheel":
            if any(
                "wheel assembly" in _norm(other.get("class", "")) and _overlap(row, other) >= 0.35
                for other in kept
            ):
                continue
        if family == "pin" and braces:
            if any(
                _aspect_ratio(brace) >= 3.0 and _overlap(row, brace) >= 0.35
                for brace in braces
            ):
                continue
        kept.append(row)
    return kept


def _pair_key(a: str, b: str) -> frozenset[str]:
    return frozenset({_norm(a), _norm(b)})


# Near-tie class pairs that should pick a winner instead of "ambiguous part".
_PIN_NAMES = (
    "short metal pin",
    "middle metal pin",
    "long metal pin",
    "screw pin",
)
RESOLVABLE_PAIRS: dict[frozenset[str], str] = {
    frozenset({a, b}): "length_pin"
    for index, a in enumerate(_PIN_NAMES)
    for b in _PIN_NAMES[index + 1:]
}
RESOLVABLE_PAIRS.update({
    frozenset({"hex nut", "tooth lock washer"}): "prefer_first",
    frozenset({"hex nut", "acorn nut"}): "prefer_first",
    frozenset({"acorn nut", "tooth lock washer"}): "prefer_first",
    frozenset({"toy car wheel", "front wheel assembly"}): "prefer_first",
    frozenset({"toy car wheel", "rear wheel assembly"}): "prefer_first",
    frozenset({"front wheel assembly", "rear wheel assembly"}): "prefer_first",
    frozenset({"toy car wheel", "pulley"}): "size_wheel_pulley",
})


def _is_assembly_lookalike(name: str) -> bool:
    family = class_family(name)
    return family in {"partial", "chassis", "brace", "bracket"}


def _pick_assembly_class(
    names: set[str],
    row: dict,
    *,
    frame_w: float | None,
    frame_area: float | None,
) -> str:
    """Choose one label for overlapping chassis / brace / bracket / partial masks."""
    cleaned = {_norm(name) for name in names if _norm(name)}
    area = _mask_or_box_area(row, frame_area) or 0.0
    short, long_side = _box_extent(row)
    aspect = long_side / max(short, 1.0)
    has_partial = any("partial" in name for name in cleaned)
    has_chassis = any(
        name in {"chassis base", "front chassis", "rear chassis"} or "chassis" in name
        for name in cleaned
    )
    has_brace = any("brace" in name for name in cleaned)
    has_bracket = any("bracket" in name for name in cleaned)

    # Large compact blobs are the assembled model / chassis, not a single brace.
    if has_partial and area >= 0.018 and aspect < 2.4:
        return "partial model"
    if has_partial and area >= 0.035:
        return "partial model"
    if has_chassis and area >= 0.018 and aspect < 2.4:
        for name in ("chassis base", "front chassis", "rear chassis"):
            if name in cleaned:
                return name
        return "chassis base"
    # Thin elongated bars keep brace labels.
    if has_brace and frame_w and long_side >= 0.18 * frame_w and aspect >= 2.2:
        brace_cut = 0.25 * frame_w
        return "long brace" if long_side >= brace_cut else "short brace"
    if has_bracket and area < 0.045:
        return "white front bracket" if "white front bracket" in cleaned else next(
            name for name in cleaned if "bracket" in name
        )
    if has_partial:
        return "partial model"
    if has_chassis:
        for name in ("chassis base", "front chassis", "rear chassis"):
            if name in cleaned:
                return name
    if has_brace:
        if frame_w:
            return "long brace" if long_side >= 0.25 * frame_w else "short brace"
        return "long brace" if "long brace" in cleaned else "short brace"
    if has_bracket:
        return "white front bracket" if "white front bracket" in cleaned else sorted(cleaned)[0]
    return sorted(cleaned)[0]


def _resolve_duplicate(
    kept: dict,
    challenger: dict,
    frame_w: float | None,
    *,
    frame_area: float | None = None,
) -> None:
    """Mutate kept to absorb challenger without leaving ambiguous_classes."""
    pair = _pair_key(str(kept.get("class")), str(challenger.get("class")))
    strategy = RESOLVABLE_PAIRS.get(pair)
    if strategy == "size_wheel_pulley":
        area = _mask_or_box_area(
            kept if kept.get("mask") is not None else challenger, frame_area
        ) or 0.0
        kept["class"] = "pulley" if area < 0.012 else "toy car wheel"
        kept["score"] = max(float(kept.get("score") or 0), float(challenger.get("score") or 0))
        kept.pop("ambiguous_classes", None)
        return
    if strategy == "length_pin" and frame_w:
        _, long_side = _box_extent(kept if kept.get("mask") is not None else challenger)
        if kept.get("mask") is None and challenger.get("mask") is not None:
            _, long_side = _box_extent(challenger)
        names = {_norm(kept.get("class")), _norm(challenger.get("class"))}
        # Competing pin classes on one mask → length bin. Do not let "screw pin"
        # win just because SAM scored that prompt higher on a T-pin.
        if "screw pin" in names and not (names & {
            "short metal pin", "middle metal pin", "long metal pin",
        }):
            label = (
                "screw pin" if long_side < 0.16 * frame_w
                else pin_label_for_length(long_side, frame_w)
            )
        else:
            label = pin_label_for_length(long_side, frame_w)
        kept["class"] = label
        kept["score"] = max(float(kept.get("score") or 0), float(challenger.get("score") or 0))
        kept.pop("ambiguous_classes", None)
        return
    if strategy == "prefer_first":
        # Keep the higher-scoring class (no ambiguous overlay).
        if float(challenger.get("score") or 0) > float(kept.get("score") or 0):
            kept["class"] = challenger["class"]
            kept["score"] = challenger["score"]
            if challenger.get("mask") is not None:
                kept["mask"] = challenger["mask"]
            if challenger.get("box") is not None:
                kept["box"] = challenger["box"]
        kept.pop("ambiguous_classes", None)
        return

    # Chassis / brace / bracket / partial model often fire on the same mask.
    names = set(kept.get("ambiguous_classes") or [kept.get("class")]) | {challenger.get("class")}
    names = {name for name in names if name}
    if any(_is_assembly_lookalike(str(name)) for name in names) and len(names) >= 2:
        kept_score = float(kept.get("score") or 0)
        challenger_score = float(challenger.get("score") or 0)
        source = challenger if (
            challenger.get("mask") is not None and kept.get("mask") is None
        ) else kept
        kept["class"] = _pick_assembly_class(
            names, source, frame_w=frame_w, frame_area=frame_area,
        )
        kept["score"] = max(kept_score, challenger_score)
        if challenger_score > kept_score:
            if challenger.get("mask") is not None:
                kept["mask"] = challenger["mask"]
            if challenger.get("box") is not None:
                kept["box"] = challenger["box"]
        kept.pop("ambiguous_classes", None)
        return

    # Default: near-tie across unrelated classes stays ambiguous.
    if float(kept.get("score") or 0) - float(challenger.get("score") or 0) < 0.12:
        kept["ambiguous_classes"] = sorted(set(
            kept.get("ambiguous_classes", [kept["class"]]) + [challenger["class"]]))


def suppress_duplicates(rows: list[dict], *, frame_area: float | None = None) -> list[dict]:
    """Same-mask de-dupe; resolvable pairs pick one label, others may stay ambiguous."""
    paper_masks = [
        row["mask"] for row in rows
        if _norm(row.get("class", "")) == "paper" and row.get("mask") is not None
    ]
    filtered = rows
    if paper_masks:
        paper = np.logical_or.reduce(paper_masks)
        filtered = []
        for row in rows:
            if _norm(row.get("class", "")) == "paper":
                continue
            mask = row.get("mask")
            if mask is None:
                filtered.append(row)
                continue
            # Printed diagrams false-fire as real parts; suppress aggressively.
            if np.count_nonzero(np.asarray(mask) & paper) / max(1, np.count_nonzero(mask)) >= 0.45:
                continue
            filtered.append(row)

    if frame_area is None:
        for row in filtered:
            mask = row.get("mask")
            if mask is not None:
                frame_area = float(np.asarray(mask).size)
                break

    frame_w = _frame_width(filtered)
    kept: list[dict] = []
    for row in sorted(filtered, key=lambda item: float(item.get("score") or 0), reverse=True):
        duplicate = next((other for other in kept if _overlap(row, other) > 0.65), None)
        if duplicate is None:
            kept.append(row)
            continue
        if duplicate.get("class") == row.get("class"):
            duplicate["score"] = max(float(duplicate.get("score") or 0), float(row.get("score") or 0))
            duplicate.pop("ambiguous_classes", None)
            continue
        _resolve_duplicate(duplicate, row, frame_w, frame_area=frame_area)
    return kept


def refine_detections(rows: list[dict], *, frame_area: float | None = None) -> list[dict]:
    """Full hygiene pipeline applied after SAM3 proposals."""
    # Reshape before size gates so pin→brace relabels use brace area limits.
    reshaped = reshape_lookalike_parts(rows, frame_area=frame_area)
    sized = [row for row in reshaped if passes_size_gate(row, frame_area=frame_area)]
    lengthed = disambiguate_length_pairs(sized)
    resolved = resolve_class_conflicts(lengthed)
    return suppress_duplicates(resolved, frame_area=frame_area)
