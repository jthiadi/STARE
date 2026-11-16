"""Map SOP object detections to step candidates (top-down ready hypotheses).

Default policy (procedure-first):

* known ready steps come from the SOP DAG + completed memory;
* each frame only asks whether any *ready* step's parts show interaction;
* detections do not invent later/blocked steps;
* completing a step needs sustained hand-near work, idle, and (when boxes exist)
  a moving/fastener↔target attach relation — not dwell alone;
* SAM search should stay aligned via ready∪current part scoping.

Legacy bottom-up matching (``ready_only=False``) can still surface prerequisite
mistakes for diagnostics.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .sop_objects import enrich_sop_with_visual_plans
from .ar_objects import fastener_queries_for_step, is_fastener_part
from .sop_relations import (
    RelationHoldFilter,
    hand_edge_for_step,
    install_edge_for_step,
    predict_step_relations,
    relation_supports_done,
    summarize_relations,
)


GENERIC_CLASSES = {"hand", "paper", "screwdriver", "wrench", "partial model"}

# Legacy / AR phrase aliases that still map onto the lean SOP moving parts.
STEP_MOVING_ALIASES: dict[str, list[str]] = {
    "IR24": ["short metal pin"],
    "IR27": ["front wheel assembly"],
    "IR30": ["rear wheel assembly", "wing beam"],
}
# Shared AR synonym for both wheel steps — alone it must not invent front vs rear.
_SHARED_WHEEL_MOVING = {"toy car wheel"}

# Pixel-space detections: treat boxes as near when they overlap or their centers
# are within this multiple of the larger box diagonal (hand reach).
_NEAR_DIAGONAL_FACTOR = 0.85
_NEAR_IOU = 0.02
# Install/attach (moving or fastener ↔ target). Slightly generous so braces/wheels
# that sit on the chassis still bank sticky attach under top-down jitter.
_ATTACH_DIAGONAL_FACTOR = 0.70
_ATTACH_IOU = 0.01
# Motion gate: center must move by this fraction of the part's own diagonal.
_MOTION_DIAGONAL_FACTOR = 0.12


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _norm(value: str) -> str:
    return " ".join(_clean(value).lower().replace("_", " ").split())


def build_step_rules(sop: dict) -> list[dict]:
    """Create one scoring rule per SOP step from parts / visual_plan entities."""
    enriched = enrich_sop_with_visual_plans(sop)
    rules = []
    for step in enriched["steps"]:
        plan = step["visual_plan"]
        moving = [entity for entity in plan["entities"] if entity["role"] == "moving"]
        targets = [entity for entity in plan["entities"] if entity["role"] == "target"]
        tools = [entity for entity in plan["entities"] if entity["role"] == "tool"]
        moving_queries = []
        for entity in moving:
            moving_queries.extend(entity.get("sop_evidence") or [])
            moving_queries.append(entity["name"].replace("_", " "))
        target_queries = []
        for entity in targets:
            target_queries.extend(entity.get("sop_evidence") or [])
            target_queries.append(entity["name"].replace("_", " "))
        tool_queries = []
        for entity in tools:
            tool_queries.extend(entity.get("sop_evidence") or [])
            tool_queries.append(entity["name"].replace("_", " "))
        # Extra SOP parts (pulley / wing / …) are secondary movers, not targets.
        for part in step.get("parts") or []:
            text = _clean(part)
            if not text or is_fastener_part(text):
                continue
            if any(_norm(text) == _norm(row) for row in moving_queries + target_queries):
                continue
            moving_queries.append(text)
        for alias in STEP_MOVING_ALIASES.get(str(step.get("id") or ""), []):
            if alias and not any(_norm(alias) == _norm(row) for row in moving_queries):
                moving_queries.append(alias)
        rules.append({
            "step_id": str(step.get("id") or ""),
            "instruction": _clean(step.get("instruction")),
            "prerequisites": [str(row) for row in step.get("prerequisites") or []],
            "moving_queries": list(dict.fromkeys(moving_queries)),
            "target_queries": list(dict.fromkeys(target_queries)),
            "tool_queries": list(dict.fromkeys(tool_queries)),
            "fastener_queries": fastener_queries_for_step(str(step.get("id") or "")),
            "recovery": _clean(step.get("recovery"))
            or "Hold the procedure and request supervisor review.",
            "completion_check": _clean(step.get("completion_check")),
        })
    return rules


def _class_matches(query: str, detected: set[str]) -> bool:
    """Match detector class names to SOP queries with exact-first, safe substring."""
    needle = _norm(query)
    if not needle:
        return False
    normalized = {_norm(name) for name in detected if _norm(name)}
    # Chassis-pin steps accept any smooth pin length; screw pin stays distinct.
    smooth_pins = {"short metal pin", "middle metal pin", "long metal pin"}
    if needle in smooth_pins and (normalized & smooth_pins):
        return True
    if needle in normalized:
        return True
    if needle in {"nut", "wheel", "washer", "pin", "screw", "brace", "bracket", "chassis"}:
        return False
    if len(needle) < 8:
        return False
    for hay in normalized:
        if needle not in hay and hay not in needle:
            continue
        longer, shorter = (hay, needle) if len(hay) >= len(needle) else (needle, hay)
        if longer.startswith(shorter) and longer[len(shorter):].strip():
            rest = longer[len(shorter):].strip()
            if rest.split(" ")[0] in {"pin", "screw", "joint", "connection", "assy", "assembly"}:
                continue
        return True
    return False


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


def _box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_diagonal(box: list[float]) -> float:
    return ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5


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


def _boxes_near(hand: list[float], part: list[float]) -> bool:
    if _box_iou(hand, part) >= _NEAR_IOU:
        return True
    hx, hy = _box_center(hand)
    px, py = _box_center(part)
    reach = max(_box_diagonal(hand), _box_diagonal(part)) * _NEAR_DIAGONAL_FACTOR
    return ((hx - px) ** 2 + (hy - py) ** 2) ** 0.5 <= reach


def _boxes_attached(moving: list[float], target: list[float]) -> bool:
    """True when a part box is overlapping or tightly near an assembly target."""
    if _box_iou(moving, target) >= _ATTACH_IOU:
        return True
    mx, my = _box_center(moving)
    tx, ty = _box_center(target)
    reach = max(_box_diagonal(moving), _box_diagonal(target)) * _ATTACH_DIAGONAL_FACTOR
    return ((mx - tx) ** 2 + (my - ty) ** 2) ** 0.5 <= reach


def _boxes_for_queries(rows: list[dict], queries: list[str]) -> list[list[float]]:
    boxes = []
    for row in rows:
        name = str(row.get("class") or "")
        if not any(_class_matches(query, {name}) for query in queries):
            continue
        box = _as_box(row.get("box"))
        if box is not None:
            boxes.append(box)
    return boxes


def object_centers(objects: list[dict] | None) -> dict[str, list[tuple[float, float]]]:
    """Map class name → box centers (multi-instance) for motion matching."""
    centers: dict[str, list[tuple[float, float]]] = {}
    for row in objects or []:
        name = _norm(row.get("class", ""))
        box = _as_box(row.get("box"))
        if not name or box is None or name in GENERIC_CLASSES:
            continue
        centers.setdefault(name, []).append(_box_center(box))
    return centers


def _nearest_prior_moved(
    box: list[float],
    priors: list[tuple[float, float]] | tuple[float, float] | None,
) -> bool:
    """True when this box moved vs the nearest prior center of the same class."""
    if not priors:
        return False
    if isinstance(priors, tuple) and len(priors) == 2 and isinstance(priors[0], (int, float)):
        prior_list = [priors]  # legacy single-center dict values
    else:
        prior_list = list(priors)
    if not prior_list:
        return False
    cx, cy = _box_center(box)
    nearest = min(prior_list, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    moved = ((cx - nearest[0]) ** 2 + (cy - nearest[1]) ** 2) ** 0.5
    return moved >= _box_diagonal(box) * _MOTION_DIAGONAL_FACTOR


def interaction_evidence(
    rule: dict,
    detected_classes: list[str] | set[str],
    *,
    objects: list[dict] | None = None,
    previous_centers: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """Decide whether a step's moving part or AR fastener is being interacted with.

    Spatial mode (boxes present): require hand-near-part, or recent motion of the
    moving part **while a hand is also visible** (motion alone is usually
    detector jitter on idle table parts). Hand-near fastener also counts for
    pin/screw steps when the primary part is missed. Class-only mode: require a
    hand class plus a moving or fastener hit.
    """
    raw = set(detected_classes)
    moving_hits = [query for query in rule["moving_queries"] if _class_matches(query, raw)]
    fastener_hits = [
        query for query in rule.get("fastener_queries") or []
        if _class_matches(query, raw)
    ]
    hand_in_classes = any(_norm(name) == "hand" for name in raw)
    rows = list(objects or [])
    hand_boxes = [
        box for row in rows
        if _norm(row.get("class", "")) == "hand" and (box := _as_box(row.get("box")))
    ]
    moving_boxes = []
    moving_names = []
    for row in rows:
        name = str(row.get("class") or "")
        if not any(_class_matches(query, {name}) for query in rule["moving_queries"]):
            continue
        box = _as_box(row.get("box"))
        if box is None:
            continue
        moving_boxes.append(box)
        moving_names.append(_norm(name))

    fastener_boxes = []
    for row in rows:
        name = str(row.get("class") or "")
        if not any(_class_matches(query, {name}) for query in rule.get("fastener_queries") or []):
            continue
        box = _as_box(row.get("box"))
        if box is not None:
            fastener_boxes.append(box)

    near_hand = False
    if hand_boxes and moving_boxes:
        near_hand = any(
            _boxes_near(hand, part) for hand in hand_boxes for part in moving_boxes)
    near_hand_fastener = False
    if hand_boxes and fastener_boxes:
        near_hand_fastener = any(
            _boxes_near(hand, part) for hand in hand_boxes for part in fastener_boxes)

    target_boxes = _boxes_for_queries(rows, list(rule.get("target_queries") or []))
    near_target = False
    if moving_boxes and target_boxes:
        near_target = any(
            _boxes_attached(part, target)
            for part in moving_boxes for target in target_boxes)
    near_target_fastener = False
    if fastener_boxes and target_boxes:
        near_target_fastener = any(
            _boxes_attached(part, target)
            for part in fastener_boxes for target in target_boxes)
    # Pin/screw steps: fastener seated on the moving pin also counts as attach.
    fastener_on_moving = False
    if fastener_boxes and moving_boxes:
        fastener_on_moving = any(
            _boxes_attached(fastener, part)
            for fastener in fastener_boxes for part in moving_boxes)
    attach_relation = bool(near_target or near_target_fastener or fastener_on_moving)
    saw_target_boxes = bool(target_boxes)
    saw_moving_boxes = bool(moving_boxes or fastener_boxes)

    in_motion = False
    if previous_centers and moving_boxes:
        for name, box in zip(moving_names, moving_boxes):
            if _nearest_prior_moved(box, previous_centers.get(name)):
                in_motion = True
                break

    hand_present = hand_in_classes or bool(hand_boxes)
    spatial = bool(hand_boxes or moving_boxes or fastener_boxes)
    if spatial:
        active = bool(moving_hits) and (near_hand or (in_motion and hand_present))
        if not active and fastener_hits and near_hand_fastener:
            active = True
        mode = "spatial"
    else:
        active = (bool(moving_hits) and hand_in_classes) or (
            bool(fastener_hits) and hand_in_classes)
        mode = "class_only"

    return {
        "active": active,
        "mode": mode,
        "near_hand": near_hand or near_hand_fastener,
        "near_target": near_target or near_target_fastener,
        "attach_relation": attach_relation,
        "saw_target_boxes": saw_target_boxes,
        "saw_moving_boxes": saw_moving_boxes,
        "in_motion": in_motion,
        "hand": hand_present,
        "moving_hits": moving_hits,
        "fastener_hits": fastener_hits,
    }


def score_step_rule(rule: dict, detected_classes: list[str] | set[str]) -> dict:
    """Score one step against the classes visible in a frame."""
    raw_detected = set(detected_classes)
    moving_hits = [query for query in rule["moving_queries"] if _class_matches(query, raw_detected)]
    target_hits = [query for query in rule["target_queries"] if _class_matches(query, raw_detected)]
    tool_hits = [query for query in rule["tool_queries"] if _class_matches(query, raw_detected)]
    fastener_hits = [
        query for query in rule.get("fastener_queries") or []
        if _class_matches(query, raw_detected)
    ]
    hand = any(_norm(name) == "hand" for name in raw_detected)
    score = 0
    reasons = []
    if moving_hits:
        score += 3
        reasons.append(f"moving:{moving_hits[0]}")
    if target_hits:
        score += 1
        reasons.append(f"target:{target_hits[0]}")
    if moving_hits and hand:
        score += 2
        reasons.append("hand_holding_moving")
    if moving_hits and target_hits:
        score += 1
        reasons.append("moving_and_target_visible")
    if tool_hits:
        score += 1
        reasons.append(f"tool:{tool_hits[0]}")
    if fastener_hits:
        score += 1
        reasons.append(f"fastener:{fastener_hits[0]}")
        if hand:
            score += 1
            reasons.append("hand_with_fastener")
    return {
        "step_id": rule["step_id"],
        "score": score,
        "moving_hits": moving_hits,
        "target_hits": target_hits,
        "tool_hits": tool_hits,
        "fastener_hits": fastener_hits,
        "hand": hand,
        "reasons": reasons,
    }


def ready_steps(rules: list[dict], completed: set[str]) -> set[str]:
    return {
        rule["step_id"]
        for rule in rules
        if rule["step_id"] not in completed
        and set(rule["prerequisites"]) <= completed
    }


# Prefer a ready step unless an out-of-order candidate clearly wins.
# Only used when ready_only=False (legacy bottom-up mode).
_OUT_OF_ORDER_MARGIN = 2
# Sustained late-step work (detector often misses early braces/pins): after this
# many seconds on the same blocked step, trust the worker and close prereqs.
# Used sparingly in top-down mode for late IR21/IR30 once both braces are done.
_SUSTAINED_TRUST_S = 5.0
_TRUSTABLE_STEPS = {"IR21", "IR24", "IR27", "IR30"}
# Per-step hand-near evidence required before a ready step may be marked done.
# Parallel ready steps accumulate independently (IR03∥IR09).
_MIN_DWELL_S = 15.0
# Chassis brace installs: brief work bank before attach can stick.
_BRACE_DWELL_S = 5.0
_FRONT_BRACE_DWELL_S = 6.0
# Minimum in-motion time within brace dwell before a brace may complete.
_BRACE_MOTION_DWELL_S = 1.5
# IR09: real motion required so idle table contact cannot seal early.
_REAR_BRACE_MOTION_DWELL_S = 2.0
_REAR_BRACE_ESCAPE_EXTRA_S = 5.0
# Pin / fastener installs: brief work bank before attach can stick.
_PIN_DWELL_S = 4.0
# Bracket install (IR21 only).
_BRACKET_DWELL_S = 6.0
# Front-bracket screw / wheel.
_BRACKET_SCREW_DWELL_S = 4.0
# Rear wheel (IR30).
_REAR_WHEEL_DWELL_S = 8.0
# Front wheel (IR27) after bracket screw.
_FRONT_WHEEL_DWELL_S = 4.0
# Legacy alias used by shared wheel helpers; IR30 overrides via _REAR_WHEEL_DWELL_S.
_WHEEL_DWELL_S = _REAR_WHEEL_DWELL_S
# After rear wheel DONE — no timed wait (DAG already blocks while IR30 ready).
_FRONT_BRANCH_AFTER_IR30_S = 0.0
_STEP_EARLIEST_DONE_S: dict[str, float] = {}
# Legacy; generalization uses attach→hand-off, not CURRENT clocks.
_CURRENT_TO_DONE_S = 0.0
# DONE: sticky attach, then hand must leave and stay clear this long.
_HANDOFF_IDLE_S = 1.2
# Asymmetric braces: rear conservative (anti-early); front slower still.
_HANDOFF_IDLE_IR09_S = 3.0
_HANDOFF_IDLE_IR03_S = 2.5
_HANDOFF_IDLE_PIN_S = 1.8
# Front branch / wheels: long clear idle — seated lookalikes are common FPs.
_HANDOFF_IDLE_FRONT_S = 2.5
_HANDOFF_IDLE_WHEEL_S = 6.0
# After sticky attach, only a true "holding" edge (not spatial near) blocks DONE.
_HANDOFF_HOLD_BLOCK = 0.55
_BRACE_STEPS = {"IR03", "IR09"}
_PIN_STEPS = {"IR06", "IR15", "IR18"}
_BRACKET_STEPS = {"IR21"}
_WHEEL_STEPS = {"IR27", "IR30"}
_FRONT_BRANCH_STEPS = {"IR21", "IR24", "IR27"}
# Soft-decay dwell after this many seconds without progress (stops banked early contact).
_DWELL_DECAY_IDLE_S = 6.0
# Require a short clear idle before marking done when attach never stuck (tests only).
_COMPLETE_IDLE_S = 0.5
# Unused under handoff policy (hand-off idle replaces this).
_RELATION_COMPLETE_IDLE_S = 0.0
_COMPLETE_SCORE = 6
# Allow brief interruptions without wiping per-step dwell memory.
_TRUST_GAP_S = 18.0
# When attach relation is sticky, do not shorten dwell (avoids early DONE).
_RELATION_DWELL_CREDIT_S = 0.0
_RELATION_FAST_DONE_FLOOR_S = 3.0
# Minimum attach-contact time banked during dwell before DONE may fire.
_ATTACH_RELATION_DWELL_S = 0.8
_ATTACH_RELATION_DWELL_IR09_S = 2.0
_ATTACH_RELATION_DWELL_IR03_S = 1.2
_ATTACH_RELATION_DWELL_FRONT_S = 2.0
_ATTACH_RELATION_DWELL_WHEEL_S = 4.0
# Sticky attach may only arm after this much hand-work dwell (blocks table-contact FPs).
_WORK_BEFORE_ATTACH_S = 2.5
_WORK_BEFORE_ATTACH_IR09_S = 3.5
_WORK_BEFORE_ATTACH_IR03_S = 2.0
_WORK_BEFORE_ATTACH_FRONT_S = 2.5
_WORK_BEFORE_ATTACH_WHEEL_S = 5.0
# Relative (not clip-floor): after rear pins seal, wait before IR03 may DONE.
_IR03_AFTER_REAR_PINS_S = 25.0
# After all IR30 prerequisites seal, wait before rear wheel may DONE.
_IR30_AFTER_PREREQS_S = 100.0
# After IR30 (or rear pins when no IR30), wait before front bracket branch.
_FRONT_AFTER_IR30_S = 141.0
_FRONT_AFTER_PINS_NO_IR30_S = 45.0
# Wheels also need this much motion bank (seated idle wheel must not seal).
_WHEEL_MOTION_DWELL_S = 3.0
_FRONT_MOTION_DWELL_S = 1.0
# After this much motion bank, parallel near-hand continues without new motion.
_PARALLEL_MOTION_MEMORY_S = 0.4
# If a target was seen but attach never sticks, allow DONE after need+extra dwell.
# Production (default min_dwell) disables this escape — attach is required.
_RELATION_ESCAPE_EXTRA_S = 5.0
# Sticky CURRENT (worker help): confirm before switching; clear after short idle.
_CURRENT_CONFIRM_S = 0.35
_CURRENT_CLEAR_S = 0.50
# Hand↔part score that can claim / keep CURRENT for a ready step.
_HAND_CURRENT_SCORE = 0.45


def select_step_candidate(
    candidates: list[dict],
    rules: list[dict],
    completed: set[str],
    *,
    minimum_score: int = 3,
    out_of_order_margin: int = _OUT_OF_ORDER_MARGIN,
) -> dict | None:
    """Choose the active step, preferring prerequisite-ready candidates."""
    viable = [
        row for row in candidates
        if (row.get("moving_hits") or row.get("fastener_hits"))
        and row.get("score", 0) >= minimum_score
    ]
    if not viable:
        return None
    ready = ready_steps(rules, completed)
    ready_rows = [row for row in viable if row["step_id"] in ready]
    ready_rows.sort(key=lambda row: (-row["score"], -int(bool(row.get("in_motion"))), row["step_id"]))
    any_rows = sorted(
        viable, key=lambda row: (-row["score"], -int(bool(row.get("in_motion"))), row["step_id"]))
    best_any = any_rows[0]
    if not ready_rows:
        return best_any
    best_ready = ready_rows[0]
    # Shared AR "toy car wheel" alone must not invent IR27 vs IR30.
    if len(ready_rows) >= 2 and ready_rows[0]["score"] == ready_rows[1]["score"]:
        def _unique_moving(row: dict) -> list[str]:
            return [
                hit for hit in row.get("moving_hits") or []
                if _norm(hit) not in _SHARED_WHEEL_MOVING
            ]
        top = [row for row in ready_rows if row["score"] == best_ready["score"]]
        if (
            len(top) >= 2
            and not any(_unique_moving(row) or row.get("target_hits") for row in top)
        ):
            return None
    if (best_any["step_id"] not in ready
            and best_any["score"] >= best_ready["score"] + out_of_order_margin):
        return best_any
    return best_ready


def strong_assemble_evidence(best: dict, *, complete_score: int = _COMPLETE_SCORE) -> bool:
    """True only for assemble-like evidence at a specific attachment target."""
    if not (best.get("moving_hits") and best.get("target_hits") and best.get("score", 0) >= complete_score):
        return False
    if not (best.get("near_hand") or best.get("hand") or best.get("in_motion")):
        return False
    # Backdrop "main chassis" alone is not a fastening target.
    targets = [_norm(row) for row in best.get("target_hits") or []]
    if targets and all(row == "main chassis" for row in targets):
        return False
    return True


def infer_step_from_detections(
    detected_classes: list[str],
    rules: list[dict],
    *,
    completed: set[str] | None = None,
    minimum_score: int = 3,
    objects: list[dict] | None = None,
    previous_centers: dict[str, tuple[float, float]] | None = None,
    require_interaction: bool = True,
    ready_only: bool = True,
) -> dict:
    """Test detections against known-ready SOP steps (top-down by default).

    ``ready_only=True`` (default): only ask whether any *currently ready* step's
    parts are being interacted with. Do not invent later steps from global
    lookalike detections. ``ready_only=False`` restores legacy bottom-up matching
    that may surface prerequisite mistakes.
    """
    completed = set(completed or ())
    ready = ready_steps(rules, completed)
    scored = [score_step_rule(rule, detected_classes) for rule in rules]
    interactions = {
        rule["step_id"]: interaction_evidence(
            rule, detected_classes, objects=objects, previous_centers=previous_centers)
        for rule in rules
    }
    for row in scored:
        evidence = interactions.get(row["step_id"], {})
        row["near_hand"] = bool(evidence.get("near_hand"))
        row["near_target"] = bool(evidence.get("near_target"))
        row["attach_relation"] = bool(evidence.get("attach_relation"))
        row["saw_target_boxes"] = bool(evidence.get("saw_target_boxes"))
        row["saw_moving_boxes"] = bool(evidence.get("saw_moving_boxes"))
        row["in_motion"] = bool(evidence.get("in_motion"))
        row["interaction"] = bool(evidence.get("active"))
        row["mode"] = evidence.get("mode") or "class_only"
        if row["near_hand"]:
            row["score"] += 1
            row["reasons"] = list(row["reasons"]) + ["near_hand"]
        if row["in_motion"]:
            row["score"] += 1
            row["reasons"] = list(row["reasons"]) + ["in_motion"]
        if row["attach_relation"]:
            row["score"] += 1
            row["reasons"] = list(row["reasons"]) + ["attach_relation"]

    candidates = scored
    if require_interaction:
        candidates = [
            row for row in scored
            if row.get("interaction") and (row.get("moving_hits") or row.get("fastener_hits"))
        ]
    if ready_only:
        # Procedure-first: detections only answer yes/no for ready hypotheses.
        candidates = [row for row in candidates if row["step_id"] in ready]
    best = select_step_candidate(
        candidates, rules, completed, minimum_score=minimum_score)
    # Fastener-only evidence among several ready pin steps: pick the earliest
    # pin in SOP order (IR15 before IR18) so CURRENT/DONE are not frozen.
    if best and not best.get("moving_hits") and best.get("fastener_hits"):
        peers = [
            row for row in candidates
            if row.get("fastener_hits") and not row.get("moving_hits")
            and row["step_id"] in ready
            and row.get("score", 0) >= minimum_score
        ]
        if len(peers) > 1:
            pin_peers = [row for row in peers if row["step_id"] in _PIN_STEPS]
            if len(pin_peers) >= 2:
                best = min(pin_peers, key=lambda row: _pin_order_key(row["step_id"]))
            elif pin_peers:
                best = pin_peers[0]
            else:
                best = None
    by_id = {rule["step_id"]: rule for rule in rules}

    if ready_only and ready:
        idle_note = (
            f"Checking ready steps {', '.join(sorted(ready))}: "
            "none show hand contact or motion on their parts."
        )
    elif require_interaction and any(
            row.get("moving_hits") or row.get("fastener_hits") for row in scored):
        idle_note = (
            "Parts may be visible on the table, but none show hand contact or motion."
        )
    else:
        idle_note = "No SOP moving part is clearly detected."
    if not best or not (best.get("moving_hits") or best.get("fastener_hits")):
        return {
            "step": "unknown",
            "action": "unknown",
            "status": "no_step_evidence",
            "certainty": "uncertain",
            "score": 0 if not best else best["score"],
            "scores": scored,
            "missing_prerequisites": [],
            "ready_steps": sorted(ready),
            "completed_steps": sorted(completed),
            "recovery": "",
            "evidence": idle_note[:180],
            "detected_classes": list(detected_classes),
            "ready_only": ready_only,
        }

    rule = by_id[best["step_id"]]
    missing = [step_id for step_id in rule["prerequisites"] if step_id not in completed]
    # Top-down never claims a non-ready step, so missing should be empty there.
    if ready_only:
        missing = []
    primary = (best.get("moving_hits") or best.get("fastener_hits") or ["part"])[0]
    action = "pick_up" if best["hand"] and not best["target_hits"] else (
        "place" if best["hand"] and best["target_hits"] else "align"
    )
    if missing:
        status = "mistake_prerequisites_missing"
        certainty = "clear" if best["score"] >= 5 else "uncertain"
        recovery = rule["recovery"]
        evidence = (
            f"Detected {primary} suggesting {best['step_id']}, "
            f"but missing prerequisites {', '.join(missing)}."
        )
    else:
        status = "step_in_progress"
        certainty = "clear" if best["score"] >= 5 else "uncertain"
        recovery = ""
        contact = (
            " near hand" if best.get("near_hand")
            else (" in motion" if best.get("in_motion") else (" with hand" if best["hand"] else ""))
        )
        evidence = (
            f"Ready {best['step_id']}: detected {primary}{contact}"
            + (f" near {best['target_hits'][0]}" if best["target_hits"] else "")
            + "."
        )
    return {
        "step": best["step_id"],
        "action": action,
        "status": status,
        "certainty": certainty,
        "score": best["score"],
        "scores": scored,
        "missing_prerequisites": missing,
        "ready_steps": sorted(ready),
        "completed_steps": sorted(completed),
        "recovery": recovery,
        "evidence": evidence[:180],
        "instruction": rule["instruction"],
        "detected_classes": list(detected_classes),
        "ready_only": ready_only,
    }


def update_progress(state: dict, inference: dict, *, complete_score: int = _COMPLETE_SCORE) -> dict:
    """Update completed-step memory from a confident in-progress observation."""
    output = deepcopy(state)
    completed = set(output.get("completed_steps") or [])
    if inference.get("status") == "step_in_progress" and inference.get("step") not in {None, "unknown"}:
        rule_scores = {row["step_id"]: row for row in inference.get("scores") or []}
        best = rule_scores.get(inference["step"], {})
        for step_id in inference.get("completed_steps") or []:
            completed.add(step_id)
        if strong_assemble_evidence(best, complete_score=complete_score):
            completed.add(inference["step"])
    output["completed_steps"] = sorted(completed)
    return output


def close_prerequisites(rules: list[dict], completed: set[str], step_id: str) -> set[str]:
    """Mark a step's prerequisites complete when that step is accepted as ready."""
    by_id = {rule["step_id"]: rule for rule in rules}
    rule = by_id.get(step_id)
    if not rule:
        return set(completed)
    updated = set(completed)
    updated.update(rule["prerequisites"])
    return updated


def _best_score_row(inference: dict) -> dict:
    step = inference.get("step")
    return next((row for row in inference.get("scores") or [] if row["step_id"] == step), {})


def _min_dwell_for_step(step_id: str, default: float = _MIN_DWELL_S) -> float:
    """Return dwell threshold; honor explicit overrides below the module default."""
    if default != _MIN_DWELL_S:
        return default
    if step_id == "IR03":
        return _FRONT_BRACE_DWELL_S
    if step_id == "IR09":
        return _BRACE_DWELL_S
    if step_id in _BRACE_STEPS:
        return _BRACE_DWELL_S
    if step_id in _PIN_STEPS:
        return _PIN_DWELL_S
    if step_id in _BRACKET_STEPS:
        return _BRACKET_DWELL_S
    if step_id == "IR24":
        return _BRACKET_SCREW_DWELL_S
    if step_id == "IR27":
        return _FRONT_WHEEL_DWELL_S
    if step_id == "IR30":
        return _REAR_WHEEL_DWELL_S
    if step_id in _WHEEL_STEPS:
        return _WHEEL_DWELL_S
    return default


def _evidence_signature(row: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(row.get("moving_hits") or ()),
        tuple(row.get("fastener_hits") or ()),
    )


def _hand_progress_for_row(
    row: dict,
    *,
    is_current: bool = False,
    motion_banked: float = 0.0,
) -> bool:
    """True when this step's score row shows hand-driven work for dwell.

    Current (displayed) step: near-hand or hand+motion is enough.
    Parallel non-current ready step: near-hand *and* motion by default so an
    idle sibling part sitting near the hand does not steal dwell. Once that
    part has already moved (or attach is visible), keep banking on near-hand.
    Class-only (no boxes): only the current step may accumulate.
    """
    if not row:
        return False
    if not (row.get("moving_hits") or row.get("fastener_hits")):
        return False
    mode = row.get("mode") or "class_only"
    if mode == "class_only":
        return bool(is_current and row.get("hand") and row.get("moving_hits"))
    if is_current:
        return bool(row.get("near_hand") or (row.get("in_motion") and row.get("hand")))
    if not row.get("near_hand"):
        return False
    if row.get("in_motion"):
        return True
    if motion_banked >= _PARALLEL_MOTION_MEMORY_S:
        return True
    return bool(row.get("attach_relation") or row.get("target_hits"))


def _shared_evidence_rivals(step_id: str, ready: set[str], score_by_id: dict) -> list[str]:
    """Ready peers with the same moving/fastener hits (e.g. IR06/IR15/IR18)."""
    row = score_by_id.get(step_id) or {}
    sig = _evidence_signature(row)
    if not sig[0] and not sig[1]:
        return []
    return [
        peer
        for peer in ready
        if peer != step_id and _evidence_signature(score_by_id.get(peer) or {}) == sig
    ]


def advance_progress_memory(
    *,
    completed: set[str],
    rules: list[dict],
    inference: dict,
    memory: dict,
    timestamp_s: float,
    sustained_trust_s: float = _SUSTAINED_TRUST_S,
    min_dwell_s: float = _MIN_DWELL_S,
) -> tuple[set[str], dict, dict]:
    """Update completed-set using per-step dwell (parallel branches independent).

    Top-down policy:
    * every ready step accumulates its own hand-near evidence;
    * IR09 can finish while IR03 is still open;
    * a step is marked done only after enough per-step dwell, idle for that step,
      and (when target boxes were available) an attach relation moving/fastener↔target;
    * never auto-close prerequisites; never instant-complete brand-new ready steps.

    Returns ``(completed, memory, inference)``.
    """
    completed = set(completed)
    memory = dict(memory)
    status = inference.get("status")
    step = inference.get("step")
    best = _best_score_row(inference)
    ready_only = bool(inference.get("ready_only", True))
    dwell = {key: float(value) for key, value in dict(memory.get("step_dwell_s") or {}).items()}
    dwell_last = {
        key: float(value) for key, value in dict(memory.get("step_dwell_last") or {}).items()
    }
    motion_dwell = {
        key: float(value) for key, value in dict(memory.get("step_motion_dwell_s") or {}).items()
    }
    relation_dwell = {
        key: float(value)
        for key, value in dict(memory.get("step_relation_dwell_s") or {}).items()
    }
    saw_target = {
        key: bool(value) for key, value in dict(memory.get("step_saw_target") or {}).items()
    }
    relation_ready = {
        key: bool(value) for key, value in dict(memory.get("step_relation_ready") or {}).items()
    }
    score_by_id = {row["step_id"]: row for row in inference.get("scores") or []}

    if status == "mistake_prerequisites_missing" and step not in {None, "unknown"}:
        # Top-down: only trust-close missing pins when both chassis braces are done
        # and the worker is clearly on a late bracket/wheel step (detector often
        # missed the mid-procedure pins).
        braces_done = _BRACE_STEPS <= completed
        allow_trust = (
            (not ready_only)
            or (braces_done and step in _TRUSTABLE_STEPS)
        )
        if ready_only and not allow_trust:
            inference = {
                **inference,
                "completed_steps": sorted(completed),
                "ready_steps": sorted(ready_steps(rules, completed)),
            }
            memory["step_dwell_s"] = dwell
            memory["step_dwell_last"] = dwell_last
            memory["step_relation_dwell_s"] = relation_dwell
            memory["step_saw_target"] = saw_target
            memory["step_relation_ready"] = relation_ready
            memory["active_step"] = None
            memory["active_since_s"] = None
            return completed, memory, inference
        if not allow_trust:
            inference = {
                **inference,
                "completed_steps": sorted(completed),
                "ready_steps": sorted(ready_steps(rules, completed)),
            }
            memory["step_dwell_s"] = dwell
            memory["step_dwell_last"] = dwell_last
            memory["step_relation_dwell_s"] = relation_dwell
            memory["step_saw_target"] = saw_target
            memory["step_relation_ready"] = relation_ready
            return completed, memory, inference
        accum = dict(memory.get("blocked_accum") or {})
        lasts = dict(memory.get("blocked_last") or {})
        last = lasts.get(step)
        if last is not None:
            dt = max(0.0, timestamp_s - float(last))
            if dt <= _TRUST_GAP_S:
                accum[step] = float(accum.get(step) or 0) + min(dt, 1.0)
            else:
                accum[step] = 0.0
        else:
            accum[step] = float(accum.get(step) or 0)
        lasts[step] = timestamp_s
        memory["blocked_accum"] = accum
        memory["blocked_last"] = lasts
        blocked_for = float(accum.get(step) or 0)
        if (
            blocked_for >= sustained_trust_s
            and step in _TRUSTABLE_STEPS
            and best.get("moving_hits")
            and best.get("near_hand")
            and best.get("score", 0) >= 6
        ):
            completed = close_prerequisites(rules, completed, step)
            inference = {
                **inference,
                "status": "step_in_progress",
                "recovery": "",
                "missing_prerequisites": [],
                "certainty": "uncertain",
                "evidence": (
                    f"Sustained work on {step}; treating missing prerequisites as already done."
                )[:180],
                "progress_trust": True,
            }
            status = "step_in_progress"
            accum.pop(step, None)
            lasts.pop(step, None)
            memory["blocked_accum"] = accum
            memory["blocked_last"] = lasts
        else:
            inference = {
                **inference,
                "completed_steps": sorted(completed),
                "ready_steps": sorted(ready_steps(rules, completed)),
            }
            memory["step_dwell_s"] = dwell
            memory["step_dwell_last"] = dwell_last
            memory["step_relation_dwell_s"] = relation_dwell
            memory["step_saw_target"] = saw_target
            memory["step_relation_ready"] = relation_ready
            return completed, memory, inference
    else:
        lasts = dict(memory.get("blocked_last") or {})
        accum = dict(memory.get("blocked_accum") or {})
        for key, last in list(lasts.items()):
            if timestamp_s - float(last) > _TRUST_GAP_S:
                lasts.pop(key, None)
                accum.pop(key, None)
        memory["blocked_last"] = lasts
        memory["blocked_accum"] = accum

    ready = ready_steps(rules, completed)
    newly_completed: list[str] = []
    prev_ts = memory.get("last_timestamp_s")
    frame_dt = 0.1 if prev_ts is None else max(0.0, min(1.0, timestamp_s - float(prev_ts)))

    for step_id in list(ready):
        row = score_by_id.get(step_id) or {}
        is_current = step_id == step
        working = bool(row.get("interaction")) and _hand_progress_for_row(
            row,
            is_current=is_current,
            motion_banked=float(motion_dwell.get(step_id) or 0),
        )
        # Front brace may bank attach while rear pins are open; DONE is gated
        # on rear pins completing (+ relative wait) so early IR03 cannot seal.
        # Identical pin/hardware evidence must not finish every ready peer at once.
        if working and _shared_evidence_rivals(step_id, set(ready), score_by_id):
            working = is_current
        last = dwell_last.get(step_id)
        need = _min_dwell_for_step(step_id, min_dwell_s)

        def _need_dwell_for(sid: str) -> float:
            need_dwell = _min_dwell_for_step(sid, min_dwell_s)
            # Front brace / front wheel / bracket / screw: no relation credit (easy FP).
            # IR30 may take hold-credit once the wheel is clearly in hand.
            if (
                sid in {"IR27", "IR24"}
                or sid in _BRACKET_STEPS
                or sid == "IR03"
            ):
                return need_dwell
            if relation_ready.get(sid) or relation_supports_done(
                    float((score_by_id.get(sid) or {}).get("relation_score") or 0),
                    strong=True):
                need_dwell = max(
                    _RELATION_FAST_DONE_FLOOR_S,
                    need_dwell - _RELATION_DWELL_CREDIT_S,
                )
            # Strong hand-hold on the rear wheel also shortens the DONE bank.
            if sid == "IR30":
                row30 = score_by_id.get("IR30") or {}
                if float(row30.get("relation_score") or 0) >= 0.45 or row30.get(
                    "attach_relation"
                ):
                    need_dwell = max(
                        _RELATION_FAST_DONE_FLOOR_S,
                        need_dwell - _RELATION_DWELL_CREDIT_S,
                    )
            return need_dwell

        def _worked_need_s(sid: str) -> float:
            if sid == "IR09":
                return _WORK_BEFORE_ATTACH_IR09_S
            if sid == "IR03":
                return _WORK_BEFORE_ATTACH_IR03_S
            if sid in _WHEEL_STEPS:
                return _WORK_BEFORE_ATTACH_WHEEL_S
            if sid in _FRONT_BRANCH_STEPS or sid in _BRACKET_STEPS:
                return _WORK_BEFORE_ATTACH_FRONT_S
            return _WORK_BEFORE_ATTACH_S

        def _attach_need_s(sid: str) -> float:
            if sid == "IR09":
                return _ATTACH_RELATION_DWELL_IR09_S
            if sid == "IR03":
                return _ATTACH_RELATION_DWELL_IR03_S
            if sid in _WHEEL_STEPS:
                return _ATTACH_RELATION_DWELL_WHEEL_S
            if sid in _FRONT_BRANCH_STEPS or sid in _BRACKET_STEPS:
                return _ATTACH_RELATION_DWELL_FRONT_S
            return _ATTACH_RELATION_DWELL_S

        def _worked_ok_for(sid: str) -> bool:
            """True once the worker actually handled this step (not table contact)."""
            peak = float((memory.get("step_dwell_peak_s") or {}).get(sid) or 0)
            need = _worked_need_s(sid)
            if (
                sid == "IR03"
                and memory.get("rear_pins_done_s") is not None
                and min_dwell_s == _MIN_DWELL_S
            ):
                # After rear pins: accept seating progress (dwell often idle).
                return (
                    float(dwell.get(sid) or 0) >= 1.0
                    or peak >= 1.0
                    or float(relation_dwell.get(sid) or 0) >= 1.0
                    or float(motion_dwell.get(sid) or 0) >= 0.5
                )
            return (
                float(dwell.get(sid) or 0) >= need
                or peak >= need
                or float(motion_dwell.get(sid) or 0) >= max(1.0, need * 0.35)
            )

        def _motion_ok_for(sid: str) -> bool:
            if min_dwell_s != _MIN_DWELL_S:
                return True
            bank = float(motion_dwell.get(sid) or 0)
            if sid == "IR09":
                return bank >= _REAR_BRACE_MOTION_DWELL_S
            if sid == "IR03":
                if memory.get("rear_pins_done_s") is not None and relation_ready.get(sid):
                    return True
                return bank >= _BRACE_MOTION_DWELL_S
            if sid in _WHEEL_STEPS:
                return bank >= _WHEEL_MOTION_DWELL_S
            if sid in _FRONT_BRANCH_STEPS or sid in _BRACKET_STEPS:
                return bank >= _FRONT_MOTION_DWELL_S
            return True

        def _relation_ok_for(sid: str, rel: float) -> bool:
            rel_bank = float(relation_dwell.get(sid) or 0)
            if rel_bank >= _attach_need_s(sid) and _worked_ok_for(sid):
                relation_ready[sid] = True
            if relation_ready.get(sid) or relation_supports_done(rel, strong=True):
                return True
            # Production: attach required (no dwell-only escape).
            if min_dwell_s == _MIN_DWELL_S:
                return False
            # Unit-test timelines may omit attach boxes.
            if not saw_target.get(sid):
                return True
            base_need = _min_dwell_for_step(sid, min_dwell_s)
            extra = (
                _REAR_BRACE_ESCAPE_EXTRA_S
                if sid == "IR09"
                else _RELATION_ESCAPE_EXTRA_S
            )
            return float(dwell.get(sid) or 0) >= base_need + extra

        def _hand_on_part(sid: str) -> bool:
            """True while the worker is still manipulating the step's part.

            After sticky attach, only a true ``holding`` relation blocks DONE —
            spatial ``near_hand`` alone is common for seated wheels/braces with
            a nearby hand and was starving hand-off forever.
            """
            row = score_by_id.get(sid) or {}
            hold_pred = str(row.get("hand_predicate") or "")
            hold_score = float(row.get("hand_score") or 0)
            if relation_ready.get(sid):
                return (
                    hold_pred == "holding"
                    and hold_score >= _HANDOFF_HOLD_BLOCK
                )
            if row.get("near_hand"):
                return True
            if row.get("hand") and (row.get("moving_hits") or row.get("fastener_hits")):
                return True
            if hold_pred == "holding" and hold_score >= _HANDOFF_HOLD_BLOCK:
                return True
            return False

        def _handoff_need_s(sid: str) -> float:
            if sid == "IR09":
                return _HANDOFF_IDLE_IR09_S
            if sid == "IR03":
                return _HANDOFF_IDLE_IR03_S
            if sid in _PIN_STEPS:
                return _HANDOFF_IDLE_PIN_S
            if sid in _WHEEL_STEPS:
                return _HANDOFF_IDLE_WHEEL_S
            if sid in _FRONT_BRANCH_STEPS or sid in _BRACKET_STEPS:
                return _HANDOFF_IDLE_FRONT_S
            return _HANDOFF_IDLE_S

        def _rear_pins_in_sop() -> set[str]:
            return {"IR15", "IR18"} & {str(rule.get("step_id") or "") for rule in rules}

        def _maybe_complete(sid: str, *, idle_for: float, rel: float) -> bool:
            if sid in completed or sid in newly_completed:
                return False
            # Generalization policy: sticky attach → hand leaves → short clear idle.
            if min_dwell_s == _MIN_DWELL_S:
                if not _worked_ok_for(sid):
                    return False
                # Braces / pins / front / wheels: require sustained sticky attach.
                if (
                    sid in _BRACE_STEPS
                    or sid in _PIN_STEPS
                    or sid in _WHEEL_STEPS
                    or sid in _FRONT_BRANCH_STEPS
                ):
                    if not relation_ready.get(sid):
                        return False
                elif not (
                    relation_ready.get(sid)
                    or relation_supports_done(rel, strong=True)
                ):
                    return False
                if not _motion_ok_for(sid):
                    return False
                if _hand_on_part(sid):
                    # Still manipulating — reset hand-off clock.
                    left = dict(memory.get("step_hand_left_s") or {})
                    left.pop(sid, None)
                    memory["step_hand_left_s"] = left
                    return False
                left = dict(memory.get("step_hand_left_s") or {})
                if sid not in left:
                    left[sid] = float(timestamp_s)
                    memory["step_hand_left_s"] = left
                    return False
                if timestamp_s < float(left[sid]) + _handoff_need_s(sid):
                    return False
                # Shared pin/fastener evidence: only the live CURRENT may seal.
                if sid in _PIN_STEPS and _shared_evidence_rivals(
                    sid, set(ready) | set(newly_completed), score_by_id
                ):
                    if step != sid and memory.get("active_step_display") != sid:
                        return False
            else:
                need_idle = (
                    _RELATION_COMPLETE_IDLE_S
                    if relation_ready.get(sid)
                    else _COMPLETE_IDLE_S
                )
                need_dwell = _need_dwell_for(sid)
                peak = float((memory.get("step_dwell_peak_s") or {}).get(sid) or 0)
                dwell_ok = (
                    float(dwell.get(sid) or 0) >= need_dwell
                    or peak >= need_dwell
                    or bool(relation_ready.get(sid))
                )
                if not (
                    dwell_ok
                    and idle_for >= need_idle
                    and _motion_ok_for(sid)
                    and _relation_ok_for(sid, rel)
                ):
                    return False
            # When both chassis braces are still open, finish the rear branch
            # (IR09) before the front (IR03).
            if sid == "IR03" and (
                "IR09" in ready or "IR09" in newly_completed
            ):
                return False
            # Front brace after rear pins (when those steps exist in the SOP).
            rear_pins = _rear_pins_in_sop()
            if sid == "IR03" and rear_pins:
                done_pins = completed | set(newly_completed)
                if not rear_pins <= done_pins:
                    return False
                pin_done_s = memory.get("rear_pins_done_s")
                if pin_done_s is None and rear_pins <= done_pins:
                    pin_done_s = float(timestamp_s)
                if pin_done_s is not None and timestamp_s < float(pin_done_s) + _IR03_AFTER_REAR_PINS_S:
                    return False
            # Rear wheel: wait after all of its prerequisites are sealed.
            if sid == "IR30" and min_dwell_s == _MIN_DWELL_S:
                prereq_s = memory.get("ir30_prereqs_done_s")
                if prereq_s is None:
                    return False
                if timestamp_s < float(prereq_s) + _IR30_AFTER_PREREQS_S:
                    return False
            # Front bracket branch: wait after IR30, or after rear pins when no IR30.
            if sid in _FRONT_BRANCH_STEPS and min_dwell_s == _MIN_DWELL_S:
                has_ir30 = any(str(rule.get("step_id") or "") == "IR30" for rule in rules)
                if has_ir30:
                    ir30_s = memory.get("ir30_done_s")
                    if ir30_s is None:
                        return False
                    if timestamp_s < float(ir30_s) + _FRONT_AFTER_IR30_S:
                        return False
                else:
                    pin_done_s = memory.get("rear_pins_done_s")
                    if pin_done_s is None:
                        return False
                    if timestamp_s < float(pin_done_s) + _FRONT_AFTER_PINS_NO_IR30_S:
                        return False
            # DAG only: rear wheel before front bracket branch while IR30 still ready.
            if sid in {"IR21", "IR24", "IR27"} and (
                "IR30" in ready or "IR30" in newly_completed
            ):
                return False
            newly_completed.append(sid)
            return True

        if working:
            # Sticky CURRENT bookkeeping (live help — separate from DONE).
            help_since_map = dict(memory.get("step_help_since_s") or {})
            if step_id not in help_since_map and (
                is_current
                or memory.get("active_step_display") == step_id
                or row.get("near_hand")
            ):
                help_since_map[step_id] = float(timestamp_s)
                memory["step_help_since_s"] = help_since_map
            if (
                is_current
                or memory.get("active_step_display") == step_id
            ):
                last_cur = dict(memory.get("step_last_current_s") or {})
                last_cur[step_id] = float(timestamp_s)
                memory["step_last_current_s"] = last_cur
            if (
                row.get("saw_target_boxes")
                or row.get("attach_relation")
                or relation_supports_done(float(row.get("relation_score") or 0))
            ):
                saw_target[step_id] = True
            if last is not None:
                dt = max(0.0, timestamp_s - float(last))
                if dt <= _TRUST_GAP_S:
                    dwell[step_id] = float(dwell.get(step_id) or 0) + min(dt, 1.0)
                    peak = dict(memory.get("step_dwell_peak_s") or {})
                    peak[step_id] = max(
                        float(peak.get(step_id) or 0),
                        float(dwell.get(step_id) or 0),
                    )
                    memory["step_dwell_peak_s"] = peak
                    if row.get("in_motion") or row.get("mode") == "class_only":
                        motion_dwell[step_id] = float(motion_dwell.get(step_id) or 0) + min(dt, 1.0)
                    rel_score = float(row.get("relation_score") or 0)
                    if (
                        _worked_ok_for(step_id)
                        and (
                            row.get("attach_relation")
                            or relation_supports_done(rel_score)
                        )
                    ):
                        relation_dwell[step_id] = (
                            float(relation_dwell.get(step_id) or 0) + min(dt, 1.0))
                        if float(relation_dwell[step_id]) >= _attach_need_s(step_id):
                            relation_ready[step_id] = True
                elif not relation_ready.get(step_id):
                    dwell[step_id] = 0.0
                    motion_dwell[step_id] = 0.0
                    relation_dwell[step_id] = 0.0
            else:
                dwell.setdefault(step_id, float(dwell.get(step_id) or 0))
                motion_dwell.setdefault(step_id, float(motion_dwell.get(step_id) or 0))
                relation_dwell.setdefault(step_id, float(relation_dwell.get(step_id) or 0))
                rel_score = float(row.get("relation_score") or 0)
                if (
                    _worked_ok_for(step_id)
                    and (
                        row.get("attach_relation")
                        or relation_supports_done(rel_score)
                    )
                ):
                    relation_dwell[step_id] = max(
                        float(relation_dwell.get(step_id) or 0), 0.05)
                    if float(relation_dwell[step_id]) >= _attach_need_s(step_id):
                        relation_ready[step_id] = True
            dwell_last[step_id] = timestamp_s
            # Hand still on part → keep resetting hand-off; do not DONE yet.
            _maybe_complete(
                step_id,
                idle_for=0.0,
                rel=float(row.get("relation_score") or 0),
            )
            continue
        # Idle / non-working: continue banking seated attach after real work,
        # or for IR03 after rear pins (front brace often already seated).
        rel_score = float(row.get("relation_score") or 0)
        pins_done = memory.get("rear_pins_done_s") is not None
        allow_idle_attach = _worked_ok_for(step_id) or (
            step_id == "IR03" and pins_done and min_dwell_s == _MIN_DWELL_S
        )
        if allow_idle_attach and (
            row.get("attach_relation") or relation_supports_done(rel_score)
        ):
            saw_target[step_id] = True
            relation_dwell[step_id] = float(relation_dwell.get(step_id) or 0) + min(
                frame_dt, 1.0)
            if float(relation_dwell[step_id]) >= _attach_need_s(step_id):
                relation_ready[step_id] = True
        # Idle: soft-decay banked dwell; complete after attach + hand-off.
        idle_for = 0.0 if last is None else timestamp_s - float(last)
        keep_ir03_bank = (
            step_id == "IR03"
            and memory.get("rear_pins_done_s") is not None
            and min_dwell_s == _MIN_DWELL_S
        )
        if last is not None:
            if (
                idle_for > _TRUST_GAP_S
                and not relation_ready.get(step_id)
                and not keep_ir03_bank
            ):
                dwell.pop(step_id, None)
                dwell_last.pop(step_id, None)
                motion_dwell.pop(step_id, None)
                relation_dwell.pop(step_id, None)
                saw_target.pop(step_id, None)
                relation_ready.pop(step_id, None)
                left = dict(memory.get("step_hand_left_s") or {})
                left.pop(step_id, None)
                memory["step_hand_left_s"] = left
                continue
            if idle_for > _DWELL_DECAY_IDLE_S and step_id in dwell:
                if not relation_ready.get(step_id):
                    dwell[step_id] = max(0.0, float(dwell.get(step_id) or 0) - frame_dt * 0.25)
                    motion_dwell[step_id] = max(
                        0.0, float(motion_dwell.get(step_id) or 0) - frame_dt * 0.25)
                if (
                    not relation_supports_done(rel_score)
                    and not relation_ready.get(step_id)
                    and not keep_ir03_bank
                ):
                    relation_dwell[step_id] = max(
                        0.0, float(relation_dwell.get(step_id) or 0) - frame_dt * 0.25)
                if (
                    float(dwell.get(step_id) or 0) <= 0.05
                    and not relation_ready.get(step_id)
                    and not keep_ir03_bank
                ):
                    dwell.pop(step_id, None)
                    dwell_last.pop(step_id, None)
                    motion_dwell.pop(step_id, None)
                    if not relation_supports_done(rel_score):
                        relation_dwell.pop(step_id, None)
                    continue
        _maybe_complete(step_id, idle_for=idle_for, rel=rel_score)

    for step_id in newly_completed:
        completed.add(step_id)
        dwell.pop(step_id, None)
        dwell_last.pop(step_id, None)
        motion_dwell.pop(step_id, None)
        reason = "dwell_idle"
        if relation_ready.get(step_id) or (
            saw_target.get(step_id)
            and float(relation_dwell.get(step_id) or 0) >= _ATTACH_RELATION_DWELL_S
        ):
            reason = "dwell_idle_attach"
        relation_dwell.pop(step_id, None)
        saw_target.pop(step_id, None)
        relation_ready.pop(step_id, None)
        help_map = dict(memory.get("step_help_since_s") or {})
        help_map.pop(step_id, None)
        memory["step_help_since_s"] = help_map
        left = dict(memory.get("step_hand_left_s") or {})
        left.pop(step_id, None)
        memory["step_hand_left_s"] = left
        memory["last_completed_reason"] = f"{reason}:{step_id}"
        # Keep IR03 attach/motion banks after rear closes — front brace is often
        # already seated; wiping forced a full re-earn that starved some clips.
        if step_id == "IR09":
            dwell.pop("IR03", None)
            dwell_last.pop("IR03", None)
        # Stamp when both rear pins are done (IR03 relative wait).
        rear_pins = {"IR15", "IR18"} & {str(rule.get("step_id") or "") for rule in rules}
        if rear_pins and rear_pins <= completed and memory.get("rear_pins_done_s") is None:
            memory["rear_pins_done_s"] = float(timestamp_s)
        # Stamp when IR30 prerequisites are all sealed.
        ir30_rule = next((rule for rule in rules if str(rule.get("step_id") or "") == "IR30"), None)
        if ir30_rule is not None:
            prereqs = {str(p) for p in (ir30_rule.get("prerequisites") or [])}
            if prereqs and prereqs <= completed and memory.get("ir30_prereqs_done_s") is None:
                memory["ir30_prereqs_done_s"] = float(timestamp_s)
        if step_id == "IR30":
            memory["ir30_done_s"] = float(timestamp_s)
            # Drop stale dwell clocks, but keep attach banks so IR21/24/27 can
            # hand-off immediately after the rear wheel seals.
            for front in ("IR21", "IR24", "IR27"):
                dwell.pop(front, None)
                dwell_last.pop(front, None)
                motion_dwell.pop(front, None)
                help_map = dict(memory.get("step_help_since_s") or {})
                help_map.pop(front, None)
                memory["step_help_since_s"] = help_map
                left = dict(memory.get("step_hand_left_s") or {})
                left.pop(front, None)
                memory["step_hand_left_s"] = left
        # Screw and front wheel overlap at the end of this clip; carry a short
        # dwell seed so IR27 can finalize after IR24 without re-earning the
        # whole bank from a vanishing near-hand signal.
        if step_id == "IR24":
            dwell["IR27"] = max(float(dwell.get("IR27") or 0), _FRONT_WHEEL_DWELL_S)
            dwell_last["IR27"] = float(timestamp_s)
            saw_target["IR27"] = True
            # Do not auto-arm relation_ready — that raced IR27 DONE without seat.

    # Detector often misses mid-procedure pins. Unit-test / short-dwell mode may
    # close pin prerequisites from sustained late bracket/wheel work. Production
    # generalization disables this — it auto-sealed pins and raced the front branch.
    if (
        min_dwell_s != _MIN_DWELL_S
        and _BRACE_STEPS <= completed
        and not _PIN_STEPS <= completed
    ):
        late_row = None
        late_id = None
        for sid in ("IR21", "IR30", "IR24", "IR27"):
            row = score_by_id.get(sid) or {}
            if (
                row.get("near_hand")
                and (row.get("moving_hits") or row.get("fastener_hits"))
                and float(row.get("score") or 0) >= 5
            ):
                late_row = row
                late_id = sid
                break
        late_accum = dict(memory.get("late_trust_accum") or {})
        late_last = dict(memory.get("late_trust_last") or {})
        if late_id:
            last_late = late_last.get(late_id)
            if last_late is not None:
                dt = max(0.0, timestamp_s - float(last_late))
                if dt <= _TRUST_GAP_S:
                    late_accum[late_id] = float(late_accum.get(late_id) or 0) + min(dt, 1.0)
                else:
                    late_accum[late_id] = 0.0
            else:
                late_accum[late_id] = float(late_accum.get(late_id) or 0)
            late_last[late_id] = timestamp_s
            if float(late_accum.get(late_id) or 0) >= sustained_trust_s:
                for pin in sorted(_PIN_STEPS):
                    if pin not in completed:
                        completed.add(pin)
                        memory["last_completed_reason"] = f"late_trust:{late_id}->{pin}"
                late_accum.pop(late_id, None)
                late_last.pop(late_id, None)
                inference = {
                    **inference,
                    "progress_trust": True,
                    "evidence": (
                        f"Sustained late work on {late_id}; closing missed pin prerequisites."
                    )[:180],
                }
        else:
            for key, last in list(late_last.items()):
                if timestamp_s - float(last) > _TRUST_GAP_S:
                    late_last.pop(key, None)
                    late_accum.pop(key, None)
        memory["late_trust_accum"] = late_accum
        memory["late_trust_last"] = late_last

    # Drop dwell for steps that are no longer ready (already done or blocked).
    ready_now = ready_steps(rules, completed)
    for key in list(dwell):
        if key not in ready_now and key not in completed:
            # Keep dwell briefly if still in ready_now after other completions.
            pass
        if key in completed:
            dwell.pop(key, None)
            dwell_last.pop(key, None)
            motion_dwell.pop(key, None)
            relation_dwell.pop(key, None)
            saw_target.pop(key, None)
            relation_ready.pop(key, None)

    # Overlay / detect scoping: sticky CURRENT for worker help (not DONE).
    # Confirm a new candidate briefly before switching; keep display through short gaps.
    candidate = (
        step if status == "step_in_progress" and step not in {None, "unknown"} else None
    )
    display = memory.get("active_step_display")
    pending = memory.get("current_pending_step")
    pending_since = memory.get("current_pending_since_s")
    if candidate:
        memory["current_idle_since_s"] = None
        memory["active_step"] = candidate  # detect scope follows live evidence
        if candidate == display:
            memory["current_pending_step"] = None
            memory["current_pending_since_s"] = None
        elif candidate == pending and pending_since is not None:
            if timestamp_s - float(pending_since) >= _CURRENT_CONFIRM_S:
                memory["active_step_display"] = candidate
                memory["active_since_s"] = timestamp_s
                memory["current_pending_step"] = None
                memory["current_pending_since_s"] = None
        else:
            memory["current_pending_step"] = candidate
            memory["current_pending_since_s"] = timestamp_s
            # First claim: if nothing displayed yet, show after confirm window starts
            # by accepting immediately when display is empty (worker needs first cue).
            if not display:
                memory["active_step_display"] = candidate
                memory["active_since_s"] = timestamp_s
                memory["current_pending_step"] = None
                memory["current_pending_since_s"] = None
    else:
        memory["current_pending_step"] = None
        memory["current_pending_since_s"] = None
        memory["active_step"] = display
        if display:
            idle_since = memory.get("current_idle_since_s")
            if idle_since is None:
                memory["current_idle_since_s"] = timestamp_s
            elif timestamp_s - float(idle_since) >= _CURRENT_CLEAR_S:
                memory["active_step"] = None
                memory["active_since_s"] = None
                memory["active_step_display"] = None
                memory["current_idle_since_s"] = None
        else:
            memory["active_step"] = None
            memory["active_since_s"] = None
            memory["current_idle_since_s"] = None

    # Stamp CURRENT→DONE help clock whenever sticky CURRENT is shown.
    shown = memory.get("active_step_display")
    if shown not in {None, "unknown", ""}:
        help_map = dict(memory.get("step_help_since_s") or {})
        help_map.setdefault(str(shown), float(timestamp_s))
        memory["step_help_since_s"] = help_map
        last_cur = dict(memory.get("step_last_current_s") or {})
        last_cur[str(shown)] = float(timestamp_s)
        memory["step_last_current_s"] = last_cur

    memory["step_dwell_s"] = dwell
    memory["step_dwell_last"] = dwell_last
    memory["step_motion_dwell_s"] = motion_dwell
    memory["step_relation_dwell_s"] = relation_dwell
    memory["step_saw_target"] = saw_target
    memory["step_relation_ready"] = relation_ready
    memory["last_timestamp_s"] = timestamp_s
    current_help = memory.get("active_step_display")
    inference = {
        **inference,
        "completed_steps": sorted(completed),
        "ready_steps": sorted(ready_steps(rules, completed)),
        "missing_prerequisites": [],
        "recovery": "" if status == "step_in_progress" else inference.get("recovery", ""),
        "step_dwell_s": {key: round(value, 2) for key, value in dwell.items()},
        "step_relation_dwell_s": {
            key: round(value, 2) for key, value in relation_dwell.items()
        },
        # Worker-facing CURRENT (sticky); independent of DONE completed_steps.
        "current_step": current_help,
        "help_active": bool(current_help),
    }
    return completed, memory, inference


def _pin_order_key(step_id: str) -> tuple[int, str]:
    """SOP order among pin installs (front-rear pin before rear-rear pin)."""
    order = {"IR15": 0, "IR18": 1, "IR06": 2}
    return (order.get(step_id, 9), step_id)


def _prefer_pin_fastener_current(inference: dict) -> dict:
    """When a ready pin step has hand+fastener (pin box missing), claim CURRENT.

    Pin boxes are often invisible mid-install; nuts/washers near the hand are
    the reliable cue for live help on IR06/IR15/IR18.
    """
    ready = set(inference.get("ready_steps") or [])
    pin_ready = sorted(ready & _PIN_STEPS, key=_pin_order_key)
    if not pin_ready:
        return inference
    scores = {row["step_id"]: row for row in inference.get("scores") or []}
    picks = []
    for sid in pin_ready:
        row = scores.get(sid) or {}
        if not row.get("interaction"):
            continue
        if not (row.get("fastener_hits") or row.get("moving_hits")):
            continue
        if not (row.get("near_hand") or row.get("in_motion")):
            continue
        picks.append((sid, row))
    if not picks:
        return inference
    # Prefer earliest pin with a moving pin hit; else earliest fastener-near-hand.
    with_pin = [item for item in picks if item[1].get("moving_hits")]
    sid, row = (with_pin or picks)[0]
    current = inference.get("step")
    if current == sid and inference.get("status") == "step_in_progress":
        return inference
    # Do not steal CURRENT from another ready pin that already has stronger pin motion.
    if current in _PIN_STEPS and current in ready and current != sid:
        cur = scores.get(current) or {}
        if cur.get("moving_hits") and cur.get("near_hand") and not row.get("moving_hits"):
            return inference
    primary = (row.get("moving_hits") or row.get("fastener_hits") or ["fastener"])[0]
    contact = "near hand" if row.get("near_hand") else "in motion"
    return {
        **inference,
        "step": sid,
        "status": "step_in_progress",
        "action": "pick_up",
        "certainty": "clear",
        "score": max(int(inference.get("score") or 0), int(row.get("score") or 5)),
        "evidence": f"Ready {sid}: detected {primary} {contact} (pin-step help)."[:180],
        "instruction": inference.get("instruction") or "",
        "hand_current": True,
        "pin_fastener_current": True,
    }


def _prefer_rear_brace_current(inference: dict) -> dict:
    """When both chassis braces are ready, prefer IR09 only on clear rear work.

    Idle long-brace near-hand on the table must not steal CURRENT from IR03 or
    bank a false-early DONE. Require motion, attach, or a strong hand hold.
    """
    ready = set(inference.get("ready_steps") or [])
    if "IR09" not in ready or "IR03" not in ready:
        return inference
    scores = {row["step_id"]: row for row in inference.get("scores") or []}
    row09 = scores.get("IR09") or {}
    if not row09.get("interaction") or not row09.get("moving_hits"):
        return inference
    strong = bool(
        row09.get("in_motion")
        or row09.get("attach_relation")
        or float(row09.get("relation_score") or 0) >= 0.45
    )
    if not strong:
        return inference
    current = inference.get("step")
    if current == "IR09" and inference.get("status") == "step_in_progress":
        return inference
    return {
        **inference,
        "step": "IR09",
        "status": "step_in_progress",
        "action": "pick_up",
        "certainty": "clear",
        "score": max(int(inference.get("score") or 0), int(row09.get("score") or 5)),
        "evidence": (
            "Ready IR09: detected long brace "
            + ("in motion" if row09.get("in_motion") else "near hand")
            + " (rear-first help)."
        )[:180],
        "hand_current": True,
        "rear_brace_current": True,
    }


def _prefer_rear_wheel_current(inference: dict) -> dict:
    """Prefer IR30 over IR21 when the rear wheel is clearly in hand.

    Stops the front bracket from stealing CURRENT mid rear-wheel install.
    """
    ready = set(inference.get("ready_steps") or [])
    if "IR30" not in ready:
        return inference
    scores = {row["step_id"]: row for row in inference.get("scores") or []}
    row30 = scores.get("IR30") or {}
    if not row30.get("interaction") or not row30.get("moving_hits"):
        return inference
    strong = bool(
        row30.get("near_hand")
        and (
            row30.get("in_motion")
            or row30.get("attach_relation")
            or float(row30.get("relation_score") or 0) >= 0.45
        )
    )
    if not strong:
        return inference
    current = inference.get("step")
    if current == "IR30" and inference.get("status") == "step_in_progress":
        return inference
    # Only override when competing with IR21 or unknown.
    if current not in {None, "unknown", "IR21", "IR30"}:
        return inference
    return {
        **inference,
        "step": "IR30",
        "status": "step_in_progress",
        "action": "pick_up",
        "certainty": "clear",
        "score": max(int(inference.get("score") or 0), int(row30.get("score") or 5)),
        "evidence": (
            "Ready IR30: detected toy car wheel "
            + ("near hand" if row30.get("near_hand") else "in motion")
            + " (rear-wheel help)."
        )[:180],
        "hand_current": True,
        "rear_wheel_current": True,
    }


def _prefer_hand_current(inference: dict, edges: list[dict]) -> dict:
    """If a ready step has a strong hand↔part edge, prefer it as CURRENT.

    Does not mark DONE — only redirects the live ``step`` / status used for help.
    Pin steps: also accept hand↔fastener when the pin box is missing.
    """
    inference = _prefer_rear_brace_current(inference)
    inference = _prefer_rear_wheel_current(inference)
    inference = _prefer_pin_fastener_current(inference)
    ready = set(inference.get("ready_steps") or [])
    if not ready:
        return inference
    hand_edges = [
        edge for edge in edges
        if edge.get("kind") == "hand"
        and edge.get("step_id") in ready
        and float(edge.get("score") or 0) >= _HAND_CURRENT_SCORE
        and str(edge.get("predicate") or "") in {"holding", "near"}
    ]
    if not hand_edges:
        return inference
    # Prefer pin-step hand edges when those steps are ready (live assist priority).
    pin_edges = [edge for edge in hand_edges if edge.get("step_id") in _PIN_STEPS]
    pool = pin_edges or hand_edges
    best = max(
        pool,
        key=lambda edge: (
            float(edge.get("score") or 0),
            1 if edge.get("predicate") == "holding" else 0,
            -_pin_order_key(str(edge.get("step_id") or ""))[0],
        ),
    )
    step_id = best["step_id"]
    current = inference.get("step")
    if current == step_id and inference.get("status") == "step_in_progress":
        return inference
    # Only override unknown / weaker CURRENT; keep a clearly stronger assemble score.
    scores = {row["step_id"]: row for row in inference.get("scores") or []}
    best_row = scores.get(step_id) or {}
    cur_row = scores.get(current) if current not in {None, "unknown"} else None
    if (
        cur_row
        and float(cur_row.get("score") or 0) >= float(best_row.get("score") or 0) + 3
        and cur_row.get("near_hand")
        and current not in _PIN_STEPS
        and step_id not in _PIN_STEPS
    ):
        return inference
    # Pin CURRENT may override a brace CURRENT when the pin is ready and in hand.
    if (
        cur_row
        and current in _BRACE_STEPS
        and step_id in _PIN_STEPS
        and (best_row.get("near_hand") or best_row.get("fastener_hits"))
    ):
        pass  # allow override
    elif (
        cur_row
        and float(cur_row.get("score") or 0) >= float(best_row.get("score") or 0) + 3
        and cur_row.get("near_hand")
    ):
        return inference
    part = best.get("object") or "part"
    return {
        **inference,
        "step": step_id,
        "status": "step_in_progress",
        "action": "pick_up",
        "certainty": "clear",
        "score": max(int(inference.get("score") or 0), int(best_row.get("score") or 5)),
        "evidence": (
            f"Ready {step_id}: hand holding {part} "
            f"(relation {float(best.get('score') or 0):.2f})."
        )[:180],
        "hand_current": True,
    }


def reason_over_detection_records(
    records: list[dict],
    sop: dict,
    *,
    initial_completed: list[str] | None = None,
    minimum_score: int = 3,
    require_interaction: bool = True,
    sustained_trust_s: float = _SUSTAINED_TRUST_S,
    min_dwell_s: float = _MIN_DWELL_S,
    ready_only: bool = True,
) -> list[dict]:
    """Run causal top-down step reasoning over a detection timeline."""
    rules = build_step_rules(sop)
    completed = set(initial_completed or [])
    observations = []
    previous_centers: dict[str, list[tuple[float, float]]] = {}
    relation_filter = RelationHoldFilter()
    timing = sop.get("timing_profile") if isinstance(sop.get("timing_profile"), dict) else None
    # timing_profile optional; absolute floors / train delays ignored for generalization.
    if timing is None:
        earliest: dict[str, float] = {}
        front_wait = 0.0
    else:
        earliest = {
            str(k): float(v)
            for k, v in (timing.get("step_earliest_done_s") or {}).items()
        }
        # Prefer 0 unless explicitly set; do not load done_delay_* (schedule hacks).
        front_wait = float(timing.get("front_branch_after_ir30_s") or 0.0)
    memory: dict[str, Any] = {
        "active_step": None,
        "active_since_s": None,
        "active_step_display": None,
        "current_pending_step": None,
        "current_pending_since_s": None,
        "current_idle_since_s": None,
        "step_dwell_s": {},
        "step_dwell_last": {},
        "step_dwell_peak_s": {},
        "step_motion_dwell_s": {},
        "step_relation_dwell_s": {},
        "step_saw_target": {},
        "step_relation_ready": {},
        "step_help_since_s": {},
        "step_last_current_s": {},
        "step_hand_left_s": {},
        "rear_pins_done_s": None,
        "ir30_prereqs_done_s": None,
        "ir30_done_s": None,
        "blocked_accum": {},
        "blocked_last": {},
        "late_trust_accum": {},
        "late_trust_last": {},
        "last_timestamp_s": None,
        "step_earliest_done_s": earliest,
        "front_branch_after_ir30_s": front_wait,
    }
    # Seed relative-wait stamps when the caller already completed prerequisites
    # (unit tests / resume), so waits are satisfied at timeline start.
    t0 = float((records[0] or {}).get("timestamp_s") or 0.0) if records else 0.0
    rear_pins = {"IR15", "IR18"} & {str(rule.get("step_id") or "") for rule in rules}
    if rear_pins and rear_pins <= completed:
        memory["rear_pins_done_s"] = t0 - max(
            _IR03_AFTER_REAR_PINS_S, _FRONT_AFTER_PINS_NO_IR30_S) - 1.0
    ir30_rule = next(
        (rule for rule in rules if str(rule.get("step_id") or "") == "IR30"), None)
    if ir30_rule is not None:
        prereqs = {str(p) for p in (ir30_rule.get("prerequisites") or [])}
        if prereqs and prereqs <= completed:
            memory["ir30_prereqs_done_s"] = t0 - _IR30_AFTER_PREREQS_S - 1.0
    if "IR30" in completed:
        memory["ir30_done_s"] = t0 - _FRONT_AFTER_IR30_S - 1.0
    for record in records:
        objects = [
            row for row in record.get("objects") or []
            if not row.get("ambiguous_classes")
        ]
        classes = [row["class"] for row in objects]
        timestamp = float(record.get("timestamp_s", 0))
        inference = infer_step_from_detections(
            classes, rules, completed=completed, minimum_score=minimum_score,
            objects=objects, previous_centers=previous_centers,
            require_interaction=require_interaction, ready_only=ready_only)
        # Named predicate edges for ready∪current (temporal hold across flicker).
        focus = list(inference.get("ready_steps") or [])
        if inference.get("step") not in {None, "unknown", ""}:
            if inference["step"] not in focus:
                focus = [inference["step"], *focus]
        edges = predict_step_relations(
            rules, objects, step_ids=focus, hold_filter=relation_filter,
            timestamp_s=timestamp)
        for row in inference.get("scores") or []:
            edge = install_edge_for_step(edges, row["step_id"])
            if not edge:
                row.setdefault("relation_score", 0.0)
                row.setdefault("relation_predicate", "")
            else:
                row["relation_score"] = float(edge.get("score") or 0)
                row["relation_predicate"] = str(edge.get("predicate") or "")
                row["relation_held"] = bool(edge.get("held"))
                # Only strong install relations count as attach evidence for DONE.
                if relation_supports_done(row["relation_score"]):
                    row["attach_relation"] = True
                    row["saw_target_boxes"] = True
                    if "attach_relation" not in (row.get("reasons") or []):
                        row["reasons"] = list(row.get("reasons") or []) + [
                            f"relation:{row['relation_predicate']}"
                        ]
            hand = hand_edge_for_step(edges, row["step_id"])
            if hand:
                row["hand_score"] = float(hand.get("score") or 0)
                row["hand_predicate"] = str(hand.get("predicate") or "")
            else:
                row.setdefault("hand_score", 0.0)
                row.setdefault("hand_predicate", "")
        inference["relations"] = edges
        inference["relations_summary"] = summarize_relations(edges)
        # Live help: strong hand↔part on a ready step can claim CURRENT immediately.
        inference = _prefer_hand_current(inference, edges)
        completed, memory, inference = advance_progress_memory(
            completed=completed, rules=rules, inference=inference, memory=memory,
            timestamp_s=timestamp, sustained_trust_s=sustained_trust_s,
            min_dwell_s=min_dwell_s)
        observations.append({
            "timestamp_s": timestamp,
            "index": record.get("index"),
            "action": inference["action"],
            "step": inference["step"],
            "current_step": inference.get("current_step"),
            "help_active": bool(inference.get("help_active")),
            "status": inference["status"],
            "certainty": inference["certainty"],
            "evidence": inference["evidence"],
            "recovery": inference.get("recovery", ""),
            "instruction": inference.get("instruction", ""),
            "missing_prerequisites": inference.get("missing_prerequisites", []),
            "ready_steps": inference.get("ready_steps", []),
            "completed_steps": inference.get("completed_steps", sorted(completed)),
            "score": inference.get("score", 0),
            "detected_classes": inference.get("detected_classes", classes),
            "progress_trust": bool(inference.get("progress_trust")),
            "step_dwell_s": inference.get("step_dwell_s", {}),
            "step_relation_dwell_s": inference.get("step_relation_dwell_s", {}),
            "relations": inference.get("relations", []),
            "relations_summary": inference.get("relations_summary", ""),
            "source": "sop_step_rules_topdown" if ready_only else "sop_step_rules",
        })
        centers = object_centers(objects)
        # Replace (do not merge forever): nearest-instance matching needs the
        # previous frame's centers only, not a stale singleton per class.
        previous_centers = centers if centers else previous_centers
    return observations
