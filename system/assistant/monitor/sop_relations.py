"""SOP step relations: named predicates + geometry scores + temporal hold.

Inspired by RelateAnything (predicate vocabulary as input; hold edges when an
endpoint briefly disappears). This module does **not** run their neural model —
it scores spatial predicates from boxes so DONE timing and overlays stay
deterministic and offline.

Public contract used by the step reasoner and the assembly overlay:

* ``predict_step_relations`` — best subject→predicate→object edges for steps
* ``RelationHoldFilter`` — keeps a calibrated score alive across short misses
* ``relation_supports_done`` — whether banked relation is strong enough for DONE
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


# Spatial predicates we can score from boxes alone (RelateAnything-style names).
SPATIAL_PREDICATES = ("attached to", "on", "through", "near", "holding")

# Hand ↔ part predicates (step *start* / in-progress), separate from install edges.
HAND_PREDICATES = ("holding", "near")

# Per-step preferred predicates (from SOP completion wording).
STEP_PREDICATES: dict[str, tuple[str, ...]] = {
    "IR03": ("attached to", "on", "fitted to"),
    "IR06": ("through", "attached to", "on"),
    "IR09": ("attached to", "on", "fitted to"),
    "IR15": ("through", "attached to", "on"),
    "IR18": ("through", "attached to", "on"),
    "IR21": ("on", "attached to"),
    "IR24": ("through", "attached to", "on"),
    "IR27": ("on", "attached to"),
    "IR30": ("on", "attached to"),
}

# Map free SOP wording onto scored spatial predicates.
_PREDICATE_ALIASES = {
    "fitted to": "attached to",
    "installed on": "on",
    "located on": "on",
    "located at": "on",
    "located through": "through",
}

# Geometry → score thresholds for DONE.
_DONE_SCORE = 0.55
_STRONG_SCORE = 0.72
_NEAR_SCORE = 0.35
# Hold a positive edge this long after one endpoint disappears (RelateAnything idea).
# Keep short so overlay lines do not trail far behind moving parts.
_HOLD_S = 0.7
# Soft decay of held score after the hold window.
_HOLD_DECAY_PER_S = 0.55


def _norm(value: str) -> str:
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


def _center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _diag(box: list[float]) -> float:
    return ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    return inter / max(1e-9, _area(a) + _area(b) - inter)


def _containment(inner: list[float], outer: list[float]) -> float:
    """Fraction of inner box area covered by outer."""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(1e-9, _area(inner))


def _aspect(box: list[float]) -> float:
    w = max(1e-6, box[2] - box[0])
    h = max(1e-6, box[3] - box[1])
    return max(w / h, h / w)


def canonical_predicate(name: str) -> str:
    text = _norm(name)
    return _PREDICATE_ALIASES.get(text, text)


def geometric_predicate_scores(subject: list[float], obj: list[float]) -> dict[str, float]:
    """Score spatial predicates from two boxes (0..1). No class labels required."""
    iou = _iou(subject, obj)
    cont_s_in_o = _containment(subject, obj)
    cont_o_in_s = _containment(obj, subject)
    cx, cy = _center(subject)
    ox, oy = _center(obj)
    dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
    reach = max(_diag(subject), _diag(obj))
    prox = max(0.0, 1.0 - dist / max(1e-6, reach * 0.85))
    thin = min(1.0, max(0.0, (_aspect(subject) - 1.2) / 3.0))

    attached = min(1.0, 0.60 * iou + 0.40 * max(cont_s_in_o, cont_o_in_s) + 0.40 * prox)
    on = min(1.0, 0.40 * cont_s_in_o + 0.25 * iou + 0.40 * prox)
    # "through": elongated subject overlapping the target (pin / screw).
    through = min(1.0, 0.50 * iou + 0.35 * cont_s_in_o + 0.30 * thin * prox)
    near = min(1.0, 0.15 * iou + 0.85 * prox)

    return {
        "attached to": round(attached, 4),
        "on": round(on, 4),
        "through": round(through, 4),
        "near": round(near, 4),
        "holding": round(min(1.0, 0.55 * near + 0.45 * attached), 4),
    }


def preferred_predicates(step_id: str) -> list[str]:
    raw = STEP_PREDICATES.get(str(step_id), ("attached to", "on", "near"))
    out = []
    for name in raw:
        canon = canonical_predicate(name)
        if canon in SPATIAL_PREDICATES and canon not in out:
            out.append(canon)
    if "near" not in out:
        out.append("near")
    return out


class RelationHoldFilter:
    """Hold relation scores briefly when a box endpoint disappears.

    Mirrors RelateAnything's temporal filter on calibrated scores: an edge that
    was confident stays available for DONE/overlay for ``hold_s`` after miss.
    """

    def __init__(self, *, hold_s: float = _HOLD_S, decay_per_s: float = _HOLD_DECAY_PER_S):
        self.hold_s = float(hold_s)
        self.decay_per_s = float(decay_per_s)
        self._rows: dict[str, dict[str, Any]] = {}

    def update(
        self,
        key: str,
        *,
        timestamp_s: float,
        score: float | None,
        payload: dict | None = None,
    ) -> dict:
        prev = self._rows.get(key)
        if score is not None and score >= _NEAR_SCORE:
            row = {
                "score": float(score),
                "raw_score": float(score),
                "last_seen_s": float(timestamp_s),
                "held": False,
                "payload": deepcopy(payload or {}),
            }
            self._rows[key] = row
            return row
        if not prev:
            return {
                "score": 0.0,
                "raw_score": 0.0,
                "last_seen_s": None,
                "held": False,
                "payload": {},
            }
        gap = float(timestamp_s) - float(prev.get("last_seen_s") or timestamp_s)
        if gap <= self.hold_s:
            row = {
                **prev,
                "score": float(prev["score"]),
                "raw_score": 0.0,
                "held": True,
            }
            self._rows[key] = row
            return row
        # Soft decay after hold window.
        decayed = max(0.0, float(prev["score"]) - (gap - self.hold_s) * self.decay_per_s)
        if decayed < 0.05:
            self._rows.pop(key, None)
            return {
                "score": 0.0,
                "raw_score": 0.0,
                "last_seen_s": prev.get("last_seen_s"),
                "held": False,
                "payload": {},
            }
        row = {
            **prev,
            "score": decayed,
            "raw_score": 0.0,
            "held": True,
        }
        self._rows[key] = row
        return row

    def get(self, key: str) -> dict:
        return dict(self._rows.get(key) or {
            "score": 0.0, "raw_score": 0.0, "last_seen_s": None, "held": False, "payload": {},
        })


def _match_queries(name: str, queries: list[str]) -> bool:
    from monitor.step_rules import _class_matches
    return any(_class_matches(query, {name}) for query in queries)


def _labeled_boxes(objects: list[dict], queries: list[str]) -> list[dict]:
    rows = []
    for obj in objects or []:
        name = str(obj.get("class") or "")
        if obj.get("ambiguous_classes"):
            continue
        if not _match_queries(name, queries):
            continue
        box = _as_box(obj.get("box"))
        if box is None:
            continue
        rows.append({"class": name, "box": box, "id": obj.get("id")})
    return rows


def best_edge_for_step(
    rule: dict,
    objects: list[dict] | None,
) -> dict | None:
    """Pick the strongest preferred install predicate between moving/fastener and target."""
    rows = list(objects or [])
    subjects = _labeled_boxes(rows, list(rule.get("moving_queries") or []))
    subjects += _labeled_boxes(rows, list(rule.get("fastener_queries") or []))
    targets = _labeled_boxes(rows, list(rule.get("target_queries") or []))
    if not subjects or not targets:
        return None
    predicates = preferred_predicates(rule.get("step_id") or "")
    best = None
    for subject in subjects:
        for target in targets:
            scores = geometric_predicate_scores(subject["box"], target["box"])
            for predicate in predicates:
                score = float(scores.get(predicate) or 0.0)
                candidate = {
                    "step_id": rule.get("step_id"),
                    "kind": "install",
                    "predicate": predicate,
                    "score": score,
                    "subject": subject["class"],
                    "object": target["class"],
                    "subject_box": subject["box"],
                    "object_box": target["box"],
                    "subject_id": subject.get("id"),
                    "object_id": target.get("id"),
                    "scores": scores,
                }
                if best is None or score > best["score"]:
                    best = candidate
    return best


def best_hand_edge_for_step(
    rule: dict,
    objects: list[dict] | None,
) -> dict | None:
    """Pick the strongest hand↔moving/fastener edge (step-start / in-progress)."""
    rows = list(objects or [])
    hands = _labeled_boxes(rows, ["hand"])
    parts = _labeled_boxes(rows, list(rule.get("moving_queries") or []))
    parts += _labeled_boxes(rows, list(rule.get("fastener_queries") or []))
    if not hands or not parts:
        return None
    best = None
    for hand in hands:
        for part in parts:
            scores = geometric_predicate_scores(hand["box"], part["box"])
            for predicate in HAND_PREDICATES:
                score = float(scores.get(predicate) or 0.0)
                candidate = {
                    "step_id": rule.get("step_id"),
                    "kind": "hand",
                    "predicate": predicate,
                    "score": score,
                    "subject": hand["class"],
                    "object": part["class"],
                    "subject_box": hand["box"],
                    "object_box": part["box"],
                    "subject_id": hand.get("id"),
                    "object_id": part.get("id"),
                    "scores": scores,
                }
                if best is None or score > best["score"]:
                    best = candidate
    return best


def _hold_edge(
    *,
    hold_filter: RelationHoldFilter | None,
    key: str,
    live: dict | None,
    timestamp_s: float,
    min_score: float,
) -> dict | None:
    """Apply optional temporal hold and return a drawable edge, or None."""
    if hold_filter is None:
        if live and float(live.get("score") or 0) >= min_score:
            return {**{k: v for k, v in live.items() if k != "scores"}, "held": False,
                    "raw_score": live["score"]}
        return None
    score = float(live["score"]) if live else None
    payload = live
    if live is None or float(live["score"]) < min_score:
        score = None
        payload = None
    held = hold_filter.update(
        key,
        timestamp_s=timestamp_s,
        score=score,
        payload={k: v for k, v in (payload or {}).items() if k != "scores"},
    )
    if float(held.get("score") or 0) < min_score:
        return None
    body = dict(held.get("payload") or {})
    if not body and live:
        body = {k: v for k, v in live.items() if k != "scores"}
    if not body:
        return None
    return {
        **body,
        "step_id": body.get("step_id") or (live or {}).get("step_id"),
        "score": round(float(held["score"]), 4),
        "raw_score": round(float(held.get("raw_score") or 0), 4),
        "held": bool(held.get("held")),
        "predicate": body.get("predicate") or (live or {}).get("predicate") or "near",
        "kind": body.get("kind") or (live or {}).get("kind") or "install",
    }


def predict_step_relations(
    rules: list[dict],
    objects: list[dict] | None,
    *,
    step_ids: list[str] | None = None,
    hold_filter: RelationHoldFilter | None = None,
    timestamp_s: float = 0.0,
    min_score: float = _NEAR_SCORE,
) -> list[dict]:
    """Return install + hand relation edges for the requested steps."""
    by_id = {rule["step_id"]: rule for rule in rules}
    wanted = list(step_ids) if step_ids is not None else [rule["step_id"] for rule in rules]
    edges = []
    for step_id in wanted:
        rule = by_id.get(step_id)
        if not rule:
            continue
        install = _hold_edge(
            hold_filter=hold_filter,
            key=f"step:{step_id}:install",
            live=best_edge_for_step(rule, objects),
            timestamp_s=timestamp_s,
            min_score=min_score,
        )
        if install:
            edges.append(install)
        hand = _hold_edge(
            hold_filter=hold_filter,
            key=f"step:{step_id}:hand",
            live=best_hand_edge_for_step(rule, objects),
            timestamp_s=timestamp_s,
            min_score=min_score,
        )
        if hand:
            edges.append(hand)
    edges.sort(key=lambda row: (
        0 if row.get("kind") == "hand" else 1,
        -float(row.get("score") or 0),
        row.get("step_id") or "",
    ))
    return edges


def relation_supports_done(score: float, *, strong: bool = False) -> bool:
    return float(score) >= (_STRONG_SCORE if strong else _DONE_SCORE)


def install_edge_for_step(edges: list[dict], step_id: str) -> dict | None:
    """Best install (part↔target) edge for DONE gating; ignores hand edges."""
    candidates = [
        edge for edge in edges
        if edge.get("step_id") == step_id and edge.get("kind") != "hand"
        and edge.get("predicate") not in HAND_PREDICATES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get("score") or 0))


def hand_edge_for_step(edges: list[dict], step_id: str) -> dict | None:
    """Best hand↔part edge for hand-off / CURRENT; ignores install edges."""
    candidates = [
        edge for edge in edges
        if edge.get("step_id") == step_id
        and (
            edge.get("kind") == "hand"
            or edge.get("predicate") in HAND_PREDICATES
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get("score") or 0))


def summarize_relations(edges: list[dict]) -> str:
    if not edges:
        return "Relations: (none)"
    bits = []
    for edge in edges[:6]:
        mark = "HOLD" if edge.get("held") else f"{float(edge.get('score') or 0):.2f}"
        bits.append(
            f"{edge.get('step_id')}:{edge.get('subject')} -{edge.get('predicate')}-> "
            f"{edge.get('object')} [{mark}]"
        )
    return "Relations: " + " | ".join(bits)


def _nearest_box_for_label(
    objects: list[dict] | None,
    class_name: str,
    prefer_id: Any = None,
    near_box: list[float] | None = None,
) -> list[float] | None:
    """Find a current box for a relation endpoint (id first, else nearest same class)."""
    want = _norm(class_name)
    if not want:
        return None
    rows = []
    for obj in objects or []:
        if obj.get("ambiguous_classes"):
            continue
        name = _norm(str(obj.get("class") or ""))
        if name != want:
            continue
        box = _as_box(obj.get("box"))
        if box is None:
            continue
        if prefer_id is not None and obj.get("id") == prefer_id:
            return box
        rows.append(box)
    if not rows:
        return None
    if near_box is None:
        return rows[0]
    nx, ny = _center(near_box)
    return min(rows, key=lambda box: (_center(box)[0] - nx) ** 2 + (_center(box)[1] - ny) ** 2)


def refresh_relation_edges_for_frame(
    edges: list[dict],
    objects: list[dict] | None,
) -> list[dict]:
    """Re-anchor relation lines to current detections so HOLD does not trail.

    Kept edges still count for the footer/summary, but drawable geometry is
    updated from the live frame. If an endpoint is missing, the edge is marked
    ``draw=False`` so the overlay skips the stale line.
    """
    refreshed = []
    for edge in edges or []:
        row = dict(edge)
        subject = _nearest_box_for_label(
            objects,
            str(row.get("subject") or ""),
            prefer_id=row.get("subject_id"),
            near_box=_as_box(row.get("subject_box")),
        )
        obj = _nearest_box_for_label(
            objects,
            str(row.get("object") or ""),
            prefer_id=row.get("object_id"),
            near_box=_as_box(row.get("object_box")),
        )
        if subject is not None:
            row["subject_box"] = subject
        if obj is not None:
            row["object_box"] = obj
        # Never draw a held edge with missing/stale-only geometry.
        if subject is None or obj is None:
            row["draw"] = False
        else:
            row["draw"] = True
            if row.get("held"):
                row["held"] = False  # live boxes → solid line at current pose
        refreshed.append(row)
    return refreshed
