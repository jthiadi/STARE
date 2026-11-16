"""Restrict SAM3 search to parts for ready / current SOP steps.

Default policy (factory intent):
* search only parts belonging to prerequisite-ready steps (and the active
  current step if set);
* do **not** peek at successor-step parts until those steps become ready
  after completion.

Optional ``include_next=True`` keeps a one-step successor lookahead for
experiments.
"""
from __future__ import annotations

from typing import Any


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def step_parts_index(sop: dict) -> dict[str, list[str]]:
    """Map step id → ordered unique part names from the SOP."""
    index: dict[str, list[str]] = {}
    for step in sop.get("steps") or []:
        step_id = str(step.get("id") or "")
        if not step_id:
            continue
        parts: list[str] = []
        seen: set[str] = set()
        for part in step.get("parts") or []:
            name = _clean(part)
            key = name.casefold()
            if not name or key in seen:
                continue
            seen.add(key)
            parts.append(name)
        index[step_id] = parts
    return index


def ready_step_ids(sop: dict, completed: set[str]) -> list[str]:
    """Prerequisite-ready, not-yet-completed steps (stable SOP order)."""
    done = {str(value) for value in completed}
    ready = []
    for step in sop.get("steps") or []:
        step_id = str(step.get("id") or "")
        if not step_id or step_id in done:
            continue
        prerequisites = {str(value) for value in (step.get("prerequisites") or [])}
        if prerequisites <= done:
            ready.append(step_id)
    return ready


def successor_step_ids(
    sop: dict,
    step_ids: set[str] | list[str],
    *,
    completed: set[str] | None = None,
) -> list[str]:
    """Steps that become ready once ``step_ids`` finish (all other prereqs done).

    Used only when ``include_next`` lookahead is enabled.
    """
    parents = {str(value) for value in step_ids}
    done = {str(value) for value in (completed or set())}
    allowed = done | parents
    if not parents:
        return []
    successors = []
    for step in sop.get("steps") or []:
        step_id = str(step.get("id") or "")
        if not step_id or step_id in allowed:
            continue
        prerequisites = {str(value) for value in (step.get("prerequisites") or [])}
        if prerequisites & parents and prerequisites <= allowed:
            successors.append(step_id)
    return successors


def scoped_step_ids(
    sop: dict,
    *,
    completed: set[str],
    current_step: str | None = None,
    include_next: bool = False,
) -> dict[str, list[str]]:
    """Return ready / current / next step id lists for detector scoping.

    Default ``include_next=False``: search = ready ∪ current only.
    """
    ready = ready_step_ids(sop, completed)
    current = []
    if current_step and current_step not in {None, "unknown", ""}:
        current = [str(current_step)]
    next_ids: list[str] = []
    if include_next:
        # Optional lookahead: successors of the focus set (not default).
        focus = set(current) or set(ready)
        next_ids = [
            step_id for step_id in successor_step_ids(sop, focus, completed=completed)
            if step_id not in set(ready) | set(current)
        ]
        for step_id in ready:
            if step_id not in current and step_id not in next_ids:
                next_ids.append(step_id)
    # Deduplicate while preserving order: current, then ready, then optional next.
    ordered: list[str] = []
    for step_id in current + ready + next_ids:
        if step_id not in ordered:
            ordered.append(step_id)
    return {
        "ready": ready,
        "current": current,
        "next": [step_id for step_id in next_ids if step_id not in set(current)],
        "search": ordered,
    }


def parts_for_steps(sop: dict, step_ids: list[str] | set[str]) -> list[str]:
    index = step_parts_index(sop)
    parts: list[str] = []
    seen: set[str] = set()
    for step_id in step_ids:
        for name in index.get(str(step_id), []):
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            parts.append(name)
    return parts


def filter_plan_classes(
    plan_classes: list[dict],
    *,
    allowed_names: set[str] | list[str],
    always: tuple[str, ...] = ("hand", "paper"),
) -> list[dict]:
    """Keep plan rows whose names are allowed (plus always-on context classes)."""
    allowed = {str(name).casefold() for name in allowed_names}
    for name in always:
        allowed.add(name.casefold())
    return [
        row for row in plan_classes
        if str(row.get("name") or "").casefold() in allowed
    ]


def scoped_plan_classes(
    sop: dict,
    plan_classes: list[dict],
    *,
    completed: set[str],
    current_step: str | None = None,
    include_next: bool = False,
    always: tuple[str, ...] = ("hand", "paper"),
    always_parts: tuple[str, ...] | list[str] = (),
) -> tuple[list[dict], dict]:
    """Classes to query this frame + scope metadata for logging.

    ``always_parts`` are extra SOP part names searched every frame (e.g. small
    fasteners) so sparse detectors can still see pins before their step is ready.
    """
    scope = scoped_step_ids(
        sop, completed=completed, current_step=current_step, include_next=include_next,
    )
    part_names = parts_for_steps(sop, scope["search"])
    extra = []
    seen = {name.casefold() for name in part_names}
    for name in always_parts:
        text = _clean(name)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        extra.append(text)
    part_names = list(part_names) + extra
    classes = filter_plan_classes(plan_classes, allowed_names=part_names, always=always)
    meta = {
        **scope,
        "parts": part_names,
        "class_names": [row.get("name") for row in classes],
        "include_next": include_next,
        "always_parts": list(extra),
    }
    return classes, meta
