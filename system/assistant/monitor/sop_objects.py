"""Build detector object lists directly from an SOP.

Factory intent: the SOP names which physical objects belong on the station and
in each step. This module turns that language into:

* a validated per-step visual_plan (reviewed plan preferred; otherwise a
  deterministic draft from ``parts``);
* a station-wide object inventory;
* open-vocabulary detector prompts (YOLO-World / SAM3 class queries).

It does not claim that those objects are currently visible. Detection still has
to run on camera frames. Action recognition remains a separate module.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from .open_vocab import validate_part_prompts
from .sop_vision import visual_plan_from_step, yolo_prompts_from_plan


SLUG = re.compile(r"[^a-z0-9]+")
TARGET_HINTS = (
    "base", "chassis", "bracket", "joint", "connection", "receiver",
    "prepared", "target", "assembly location",
)
MOVING_HINTS = (
    "pin", "screw", "brace", "wheel", "nut", "washer", "hardware", "bracket",
    "assy", "assembly", "pulley", "wing",
)
FASTEN_HINTS = ("pin", "screw", "nut", "bolt", "fasten", "washer")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def slugify_entity_name(text: str, *, used: set[str] | None = None) -> str:
    """Convert SOP wording into a stable snake_case entity id."""
    words = SLUG.sub("_", _clean(text).lower()).strip("_")
    # Drop leading count words so "two short braces" → short_braces-ish.
    words = re.sub(r"^(?:\d+|two|three|four|five|a|an|the)_+", "", words)
    if not words or not words[0].isalpha():
        words = f"part_{words or 'object'}"
    words = words[:64]
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", words):
        words = "part_object"
    used = used if used is not None else set()
    candidate = words
    index = 2
    while candidate in used:
        suffix = f"_{index}"
        candidate = f"{words[:64 - len(suffix)]}{suffix}"
        index += 1
    used.add(candidate)
    return candidate


def _looks_like(text: str, hints: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def _target_score(text: str) -> int:
    """Prefer clear receivers (base/joint/connection) over the part being installed."""
    lowered = text.lower()
    score = 0
    if "base" in lowered:
        score += 6
    if "partial model" in lowered or "partial" in lowered:
        # Prepared assembly / backdrop — strong receiver for bracket install (IR21).
        score += 6
    if any(token in lowered for token in ("joint", "connection", "receiver")):
        score += 5
    if "prepared" in lowered:
        score += 4
    # Bracket is a receiver for screw/wheel steps, but weaker than partial model so
    # IR21 (install bracket onto model) keeps the bracket as the moving part.
    if "bracket" in lowered and not any(token in lowered for token in ("screw", "pin")):
        score += 3
    if re.search(r"\brear chassis\b", lowered) and "pin" not in lowered and "joint" not in lowered:
        score += 5
    if re.search(r"\bfront chassis\b", lowered) and "pin" not in lowered and "joint" not in lowered:
        score += 4
    if "main chassis" in lowered:
        # Backdrop / already-built car — never preferred over a specific receiver.
        score -= 2
    if "chassis" in lowered:
        score += 1
    if any(token in lowered for token in ("pin", "screw", "brace", "wheel", "hardware", "washer", "nut")):
        score -= 4
    return score


def _moving_score(text: str) -> int:
    lowered = text.lower()
    score = 0
    if any(token in lowered for token in ("pin", "screw", "brace", "wheel", "pulley")):
        score += 5
    if re.search(r"\bwing\b", lowered) or "wing beam" in lowered:
        score += 4
    if any(token in lowered for token in ("nut", "washer", "hardware")):
        # Support fasteners help detection but should not steal the moving role
        # from the pin/brace/wheel the SOP is installing.
        score += 1
    if "assembly" in lowered or "assy" in lowered:
        score += 3
    if "bracket" in lowered and "connection" not in lowered:
        score += 4
    if "partial model" in lowered or "partial" in lowered:
        score -= 4
    if "base" in lowered or "joint" in lowered or "connection" in lowered:
        score -= 3
    return score


def _step_corpus(step: dict) -> str:
    return " ".join([
        _clean(step.get("instruction")),
        _clean(step.get("completion_check")),
        *[_clean(value) for value in step.get("parts", []) or []],
        *[_clean(value) for value in step.get("tools", []) or []],
    ]).lower()


def draft_visual_plan_from_parts(step: dict) -> dict:
    """Deterministically draft a minimal visual_plan from SOP parts/instruction.

    This is a PIC-review candidate, not factory-approved identity. Evidence
    strings are copied from the SOP so :func:`visual_plan_from_step` can trace
    every entity back to station language.
    """
    if not isinstance(step, dict):
        raise ValueError("Each SOP step must be an object.")
    parts = [_clean(value) for value in step.get("parts", []) or [] if _clean(value)]
    tools = [_clean(value) for value in step.get("tools", []) or [] if _clean(value)]
    instruction = _clean(step.get("instruction"))
    if len(parts) < 2:
        raise ValueError(
            f"Step {step.get('id', '?')} needs at least two parts to draft objects "
            "from the SOP without a reviewed visual_plan."
        )

    target_index = max(range(len(parts)), key=lambda index: (_target_score(parts[index]), index))
    moving_candidates = [index for index in range(len(parts)) if index != target_index]
    if not moving_candidates:
        raise ValueError(f"Step {step.get('id', '?')} cannot separate moving and target parts.")
    moving_index = max(
        moving_candidates,
        key=lambda index: (_moving_score(parts[index]), -index),
    )
    if _target_score(parts[target_index]) <= 0 and _moving_score(parts[moving_index]) <= 0:
        # Fall back to first/last when wording is too generic.
        moving_index, target_index = 0, len(parts) - 1
        if moving_index == target_index:
            raise ValueError(f"Step {step.get('id', '?')} cannot separate moving and target parts.")

    used: set[str] = set()
    moving_name = slugify_entity_name(parts[moving_index], used=used)
    target_name = slugify_entity_name(parts[target_index], used=used)
    verb = "fasten" if _looks_like(parts[moving_index], FASTEN_HINTS) else "attach"
    entities = [
        {
            "name": moving_name,
            "role": "moving",
            "description": f"The handled SOP part: {parts[moving_index]}.",
            "sop_evidence": [parts[moving_index]],
        },
        {
            "name": target_name,
            "role": "target",
            "description": f"The receiving SOP part: {parts[target_index]}.",
            "sop_evidence": [parts[target_index]],
        },
    ]
    tool_names = []
    for tool in tools[:4]:
        if tool.lower() not in _step_corpus({"parts": parts, "tools": tools,
                                             "instruction": instruction}):
            continue
        name = slugify_entity_name(tool, used=used)
        entities.append({
            "name": name,
            "role": "tool",
            "description": f"SOP tool: {tool}.",
            "sop_evidence": [tool],
        })
        tool_names.append(name)

    plan = {
        "action": {"verb": verb, "subject": moving_name, "object": target_name},
        "entities": entities,
        "tools": tool_names,
        "completion_evidence": [
            {
                "id": "part_at_target",
                "description": f"{parts[moving_index]} remains at {parts[target_index]}.",
                "source": "sam2_temporal_relation",
            },
            {
                "id": "correct_final_state",
                "description": (
                    _clean(step.get("completion_check"))
                    or f"{parts[moving_index]} is correctly located on {parts[target_index]}."
                ),
                "source": "vlm_visual_check",
            },
        ],
        "required_evidence": ["part_at_target", "correct_final_state"],
        "draft_source": "deterministic_parts_from_sop",
        "requires_pic_review": True,
    }
    return visual_plan_from_step({**step, "visual_plan": plan})


def ensure_step_visual_plan(step: dict) -> dict:
    """Return a validated visual_plan, drafting from parts when absent."""
    if isinstance(step.get("visual_plan"), dict):
        return visual_plan_from_step(step)
    return draft_visual_plan_from_parts(step)


def enrich_sop_with_visual_plans(sop: dict) -> dict:
    """Copy an SOP and ensure every step has a validated visual_plan."""
    if not isinstance(sop, dict) or not isinstance(sop.get("steps"), list):
        raise ValueError("SOP must contain a steps array.")
    output = deepcopy(sop)
    for step in output["steps"]:
        step["visual_plan"] = ensure_step_visual_plan(step)
        if not step.get("parts"):
            # Keep parts discoverable for attribute checks and planners.
            step["parts"] = [
                evidence
                for entity in step["visual_plan"]["entities"]
                for evidence in entity.get("sop_evidence", [])
            ]
    return output


def detector_prompts_for_step(step: dict) -> list[dict]:
    """Build open-vocabulary prompts for one SOP step's objects."""
    plan = ensure_step_visual_plan(step)
    enriched = {**step, "visual_plan": plan}
    rows = []
    for row in yolo_prompts_from_plan(plan):
        role = row["role"] if row["role"] in {"moving", "target", "context"} else "context"
        rows.append({**row, "role": role})
    return validate_part_prompts(rows, enriched)


def station_object_inventory(sop: dict) -> dict:
    """Union of every detectable object named by the SOP."""
    enriched = enrich_sop_with_visual_plans(sop)
    objects: dict[str, dict] = {}
    for step in enriched["steps"]:
        step_id = str(step.get("id") or "")
        plan = step["visual_plan"]
        for entity in plan["entities"]:
            name = entity["name"]
            row = objects.setdefault(name, {
                "name": name,
                "role": entity["role"],
                "description": entity["description"],
                "sop_evidence": [],
                "steps": [],
                "detector_prompt": name.replace("_", " "),
            })
            for evidence in entity.get("sop_evidence", []):
                if evidence not in row["sop_evidence"]:
                    row["sop_evidence"].append(evidence)
            if step_id and step_id not in row["steps"]:
                row["steps"].append(step_id)
            # Prefer a concrete evidence phrase as the detector query.
            if row["sop_evidence"]:
                row["detector_prompt"] = row["sop_evidence"][0]
    return {
        "title": enriched.get("title"),
        "step_count": len(enriched["steps"]),
        "object_count": len(objects),
        "objects": sorted(objects.values(), key=lambda row: row["name"]),
        "steps": [
            {
                "id": step.get("id"),
                "instruction": step.get("instruction"),
                "parts": list(step.get("parts") or []),
                "entity_names": [entity["name"] for entity in step["visual_plan"]["entities"]],
                "action": step["visual_plan"]["action"],
                "detector_prompts": detector_prompts_for_step(step),
            }
            for step in enriched["steps"]
        ],
        "enriched_sop": enriched,
    }


def station_detector_classes(
    sop: dict,
    *,
    include_hand: bool = True,
    include_generic_tools: bool = False,
    include_paper: bool = True,
    include_ar_objects: bool = True,
    maximum: int = 40,
) -> list[dict]:
    """Station-wide class list for text-prompted detectors (SAM3 / YOLO-World).

    Classes come from SOP ``parts`` (with SAM3 prompts). Generic tools such as
    screwdriver/wrench are off by default. ``hand`` / ``paper`` are optional
    station context (interaction + printed-diagram suppression).
    """
    from .ar_objects import ar_detector_classes_for_sop, detector_prompt_for_part, enrich_sop_parts_with_ar_objects

    enriched = enrich_sop_parts_with_ar_objects(sop) if include_ar_objects else sop
    classes: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, prompt: str, *, entity: str, role: str, steps: list, source: str) -> None:
        key = name.casefold()
        if key in seen:
            for existing in classes:
                if existing["name"].casefold() == key:
                    for step_id in steps:
                        if step_id not in existing["steps"]:
                            existing["steps"].append(step_id)
                    break
            return
        if len(classes) >= maximum:
            return
        seen.add(key)
        classes.append({
            "name": name[:50],
            "prompt": prompt[:80],
            "entity": entity,
            "role": role,
            "steps": list(steps),
            "source": source,
        })

    for row in ar_detector_classes_for_sop(enriched):
        _add(
            row["name"], row["prompt"],
            entity=row["entity"], role=row["role"],
            steps=list(row["steps"]), source=row.get("source", "sop_parts"))

    if include_hand and "hand" not in seen:
        _add("hand", detector_prompt_for_part("hand"),
             entity="hand", role="context", steps=[],
             source="generic interaction context")
    if include_paper and "paper" not in seen:
        _add("paper", detector_prompt_for_part("paper"),
             entity="paper", role="context", steps=[],
             source="station context for printed-diagram suppression")
    if include_generic_tools:
        for tool in ("screwdriver", "wrench"):
            if tool not in seen:
                _add(tool, tool, entity=tool, role="context", steps=[],
                     source="generic tool search hypothesis, not SOP-required or confirmed present")
    return classes


def detection_plan_from_sop(sop: dict, *, sop_path: str | None = None) -> dict:
    """Build the automatic-assembly style plan.json payload from SOP objects."""
    inventory = station_object_inventory(sop)
    classes = station_detector_classes(sop)
    return {
        "classes": classes,
        "sop": sop_path,
        "steps": [
            {
                "id": step["id"],
                "instruction": step["instruction"],
                "parts": step["parts"],
                "prerequisites": list(
                    next(row for row in inventory["enriched_sop"]["steps"]
                         if row.get("id") == step["id"]).get("prerequisites", [])
                ),
                "completion_check": next(
                    row for row in inventory["enriched_sop"]["steps"]
                    if row.get("id") == step["id"]
                ).get("completion_check", ""),
                "entities": step["entity_names"],
                "action": step["action"],
            }
            for step in inventory["steps"]
        ],
        "object_inventory": {
            "object_count": inventory["object_count"],
            "objects": [
                {key: row[key] for key in ("name", "role", "detector_prompt", "steps", "sop_evidence")}
                for row in inventory["objects"]
            ],
        },
        "runtime_clicks": 0,
        "reference_images": 0,
        "ground_truth_used": False,
        "planner": "sop_object_inventory",
        "identity_source": "SOP parts, visual_plan entities, and IndustReal AR object nouns",
        "automatic_object_recognition": False,
        "note": (
            "Class names come from the SOP plus AR-aligned fasteners (nut/washer/pin). "
            "They are search queries for a detector, not proof that those objects are visible."
        ),
    }
