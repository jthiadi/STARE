"""IndustReal action-recognition object nouns for SOP detection.

These names come from the public IndustReal AR vocabulary (take_nut, fit_washer,
…), not from a specific recording's labels. Inference must never open
``data/evaluation/official/...`` AR CSVs.

The enriched SOP (`data/industreal_assembly.template.json`) is the source of
truth for which parts to search. This module supplies SAM3 prompts, optional
gap-fill, and soft activation fasteners.
"""
from __future__ import annotations

from typing import Any


_PART_PROMPTS_RAW: dict[str, str] = {
    "hex nut": "small pink hexagonal plastic nut fastener",
    "acorn nut": "small pink acorn cap nut fastener",
    "tooth lock washer": "small pink tooth lock washer fastener",
    "round flat washer": "small thin metal flat washer fastener",
    # T-pins: T-handle + smooth shaft. Do not say "screw pin" — SAM matches that
    # phrase to every grey rod and then the hygiene filter used to lock the label.
    "short metal pin": "short grey T-handle pin with a smooth unthreaded shaft",
    "middle metal pin": "medium grey T-handle pin, longer smooth unthreaded shaft",
    "long metal pin": "long grey T-handle pin with a long smooth unthreaded shaft",
    "screw pin": "small grey machine screw with a threaded shaft and screw head",
    "short brace": "short white perforated plastic brace bar with holes",
    "long brace": "long white perforated plastic brace bar with holes",
    # In-progress assembly body on this video (not a separate chassis-base class).
    "partial model": (
        "white perforated plastic toy car chassis body in progress of assembly"
    ),
    "toy car wheel": "black rubber toy car wheel with tire",
    "white front bracket": (
        "small white plastic U-shaped front bracket with holes for the wheel axle"
    ),
    # Do not say "pulley wheel" — SAM then labels black tires as pulleys.
    "pulley": "small white grooved plastic belt pulley, not a rubber tire",
    # One class: official AR splits take_wing vs take_wing_beam, but they are
    # the same rear-assembly plastic piece family and must not be two SAM queries.
    "wing": "white plastic toy-car wing piece including the wing-beam bar",
    "wing beam": "white plastic toy-car wing piece including the wing-beam bar",
    "front wheel assembly": "front wheel assembly",
    "rear wheel assembly": "rear wheel assembly",
    # Legacy aliases kept for hygiene / old detections only — not SOP search classes.
    "chassis base": "plastic chassis base",
    "front chassis": "front chassis",
    "rear chassis": "rear chassis",
    "hand": "human hand",
    "paper": "printed paper assembly instruction sheet with diagrams",
}

# Extra prompt variants tried when a READY step needs rare small parts.
PART_PROMPT_VARIANTS: dict[str, tuple[str, ...]] = {
    "white front bracket": (
        "small white plastic U-shaped front bracket with holes for the wheel axle",
        "white plastic steering bracket for toy car front wheel",
        "small white perforated plastic front mount bracket",
    ),
    "short metal pin": (
        "short grey T-handle pin with a smooth unthreaded shaft",
        "small grey T-shaped pin lying on the table",
    ),
    "middle metal pin": (
        "medium grey T-handle pin, longer smooth unthreaded shaft",
        "medium-length grey T-shaped pin rod",
    ),
    "long metal pin": (
        "long grey T-handle pin with a long smooth unthreaded shaft",
        "long grey T-shaped pin rod on the table",
    ),
    "screw pin": (
        "small grey machine screw with a threaded shaft and screw head",
        "tiny grey threaded screw axle, not a T-handle pin",
    ),
    "pulley": (
        "small white grooved plastic belt pulley, not a rubber tire",
        "tiny white plastic grooved pulley disc",
    ),
}

PART_DETECTOR_PROMPTS: dict[str, str] = {
    key.casefold(): value for key, value in _PART_PROMPTS_RAW.items()
}

AR_OBJECT_NAMES: dict[str, str] = {
    "nut": "hex nut",
    "acorn_nut": "acorn nut",
    "tooth_washer": "tooth lock washer",
    "round_washer": "round flat washer",
    "short_pin": "short metal pin",
    "middle_pin": "middle metal pin",
    "long_pin": "long metal pin",
    "screw_pin": "screw pin",
    "short_brace": "short brace",
    "long_brace": "long brace",
    "partial_model": "partial model",
    "wheel": "toy car wheel",
    "pulley": "pulley",
    "wing_beam": "wing",
    "wing": "wing",
}

# Display-name aliases collapsed before SAM search / SOP class lists.
PART_CLASS_ALIASES: dict[str, str] = {
    "wing beam": "wing",
}

AR_OBJECT_PROMPTS: dict[str, str] = {
    key: PART_DETECTOR_PROMPTS[AR_OBJECT_NAMES[key].casefold()]
    for key in AR_OBJECT_NAMES
}

# Gap-fill only when a step's parts omit these. Prefer editing the SOP instead.
STEP_AR_OBJECTS: dict[str, list[str]] = {
    "IR03": ["short_brace", "partial_model"],
    "IR06": ["short_pin", "middle_pin", "tooth_washer", "nut"],
    "IR09": ["long_brace", "partial_model"],
    "IR15": ["short_pin", "middle_pin", "tooth_washer", "nut"],
    "IR18": ["short_pin", "middle_pin", "tooth_washer", "nut"],
    "IR21": ["partial_model"],  # white front bracket comes from SOP parts
    "IR24": ["screw_pin", "round_washer", "acorn_nut"],
    "IR27": ["wheel", "nut"],
    "IR30": [
        "wheel", "pulley", "wing", "long_pin", "middle_pin",
        "nut", "tooth_washer", "partial_model",
    ],
}

STEP_ACTIVATION_FASTENERS: dict[str, list[str]] = {
    "IR06": ["tooth_washer", "nut"],
    "IR15": ["tooth_washer", "nut"],
    "IR18": ["tooth_washer", "nut"],
    "IR24": ["round_washer", "acorn_nut", "nut"],
}

HARDWARE_EXPANSIONS: dict[str, list[str]] = {
    "retaining hardware": ["tooth lock washer", "hex nut"],
    "connection hardware": ["hex nut", "acorn nut"],
}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def canonical_part_name(name: str) -> str:
    """Collapse detector aliases (e.g. wing beam → wing)."""
    text = _clean(name)
    if not text:
        return text
    return PART_CLASS_ALIASES.get(text.casefold(), text)


def detector_prompt_for_part(name: str) -> str:
    """Return the SAM3 text prompt for a SOP part display name."""
    text = canonical_part_name(name)
    if not text:
        return text
    return PART_DETECTOR_PROMPTS.get(text.casefold(), text)


def detector_prompt_variants_for_part(name: str) -> list[str]:
    """Primary prompt plus optional recall variants for rare parts."""
    text = canonical_part_name(name)
    if not text:
        return []
    primary = detector_prompt_for_part(text)
    variants = list(PART_PROMPT_VARIANTS.get(text.casefold()) or ())
    out = []
    for prompt in (primary, *variants):
        if prompt and prompt not in out:
            out.append(prompt)
    return out


def is_fastener_part(text: str) -> bool:
    """True for nut/washer-only parts, not pins/screws/braces/wheels."""
    lowered = _clean(text).lower()
    if not lowered:
        return False
    if any(token in lowered for token in (
        "pin", "screw", "brace", "wheel", "bracket", "chassis", "assy",
        "assembly", "tire", "model", "pulley", "wing",
    )):
        return False
    return any(token in lowered for token in ("nut", "washer"))


def expand_hardware_phrases(parts: list[str]) -> list[str]:
    """Expand vague hardware wording into detectable fastener names."""
    output: list[str] = []
    for part in parts:
        text = _clean(part)
        if not text:
            continue
        key = text.casefold()
        if key in HARDWARE_EXPANSIONS:
            for item in HARDWARE_EXPANSIONS[key]:
                item = canonical_part_name(item)
                if item.casefold() not in {row.casefold() for row in output}:
                    output.append(item)
            continue
        text = canonical_part_name(text)
        if text.casefold() not in {row.casefold() for row in output}:
            output.append(text)
    return output


def enrich_sop_parts_with_ar_objects(sop: dict) -> dict:
    """Copy an SOP and fill any missing necessary AR-aligned fasteners."""
    from copy import deepcopy

    output = deepcopy(sop)
    for step in output.get("steps") or []:
        step_id = str(step.get("id") or "")
        parts = expand_hardware_phrases(list(step.get("parts") or []))
        for key in STEP_AR_OBJECTS.get(step_id, []):
            name = canonical_part_name(AR_OBJECT_NAMES.get(key, key.replace("_", " ")))
            if name.casefold() not in {row.casefold() for row in parts}:
                parts.append(name)
        step["parts"] = parts
    return output


def _role_for_part(name: str) -> str:
    key = _clean(name).casefold()
    if is_fastener_part(name):
        return "tool"
    if "partial" in key or key == "paper" or key == "hand":
        return "context"
    if any(token in key for token in ("chassis", "base", "bracket", "joint", "connection")):
        if "wheel" not in key and "brace" not in key:
            return "target"
    return "moving"


def ar_detector_classes_for_sop(sop: dict) -> list[dict]:
    """Classes derived from SOP step parts, with SAM3 prompts attached."""
    by_name: dict[str, dict] = {}
    for step in sop.get("steps") or []:
        step_id = str(step.get("id") or "")
        for part in step.get("parts") or []:
            name = canonical_part_name(part)
            if not name:
                continue
            key = name.casefold()
            row = by_name.setdefault(key, {
                "name": name[:50],
                "prompt": detector_prompt_for_part(name)[:80],
                "entity": key.replace(" ", "_")[:40],
                "role": _role_for_part(name),
                "steps": [],
                "source": "sop_parts",
            })
            if step_id and step_id not in row["steps"]:
                row["steps"].append(step_id)
    return sorted(by_name.values(), key=lambda row: row["name"])


def fastener_queries_for_step(step_id: str) -> list[str]:
    """Display names / prompts used by step scoring for AR fasteners."""
    queries = []
    for key in STEP_ACTIVATION_FASTENERS.get(step_id, []):
        queries.append(AR_OBJECT_NAMES.get(key, key.replace("_", " ")))
        queries.append(AR_OBJECT_PROMPTS.get(key, key.replace("_", " ")))
    return list(dict.fromkeys(queries))
