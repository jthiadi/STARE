"""Validated SOP dependencies and event-driven procedure tracking (no ML required)."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any


def validate_sop(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("The SOP must be a JSON object.")
    # Accept a single SOPRAG graph as well as the monitoring interchange format.
    if "flow_graph" in value:
        value = {"title": value.get("procedure_card", {}).get("title", "Imported SOP"),
                 "steps": value["flow_graph"].get("steps", [])}
    steps = value.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 200:
        raise ValueError("The SOP needs between 1 and 200 steps.")
    result = {"title": str(value.get("title", "Untitled procedure"))[:200], "steps": []}
    # Preserve the reviewed SOP contract needed by the vision/verifier pipeline.
    # Execution still depends on the validated step IDs and graph below.
    for field in ("schema_version", "status", "station", "approval",
                  "execution_policy", "source_documents", "evaluation_sources",
                  "visual_reference_policy"):
        if field in value:
            result[field] = deepcopy(value[field])
    if 'camera_region' in value:
        region = value['camera_region']
        if (not isinstance(region, list) or len(region) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in region)
                or region[0] >= region[2] or region[1] >= region[3]):
            raise ValueError('camera_region must be normalized [left, top, right, bottom].')
        result['camera_region'] = list(region)
    ids = set()
    for raw in steps:
        if not isinstance(raw, dict):
            raise ValueError("Each step must be an object.")
        step_id = str(raw.get("id", "")).strip()
        instruction = str(raw.get("instruction", "")).strip()
        if not step_id or not instruction or step_id in ids:
            raise ValueError("Every step needs a unique ID and an instruction.")
        prerequisites = raw.get("prerequisites", [])
        if not isinstance(prerequisites, list):
            raise ValueError(f"{step_id}: prerequisites must be an array of step IDs.")
        ids.add(step_id)
        verification = raw.get("verification", "unspecified")
        if verification not in ("visual", "external", "unspecified"):
            raise ValueError(f"{step_id}: verification must be visual, external or unspecified.")
        for field in ("parts", "tools"):
            if not isinstance(raw.get(field, []), list) or any(not isinstance(x, str) for x in raw.get(field, [])):
                raise ValueError(f"{step_id}: {field} must be an array of names.")
        clean_step = {"id": step_id, "instruction": instruction,
                      "prerequisites": list(dict.fromkeys(str(p) for p in prerequisites)),
                      "recovery": str(raw.get("recovery") or "").strip(),
                      "parts": list(dict.fromkeys(raw.get("parts", [])))[:30],
                      "tools": list(dict.fromkeys(raw.get("tools", [])))[:30],
                      "completion_check": str(raw.get("completion_check") or "").strip()[:2000],
                      "verification": verification,
                      "source_quote": str(raw.get("source_quote") or "")[:2000]}
        for field in ("official_step_id", "component_state_index", "review_status"):
            if field in raw:
                clean_step[field] = deepcopy(raw[field])
        if "visual_plan" in raw:
            # Import lazily so ordinary deterministic SOPs stay lightweight.
            from .sop_vision import visual_plan_from_step
            clean_step["visual_plan"] = visual_plan_from_step(raw)
        references = raw.get("visual_references", [])
        if not isinstance(references, list):
            raise ValueError(f"{step_id}: visual_references must be an array.")
        clean_references = []
        for reference in references:
            if not isinstance(reference, dict):
                raise ValueError(f"{step_id}: each visual reference must be an object.")
            required = ("entity", "role", "path", "sha256", "source_document_id",
                        "source_type", "factory_pic_approved")
            if any(field not in reference for field in required):
                raise ValueError(f"{step_id}: visual reference metadata is incomplete.")
            if reference["role"] not in ("moving", "target", "completed_state"):
                raise ValueError(f"{step_id}: invalid visual reference role.")
            digest = str(reference["sha256"]).lower()
            if not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError(f"{step_id}: visual reference needs a SHA-256 digest.")
            if not isinstance(reference["factory_pic_approved"], bool):
                raise ValueError(f"{step_id}: factory_pic_approved must be boolean.")
            clean_references.append({
                "entity": str(reference["entity"])[:160],
                "role": reference["role"],
                "path": str(reference["path"])[:500],
                "sha256": digest,
                "source_document_id": str(reference["source_document_id"])[:160],
                "source_type": str(reference["source_type"])[:160],
                "factory_pic_approved": reference["factory_pic_approved"],
            })
        if clean_references:
            clean_step["visual_references"] = clean_references
        result["steps"].append(clean_step)
    by_id = {s["id"]: s for s in result["steps"]}
    visited, visiting = set(), set()

    def visit(step_id):
        if step_id not in by_id:
            raise ValueError(f"Unknown prerequisite: {step_id}")
        if step_id in visiting:
            raise ValueError("The SOP has a circular dependency.")
        if step_id in visited:
            return
        visiting.add(step_id)
        for prerequisite in by_id[step_id]["prerequisites"]:
            visit(prerequisite)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in ids:
        visit(step_id)
    return result


def sop_from_markdown(text: str) -> dict[str, Any]:
    """Parse explicit step tables or numbered instructions without guessing recovery."""
    title = re.search(r"^#\s+(.+)", text, re.M)
    steps = []
    headers = None
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            headers = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        lower = [c.lower() for c in cells]
        if "step" in lower and "instruction" in lower:
            headers = lower
            continue
        if not headers or all(re.fullmatch(r"[:\-\s]*", c) for c in cells):
            continue
        row = dict(zip(headers, cells))
        if not row.get("step") or not row.get("instruction"):
            continue
        prereq = row.get("prerequisites", "")
        steps.append({"id": row["step"], "instruction": row["instruction"],
                      "prerequisites": [] if prereq.lower() in ("none", "-", "", "n/a")
                      else [s for s in re.split(r"[,;\s]+", prereq) if s],
                      "recovery": row.get("recovery", "")})
    if not steps:
        for match in re.finditer(r"^\s*(\d+)[.)]\s+(.+)$", text, re.M):
            step_id = f"S{int(match[1]):02d}"
            steps.append({"id": step_id, "instruction": match[2],
                          "prerequisites": [steps[-1]["id"]] if steps else []})
    if not steps:
        raise ValueError("No explicit step table or numbered steps found. Use JSON with id, instruction, prerequisites and optional recovery, or a Markdown step table.")
    return validate_sop({"title": title[1] if title else "Imported procedure", "steps": steps})


class Workflow:
    def __init__(self, sop: dict[str, Any]):
        self.sop = validate_sop(sop)
        self.by_id = {s["id"]: s for s in self.sop["steps"]}
        self.completed: set[str] = set()
        self.events: list[dict[str, Any]] = []
        self.issues: dict[str | None, dict[str, Any]] = {}
        self.last_time = -1.0
        self.revision = 0

    def reset(self):
        self.revision += 1
        self.completed.clear()
        self.events.clear()
        self.issues.clear()
        self.last_time = -1.0

    def _invalidate(self, step_id: str):
        invalid = {step_id}
        while True:
            new = {s["id"] for s in self.sop["steps"]
                   if any(p in invalid for p in s["prerequisites"])} - invalid
            if not new:
                break
            invalid.update(new)
        self.completed.difference_update(invalid)
        return sorted(invalid)

    def observe(self, step_id: str, *, timestamp: float, confidence: float = 1.0,
                kind: str = "complete", source: str = "model") -> dict[str, Any]:
        if step_id not in self.by_id:
            raise ValueError(f"Unmapped model step: {step_id}. Check the class-to-SOP mapping.")
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp < self.last_time:
            raise ValueError("Observations must be in chronological order. Reset after seeking backwards.")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between zero and one.")
        if kind not in ("complete", "incorrect", "remove"):
            raise ValueError("Observation kind must be complete, incorrect or remove.")
        self.last_time = timestamp
        step = self.by_id[step_id]
        event = {"step_id": step_id, "timestamp": round(timestamp, 2),
                 "confidence": round(confidence, 3), "kind": kind, "source": source}
        missing = [p for p in step["prerequisites"] if p not in self.completed]
        if kind in ("incorrect", "remove"):
            event["invalidated"] = self._invalidate(step_id)
            event["status"] = "needs_attention" if kind == "incorrect" else "rework"
            self.issues[step_id] = {"step_id": step_id, "reason": "Incorrect completion detected." if kind == "incorrect"
                          else "A completed part was removed. Dependent steps need to be verified again.",
                          "missing": missing, "recovery": step["recovery"]}
        elif missing:
            event["status"] = "out_of_order"
            self.issues[step_id] = {"step_id": step_id, "reason": "Prerequisite steps have not been confirmed.",
                          "missing": missing, "recovery": step["recovery"]}
        elif step_id in self.completed:
            return self.snapshot()  # Held visual states must not flood the history.
        else:
            self.completed.add(step_id)
            event["status"] = "completed"
            # Completing a prerequisite alone does not resolve an observed wrong step.
            self.issues.pop(step_id, None)
        # Suppress identical repeated error notifications while preserving state.
        self.revision += 1
        if not self.events or any(self.events[-1].get(k) != event.get(k)
                                  for k in ("step_id", "status", "kind")):
            self.events.append(event)
            self.events = self.events[-500:]
        return self.snapshot()

    def snapshot(self):
        ready = [s["id"] for s in self.sop["steps"] if s["id"] not in self.completed
                 and all(p in self.completed for p in s["prerequisites"])]
        issues = deepcopy(list(self.issues.values()))
        for issue in issues:
            if issue["step_id"] is not None:
                issue["missing"] = [p for p in self.by_id[issue["step_id"]]["prerequisites"] if p not in self.completed]
        return {"sop": deepcopy(self.sop), "completed": sorted(self.completed), "ready": ready,
                "current": ready[:1], "workflow_revision": self.revision,
                "issue": issues[0] if issues else None, "issues": issues, "events": deepcopy(self.events),
                "progress": len(self.completed) / len(self.sop["steps"]),
                "status": "attention" if issues else "complete" if not ready else "tracking"}
