"""SOP-derived entity localization and conservative completion evidence.

The active SOP defines semantic roles and observable completion conditions. A
local VLM proposes boxes without station training, SAM2 tracks those boxes, and
this module refuses to mark completion until temporal geometry and a separate
terminal visual check agree. Dataset labels are not accepted by any API here.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from PIL import Image
import cv2
import numpy as np

from .sam2_focus import validate_prompt_box


NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
EVIDENCE_SOURCES = {"sam2_temporal_relation", "vlm_visual_check", "external_sensor"}


def _text(value: Any, label: str, maximum: int = 500) -> str:
    value = " ".join(str(value).strip().split())
    if not value or len(value) > maximum:
        raise ValueError(f"{label} must contain 1-{maximum} characters.")
    return value


def visual_plan_from_step(step: dict) -> dict:
    """Validate one PIC-reviewed visual plan embedded in an SOP step."""
    if not isinstance(step, dict) or not isinstance(step.get("visual_plan"), dict):
        raise ValueError("The active SOP step needs a reviewed visual_plan.")
    plan = deepcopy(step["visual_plan"])
    action = plan.get("action")
    if not isinstance(action, dict) or set(action) != {"verb", "subject", "object"}:
        raise ValueError("visual_plan.action needs verb, subject, and object.")
    action = {key: _text(value, f"action.{key}", 80) for key, value in action.items()}

    rows = plan.get("entities")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 8:
        raise ValueError("visual_plan.entities must contain 2-8 entities.")
    corpus = " ".join([
        str(step.get("instruction", "")), str(step.get("completion_check", "")),
        *[str(value) for value in step.get("parts", [])],
        *[str(value) for value in step.get("tools", [])],
    ]).lower()
    entities, names, roles = [], set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every visual entity must be an object.")
        name = str(row.get("name", "")).strip()
        role = row.get("role")
        description = _text(row.get("description", ""), f"entity {name} description")
        evidence = row.get("sop_evidence")
        if not NAME.fullmatch(name) or name in names:
            raise ValueError("Visual entity names must be unique snake_case identifiers.")
        if role not in {"moving", "target", "tool", "context"}:
            raise ValueError("Visual entity roles must be moving, target, tool, or context.")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"Visual entity {name} needs SOP traceability evidence.")
        evidence = [_text(value, f"entity {name} SOP evidence", 120) for value in evidence]
        if any(value.lower() not in corpus for value in evidence):
            raise ValueError(f"Visual entity {name} contains evidence absent from the SOP step.")
        names.add(name); roles.add(role)
        entities.append({"name": name, "role": role, "description": description,
                         "sop_evidence": evidence})
    if not {"moving", "target"} <= roles:
        raise ValueError("The visual plan needs both moving and target entities.")
    if action["subject"] not in names or action["object"] not in names:
        raise ValueError("The action subject/object must name visual entities.")

    tools = plan.get("tools", [])
    if not isinstance(tools, list) or any(name not in names for name in tools):
        raise ValueError("visual_plan.tools must name tool entities.")
    declared_tools = {row["name"] for row in entities if row["role"] == "tool"}
    if set(tools) != declared_tools:
        raise ValueError("Every tool entity must be declared in visual_plan.tools.")

    checks = plan.get("completion_evidence")
    if not isinstance(checks, list) or not checks:
        raise ValueError("The visual plan needs completion evidence.")
    clean_checks, check_ids = [], set()
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("Every completion evidence row must be an object.")
        check_id = str(check.get("id", "")).strip()
        source = check.get("source")
        if not NAME.fullmatch(check_id) or check_id in check_ids or source not in EVIDENCE_SOURCES:
            raise ValueError("Completion evidence IDs/sources are invalid.")
        check_ids.add(check_id)
        clean_checks.append({"id": check_id, "source": source,
                             "description": _text(check.get("description", ""),
                                                  f"completion evidence {check_id}")})
    required = plan.get("required_evidence")
    if not isinstance(required, list) or not required or any(value not in check_ids for value in required):
        raise ValueError("required_evidence must name defined completion evidence.")
    return {"action": action, "entities": entities, "tools": list(tools),
            "completion_evidence": clean_checks, "required_evidence": list(required)}


def _box_schema() -> dict:
    return {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            "minItems": 4, "maxItems": 4}


def entity_localization_schema(plan: dict) -> dict:
    required = [entity["name"] for entity in plan["entities"]]
    properties = {name: _box_schema() for name in required}
    return {"type": "object", "additionalProperties": False,
            "required": required, "properties": properties}


def _iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = (first[2] - first[0]) * (first[3] - first[1])
    b = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (a + b - intersection) if a + b > intersection else 0.0


def _intersection_over_smaller(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    areas = [(row[2] - row[0]) * (row[3] - row[1]) for row in (first, second)]
    return intersection / min(areas) if min(areas) else 0.0


def crop_appearance(image: Image.Image, box: list[float]) -> dict:
    """Independent, label-free evidence that a box contains a visual object."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = (round(box[0] * width), round(box[1] * height),
                      round(box[2] * width), round(box[3] * height))
    crop = rgb[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if not crop.size:
        return {"valid": False, "foreground_fraction": 0.0, "edge_fraction": 0.0}
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    foreground = np.logical_or(saturation > 45, value < 170)
    light_object = (saturation < 55) & (value > 175)
    edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), 60, 150) > 0
    foreground_fraction, light_fraction = float(foreground.mean()), float(light_object.mean())
    edge_fraction = float(edges.mean())
    return {"valid": (foreground_fraction >= .12 or light_fraction >= .2)
                     and edge_fraction >= .008,
            "foreground_fraction": foreground_fraction, "light_fraction": light_fraction,
            "edge_fraction": edge_fraction}


def salient_object_proposals(image: Image.Image, maximum: int = 6) -> list[dict]:
    """Create class-agnostic proposals from dark/saturated rigid-object pixels.

    This assumes the light, uncluttered work surface visible in the active
    station recording. It is a deployment adapter, not a universal detector.
    The VLM assigns SOP identities; these components never assign labels.
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    magenta = (hue >= 135) & (hue <= 179) & (saturation > 70)
    seed = ((value < 105) | magenta).astype(np.uint8)
    scale = max(9, round(min(width, height) * .029))
    if scale % 2 == 0:
        scale += 1
    kernel = np.ones((scale, scale), np.uint8)
    expanded = cv2.dilate(cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel), kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(expanded)
    boxes = []
    for x, y, box_width, box_height, area in stats[1:]:
        fraction = area / (width * height)
        if not .004 <= fraction <= .5:
            continue
        boxes.append([x / width, y / height, (x + box_width) / width, (y + box_height) / height])
    # Keep touching components separate. During assembly, a hand or the moving
    # part may already touch the receiver; merging them would destroy identity.
    proposals = []
    for box in boxes:
        padding = .012
        box = validate_prompt_box([max(0.0, box[0] - padding), max(0.0, box[1] - padding),
                                   min(1.0, box[2] + padding), min(1.0, box[3] + padding)])
        appearance = crop_appearance(image, box)
        if appearance["valid"]:
            proposals.append({"box": box, "appearance": appearance,
                              "area": (box[2] - box[0]) * (box[3] - box[1])})
    proposals.sort(key=lambda row: row["area"], reverse=True)
    output = []
    for index, row in enumerate(proposals[:maximum], 1):
        output.append({"id": f"pixel_{index}", "source": "pixel_saliency", **row})
    return output


def light_object_proposals(image: Image.Image, maximum: int = 6) -> list[dict]:
    """Propose light rigid parts on a neutral work surface.

    This is a class-agnostic complement to ``salient_object_proposals``.  It
    fixes the otherwise systematic omission of white perforated pieces in the
    IndustReal station.  Identity is still assigned by DINO/VLM and audited;
    this thresholding stage never names a part.
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    threshold = max(170.0, float(np.median(value)) + 3.0)
    seed = ((saturation < 55) & (value > threshold)).astype(np.uint8)
    close_scale = max(7, round(min(width, height) * .015))
    if close_scale % 2 == 0:
        close_scale += 1
    closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE,
                              np.ones((close_scale, close_scale), np.uint8))
    dilate_scale = max(3, close_scale // 2)
    expanded = cv2.dilate(closed, np.ones((dilate_scale, dilate_scale), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(expanded)
    proposals = []
    for x, y, box_width, box_height, area in stats[1:]:
        fraction = area / (width * height)
        aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
        if not .0005 <= fraction <= .12 or aspect > 5:
            continue
        padding = .008
        box = validate_prompt_box([
            max(0.0, x / width - padding), max(0.0, y / height - padding),
            min(1.0, (x + box_width) / width + padding),
            min(1.0, (y + box_height) / height + padding),
        ])
        appearance = crop_appearance(image, box)
        if appearance["valid"]:
            proposals.append({"box": box, "appearance": appearance,
                              "area": (box[2] - box[0]) * (box[3] - box[1])})
    proposals.sort(key=lambda row: row["area"], reverse=True)
    return [{"id": f"light_{index}", "source": "light_saliency", **row}
            for index, row in enumerate(proposals[:maximum], 1)]


def motion_guided_light_proposals(previous: Image.Image | None, current: Image.Image,
                                  maximum: int = 4) -> list[dict]:
    """Find light rigid regions that changed between two camera samples.

    Motion is used only to propose pixels, never to name the part or claim an
    action.  This is especially useful when a white component has little
    contrast against the IndustReal table but moves with the worker.
    """
    if previous is None:
        return []
    before = np.asarray(previous.convert("RGB").resize(current.size))
    after = np.asarray(current.convert("RGB"))
    height, width = after.shape[:2]
    difference = cv2.cvtColor(cv2.absdiff(before, after), cv2.COLOR_RGB2GRAY)
    moving = difference > 14
    motion_scale = max(9, round(min(width, height) * .021))
    if motion_scale % 2 == 0:
        motion_scale += 1
    moving = cv2.dilate(moving.astype(np.uint8),
                        np.ones((motion_scale, motion_scale), np.uint8)) > 0
    hsv = cv2.cvtColor(after, cv2.COLOR_RGB2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    threshold = max(168.0, float(np.median(value)))
    light = (saturation < 70) & (value > threshold) & moving
    close_scale = max(7, motion_scale // 2)
    joined = cv2.morphologyEx(light.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((close_scale, close_scale), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    rows = []
    for x, y, box_width, box_height, area in stats[1:]:
        fraction = area / (width * height)
        aspect = max(box_width / max(1, box_height), box_height / max(1, box_width))
        if not .00025 <= fraction <= .08 or aspect > 6:
            continue
        padding = .018
        box = validate_prompt_box([
            max(0.0, x / width - padding), max(0.0, y / height - padding),
            min(1.0, (x + box_width) / width + padding),
            min(1.0, (y + box_height) / height + padding),
        ])
        appearance = crop_appearance(current, box)
        if appearance["valid"]:
            rows.append({"box": box, "appearance": appearance,
                         "area": (box[2] - box[0]) * (box[3] - box[1])})
    rows.sort(key=lambda row: row["area"], reverse=True)
    return [{"id": f"motion_{index}", "source": "motion_guided_light", **row}
            for index, row in enumerate(rows[:maximum], 1)]


def yolo_prompts_from_plan(plan: dict) -> list[dict]:
    """Create station-training-free detector prompts from the reviewed plan."""
    rows = []
    for entity in plan["entities"]:
        aliases = []
        for value in [*entity.get("sop_evidence", []), entity["description"]]:
            value = " ".join(str(value).strip().split())
            if value and value.lower() != entity["name"].replace("_", " ").lower() and value not in aliases:
                aliases.append(value[:100])
        rows.append({"name": entity["name"], "prompt": entity["name"].replace("_", " "),
                     "aliases": aliases[:3], "role": entity["role"], "count": 1,
                     "sop_evidence": entity["sop_evidence"][0]})
    return rows


def merge_candidate_proposals(pixel_rows: list[dict], yolo_result: dict | None,
                              maximum: int = 12) -> list[dict]:
    """Merge class-agnostic and YOLO-World boxes for one VLM ranking pass."""
    output = [dict(row) for row in pixel_rows]
    if isinstance(yolo_result, dict):
        for index, row in enumerate(yolo_result.get("objects", []), 1):
            try:
                box = validate_prompt_box(row.get("box"))
            except (TypeError, ValueError):
                continue
            if any(_iou(box, old["box"]) >= .85 for old in output):
                continue
            output.append({"id": f"yolo_{index}", "source": "yolo_world", "box": box,
                           "semantic_hint": row.get("semantic_name") or row.get("name"),
                           "matched_prompt": row.get("matched_prompt"),
                           "confidence": float(row.get("confidence", 0)),
                           "area": (box[2] - box[0]) * (box[3] - box[1])})
    return output[:maximum]


def proposals_in_workspace(proposals: list[dict], workspace_roi: list[float] | None) -> list[dict]:
    """Keep candidates whose centre lies in the PIC-calibrated work area."""
    if workspace_roi is None:
        return [dict(row) for row in proposals]
    roi = validate_prompt_box(workspace_roi)
    output = []
    for row in proposals:
        box = row["box"]
        centre_x, centre_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if roi[0] <= centre_x <= roi[2] and roi[1] <= centre_y <= roi[3]:
            output.append(dict(row))
    return output


def prompt_points_for_entity(image: Image.Image, box: list[float], description: str,
                             maximum_points: int = 4) -> list[dict]:
    """Create positive expected-colour points and negative skin points for SAM2."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    expected = np.zeros((height, width), dtype=bool)
    words = set(re.findall(r"\b(?:black|white|grey|gray|magenta|pink|red|blue|green|yellow|orange)\b",
                           description.lower()))
    if "black" in words:
        expected |= value < 90
    if words & {"magenta", "pink"}:
        expected |= (hue >= 135) & (hue <= 179) & (saturation > 70)
    if words & {"grey", "gray"}:
        expected |= (saturation < 55) & (value >= 55) & (value <= 180)
    if "white" in words:
        expected |= (saturation < 55) & (value >= 160)
    if not expected.any():
        expected |= (value < 120) | (saturation > 70)
    skin = (hue <= 24) & (saturation >= 35) & (saturation <= 190) & (value >= 75)
    x1, y1, x2, y2 = (max(0, round(box[0] * width)), max(0, round(box[1] * height)),
                      min(width, round(box[2] * width)), min(height, round(box[3] * height)))
    region = np.zeros((height, width), dtype=bool); region[y1:y2, x1:x2] = True

    def component_points(mask: np.ndarray, count: int) -> list[tuple[int, int]]:
        total, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        rows = sorted(((int(stats[index, cv2.CC_STAT_AREA]), centroids[index])
                       for index in range(1, total) if stats[index, cv2.CC_STAT_AREA] >= 12),
                      key=lambda row: row[0], reverse=True)
        return [(round(point[0]), round(point[1])) for _, point in rows[:count]]

    positives = component_points(expected & region & ~skin, max(1, maximum_points - 2))
    negatives = component_points(skin & region, maximum_points - len(positives))
    points = [{"point": [x / width, y / height], "label": 1} for x, y in positives]
    points.extend({"point": [x / width, y / height], "label": 0} for x, y in negatives)
    return points[:maximum_points]


def semantic_mask_quality(image: Image.Image, mask: np.ndarray, description: str) -> dict:
    """Reject a SAM2 initialization dominated by skin or uninformative background."""
    mask = np.asarray(mask, dtype=bool)
    rgb = np.asarray(image.convert("RGB"))
    if mask.shape != rgb.shape[:2] or not mask.any():
        return {"valid": False, "distinctive_fraction": 0.0, "skin_fraction": 0.0,
                "reason": "The initial mask is empty or has the wrong size."}
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    words = set(re.findall(r"\b(?:black|grey|gray|magenta|pink|red|blue|green|yellow|orange)\b",
                           description.lower()))
    distinctive = np.zeros(mask.shape, dtype=bool)
    if "black" in words:
        distinctive |= value < 90
    if words & {"magenta", "pink"}:
        distinctive |= (hue >= 135) & (hue <= 179) & (saturation > 70)
    if words & {"grey", "gray"}:
        distinctive |= (saturation < 55) & (value >= 55) & (value <= 180)
    if "white" in words:
        distinctive |= (saturation < 55) & (value >= 160)
    if "red" in words:
        distinctive |= ((hue <= 8) | (hue >= 172)) & (saturation > 90)
    skin = (hue <= 24) & (saturation >= 35) & (saturation <= 190) & (value >= 75)
    area = int(mask.sum())
    distinctive_fraction = float((distinctive & mask).sum() / area)
    skin_fraction = float((skin & mask).sum() / area)
    valid = (not words or distinctive_fraction >= .08) and skin_fraction <= .5
    reason = ("Expected appearance is present and skin does not dominate."
              if valid else "The mask is dominated by a hand/background or lacks expected part colours.")
    return {"valid": valid, "distinctive_fraction": distinctive_fraction,
            "skin_fraction": skin_fraction, "reason": reason}


def candidate_routing_schema(plan: dict, proposals: list[dict],
                             allowed_ids: dict[str, list[str]] | None = None) -> dict:
    all_choices = [row["id"] for row in proposals]
    names = [row["name"] for row in plan["entities"]]
    properties = {}
    for name in names:
        choices = (allowed_ids or {}).get(name, all_choices)
        choices = [value for value in choices if value in all_choices]
        properties[name] = {"type": "string", "enum": [*choices, "unknown"]}
    return {"type": "object", "additionalProperties": False, "required": names,
            "properties": properties}


def localization_from_routing(value: Any, plan: dict, proposals: list[dict],
                              image: Image.Image) -> dict:
    names = [row["name"] for row in plan["entities"]]
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("The VLM must route every entity exactly once.")
    by_id = {row["id"]: row for row in proposals}
    selected = [value[name] for name in names]
    missing = [{"name": name, "reason": "No unambiguous visual proposal."}
               for name in names if value[name] not in by_id]
    if len([choice for choice in selected if choice in by_id]) != len(set(
            choice for choice in selected if choice in by_id)):
        missing.append({"name": "identity_routing", "reason": "Two identities selected the same pixels."})
    selected_boxes = [by_id[choice]["box"] for choice in selected if choice in by_id]
    for index, first in enumerate(selected_boxes):
        for second in selected_boxes[index + 1:]:
            if _intersection_over_smaller(first, second) > .2:
                missing.append({"name": "identity_routing",
                                "reason": "Selected proposals belong to the same overlapping physical region."})
    objects = []
    if not missing:
        for object_id, entity in enumerate(plan["entities"], 1):
            proposal = by_id[value[entity["name"]]]
            appearance = crop_appearance(image, proposal["box"])
            if not appearance["valid"]:
                missing.append({"name": entity["name"], "reason": "Candidate failed pixel evidence."})
                continue
            objects.append({"object_id": object_id, "name": entity["name"], "role": entity["role"],
                            "box": proposal["box"], "visibility": "model_routed_proposal",
                            "evidence": f"VLM selected {proposal['id']}; independent pixel gate passed.",
                            "appearance": appearance,
                            "prompt_points": prompt_points_for_entity(
                                image, proposal["box"], entity["description"])})
    complete = len(objects) == len(names) and not missing
    return {"objects": objects if complete else [], "candidates": proposals,
            "missing": missing, "complete": complete,
            "decision": "ACCEPT" if complete else "ABSTAIN", "raw": value}


def validate_entity_localization(value: Any, plan: dict) -> dict:
    names = [row["name"] for row in plan["entities"]]
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("The localizer must return every SOP visual entity exactly once.")
    objects, missing = [], []
    for object_id, name in enumerate(names, 1):
        row = value.get(name)
        try:
            raw = row
            if not isinstance(raw, list) or len(raw) != 4:
                raise ValueError
            box = validate_prompt_box([float(number) / 1000 for number in raw])
        except (TypeError, ValueError):
            box = None
        if box is None:
            missing.append({"name": name, "reason": "The VLM did not return a usable box."})
            continue
        role = next(item["role"] for item in plan["entities"] if item["name"] == name)
        objects.append({"object_id": object_id, "name": name, "role": role,
                        "box": box, "visibility": "model_proposed",
                        "evidence": "Box proposed from the active SOP description."})
    for index, first in enumerate(objects):
        for second in objects[index + 1:]:
            if _iou(first["box"], second["box"]) > .55:
                missing.extend([{"name": first["name"], "reason": "Localization boxes overlap ambiguously."},
                                {"name": second["name"], "reason": "Localization boxes overlap ambiguously."}])
    complete = len(objects) == len(names) and not missing
    return {"objects": objects if complete else [], "candidates": objects,
            "missing": missing, "complete": complete,
            "decision": "ACCEPT" if complete else "ABSTAIN"}


class SopEntityLocalizer:
    """Use a local structured-output VLM to ground reviewed SOP entities."""
    def __init__(self, generator, detector=None, reference_matcher=None,
                 require_references: bool = False, workspace_roi: list[float] | None = None):
        self.generator = generator
        self.detector = detector
        self.reference_matcher = reference_matcher
        self.require_references = require_references
        self.workspace_roi = (validate_prompt_box(workspace_roi)
                              if workspace_roi is not None else None)

    def _refine_moving_entities(self, image: Image.Image, plan: dict, result: dict) -> dict:
        if not result["complete"]:
            return result
        refined = []
        raw_refinements = {}
        entities = {row["name"]: row for row in plan["entities"]}
        for obj in result["objects"]:
            entity = entities[obj["name"]]
            if entity["role"] not in {"moving", "tool"}:
                refined.append(obj)
                continue
            box = obj["box"]
            width, height = box[2] - box[0], box[3] - box[1]
            parent = [max(0.0, box[0] - width * .35), max(0.0, box[1] - height * .35),
                      min(1.0, box[2] + width * .08), min(1.0, box[3] + height * .08)]
            crop = image.crop((round(parent[0] * image.width), round(parent[1] * image.height),
                               round(parent[2] * image.width), round(parent[3] * image.height)))
            schema = {"type": "object", "additionalProperties": False, "required": ["found", "box"],
                      "properties": {"found": {"type": "boolean"}, "box": _box_schema()}}
            prompt = (
                "Draw one tight box around only the named rigid physical object in this CURRENT CAMERA CROP. "
                "Exclude hands, fingers, skin, table, shadows and printed images. Include all visible connected "
                "pieces belonging to the object. Coordinates are 0 to 1000 relative to this crop. Use [0,0,1,1] "
                "if the object cannot be separated from the hand and set found=false; otherwise set found=true.\n"
                "ENTITY: " + str(entity))
            raw = self.generator.generate(
                prompt, [("CURRENT CAMERA CROP; localize only the named rigid object.", crop)],
                max_tokens=60, schema=schema)
            raw_refinements[obj["name"]] = raw
            try:
                if raw.get("found") is not True:
                    raise ValueError
                relative = validate_prompt_box([float(number) / 1000 for number in raw["box"]])
                parent_width, parent_height = parent[2] - parent[0], parent[3] - parent[1]
                refined_box = validate_prompt_box([
                    parent[0] + relative[0] * parent_width,
                    parent[1] + relative[1] * parent_height,
                    parent[0] + relative[2] * parent_width,
                    parent[1] + relative[3] * parent_height,
                ])
            except (KeyError, TypeError, ValueError):
                return {**result, "objects": [], "complete": False, "decision": "ABSTAIN",
                        "missing": [{"name": obj["name"],
                                     "reason": "Moving-object refinement was uncertain."}],
                        "refinements": raw_refinements}
            appearance = crop_appearance(image, refined_box)
            original_area = (box[2] - box[0]) * (box[3] - box[1])
            refined_area = (refined_box[2] - refined_box[0]) * (refined_box[3] - refined_box[1])
            if (not appearance["valid"] or _intersection_over_smaller(refined_box, box) < .1
                    or not .2 <= refined_area / original_area <= 1.4):
                return {**result, "objects": [], "complete": False, "decision": "ABSTAIN",
                        "missing": [{"name": obj["name"],
                                     "reason": "Refined box failed the independent pixel/consistency gate."}],
                        "refinements": raw_refinements}
            refined.append({**obj, "box": refined_box, "appearance": appearance,
                            "evidence": obj["evidence"] + " Moving/tool refinement passed.",
                            "prompt_points": prompt_points_for_entity(
                                image, refined_box, entity["description"])})
        return {**result, "objects": refined, "refinements": raw_refinements}

    def _audit_selected_entities(self, image: Image.Image, plan: dict, result: dict) -> dict:
        """Reject printed instructions, hands, and semantically wrong crops."""
        if not result["complete"]:
            return result
        schema = {"type": "object", "additionalProperties": False,
                  "required": ["named_entity_visible", "printed_media_dominant",
                               "hand_dominant", "certainty", "evidence"],
                  "properties": {
                      "named_entity_visible": {"type": "string", "enum": ["yes", "no", "unknown"]},
                      "printed_media_dominant": {"type": "string", "enum": ["yes", "no", "unknown"]},
                      "hand_dominant": {"type": "string", "enum": ["yes", "no", "unknown"]},
                      "certainty": {"type": "string", "enum": ["clear", "uncertain"]},
                      "evidence": {"type": "string", "minLength": 1, "maxLength": 160}}}
        entities = {row["name"]: row for row in plan["entities"]}
        audits, failed = {}, []
        for obj in result["objects"]:
            box = obj["box"]
            crop = image.crop((round(box[0] * image.width), round(box[1] * image.height),
                               round(box[2] * image.width), round(box[3] * image.height)))
            entity = entities[obj["name"]]
            prompt = (
                "Audit whether this candidate crop really contains the named 3D physical SOP entity. "
                "Printed instruction sheets, photographs, diagrams and screens are not physical parts. "
                "A hand touching a part is allowed only when the rigid part—not skin—dominates. Return unknown "
                "when identity is ambiguous.\nENTITY: " + str(entity))
            try:
                raw = self.generator.generate(prompt, [("SELECTED CANDIDATE CROP", crop)],
                                              max_tokens=120, schema=schema)
            except Exception as error:
                raw = {"named_entity_visible": "unknown", "printed_media_dominant": "unknown",
                       "hand_dominant": "unknown", "certainty": "uncertain",
                       "evidence": str(error)[:160] or type(error).__name__}
            audits[obj["name"]] = raw
            valid = (raw.get("named_entity_visible") == "yes"
                     and raw.get("printed_media_dominant") == "no"
                     and raw.get("hand_dominant") == "no"
                     and raw.get("certainty") == "clear")
            if not valid:
                failed.append({"name": obj["name"],
                               "reason": "Selected crop failed the physical-entity audit: "
                                         + str(raw.get("evidence", "uncertain"))})
        if failed:
            return {**result, "objects": [], "proposed_objects": result["objects"],
                    "complete": False, "decision": "ABSTAIN", "missing": failed,
                    "entity_audits": audits}
        return {**result, "entity_audits": audits}

    def locate(self, image: Image.Image, plan: dict,
               extra_proposals: list[dict] | None = None) -> dict:
        requirements = [{key: entity[key] for key in ("name", "role", "description")}
                        for entity in plan["entities"]]
        pixel_proposals = salient_object_proposals(image)
        light_proposals = light_object_proposals(image)
        motion_proposals = [dict(row) for row in (extra_proposals or [])]
        yolo_result = None
        if self.detector is not None:
            try:
                yolo_result = self.detector.locate(
                    image, yolo_prompts_from_plan(plan), confidence=.025, image_size=960)
            except Exception as error:
                yolo_result = {"objects": [], "missing": [], "complete": False,
                               "error": str(error)[:240] or type(error).__name__}
        proposals = proposals_in_workspace(
            merge_candidate_proposals([*motion_proposals, *pixel_proposals, *light_proposals],
                                      yolo_result, maximum=16),
            self.workspace_roi)
        reference_matching = None
        reference_images = []
        allowed_ids = None
        if self.reference_matcher is not None:
            reference_matching = self.reference_matcher.rank(image, proposals, plan)
            if reference_matching["decision"] != "GUIDED":
                missing = reference_matching.get("missing", [])
                if missing and isinstance(missing[0], str):
                    missing = [{"name": name, "reason": reference_matching["reason"]}
                               for name in missing]
                return {"objects": [], "candidates": proposals, "missing": missing,
                        "complete": False, "decision": "ABSTAIN", "raw": {},
                        "reference_matching": reference_matching,
                        "proposal_sources": {
                            "pixel_saliency": len(pixel_proposals),
                            "light_saliency": len(light_proposals),
                            "motion_guided_light": len(motion_proposals),
                            "yolo_world": len(yolo_result.get("objects", [])) if yolo_result else 0,
                            "yolo_world_error": yolo_result.get("error") if yolo_result else None,
                        },
                        "model": self.generator.status().get("model_id"),
                        "station_training": False, "source": "active_sop_visual_plan"}
            score_maps = {
                name: {row["candidate_id"]: row["score"] for row in rows}
                for name, rows in reference_matching["rankings"].items()
            }
            selected = set(reference_matching["shortlisted_ids"])
            proposals = [{**row, "reference_scores": {
                            name: scores.get(row["id"]) for name, scores in score_maps.items()
                            if row["id"] in scores}}
                         for row in proposals if row["id"] in selected]
            reference_images = self.reference_matcher.prompt_images(plan)
            allowed_ids = reference_matching.get("allowed_ids")
        elif self.require_references:
            return {"objects": [], "candidates": proposals,
                    "missing": [{"name": row["name"],
                                 "reason": "PIC-approved entity references are required."}
                                for row in requirements],
                    "complete": False, "decision": "ABSTAIN", "raw": {},
                    "reference_matching": {"decision": "ABSTAIN", "complete": False,
                                           "reason": "No entity reference matcher was configured."},
                    "model": self.generator.status().get("model_id"),
                    "station_training": False, "source": "active_sop_visual_plan"}
        if len(proposals) < len(requirements):
            return {"objects": [], "candidates": proposals,
                    "missing": [{"name": row["name"], "reason": "Not enough visual proposals."}
                                for row in requirements],
                    "complete": False, "decision": "ABSTAIN", "raw": {},
                    "model": self.generator.status().get("model_id"),
                    "station_training": False, "source": "active_sop_visual_plan"}
        images = list(reference_images)
        for proposal in proposals:
            box = proposal["box"]
            crop = image.crop((round(box[0] * image.width), round(box[1] * image.height),
                               round(box[2] * image.width), round(box[3] * image.height)))
            images.append((f"{proposal['id']}: crop from the same CURRENT CAMERA frame.", crop))
        candidate_metadata = [{key: row.get(key) for key in
                               ("id", "source", "semantic_hint", "matched_prompt", "confidence",
                                "reference_scores")
                               if row.get(key) is not None} for row in proposals]
        prompt = (
            "Match each named PHYSICAL SOP entity to one candidate crop from the CURRENT CAMERA frame. Ignore "
            "printed photographs, instruction sheets, diagrams, hands, skin, and unrelated loose pieces. Use each "
            "candidate at most once. A moving entity is the separate rigid item being handled or approaching; the "
            "target is the already-assembled receiving structure. Never label two crops from the same physical "
            "assembly as moving and target. REFERENCE images show appearance only and are never live "
            "completion evidence. DINOv2 reference scores are similarity hints, not proof. Return unknown rather "
            "than guessing. Return only one candidate ID under each "
            "supplied entity key.\nVISUAL ENTITIES:\n"
            + str(requirements) + "\nCANDIDATE METADATA (hints are proposals, not truth):\n"
            + str(candidate_metadata))
        raw = self.generator.generate(
            prompt, images, max_tokens=120,
            schema=candidate_routing_schema(plan, proposals, allowed_ids))
        result = localization_from_routing(raw, plan, proposals, image)
        attempts = [raw]
        if not result["complete"]:
            subject, target = plan["action"]["subject"], plan["action"]["object"]
            target_proposal = next((row for row in proposals if row["id"] == raw.get(target)), None)
            subject_allowed = (allowed_ids or {}).get(subject)
            choices = ([row["id"] for row in proposals
                        if target_proposal is None
                        or (row["id"] != target_proposal["id"]
                            and _intersection_over_smaller(row["box"], target_proposal["box"]) <= .2)]
                       + ["unknown"])
            if subject_allowed is not None:
                choices = [value for value in choices
                           if value == "unknown" or value in subject_allowed]
            retry_schema = {"type": "object", "additionalProperties": False,
                            "required": [subject],
                            "properties": {subject: {"type": "string", "enum": choices}}}
            retry_prompt = (
                "The initial moving selection overlapped the receiving target. Keep the target fixed and choose only "
                "the separate rigid moving item being handled or approaching it. Do not choose a hand, printed image, "
                "part of the connected receiver, or an overlapping crop. Return unknown if no candidate is safe.\n"
                f"MOVING ENTITY: {next(row for row in requirements if row['name'] == subject)}\n"
                f"FIXED TARGET: {raw.get(target)}")
            fixed = self.generator.generate(retry_prompt, images, max_tokens=60, schema=retry_schema)
            repaired = {**raw, subject: fixed[subject]}
            result = localization_from_routing(repaired, plan, proposals, image)
            attempts.append(repaired)
        result = self._refine_moving_entities(image, plan, result)
        result = self._audit_selected_entities(image, plan, result)
        return {**result, "attempts": attempts, "reference_matching": reference_matching,
                "proposal_sources": {
                    "pixel_saliency": len(pixel_proposals),
                    "light_saliency": len(light_proposals),
                    "motion_guided_light": len(motion_proposals),
                    "yolo_world": len(yolo_result.get("objects", [])) if yolo_result else 0,
                    "yolo_world_error": yolo_result.get("error") if yolo_result else None,
                },
                "model": self.generator.status().get("model_id"),
                "station_training": False, "source": "active_sop_visual_plan"}


def completion_check_schema(plan: dict) -> dict:
    properties = {
        check_id: {"type": "string", "enum": ["yes", "no", "unknown"]}
        for check_id in plan["required_evidence"] if check_id != "part_at_target"
    }
    properties.update({
        "worker_holding_part": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "certainty": {"type": "string", "enum": ["clear", "uncertain"]},
        "evidence": {"type": "string", "minLength": 1, "maxLength": 160},
    })
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def validate_completion_check(value: Any, plan: dict) -> dict:
    expected = set(completion_check_schema(plan)["required"])
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Invalid terminal completion evidence.")
    evidence = _text(value.get("evidence", ""), "terminal evidence", 500)
    certainty = value.get("certainty")
    required_visual = [item for item in plan["required_evidence"] if item != "part_at_target"]
    states = {name: value.get(name) for name in required_visual}
    if certainty not in {"clear", "uncertain"} or value.get("worker_holding_part") not in {"yes", "no", "unknown"}:
        raise ValueError("Invalid terminal evidence certainty.")
    if any(state not in {"yes", "no", "unknown"} for state in states.values()):
        raise ValueError("Invalid terminal evidence state.")
    passed = certainty == "clear" and all(state == "yes" for state in states.values())
    return {**states, "worker_holding_part": value["worker_holding_part"],
            "certainty": certainty, "evidence": evidence, "passed": passed}


class SopCompletionVerifier:
    """Ask only for final visible evidence; temporal contact remains deterministic."""
    def __init__(self, generator):
        self.generator = generator

    def verify(self, image: Image.Image, plan: dict) -> dict:
        checks = [row for row in plan["completion_evidence"]
                  if row["id"] in plan["required_evidence"] and row["id"] != "part_at_target"]
        prompt = (
            "Inspect only this CURRENT CAMERA frame. This check runs after a separate tracker found stable contact, "
            "but contact alone is not completion. Judge each listed visible result as yes, no, or unknown. Do not "
            "infer hidden tightness. Printed photos and instructions are not evidence. worker_holding_part asks whether "
            "a hand is still supporting or positioning the moving subassembly; this is phase information and does not "
            "by itself decide completion. Use certainty=clear only when every required feature is visible.\n"
            f"ACTION: {plan['action']}\nCHECKS: {checks}")
        raw = self.generator.generate(
            prompt, [("CURRENT CAMERA FRAME; judge this frame only.", image)],
            max_tokens=180, schema=completion_check_schema(plan))
        return {**validate_completion_check(raw, plan), "raw": raw,
                "model": self.generator.status().get("model_id")}


def build_vision_event(plan: dict, timestamp_s: float, relation: dict,
                       terminal: dict | None = None) -> dict:
    """Produce the stable interface expected by the downstream verifier."""
    relation_state = relation.get("state", "unknown")
    part_at_target = "yes" if relation_state == "stable_contact_candidate" else (
        "no" if relation_state == "separate" else "unknown")
    terminal_passed = bool(terminal and terminal.get("passed"))
    if relation_state == "unknown":
        phase, status = "unknown", "unknown"
    elif relation_state == "separate":
        phase, status = "approaching", "started"
    elif relation_state == "contact_candidate":
        phase, status = "placing", "ongoing"
    elif terminal_passed:
        phase, status = "complete", "completed"
    else:
        phase, status = "fastening_or_verifying", "ongoing"
    visual_checks = {name: terminal.get(name, "unknown") if terminal else "unknown"
                     for name in plan["required_evidence"] if name != "part_at_target"}
    return {
        "timestamp_s": float(timestamp_s),
        "action": dict(plan["action"]),
        "action_status": status,
        "phase": phase,
        "confidence": None,
        "certainty": terminal.get("certainty", "uncertain") if terminal_passed else "uncertain",
        "relation": {
            "part_at_target": part_at_target,
            **visual_checks,
            "tool_contact": ("not_required" if not plan["tools"]
                             else relation.get("tool_contact", "unknown")),
            "stable_attachment": part_at_target,
        },
        "tools": list(plan["tools"]),
        "evidence": (terminal.get("evidence") if terminal else relation.get("reason", "No evidence.")),
    }
