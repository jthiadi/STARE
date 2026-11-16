"""Preflight checks for the next IndustReal end-to-end experiment.

The checker keeps inference inputs separate from held-out labels, validates the
dataset-grounded SOP, and reports deployment approvals as separate gates.  It
does not run a model and never exposes evaluation labels to inference code.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .settings import (
    DINO_V2_ID,
    DINO_V2_REVISION,
    ROOT,
    SAM2_ID,
    SAM2_REVISION,
    VLM_ID,
    VLM_REVISION,
    YOLO_WORLD_ID,
)
from .sop_vision import visual_plan_from_step
from .workflow import Workflow, validate_sop


OFFICIAL_STEPS = {
    "IR03": (3, 1, "Install front chassis"),
    "IR06": (6, 2, "Install front chassis pin"),
    "IR09": (9, 3, "Install rear chassis"),
    "IR15": (15, 5, "Install front rear chassis pin"),
    "IR18": (18, 6, "Install rear rear chassis pin"),
    "IR21": (21, 7, "Install front bracket"),
    "IR24": (24, 8, "Install front bracket screw"),
    "IR27": (27, 9, "Install front wheel assy"),
    "IR30": (30, 10, "Install rear wheel assy"),
}

# This is a partial order, not the chronological order of one recording.
# The two chassis branches can run independently.  Once both are ready, the
# front-wheel and rear-wheel branches can also run independently.
EXPECTED_PREREQUISITES = {
    "IR03": [],
    "IR06": ["IR03"],
    "IR09": [],
    "IR15": ["IR09"],
    "IR18": ["IR09"],
    "IR21": ["IR03", "IR06", "IR09", "IR15", "IR18"],
    "IR24": ["IR21"],
    "IR27": ["IR24"],
    "IR30": ["IR03", "IR06", "IR09", "IR15", "IR18"],
}


def _path(value: Any) -> Path:
    candidate = (ROOT / str(value)).resolve()
    root = ROOT.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path escapes the project workspace: {value}")
    return candidate


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(checks: list[dict], check_id: str, passed: bool, detail: str,
           *, gate: str = "research", warning: bool = False) -> None:
    checks.append({
        "id": check_id,
        "status": "pass" if passed else "warning" if warning else "fail",
        "gate": gate,
        "detail": detail,
    })


def _check_sop(sop: dict, checks: list[dict]) -> None:
    try:
        validate_sop(sop)
    except (TypeError, ValueError) as error:
        _check(checks, "sop_schema_valid", False, str(error))
        return
    _check(checks, "sop_schema_valid", True, "SOP IDs, prerequisites, and graph are structurally valid.")

    by_id = {row.get("id"): row for row in sop.get("steps", [])}
    mapping_ok = set(by_id) == set(OFFICIAL_STEPS)
    mapping_errors = []
    for step_id, expected in OFFICIAL_STEPS.items():
        row = by_id.get(step_id, {})
        actual = (row.get("official_step_id"), row.get("component_state_index"), row.get("source_quote"))
        if actual != expected:
            mapping_ok = False
            mapping_errors.append(f"{step_id}: expected {expected}, found {actual}")
    _check(checks, "official_step_mapping_valid", mapping_ok,
           "All nine assembly steps match the official IDs, state indexes, and names."
           if mapping_ok else "; ".join(mapping_errors))

    procedure_source = next((row for row in sop.get("source_documents", [])
                             if row.get("id") == "official_procedure_info"), {})
    source_mapping = {}
    try:
        source_path = _path(procedure_source.get("path", ""))
        source_rows = json.loads(source_path.read_text(encoding="utf-8"))
        for row in source_rows:
            if row.get("expected_in_assy") is True:
                step_id = f"IR{int(row['id']):02d}"
                source_mapping[step_id] = (int(row["id"]), int(row["state_idx"]), row["description"])
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _check(checks, "official_procedure_source_valid", False,
               f"Could not read official procedure metadata: {error}")
    else:
        source_ok = source_mapping == OFFICIAL_STEPS
        _check(checks, "official_procedure_source_valid", source_ok,
               "The cached official procedure metadata independently confirms the nine-step mapping."
               if source_ok else f"Official source mapping differs: {source_mapping}.")

    actual_prerequisites = {step_id: row.get("prerequisites", []) for step_id, row in by_id.items()}
    graph_ok = actual_prerequisites == EXPECTED_PREREQUISITES
    _check(checks, "partial_order_valid", graph_ok,
           "SOP uses two independent chassis branches and two independent final wheel branches."
           if graph_ok else f"Expected {EXPECTED_PREREQUISITES}; found {actual_prerequisites}.")

    policy = sop.get("execution_policy", {})
    display = policy.get("canonical_display_order")
    display_ok = (policy.get("type") == "partial_order_dag" and isinstance(display, list)
                  and len(display) == len(by_id) and set(display) == set(by_id))
    _check(checks, "display_order_valid", display_ok,
           "Canonical display order contains every step once; it does not override the prerequisite graph.")

    visual_errors = []
    for step_id, row in by_id.items():
        try:
            visual_plan_from_step(row)
        except (TypeError, ValueError) as error:
            visual_errors.append(f"{step_id}: {error}")
    _check(checks, "visual_plans_complete", not visual_errors,
           "Every step has traceable moving/target entities and both temporal and terminal completion evidence."
           if not visual_errors else "; ".join(visual_errors))

    policy = sop.get("visual_reference_policy", {})
    reference_errors = []
    for step_id, row in by_id.items():
        references = row.get("visual_references", [])
        if not references:
            reference_errors.append(f"{step_id}: no SOP-source visual reference")
            continue
        for reference in references:
            try:
                path = _path(reference.get("path", ""))
                valid = (path.is_file() and _sha256(path) == reference.get("sha256")
                         and reference.get("source_document_id") == "official_state_overview"
                         and reference.get("source_type") == "official_sop_illustration"
                         and reference.get("factory_pic_approved") is False)
            except (OSError, TypeError, ValueError):
                valid = False
            if not valid:
                reference_errors.append(f"{step_id}: invalid or changed SOP-source reference")
    policy_ok = (policy.get("not_ground_truth") is True
                 and policy.get("factory_pic_approved") is False
                 and policy.get("source_document_id") == "official_state_overview")
    _check(checks, "sop_source_visual_references_valid", policy_ok and not reference_errors,
           "Every step has a hashed component illustration extracted from the cited SOP source; none is treated as ground truth or PIC approval."
           if policy_ok and not reference_errors else "; ".join(reference_errors) or "Invalid visual-reference policy.")

    approval = sop.get("approval", {})
    research_ok = approval.get("approved_for_research_experiment") is True
    _check(checks, "sop_research_approval", research_ok,
           "Candidate SOP is explicitly approved only for the research experiment.")
    factory_ok = (approval.get("approved_for_factory_operation") is True
                  and approval.get("status") == "factory_pic_approved")
    _check(checks, "factory_pic_sop_approval", factory_ok,
           "A factory PIC still needs to approve the instructions, dependencies, and visible checks.", gate="factory")

    step_review_ok = all(row.get("review_status") == "factory_pic_approved" for row in by_id.values())
    _check(checks, "factory_pic_step_review", step_review_ok,
           "Each step is still marked pending factory PIC review.", gate="factory")

    # The current recovery is deliberately a stop/escalate fallback.  It is
    # safe for a research UI, but it is not a validated correction procedure.
    recovery_ok = all(row.get("recovery") for row in by_id.values())
    _check(checks, "safe_recovery_fallback", recovery_ok,
           "Every step has a stop-and-supervisor fallback; no fabricated correction is auto-prescribed.")
    _check(checks, "factory_pic_recovery_approval", False,
           "Step-specific corrective actions are not present because the dataset does not certify them.", gate="factory")


def _check_sources(sop: dict, checks: list[dict]) -> None:
    for source in sop.get("source_documents", []):
        if "path" not in source:
            continue
        path = _path(source["path"])
        exists = path.is_file()
        matches = exists and (not source.get("sha256") or _sha256(path) == source["sha256"])
        _check(checks, f"source_{source.get('id', path.stem)}", matches,
               f"Verified {source['path']} and its SHA-256."
               if matches else f"Missing or changed source document: {source['path']}.")


def _check_video(inference: dict, checks: list[dict]) -> None:
    video = _path(inference.get("video", ""))
    source = _path(inference.get("video_source", ""))
    if not video.is_file() or not source.is_file():
        _check(checks, "video_readable", False, "Video or video-provenance sidecar is missing.")
        return
    try:
        import cv2
        cap = cv2.VideoCapture(str(video))
        opened = cap.isOpened()
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        ok = opened and math.isclose(fps, 10.0, rel_tol=0, abs_tol=.01) and frames == 2754 \
            and (width, height) == (1280, 720)
        _check(checks, "video_readable", ok,
               f"Video opens as {width}x{height}, {fps:.2f} FPS, {frames} frames."
               if opened else "OpenCV could not open the video.")
    except (ImportError, OSError, ValueError) as error:
        _check(checks, "video_readable", False, f"Video probe failed: {error}")

    provenance = _load_json(source)
    identity_ok = (provenance.get("dataset") == "IndustReal"
                   and provenance.get("archive_member") == "03_assy_0_1.mp4"
                   and provenance.get("bytes") == video.stat().st_size)
    _check(checks, "video_provenance_valid", identity_ok,
           "Video identity and byte size match its source sidecar."
           if identity_ok else "Video provenance does not match the selected recording.")


def _check_models(inference: dict, checks: list[dict]) -> None:
    expected_files = {
        "entity_reference_encoder": ["config.json", "model.safetensors"],
        "semantic_localizer_and_terminal_verifier": ["config.json", "model.safetensors"],
        "tracker": ["config.json", "model.safetensors"],
    }
    models = inference.get("models", {})
    missing = []
    for name, files in expected_files.items():
        folder = _path(models.get(name, {}).get("path", ""))
        if not folder.is_dir() or any(not (folder / file).is_file() for file in files):
            missing.append(name)
    yolo = _path(models.get("open_vocabulary_proposals", {}).get("path", ""))
    if not yolo.is_file():
        missing.append("open_vocabulary_proposals")
    _check(checks, "model_files_present", not missing,
           "Pinned DINOv2, Qwen3-VL, SAM2.1, and YOLO-World files are present."
           if not missing else "Missing model files: " + ", ".join(missing))

    declared = {
        "entity_reference_encoder": (DINO_V2_ID, DINO_V2_REVISION),
        "semantic_localizer_and_terminal_verifier": (VLM_ID, VLM_REVISION),
        "tracker": (SAM2_ID, SAM2_REVISION),
        "open_vocabulary_proposals": (YOLO_WORLD_ID, None),
    }
    revision_errors = []
    for name, (model_id, revision) in declared.items():
        row = models.get(name, {})
        if row.get("id") != model_id or (revision is not None and row.get("revision") != revision):
            revision_errors.append(name)
    _check(checks, "model_manifest_matches_code", not revision_errors,
           "Model IDs and pinned revisions match monitor/settings.py."
           if not revision_errors else "Model manifest/code mismatch: " + ", ".join(revision_errors))

    runtime = inference.get("runtime_policy", {})
    scheduling = runtime.get("model_scheduling")
    scheduling_ok = (runtime.get("device") == "cuda" and runtime.get("batch_size") == 1
                     and scheduling in {"sequential", "stage_scoped"}
                     and runtime.get("release_between_stages") is True)
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        total_bytes = int(torch.cuda.mem_get_info()[1]) if cuda_ok else 0
        device_name = torch.cuda.get_device_name(0) if cuda_ok else "none"
    except (ImportError, RuntimeError) as error:
        cuda_ok, total_bytes, device_name = False, 0, str(error)
    hardware_ok = scheduling_ok and cuda_ok and total_bytes >= 5_500_000_000
    _check(checks, "runtime_memory_policy_valid", hardware_ok,
           f"CUDA device {device_name} has {total_bytes / 1024**3:.2f} GiB; {scheduling} batch-1 model scheduling is configured."
           if cuda_ok else f"CUDA runtime is unavailable: {device_name}.")


def _check_evaluation(manifest: dict, sop: dict, checks: list[dict]) -> None:
    evaluation = manifest.get("evaluation", {})
    provenance_path = _path(evaluation.get("provenance", ""))
    if not provenance_path.is_file():
        _check(checks, "evaluation_provenance_valid", False, "Evaluation provenance is missing.")
        return
    provenance = _load_json(provenance_path)
    hashes = {row["name"]: row["sha256"] for row in provenance.get("files", [])}
    errors = []
    for key in ("psr_events", "psr_states", "action_intervals", "assembly_boxes"):
        path = _path(evaluation.get(key, ""))
        if not path.is_file():
            errors.append(f"missing {key}")
        elif hashes.get(path.name) != _sha256(path):
            errors.append(f"hash mismatch for {path.name}")
    _check(checks, "evaluation_provenance_valid", not errors,
           "All official evaluation files match the recorded SHA-256 values."
           if not errors else "; ".join(errors))

    label_path = _path(evaluation.get("psr_events", ""))
    observed = {}
    event_rows = []
    if label_path.is_file():
        with label_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle):
                if row:
                    observed[f"IR{int(row[1]):02d}"] = row[2]
                    event_rows.append((int(Path(row[0]).stem), f"IR{int(row[1]):02d}"))
    expected_names = {step_id: values[2] for step_id, values in OFFICIAL_STEPS.items()}
    coverage_ok = observed == expected_names
    _check(checks, "evaluation_step_coverage", coverage_ok,
           "The selected recording contains held-out completion labels for all nine SOP steps."
           if coverage_ok else f"Expected {expected_names}; found {observed}.")

    order_errors = []
    if sop and event_rows:
        workflow = Workflow(sop)
        for frame, step_id in event_rows:
            snapshot = workflow.observe(step_id, timestamp=frame / 10.0, source="evaluation_audit")
            if snapshot["events"] and snapshot["events"][-1].get("status") == "out_of_order":
                order_errors.append(step_id)
    _check(checks, "selected_recording_order_accepted", not order_errors,
           "The partial-order SOP accepts the recording's alternate branch order without a false wrong-order alarm."
           if not order_errors else "False out-of-order alarms for: " + ", ".join(order_errors))

    inference_text = json.dumps(manifest.get("inference", {}), sort_keys=True).lower()
    forbidden = ["data/evaluation/", "psr_labels", "ar_labels", "od_labels"]
    leakage_free = not any(token in inference_text for token in forbidden)
    policy_ok = manifest.get("inference", {}).get("ground_truth_access") == \
        "forbidden_until_predictions_are_finalized"
    start_ok = manifest.get("inference", {}).get("start_s") == 0.0 \
        and manifest.get("inference", {}).get("process_to_end") is True
    _check(checks, "no_ground_truth_leakage", leakage_free and policy_ok and start_ok,
           "Inference starts at frame zero and contains no label paths or annotation-timed activation window."
           if leakage_free and policy_ok and start_ok else
           "Inference configuration contains evaluation data or a ground-truth-centred time window.")


def _check_references(inference: dict, checks: list[dict]) -> None:
    requirements_path = _path(inference.get("reference_requirements", ""))
    if not requirements_path.is_file():
        _check(checks, "approved_reference_coverage_complete", False,
               "Reference-requirements file is missing.", gate="reference")
        return
    requirements = _load_json(requirements_path)
    entities = requirements.get("entities", [])
    ready = bool(entities) and all(row.get("status") == "ready" for row in entities)
    missing = [row.get("name") for row in entities if row.get("status") != "ready"]
    _check(checks, "approved_reference_coverage_complete", ready,
           "All required entities have approved reference coverage."
           if ready else "Missing approved references for: " + ", ".join(missing), gate="reference")


def audit_end_to_end(manifest_path: Path | str = ROOT / "data/industreal_end_to_end.manifest.json") -> dict:
    manifest_path = _path(manifest_path)
    manifest = _load_json(manifest_path)
    checks: list[dict] = []
    if manifest.get("schema_version") != 1:
        _check(checks, "manifest_schema_valid", False, "Manifest must use schema_version 1.")
        return {"manifest": str(manifest_path), "checks": checks, "ready": {}}
    _check(checks, "manifest_schema_valid", True, "End-to-end manifest uses schema_version 1.")

    inference = manifest.get("inference", {})
    sop_path = _path(inference.get("sop", ""))
    if not sop_path.is_file():
        _check(checks, "sop_file_present", False, "Configured SOP file is missing.")
        sop = {}
    else:
        sop = _load_json(sop_path)
        _check(checks, "sop_file_present", True, f"Using {sop_path.relative_to(ROOT)}.")
        _check_sop(sop, checks)
        _check_sources(sop, checks)
    _check_video(inference, checks)
    _check_models(inference, checks)
    _check_evaluation(manifest, sop, checks)
    _check_references(inference, checks)

    research_failures = [row for row in checks if row["gate"] == "research" and row["status"] == "fail"]
    reference_failures = [row for row in checks if row["gate"] == "reference" and row["status"] == "fail"]
    factory_failures = [row for row in checks if row["gate"] == "factory" and row["status"] == "fail"]
    research_ready = not research_failures
    reference_ready = research_ready and not reference_failures
    factory_ready = reference_ready and not factory_failures
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "recording_id": manifest.get("recording_id"),
        "checks": checks,
        "ready": {
            "research_end_to_end_implementation": research_ready,
            "reference_guided_evaluation": reference_ready,
            "factory_operation": factory_ready,
        },
        "blocking": {
            "research": [row["id"] for row in research_failures],
            "reference": [row["id"] for row in reference_failures],
            "factory": [row["id"] for row in factory_failures],
        },
    }
