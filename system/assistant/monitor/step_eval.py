"""Post-inference evaluation of SOP step-rule predictions vs IndustReal PSR.

Labels are opened only after predictions are validated. This module never feeds
PSR/AR into detection or step inference.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from .psr_metrics import score_psr_events
from .readiness import EXPECTED_PREREQUISITES, OFFICIAL_STEPS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_psr_events(path: Path, *, source_fps: float = 10.0) -> list[dict]:
    """Load official completion events as IR step IDs."""
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            official_id = int(row[1])
            ir_id = next((key for key, value in OFFICIAL_STEPS.items()
                          if value[0] == official_id), None)
            if ir_id is None:
                continue
            rows.append({
                "step_id": ir_id,
                "official_step_id": official_id,
                "timestamp_s": int(Path(row[0]).stem) / float(source_fps),
                "description": row[2] if len(row) > 2 else "",
            })
    return rows


def load_psr_raw_states(path: Path, *, source_fps: float = 10.0) -> list[dict]:
    """Load sparse component-state snapshots."""
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            states = [int(value) for value in row[1:]]
            rows.append({
                "timestamp_s": int(Path(row[0]).stem) / float(source_fps),
                "states": states,
            })
    rows.sort(key=lambda item: item["timestamp_s"])
    return rows


def completed_steps_at(raw_rows: list[dict], timestamp_s: float) -> set[str]:
    """Latest PSR raw snapshot at or before timestamp → completed IR ids."""
    latest = None
    for row in raw_rows:
        if row["timestamp_s"] <= timestamp_s + 1e-9:
            latest = row
        else:
            break
    if latest is None:
        return set()
    completed = set()
    for ir_id, (_official, state_index, _name) in OFFICIAL_STEPS.items():
        if state_index < len(latest["states"]) and latest["states"][state_index] == 1:
            completed.add(ir_id)
    return completed


def ready_steps_at(completed: set[str]) -> set[str]:
    return {
        step_id
        for step_id, prerequisites in EXPECTED_PREREQUISITES.items()
        if step_id not in completed and set(prerequisites) <= completed
    }


def _set_scores(predicted: set[str], truth: set[str]) -> dict:
    true_positives = len(predicted & truth)
    false_positives = len(predicted - truth)
    false_negatives = len(truth - predicted)
    if not predicted and not truth:
        precision = recall = f1 = jaccard = 1.0
    else:
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        union = len(predicted | truth)
        jaccard = true_positives / union if union else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted": sorted(predicted),
        "truth": sorted(truth),
    }


def extract_completion_events(observations: list[dict]) -> list[dict]:
    """Emit chronological completion events when a step first enters completed_steps."""
    seen: set[str] = set()
    events = []
    for row in observations:
        completed = set(row.get("completed_steps") or [])
        newly = sorted(completed - seen)
        timestamp = float(row.get("timestamp_s", 0))
        for step_id in newly:
            if step_id in OFFICIAL_STEPS:
                events.append({
                    "step_id": step_id,
                    "timestamp_s": timestamp,
                    "confidence": float(row.get("score", 1) or 1) / 10.0,
                })
        seen |= completed
    events.sort(key=lambda item: (item["timestamp_s"], item["step_id"]))
    return events


def extract_current_help_events(observations: list[dict]) -> list[dict]:
    """First sticky CURRENT (worker-help) observation per step.

    Prefers ``current_step`` / ``help_active`` when present; falls back to
    ``step`` while ``status == step_in_progress``.
    """
    first: dict[str, dict] = {}
    for row in observations:
        step_id = row.get("current_step") if row.get("help_active") else None
        if step_id in {None, "unknown", ""}:
            if row.get("status") == "step_in_progress":
                step_id = row.get("step")
        if step_id in {None, "unknown", ""}:
            continue
        if step_id not in OFFICIAL_STEPS:
            continue
        if step_id not in first:
            first[step_id] = {
                "step_id": step_id,
                "timestamp_s": float(row.get("timestamp_s", 0)),
                "confidence": float(row.get("score", 1) or 1) / 10.0,
                "status": row.get("status"),
                "source": "current_help",
            }
    return sorted(first.values(), key=lambda item: (item["timestamp_s"], item["step_id"]))


def extract_first_evidence_events(observations: list[dict]) -> list[dict]:
    """First clear in-progress observation per step (diagnostic timing, not completion)."""
    first: dict[str, dict] = {}
    for row in observations:
        step_id = row.get("step")
        if step_id in {None, "unknown"}:
            continue
        if row.get("status") not in {"step_in_progress", "mistake_prerequisites_missing"}:
            continue
        if row.get("certainty") == "uncertain" and row.get("score", 0) < 5:
            continue
        if step_id not in first:
            first[step_id] = {
                "step_id": step_id,
                "timestamp_s": float(row.get("timestamp_s", 0)),
                "confidence": float(row.get("score", 1) or 1) / 10.0,
                "status": row.get("status"),
            }
    return sorted(first.values(), key=lambda item: (item["timestamp_s"], item["step_id"]))



def score_step_observations(
    observations: list[dict],
    *,
    psr_events_path: Path,
    psr_raw_path: Path,
    source_fps: float = 10.0,
) -> dict:
    """Score a finalized observation timeline against held-out PSR labels."""
    if not isinstance(observations, list) or not observations:
        raise ValueError("Observations must be a non-empty list.")
    for row in observations:
        if not isinstance(row, dict) or "timestamp_s" not in row:
            raise ValueError("Every observation needs timestamp_s.")

    events = load_psr_events(psr_events_path, source_fps=source_fps)
    raw_rows = load_psr_raw_states(psr_raw_path, source_fps=source_fps)

    completed_scores = []
    ready_hits = 0
    ready_total = 0
    abstentions = 0
    mistake_tp = 0
    mistake_fp = 0
    mistake_total = 0

    for row in observations:
        timestamp = float(row["timestamp_s"])
        truth_completed = completed_steps_at(raw_rows, timestamp)
        truth_ready = ready_steps_at(truth_completed)
        predicted_completed = {step for step in row.get("completed_steps") or []
                               if step in OFFICIAL_STEPS}
        completed_scores.append({
            "timestamp_s": timestamp,
            **_set_scores(predicted_completed, truth_completed),
            "ready_truth": sorted(truth_ready),
        })

        predicted_step = row.get("step") or "unknown"
        if predicted_step in {None, "unknown"} or row.get("status") == "no_step_evidence":
            abstentions += 1
        else:
            ready_total += 1
            if predicted_step in truth_ready or predicted_step in truth_completed:
                ready_hits += 1

        if row.get("status") == "mistake_prerequisites_missing":
            mistake_total += 1
            missing = set(row.get("missing_prerequisites") or [])
            # True positive: at least one claimed-missing prereq is truly incomplete.
            if missing - truth_completed:
                mistake_tp += 1
            else:
                mistake_fp += 1

    def _mean(key: str) -> float | None:
        values = [row[key] for row in completed_scores]
        return sum(values) / len(values) if values else None

    completion_events = extract_completion_events(observations)
    first_evidence = extract_first_evidence_events(observations)
    current_help = extract_current_help_events(observations)
    truth_for_psr = [{"step_id": row["official_step_id"], "timestamp_s": row["timestamp_s"]}
                     for row in events]
    predicted_for_psr = [
        {"step_id": row["step_id"], "timestamp_s": row["timestamp_s"],
         "confidence": row.get("confidence", 1.0)}
        for row in completion_events
    ]
    evidence_for_psr = [
        {"step_id": row["step_id"], "timestamp_s": row["timestamp_s"],
         "confidence": row.get("confidence", 1.0)}
        for row in first_evidence
    ]
    help_for_psr = [
        {"step_id": row["step_id"], "timestamp_s": row["timestamp_s"],
         "confidence": row.get("confidence", 1.0)}
        for row in current_help
    ]
    help_metrics = score_psr_events(truth_for_psr, help_for_psr)
    # Live-help timing vs GT DONE: negative = help before official completion (good).
    help_latencies = []
    gt_by_ir = {row["step_id"]: row for row in events}
    for row in current_help:
        gt = gt_by_ir.get(row["step_id"])
        if not gt:
            continue
        help_latencies.append({
            "step_id": row["step_id"],
            "ground_truth_s": gt["timestamp_s"],
            "help_s": row["timestamp_s"],
            "help_minus_gt_s": row["timestamp_s"] - gt["timestamp_s"],
        })

    return {
        "sample_count": len(observations),
        "abstention_rate": abstentions / len(observations),
        "completed_set": {
            "mean_precision": _mean("precision"),
            "mean_recall": _mean("recall"),
            "mean_f1": _mean("f1"),
            "mean_jaccard": _mean("jaccard"),
            "per_sample": completed_scores,
        },
        "ready_or_completed_membership": {
            "hits": ready_hits,
            "total_non_abstain": ready_total,
            "hit_rate": ready_hits / ready_total if ready_total else None,
            "note": (
                "Predicted step is counted correct if it is ready or already complete "
                "under the official DAG at that time. Unique current-step Top-1 is not GT."
            ),
        },
        "mistake_prerequisites": {
            "count": mistake_total,
            "true_positives": mistake_tp,
            "false_positives": mistake_fp,
            "precision": mistake_tp / mistake_total if mistake_total else None,
            "note": (
                "Evaluates prerequisite-logic mistakes vs PSR completed states. "
                "This recording has no official worker-error events."
            ),
        },
        "completion_events": {
            "predicted": completion_events,
            "metrics": score_psr_events(truth_for_psr, predicted_for_psr),
            "note": "Strict PSR matching on steps first marked completed by the reasoner.",
        },
        "first_evidence_events": {
            "predicted": first_evidence,
            "metrics": score_psr_events(truth_for_psr, evidence_for_psr),
            "note": (
                "Diagnostic only: first clear object evidence per step. "
                "Early evidence before GT completion counts as a false positive here."
            ),
        },
        "current_help_events": {
            "predicted": current_help,
            "metrics": help_metrics,
            "latency_vs_gt": help_latencies,
            "mean_help_minus_gt_s": (
                sum(row["help_minus_gt_s"] for row in help_latencies) / len(help_latencies)
                if help_latencies else None
            ),
            "steps_helped": len(help_latencies),
            "steps_gt": len(events),
            "help_coverage": (
                len(help_latencies) / len(events) if events else None
            ),
            "note": (
                "Worker-facing sticky CURRENT (hand↔part). Negative help_minus_gt means "
                "help appeared before the official DONE time — desired for assist. "
                "PSR-style F1 still treats early help as FP; prefer help_coverage + "
                "mean_help_minus_gt_s for live-assist claims."
            ),
        },
        "ground_truth_events": events,
        "labels": {
            "psr_events": str(psr_events_path),
            "psr_raw": str(psr_raw_path),
            "psr_events_sha256": sha256(psr_events_path),
            "psr_raw_sha256": sha256(psr_raw_path),
            "source_fps": source_fps,
        },
    }


def validate_prediction_bundle(value: dict) -> list[dict]:
    """Refuse to score until the prediction bundle is finalized and label-free."""
    if not isinstance(value, dict):
        raise ValueError("Prediction bundle must be an object.")
    if value.get("predictions_finalized") is not True:
        raise ValueError("Refusing to score before predictions are finalized.")
    if value.get("ground_truth_used_for_inference") is not False:
        raise ValueError("Prediction bundle must state that ground truth was not used.")
    blob = json_dumps_lower(value)
    forbidden = ("psr_labels", "ar_labels", "od_labels", "data/evaluation/")
    if any(token in blob for token in forbidden):
        raise ValueError("Prediction bundle appears to embed evaluation label paths.")
    rows = value.get("observations")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prediction bundle needs a non-empty observations array.")
    return rows


def json_dumps_lower(value: Any) -> str:
    import json
    return json.dumps(value, sort_keys=True).lower()
