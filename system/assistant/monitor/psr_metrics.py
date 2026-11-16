"""IndustReal-style event metrics for finalized end-to-end predictions."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def _step_id(value) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    text = str(value).strip().upper()
    if text.startswith("IR"):
        text = text[2:]
    if not text.isdigit():
        raise ValueError(f"Invalid procedure step ID: {value}")
    return int(text)


def _events(rows: Iterable[dict]) -> list[dict]:
    output = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every procedure event must be an object.")
        step_id = _step_id(row.get("step_id"))
        timestamp = float(row.get("timestamp_s"))
        if timestamp < 0:
            raise ValueError("Procedure event timestamps must be non-negative.")
        output.append({"step_id": step_id, "timestamp_s": timestamp,
                       "confidence": float(row.get("confidence", 1.0))})
    if any(first["timestamp_s"] > second["timestamp_s"]
           for first, second in zip(output, output[1:])):
        raise ValueError("Procedure events must be chronological.")
    return output


def weighted_damerau_levenshtein(first: list[int], second: list[int]) -> int:
    """Adjacent-transposition edit distance used by the IndustReal POS setup.

    Insertions/deletions cost 1, substitutions cost 2, and adjacent
    transpositions cost 1.  A substitution therefore cannot be cheaper than a
    delete plus insert, matching the public IndustReal evaluation code.
    """
    rows, cols = len(first) + 1, len(second) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            substitution = 0 if first[i - 1] == second[j - 1] else 2
            best = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + substitution,
            )
            if (i > 1 and j > 1 and first[i - 1] == second[j - 2]
                    and first[i - 2] == second[j - 1]):
                best = min(best, distance[i - 2][j - 2] + 1)
            distance[i][j] = best
    return distance[-1][-1]


def procedure_order_similarity(ground_truth_ids: list[int], prediction_ids: list[int]) -> float | None:
    if not ground_truth_ids:
        return None
    distance = weighted_damerau_levenshtein(ground_truth_ids, prediction_ids)
    return 1.0 - min(distance / len(ground_truth_ids), 1.0)


def score_psr_events(ground_truth: Iterable[dict], predictions: Iterable[dict]) -> dict:
    """Score finalized completion events without matching an early prediction.

    For each ground-truth completion, the earliest unused prediction of the
    same step at or after completion is a true positive. Predictions before the
    real completion (and duplicates) are false positives. Unmatched ground
    truths are false negatives.
    """
    truth = _events(ground_truth)
    predicted = _events(predictions)
    truth_by_step: dict[int, list[dict]] = defaultdict(list)
    predicted_by_step: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for row in truth:
        truth_by_step[row["step_id"]].append(row)
    for index, row in enumerate(predicted):
        predicted_by_step[row["step_id"]].append((index, row))

    used_predictions: set[int] = set()
    delays = []
    matches = []
    false_negatives = 0
    for step_id, truth_rows in truth_by_step.items():
        candidates = predicted_by_step.get(step_id, [])
        for true_row in truth_rows:
            match = next(((index, row) for index, row in candidates
                          if index not in used_predictions
                          and row["timestamp_s"] >= true_row["timestamp_s"]), None)
            if match is None:
                false_negatives += 1
                continue
            index, predicted_row = match
            used_predictions.add(index)
            delay = predicted_row["timestamp_s"] - true_row["timestamp_s"]
            delays.append(delay)
            matches.append({"step_id": step_id,
                            "ground_truth_s": true_row["timestamp_s"],
                            "prediction_s": predicted_row["timestamp_s"],
                            "delay_s": delay})

    true_positives = len(used_predictions)
    false_positives = len(predicted) - true_positives
    precision = true_positives / (true_positives + false_positives) \
        if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) \
        if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    truth_ids = [row["step_id"] for row in truth]
    predicted_ids = [row["step_id"] for row in predicted]
    return {
        "procedure_order_similarity": procedure_order_similarity(truth_ids, predicted_ids),
        "system_f1": f1,
        "precision": precision,
        "recall": recall,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "average_completion_delay_seconds": sum(delays) / len(delays) if delays else None,
        "matched_events": matches,
        "ground_truth_sequence": truth_ids,
        "prediction_sequence": predicted_ids,
    }

