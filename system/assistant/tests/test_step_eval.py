"""Unit tests for sealed step-rule evaluation against PSR fixtures."""
import json
import tempfile
import unittest
from pathlib import Path

from monitor.settings import ROOT
from monitor.step_eval import (
    completed_steps_at,
    extract_completion_events,
    load_psr_events,
    load_psr_raw_states,
    ready_steps_at,
    score_step_observations,
    validate_prediction_bundle,
)


PSR_EVENTS = ROOT / "data/evaluation/official/03_assy_0_1/PSR_labels.csv"
PSR_RAW = ROOT / "data/evaluation/official/03_assy_0_1/PSR_labels_raw.csv"


class StepEvalTests(unittest.TestCase):
    def test_psr_timeline_matches_known_completions(self):
        events = load_psr_events(PSR_EVENTS)
        by_time = {}
        for row in events:
            by_time.setdefault(row["timestamp_s"], []).append(row["step_id"])
        self.assertEqual(sorted(by_time[47.7]), ["IR09", "IR15"])
        self.assertEqual(by_time[165.5], ["IR30"])
        self.assertEqual(sorted(by_time[268.9]), ["IR21", "IR24", "IR27"])

    def test_completed_and_ready_sets_follow_raw_states(self):
        raw = load_psr_raw_states(PSR_RAW)
        before = completed_steps_at(raw, 40.0)
        self.assertEqual(before, set())
        after_rear = completed_steps_at(raw, 70.0)
        self.assertTrue({"IR09", "IR15", "IR18"} <= after_rear)
        ready = ready_steps_at(after_rear)
        self.assertIn("IR03", ready)
        self.assertNotIn("IR21", ready)

    def test_refuses_unfinalized_or_leaky_predictions(self):
        with self.assertRaisesRegex(ValueError, "finalized"):
            validate_prediction_bundle({"ground_truth_used_for_inference": False,
                                        "observations": [{"timestamp_s": 1}]})
        with self.assertRaisesRegex(ValueError, "ground truth"):
            validate_prediction_bundle({
                "predictions_finalized": True,
                "ground_truth_used_for_inference": True,
                "observations": [{"timestamp_s": 1}],
            })
        with self.assertRaisesRegex(ValueError, "evaluation"):
            validate_prediction_bundle({
                "predictions_finalized": True,
                "ground_truth_used_for_inference": False,
                "labels": "data/evaluation/official/03_assy_0_1/PSR_labels.csv",
                "observations": [{"timestamp_s": 1}],
            })

    def test_perfect_completed_timeline_scores_high(self):
        raw = load_psr_raw_states(PSR_RAW)
        observations = []
        for timestamp in (0.0, 50.0, 120.0, 170.0, 270.0):
            completed = sorted(completed_steps_at(raw, timestamp))
            observations.append({
                "timestamp_s": timestamp,
                "step": "unknown",
                "status": "no_step_evidence",
                "completed_steps": completed,
                "missing_prerequisites": [],
            })
        metrics = score_step_observations(
            observations, psr_events_path=PSR_EVENTS, psr_raw_path=PSR_RAW)
        self.assertAlmostEqual(metrics["completed_set"]["mean_f1"], 1.0)
        self.assertAlmostEqual(metrics["completed_set"]["mean_jaccard"], 1.0)

    def test_ready_membership_and_mistake_precision(self):
        observations = [
            {
                "timestamp_s": 50.0,
                "step": "IR03",
                "status": "step_in_progress",
                "certainty": "clear",
                "score": 5,
                "completed_steps": ["IR09", "IR15", "IR18"],
                "missing_prerequisites": [],
            },
            {
                "timestamp_s": 40.0,
                "step": "IR30",
                "status": "mistake_prerequisites_missing",
                "certainty": "clear",
                "score": 5,
                "completed_steps": [],
                "missing_prerequisites": ["IR09"],
            },
            {
                "timestamp_s": 170.0,
                "step": "IR30",
                "status": "mistake_prerequisites_missing",
                "certainty": "clear",
                "score": 5,
                "completed_steps": ["IR03", "IR06", "IR09", "IR15", "IR18", "IR30"],
                "missing_prerequisites": ["IR09"],
            },
        ]
        metrics = score_step_observations(
            observations, psr_events_path=PSR_EVENTS, psr_raw_path=PSR_RAW)
        self.assertEqual(metrics["ready_or_completed_membership"]["hits"], 2)
        self.assertEqual(metrics["ready_or_completed_membership"]["total_non_abstain"], 3)
        self.assertEqual(metrics["mistake_prerequisites"]["true_positives"], 1)
        self.assertEqual(metrics["mistake_prerequisites"]["false_positives"], 1)

    def test_completion_event_extraction_is_causal(self):
        events = extract_completion_events([
            {"timestamp_s": 10, "completed_steps": ["IR09"]},
            {"timestamp_s": 20, "completed_steps": ["IR09", "IR15"]},
            {"timestamp_s": 30, "completed_steps": ["IR09", "IR15"]},
        ])
        self.assertEqual([row["step_id"] for row in events], ["IR09", "IR15"])
        self.assertEqual(events[1]["timestamp_s"], 20)


if __name__ == "__main__":
    unittest.main()
