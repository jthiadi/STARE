"""Deterministic SOP object → step / mistake / recovery rules."""
import json
import unittest
from pathlib import Path

from monitor.settings import ROOT
from monitor.step_rules import (
    build_step_rules,
    close_prerequisites,
    infer_step_from_detections,
    reason_over_detection_records,
    score_step_rule,
)


TEMPLATE = ROOT / "data/industreal_assembly.template.json"


class StepRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sop = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        cls.rules = build_step_rules(cls.sop)

    def test_rules_cover_every_template_step(self):
        self.assertEqual([row["step_id"] for row in self.rules],
                         [step["id"] for step in self.sop["steps"]])
        ir30 = next(row for row in self.rules if row["step_id"] == "IR30")
        self.assertTrue(any("wheel" in query.lower() for query in ir30["moving_queries"]))
        self.assertTrue(any("rear wheel" in query.lower() for query in ir30["moving_queries"]))
        self.assertIn("IR09", ir30["prerequisites"])

    def test_rear_wheel_and_hand_map_to_ir30(self):
        result = infer_step_from_detections(
            ["hand", "rear wheel assembly"], self.rules,
            completed={"IR03", "IR06", "IR09", "IR15", "IR18"})
        self.assertEqual(result["step"], "IR30")
        self.assertEqual(result["status"], "step_in_progress")
        self.assertEqual(result["action"], "pick_up")
        self.assertFalse(result["recovery"])

    def test_idle_part_without_hand_does_not_claim_a_step(self):
        result = infer_step_from_detections(
            ["rear wheel assembly", "white front bracket"], self.rules,
            completed={"IR03", "IR06", "IR09", "IR15", "IR18"})
        self.assertEqual(result["step"], "unknown")
        self.assertEqual(result["status"], "no_step_evidence")
        self.assertIn("hand contact or motion", result["evidence"].lower())

    def test_hand_far_from_part_does_not_claim_a_step(self):
        objects = [
            {"class": "hand", "box": [10.0, 10.0, 40.0, 40.0]},
            {"class": "rear wheel assembly", "box": [400.0, 300.0, 480.0, 380.0]},
        ]
        result = infer_step_from_detections(
            [row["class"] for row in objects], self.rules,
            completed={"IR03", "IR06", "IR09", "IR15", "IR18"},
            objects=objects)
        self.assertEqual(result["step"], "unknown")
        self.assertFalse(any(row.get("near_hand") for row in result["scores"]
                             if row["step_id"] == "IR30"))

    def test_hand_near_part_claims_step(self):
        objects = [
            {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
            {"class": "rear wheel assembly", "box": [140.0, 120.0, 220.0, 200.0]},
        ]
        result = infer_step_from_detections(
            [row["class"] for row in objects], self.rules,
            completed={"IR03", "IR06", "IR09", "IR15", "IR18"},
            objects=objects)
        self.assertEqual(result["step"], "IR30")
        self.assertEqual(result["status"], "step_in_progress")
        ir30 = next(row for row in result["scores"] if row["step_id"] == "IR30")
        self.assertTrue(ir30["near_hand"])

    def test_motion_alone_without_hand_does_not_activate(self):
        # Detector jitter on an idle part must not look like work in progress.
        objects = [
            {"class": "front wheel assembly", "box": [200.0, 200.0, 280.0, 280.0]},
        ]
        previous = {"front wheel assembly": (120.0, 120.0)}
        result = infer_step_from_detections(
            [row["class"] for row in objects], self.rules,
            completed={"IR03", "IR06", "IR09", "IR15", "IR18", "IR21", "IR24"},
            objects=objects, previous_centers=previous)
        self.assertEqual(result["step"], "unknown")

    def test_motion_with_hand_visible_can_activate(self):
        objects = [
            {"class": "hand", "box": [10.0, 10.0, 40.0, 40.0]},
            {"class": "front wheel assembly", "box": [200.0, 200.0, 280.0, 280.0]},
        ]
        previous = {"front wheel assembly": (120.0, 120.0)}
        result = infer_step_from_detections(
            [row["class"] for row in objects], self.rules,
            completed={"IR03", "IR06", "IR09", "IR15", "IR18", "IR21", "IR24"},
            objects=objects, previous_centers=previous)
        self.assertEqual(result["step"], "IR27")
        self.assertEqual(result["status"], "step_in_progress")
        ir27 = next(row for row in result["scores"] if row["step_id"] == "IR27")
        self.assertTrue(ir27["in_motion"])
        self.assertFalse(ir27["near_hand"])

    def test_topdown_ignores_blocked_wheel_at_cold_start(self):
        result = infer_step_from_detections(
            ["hand", "rear wheel assembly"], self.rules, completed=set())
        self.assertEqual(result["step"], "unknown")
        self.assertEqual(result["status"], "no_step_evidence")
        self.assertEqual(result["ready_steps"], ["IR03", "IR09"])

    def test_legacy_bottomup_still_surfaces_prerequisite_mistakes(self):
        result = infer_step_from_detections(
            ["hand", "rear wheel assembly"], self.rules, completed=set(),
            ready_only=False)
        self.assertEqual(result["step"], "IR30")
        self.assertEqual(result["status"], "mistake_prerequisites_missing")
        self.assertIn("IR09", result["missing_prerequisites"])
        self.assertTrue(result["recovery"])

    def test_front_bracket_maps_to_ir21(self):
        completed = close_prerequisites(self.rules, set(), "IR21")
        result = infer_step_from_detections(
            ["hand", "white front bracket"], self.rules, completed=completed)
        self.assertEqual(result["step"], "IR21")
        self.assertEqual(result["status"], "step_in_progress")

    def test_topdown_front_wheel_ignored_until_ir24_ready(self):
        # Chassis branch done, but IR21/IR24 not done → IR27 not ready.
        completed = {"IR03", "IR06", "IR09", "IR15", "IR18"}
        result = infer_step_from_detections(
            ["hand", "front wheel assembly"], self.rules, completed=completed)
        self.assertEqual(result["step"], "unknown")
        self.assertIn("IR21", result["ready_steps"])
        self.assertIn("IR30", result["ready_steps"])

    def test_legacy_front_wheel_without_bracket_screw_is_a_mistake(self):
        completed = {"IR03", "IR06", "IR09", "IR15", "IR18"}
        result = infer_step_from_detections(
            ["hand", "front wheel assembly"], self.rules, completed=completed,
            ready_only=False)
        self.assertEqual(result["step"], "IR27")
        self.assertEqual(result["status"], "mistake_prerequisites_missing")
        self.assertIn("IR24", result["missing_prerequisites"])

    def test_timeline_topdown_ignores_early_wheel_until_ready(self):
        records = [
            {"index": 0, "timestamp_s": 245.0,
             "objects": [{"class": "hand"}, {"class": "rear wheel assembly"}]},
            {"index": 1, "timestamp_s": 246.0,
             "objects": [{"class": "hand"}, {"class": "front wheel assembly"}]},
        ]
        rows = reason_over_detection_records(records, self.sop, initial_completed=[])
        self.assertEqual(rows[0]["status"], "no_step_evidence")
        self.assertEqual(rows[0]["step"], "unknown")
        # After chassis prereqs, IR30 is ready; IR27 still needs IR21/IR24.
        boot = list(close_prerequisites(self.rules, set(), "IR30"))
        rows = reason_over_detection_records(records, self.sop, initial_completed=boot)
        self.assertEqual(rows[0]["status"], "step_in_progress")
        self.assertEqual(rows[0]["step"], "IR30")
        self.assertEqual(rows[1]["step"], "unknown")

    def test_timeline_ignores_idle_table_clutter(self):
        records = [
            {"index": 0, "timestamp_s": 10.0,
             "objects": [
                 {"class": "white front bracket", "box": [50.0, 50.0, 120.0, 100.0]},
                 {"class": "rear wheel assembly", "box": [300.0, 200.0, 380.0, 280.0]},
             ]},
            {"index": 1, "timestamp_s": 11.0,
             "objects": [
                 {"class": "hand", "box": [140.0, 90.0, 190.0, 140.0]},
                 {"class": "white front bracket", "box": [150.0, 100.0, 220.0, 160.0]},
             ]},
        ]
        completed = list(close_prerequisites(self.rules, set(), "IR21"))
        rows = reason_over_detection_records(
            records, self.sop, initial_completed=completed)
        self.assertEqual(rows[0]["step"], "unknown")
        self.assertEqual(rows[1]["step"], "IR21")
        self.assertEqual(rows[1]["status"], "step_in_progress")

    def test_prefers_ready_step_over_similar_out_of_order(self):
        # Hand near short braces (IR03 ready) should beat white bracket (IR21 blocked).
        objects = [
            {"class": "hand", "box": [100.0, 100.0, 150.0, 150.0]},
            {"class": "short brace", "box": [120.0, 110.0, 200.0, 180.0]},
            {"class": "white front bracket", "box": [130.0, 120.0, 190.0, 170.0]},
        ]
        result = infer_step_from_detections(
            [row["class"] for row in objects], self.rules, completed=set(), objects=objects)
        self.assertEqual(result["step"], "IR03")
        self.assertEqual(result["status"], "step_in_progress")

    def test_legacy_sustained_blocked_work_trusts_prerequisites(self):
        records = []
        for index, timestamp in enumerate([0.0, 2.0, 4.0, 6.0, 8.0, 9.0]):
            records.append({
                "index": index,
                "timestamp_s": timestamp,
                "objects": [
                    {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
                    {"class": "rear wheel assembly", "box": [140.0, 120.0, 220.0, 200.0]},
                ],
            })
        rows = reason_over_detection_records(
            records, self.sop, initial_completed=[], sustained_trust_s=5.0,
            min_dwell_s=3.0, ready_only=False)
        self.assertEqual(rows[0]["status"], "mistake_prerequisites_missing")
        trusted = next(row for row in rows if row.get("progress_trust"))
        self.assertEqual(trusted["status"], "step_in_progress")
        self.assertIn("IR09", trusted["completed_steps"])

    def test_dwell_then_idle_marks_step_complete(self):
        records = [
            {"index": 0, "timestamp_s": 0.0,
             "objects": [{"class": "hand"}, {"class": "short brace"}]},
            {"index": 1, "timestamp_s": 1.0,
             "objects": [{"class": "hand"}, {"class": "short brace"}]},
            {"index": 2, "timestamp_s": 2.0,
             "objects": [{"class": "hand"}, {"class": "short brace"}]},
            {"index": 3, "timestamp_s": 3.0,
             "objects": [{"class": "hand"}, {"class": "short brace"}]},
            {"index": 4, "timestamp_s": 4.5,
             "objects": [{"class": "hand"}]},
        ]
        rows = reason_over_detection_records(
            records, self.sop, initial_completed=["IR09", "IR15", "IR18"],
            min_dwell_s=2.5, sustained_trust_s=99)
        self.assertEqual(rows[0]["status"], "step_in_progress")
        self.assertEqual(rows[0]["step"], "IR03")
        self.assertIn("IR03", rows[-1]["completed_steps"])

    def test_dwell_without_attach_does_not_complete_when_target_visible(self):
        # Hand works the brace, but the partial model stays far away → no DONE.
        records = []
        for index, timestamp in enumerate([float(i) for i in range(0, 14)] + [15.5]):
            if timestamp < 14.5:
                objects = [
                    {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
                    {"class": "long brace", "box": [140.0, 120.0, 260.0, 180.0]},
                    {"class": "partial model", "box": [500.0, 400.0, 620.0, 520.0]},
                ]
            else:
                objects = [
                    {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
                    {"class": "partial model", "box": [500.0, 400.0, 620.0, 520.0]},
                ]
            records.append({"index": index, "timestamp_s": timestamp, "objects": objects})
        rows = reason_over_detection_records(
            records, self.sop, initial_completed=[], min_dwell_s=10.0)
        self.assertNotIn("IR09", rows[-1]["completed_steps"])

    def test_dwell_with_attach_marks_step_complete(self):
        # Brace overlaps the partial model while worked, then idle → DONE.
        records = []
        for index, timestamp in enumerate([float(i) for i in range(0, 14)] + [15.5]):
            if timestamp < 14.5:
                objects = [
                    {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
                    {"class": "long brace", "box": [140.0, 120.0, 260.0, 180.0]},
                    {"class": "partial model", "box": [150.0, 110.0, 280.0, 220.0]},
                ]
            else:
                objects = [
                    {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
                    {"class": "partial model", "box": [150.0, 110.0, 280.0, 220.0]},
                ]
            records.append({"index": index, "timestamp_s": timestamp, "objects": objects})
        rows = reason_over_detection_records(
            records, self.sop, initial_completed=[], min_dwell_s=10.0)
        self.assertIn("IR09", rows[-1]["completed_steps"])
        self.assertNotIn("IR03", rows[-1]["completed_steps"])

    def test_parallel_ready_branch_can_finish_independently(self):
        # IR09 (long brace) gets sustained hand work then idle, while IR03 never works.
        records = []
        for index, timestamp in enumerate(
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                 11.0, 12.0, 13.0, 14.0, 15.0, 16.5]):
            if timestamp < 15.5:
                objects = [
                    {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
                    {"class": "long brace", "box": [140.0, 120.0, 260.0, 180.0]},
                ]
            else:
                objects = [{"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]}]
            records.append({"index": index, "timestamp_s": timestamp, "objects": objects})
        rows = reason_over_detection_records(
            records, self.sop, initial_completed=[], min_dwell_s=12.0)
        self.assertIn("IR09", rows[-1]["completed_steps"])
        self.assertNotIn("IR03", rows[-1]["completed_steps"])
        self.assertIn("IR03", rows[-1]["ready_steps"])

    def test_front_brace_waits_while_rear_brace_still_ready(self):
        # Short brace alone is enough to work IR03, but IR09 stays ready so IR03
        # must not finalize first.
        records = []
        for index, timestamp in enumerate([float(i) for i in range(0, 18)] + [19.5]):
            if timestamp < 18.5:
                objects = [
                    {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
                    {"class": "short brace", "box": [140.0, 120.0, 220.0, 180.0]},
                ]
            else:
                objects = [{"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]}]
            records.append({"index": index, "timestamp_s": timestamp, "objects": objects})
        rows = reason_over_detection_records(
            records, self.sop, initial_completed=[], min_dwell_s=12.0)
        self.assertNotIn("IR03", rows[-1]["completed_steps"])
        self.assertIn("IR03", rows[-1]["ready_steps"])
        self.assertIn("IR09", rows[-1]["ready_steps"])

    def test_idle_parallel_part_near_hand_does_not_finish_sibling(self):
        # Short brace is worked; long brace sits near the hand but never moves.
        # IR03 must not finalize while IR09 remains ready; IR09 must not false-complete.
        records = []
        long_idle = [230.0, 120.0, 320.0, 180.0]
        timestamps = [float(i) for i in range(0, 17)] + [18.5]
        for index, timestamp in enumerate(timestamps):
            if timestamp < 17.5:
                sx = 140.0 + timestamp * 3.0
                objects = [
                    {"class": "hand", "box": [100.0 + timestamp * 3.0, 100.0, 160.0 + timestamp * 3.0, 160.0]},
                    {"class": "short brace", "box": [sx, 120.0, sx + 80.0, 180.0]},
                    {"class": "long brace", "box": list(long_idle)},
                ]
            else:
                objects = [
                    {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
                    {"class": "long brace", "box": list(long_idle)},
                ]
            records.append({"index": index, "timestamp_s": timestamp, "objects": objects})
        rows = reason_over_detection_records(
            records, self.sop, initial_completed=[], min_dwell_s=12.0)
        self.assertNotIn("IR09", rows[-1]["completed_steps"])
        self.assertIn("IR09", rows[-1]["ready_steps"])
        self.assertIn("IR03", rows[-1]["ready_steps"])

    def test_motion_only_does_not_complete_a_step(self):
        records = [
            {"index": 0, "timestamp_s": 0.0,
             "objects": [
                 {"class": "hand", "box": [10.0, 10.0, 40.0, 40.0]},
                 {"class": "short brace", "box": [200.0, 200.0, 280.0, 280.0]},
             ]},
            {"index": 1, "timestamp_s": 1.0,
             "objects": [
                 {"class": "hand", "box": [10.0, 10.0, 40.0, 40.0]},
                 {"class": "short brace", "box": [260.0, 200.0, 340.0, 280.0]},
             ]},
            {"index": 2, "timestamp_s": 20.0,
             "objects": [{"class": "hand", "box": [10.0, 10.0, 40.0, 40.0]}]},
        ]
        rows = reason_over_detection_records(
            records, self.sop, initial_completed=[], min_dwell_s=2.5)
        self.assertNotIn("IR03", rows[-1]["completed_steps"])

    def test_hand_holding_relation_can_claim_current_help(self):
        # Short brace near hand with no target → CURRENT IR03 via hand relation path.
        records = [
            {"index": 0, "timestamp_s": 0.0,
             "objects": [
                 {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
                 {"class": "short brace", "box": [120.0, 110.0, 200.0, 170.0]},
             ]},
            {"index": 1, "timestamp_s": 0.4,
             "objects": [
                 {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
                 {"class": "short brace", "box": [122.0, 112.0, 202.0, 172.0]},
             ]},
        ]
        rows = reason_over_detection_records(
            records, self.sop, initial_completed=["IR09", "IR15", "IR18"],
            min_dwell_s=99.0, sustained_trust_s=99)
        self.assertTrue(any(row.get("help_active") for row in rows))
        self.assertEqual(rows[-1].get("current_step"), "IR03")
        self.assertNotIn("IR03", rows[-1]["completed_steps"])

    def test_sticky_attach_then_near_hand_still_handoff_done(self):
        # Seated wheel with a nearby (not gripping) hand must seal under production
        # attach→hand-off — spatial near alone must not starve DONE forever.
        records = []
        for index, timestamp in enumerate([float(i) for i in range(0, 50)]):
            if timestamp < 20.0:
                # Move + grip + seat the wheel on the chassis (banks motion).
                x = 40.0 + timestamp * 25.0
                objects = [
                    {"class": "hand", "box": [x, 140.0, x + 60.0, 200.0]},
                    {"class": "toy car wheel", "box": [x + 10.0, 150.0, x + 70.0, 210.0]},
                    {"class": "partial model", "box": [145.0, 145.0, 400.0, 280.0]},
                ]
            else:
                # Hand near but not overlapping the seated wheel.
                objects = [
                    {"class": "hand", "box": [40.0, 40.0, 90.0, 90.0]},
                    {"class": "toy car wheel", "box": [250.0, 150.0, 310.0, 210.0]},
                    {"class": "partial model", "box": [145.0, 145.0, 400.0, 280.0]},
                ]
            records.append({"index": index, "timestamp_s": timestamp, "objects": objects})
        rows = reason_over_detection_records(
            records,
            self.sop,
            initial_completed=["IR03", "IR06", "IR09", "IR15", "IR18"],
        )
        self.assertIn("IR30", rows[-1]["completed_steps"])

    def test_ambiguous_or_generic_only_does_not_claim_a_step(self):
        result = infer_step_from_detections(["hand", "wrench"], self.rules, completed=set())
        self.assertEqual(result["step"], "unknown")
        scored = score_step_rule(self.rules[0], ["hand"])
        self.assertLess(scored["score"], 3)


if __name__ == "__main__":
    unittest.main()
