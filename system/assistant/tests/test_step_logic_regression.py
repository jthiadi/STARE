"""Regression tests for step-claim correctness (overlay must match SOP intent)."""
import json
import unittest

from monitor.settings import ROOT
from monitor.sop_objects import draft_visual_plan_from_parts
from monitor.step_rules import (
    build_step_rules,
    infer_step_from_detections,
    reason_over_detection_records,
)


TEMPLATE = ROOT / "data/industreal_assembly.template.json"


class StepLogicRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sop = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        cls.rules = build_step_rules(cls.sop)

    def test_nut_and_hand_does_not_claim_wheel_steps(self):
        chassis = {"IR03", "IR06", "IR09", "IR15", "IR18"}
        objects = [
            {"class": "hand", "box": [100.0, 100.0, 150.0, 150.0]},
            {"class": "hex nut", "box": [120.0, 110.0, 160.0, 150.0]},
        ]
        result = infer_step_from_detections(
            [row["class"] for row in objects], self.rules,
            completed=chassis, objects=objects)
        self.assertNotIn(result["step"], {"IR27", "IR30", "IR21"})

    def test_toy_car_wheel_alone_does_not_prefer_front_over_rear(self):
        completed = {"IR03", "IR06", "IR09", "IR15", "IR18", "IR21", "IR24"}
        objects = [
            {"class": "hand", "box": [100.0, 100.0, 150.0, 150.0]},
            {"class": "toy car wheel", "box": [120.0, 110.0, 200.0, 190.0]},
        ]
        result = infer_step_from_detections(
            [row["class"] for row in objects], self.rules,
            completed=completed, objects=objects)
        # Ambiguous AR wheel synonym must not invent IR27 vs IR30.
        self.assertEqual(result["step"], "unknown")

    def test_front_and_rear_wheel_assemblies_disambiguate(self):
        completed = {"IR03", "IR06", "IR09", "IR15", "IR18", "IR21", "IR24"}
        front = [
            {"class": "hand", "box": [100.0, 100.0, 150.0, 150.0]},
            {"class": "front wheel assembly", "box": [120.0, 110.0, 220.0, 200.0]},
        ]
        rear = [
            {"class": "hand", "box": [100.0, 100.0, 150.0, 150.0]},
            {"class": "rear wheel assembly", "box": [120.0, 110.0, 220.0, 200.0]},
        ]
        self.assertEqual(
            infer_step_from_detections(
                [r["class"] for r in front], self.rules, completed=completed, objects=front)["step"],
            "IR27")
        self.assertEqual(
            infer_step_from_detections(
                [r["class"] for r in rear], self.rules, completed=completed, objects=rear)["step"],
            "IR30")

    def test_early_bracket_trust_does_not_poison_chassis_memory(self):
        records = []
        for index, timestamp in enumerate([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]):
            records.append({
                "index": index,
                "timestamp_s": timestamp,
                "objects": [
                    {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0]},
                    {"class": "white front bracket", "box": [140.0, 120.0, 220.0, 200.0]},
                ],
            })
        rows = reason_over_detection_records(
            records, self.sop, initial_completed=[], sustained_trust_s=8.0, min_dwell_s=3.0)
        # Top-down: IR21 is not ready at cold start, so never claim it or complete chassis.
        for row in rows:
            self.assertNotIn("IR03", row["completed_steps"])
            self.assertTrue(
                row["status"] == "no_step_evidence"
                or row["step"] == "unknown"
                or row["step"] in {"IR03", "IR09"})

    def test_ir27_target_is_front_bracket_not_main_chassis(self):
        ir27 = next(step for step in self.sop["steps"] if step["id"] == "IR27")
        plan = draft_visual_plan_from_parts(ir27)
        self.assertEqual(plan["action"]["object"], "white_front_bracket")

    def test_ir30_target_is_partial_model_not_legacy_chassis(self):
        ir30 = next(step for step in self.sop["steps"] if step["id"] == "IR30")
        plan = draft_visual_plan_from_parts(ir30)
        self.assertEqual(plan["action"]["object"], "partial_model")

    def test_fastener_not_duplicated_as_near_target_in_evidence(self):
        objects = [
            {"class": "hand", "box": [100.0, 100.0, 150.0, 150.0]},
            {"class": "hex nut", "box": [120.0, 110.0, 160.0, 150.0]},
        ]
        result = infer_step_from_detections(
            [row["class"] for row in objects], self.rules,
            completed={"IR03"}, objects=objects)
        self.assertEqual(result["step"], "IR06")
        self.assertNotIn("near hex nut", result["evidence"].lower())

    def test_wheel_substring_does_not_activate_opposite_step(self):
        completed = {"IR03", "IR06", "IR09", "IR15", "IR18", "IR21", "IR24"}
        objects = [
            {"class": "hand", "box": [100.0, 100.0, 150.0, 150.0]},
            {"class": "rear wheel assembly", "box": [120.0, 110.0, 220.0, 200.0]},
        ]
        result = infer_step_from_detections(
            [row["class"] for row in objects], self.rules,
            completed=completed, objects=objects)
        self.assertEqual(result["step"], "IR30")
        ir27 = next(row for row in result["scores"] if row["step_id"] == "IR27")
        self.assertFalse(ir27.get("interaction"))


if __name__ == "__main__":
    unittest.main()
