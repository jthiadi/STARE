"""Pin-step CURRENT / fastener tie-break helpers."""
from __future__ import annotations

import unittest

from monitor.step_rules import (
    _pin_order_key,
    _prefer_pin_fastener_current,
    infer_step_from_detections,
    build_step_rules,
)


class PinCurrentTests(unittest.TestCase):
    def test_pin_order_prefers_ir15_before_ir18(self):
        self.assertLess(_pin_order_key("IR15"), _pin_order_key("IR18"))

    def test_fastener_only_picks_earliest_ready_pin(self):
        sop = {
            "steps": [
                {"id": "IR09", "instruction": "rear", "prerequisites": [],
                 "parts": ["long brace", "partial model"]},
                {"id": "IR15", "instruction": "pin a", "prerequisites": ["IR09"],
                 "parts": ["short metal pin", "hex nut", "tooth lock washer", "partial model"]},
                {"id": "IR18", "instruction": "pin b", "prerequisites": ["IR09"],
                 "parts": ["short metal pin", "hex nut", "tooth lock washer", "partial model"]},
            ]
        }
        rules = build_step_rules(sop)
        objects = [
            {"class": "hand", "box": [10, 10, 40, 40], "score": 0.9},
            {"class": "hex nut", "box": [20, 20, 35, 35], "score": 0.9},
            {"class": "partial model", "box": [100, 100, 200, 200], "score": 0.9},
        ]
        classes = [row["class"] for row in objects]
        inference = infer_step_from_detections(
            classes, rules, completed={"IR09"}, objects=objects, ready_only=True)
        self.assertEqual(inference["step"], "IR15")
        self.assertEqual(inference["status"], "step_in_progress")

    def test_prefer_pin_fastener_current_sets_help(self):
        inference = {
            "step": "unknown",
            "status": "no_step_evidence",
            "score": 0,
            "ready_steps": ["IR15", "IR18"],
            "scores": [
                {
                    "step_id": "IR15",
                    "score": 5,
                    "interaction": True,
                    "near_hand": True,
                    "fastener_hits": ["hex nut"],
                    "moving_hits": [],
                },
                {
                    "step_id": "IR18",
                    "score": 5,
                    "interaction": True,
                    "near_hand": True,
                    "fastener_hits": ["hex nut"],
                    "moving_hits": [],
                },
            ],
        }
        out = _prefer_pin_fastener_current(inference)
        self.assertEqual(out["step"], "IR15")
        self.assertTrue(out.get("pin_fastener_current"))


if __name__ == "__main__":
    unittest.main()
