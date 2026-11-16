"""AR-aligned fastener vocabulary for SOP detection."""
import json
import unittest

from monitor.ar_objects import (
    AR_OBJECT_PROMPTS,
    ar_detector_classes_for_sop,
    enrich_sop_parts_with_ar_objects,
    expand_hardware_phrases,
)
from monitor.settings import ROOT
from monitor.sop_objects import detection_plan_from_sop, draft_visual_plan_from_parts
from monitor.step_rules import build_step_rules, infer_step_from_detections


TEMPLATE = ROOT / "data/industreal_assembly.template.json"


class ArObjectTests(unittest.TestCase):
    def test_hardware_phrase_expands_to_nut_and_washer(self):
        parts = expand_hardware_phrases(["front chassis pin", "retaining hardware"])
        self.assertIn("front chassis pin", parts)
        self.assertTrue(any("nut" in row.lower() for row in parts))
        self.assertTrue(any("washer" in row.lower() for row in parts))

    def test_template_plan_is_lean_and_sop_derived(self):
        sop = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        plan = detection_plan_from_sop(sop)
        names = {row["name"].lower() for row in plan["classes"]}
        self.assertIn("hex nut", names)
        self.assertIn("tooth lock washer", names)
        self.assertIn("short brace", names)
        self.assertIn("partial model", names)
        self.assertIn("pulley", names)
        self.assertIn("wing", names)
        self.assertNotIn("wing beam", names)
        pulley = next(row for row in plan["classes"] if row["name"].lower() == "pulley")
        self.assertNotIn("wheel", pulley["prompt"].lower())
        self.assertIn("hand", names)
        self.assertIn("paper", names)
        self.assertNotIn("screwdriver", names)
        self.assertNotIn("wrench", names)
        self.assertNotIn("chassis base", names)
        self.assertNotIn("front chassis", names)
        self.assertNotIn("rear chassis", names)
        # No duplicate fine-grained unused SOP pin aliases.
        self.assertNotIn("front chassis pin", names)
        self.assertNotIn("front rear chassis pin", names)
        nut = next(row for row in plan["classes"] if row["name"].lower() == "hex nut")
        self.assertEqual(nut["source"], "sop_parts")
        partial = next(row for row in plan["classes"] if row["name"].lower() == "partial model")
        self.assertIn("chassis body", partial["prompt"].lower())
        wing = next(row for row in plan["classes"] if row["name"].lower() == "wing")
        self.assertIn("wing-beam", wing["prompt"].lower())

    def test_wing_beam_collapses_to_wing(self):
        from monitor.ar_objects import canonical_part_name

        self.assertEqual(canonical_part_name("wing beam"), "wing")
        classes = ar_detector_classes_for_sop({
            "steps": [{"id": "IR30", "parts": ["wing", "wing beam", "pulley"]}],
        })
        names = [row["name"] for row in classes]
        self.assertEqual(names.count("wing"), 1)
        self.assertNotIn("wing beam", names)

    def test_hand_near_nut_can_activate_pin_step(self):
        sop = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        rules = build_step_rules(sop)
        objects = [
            {"class": "hand", "box": [100.0, 100.0, 150.0, 150.0]},
            {"class": "hex nut", "box": [120.0, 110.0, 160.0, 150.0]},
        ]
        result = infer_step_from_detections(
            [row["class"] for row in objects], rules,
            completed={"IR03"}, objects=objects)
        self.assertEqual(result["step"], "IR06")
        self.assertIn(result["status"], {"step_in_progress", "no_step_evidence"})
        if result["status"] == "step_in_progress":
            self.assertEqual(result["step"], "IR06")

    def test_enrich_adds_ar_objects_without_touching_eval_labels(self):
        sop = {"steps": [{"id": "IR06", "parts": ["front chassis pin", "front chassis"],
                          "instruction": "Install pin", "prerequisites": ["IR03"]}]}
        enriched = enrich_sop_parts_with_ar_objects(sop)
        parts = " ".join(enriched["steps"][0]["parts"]).lower()
        self.assertIn("nut", parts)
        classes = ar_detector_classes_for_sop(enriched)
        self.assertTrue(any("nut" in row["name"].lower() for row in classes))


if __name__ == "__main__":
    unittest.main()
