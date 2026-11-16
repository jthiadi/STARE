"""SOP → station object inventory → detector prompts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from monitor.settings import ROOT
from monitor.sop_objects import (
    detection_plan_from_sop,
    detector_prompts_for_step,
    draft_visual_plan_from_parts,
    enrich_sop_with_visual_plans,
    slugify_entity_name,
    station_detector_classes,
    station_object_inventory,
)
from monitor.sop_vision import visual_plan_from_step
from scripts.kit_sop.run_automatic_assembly import make_plan


TEMPLATE = ROOT / "data/industreal_assembly.template.json"
SEQUENTIAL = ROOT / "data/industreal_sequential.demo.json"


class SopObjectInventoryTests(unittest.TestCase):
    def test_slugify_is_stable_and_unique(self):
        used = set()
        first = slugify_entity_name("two short white perforated braces", used=used)
        second = slugify_entity_name("two short white perforated braces", used=used)
        self.assertTrue(first.startswith("short_white_perforated_braces") or
                        first.startswith("white_perforated_braces") or
                        "brace" in first)
        self.assertNotEqual(first, second)

    def test_template_parts_draft_valid_visual_plans(self):
        sop = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        for step in sop["steps"]:
            plan = draft_visual_plan_from_parts(step)
            self.assertIn(plan["action"]["subject"],
                          {row["name"] for row in plan["entities"]})
            roles = {row["role"] for row in plan["entities"]}
            self.assertIn("moving", roles)
            self.assertIn("target", roles)
            # Must remain traceable through the official validator.
            visual_plan_from_step({**step, "visual_plan": plan})
        ir03 = draft_visual_plan_from_parts(next(step for step in sop["steps"]
                                                 if step["id"] == "IR03"))
        self.assertEqual(ir03["action"]["object"], "partial_model")
        self.assertIn("brace", ir03["action"]["subject"])
        ir06 = draft_visual_plan_from_parts(next(step for step in sop["steps"]
                                                 if step["id"] == "IR06"))
        self.assertIn("pin", ir06["action"]["subject"])
        self.assertEqual(ir06["action"]["object"], "partial_model")

    def test_template_builds_station_inventory_and_detector_prompts(self):
        sop = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        inventory = station_object_inventory(sop)
        self.assertGreaterEqual(inventory["object_count"], 6)
        self.assertEqual(inventory["step_count"], 9)
        names = {row["name"] for row in inventory["objects"]}
        self.assertTrue(any("partial" in name for name in names))
        plan = detection_plan_from_sop(sop)
        plan_names = {row["name"] for row in plan["classes"]}
        self.assertFalse(any(name in {"chassis base", "front chassis", "rear chassis"}
                             for name in plan_names))
        self.assertTrue(any("wheel" in name for name in plan_names))
        self.assertTrue(any("pulley" in name for name in plan_names))
        self.assertTrue(any("wing" in name for name in plan_names))
        ir27 = next(row for row in inventory["steps"] if row["id"] == "IR27")
        prompts = ir27["detector_prompts"]
        self.assertTrue(any(row["role"] == "moving" for row in prompts))
        self.assertTrue(any(row["role"] == "target" for row in prompts))
        self.assertTrue(any("wheel" in row["sop_evidence"].lower() for row in prompts))

    def test_reviewed_sequential_plans_are_preferred(self):
        sop = json.loads(SEQUENTIAL.read_text(encoding="utf-8"))
        inventory = station_object_inventory(sop)
        ir03 = next(row for row in inventory["steps"] if row["id"] == "IR03")
        self.assertEqual(ir03["action"]["subject"], "front_chassis_brace")
        self.assertIn("front_chassis_brace", ir03["entity_names"])
        classes = station_detector_classes(sop, include_generic_tools=False)
        self.assertTrue(any(row["source"] in {"sop_parts", "sop_object_inventory"} for row in classes))
        self.assertTrue(any(row["name"] == "hand" for row in classes))
        self.assertFalse(any(row["name"] in {"screwdriver", "wrench"} for row in classes))

    def test_detector_prompts_reject_untraceable_inventions(self):
        step = {
            "id": "X1",
            "instruction": "Install the red wheel on the white bracket.",
            "parts": ["red wheel", "white bracket"],
            "tools": [],
            "visual_plan": {
                "action": {"verb": "attach", "subject": "wheel", "object": "bracket"},
                "entities": [
                    {"name": "wheel", "role": "moving", "description": "Red wheel.",
                     "sop_evidence": ["red wheel"]},
                    {"name": "bracket", "role": "target", "description": "White bracket.",
                     "sop_evidence": ["white bracket"]},
                    {"name": "laser_cutter", "role": "tool", "description": "Invented tool.",
                     "sop_evidence": ["laser cutter"]},
                ],
                "tools": ["laser_cutter"],
                "completion_evidence": [
                    {"id": "part_at_target", "description": "Contact.",
                     "source": "sam2_temporal_relation"},
                    {"id": "correct_final_state", "description": "Finished.",
                     "source": "vlm_visual_check"},
                ],
                "required_evidence": ["part_at_target", "correct_final_state"],
            },
        }
        with self.assertRaisesRegex(ValueError, "absent"):
            detector_prompts_for_step(step)

    def test_bare_instruction_without_parts_cannot_invent_objects(self):
        with self.assertRaisesRegex(ValueError, "two parts"):
            draft_visual_plan_from_parts({
                "id": "X", "instruction": "Do the next assembly action.", "parts": [],
            })

    def test_detection_plan_is_ready_for_automatic_assembly(self):
        sop = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        plan = detection_plan_from_sop(sop, sop_path=str(TEMPLATE))
        self.assertEqual(plan["planner"], "sop_object_inventory")
        self.assertFalse(plan["ground_truth_used"])
        self.assertGreaterEqual(len(plan["classes"]), 6)
        self.assertTrue(all("prompt" in row for row in plan["classes"]))
        self.assertIn("object_inventory", plan)

    def test_make_plan_uses_sop_inventory_without_vlm(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            args = mock.Mock(sop=TEMPLATE, output=output)
            with mock.patch("monitor.general.GeneralPredictor") as predictor:
                make_plan(args)
                predictor.assert_not_called()
            plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["planner"], "sop_object_inventory")
            self.assertTrue(any(row["source"] == "sop_parts"
                                for row in plan["classes"]))

    def test_enrich_preserves_existing_reviewed_plan(self):
        sop = {
            "title": "tiny",
            "steps": [{
                "id": "S1",
                "instruction": "Attach the red wheel assembly to the white bracket.",
                "parts": ["red wheel assembly", "white bracket"],
                "tools": [],
                "completion_check": "The red wheel assembly is connected to the white bracket.",
                "visual_plan": {
                    "action": {"verb": "attach", "subject": "wheel_assembly", "object": "bracket"},
                    "entities": [
                        {"name": "wheel_assembly", "role": "moving",
                         "description": "Red wheel assembly.",
                         "sop_evidence": ["red wheel assembly"]},
                        {"name": "bracket", "role": "target",
                         "description": "White receiving bracket.",
                         "sop_evidence": ["white bracket"]},
                    ],
                    "tools": [],
                    "completion_evidence": [
                        {"id": "part_at_target", "description": "Parts remain in contact.",
                         "source": "sam2_temporal_relation"},
                        {"id": "correct_final_state", "description": "Finished.",
                         "source": "vlm_visual_check"},
                    ],
                    "required_evidence": ["part_at_target", "correct_final_state"],
                },
            }],
        }
        enriched = enrich_sop_with_visual_plans(sop)
        self.assertEqual(
            enriched["steps"][0]["visual_plan"]["action"]["subject"], "wheel_assembly")


if __name__ == "__main__":
    unittest.main()
