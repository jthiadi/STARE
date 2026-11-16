"""Ready / current SOP part scoping for SAM3 queries (no successor lookahead)."""
import json
import unittest

from monitor.settings import ROOT
from monitor.sop_detect_scope import (
    parts_for_steps,
    ready_step_ids,
    scoped_plan_classes,
    scoped_step_ids,
)
from monitor.sop_objects import detection_plan_from_sop


TEMPLATE = ROOT / "data/industreal_assembly.template.json"


class SopDetectScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sop = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        cls.plan = detection_plan_from_sop(cls.sop)

    def test_cold_start_ready_is_parallel_brace_steps(self):
        ready = ready_step_ids(self.sop, set())
        self.assertEqual(ready, ["IR03", "IR09"])

    def test_default_scope_is_ready_only_no_successors(self):
        scope = scoped_step_ids(self.sop, completed=set(), current_step=None)
        self.assertEqual(scope["ready"], ["IR03", "IR09"])
        self.assertEqual(scope["search"], ["IR03", "IR09"])
        self.assertEqual(scope["next"], [])
        # Pin / wheel successors stay out until their steps become ready.
        self.assertNotIn("IR06", scope["search"])
        self.assertNotIn("IR15", scope["search"])
        self.assertNotIn("IR18", scope["search"])
        self.assertNotIn("IR27", scope["search"])
        self.assertNotIn("IR30", scope["search"])

    def test_cold_start_parts_are_braces_and_model_only(self):
        scope = scoped_step_ids(self.sop, completed=set())
        parts = {name.casefold() for name in parts_for_steps(self.sop, scope["search"])}
        self.assertEqual(parts, {"short brace", "long brace", "partial model"})

    def test_after_ir09_opens_rear_pin_steps(self):
        completed = {"IR09"}
        scope = scoped_step_ids(self.sop, completed=completed)
        self.assertEqual(scope["ready"], ["IR03", "IR15", "IR18"])
        self.assertEqual(scope["search"], ["IR03", "IR15", "IR18"])
        parts = {name.casefold() for name in parts_for_steps(self.sop, scope["search"])}
        self.assertIn("short brace", parts)
        self.assertIn("short metal pin", parts)
        self.assertIn("hex nut", parts)
        self.assertNotIn("long brace", parts)  # IR09 done → not searching its parts
        self.assertNotIn("toy car wheel", parts)

    def test_after_chassis_branch_ready_are_ir21_and_ir30(self):
        completed = {"IR03", "IR06", "IR09", "IR15", "IR18"}
        scope = scoped_step_ids(self.sop, completed=completed, current_step="IR21")
        self.assertEqual(scope["current"], ["IR21"])
        self.assertEqual(set(scope["ready"]), {"IR21", "IR30"})
        self.assertNotIn("IR24", scope["search"])  # not ready until IR21 completes
        parts = {name.casefold() for name in parts_for_steps(self.sop, scope["search"])}
        self.assertIn("white front bracket", parts)
        self.assertIn("toy car wheel", parts)
        self.assertIn("pulley", parts)

    def test_optional_lookahead_still_available(self):
        scope = scoped_step_ids(
            self.sop, completed=set(), include_next=True)
        self.assertIn("IR06", scope["search"])
        self.assertIn("IR15", scope["search"])

    def test_scoped_plan_keeps_hand_and_paper(self):
        classes, meta = scoped_plan_classes(
            self.sop, self.plan["classes"], completed=set())
        names = {row["name"].casefold() for row in classes}
        self.assertIn("hand", names)
        self.assertIn("paper", names)
        self.assertIn("short brace", names)
        self.assertIn("long brace", names)
        self.assertNotIn("short metal pin", names)
        self.assertNotIn("hex nut", names)
        self.assertNotIn("toy car wheel", names)
        self.assertEqual(meta["ready"], ["IR03", "IR09"])
        self.assertFalse(meta["include_next"])


if __name__ == "__main__":
    unittest.main()
