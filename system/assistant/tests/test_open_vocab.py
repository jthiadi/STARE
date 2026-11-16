"""Model-free tests for SOP-driven YOLO-World proposal policy."""
import unittest

from monitor.open_vocab import RelocalizationGate, validate_part_prompts


STEP = {
    "id": "IR27",
    "instruction": "Attach the front wheel and magenta wing assembly to the front bracket.",
    "parts": ["two black front wheels", "grey axle", "magenta wing", "white front bracket"],
}


class PromptPolicyTests(unittest.TestCase):
    def test_prompts_are_traceable_to_active_sop(self):
        rows = validate_part_prompts([
            {"name": "front wheels", "prompt": "black toy wheel", "aliases": ["wheel"],
             "role": "moving", "count": 2, "sop_evidence": "two black front wheels"},
            {"name": "front bracket", "prompt": "white perforated toy bracket",
             "role": "target", "count": 1, "sop_evidence": "white front bracket"},
        ], STEP)
        self.assertEqual(rows[0]["name"], "front_wheels")
        self.assertEqual(rows[0]["count"], 2)

    def test_unrelated_detector_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not traceable"):
            validate_part_prompts([
                {"name": "screwdriver", "prompt": "screwdriver", "role": "moving",
                 "sop_evidence": "screwdriver"},
                {"name": "front bracket", "prompt": "bracket", "role": "target",
                 "sop_evidence": "white front bracket"},
            ], STEP)

    def test_relocalization_requires_persistent_failure_and_cooldown(self):
        gate = RelocalizationGate(invalid_samples_before_retry=2,
                                  minimum_seconds_between_retries=3)
        self.assertFalse(gate.observe(False, 1))
        self.assertTrue(gate.observe(False, 2))
        self.assertFalse(gate.observe(False, 3))
        self.assertFalse(gate.observe(False, 4))
        self.assertTrue(gate.observe(False, 5))
        self.assertFalse(gate.observe(True, 6))


if __name__ == "__main__":
    unittest.main()
