"""Tests for the non-model end-to-end preflight contract."""
import unittest

from monitor.readiness import EXPECTED_PREREQUISITES, OFFICIAL_STEPS


class EndToEndReadinessTests(unittest.TestCase):
    def test_official_mapping_and_partial_order_cover_same_steps(self):
        self.assertEqual(set(OFFICIAL_STEPS), set(EXPECTED_PREREQUISITES))
        self.assertEqual(EXPECTED_PREREQUISITES["IR03"], [])
        self.assertEqual(EXPECTED_PREREQUISITES["IR09"], [])
        self.assertNotIn("IR30", EXPECTED_PREREQUISITES["IR21"])


if __name__ == "__main__":
    unittest.main()
