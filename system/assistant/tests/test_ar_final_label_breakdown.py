"""What each part of the IndustReal final label adds (paper Table Y): the script must rebuild the deployed recipe step by step,
end exactly at the locked val 73.81 / test 75.07, and report only test for rows that use the TCN (the deployed TCN saw the
val videos, so its val numbers are inflated)."""
import importlib.util
import json
import unittest
from pathlib import Path

from assist.extract import action_end

STARE = Path(__file__).resolve().parents[3]
SCRIPT = STARE / "metrics/ar/score_final_label_breakdown.py"
RESULT = STARE / "metrics/ar/results/final_label_breakdown.json"


def load():
    spec = importlib.util.spec_from_file_location("breakdown", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StepsTest(unittest.TestCase):
    def test_the_steps_end_at_the_deployed_recipe(self):
        mod = load()
        name, w, strength = mod.STEPS[-1]
        self.assertEqual(name, "+ order prior (final label)")
        self.assertEqual((w, strength), (action_end.WEIGHT, action_end.STRENGTH))

    def test_the_steps_add_one_part_at_a_time(self):
        self.assertEqual([s[0] for s in load().STEPS],
                         ["TCN alone", "+ TCN (w = 0.95)", "+ order prior (final label)"])


@unittest.skipUnless((STARE / "system/assistant/output/assist/03_main_0_1/live_probs.npz").exists(), "needs the saved TCN outputs")
class ResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load().main()
        cls.r = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_the_full_recipe_gives_the_locked_numbers(self):
        self.assertEqual(self.r["val_full_recipe"], 73.81)
        self.assertEqual(self.r["test"]["+ order prior (final label)"], 75.07)

    def test_the_rows_match_the_paper_table(self):
        t = self.r["test"]
        self.assertEqual(t["TCN alone"], 55.74)
        self.assertEqual(t["VideoMAE alone"], 62.18)
        self.assertEqual(t["Mean of 3 V-JEPA 2 runs"], 72.76)
        self.assertEqual(t["+ VideoMAE (mean of 4)"], 73.25)
        self.assertEqual(t["+ TCN (w = 0.95)"], 73.74)

    def test_no_val_numbers_that_use_the_tcn(self):
        self.assertNotIn("val", self.r)


if __name__ == "__main__":
    unittest.main()
