"""Paper Table 3: MECCANO action recognition (62.45, val 66.12) recomputed on a CPU from the saved model outputs, with the
VideoMAE run that was chosen on val. The script must not score test with any other choice."""
import importlib.util
import json
import unittest
from pathlib import Path

STARE = Path(__file__).resolve().parents[3]
SCRIPT = STARE / "metrics/ar/score_meccano_final_labels.py"
RESULT = STARE / "metrics/ar/results/meccano_final_labels.json"
OUT = STARE / "system/assistant/output/meccano_action_end"


def load():
    spec = importlib.util.spec_from_file_location("meccano_score", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ChoiceTest(unittest.TestCase):
    def test_the_videomae_run_comes_from_the_saved_val_choice(self):
        self.assertEqual(load().chosen_vmae({"videomae_run": "off_s3", "val": {}}), "off_s3")


@unittest.skipUnless((OUT / "vmae_probs.npz").exists() and (STARE / "data/MECCANO").exists()
                     and importlib.util.find_spec("torch") is not None,
                     "needs the saved MECCANO outputs, the MECCANO annotations and PyTorch")
class ResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load().main()
        cls.r = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_the_final_label_gives_the_paper_numbers(self):
        self.assertEqual((self.r["val"], self.r["test"]), (66.12, 62.45))

    def test_the_test_segments_are_all_scored(self):
        self.assertEqual(self.r["test_segments"], 2823)


if __name__ == "__main__":
    unittest.main()
