"""MECCANO action-end label (62.45, past only): IndustReal recipe unchanged, the one choice made on val, test scored once,
and the data read from STARE's own data/MECCANO."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assist.extract import action_end
from meccano_action_end import action_end as mec
from meccano_action_end import data


class RecipeTest(unittest.TestCase):
    def test_the_members_are_the_industreal_recipe_with_the_meccano_videomae(self):
        self.assertEqual(set(action_end.MEMBERS) - {"vmae_full"}, set(mec.VJ_MEMBERS))

    def test_the_settings_are_the_industreal_ones_unchanged(self):
        self.assertEqual((mec.WEIGHT, mec.STRENGTH, mec.SMOOTHING),
                         (action_end.WEIGHT, action_end.STRENGTH, action_end.SMOOTHING))

    def test_the_tcn_is_the_past_only_one(self):
        self.assertIn("causal", mec.TCN_RUN)
        self.assertIn("causal", mec.FEATS_CAUSAL.name)


class PickVideoMAETest(unittest.TestCase):
    def test_the_videomae_run_with_the_best_val_score_is_used(self):
        self.assertEqual(mec.pick_vmae({"off_s2": 61.0, "off_s3": 63.5}), "off_s3")

    def test_a_tie_goes_to_the_first_run(self):
        self.assertEqual(mec.pick_vmae({"off_s3": 60.0, "off_s2": 60.0}), "off_s2")


class SingleLookTest(unittest.TestCase):
    def test_test_needs_the_val_result_first(self):
        folder = Path(tempfile.mkdtemp())
        with self.assertRaises(FileNotFoundError):
            mec.check_can_score_test(folder / "val.json", folder / "test.json")

    def test_a_second_look_at_test_is_refused(self):
        folder = Path(tempfile.mkdtemp())
        (folder / "val.json").write_text("{}")
        (folder / "test.json").write_text("{}")
        with self.assertRaises(FileExistsError):
            mec.check_can_score_test(folder / "val.json", folder / "test.json")


class DataInStareTest(unittest.TestCase):
    @unittest.skipIf("MECCANO_ROOT" in os.environ, "MECCANO_ROOT overrides the default")
    def test_the_dataset_is_stare_data_meccano(self):
        stare = Path(__file__).resolve().parents[3]
        self.assertEqual(data.DATA.resolve(), (stare / "data" / "MECCANO").resolve())

    def test_outputs_are_in_output_meccano_action_end(self):
        self.assertEqual(data.OUT.name, "meccano_action_end")


if __name__ == "__main__":
    unittest.main()
