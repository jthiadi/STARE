"""Which recordings count as "with errors" in the IndustReal PSR protocol."""
import unittest

from assist.common import load_config
from assist.evaluation.psr_eval import contains_errors, determine_performance, load_gt, PUBLISHED_PSR
from assist.industreal import SPLITS, psr_states, task_of

PROC = [{"id": 1, "expected_in_assy": True, "expected_in_main": False},
        {"id": 2, "expected_in_assy": True, "expected_in_main": True},
        {"id": 3, "expected_in_assy": False, "expected_in_main": True}]
GOOD = [(10, [1, 0]), (20, [1, 1])]


def steps(*ids):
    return [{"id": i} for i in ids]


class ContainsErrorsTest(unittest.TestCase):
    def test_all_steps_done_and_no_wrong_part_has_no_errors(self):
        self.assertFalse(contains_errors(steps(1, 2), PROC, "assy", GOOD))

    def test_a_missing_expected_step_is_an_error(self):
        self.assertTrue(contains_errors(steps(1), PROC, "assy", GOOD))

    def test_a_step_only_expected_in_the_other_task_is_not_missing(self):
        self.assertFalse(contains_errors(steps(2, 3), PROC, "main", GOOD))

    def test_a_wrongly_installed_part_is_an_error_even_if_fixed_later(self):
        self.assertTrue(contains_errors(steps(1, 2), PROC, "assy", [(10, [1, -1]), (20, [1, 1])]))

    def test_doing_all_steps_in_the_wrong_order_is_not_an_error(self):
        self.assertFalse(contains_errors(steps(2, 1), PROC, "assy", GOOD))


class PublishedPsrTest(unittest.TestCase):
    def test_the_wacv_row_is_the_table_printed_in_the_paper(self):
        wacv = next(r for r in PUBLISHED_PSR if "WACV" in r["name"])
        self.assertEqual((wacv["f1"], wacv["pos"], wacv["delay_s"]), (0.883, 0.797, 22.4))
        self.assertEqual(wacv["with_errors"], {"f1": 0.816, "pos": 0.731, "delay_s": 20.4})

    def test_the_figures_from_the_storm_paper_are_named_as_such(self):
        row = next(r for r in PUBLISHED_PSR if r["f1"] == 0.891)
        self.assertIn("as reported in STORM-PSR", row["name"])
        self.assertNotIn("WACV", row["name"])


class OfficialScoringTest(unittest.TestCase):
    """Worked through by hand from IndustReal's own determine_performance, which must be matched exactly, quirks included."""
    PROC = [{"id": 1}, {"id": 2}]

    @staticmethod
    def entries(*pairs):
        return [{"frame": f, "id": i, "conf": 1} for f, i in pairs]

    def score(self, gt, pred):
        return determine_performance(self.entries(*gt), self.entries(*pred), self.PROC)

    def test_a_correct_late_prediction_is_a_true_positive_with_its_delay(self):
        m = self.score([(100, 1)], [(130, 1)])
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["avg_delay"]), (1, 0, 0, 30))

    def test_with_two_predictions_the_later_one_is_matched_not_the_nearest(self):
        m = self.score([(100, 1)], [(90, 1), (120, 1)])
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["avg_delay"]), (1, 1, 0, 20))

    def test_an_early_single_prediction_is_paired_with_the_first_entry(self):
        """A quirk of the official code: no later prediction exists, so its search falls back to entry 0."""
        m = self.score([(50, 1), (100, 2)], [(150, 1), (90, 2)])
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["avg_delay"]), (2, 0, 0, 75))

    def test_no_delay_at_all_counts_as_100_frames(self):
        self.assertEqual(self.score([(100, 1)], [])["avg_delay"], 100)


class RealRecordingsTest(unittest.TestCase):
    def test_35_of_84_recordings_have_errors_and_15_of_them_are_in_test(self):
        """The paper says 35 videos (42%) hold a missing or incorrectly completed step."""
        import json
        from pathlib import Path
        cfg = load_config()
        labels = cfg["paths"]["labels"]
        if not Path(labels).is_dir():
            self.skipTest("IndustReal labels not available")
        proc = json.loads(Path(cfg["paths"]["procedure_info"]).read_text(encoding="utf-8"))
        counts = {split: sum(contains_errors(load_gt(rid, labels), proc, task_of(rid), psr_states(rid, labels))
                             for rid in SPLITS[split]) for split in SPLITS}
        self.assertEqual(sum(counts.values()), 35)
        self.assertEqual(counts["test"], 15)


if __name__ == "__main__":
    unittest.main()
