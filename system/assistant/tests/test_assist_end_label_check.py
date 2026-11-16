"""The action-end label check: labels arrive when the GPU is done with them, and the val rule is fixed in code."""
import unittest

from scripts.mistakes.end_label_check import SETTINGS, choose, passes, ready_rows

BASE = {"found": 4, "false_alarms": 10}


class ReadyRowsTest(unittest.TestCase):
    def test_each_label_arrives_after_its_action_ended_plus_the_queue(self):
        rows = [{"ended_at": 10.0, "label": "fit_a", "conf": 0.8}, {"ended_at": 10.5, "label": "fit_b", "conf": 0.6}]
        self.assertEqual(ready_rows(rows, 1.0), [(11.0, "fit_a", 0.8), (12.0, "fit_b", 0.6)])

    def test_rows_out_of_order_are_queued_by_end_time(self):
        rows = [{"ended_at": 20.0, "label": "x", "conf": 1.0}, {"ended_at": 5.0, "label": "y", "conf": 1.0}]
        self.assertEqual([r[1] for r in ready_rows(rows, 1.0)], ["y", "x"])


class RuleTest(unittest.TestCase):
    def test_two_new_finds_with_no_more_new_false_alarms_pass(self):
        self.assertTrue(passes(BASE, {"found": 6, "false_alarms": 12}))
        self.assertFalse(passes(BASE, {"found": 6, "false_alarms": 13}))
        self.assertFalse(passes(BASE, {"found": 5, "false_alarms": 10}))

    def test_a_quarter_fewer_false_alarms_without_losing_a_find_passes(self):
        self.assertTrue(passes(BASE, {"found": 4, "false_alarms": 7}))
        self.assertFalse(passes(BASE, {"found": 3, "false_alarms": 2}))
        self.assertFalse(passes(BASE, {"found": 4, "false_alarms": 8}))

    def test_choose_takes_most_finds_then_fewest_false_alarms(self):
        rows = [{"found": 6, "false_alarms": 12}, {"found": 6, "false_alarms": 11}, {"found": 4, "false_alarms": 3}]
        self.assertEqual(choose(BASE, rows), {"found": 6, "false_alarms": 11})

    def test_nothing_is_chosen_when_nothing_passes(self):
        self.assertIsNone(choose(BASE, [{"found": 4, "false_alarms": 10}]))


class SettingsTest(unittest.TestCase):
    def test_settings_are_distinct_and_include_the_plain_engine(self):
        keys = [tuple(sorted((k, str(v)) for k, v in s.items())) for s in SETTINGS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn(None, [s["final_label"] for s in SETTINGS])


if __name__ == "__main__":
    unittest.main()
