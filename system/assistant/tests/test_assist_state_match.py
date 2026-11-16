"""The state-match check: after an install, does the view look like the state the procedure expects?"""
import unittest

import numpy as np

from assist.engine.engine import Engine
from scripts.mistakes.state_match import SETTINGS, crop_box, similarities, thresholds
from test_assist_engine import COMPONENTS, CFG, toy_task
from test_assist_part_check import fitted_a

EXPECTED = "110000"                       # toy task: base and part a on
CHECK = {"thr": {EXPECTED: 0.8}, "window_s": 10, "looks": 2}


def fitted(check=CHECK):
    return fitted_a(None if check is None else check)


def installs(eng):
    return [a for a in eng.alerts if a["type"] == "incorrect_install"]


class EngineMatchTest(unittest.TestCase):
    def test_views_unlike_the_expected_state_raise_the_alert(self):
        eng = fitted()
        eng.on_state_match(4.0, {EXPECTED: 0.5}, 1.0)
        self.assertEqual(installs(eng), [])
        eng.on_state_match(4.5, {EXPECTED: 0.5}, 1.0)
        self.assertEqual([(a["step"], a["t"], a["source"]) for a in installs(eng)], [("a", 4.5, "match")])

    def test_views_like_the_expected_state_are_fine(self):
        eng = fitted()
        for t in (4.0, 4.5, 5.0):
            eng.on_state_match(t, {EXPECTED: 0.95}, 1.0)
        self.assertEqual(installs(eng), [])

    def test_a_good_view_restarts_the_count(self):
        eng = fitted()
        eng.on_state_match(4.0, {EXPECTED: 0.5}, 1.0)
        eng.on_state_match(4.5, {EXPECTED: 0.95}, 1.0)
        eng.on_state_match(5.0, {EXPECTED: 0.5}, 1.0)
        self.assertEqual(installs(eng), [])

    def test_an_expected_state_with_no_reference_is_skipped(self):
        eng = fitted({**CHECK, "thr": {"999999": 0.8}})
        eng.on_state_match(4.0, {"999999": 0.1}, 1.0)
        eng.on_state_match(4.5, {"999999": 0.1}, 1.0)
        self.assertEqual(installs(eng), [])

    def test_looks_after_the_window_are_ignored(self):
        eng = fitted()
        eng.on_state_match(14.0, {EXPECTED: 0.5}, 1.0)
        eng.on_state_match(14.5, {EXPECTED: 0.5}, 1.0)
        self.assertEqual(installs(eng), [])

    def test_hands_over_the_assembly_do_not_count(self):
        eng = fitted()
        eng.on_hands(3.5, True)
        eng.on_state_match(4.0, {EXPECTED: 0.5}, 1.0)
        eng.on_state_match(4.5, {EXPECTED: 0.5}, 1.0)
        self.assertEqual(installs(eng), [])

    def test_step_status_never_changes(self):
        plain, checked = fitted(), fitted()
        checked.on_state_match(4.0, {EXPECTED: 0.5}, 1.0)
        checked.on_state_match(4.5, {EXPECTED: 0.5}, 1.0)
        self.assertEqual(installs(checked)[0]["step"], "a")
        self.assertEqual(plain.status, checked.status)
        self.assertEqual(plain.log, checked.log)

    def test_without_the_setting_views_do_nothing(self):
        eng = fitted(None)
        eng.on_state_match(4.0, {EXPECTED: 0.1}, 1.0)
        eng.on_state_match(4.5, {EXPECTED: 0.1}, 1.0)
        self.assertEqual(eng.alerts, [])


class SimilaritiesTest(unittest.TestCase):
    def test_each_row_gets_its_cosine_to_every_reference(self):
        rows = np.array([[1.0, 0.0], [0.0, 1.0]])
        refs = np.array([[1.0, 0.0], [0.6, 0.8]])
        self.assertTrue(np.allclose(similarities(rows, refs), [[1.0, 0.6], [0.0, 0.8]]))


class ThresholdsTest(unittest.TestCase):
    def test_at_most_this_share_of_correct_views_falls_below(self):
        sims = np.linspace(0, 1, 1001)
        thr = thresholds({"110": sims}, 0.01)["110"]
        self.assertLessEqual((sims < thr).mean(), 0.01)
        self.assertGreater(thr, 0.0)

    def test_a_state_with_too_few_correct_views_gets_no_threshold(self):
        self.assertEqual(thresholds({"110": np.ones(5)}, 0.01), {})


class CropTest(unittest.TestCase):
    def test_a_box_in_1280_pixels_becomes_cache_pixels(self):
        self.assertEqual(crop_box([100, 200, 400, 80]), (75, 150, 375, 210))

    def test_a_box_is_kept_inside_the_frame(self):
        self.assertEqual(crop_box([1200, 700, 400, 80]), (900, 525, 960, 540))


class SettingsTest(unittest.TestCase):
    def test_the_grid_is_small_because_val_has_only_5_wrong_installs(self):
        self.assertLessEqual(len(SETTINGS), 12)


if __name__ == "__main__":
    unittest.main()
