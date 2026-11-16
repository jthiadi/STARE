"""The final label on screen: it may only appear once the pass over the action has finished."""
import inspect
import unittest

import numpy as np
from PIL import Image

from assist.tools.end_label_view import ACTION_PASS_S, chip_text, draw_chip, draw_strip, shown_final, with_times
from assist.tools.overlay_cards import label_column, render


def rows(*ended):
    return [{"start": t - 2, "end": t - 0.4, "ended_at": t, "label": "take_nut", "conf": 0.9} for t in ended]


class TimesTest(unittest.TestCase):
    def test_the_pass_time_is_the_one_measured_on_a_free_gpu(self):
        self.assertEqual(ACTION_PASS_S, 1.21)

    def test_a_label_is_ready_one_pass_after_the_action_ended(self):
        self.assertEqual(with_times(rows(5.0), 1.0)[0]["label_at"], 6.0)

    def test_a_label_waits_while_the_previous_pass_is_still_running(self):
        self.assertEqual(with_times(rows(5.0, 5.5), 1.0)[1]["label_at"], 7.0)


class ShownFinalTest(unittest.TestCase):
    def setUp(self):
        self.track = with_times(rows(5.0, 20.0), 1.0)

    def test_nothing_shows_before_the_label_is_ready(self):
        self.assertIsNone(shown_final(self.track, 5.9, 4.0))

    def test_it_shows_once_ready(self):
        self.assertEqual(shown_final(self.track, 6.0, 4.0)["ended_at"], 5.0)

    def test_it_goes_away_after_the_hold(self):
        self.assertIsNone(shown_final(self.track, 10.0, 4.0))

    def test_a_newer_label_replaces_an_older_one(self):
        track = with_times(rows(5.0, 6.5), 1.0)
        self.assertEqual(shown_final(track, 7.6, 4.0)["ended_at"], 6.5)


class StripTest(unittest.TestCase):
    def picture(self, t):
        image = Image.new("RGB", (100, 80), (255, 255, 255))
        draw_strip(image, with_times([{"start": 10.0, "end": 20.0, "ended_at": 20.0, "label": "take_nut"}], 1.0), t, 100.0, 0, 100)
        return np.asarray(image)

    def test_an_action_stays_off_the_strip_until_its_label_is_ready(self):
        self.assertTrue((self.picture(20.9) == 255).all())

    def test_it_appears_on_the_strip_once_ready(self):
        self.assertFalse((self.picture(21.0) == 255).all())


class ChipTest(unittest.TestCase):
    def test_the_chip_names_what_it_is(self):
        self.assertEqual(chip_text("take_nut"), "Final label: Taking the nut")

    def test_the_chip_draws_in_the_video_corner(self):
        image = Image.new("RGB", (1440, 810), (255, 255, 255))
        draw_chip(image, "take_nut")
        pixels = np.asarray(image)
        self.assertFalse((pixels[760:800, 20:300] == 255).all())
        self.assertTrue((pixels[:600] == 255).all())


class OptionTest(unittest.TestCase):
    def test_the_overlay_has_no_final_label_unless_asked(self):
        self.assertIs(inspect.signature(render).parameters["action_end"].default, False)

    def test_the_timeline_names_the_strip_only_when_it_is_shown(self):
        plain, named = (np.asarray(label_column("next", False, final)) for final in (False, True))
        self.assertFalse((plain == named).all())
        self.assertTrue((np.asarray(label_column("next", False)) == plain).all())


if __name__ == "__main__":
    unittest.main()
