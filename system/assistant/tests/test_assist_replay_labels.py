"""The replay with final labels: each action shows its final label over its whole span, and the video says so."""
import inspect
import unittest

import numpy as np

from assist.tools.end_label_view import final_events
from assist.tools.overlay_cards import label_column, render
from assist.tools.six_cases import REPLAY_FOOTER, REPLAY_CASES, clip_name


class FinalEventsTest(unittest.TestCase):
    def test_each_action_shows_its_final_label_from_its_start_to_its_end(self):
        rows = [{"start": 2.0, "end": 4.0, "ended_at": 4.4, "label": "take_nut", "conf": 0.9}]
        self.assertEqual(final_events(rows), [{"label": "take_nut", "start": 2.0, "end": 4.0, "known_at": 2.0,
                                               "ended_at": 4.0, "conf": 0.9}])


class LabelColumnTest(unittest.TestCase):
    def test_the_row_is_renamed_when_it_shows_final_labels(self):
        live, final = (np.asarray(label_column("next", False, row_name=name)) for name in (None, "Final labels"))
        self.assertFalse((live == final).all())
        self.assertTrue((np.asarray(label_column("next", False)) == live).all())


class ReplayOptionTest(unittest.TestCase):
    def test_the_overlay_only_replays_final_labels_when_asked(self):
        self.assertIs(inspect.signature(render).parameters["final_actions"].default, False)

    def test_every_replay_card_says_the_labels_were_filled_in_afterwards(self):
        self.assertIn("filled in after each action ends", REPLAY_FOOTER)
        self.assertEqual(len(REPLAY_CASES), 6)

    def test_the_replay_clips_never_overwrite_the_existing_ones(self):
        self.assertEqual(clip_name("six_cases_replay", REPLAY_CASES[0]), "six_cases_replay_1_skipped_step.mp4")


if __name__ == "__main__":
    unittest.main()
