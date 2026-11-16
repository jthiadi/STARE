"""Where a clip's alert sounds land in the cut-together reel."""
import unittest

from assist.tools.add_sound import SPEECH_DELAY
from assist.tools.reel_cards import clip_sounds

MISTAKE = {"t": 128.5, "kind": "mistake", "spoken": "Stop.", "outcome": "caught"}
WARN = {"t": 90.0, "kind": "warning", "spoken": "Wait."}
LENGTH = {"Stop.": 5.0, "Wait.": 2.0}


class ClipSoundsTest(unittest.TestCase):
    def test_an_alert_lands_at_its_place_in_the_reel(self):
        tones, speech = clip_sounds([MISTAKE], 84, 136, 10.0, 52.0, LENGTH)
        self.assertAlmostEqual(tones[0][0], 10.0 + 128.5 - 84)
        self.assertEqual(tones[0][1], "mistake")
        self.assertAlmostEqual(speech[0][0], 10.0 + 128.5 - 84 + SPEECH_DELAY["mistake"])

    def test_alerts_outside_the_clip_make_no_sound(self):
        self.assertEqual(clip_sounds([WARN], 100, 136, 0.0, 36.0, LENGTH), ([], []))
        self.assertEqual(clip_sounds([MISTAKE], 84, 120, 0.0, 36.0, LENGTH), ([], []))

    def test_speech_stops_when_the_clip_ends(self):
        _, speech = clip_sounds([MISTAKE], 120, 130, 0.0, 10.0, LENGTH)
        self.assertAlmostEqual(speech[0][2], 10.0 - (8.5 + SPEECH_DELAY["mistake"]))

    def test_a_sentence_that_would_start_after_the_cut_is_dropped(self):
        _, speech = clip_sounds([{**MISTAKE, "t": 129.9}], 120, 130, 0.0, 10.0, LENGTH)
        self.assertEqual(speech, [])

    def test_a_clip_with_no_stop_runs_to_its_held_end(self):
        tones, speech = clip_sounds([MISTAKE], 120, None, 0.0, 20.0, LENGTH)
        self.assertEqual(len(tones), 1)
        self.assertAlmostEqual(speech[0][2], 5.0)

    def test_the_input_is_not_modified(self):
        alert = dict(MISTAKE)
        clip_sounds([alert], 84, 136, 10.0, 52.0, LENGTH)
        self.assertEqual(alert["t"], 128.5)


if __name__ == "__main__":
    unittest.main()
