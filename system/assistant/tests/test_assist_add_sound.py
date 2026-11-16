"""The audio track of an overlay video follows the same rules as the dashboard's sound."""
import unittest

import numpy as np

from assist.tools.add_sound import RATE, SPEECH_DELAY, firm, mix, plan, tone_mistake, tone_warning

WARN = {"t": 10.0, "kind": "warning", "spoken": "Wait."}
MISTAKE = {"t": 20.0, "kind": "mistake", "spoken": "Out of order.", "outcome": "caught"}
LENGTH = {"Wait.": 2.0, "Out of order.": 9.0, "Skipped.": 5.0}


def plain(alerts):
    tones, speech = plan(alerts, LENGTH)
    return tones, speech


class PlanTest(unittest.TestCase):
    def test_a_warning_gets_its_tone_and_line(self):
        tones, speech = plain([WARN])
        self.assertEqual(tones, [(10.0, "warning")])
        self.assertEqual(speech, [(10.0 + SPEECH_DELAY["warning"], "Wait.", 2.0)])

    def test_a_mistake_gets_its_tone_and_reason(self):
        tones, speech = plain([MISTAKE])
        self.assertEqual(tones, [(20.0, "mistake")])
        self.assertEqual(speech, [(20.0 + SPEECH_DELAY["mistake"], "Out of order.", 9.0)])

    def test_silent_and_duplicate_alerts_make_no_sound(self):
        alerts = [{"t": 5.0, "kind": "status", "spoken": None}, {**MISTAKE, "outcome": "duplicate"}]
        self.assertEqual(plain(alerts), ([], []))

    def test_two_mistakes_at_one_moment_play_once(self):
        tones, speech = plain([MISTAKE, {**MISTAKE, "spoken": "Skipped."}])
        self.assertEqual(len(tones), 1)
        self.assertEqual([s[1] for s in speech], ["Out of order."])

    def test_alerts_right_after_a_mistake_are_silent(self):
        tones, _ = plain([MISTAKE, {**WARN, "t": 21.5}])
        self.assertEqual(len(tones), 1)

    def test_a_new_mistake_cuts_the_old_sentence(self):
        _, speech = plain([MISTAKE, {**MISTAKE, "t": 25.0, "spoken": "Skipped."}])
        self.assertEqual(speech[0][:2], (20.0 + SPEECH_DELAY["mistake"], "Out of order."))
        self.assertAlmostEqual(speech[0][2], 25.0 - 20.0 - SPEECH_DELAY["mistake"])

    def test_a_warning_during_a_reason_is_a_tone_only(self):
        tones, speech = plain([MISTAKE, {**WARN, "t": 25.0}])
        self.assertEqual(len(tones), 2)
        self.assertEqual([s[1] for s in speech], ["Out of order."])

    def test_a_warning_after_the_reason_is_spoken(self):
        _, speech = plain([MISTAKE, {**WARN, "t": 40.0}])
        self.assertEqual([s[1] for s in speech], ["Out of order.", "Wait."])


class ToneTest(unittest.TestCase):
    @staticmethod
    def pitch(sound):
        """Main pitch of the first and last quarter."""
        n = len(sound) // 4

        def main(part):
            return np.fft.rfftfreq(len(part), 1 / RATE)[np.abs(np.fft.rfft(part)).argmax()]

        return main(sound[:n]), main(sound[-n:])

    def test_the_warning_rises_and_the_mistake_falls(self):
        first, last = self.pitch(tone_warning())
        self.assertGreater(last, first)
        first, last = self.pitch(tone_mistake())
        self.assertLess(last, first)

    def test_the_mistake_is_lower_and_louder(self):
        self.assertLess(self.pitch(tone_mistake())[0], self.pitch(tone_warning())[0])
        self.assertGreater(np.abs(tone_mistake()).max(), np.abs(tone_warning()).max())

    def test_the_tones_are_shorter_than_their_speech_delay(self):
        self.assertLessEqual(len(tone_warning()) / RATE, SPEECH_DELAY["warning"])
        self.assertLessEqual(len(tone_mistake()) / RATE, SPEECH_DELAY["mistake"])


class FirmTest(unittest.TestCase):
    @staticmethod
    def voice_like():
        """A quiet stretch then a loud stretch, with low rumble under both."""
        t = np.arange(RATE * 2) / RATE
        level = np.where(t < 1, 0.05, 0.5)
        return (level * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)

    @staticmethod
    def strength(sound, frequency):
        return np.abs(np.fft.rfft(sound))[int(frequency * len(sound) / RATE)]

    def test_quiet_parts_come_up_towards_loud_parts(self):
        t = np.arange(RATE * 2) / RATE                # no rumble here: it would hide how quiet the quiet part is
        before = (np.where(t < 1, 0.05, 0.5) * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        after = firm(before)
        gap = lambda s: np.abs(s[RATE + 100:RATE + 5000]).mean() / np.abs(s[100:5000]).mean()
        self.assertLess(gap(after), gap(before) * 0.7)

    def test_low_rumble_is_taken_out(self):
        before, after = self.voice_like(), firm(self.voice_like())
        self.assertLess(self.strength(after, 60) / self.strength(after, 1000),
                        self.strength(before, 60) / self.strength(before, 1000) / 3)

    def test_it_is_as_loud_as_it_can_be_without_clipping(self):
        peak = np.abs(firm(self.voice_like())).max()
        self.assertLessEqual(peak, 1.0)
        self.assertGreater(peak, 0.85)

    def test_silence_stays_silence(self):
        self.assertEqual(np.abs(firm(np.zeros(RATE, np.float32))).max(), 0)


class MixTest(unittest.TestCase):
    def test_sounds_land_at_their_times(self):
        one = np.ones(RATE // 10, np.float32) * 0.5
        out = mix(3.0, [(1.0, one)])
        self.assertEqual(len(out), 3 * RATE)
        self.assertEqual(out[int(0.5 * RATE)], 0)
        self.assertGreater(out[int(1.05 * RATE)], 0)

    def test_the_track_grows_to_fit_a_sound_past_the_end(self):
        self.assertEqual(len(mix(1.0, [(0.9, np.ones(RATE, np.float32) * 0.1)])), int(1.9 * RATE))

    def test_it_never_clips(self):
        loud = np.ones(RATE, np.float32)
        self.assertLessEqual(np.abs(mix(1.0, [(0.0, loud), (0.0, loud)])).max(), 1.0)


if __name__ == "__main__":
    unittest.main()
