"""When the dashboard makes a sound (the logic in static/sound.js, run with Node)."""
import json
import shutil
import subprocess
import unittest

from assist.common import ROOT

SOUND = (ROOT / "assist/dashboard/static/sound.js").as_posix()
NODE = shutil.which("node")
WARN = {"t": 10.0, "kind": "warning", "spoken": "Wait."}
MISTAKE = {"t": 10.5, "kind": "mistake", "spoken": "Out of order.", "outcome": "caught"}
SECOND = {"t": 10.5, "kind": "mistake", "spoken": "Skipped.", "outcome": "caught"}
DUPLICATE = {"t": 11.0, "kind": "mistake", "spoken": "Installed wrong.", "outcome": "duplicate"}
SILENT = {"t": 10.2, "kind": "status", "spoken": None}


def call(function: str, *args):
    code = (f"const s = require({json.dumps(SOUND)});"
            f"console.log(JSON.stringify(s.{function}(...{json.dumps(args)}) ?? null));")
    return json.loads(subprocess.run([NODE, "-e", code], capture_output=True, text=True, check=True).stdout)


@unittest.skipUnless(NODE, "Node is not installed")
class DueAlertsTest(unittest.TestCase):
    def test_an_alert_sounds_when_playback_reaches_it(self):
        self.assertEqual(call("dueAlerts", [WARN], 9.9, 10.1), [WARN])

    def test_an_alert_does_not_sound_again_on_the_next_frame(self):
        self.assertEqual(call("dueAlerts", [WARN], 10.1, 10.2), [])

    def test_dragging_forward_past_an_alert_is_silent(self):
        self.assertEqual(call("dueAlerts", [WARN], 2.0, 50.0), [])

    def test_dragging_back_is_silent(self):
        self.assertEqual(call("dueAlerts", [WARN], 50.0, 2.0), [])

    def test_silent_and_duplicate_alerts_never_sound(self):
        self.assertEqual(call("dueAlerts", [SILENT, DUPLICATE], 9.9, 11.5), [])


@unittest.skipUnless(NODE, "Node is not installed")
class ChooseTest(unittest.TestCase):
    def test_a_mistake_goes_before_a_warning(self):
        self.assertEqual(call("choose", [WARN, MISTAKE], 10.5, -100), MISTAKE)

    def test_two_mistakes_at_one_moment_play_once(self):
        self.assertEqual(call("choose", [MISTAKE, SECOND], 10.5, -100), MISTAKE)

    def test_right_after_a_mistake_nothing_else_plays(self):
        self.assertIsNone(call("choose", [SECOND], 11.0, 10.5))

    def test_later_alerts_play_again(self):
        self.assertEqual(call("choose", [SECOND], 13.0, 10.5), SECOND)

    def test_after_rewinding_past_a_mistake_earlier_alerts_play(self):
        self.assertEqual(call("choose", [WARN], 10.0, 128.5), WARN)

    def test_nothing_due_plays_nothing(self):
        self.assertIsNone(call("choose", [], 13.0, -100))


@unittest.skipUnless(NODE, "Node is not installed")
class VoicesTest(unittest.TestCase):
    VOICES = [{"name": "Hanhan", "lang": "zh-TW"}, {"name": "Zira", "lang": "en-US"}, {"name": "George", "lang": "en_GB"},
              {"name": "David", "lang": "en-US"}]

    def test_only_english_voices_are_offered_in_name_order(self):
        self.assertEqual([v["name"] for v in call("englishVoices", self.VOICES)], ["David", "George", "Zira"])

    def test_the_saved_voice_is_used(self):
        self.assertEqual(call("chooseVoice", self.VOICES, "Zira")["name"], "Zira")

    def test_with_nothing_saved_the_firm_voice_is_used(self):
        self.assertEqual(call("chooseVoice", self.VOICES, None)["name"], "David")

    def test_a_voice_that_is_gone_falls_back_to_the_firm_voice(self):
        self.assertEqual(call("chooseVoice", self.VOICES, "Nobody")["name"], "David")

    def test_browser_default_stays_available_when_there_is_no_firm_voice(self):
        self.assertIsNone(call("chooseVoice", self.VOICES[:3], None))


if __name__ == "__main__":
    unittest.main()
