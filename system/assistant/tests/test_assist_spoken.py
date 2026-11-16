"""What the dashboard says aloud for an alert: short, firm commands that carry over factory noise."""
import unittest

from assist.dashboard.data import spoken_text


def alert(kind, title="", why=""):
    return {"type": kind, "kind": "mistake", "title": title, "why": why}


class SpokenTextTest(unittest.TestCase):
    def test_wrong_order_says_stop_and_what_to_do_first(self):
        text = spoken_text(alert("wrong_order", "Out of order: front pin of the rear chassis",
                                 "The front pin of the rear chassis was done before short rear chassis (short brace). "
                                 "The pins go through it."))
        self.assertEqual(text, "Stop. Wrong order. Front pin of the rear chassis. First do: short rear chassis.")

    def test_wrong_order_names_every_missing_step(self):
        text = spoken_text(alert("wrong_order", "Out of order: rear wheel assembly",
                                 "The rear wheel assembly was done before remove the rear chassis (long brace) and short "
                                 "rear chassis (short brace). More."))
        self.assertEqual(text, "Stop. Wrong order. Rear wheel assembly. First do: remove the rear chassis and short rear chassis.")

    def test_a_skipped_step_is_a_command(self):
        text = spoken_text(alert("skipped_step", "Skipped step: remove the rear chassis (long brace)", "Long reason."))
        self.assertEqual(text, "Stop. Step skipped. Remove the rear chassis. Do it now.")

    def test_a_wrong_install_is_a_command(self):
        text = spoken_text(alert("incorrect_install", "Installed incorrectly: front bracket", "Long reason."))
        self.assertEqual(text, "Stop. Installed wrong. Front bracket. Fix it now.")

    def test_an_uncertain_install_asks_for_a_check_without_saying_stop(self):
        text = spoken_text(alert("possible_incorrect_install", "Check the front bracket", "Long reason."))
        self.assertEqual(text, "Check now. Front bracket. Not confirmed.")

    def test_a_wrong_part_is_a_command(self):
        text = spoken_text(alert("wrong_part", "Wrong part: short rear chassis", "Long reason."))
        self.assertEqual(text, "Stop. Wrong part. Short rear chassis. Remove it.")

    def test_an_early_warning_is_one_short_line(self):
        text = spoken_text({"type": "early_warning", "kind": "warning", "title": "Wait: front pin (left) is not next",
                            "why": "You picked up the part."})
        self.assertEqual(text, "Wait. The front pin is not next.")

    def test_an_early_warning_on_a_removal_reads_correctly(self):
        text = spoken_text({"type": "early_warning", "kind": "warning", "title": "Wait: remove the rear chassis is not next",
                            "why": "x"})
        self.assertEqual(text, "Wait. Removing the rear chassis is not next.")

    def test_cannot_verify_says_nothing(self):
        self.assertIsNone(spoken_text({"type": "cannot_verify", "kind": "status", "title": "Cannot verify", "why": "x"}))

    def test_brackets_inside_brackets_are_removed_whole(self):
        text = spoken_text(alert("skipped_step", "Skipped step: front pin (a (b) c)", "Long reason."))
        self.assertEqual(text, "Stop. Step skipped. Front pin. Do it now.")


if __name__ == "__main__":
    unittest.main()
