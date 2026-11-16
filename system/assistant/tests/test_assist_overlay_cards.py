"""The card overlay: which card shows, what it says, and which checklist rows fade."""
import unittest

import numpy as np
from PIL import Image

from assist.tools.overlay_cards import LABEL_W, card_text, ring_segments, row_style, shadow_mask, with_shadows
from assist.tools.render_overlay import H, PX, VH, W
from test_assist_engine import toy_task

TASK = toy_task()


def alert(kind, atype, step="b", **extra):
    titles = {"early_warning": "Wait: part b is not next", "wrong_order": "Out of order: part b",
              "skipped_step": "Skipped step: part b", "incorrect_install": "Installed incorrectly: part b",
              "possible_incorrect_install": "Check the part b", "wrong_part": "Wrong part: short brace"}
    return {"kind": kind, "type": atype, "step": step, "t": 10.0, "title": titles[atype], **extra}


class CardTextTest(unittest.TestCase):
    def test_with_no_alert_the_card_says_what_is_next(self):
        c = card_text(None, TASK, ["a"], False, 0)
        self.assertEqual((c["state"], c["heading"], c["title"]), ("next", "NEXT", "Install the part a"))

    def test_other_allowed_steps_are_the_subtext(self):
        c = card_text(None, TASK, ["a", "c"], False, 0)
        self.assertEqual(c["sub"], "Also allowed: part c")

    def test_when_everything_is_done_the_card_says_so(self):
        c = card_text(None, TASK, [], False, 0)
        self.assertEqual((c["state"], c["heading"]), ("next", "ALL STEPS DONE"))

    def test_an_early_warning_is_amber_and_says_what_comes_first(self):
        c = card_text(alert("warning", "early_warning", missing=["a"]), TASK, ["a"], False, 0)
        self.assertEqual((c["state"], c["heading"], c["title"]), ("warn", "WARNING", "Wait: Part b not next"))
        self.assertEqual(c["sub"], "First do: part a")

    def test_the_subtext_leaves_out_bracketed_asides_so_it_fits(self):
        task = toy_task()
        task.steps["a"].name = "rear chassis (long brace)"
        c = card_text(alert("warning", "early_warning", missing=["a"]), task, ["a"], False, 0)
        self.assertEqual(c["sub"], "First do: rear chassis")

    def test_a_wrong_order_mistake_is_red_with_the_step_as_its_title(self):
        c = card_text(alert("mistake", "wrong_order", missing=["a"]), TASK, ["a"], False, 0)
        self.assertEqual((c["state"], c["heading"], c["title"]), ("mistake", "MISTAKE", "Part b"))
        self.assertEqual(c["sub"], "Out of order. First do: part a")

    def test_red_does_not_depend_on_the_steps_severity(self):
        c = card_text({**alert("mistake", "skipped_step"), "severity": "warn"}, TASK, ["a"], False, 0)
        self.assertEqual((c["state"], c["sub"]), ("mistake", "Step skipped. Do it now."))

    def test_an_uncertain_install_is_amber_and_asks_for_a_check(self):
        c = card_text(alert("mistake", "possible_incorrect_install"), TASK, ["a"], False, 0)
        self.assertEqual((c["state"], c["heading"], c["title"]), ("warn", "CHECK", "Part b"))

    def test_a_wrong_part_names_the_part(self):
        c = card_text(alert("mistake", "wrong_part", step=None), TASK, ["a"], False, 0)
        self.assertEqual((c["state"], c["title"], c["sub"]), ("mistake", "Short brace", "Wrong part. Remove it."))

    def test_the_end_card_counts_the_mistakes(self):
        c = card_text(None, TASK, [], True, 5)
        self.assertEqual((c["state"], c["heading"], c["title"]), ("mistake", "TASK ENDED", "5 mistakes flagged"))
        self.assertEqual(card_text(None, TASK, [], True, 1)["title"], "1 mistake flagged")
        self.assertEqual(card_text(None, TASK, [], True, 0)["state"], "next")


class RowStyleTest(unittest.TestCase):
    def test_a_step_done_correctly_fades(self):
        self.assertEqual(row_style("a", {"a": "done"}, set(), ["b"]), "faded")

    def test_a_done_step_with_a_mistake_stays_visible(self):
        self.assertEqual(row_style("a", {"a": "done"}, {"a"}, ["b"]), "flagged")

    def test_a_flagged_step_that_is_not_done_stays_visible(self):
        self.assertEqual(row_style("a", {}, {"a"}, ["b"]), "flagged")

    def test_the_next_step_stands_out(self):
        self.assertEqual(row_style("b", {"a": "done"}, set(), ["b"]), "next")

    def test_a_step_in_progress_stands_out(self):
        self.assertEqual(row_style("b", {"b": "placed"}, set(), ["a"]), "active")

    def test_a_later_step_stays_plain(self):
        self.assertEqual(row_style("c", {}, set(), ["a"]), "pending")


class ShadowTest(unittest.TestCase):
    def test_a_shadow_is_darkest_at_the_edge_that_casts_it_and_fades_out(self):
        mask = np.asarray(shadow_mask(20, 4, 90, dark_on_right=True))
        self.assertEqual(mask.shape, (4, 20))
        self.assertEqual(mask[0, -1], 90)
        self.assertEqual(mask[0, 0], 0)
        self.assertTrue((np.diff(mask[0].astype(int)) >= 0).all())

    def test_it_can_fall_the_other_way(self):
        mask = np.asarray(shadow_mask(20, 4, 90, dark_on_right=False))
        self.assertEqual(mask[0, 0], 90)
        self.assertEqual(mask[0, -1], 0)

    def test_shadows_only_darken_and_leave_everything_else_alone(self):
        flat = Image.new("RGB", (W, H), (200, 200, 200))
        out = np.asarray(with_shadows(flat))
        self.assertEqual(tuple(out[H // 2, PX + 5]), (200, 200, 200))     # the panel itself
        self.assertLess(out[H // 2, PX - 1, 0], 200)                      # just left of the panel
        self.assertEqual(tuple(out[H // 2, PX - 60]), (200, 200, 200))    # further away
        self.assertLess(out[VH + 50, LABEL_W + 1, 0], 200)                # just right of the label column
        self.assertEqual(tuple(out[VH // 2, LABEL_W + 1]), (200, 200, 200))   # that one is only beside the timeline


class RingTest(unittest.TestCase):
    def test_done_segments_come_first_from_the_top_with_one_per_step(self):
        self.assertEqual(ring_segments(7, 10), [True] * 7 + [False] * 3)

    def test_it_uses_the_tasks_real_number_of_steps(self):
        self.assertEqual(len(ring_segments(0, 8)), 8)


if __name__ == "__main__":
    unittest.main()
