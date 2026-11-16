"""The combined check: the per-part model only adds "installed wrong" alerts, right after an install."""
import unittest

import numpy as np

from assist.engine.engine import Engine
from assist.engine.events import Event
from scripts.mistakes.part_check import choose, passes, wrong_rows
from test_assist_engine import COMPONENTS, CFG, toy_task

CHECK = {"thr": 0.7, "window_s": 10, "looks": 2}
A = COMPONENTS.index("a")


def wrong(component: int = A, p: float = 0.9) -> list[float]:
    return [p if k == component else 0.0 for k in range(len(COMPONENTS))]


def fitted_a(check=CHECK) -> Engine:
    """Part a was just fastened and the state shows it, at t = 3."""
    eng = Engine(toy_task(), COMPONENTS, {**CFG, "part_check": check} if check else CFG, use_state=True)
    fit, nut = Event("fit_a", 1, 2, known_at=1, ended_at=2, conf=1.0), Event("fit_nut", 2, 3, known_at=2, ended_at=3, conf=1.0)
    eng.on_action_start(1, fit)
    eng.on_action_end(2, fit)
    eng.on_action_start(2, nut)
    eng.on_action_end(3, nut)
    eng.on_state(3, [1, 1, 0, 0, 0, 0], 1.0)
    return eng


def installs(eng: Engine) -> list[dict]:
    return [a for a in eng.alerts if a["type"] == "incorrect_install"]


class PartCheckEngineTest(unittest.TestCase):
    def test_repeated_wrong_readings_after_install_raise_the_alert(self):
        eng = fitted_a()
        eng.on_part_wrong(4.0, wrong(), 1.0)
        self.assertEqual(installs(eng), [])          # one look is not enough
        eng.on_part_wrong(4.5, wrong(), 1.0)
        self.assertEqual([(a["step"], a["t"], a["source"]) for a in installs(eng)], [("a", 4.5, "part")])

    def test_a_low_threshold_is_not_mistaken_for_low_confidence(self):
        """The reading passed its own threshold; only the detection's confidence decides "cannot verify"."""
        eng = fitted_a({**CHECK, "thr": 0.1})
        eng.on_part_wrong(4.0, wrong(p=0.2), 1.0)
        eng.on_part_wrong(4.5, wrong(p=0.2), 1.0)
        self.assertEqual([a["type"] for a in eng.alerts], ["incorrect_install"])

    def test_readings_after_the_window_are_ignored(self):
        eng = fitted_a()
        eng.on_part_wrong(14.0, wrong(), 1.0)
        eng.on_part_wrong(14.5, wrong(), 1.0)
        self.assertEqual(installs(eng), [])

    def test_a_part_that_was_not_just_installed_is_ignored(self):
        eng = fitted_a()
        b = COMPONENTS.index("b")
        eng.on_part_wrong(4.0, wrong(b), 1.0)
        eng.on_part_wrong(4.5, wrong(b), 1.0)
        self.assertEqual(installs(eng), [])

    def test_a_low_reading_restarts_the_count(self):
        eng = fitted_a()
        eng.on_part_wrong(4.0, wrong(), 1.0)
        eng.on_part_wrong(4.5, wrong(p=0.1), 1.0)
        eng.on_part_wrong(5.0, wrong(), 1.0)
        self.assertEqual(installs(eng), [])

    def test_looks_with_hands_over_the_assembly_do_not_count(self):
        eng = fitted_a()
        eng.on_hands(3.5, True)
        eng.on_part_wrong(4.0, wrong(), 1.0)
        eng.on_part_wrong(4.5, wrong(), 1.0)
        self.assertEqual(installs(eng), [])

    def test_looks_with_a_weak_detection_do_not_count(self):
        eng = fitted_a()
        eng.on_part_wrong(4.0, wrong(), 0.2)
        eng.on_part_wrong(4.5, wrong(), 0.2)
        self.assertEqual(installs(eng), [])

    def test_step_status_never_changes(self):
        plain, checked = fitted_a(), fitted_a()
        checked.on_part_wrong(4.0, wrong(), 1.0)
        checked.on_part_wrong(4.5, wrong(), 1.0)
        self.assertEqual(installs(checked)[0]["step"], "a")
        self.assertEqual(plain.status, checked.status)
        self.assertEqual(plain.log, checked.log)

    def test_alert_is_not_raised_twice(self):
        eng = fitted_a()
        for t in (4.0, 4.5, 5.0, 5.5):
            eng.on_part_wrong(t, wrong(), 1.0)
        self.assertEqual(len(installs(eng)), 1)

    def test_without_the_setting_readings_do_nothing(self):
        eng = fitted_a(check=None)
        eng.on_part_wrong(4.0, wrong(), 1.0)
        eng.on_part_wrong(4.5, wrong(), 1.0)
        self.assertEqual(eng.alerts, [])

    def test_a_pending_verification_is_settled_by_the_alert(self):
        """Fastened but the state never shows it: one alert for this step, not a second "possible" one."""
        eng = Engine(toy_task(), COMPONENTS, {**CFG, "part_check": CHECK}, use_state=True)
        fit, nut = Event("fit_a", 1, 2, known_at=1, ended_at=2, conf=1.0), Event("fit_nut", 2, 3, known_at=2, ended_at=3, conf=1.0)
        eng.on_action_start(1, fit)
        eng.on_action_end(2, fit)
        eng.on_action_start(2, nut)
        eng.on_action_end(3, nut)
        eng.on_part_wrong(4.0, wrong(), 1.0)
        eng.on_part_wrong(4.5, wrong(), 1.0)
        eng.tick(30)
        self.assertEqual([a["type"] for a in eng.alerts], ["incorrect_install"])


class PassRuleTest(unittest.TestCase):
    BASE = {"install_found": 0, "false_alarms": 12, "psr_f1": 0.5, "psr_pos": 0.4, "found": 1}

    def row(self, **kw):
        return {**self.BASE, **kw}

    def test_two_finds_with_two_new_false_alarms_pass(self):
        self.assertTrue(passes(self.BASE, self.row(install_found=2, false_alarms=14)))

    def test_one_find_is_not_enough(self):
        self.assertFalse(passes(self.BASE, self.row(install_found=1, false_alarms=12)))

    def test_more_new_false_alarms_than_finds_fail(self):
        self.assertFalse(passes(self.BASE, self.row(install_found=2, false_alarms=15)))

    def test_lower_step_scores_fail(self):
        self.assertFalse(passes(self.BASE, self.row(install_found=2, psr_f1=0.49)))
        self.assertFalse(passes(self.BASE, self.row(install_found=2, psr_pos=0.39)))

    def test_choose_prefers_more_finds_then_fewer_false_alarms(self):
        rows = [self.row(install_found=2, false_alarms=13, found=3), self.row(install_found=3, false_alarms=15, found=4),
                self.row(install_found=3, false_alarms=14, found=4)]
        self.assertEqual(choose(self.BASE, rows), rows[2])

    def test_choose_returns_none_when_nothing_passes(self):
        self.assertIsNone(choose(self.BASE, [self.row()]))


class WrongRowsTest(unittest.TestCase):
    def test_keeps_time_confidence_and_the_wrong_column(self):
        rows = [{"t": 1.5, "conf": 0.8}, {"t": 2.0, "conf": 0.6}]
        p = np.zeros((2, 11, 3), np.float32)
        p[0, 4, 2] = 0.9
        out = wrong_rows(rows, p)
        self.assertEqual([(r[0], r[2]) for r in out], [(1.5, 0.8), (2.0, 0.6)])
        self.assertAlmostEqual(out[0][1][4], 0.9, places=5)
        self.assertEqual(len(out[0][1]), 11)

    def test_rows_and_predictions_must_line_up(self):
        with self.assertRaises(ValueError):
            wrong_rows([{"t": 1.5, "conf": 0.8}], np.zeros((2, 11, 3), np.float32))


if __name__ == "__main__":
    unittest.main()
