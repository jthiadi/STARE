"""The action-end label as extra evidence in the engine: it may only add alerts, never move a step, and is off by default.

Same toy task as ``test_assist_engine``: a, then b (b requires a), pins p and q, then c in a later subgoal.
"""
import unittest

from assist.engine.engine import Engine
from test_assist_engine import CFG, COMPONENTS, install, run, toy_task

RULE = {"install": True, "skip_grace_s": None, "min_conf": 0.5}
OLD_STATE = [1, 0, 0, 0, 0, 0]


def engine(rule=RULE, use_state=True):
    cfg = dict(CFG) if rule is None else {**CFG, "final_label": rule}
    return Engine(toy_task(), COMPONENTS, cfg, use_state)


def types(eng):
    return [a["type"] for a in eng.alerts]


class OffByDefaultTest(unittest.TestCase):
    def test_without_the_setting_a_final_label_does_nothing(self):
        eng = engine(rule=None)
        for t in (1, 3, 5, 7):
            eng.on_state(t, OLD_STATE, 0.9)
        eng.on_final_label(3, "fit_a", 0.9)
        eng.finish(20)
        self.assertNotIn("possible_incorrect_install", types(eng))


class InstallTest(unittest.TestCase):
    def test_an_install_the_live_labels_missed_is_checked_against_the_state(self):
        eng = engine()
        eng.on_final_label(3, "fit_a", 0.9)
        for t in (4, 6, 8, 12):
            eng.on_state(t, OLD_STATE, 0.9)
        eng.finish(20)
        alert = [a for a in eng.alerts if a["type"] == "possible_incorrect_install"][0]
        self.assertEqual((alert["step"], alert["t"], alert["source"]), ("a", 13, "final_label"))

    def test_the_expected_state_clears_the_check(self):
        eng = engine()
        eng.on_final_label(3, "fit_a", 0.9)
        for t in (4, 6):
            eng.on_state(t, [1, 1, 0, 0, 0, 0], 0.9)
        eng.finish(20)
        self.assertNotIn("possible_incorrect_install", types(eng))

    def test_a_label_below_the_confidence_is_ignored(self):
        eng = engine()
        eng.on_final_label(3, "fit_a", 0.4)
        for t in (4, 6, 8, 12):
            eng.on_state(t, OLD_STATE, 0.9)
        eng.finish(20)
        self.assertNotIn("possible_incorrect_install", types(eng))

    def test_an_install_whose_prerequisites_are_not_done_is_not_checked(self):
        eng = engine()
        eng.on_final_label(3, "fit_b", 0.9)          # b needs a, which the state has not shown
        for t in (4, 6, 8, 12):
            eng.on_state(t, OLD_STATE, 0.9)
        eng.finish(20)
        self.assertNotIn("possible_incorrect_install", [a["type"] for a in eng.alerts if a["step"] == "b"])

    def test_fastening_is_not_an_install(self):
        eng = engine()
        eng.on_final_label(3, "fit_nut", 0.9)
        for t in (4, 6, 8, 12):
            eng.on_state(t, OLD_STATE, 0.9)
        eng.finish(20)
        self.assertNotIn("possible_incorrect_install", types(eng))

    def test_without_a_state_source_nothing_is_checked(self):
        eng = engine(use_state=False)
        eng.on_final_label(3, "fit_a", 0.9)
        eng.finish(20)
        self.assertNotIn("possible_incorrect_install", types(eng))


class SkipTest(unittest.TestCase):
    SKIP = {"install": False, "skip_grace_s": 5, "min_conf": 0.5}

    def test_a_later_subgoal_action_with_an_earlier_step_not_done_is_a_skip(self):
        eng = engine(self.SKIP, use_state=False)
        eng.on_final_label(10, "fit_c", 0.9)
        eng.tick(16)
        skipped = sorted(a["step"] for a in eng.alerts if a["type"] == "skipped_step")
        self.assertEqual(skipped, ["a", "b", "p", "q"])
        self.assertTrue(all(a["t"] == 15 and a["source"] == "final_label" for a in eng.alerts
                            if a["type"] == "skipped_step"))

    def test_a_step_done_within_the_grace_is_not_skipped(self):
        eng = engine(self.SKIP, use_state=True)
        eng.on_state(1, [1, 1, 1, 1, 1, 0], 0.9)       # a, b, p, q all shown done
        eng.on_final_label(10, "fit_c", 0.9)
        eng.tick(16)
        self.assertNotIn("skipped_step", types(eng))

    def test_off_when_no_grace_is_set(self):
        eng = engine({"install": False, "skip_grace_s": None, "min_conf": 0.5}, use_state=False)
        eng.on_final_label(10, "fit_c", 0.9)
        eng.tick(16)
        self.assertNotIn("skipped_step", types(eng))


class NeverMovesAStepTest(unittest.TestCase):
    def test_the_step_log_is_the_same_with_and_without_final_labels(self):
        actions = install("a", 0) + install("b", 10) + install("c", 30)
        states = [(t, [1, 1, 0, 0, 0, 0], 0.9) for t in range(4, 40, 2)]
        plain = run(actions, use_state=True, states=states, end=50)
        eng = engine({"install": True, "skip_grace_s": 5, "min_conf": 0.0})
        stream = [(t, 2, "state", s) for t, s, _ in states]
        stream += [(e, 5, "final", label) for label, _, e, _ in actions]
        from assist.engine.events import Event
        for label, s, e, conf in actions:
            ev = Event(label, s, e, known_at=s, ended_at=e, conf=conf)
            stream += [(s, 0, "start", ev), (e, 1, "end", ev)]
        for t, _, kind, x in sorted(stream, key=lambda r: (r[0], r[1])):
            if kind == "start":
                eng.on_action_start(t, x)
            elif kind == "end":
                eng.on_action_end(t, x)
            elif kind == "state":
                eng.on_state(t, x, 0.9)
            else:
                eng.on_final_label(t, x, 0.9)
        eng.finish(50)
        steps = lambda log: [e for e in log if e["status"] != "verified"]      # noqa: E731
        self.assertEqual(steps(plain.log), steps(eng.log))
        self.assertEqual(plain.status, eng.status)


if __name__ == "__main__":
    unittest.main()
