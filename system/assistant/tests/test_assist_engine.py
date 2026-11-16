"""Engine rules on small synthetic event sequences (no dataset needed).

Toy task, components [base, a, b, p, q, c]:
  subgoal 1: a, then b (b requires a); pins p and q share the action "plug_pin", p requires a, q requires b
  subgoal 2: c
"""
import unittest

from assist.engine.engine import Engine
from assist.engine.events import Event
from assist.engine.explain import redo_plan
from assist.engine.graph import Step, Task

COMPONENTS = ["base", "a", "b", "p", "q", "c"]
CFG = {"verify_window_s": 10, "state_min_conf": 0.5, "state_stable": 1, "low_conf": 0.45,
       "alert_cooldown_s": 20, "min_frames": 5, "order_grace_s": 5,
       "stale_state_s": 30}


def toy_task(with_c: bool = True) -> Task:
    def step(sid, take, do, requires, subgoal, done_on="fasten"):
        return Step(id=sid, name=f"part {sid}", component=sid, value=1, take=take, do=do, done_on=done_on,
                    requires=requires, severity="stop", subgoal=subgoal)
    steps = {"a": step("a", ["take_a"], ["fit_a"], [], 0),
             "b": step("b", ["take_b"], ["fit_b"], ["a"], 0),
             "p": step("p", ["take_pin"], ["plug_pin"], ["a"], 0),
             "q": step("q", ["take_pin"], ["plug_pin"], ["b"], 0),
             "c": step("c", ["take_c"], ["fit_c"], [], 1)}
    subgoals = [("first", ["a", "b", "p", "q"]), ("second", ["c"])]
    if not with_c:
        steps.pop("c")
        subgoals.pop()
    return Task("toy", "Toy assembly", [1, 0, 0, 0, 0, 0], steps, subgoals, ["fit_nut"])


def run(actions, use_state=False, states=(), hands=(), end=None):
    """actions: (label, start, end[, conf]); states: (t, state, conf); hands: (t, occluded)."""
    eng = Engine(toy_task(), COMPONENTS, CFG, use_state)
    stream = []
    for a in actions:
        label, s, e = a[:3]
        conf = a[3] if len(a) > 3 else 1.0
        ev = Event(label, s, e, known_at=s, ended_at=e, conf=conf)
        stream += [(s, 0, "start", ev), (e, 1, "end", ev)]
    stream += [(t, 2, "state", (st, c)) for t, st, c in states]
    stream += [(t, 3, "hands", o) for t, o in hands]
    for t, _, kind, x in sorted(stream, key=lambda r: (r[0], r[1])):
        if kind == "start":
            eng.on_action_start(t, x)
        elif kind == "end":
            eng.on_action_end(t, x)
        elif kind == "state":
            eng.on_state(t, *x)
        else:
            eng.on_hands(t, x)
    eng.finish(end if end is not None else max(r[0] for r in stream) + 1)
    return eng


def install(label, t, conf=1.0):
    """take, fit, fasten for one part, 3 s in total starting at t."""
    return [(f"take_{label}", t, t + 1, conf), (f"fit_{label}", t + 1, t + 2, conf), ("fit_nut", t + 2, t + 3, conf)]


CORRECT = (install("a", 0) + install("b", 10)
           + [("take_pin", 20, 21), ("plug_pin", 21, 22), ("fit_nut", 22, 23)]
           + [("take_pin", 30, 31), ("plug_pin", 31, 32), ("fit_nut", 32, 33)]
           + install("c", 40))


def types(eng):
    return [a["type"] for a in eng.alerts]


class EngineRulesTest(unittest.TestCase):
    def test_correct_run_has_no_alerts(self):
        eng = run(CORRECT)
        self.assertEqual(eng.alerts, [])
        self.assertEqual(eng.progress(), (5, 5))

    def test_take_before_prerequisite_is_an_early_warning(self):
        eng = run([("take_b", 0, 1)] + CORRECT[:3] + CORRECT[3:])
        warn = eng.alerts[0]
        self.assertEqual((warn["type"], warn["step"], warn["missing"]), ("early_warning", "b", ["a"]))
        self.assertEqual(warn["t"], 0)
        self.assertNotIn("wrong_order", types(eng))     # the worker then did it right: warning only

    def test_finishing_before_prerequisite_is_wrong_order(self):
        eng = run(install("b", 0) + install("a", 10) + CORRECT[6:])
        mistake = [a for a in eng.alerts if a["type"] == "wrong_order"][0]
        self.assertEqual((mistake["step"], mistake["missing"], mistake["t"]), ("b", ["a"], 8))   # 3 s + grace
        self.assertEqual(mistake["fix_steps"], ["Remove the part b", "Install the part a", "Install the part b"])

    def test_prerequisite_finished_within_grace_counts_as_together(self):
        eng = run([("fit_b", 0, 1), ("fit_nut", 1, 2), ("fit_a", 2, 3), ("fit_nut", 3, 4)] + CORRECT[6:])
        self.assertNotIn("wrong_order", types(eng))

    def test_later_subgoal_with_missing_step_is_skipped(self):
        eng = run(install("a", 0) + CORRECT[6:12] + install("c", 40), end=60)   # b and q never done
        skipped = {a["step"]: a for a in eng.alerts if a["type"] == "skipped_step"}
        self.assertIn("b", skipped)
        self.assertEqual(skipped["b"]["trigger"], "c")
        self.assertEqual(skipped["b"]["t"], 48)

    def test_task_end_with_missing_step_is_skipped(self):
        eng = run(CORRECT[:-3])                                           # c never done
        self.assertEqual([(a["type"], a["step"]) for a in eng.alerts], [("skipped_step", "c")])

    def test_ambiguous_action_takes_the_legal_reading(self):
        # first pin after a only: p is legal, q (needs b) is not -> assigned to p, no warning
        eng = run(install("a", 0) + [("take_pin", 5, 6), ("plug_pin", 6, 7), ("fit_nut", 7, 8)], end=9)
        self.assertEqual(eng.status["p"], 3)
        self.assertEqual(eng.status["q"], 0)
        self.assertNotIn("early_warning", types(eng))

    def test_expected_state_confirms_install(self):
        eng = run(CORRECT, use_state=True, states=[(4, [1, 1, 0, 0, 0, 0], 0.9)])
        self.assertNotIn("possible_incorrect_install", types(eng))
        self.assertTrue(any(e["status"] == "verified" and e["step"] == "a" for e in eng.log))

    def test_missing_expected_state_is_possible_incorrect_install(self):
        # the detector keeps seeing the old state after part a is fastened
        states = [(t, [1, 0, 0, 0, 0, 0], 0.9) for t in (4, 6, 8)]
        eng = run(install("a", 0), use_state=True, states=states, end=20)
        alert = [a for a in eng.alerts if a["type"] == "possible_incorrect_install"][0]
        self.assertEqual((alert["step"], alert["t"]), ("a", 13))

    def test_error_state_is_incorrect_install(self):
        eng = run(install("a", 0), use_state=True, states=[(5, "error_state", 0.9)], end=20)
        alert = [a for a in eng.alerts if a["type"] == "incorrect_install"][0]
        self.assertEqual((alert["step"], alert["t"]), ("a", 5))
        self.assertEqual(alert["fix_steps"][:2], ["Remove the part a", "Install the part a"])

    def test_hands_over_assembly_is_cannot_verify(self):
        hands = [(t, True) for t in range(3, 14)]
        eng = run(install("a", 0), use_state=True, states=[(4, [1, 0, 0, 0, 0, 0], 0.9)], hands=hands, end=20)
        self.assertNotIn("possible_incorrect_install", types(eng))
        cv = [a for a in eng.alerts if a["type"] == "cannot_verify"][0]
        self.assertIn("Hands covered", cv["why"])

    def test_low_confidence_mistake_is_cannot_verify(self):
        eng = run(install("b", 0, conf=0.3) + install("a", 10) + CORRECT[6:])
        self.assertNotIn("wrong_order", types(eng))
        self.assertIn("cannot_verify", types(eng))

    def test_unused_component_in_state_is_wrong_part(self):
        eng = Engine(toy_task(with_c=False), COMPONENTS, CFG, use_state=True)
        eng.on_state(5, [1, 0, 0, 0, 0, 1], 0.9)
        self.assertEqual([(a["type"], a["component"]) for a in eng.alerts], [("wrong_part", "c")])

    def test_alerts_are_explained(self):
        eng = run([("take_b", 0, 1)] + CORRECT)
        for a in eng.alerts:
            for key in ("title", "why", "fix_steps", "rule", "evidence"):
                self.assertIn(key, a)


class HiddenStepsTest(unittest.TestCase):
    """Maintenance-like swap: a pin is taken out and put back, so the state looks the same before and after."""

    def task(self):
        def step(sid, comp, value, requires, subgoal):
            return Step(id=sid, name=sid.replace("_", " "), component=comp, value=value, take=[], do=[],
                        done_on="do", requires=requires, severity="stop", subgoal=subgoal)
        steps = {"remove_pin": step("remove_pin", "pin", 0, [], 0),
                 "remove_long": step("remove_long", "long", 0, ["remove_pin"], 0),
                 "short": step("short", "short", 1, ["remove_long"], 1),
                 "pin": step("pin", "pin", 1, ["short"], 1)}
        return Task("swap", "Swap", [1, 1, 1, 0], steps,
                    [("strip", ["remove_pin", "remove_long"]), ("rebuild", ["short", "pin"])], [])

    def test_unseen_pin_removal_and_reinsertion_are_inferred(self):
        eng = Engine(self.task(), ["base", "pin", "long", "short"], CFG, use_state=True)
        eng.on_state(10, [1, 1, 0, 1], 0.9)        # long brace gone, short one on, pin (back) in
        eng.finish(60)
        self.assertEqual(eng.alerts, [])
        self.assertEqual(eng.progress(), (4, 4))

    def test_pin_left_out_is_still_skipped(self):
        eng = Engine(self.task(), ["base", "pin", "long", "short"], CFG, use_state=True)
        eng.on_state(10, [1, 0, 0, 1], 0.9)        # pin never put back
        eng.on_state(50, [1, 0, 0, 1], 0.9)
        eng.finish(60)
        self.assertEqual([(a["type"], a["step"]) for a in eng.alerts], [("skipped_step", "pin")])


class FixPlanTest(unittest.TestCase):
    def test_dependents_come_off_first(self):
        task = toy_task()
        status = {"a": 0, "b": 3, "p": 0, "q": 3, "c": 0}   # b and q are on, a is missing
        self.assertEqual(redo_plan(task, status, [], ["a"]),
                         ["Remove the part q", "Remove the part b", "Install the part a",
                          "Install the part b", "Install the part q"])


if __name__ == "__main__":
    unittest.main()
