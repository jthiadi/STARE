import unittest

import numpy as np

from assist.engine.events import Event
from assist.extract.online_vjepa import frame_scores, shown_probs, updates

FPS = 10.0
K = 74


def ev(start, end, known_at, ended_at, label="take_nut"):
    return Event(label, start, end, known_at=known_at, ended_at=ended_at)


def live_probs(n, seed=0):
    p = np.random.default_rng(seed).random((n, K + 1))
    return p / p.sum(1, keepdims=True)


def members(n, seed=1):
    p = np.random.default_rng(seed).random((n, K))
    return p / p.sum(1, keepdims=True)


FLAT = (np.ones(K), np.ones((K, K)))         # uniform order counts


class UpdatesTest(unittest.TestCase):
    def test_updates_never_pass_the_event_end(self):
        upd = updates([ev(0.3, 2.0, 0.7, 2.5)], FPS, 40)
        self.assertEqual([u[3] for u in upd], [7, 12, 17, 22, 25])      # every 0.5 s from known_at, then ended_at
        self.assertTrue(all(start == 3 and end <= at for _, start, end, at in upd))
        self.assertEqual(upd[-1], (0, 3, 19, 25))                       # final window = the action's own frames

    def test_short_event_still_gets_its_final_update(self):
        upd = updates([ev(0.3, 0.8, 0.7, 0.9)], FPS, 40)
        self.assertEqual(upd, [(0, 3, 7, 7), (0, 3, 7, 9)])

    def test_event_running_at_video_end_is_clipped(self):
        upd = updates([ev(3.0, 4.2, 3.4, 4.2)], FPS, 40)
        self.assertTrue(all(end <= at <= 39 for _, _, end, at in upd))
        self.assertEqual(upd[-1][3], 39)


class ShownProbsTest(unittest.TestCase):
    def setUp(self):
        self.events = [ev(0.3, 2.0, 0.7, 2.5), ev(3.0, 4.0, 3.4, 4.5)]
        self.n = 60
        self.live = live_probs(self.n)
        self.upd = updates(self.events, FPS, self.n)
        self.member = members(len(self.upd))

    def shown(self, member=None, strength=0.5, w=0.7):
        return shown_probs(self.live, self.events, self.upd, self.member if member is None else member, w,
                           strength, 0.1, *FLAT, delay=5)

    def test_frames_before_first_result_show_live(self):
        s = self.shown()
        changed = np.where((s != self.live).any(1))[0]
        self.assertEqual(changed.min(), 7 + 5)                           # first update at 7, shown 0.5 s later
        np.testing.assert_allclose(s[:12], self.live[:12])

    def test_no_frame_uses_an_update_before_it_is_ready(self):
        base = self.shown()
        for j, (_, _, _, at) in enumerate(self.upd):
            changed = self.member.copy()
            changed[j] = members(1, seed=100 + j)[0]
            np.testing.assert_allclose(self.shown(member=changed)[:at + 5], base[:at + 5])

    def test_frames_between_events_show_live(self):
        np.testing.assert_allclose(self.shown()[25:39], self.live[25:39])

    def test_strength_zero_mixes_each_frame_with_the_latest_result(self):
        s = self.shown(strength=0.0, w=0.7)
        _, _, _, at = self.upd[0]
        f = at + 5
        a = 0.3 * np.log(self.live[f, 1:]) + 0.7 * np.log(self.member[0] + 1e-8)
        a = np.exp(a - a.max())
        a /= a.sum()
        self.assertAlmostEqual(s[f, 0], self.live[f, 0])                 # background kept from the live model
        np.testing.assert_allclose(s[f, 1:], (1 - self.live[f, 0]) * a, rtol=1e-6)

    def test_weight_zero_is_the_live_model(self):
        np.testing.assert_allclose(self.shown(w=0.0), self.live, rtol=1e-6)

    def test_later_events_do_not_change_earlier_labels(self):
        changed = self.member.copy()
        second = [j for j, u in enumerate(self.upd) if u[0] == 1]
        changed[second] = members(len(second), seed=9)
        a, b = self.shown(), self.shown(member=changed)
        np.testing.assert_allclose(a[:30], b[:30])
        self.assertFalse(np.allclose(a[30:], b[30:]))


class FrameScoresTest(unittest.TestCase):
    def test_scores(self):
        gt = np.array([0, 0, 5, 5, 5, 0, 7, 7])
        shown = np.zeros((8, K + 1))
        for f, a in enumerate([0, 3, 5, 5, 5, 5, 7, 3]):
            shown[f, a] = 1.0
        r = frame_scores(shown, gt, [(2, 4, 5), (6, 7, 7)])
        self.assertAlmostEqual(r["frame_acc"], 100 * 5 / 8)
        self.assertAlmostEqual(r["action_frame_acc"], 100 * 4 / 5)
        self.assertAlmostEqual(r["end_acc"], 50.0)                       # right at frame 4, wrong at frame 7


class VariantTest(unittest.TestCase):
    def test_a_fresh_worker_reads_the_chosen_variants_folder(self):
        """Spawned workers re-import the module, so they start on the default folder until told otherwise."""
        from unittest.mock import patch

        import assist.extract.run_ar_vjepa as m

        seen = {}
        default = m.VARIANTS["release"]
        self.addCleanup(m._use_variant, "release")
        self.assertEqual(m.OUT, default[0])
        with patch.object(m, "_load_split", side_effect=lambda split: seen.update(out=m.OUT)), \
                patch.object(m, "transition_counts"), patch.object(m, "load_mapping"):
            m._init("val", "prefix")
        self.assertEqual(seen["out"], m.VARIANTS["prefix"][0])


class ValRuleTest(unittest.TestCase):
    LIVE = {"frame_acc": 44.0, "end_acc": 38.0}

    @staticmethod
    def cand(weight, frame_acc, end_acc, recall=7.7, fa=25.6, f1=4.9, warned=5, early=40.0):
        return {"weight": weight, "frame_acc": frame_acc, "end_acc": end_acc, "mistake_recall": recall,
                "false_alarms_per_h": fa, "mistake_f1": f1, "warned": warned, "early_warnings_per_h": early}

    def rule(self, cands):
        from assist.extract.run_ar_vjepa import passing_setting
        return passing_setting(cands, self.LIVE, self.cand(0.0, 44.0, 38.0))

    def test_better_end_labels_alone_do_not_pass(self):
        self.assertIsNone(self.rule([self.cand(0.8, 40.0, 43.0)]))

    def test_it_must_beat_the_live_model_not_equal_it(self):
        self.assertIsNone(self.rule([self.cand(0.0, 44.0, 38.0)]))

    def test_fewer_mistakes_found_does_not_pass(self):
        self.assertIsNone(self.rule([self.cand(0.1, 45.0, 39.0, recall=0.0)]))

    def test_more_false_alarms_do_not_pass(self):
        self.assertIsNone(self.rule([self.cand(0.1, 45.0, 39.0, fa=26.0)]))

    def test_ties_go_to_the_smaller_weight(self):
        best = self.rule([self.cand(0.8, 40.0, 43.0), self.cand(0.2, 44.5, 41.0), self.cand(0.1, 45.0, 40.0)])
        self.assertEqual(best["weight"], 0.1)


class EngineInputTest(unittest.TestCase):
    def test_pred_vjepa_reads_ar_vjepa_jsonl(self):
        import tempfile
        from pathlib import Path

        from assist.common import write_jsonl
        from assist.run_engine import action_events

        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"f": f, "t": f / FPS, "top": [["take_nut" if 3 <= f < 15 else "background", 0.9]]}
                    for f in range(30)]
            write_jsonl(Path(tmp) / "v" / "ar_vjepa.jsonl", rows)
            cfg = {"fps": FPS, "paths": {"out": Path(tmp)},
                   "events": {"min_frames": 5, "min_conf": 0.3, "ignore": ["background"]}}
            ev = action_events("v", "pred_vjepa", cfg)
        self.assertEqual([e.label for e in ev], ["take_nut"])


if __name__ == "__main__":
    unittest.main()
