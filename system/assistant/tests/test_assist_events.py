import unittest

from assist.engine.events import build_events, events_from_segments

FPS = 10.0


def frames(*runs):
    """runs of (label, n_frames, conf) -> per-frame (t, label, conf)."""
    out, i = [], 0
    for label, n, conf in runs:
        for _ in range(n):
            out.append((i / FPS, label, conf))
            i += 1
    return out


class BuildEventsTest(unittest.TestCase):
    def test_label_must_hold_n_frames(self):
        ev = build_events(frames(("take_nut", 4, 0.9), ("background", 10, 0.9)), FPS, 5, 0.3)
        self.assertEqual(ev, [])

    def test_event_times_are_causal(self):
        ev = build_events(frames(("background", 3, 0.9), ("take_nut", 10, 0.9), ("background", 10, 0.9)),
                          FPS, 5, 0.3)
        self.assertEqual(len(ev), 1)
        e = ev[0]
        self.assertEqual(e.label, "take_nut")
        self.assertAlmostEqual(e.start, 0.3)
        self.assertAlmostEqual(e.known_at, 0.7)      # confirmed on its 5th frame
        self.assertAlmostEqual(e.end, 1.3)
        self.assertAlmostEqual(e.ended_at, 1.7)      # background held 5 frames
        self.assertGreaterEqual(e.known_at, e.start)

    def test_short_flicker_does_not_split_event(self):
        ev = build_events(frames(("fit_nut", 8, 0.9), ("tighten_nut", 2, 0.9), ("fit_nut", 8, 0.9),
                                 ("background", 6, 0.9)), FPS, 5, 0.3)
        self.assertEqual([e.label for e in ev], ["fit_nut"])

    def test_low_confidence_frames_are_ignored(self):
        ev = build_events(frames(("take_wheel", 10, 0.1)), FPS, 5, 0.3)
        self.assertEqual(ev, [])

    def test_switch_between_actions(self):
        ev = build_events(frames(("take_nut", 6, 0.8), ("fit_nut", 6, 0.6)), FPS, 5, 0.3)
        self.assertEqual([e.label for e in ev], ["take_nut", "fit_nut"])
        self.assertAlmostEqual(ev[0].end, ev[1].start)
        self.assertAlmostEqual(ev[1].conf, 0.6)

    def test_ground_truth_segments(self):
        ev = events_from_segments([{"action": "take_nut", "start": 1.0, "end": 2.0}])
        self.assertEqual((ev[0].known_at, ev[0].ended_at), (1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
