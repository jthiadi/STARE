"""v9's visual checks as the "part reads as wrong" input of the engine."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.mistakes.v9_checks import SETTINGS, link, same_steps, wrong_rows
from test_assist_engine import COMPONENTS
from test_assist_part_check import fitted_a

CHECK = {"thr": 0.8, "window_s": 10, "looks": 2}


def attached(part_a: float, n: int = 1) -> np.ndarray:
    """P(attached) for the toy task's parts after the base; part a comes first."""
    return np.array([[part_a] + [1.0] * (len(COMPONENTS) - 2)] * n, np.float32)


def installs(eng):
    return [a for a in eng.alerts if a["type"] == "incorrect_install"]


class LinkTest(unittest.TestCase):
    def test_their_folders_get_our_label_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dataset = Path(tmp) / "official", Path(tmp) / "dataset"
            (source / "03_assy_0_1").mkdir(parents=True)
            (source / "03_assy_0_1" / "PSR_labels.csv").write_text("x", encoding="utf-8")
            (source / "03_assy_0_1" / "OD_labels.json").write_text("big", encoding="utf-8")
            link(dataset, source, {"test": ["03_assy_0_1"]})
            target = dataset / "test" / "test_annotations" / "03_assy_0_1"
            self.assertEqual((target / "PSR_labels.csv").read_text(encoding="utf-8"), "x")
            self.assertFalse((target / "OD_labels.json").exists())


class WrongRowsTest(unittest.TestCase):
    def test_a_part_reading_absent_counts_as_wrong_and_the_base_never_does(self):
        rows = wrong_rows(np.array([40]), np.array([[0.2, 0.9]]), np.array([[0.4, 0.7]]), np.array([0.5]))
        time, wrong, conf = rows[0]
        self.assertEqual(time, 4.0)
        self.assertEqual(wrong[0], 0.0)
        self.assertAlmostEqual(wrong[1], 0.7, places=5)      # the two checks are averaged
        self.assertAlmostEqual(wrong[2], 0.2, places=5)
        self.assertEqual(conf, 1.0)

    def test_a_poor_view_has_no_confidence(self):
        rows = wrong_rows(np.array([40]), attached(0.1), attached(0.1), np.array([0.05]))
        self.assertEqual(rows[0][2], 0.0)

    def test_the_two_checks_must_cover_the_same_frames(self):
        with self.assertRaises(ValueError):
            wrong_rows(np.array([40, 42]), attached(0.1), attached(0.1, 2), np.array([0.5, 0.5]))


class EngineTest(unittest.TestCase):
    def feed(self, eng, frame, part_a, quality=0.5):
        for t, wrong, conf in wrong_rows(np.array([frame]), attached(part_a), attached(part_a), np.array([quality])):
            eng.on_part_wrong(t, wrong, conf)

    def test_a_part_still_absent_after_its_install_raises_the_alert(self):
        eng = fitted_a(CHECK)
        self.feed(eng, 40, 0.1)
        self.assertEqual(installs(eng), [])
        self.feed(eng, 42, 0.1)
        self.assertEqual([(a["step"], a["source"]) for a in installs(eng)], [("a", "part")])

    def test_a_part_seen_attached_raises_nothing(self):
        eng = fitted_a(CHECK)
        for frame in (40, 42, 44):
            self.feed(eng, frame, 0.95)
        self.assertEqual(installs(eng), [])

    def test_poor_views_are_not_counted(self):
        eng = fitted_a(CHECK)
        for frame in (40, 42, 44):
            self.feed(eng, frame, 0.1, quality=0.05)
        self.assertEqual(installs(eng), [])


class SameStepsTest(unittest.TestCase):
    DONE = {"step": "a", "status": "done", "t": 5.0, "source": "state"}
    NOTE = {"step": "a", "status": "verified", "t": 9.0, "source": "state"}

    def test_a_verified_note_does_not_change_the_steps(self):
        self.assertTrue(same_steps([self.DONE], [self.DONE, self.NOTE]))

    def test_a_step_finished_at_another_time_does(self):
        self.assertFalse(same_steps([self.DONE], [{**self.DONE, "t": 6.0}]))


class SettingsTest(unittest.TestCase):
    def test_the_grid_was_fixed_before_any_result(self):
        self.assertEqual(len(SETTINGS), 18)
        self.assertEqual({s["looks"] for s in SETTINGS}, {5, 10, 25})     # 1, 2 and 5 s of looks in a row


if __name__ == "__main__":
    unittest.main()
