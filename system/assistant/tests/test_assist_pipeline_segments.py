"""The pipeline's models on the paper's segments: the test must see each segment's own frames, and settings come from val."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.recognition.pipeline_on_segments import guard_single_look, pick_on_val, tcn_on_segment


class TcnOnSegmentTest(unittest.TestCase):
    def setUp(self):
        self.frames = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    def test_only_the_segments_own_frames_count(self):
        np.testing.assert_allclose(tcn_on_segment(self.frames, 0, 1), [1.0, 0.0])

    def test_a_frame_after_the_segment_changes_nothing(self):
        changed = self.frames.copy()
        changed[2] = [0.0, 1.0, 0.0]
        np.testing.assert_allclose(tcn_on_segment(changed, 3, 3), tcn_on_segment(self.frames, 3, 3))

    def test_background_is_left_out_and_each_frame_still_sums_to_one(self):
        out = tcn_on_segment(np.array([[0.8, 0.1, 0.1]]), 0, 0)
        np.testing.assert_allclose(out, [0.5, 0.5])

    def test_an_end_past_the_last_frame_stops_at_the_last_frame(self):
        np.testing.assert_allclose(tcn_on_segment(self.frames, 3, 99), [0.0, 1.0])


class PickOnValTest(unittest.TestCase):
    def test_the_highest_val_score_wins(self):
        grid = [{"w": 0.2, "strength": 0.0, "smoothing": 0.1, "val_top1": 60.0},
                {"w": 0.5, "strength": 0.3, "smoothing": 0.1, "val_top1": 61.0}]
        self.assertEqual(pick_on_val(grid)["w"], 0.5)

    def test_a_tie_goes_to_the_smaller_prior_then_the_smaller_weight(self):
        grid = [{"w": 0.5, "strength": 0.3, "smoothing": 0.1, "val_top1": 60.0},
                {"w": 0.7, "strength": 0.0, "smoothing": 0.1, "val_top1": 60.0},
                {"w": 0.4, "strength": 0.0, "smoothing": 0.1, "val_top1": 60.0}]
        self.assertEqual(pick_on_val(grid)["w"], 0.4)

    def test_a_row_holding_a_test_score_is_refused(self):
        with self.assertRaises(ValueError):
            pick_on_val([{"w": 0.5, "strength": 0.0, "smoothing": 0.1, "val_top1": 60.0, "test_top1": 70.0}])


class SingleLookTest(unittest.TestCase):
    def test_a_saved_result_blocks_a_second_look_at_test(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / "result.json"
            guard_single_look(saved)
            saved.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                guard_single_look(saved)


if __name__ == "__main__":
    unittest.main()
