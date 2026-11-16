"""Turning the v9 PSR system's saved step predictions into inputs our scoring and engine can read."""
import unittest

from scripts.mistakes.compare_v9 import step_entries, state_rows

PROC = [{"id": 0, "description": "Install base", "state_idx": 0, "install": True},
        {"id": 3, "description": "Install front chassis", "state_idx": 1, "install": True},
        {"id": 4, "description": "Incorrectly installed front chassis", "state_idx": 1, "install": True},
        {"id": 5, "description": "Remove front chassis", "state_idx": 1, "install": False}]
BOXES = [{"t": 0.5, "box": [1, 2, 3, 4]}, {"t": 2.0, "box": [5, 6, 7, 8]}]


class StepEntriesTest(unittest.TestCase):
    def test_a_description_becomes_its_official_step(self):
        entries = step_entries([[120, "Install front chassis"]], PROC)
        self.assertEqual((entries[0]["frame"], entries[0]["id"]), (120, 3))

    def test_an_unknown_description_is_an_error(self):
        with self.assertRaises(KeyError):
            step_entries([[1, "Polish the car"]], PROC)


class StateRowsTest(unittest.TestCase):
    def rows(self, steps, frames=30):
        return state_rows(steps, [1, 0], PROC, frames, BOXES, fps=10, every=5)

    def test_one_reading_every_half_second(self):
        self.assertEqual([r["f"] for r in self.rows([], 15)], [0, 5, 10])
        self.assertEqual([r["t"] for r in self.rows([], 15)], [0.0, 0.5, 1.0])

    def test_a_step_shows_from_its_own_frame_on_and_never_before(self):
        states = {r["f"]: r["state"] for r in self.rows([[10, "Install front chassis"]])}
        self.assertEqual(states[5], [1, 0])
        self.assertEqual(states[10], [1, 1])
        self.assertEqual(states[25], [1, 1])

    def test_a_removal_takes_the_part_off_again(self):
        states = {r["f"]: r["state"] for r in self.rows([[5, "Install front chassis"], [15, "Remove front chassis"]])}
        self.assertEqual(states[10], [1, 1])
        self.assertEqual(states[15], [1, 0])

    def test_the_box_is_the_last_one_seen_so_far(self):
        boxes = {r["f"]: r["box"] for r in self.rows([])}
        self.assertIsNone(boxes[0])
        self.assertEqual(boxes[5], [1, 2, 3, 4])
        self.assertEqual(boxes[25], [5, 6, 7, 8])

    def test_every_reading_counts_as_a_confident_view(self):
        self.assertTrue(all(r["conf"] == 1.0 for r in self.rows([])))


if __name__ == "__main__":
    unittest.main()
