"""Per-part state spike: labels, crops and the probabilities -> state rule (no GPU needed)."""
import unittest

import numpy as np

from scripts.mistakes.part_state import square, state_at, to_classes, to_state


class PartStateTest(unittest.TestCase):
    def test_state_at_uses_the_last_change_or_the_initial_state(self):
        changes = [(10, [1, 0]), (20, [1, -1])]
        self.assertEqual(state_at(changes, [0, 0], 5), [0, 0])
        self.assertEqual(state_at(changes, [0, 0], 10), [1, 0])
        self.assertEqual(state_at(changes, [0, 0], 25), [1, -1])

    def test_classes_map_wrong_to_the_third_class(self):
        self.assertEqual(to_classes([0, 1, -1]), [0, 1, 2])

    def test_wrong_only_above_the_threshold(self):
        p = np.array([[0.1, 0.5, 0.4], [0.8, 0.1, 0.1]])
        self.assertEqual(to_state(p, 0.3)[0], [-1, 0])
        self.assertEqual(to_state(p, 0.5)[0], [1, 0])
        self.assertAlmostEqual(to_state(p, 0.5)[1], (0.5 + 0.8) / 2)

    def test_square_stays_inside_the_frame(self):
        for box in ([0, 0, 200, 100], [1200, 650, 80, 70], [100, 100, 1200, 700]):
            x1, y1, x2, y2 = square(box)
            self.assertEqual(x2 - x1, y2 - y1)
            self.assertTrue(0 <= x1 and x2 <= 960 and 0 <= y1 and y2 <= 540, box)


if __name__ == "__main__":
    unittest.main()
