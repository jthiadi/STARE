import unittest

from assist.evaluation.baseline import state_changes


class StateChangesTest(unittest.TestCase):
    def test_steps_become_states_one_per_frame(self):
        steps = [{"frame": 10, "id": 3}, {"frame": 10, "id": 6}, {"frame": 20, "id": 4}, {"frame": 30, "id": 5}]
        self.assertEqual(state_changes(steps, [1, 0, 0]), [
            (0, [1, 0, 0]),
            (10, [1, 1, 1]),      # two installs on the same frame: one change
            (20, [1, -1, 1]),     # 3k + 1: installed incorrectly
            (30, [1, 0, 1]),      # 3k + 2: removed
        ])

    def test_no_steps_keeps_the_initial_state(self):
        self.assertEqual(state_changes([], [1, 1]), [(0, [1, 1])])


if __name__ == "__main__":
    unittest.main()
