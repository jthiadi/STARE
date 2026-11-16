import unittest

import numpy as np

from scripts.mistakes.state_anomaly import flag


def rows(n):
    return [{"t": i / 2, "state": "11100000000", "conf": 0.9, "box": [0, 0, 10, 10]} for i in range(n)]


class FlagTest(unittest.TestCase):
    def test_only_runs_of_k_anomalous_visible_rows_become_error_state(self):
        score = np.array([3.0, 3.0, 0.5, 3.0, 3.0, 3.0, np.nan, 3.0])
        out = flag(rows(8), score, np.zeros(8, bool), thr=2.0, k=3)
        self.assertEqual([r["state"] == "error_state" for r in out],
                         [False, False, False, True, True, True, False, False])

    def test_rows_with_hands_over_the_assembly_are_never_flagged(self):
        covered = np.array([False, True, False, False])
        out = flag(rows(4), np.full(4, 3.0), covered, thr=2.0, k=2)
        self.assertEqual([r["state"] == "error_state" for r in out], [False, False, True, True])

    def test_input_rows_are_not_changed(self):
        r = rows(3)
        flag(r, np.full(3, 3.0), np.zeros(3, bool), thr=2.0, k=2)
        self.assertTrue(all(x["state"] == "11100000000" for x in r))


if __name__ == "__main__":
    unittest.main()
