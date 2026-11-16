"""State-detector rows written by ``assist.extract.run_asd``."""
import unittest

from assist.extract.run_asd import WEIGHTS_CONF_SHIFT, jsonl_rows


class JsonlRowsTest(unittest.TestCase):
    ROW = {"frame": 30, "state": "10000000000", "conf": 0.6, "box": [1, 2, 3, 4]}

    def test_published_predictions_keep_their_confidence(self):
        self.assertEqual(jsonl_rows([self.ROW], 10.0, "release"),
                         [{"f": 30, "t": 3.0, "state": "10000000000", "conf": 0.6, "box": [1, 2, 3, 4]}])

    def test_reproduced_predictions_are_shifted_to_match_the_published_ones(self):
        self.assertAlmostEqual(jsonl_rows([self.ROW], 10.0, "weights")[0]["conf"], 0.6 + WEIGHTS_CONF_SHIFT)

    def test_shifted_confidence_stays_at_most_one(self):
        self.assertEqual(jsonl_rows([{**self.ROW, "conf": 0.99}], 10.0, "weights")[0]["conf"], 1.0)


if __name__ == "__main__":
    unittest.main()
