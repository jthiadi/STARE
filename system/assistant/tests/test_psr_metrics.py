"""Small deterministic tests for IndustReal-style event metrics."""
import unittest

from monitor.psr_metrics import procedure_order_similarity, score_psr_events


class PsrMetricTests(unittest.TestCase):
    def test_perfect_events_have_perfect_scores_and_delay(self):
        truth = [{"step_id": 3, "timestamp_s": 10}, {"step_id": 9, "timestamp_s": 20}]
        prediction = [{"step_id": "IR03", "timestamp_s": 11},
                      {"step_id": "IR09", "timestamp_s": 23}]
        result = score_psr_events(truth, prediction)
        self.assertEqual(result["procedure_order_similarity"], 1.0)
        self.assertEqual(result["system_f1"], 1.0)
        self.assertEqual(result["average_completion_delay_seconds"], 2.0)

    def test_early_and_duplicate_predictions_are_false_positives(self):
        truth = [{"step_id": 3, "timestamp_s": 10}]
        prediction = [{"step_id": 3, "timestamp_s": 8},
                      {"step_id": 3, "timestamp_s": 12},
                      {"step_id": 3, "timestamp_s": 13}]
        result = score_psr_events(truth, prediction)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 2)
        self.assertEqual(result["false_negatives"], 0)

    def test_adjacent_swap_has_one_edit_penalty(self):
        self.assertAlmostEqual(procedure_order_similarity([3, 6, 9], [3, 9, 6]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
