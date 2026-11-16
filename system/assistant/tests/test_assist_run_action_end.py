"""The action-end label on detected actions: which frames it reads, how it is scored, and how late it arrives."""
import unittest

from assist.engine.events import Event
from assist.extract.run_action_end import best_overlap_accuracy, final_spans, label_delays


def event(start, end, known, ended, label="a"):
    return Event(label, start, end, known_at=known, ended_at=ended)


class FinalSpansTest(unittest.TestCase):
    def test_a_span_runs_from_the_detected_start_to_the_detected_end(self):
        self.assertEqual(final_spans([event(1.0, 2.0, 1.4, 2.4)], 10, 100), [(10, 19)])

    def test_it_never_reads_past_the_moment_the_end_was_confirmed(self):
        self.assertEqual(final_spans([event(1.0, 3.0, 1.4, 2.0)], 10, 100), [(10, 20)])

    def test_it_stops_at_the_last_frame_of_the_video(self):
        self.assertEqual(final_spans([event(9.0, 12.0, 9.4, 12.4)], 10, 100), [(90, 99)])

    def test_a_span_always_holds_at_least_its_first_frame(self):
        self.assertEqual(final_spans([event(5.0, 5.0, 5.0, 5.0)], 10, 100), [(50, 50)])


class AccuracyTest(unittest.TestCase):
    def test_a_true_action_takes_the_label_of_the_detection_overlapping_it_most(self):
        spans, labels = [(0, 2), (3, 9)], [1, 2]
        self.assertEqual(best_overlap_accuracy(spans, labels, [(2, 8, 2)]), (1, 1))

    def test_a_true_action_with_no_detection_counts_as_wrong(self):
        self.assertEqual(best_overlap_accuracy([(0, 2)], [1], [(5, 9, 1)]), (0, 1))

    def test_the_wrong_label_counts_as_wrong(self):
        self.assertEqual(best_overlap_accuracy([(0, 9)], [3], [(0, 9, 1)]), (0, 1))


class DelayTest(unittest.TestCase):
    def test_spaced_actions_wait_only_for_their_own_pass(self):
        self.assertEqual(label_delays([1.0, 5.0], 0.5), [0.5, 0.5])

    def test_a_pass_still_running_makes_the_next_label_wait(self):
        self.assertEqual(label_delays([1.0, 1.2], 1.0), [1.0, 1.8])


if __name__ == "__main__":
    unittest.main()
