"""Invariants for automatic, annotation-free assembly inference."""
import unittest
import numpy as np
from scripts.kit_sop.run_automatic_assembly import filter_detections, step_context, latest_at


def _mask(h, w, y0, x0, y1, x1):
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class AutomaticAssemblyTests(unittest.TestCase):
    def test_future_observation_is_not_used_for_earlier_video_frame(self):
        rows = [{'timestamp_s': 5, 'step': 'S1'}, {'timestamp_s': 10, 'step': 'S2'}]
        self.assertIsNone(latest_at(rows,4))
        self.assertEqual(latest_at(rows,9)['step'],'S1')
    def test_eval_paths_and_references_never_enter_model_context(self):
        sop = {'evaluation_sources': ['secret_labels.csv'], 'steps': [{
            'id': 'S1', 'instruction': 'Attach bracket', 'parts': ['bracket'],
            'visual_references': [{'path': 'annotated_test_frame.png'}],
            'completion_timestamp': 123}]}
        context = str(step_context(sop))
        self.assertNotIn('secret_labels', context)
        self.assertNotIn('annotated_test', context)
        self.assertNotIn('123', context)
        self.assertIn('Attach bracket', context)

    def test_printed_part_is_suppressed_but_real_part_survives(self):
        # Full-frame masks: printed nut on paper vs real nut elsewhere.
        paper = _mask(200, 200, 0, 0, 80, 80)
        printed = _mask(200, 200, 10, 10, 18, 18)
        real = _mask(200, 200, 150, 150, 158, 158)
        rows = filter_detections([
            {'class':'paper','mask':paper,'score':.9},
            {'class':'nut','mask':printed,'score':.8},
            {'class':'nut','mask':real,'score':.8}])
        self.assertEqual(len(rows),1)
        self.assertTrue(np.array_equal(rows[0]['mask'],real))

    def test_conflicting_labels_remain_ambiguous(self):
        # Same tiny fastener region claimed as nut and washer.
        mask = _mask(200, 200, 90, 90, 98, 98)
        rows = filter_detections([{'class':'nut','score':.81,'mask':mask},
                                  {'class':'washer','score':.78,'mask':mask}])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['ambiguous_classes'],['nut','washer'])

    def test_separate_same_class_instances_survive(self):
        a = _mask(200, 200, 10, 10, 18, 18)
        b = _mask(200, 200, 150, 150, 158, 158)
        rows = filter_detections([{'class':'nut','score':.8,'mask':a},
                                  {'class':'nut','score':.7,'mask':b}])
        self.assertEqual(len(rows),2)


if __name__=='__main__':
    unittest.main()
