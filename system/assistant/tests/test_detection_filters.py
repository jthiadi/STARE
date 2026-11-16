"""Detection size gates, pin length labels, and conflict resolution."""
import unittest

import numpy as np

from monitor.detection_filters import (
    class_family,
    disambiguate_length_pairs,
    passes_size_gate,
    refine_detections,
    resolve_class_conflicts,
    suppress_duplicates,
)


def _mask(h, w, y0, x0, y1, x1):
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class DetectionFilterTests(unittest.TestCase):
    def test_large_washer_fails_size_gate(self):
        mask = _mask(200, 200, 40, 40, 120, 120)
        row = {"class": "round flat washer", "score": 0.9, "mask": mask,
               "box": [40.0, 40.0, 120.0, 120.0]}
        self.assertEqual(class_family(row["class"]), "washer")
        self.assertFalse(passes_size_gate(row))

    def test_small_washer_passes_size_gate(self):
        mask = _mask(200, 200, 90, 90, 100, 100)
        row = {"class": "round flat washer", "score": 0.8, "mask": mask,
               "box": [90.0, 90.0, 100.0, 100.0]}
        self.assertTrue(passes_size_gate(row))

    def test_washer_overlapping_wheel_is_dropped(self):
        wheel = {
            "class": "toy car wheel", "score": 0.85,
            "mask": _mask(200, 200, 50, 50, 120, 120),
            "box": [50.0, 50.0, 120.0, 120.0],
        }
        washer = {
            "class": "round flat washer", "score": 0.8,
            "mask": _mask(200, 200, 80, 80, 88, 88),
            "box": [80.0, 80.0, 88.0, 88.0],
        }
        self.assertTrue(passes_size_gate(washer))
        kept = resolve_class_conflicts([wheel, washer])
        names = {row["class"] for row in kept}
        self.assertIn("toy car wheel", names)
        self.assertNotIn("round flat washer", names)

    def test_refine_keeps_wheel_drops_huge_washer(self):
        rows = [
            {"class": "toy car wheel", "score": 0.9,
             "mask": _mask(200, 200, 50, 50, 100, 100),
             "box": [50.0, 50.0, 100.0, 100.0]},
            {"class": "round flat washer", "score": 0.88,
             "mask": _mask(200, 200, 40, 40, 120, 120),
             "box": [40.0, 40.0, 120.0, 120.0]},
            {"class": "hex nut", "score": 0.7,
             "mask": _mask(200, 200, 10, 10, 18, 18),
             "box": [10.0, 10.0, 18.0, 18.0]},
        ]
        cleaned = refine_detections(rows, frame_area=200 * 200)
        names = {row["class"] for row in cleaned}
        self.assertIn("toy car wheel", names)
        self.assertIn("hex nut", names)
        self.assertNotIn("round flat washer", names)

    def test_nut_on_pin_head_is_dropped(self):
        pin = {
            "class": "short metal pin", "score": 0.9,
            "mask": _mask(200, 200, 40, 40, 55, 100),
            "box": [40.0, 40.0, 100.0, 55.0],
        }
        nut = {
            "class": "hex nut", "score": 0.7,
            "mask": _mask(200, 200, 42, 42, 52, 52),
            "box": [42.0, 42.0, 52.0, 52.0],
        }
        kept = resolve_class_conflicts([pin, nut])
        names = {row["class"] for row in kept}
        self.assertIn("short metal pin", names)
        self.assertNotIn("hex nut", names)

    def test_pin_length_splits_short_middle_long(self):
        short = {
            "class": "screw pin", "score": 0.9,
            "mask": _mask(200, 200, 22, 10, 30, 40),  # thin short rod
            "box": [10.0, 22.0, 40.0, 30.0],
        }
        long = {
            "class": "short metal pin", "score": 0.9,
            "mask": _mask(200, 200, 52, 10, 60, 140),  # thin long rod
            "box": [10.0, 52.0, 140.0, 60.0],
        }
        out = disambiguate_length_pairs([short, long])
        by_len = {round(row["box"][2] - row["box"][0]): row["class"] for row in out}
        # Unique short screw-pin label is kept; long smooth rod → long metal pin.
        self.assertEqual(by_len[30], "screw pin")
        self.assertEqual(by_len[130], "long metal pin")

    def test_overlapping_pin_labels_are_not_ambiguous(self):
        # Thin rod (short side < 0.048 * frame) stays a pin, not a brace.
        mask = _mask(200, 200, 46, 10, 54, 140)
        rows = [
            {"class": "short metal pin", "score": 0.91, "mask": mask,
             "box": [10.0, 46.0, 140.0, 54.0]},
            {"class": "screw pin", "score": 0.90, "mask": mask,
             "box": [10.0, 46.0, 140.0, 54.0]},
        ]
        cleaned = refine_detections(rows, frame_area=200 * 200)
        self.assertEqual(len(cleaned), 1)
        # Long smooth rod must not be forced to screw pin.
        self.assertEqual(cleaned[0]["class"], "long metal pin")
        self.assertNotIn("ambiguous_classes", cleaned[0])

    def test_short_tpin_beats_overlapping_screw_pin(self):
        mask = _mask(200, 200, 46, 10, 54, 40)
        rows = [
            {"class": "short metal pin", "score": 0.70, "mask": mask,
             "box": [10.0, 46.0, 40.0, 54.0]},
            {"class": "screw pin", "score": 0.90, "mask": mask,
             "box": [10.0, 46.0, 40.0, 54.0]},
        ]
        cleaned = refine_detections(rows, frame_area=200 * 200)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["class"], "short metal pin")
        self.assertNotIn("ambiguous_classes", cleaned[0])

    def test_nut_washer_near_tie_picks_winner(self):
        mask = _mask(200, 200, 90, 90, 98, 98)
        rows = [
            {"class": "hex nut", "score": 0.81, "mask": mask, "box": [90.0, 90.0, 98.0, 98.0]},
            {"class": "tooth lock washer", "score": 0.78, "mask": mask,
             "box": [90.0, 90.0, 98.0, 98.0]},
        ]
        cleaned = suppress_duplicates(disambiguate_length_pairs(rows))
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["class"], "hex nut")
        self.assertNotIn("ambiguous_classes", cleaned[0])

    def test_length_splits_short_and_long_brace(self):
        short = {
            "class": "long brace", "score": 0.8,
            "mask": _mask(200, 200, 20, 10, 30, 50),
            "box": [10.0, 20.0, 50.0, 30.0],
        }
        long = {
            "class": "short brace", "score": 0.8,
            "mask": _mask(200, 200, 50, 10, 60, 140),
            "box": [10.0, 50.0, 140.0, 60.0],
        }
        out = disambiguate_length_pairs([short, long])
        by_box = {(round(row["box"][2] - row["box"][0])): row["class"] for row in out}
        self.assertEqual(by_box[40], "short brace")
        self.assertEqual(by_box[130], "long brace")

    def test_squat_pin_blob_is_dropped(self):
        blob = {
            "class": "short metal pin", "score": 0.9,
            "mask": _mask(200, 200, 40, 40, 70, 75),  # ~aspect 1.17
            "box": [40.0, 40.0, 75.0, 70.0],
        }
        cleaned = refine_detections([blob], frame_area=200 * 200)
        self.assertEqual(cleaned, [])

    def test_thick_elongated_pin_becomes_brace(self):
        bar = {
            "class": "screw pin", "score": 0.85,
            "mask": _mask(200, 200, 30, 5, 55, 160),  # short=25, long=155
            "box": [5.0, 30.0, 160.0, 55.0],
        }
        cleaned = refine_detections([bar], frame_area=200 * 200)
        self.assertEqual(len(cleaned), 1)
        self.assertIn(cleaned[0]["class"], {"long brace", "short brace"})

    def test_oversized_bracket_becomes_partial_model(self):
        mid = {
            "class": "white front bracket", "score": 0.8,
            "mask": _mask(200, 200, 50, 50, 110, 100),  # area ≈ 0.075
            "box": [50.0, 50.0, 100.0, 110.0],
        }
        cleaned = refine_detections([mid], frame_area=200 * 200)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["class"], "partial model")

    def test_wrench_leftover_is_dropped(self):
        tool = {
            "class": "wrench", "score": 0.7,
            "mask": _mask(200, 200, 20, 20, 40, 80),
            "box": [20.0, 20.0, 80.0, 40.0],
        }
        cleaned = refine_detections([tool], frame_area=200 * 200)
        self.assertEqual(cleaned, [])

    def test_partial_model_not_left_ambiguous_with_brace(self):
        # Large compact assembly mask claimed as both brace and partial model.
        mask = _mask(200, 200, 40, 40, 120, 130)  # area ≈ 0.18 before size gate;
        # Stay inside partial max 0.22 and brace max 0.12: use ~0.06 area.
        mask = _mask(200, 200, 50, 40, 110, 120)  # 60*80=4800/40000=0.12
        rows = [
            {"class": "long brace", "score": 0.94, "mask": mask,
             "box": [40.0, 50.0, 120.0, 110.0]},
            {"class": "partial model", "score": 0.90, "mask": mask,
             "box": [40.0, 50.0, 120.0, 110.0]},
            {"class": "chassis base", "score": 0.88, "mask": mask,
             "box": [40.0, 50.0, 120.0, 110.0]},
        ]
        cleaned = refine_detections(rows, frame_area=200 * 200)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["class"], "partial model")
        self.assertNotIn("ambiguous_classes", cleaned[0])

    def test_small_bracket_wins_over_brace_on_compact_mask(self):
        mask = _mask(200, 200, 80, 80, 105, 110)  # area ≈ 0.01875
        rows = [
            {"class": "short brace", "score": 0.80, "mask": mask,
             "box": [80.0, 80.0, 110.0, 105.0]},
            {"class": "white front bracket", "score": 0.79, "mask": mask,
             "box": [80.0, 80.0, 110.0, 105.0]},
        ]
        cleaned = refine_detections(rows, frame_area=200 * 200)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["class"], "white front bracket")
        self.assertNotIn("ambiguous_classes", cleaned[0])

    def test_pulley_is_not_wheel_family(self):
        self.assertEqual(class_family("pulley"), "pulley")
        self.assertEqual(class_family("toy car wheel"), "wheel")

    def test_tire_sized_pulley_becomes_wheel(self):
        blob = {
            "class": "pulley", "score": 0.8,
            "mask": _mask(200, 200, 40, 40, 90, 90),  # area 0.0625
            "box": [40.0, 40.0, 90.0, 90.0],
        }
        cleaned = refine_detections([blob], frame_area=200 * 200)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["class"], "toy car wheel")

    def test_small_pulley_beats_overlapping_wheel_label(self):
        mask = _mask(200, 200, 90, 90, 110, 110)  # area 0.01
        rows = [
            {"class": "toy car wheel", "score": 0.9, "mask": mask,
             "box": [90.0, 90.0, 110.0, 110.0]},
            {"class": "pulley", "score": 0.6, "mask": mask,
             "box": [90.0, 90.0, 110.0, 110.0]},
        ]
        cleaned = refine_detections(rows, frame_area=200 * 200)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["class"], "pulley")


if __name__ == "__main__":
    unittest.main()
