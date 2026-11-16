"""Kit card labels and hand-touched overlay selection."""
import unittest

import numpy as np

from monitor.kit_catalog import (
    KIT_PART_LABELS,
    KIT_PART_PROMPTS,
    ar_object_for_kit,
    allowed_detect_label,
    collapse_overlapping_lookalikes,
    hold_touched_parts,
    kit_detection_plan,
    kit_label_for_detector,
    kit_prompt_for,
    select_hand_touched_parts,
    select_primary_ar_object,
    stabilize_detection_records,
)


class KitCatalogTests(unittest.TestCase):
    def test_labels_match_parts_card_word_for_word(self):
        self.assertEqual(KIT_PART_LABELS, (
            "nut",
            "wing",
            "acorn nut",
            "pulley",
            "round washer",
            "tooth washer",
            "wheel",
            "base",
            "front bracket",
            "short brace",
            "long brace",
            "pin long",
            "pin middle",
            "pin short",
            "screw pin",
        ))

    def test_kit_plan_has_only_card_classes_plus_hand_and_paper(self):
        plan = kit_detection_plan()
        names = [row["name"] for row in plan["classes"]]
        self.assertEqual(names, [*KIT_PART_LABELS, "hand", "paper"])
        self.assertEqual(plan["discarded"], ["paper"])
        for name in KIT_PART_LABELS:
            prompt = kit_prompt_for(name)
            self.assertLessEqual(len(prompt), 80)
            self.assertTrue(prompt)
            self.assertIn(name, KIT_PART_PROMPTS)
        self.assertIn("pink", kit_prompt_for("nut").lower())
        self.assertIn("black", kit_prompt_for("wheel").lower())
        self.assertIn("white", kit_prompt_for("base").lower())
        self.assertIn("groove", kit_prompt_for("pulley").lower())
        self.assertNotIn("hex nut", names)
        self.assertNotIn("toy car wheel", names)
        self.assertNotIn("partial model", names)
        self.assertNotIn("short metal pin", names)

    def test_detector_names_map_to_card_strings(self):
        self.assertEqual(kit_label_for_detector("hex nut"), "nut")
        self.assertEqual(kit_label_for_detector("tooth lock washer"), "tooth washer")
        self.assertEqual(kit_label_for_detector("round flat washer"), "round washer")
        self.assertEqual(kit_label_for_detector("toy car wheel"), "wheel")
        self.assertEqual(kit_label_for_detector("partial model"), "base")
        self.assertEqual(kit_label_for_detector("white front bracket"), "front bracket")
        self.assertEqual(kit_label_for_detector("short metal pin"), "pin short")
        self.assertEqual(kit_label_for_detector("middle metal pin"), "pin middle")
        self.assertEqual(kit_label_for_detector("long metal pin"), "pin long")
        self.assertEqual(kit_label_for_detector("screw pin"), "screw pin")
        self.assertEqual(kit_label_for_detector("acorn nut"), "acorn nut")
        self.assertEqual(kit_label_for_detector("wing beam"), "wing")

    def test_non_kit_classes_are_not_shown(self):
        self.assertIsNone(kit_label_for_detector("hand"))
        self.assertIsNone(kit_label_for_detector("paper"))
        self.assertIsNone(kit_label_for_detector("front chassis"))
        self.assertIsNone(kit_label_for_detector("rear wheel assembly"))

    def test_only_parts_overlapping_the_hand_are_kept(self):
        hand_mask = np.zeros((40, 40), dtype=bool)
        nut_mask = np.zeros((40, 40), dtype=bool)
        far_mask = np.zeros((40, 40), dtype=bool)
        hand_mask[10:22, 10:22] = True
        nut_mask[12:20, 12:20] = True
        far_mask[30:38, 30:38] = True
        objects = [
            {"id": 1, "class": "hand", "score": 0.9, "box": [10, 10, 22, 22]},
            {"id": 2, "class": "hex nut", "score": 0.8, "box": [12, 12, 20, 20]},
            {"id": 3, "class": "hex nut", "score": 0.95, "box": [30, 30, 38, 38]},
            {"id": 4, "class": "toy car wheel", "score": 0.7, "box": [30, 30, 38, 38]},
        ]
        masks = {"1": hand_mask, "2": nut_mask, "3": far_mask, "4": far_mask}
        kept = select_hand_touched_parts(objects, masks=masks)
        self.assertEqual([row["kit_label"] for row in kept], ["nut"])
        self.assertEqual(kept[0]["id"], 2)

    def test_one_box_per_kit_class_and_max_parts(self):
        hand = np.zeros((30, 30), dtype=bool)
        a = np.zeros((30, 30), dtype=bool)
        b = np.zeros((30, 30), dtype=bool)
        c = np.zeros((30, 30), dtype=bool)
        d = np.zeros((30, 30), dtype=bool)
        hand[5:25, 5:25] = True
        a[6:14, 6:14] = True
        b[8:16, 8:16] = True
        c[10:18, 10:18] = True
        d[12:20, 12:20] = True
        objects = [
            {"id": 0, "class": "hand", "score": 1.0, "box": [5, 5, 25, 25]},
            {"id": 1, "class": "hex nut", "score": 0.4, "box": [6, 6, 14, 14]},
            {"id": 2, "class": "hex nut", "score": 0.9, "box": [8, 8, 16, 16]},
            {"id": 3, "class": "short metal pin", "score": 0.8, "box": [10, 10, 18, 18]},
            {"id": 4, "class": "short brace", "score": 0.7, "box": [12, 12, 20, 20]},
            {"id": 5, "class": "pulley", "score": 0.6, "box": [12, 12, 20, 20]},
        ]
        masks = {"0": hand, "1": a, "2": b, "3": c, "4": d, "5": d}
        kept = select_hand_touched_parts(objects, masks=masks, max_parts=3)
        labels = [row["kit_label"] for row in kept]
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(set(labels)), 3)
        self.assertNotIn("hand", labels)
        nuts = [row for row in kept if row["kit_label"] == "nut"]
        if nuts:
            self.assertEqual(nuts[0]["id"], 2)

    def test_default_is_one_primary_object_like_ar(self):
        hand = np.zeros((40, 40), dtype=bool)
        nut = np.zeros((40, 40), dtype=bool)
        base = np.zeros((40, 40), dtype=bool)
        hand[10:28, 10:28] = True
        nut[14:20, 14:20] = True
        base[8:30, 8:30] = True
        objects = [
            {"id": 0, "class": "hand", "score": 1.0, "box": [10, 10, 28, 28]},
            {"id": 1, "class": "hex nut", "score": 0.7, "box": [14, 14, 20, 20]},
            {"id": 2, "class": "partial model", "score": 0.95, "box": [8, 8, 30, 30]},
        ]
        primary = select_primary_ar_object(
            objects, masks={"0": hand, "1": nut, "2": base},
        )
        self.assertIsNotNone(primary)
        self.assertEqual(primary["kit_label"], "nut")
        self.assertEqual(primary["ar_object"], "nut")
        self.assertEqual(ar_object_for_kit("pin short"), "pin_short")
        self.assertEqual(ar_object_for_kit("base"), "partial_model")
        # Default max_parts is 1.
        kept = select_hand_touched_parts(
            objects, masks={"0": hand, "1": nut, "2": base},
        )
        self.assertEqual(len(kept), 1)

    def test_huge_hand_box_does_not_count_as_touching_a_pile_part(self):
        hand_mask = np.zeros((80, 120), dtype=bool)
        washer_mask = np.zeros((80, 120), dtype=bool)
        hand_mask[50:70, 10:40] = True
        washer_mask[5:15, 95:110] = True
        objects = [
            {"id": 1, "class": "hand", "score": 0.9, "box": [0, 0, 119, 79]},
            {"id": 2, "class": "tooth lock washer", "score": 0.8,
             "box": [95, 5, 110, 15]},
        ]
        masks = {"1": hand_mask, "2": washer_mask}
        self.assertEqual(
            select_hand_touched_parts(objects, masks=masks),
            [],
        )

    def test_part_box_on_hand_silhouette_counts_even_if_masks_are_disjoint(self):
        hand_mask = np.zeros((40, 40), dtype=bool)
        nut_mask = np.zeros((40, 40), dtype=bool)
        hand_mask[10:22, 10:22] = True
        nut_mask[18:24, 18:24] = True
        # Nut mask barely misses the undilated hand; the box still sits on it.
        objects = [
            {"id": 1, "class": "hand", "score": 0.9, "box": [10, 10, 22, 22]},
            {"id": 2, "class": "hex nut", "score": 0.8, "box": [12, 12, 24, 24]},
        ]
        kept = select_hand_touched_parts(
            objects, masks={"1": hand_mask, "2": nut_mask},
        )
        self.assertEqual([row["kit_label"] for row in kept], ["nut"])
        objects = [
            {"id": 1, "class": "hex nut", "score": 0.9, "box": [10, 10, 20, 20]},
        ]
        self.assertEqual(select_hand_touched_parts(objects), [])

    def test_only_kit_parts_and_hand_are_allowed(self):
        self.assertEqual(allowed_detect_label("hex nut"), "nut")
        self.assertEqual(allowed_detect_label("hand"), "hand")
        self.assertIsNone(allowed_detect_label("paper"))
        self.assertIsNone(allowed_detect_label("front chassis"))

    def test_same_frame_nut_and_washer_collapse_to_one(self):
        rows = collapse_overlapping_lookalikes([
            {"id": 1, "class": "nut", "kit_label": "nut", "score": 0.9,
             "box": [10.0, 10.0, 20.0, 20.0]},
            {"id": 2, "class": "tooth washer", "kit_label": "tooth washer",
             "score": 0.7, "box": [11.0, 11.0, 21.0, 21.0]},
        ])
        self.assertEqual([row["kit_label"] for row in rows], ["nut"])

    def test_alternating_nut_washer_does_not_blink(self):
        records = []
        for index, name in enumerate(
            ["hex nut", "tooth lock washer", "hex nut", "tooth lock washer",
             "hex nut", "round flat washer"]
        ):
            records.append({
                "index": index,
                "timestamp_s": index / 10.0,
                "objects": [
                    {"id": 1, "class": name, "score": 0.8,
                     "box": [40.0, 40.0, 70.0, 70.0]},
                    {"id": 2, "class": "hand", "score": 0.9,
                     "box": [30.0, 30.0, 80.0, 80.0]},
                ],
            })
        stable = stabilize_detection_records(records)
        labels = [
            next(obj["class"] for obj in rec["objects"] if obj["class"] != "hand")
            for rec in stable
        ]
        self.assertEqual(labels, ["nut"] * 6)
        self.assertTrue(all(
            any(obj["class"] == "hand" for obj in rec["objects"])
            for rec in stable
        ))
        self.assertFalse(any(
            obj["class"] == "paper"
            for rec in stable for obj in rec["objects"]
        ))

    def test_sustained_new_class_can_switch(self):
        records = []
        for index in range(12):
            name = "hex nut" if index < 2 else "tooth lock washer"
            records.append({
                "index": index,
                "objects": [{
                    "id": 1, "class": name, "score": 0.75,
                    "box": [10.0, 10.0, 30.0, 30.0],
                }],
            })
        stable = stabilize_detection_records(records)
        labels = [rec["objects"][0]["class"] for rec in stable]
        self.assertEqual(labels[0], "nut")
        self.assertEqual(labels[-1], "tooth washer")

    def test_touch_hold_keeps_label_across_a_short_gap(self):
        current = []
        previous = [{"kit_label": "nut", "class": "nut", "box": [1, 2, 3, 4]}]
        kept, ages = hold_touched_parts(current, previous, {}, hold=4, max_parts=3)
        self.assertEqual([row["kit_label"] for row in kept], ["nut"])
        self.assertTrue(kept[0].get("held"))
        self.assertEqual(ages["nut"], 1)


if __name__ == "__main__":
    unittest.main()
