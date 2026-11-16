"""Named SOP relations + temporal hold (RelateAnything-inspired, geometric)."""
import unittest

from monitor.sop_relations import (
    RelationHoldFilter,
    geometric_predicate_scores,
    predict_step_relations,
    relation_supports_done,
    summarize_relations,
)


class SopRelationTests(unittest.TestCase):
    def test_attached_score_high_for_overlapping_boxes(self):
        subject = [100.0, 100.0, 180.0, 160.0]
        target = [120.0, 110.0, 260.0, 220.0]
        scores = geometric_predicate_scores(subject, target)
        self.assertGreaterEqual(scores["attached to"], 0.50)
        self.assertGreaterEqual(scores["on"], 0.35)

    def test_near_score_low_when_far(self):
        subject = [10.0, 10.0, 40.0, 40.0]
        target = [400.0, 300.0, 520.0, 420.0]
        scores = geometric_predicate_scores(subject, target)
        self.assertLess(scores["attached to"], 0.35)
        self.assertLess(scores["near"], 0.35)

    def test_hold_filter_keeps_edge_after_miss(self):
        filt = RelationHoldFilter(hold_s=1.5)
        filt.update(
            "step:IR09",
            timestamp_s=10.0,
            score=0.8,
            payload={
                "step_id": "IR09",
                "predicate": "attached to",
                "subject": "long brace",
                "object": "partial model",
                "subject_box": [1, 1, 2, 2],
                "object_box": [1.5, 1.5, 3, 3],
            },
        )
        held = filt.update("step:IR09", timestamp_s=11.0, score=None, payload=None)
        self.assertTrue(held["held"])
        self.assertGreaterEqual(held["score"], 0.7)

    def test_predict_step_relations_emits_labeled_edge(self):
        rules = [{
            "step_id": "IR09",
            "moving_queries": ["long brace"],
            "target_queries": ["partial model"],
            "fastener_queries": [],
        }]
        objects = [
            {"class": "long brace", "box": [140.0, 120.0, 260.0, 180.0], "id": 1},
            {"class": "partial model", "box": [150.0, 110.0, 280.0, 220.0], "id": 2},
        ]
        edges = predict_step_relations(rules, objects, step_ids=["IR09"])
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["kind"], "install")
        self.assertIn(edges[0]["predicate"], {"attached to", "on"})
        self.assertGreaterEqual(edges[0]["score"], 0.50)
        self.assertIn("attached to", summarize_relations(edges))

    def test_predict_emits_hand_holding_edge(self):
        rules = [{
            "step_id": "IR03",
            "moving_queries": ["short brace"],
            "target_queries": ["partial model"],
            "fastener_queries": [],
        }]
        objects = [
            {"class": "hand", "box": [100.0, 100.0, 160.0, 160.0], "id": 0},
            {"class": "short brace", "box": [120.0, 110.0, 200.0, 170.0], "id": 1},
            {"class": "partial model", "box": [400.0, 300.0, 520.0, 420.0], "id": 2},
        ]
        edges = predict_step_relations(rules, objects, step_ids=["IR03"])
        kinds = {edge["kind"] for edge in edges}
        self.assertIn("hand", kinds)
        hand = next(edge for edge in edges if edge["kind"] == "hand")
        self.assertEqual(hand["subject"], "hand")
        self.assertEqual(hand["object"], "short brace")
        self.assertIn(hand["predicate"], {"holding", "near"})
        self.assertGreaterEqual(hand["score"], 0.35)
        self.assertIn("holding", summarize_relations(edges) + hand["predicate"])

    def test_refresh_moves_held_edge_to_live_boxes(self):
        from monitor.sop_relations import refresh_relation_edges_for_frame
        edges = [{
            "step_id": "IR03",
            "kind": "hand",
            "predicate": "holding",
            "score": 0.7,
            "subject": "hand",
            "object": "short brace",
            "subject_box": [10.0, 10.0, 40.0, 40.0],
            "object_box": [50.0, 50.0, 80.0, 80.0],
            "held": True,
        }]
        objects = [
            {"class": "hand", "box": [200.0, 200.0, 260.0, 260.0], "id": 1},
            {"class": "short brace", "box": [220.0, 210.0, 300.0, 270.0], "id": 2},
        ]
        out = refresh_relation_edges_for_frame(edges, objects)
        self.assertTrue(out[0]["draw"])
        self.assertFalse(out[0]["held"])
        self.assertEqual(out[0]["subject_box"], [200.0, 200.0, 260.0, 260.0])

    def test_relation_supports_done_thresholds(self):
        self.assertFalse(relation_supports_done(0.4))
        self.assertTrue(relation_supports_done(0.6))
        self.assertFalse(relation_supports_done(0.6, strong=True))
        self.assertTrue(relation_supports_done(0.75, strong=True))


if __name__ == "__main__":
    unittest.main()
