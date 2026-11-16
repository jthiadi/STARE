"""The action-end label: the live assistant and the paper's test must share one function and one locked recipe."""
import unittest

import numpy as np

from assist.extract.action_end import MEMBERS, SAMPLINGS, SMOOTHING, STRENGTH, WEIGHT, end_labels, mean_clip


def frames_of(*labels, classes=3):
    """(T, 1 + classes) frame probabilities with all the weight on one action per frame (1-based), column 0 background."""
    out = np.full((len(labels), classes + 1), 1e-6)
    out[np.arange(len(labels)), labels] = 1.0
    return out / out.sum(1, keepdims=True)


class RecipeTest(unittest.TestCase):
    def test_the_recipe_chosen_on_val_is_locked(self):
        self.assertEqual(MEMBERS, ("vmae_full", "vjft_area_k12", "vjft_hand_k12", "vjft_hand_k12_s1"))
        self.assertEqual((WEIGHT, STRENGTH, SMOOTHING, SAMPLINGS), (0.95, 0.2, 0.01, (0.5, 0.25, 0.75)))


class EndLabelsTest(unittest.TestCase):
    def setUp(self):
        self.first, self.pairs = np.ones(3), np.ones((3, 3))
        self.clip = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

    def label(self, frames, spans, clip=None, w=0.95, strength=0.0, first=None, pairs=None):
        return end_labels(frames, spans, self.clip if clip is None else clip, w, strength, 0.01,
                          self.first if first is None else first, self.pairs if pairs is None else pairs).tolist()

    def test_at_weight_one_only_the_clip_models_decide(self):
        self.assertEqual(self.label(frames_of(3, 3, 3, 3), [(0, 1), (2, 3)], w=1.0), [1, 2])

    def test_at_weight_zero_only_the_tcn_decides(self):
        self.assertEqual(self.label(frames_of(3, 3, 2, 2), [(0, 1), (2, 3)], w=0.0), [3, 2])

    def test_frames_after_the_action_change_nothing(self):
        spans = [(0, 1), (2, 3)]
        changed = frames_of(1, 1, 2, 2, 3, 3)
        same = frames_of(1, 1, 2, 2, 1, 1)
        self.assertEqual(self.label(changed, spans, w=0.5), self.label(same, spans, w=0.5))

    def test_a_later_action_never_changes_an_earlier_label(self):
        frames = frames_of(1, 1, 2, 2)
        first = np.array([1.0, 1.0, 1.0])
        pairs = np.array([[1.0, 50.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        one = self.label(frames, [(0, 1)], clip=self.clip[:1], w=0.5, strength=1.0, first=first, pairs=pairs)
        two = self.label(frames, [(0, 1), (2, 3)], w=0.5, strength=1.0, first=first, pairs=pairs)
        self.assertEqual(two[:1], one)

    def test_the_prior_uses_the_earlier_action(self):
        flat = np.full((2, 3), 1 / 3)
        flat[0] = [0.9, 0.05, 0.05]
        pairs = np.array([[1.0, 1000.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertEqual(self.label(frames_of(1, 1, 1, 1), [(0, 1), (2, 3)], clip=flat, w=1.0, strength=1.0, pairs=pairs)[1], 2)

    def test_no_prior_means_each_action_alone(self):
        self.assertEqual(self.label(frames_of(1, 1, 2, 2), [(0, 1), (2, 3)], w=0.5, strength=0.0), [1, 2])


class MeanClipTest(unittest.TestCase):
    def test_the_members_are_averaged_in_equal_shares(self):
        out = mean_clip({"a": np.array([[1.0, 0.0]]), "b": np.array([[0.0, 1.0]]), "unused": np.array([[1.0, 0.0]])}, ("a", "b"))
        np.testing.assert_allclose(out, [[0.5, 0.5]])

    def test_a_missing_member_is_an_error_not_a_lighter_recipe(self):
        with self.assertRaises(KeyError):
            mean_clip({"a": np.array([[1.0, 0.0]])}, ("a", "b"))


if __name__ == "__main__":
    unittest.main()
