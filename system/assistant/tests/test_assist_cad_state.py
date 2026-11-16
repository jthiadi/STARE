"""The CAD view must show the state the system reported at that moment, and never pretend a wrong state has a model."""
import inspect
import unittest

import numpy as np

from assist.tools.cad_state import INSET_H, INSET_W, align, draw_model, inset, model_number, shown_state, state_name, state_track
from scripts.mistakes.fbx_mesh import Piece

CATEGORIES = ["background", "10000000000", "11110111111", "error_state"]


def block(size=(10.0, 20.0, 40.0), centre=(0.0, 0.0, 0.0), color=(1.0, 1.0, 1.0), material="White.001") -> Piece:
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * np.array(size) + np.array(centre)
    tris = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return Piece(corners, tris, color, material)


class StateNameTest(unittest.TestCase):
    def test_a_list_and_a_string_name_the_same_state(self):
        self.assertEqual(state_name([1, 0, 0]), "100")
        self.assertEqual(state_name("100"), "100")

    def test_the_error_state_keeps_its_name(self):
        self.assertEqual(state_name("error_state"), "error_state")

    def test_a_missing_state_has_no_name(self):
        self.assertIsNone(state_name(None))


class ModelNumberTest(unittest.TestCase):
    def test_a_file_number_is_the_position_in_the_detector_categories(self):
        self.assertEqual(model_number("10000000000", CATEGORIES), 1)
        self.assertEqual(model_number("11110111111", CATEGORIES), 2)

    def test_the_error_state_has_no_model(self):
        self.assertIsNone(model_number("error_state", CATEGORIES))

    def test_background_and_unknown_codes_have_no_model(self):
        self.assertIsNone(model_number("background", CATEGORIES))
        self.assertIsNone(model_number("10101010101", CATEGORIES))


class ShownStateTest(unittest.TestCase):
    def track(self, *rows):
        return state_track([{"t": t, "state": s} for t, s in rows], CATEGORIES)

    def test_nothing_is_shown_before_the_first_reading(self):
        self.assertEqual(shown_state(self.track((5.0, "10000000000")), 4.9), (None, False))

    def test_the_latest_reading_so_far_is_shown_never_a_later_one(self):
        track = self.track((5.0, "10000000000"), (9.0, "11110111111"))
        self.assertEqual(shown_state(track, 8.9), ("10000000000", False))
        self.assertEqual(shown_state(track, 9.0), ("11110111111", False))

    def test_a_wrong_state_keeps_the_last_model_and_is_flagged(self):
        track = self.track((5.0, "10000000000"), (9.0, "error_state"))
        self.assertEqual(shown_state(track, 10.0), ("10000000000", True))

    def test_the_flag_clears_when_a_correct_state_returns(self):
        track = self.track((5.0, "10000000000"), (9.0, "error_state"), (12.0, "11110111111"))
        self.assertEqual(shown_state(track, 13.0), ("11110111111", False))

    def test_a_state_without_a_model_shows_no_model_rather_than_an_old_one(self):
        track = self.track((5.0, "10000000000"), (9.0, "10101010101"))
        self.assertEqual(shown_state(track, 10.0), (None, False))

    def test_a_list_reading_works_like_the_string(self):
        track = state_track([{"t": 1.0, "state": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}], CATEGORIES)
        self.assertEqual(shown_state(track, 2.0), ("10000000000", False))


class AlignTest(unittest.TestCase):
    def test_a_turned_and_moved_model_returns_to_the_reference_pose(self):
        reference = [block(), block((6.0, 6.0, 6.0), (30.0, 0.0, 0.0), material="Magenta.001")]
        a = np.radians(35)
        turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        moved = [Piece(p.verts @ turn.T + np.array([100.0, -50.0, 7.0]), p.tris, p.color, p.material) for p in reference]
        out = align(moved, reference)
        self.assertLess(np.abs(np.concatenate([p.verts for p in out]) - np.concatenate([p.verts for p in reference])).max(), 1e-6)


class DrawTest(unittest.TestCase):
    def test_a_model_is_drawn_on_the_background_at_the_asked_size(self):
        image = draw_model([block()], size=(80, 60), centre=np.zeros(3), span=60.0, background=(1, 2, 3))
        pixels = np.asarray(image)
        self.assertEqual(pixels.shape, (60, 80, 3))
        self.assertEqual(tuple(pixels[0, 0]), (1, 2, 3))
        self.assertTrue((pixels != np.array([1, 2, 3])).any())


class SeeThroughTest(unittest.TestCase):
    def test_a_model_drawn_without_background_leaves_the_rest_clear(self):
        pixels = np.asarray(draw_model([block()], size=(80, 60), centre=np.zeros(3), span=60.0, background=None))
        self.assertEqual(pixels.shape, (60, 80, 4))
        self.assertEqual(pixels[0, 0, 3], 0)
        self.assertEqual(pixels[:, :, 3].max(), 255)

    def test_the_panel_lets_the_video_show_through_but_its_words_stay_solid(self):
        alpha = np.asarray(inset(None, True, ("background",)))[:, :, 3]
        self.assertTrue(60 < alpha[INSET_H // 2 - 40, INSET_W - 20] < 220)
        self.assertEqual(alpha[8:32, 10:200].max(), 255)
        self.assertEqual(alpha[INSET_H - 27, INSET_W // 2 + 120], 255)

    def test_the_panel_is_dark_gray_and_its_title_light(self):
        pixels = np.asarray(inset(None, False, ("background",)))
        self.assertTrue((pixels[INSET_H // 2 - 40, INSET_W - 20, :3] < 90).all())
        title = pixels[8:32, 10:200]
        self.assertGreater(title[title[:, :, 3] == 255, :3].max(), 200)


class RenderOptionTest(unittest.TestCase):
    def test_the_overlay_has_no_cad_view_unless_asked(self):
        from assist.tools.overlay_cards import render
        self.assertIs(inspect.signature(render).parameters["cad"].default, False)


if __name__ == "__main__":
    unittest.main()
