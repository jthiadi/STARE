"""The small FBX helpers behind the wrong-install renders."""
import unittest

import numpy as np

from scripts.mistakes.fbx_mesh import components, triangulate


class TriangulateTest(unittest.TestCase):
    def test_a_quad_becomes_two_triangles_of_the_same_polygon(self):
        tris, owner = triangulate(np.array([0, 1, 2, ~3]))
        self.assertEqual(tris.tolist(), [[0, 1, 2], [0, 2, 3]])
        self.assertEqual(owner.tolist(), [0, 0])

    def test_polygons_are_numbered_in_order(self):
        _, owner = triangulate(np.array([0, 1, ~2, 3, 4, ~5]))
        self.assertEqual(owner.tolist(), [0, 1])


class ComponentsTest(unittest.TestCase):
    def test_touching_triangles_are_one_piece_and_separate_ones_are_not(self):
        labels = components(7, np.array([[0, 1, 2], [2, 3, 4], [5, 6, 5]]))
        self.assertEqual(len(set(labels[:5].tolist())), 1)
        self.assertEqual(labels[5], labels[6])
        self.assertNotEqual(labels[0], labels[5])


if __name__ == "__main__":
    unittest.main()
