import unittest2 as unittest
import numpy as np


class TestFeatures(unittest.TestCase):
    def test_distance_between_points(self):
        from icllib.features import distance_between_points

        arr = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(distance_between_points(arr), 2.0)

        arr2 = np.array([[0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0]])
        self.assertEqual(distance_between_points(arr2), 2.0)
