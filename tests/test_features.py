"""Tests for functions in features module."""
import unittest2 as unittest
import numpy as np


class TestFeatures(unittest.TestCase):
    """Test case for functions in features module."""

    def test_distance_between_points(self):
        """Test disstance between points with few simple test vectors."""
        from icllib.features import distance_between_points

        arr = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(distance_between_points(arr), 2.0)

        arr2 = np.array([[0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0]])
        self.assertEqual(distance_between_points(arr2), 2.0)

    def test_distance_between_vector_and_point(self):
        """Test distance between vector and points with simple test vector."""
        from icllib.features import distance_between_vector_and_point

        arr2 = np.array([[0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0]])
        p = np.array([0.0, 0.0, 0.0])
        dists = np.array([0.0, 1.0, 0.0])

        self.assertEqual(
            np.array_equal(distance_between_vector_and_point(arr2, p), dists),
            True)
