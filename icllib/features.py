"""Collection of functions related to feature extraction from gazedata."""
import numpy as np


def distance_between_points(points):
    """Calculate D-dimensional euclidean distance between vertices.

    Input NxD NumPy array, where D is dimensions.
    """
    diff = points[1:] - points[:-1]

    return np.sqrt(np.power(diff, 2).sum(axis=1)).sum()


def distance_between_vector_and_point(vec, point):
    """Calculate distance for each element in vector vec from point.

    Input vec NxD Numpy array of vectors,
          points D Numpy vector.

    Trailing axes should have the same dimensions for NumPy broadcasting to
    work.
    """
    diff = vec - point

    return np.sqrt(np.power(diff, 2).sum(axis=1))
