import numpy as np


def distance_between_points(points):
    """Calculate D-dimensional euclidean distance between vertices

       Input NxD NumPy array, where D is dimensions.
    """

    diff = points[1:] - points[:-1]

    return np.sqrt(np.power(diff, 2).sum(axis=1)).sum()
