import numpy as np
import scipy.spatial as spatial


def remove_points_far_away_from_centroid(points, cutoff=1.0):
    # find the points` centroid
    centroid = np.mean(points, axis=0)
    distances = np.empty(len(points))
    for i in range(len(points)):
        # exclude points that have zero depth
        if points[i][2] != 0:
            dist = np.linalg.norm(centroid - points[i])
            distances[i] = dist

    # keep only points that are under the cutoff value
    return distances < cutoff


def remove_points_with_less_neighbours(points, nb_neighbours, radius=0.03):
    tree = spatial.cKDTree(points)
    neighbours = np.zeros(len(points))
    for i in range(len(points)):
        # exclude points that have zero depth
        if points[i][2] != 0:
            if neighbours[i] == 0:
                nearest = tree.query_ball_point(points[i], radius)
                if len(nearest) > nb_neighbours:
                    neighbours[nearest] = 1

    # keep only points that have no. of neighbours above the threshold
    return neighbours == 1

