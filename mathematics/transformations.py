import numpy as np

def remove_points_far_away_from_centroid(points, cutoff=1.0):
    # find the points` centroid
    centroid = np.mean(points, axis=0)
    distances = np.empty(len(points))
    for i in range(len(points)):
        if points[i][2] != 0:
            dist = np.linalg.norm(centroid - points[i])
            distances[i] = dist

    # keep only points that are under the threshold
    indices = distances < cutoff
    return indices
