import time
import pyrealsense2 as rs
import numpy as np
import scipy.spatial as spatial


def get_texture_for_pointcloud(vertices, tex_coords, texture_data, w, h, bytes_per_pixel, stride_in_bytes):
    # Reference: https://github.com/Resays/xyz_rgb_realsense/blob/ede2bf9cc81d67ff0a7b616a5c70ff529e43bfe3/xyz_rgb_realsense.cpp
    start_time = time.time()
    # get information from color frame
    texture = []
    length = len(vertices)
    for i in range(length):
        texture.append(get_texture_color(tex_coords[i], texture_data, w, h, bytes_per_pixel, stride_in_bytes))

    print("It took: ", time.time() - start_time)
    return np.asanyarray(texture).reshape(-1, 3)


def get_texture_color(tex_coords: rs.texture_coordinate, texture_data, w, h, bytes_per_pixel, stride_in_bytes):
    """
    Returns a tuple in (R,G,B) format.
    """
    x = min(max(int(tex_coords[0] * w + 0.5), 0), w - 1)
    y = min(max(int(tex_coords[1] * h + 0.5), 0), h - 1)
    idx = x * bytes_per_pixel + y * stride_in_bytes
    return [texture_data[idx][0], texture_data[idx + 1][0], texture_data[idx + 2][0]]


def remove_points_far_away_from_centroid(points, cutoff=1.0):
    """
    Removes points that are above the cutoff specified distance
    regarding the centroid of the point cloud.
    """
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
    """
    It searches neighbors for each point in a given radius.
    If the number of neighbors is greater than the number specified,
    then the point and his neighbours will be kept, otherwise it will be dropped.
    """
    tree = spatial.cKDTree(points)
    # keep track of visited neighbors
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


def remove_statistical_outliers(point_cloud, nb_neighbours: int, std_ratio: float):
    """
    For a given point cloud it searches only for the valid points
    removing outliers from the point cloud.
    For each point, we compute the mean distance from it to all its neighbors.
    By assuming that the resulted distribution is Gaussian with a mean and a standard deviation,
    all points whose mean distances are outside an interval defined by the global distances mean
    and standard deviation can be considered as outliers and trimmed from the dataset.
    """
    # create a KDTree structure for the given point cloud
    kd_tree = spatial.cKDTree(point_cloud)

    # get distances for each point to his nearest <nb_neighbours> points
    distances, indices = kd_tree.query(point_cloud, k=nb_neighbours, p=2, n_jobs=-1)

    # compute mean distance for each point in point cloud
    distances_squared = np.square(distances)

    # get mean of distances squared for each point
    mean_distances_squared = np.mean(distances_squared, axis=1)

    # compute cloud mean distance squared
    cloud_mean = np.mean(distances_squared)

    # get standard deviation
    std_dev = np.std(distances_squared)

    # compute the threshold used to remove outliers
    upper_threshold = cloud_mean + std_ratio * std_dev
    bottom_threshold = cloud_mean - std_ratio * std_dev

    # get indices from cloud_mean < sigma
    positive = mean_distances_squared < upper_threshold
    negative = bottom_threshold < mean_distances_squared

    # mask indices from -sigma < cloud_mean < +sigma from distribution
    return positive & negative


def down_sample_point_cloud(point_cloud):
    """
    Down sample point cloud to remove points that are to close to each other
    and do not carry additional information.
    Returns:
        Indices of points that need to be kept in point cloud.
    """
    # create a KDTree structure to find nearest neighbors
    kd_tree = spatial.cKDTree(point_cloud)
    # get the distances and indices between each point and 5 nearest neighbors
    dist, indices = kd_tree.query(point_cloud, k=5, p=2, n_jobs=-1)

    # create an array with boolean values to keep or drop indices
    keep_indices = np.full(len(point_cloud), False, dtype=bool)
    for i in range(len(point_cloud)):
        distances = dist[i]
        inds = indices[i]
        # keep only indices that are above median value in distances
        valid = dist[i, :] > np.median(distances)
        for index in inds[valid]:
            keep_indices[index] = True

    print("Valid indices acquired.")
    return keep_indices
