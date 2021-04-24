from input_output.ply import import_pointcloud_from_ply, export_numpy_array_to_ply
from mathematics.transformations import remove_points_far_away_from_centroid, remove_points_with_less_neighbours
import numpy as np

from processing.icp import icp


def test_removal_with_radius(file_name="test.ply"):
    points, colors = import_pointcloud_from_ply(file_name)
    nb_neighbours = 35
    radius = 0.05
    indices = remove_points_with_less_neighbours(points, nb_neighbours, radius)
    colors[indices] = (0, 255, 0)
    export_numpy_array_to_ply(points[indices], colors[indices],
                              f'removed_neighbours_less_than_{nb_neighbours}_radius_{radius}.ply',
                              rotate_columns=False)

    indices = np.invert(indices)
    colors[indices] = (0, 0, 255)
    export_numpy_array_to_ply(points[indices], colors[indices],
                              f'what_has_been_removed_{nb_neighbours}_radius_{radius}.ply',
                              rotate_columns=False)


def test_removal_centroid(file_name="test.ply", cutoff=1.0):
    points, colors = import_pointcloud_from_ply(file_name)
    indices = remove_points_far_away_from_centroid(points, cutoff)
    colors[indices] = (255, 0, 0)
    export_numpy_array_to_ply(points[indices], colors[indices], f'removed_with_cutoff_{cutoff}.ply',
                              rotate_columns=False)

    indices = np.invert(indices)
    colors[indices] = (0, 255, 255)
    export_numpy_array_to_ply(points[indices], colors[indices],
                              f'what_has_been_removed_with_cutoff_{cutoff}.ply',
                              rotate_columns=False)


def test_icp(file_name_1="test_nou1.ply", file_name_2="test_nou2.ply", nb_neighbours=35, radius=0.05):
    # remove outliers from point cloud A
    points_a, colors_a = import_pointcloud_from_ply(file_name_1)
    indices = remove_points_with_less_neighbours(points_a, nb_neighbours, radius)
    indices = np.invert(indices)
    points_a[indices] = (0, 0, 0)

    # remove outliers from point cloud B
    points_b, colors_b = import_pointcloud_from_ply(file_name_2)
    indices = remove_points_with_less_neighbours(points_b, nb_neighbours, radius)
    indices = np.invert(indices)
    points_b[indices] = (0, 0, 0)

    # perform icp on point clouds without outliers
    T, distances, index, result = icp(points_a, points_b)
    print("iterations: ", index)
    print("Transformation:\n", T)
    result = np.delete(result, 3, axis=1)
    export_numpy_array_to_ply(result, colors_a,
                              f'after_icp_{file_name_1}.ply',
                              rotate_columns=False)


if __name__ == "__main__":
    # test_removal_centroid("test_nou1.ply", cutoff=1)
    # test_removal_with_radius("test_nou1.ply")
    test_icp()
