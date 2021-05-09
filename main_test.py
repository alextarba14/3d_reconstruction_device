from input_output.ply import import_pointcloud_from_ply, export_numpy_array_to_ply
from mathematics.transformations import remove_points_far_away_from_centroid, remove_points_with_less_neighbours
import numpy as np
import open3d as o3d

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


def test_icp(file_name1="test_nou1.ply", file_name2="test_nou2.ply", nb_neighbours=35, radius=0.05):
    # remove outliers from point cloud A
    points_a, colors_a = import_pointcloud_from_ply(file_name1)
    # indices = remove_points_with_less_neighbours(points_a, nb_neighbours, radius)
    # indices = np.invert(indices)
    # points_a[indices] = (0, 0, 0)

    # remove outliers from point cloud B
    points_b, colors_b = import_pointcloud_from_ply(file_name2)
    # indices = remove_points_with_less_neighbours(points_b, nb_neighbours, radius)
    # indices = np.invert(indices)
    # points_b[indices] = (0, 0, 0)

    invalid_indices = points_a[:, 2] == 0

    initial_transformation = np.eye(4,4)
    # perform icp on point clouds without outliers
    T = icp(points_a, points_b, initial_transformation)
    print("Transformation:\n", T)

    ones = np.ones((len(points_a), 1))
    points_a = np.append(points_a, ones, axis=1)
    # inv_T = np.linalg.inv(T)
    test = points_a @ T

    # remove the extra column of ones used in homogeneous coordinates
    test = np.delete(test, 3, axis=1)
    # result = np.delete(result, 3, axis=1)

    export_numpy_array_to_ply(test, colors_a,
                              f'after_icp_test_{file_name1}',
                              rotate_columns=False)


def test_open3d(file_name1="test.ply", file_name2="test2.ply"):
    points_a, colors_a = import_pointcloud_from_ply(file_name1)
    source = o3d.io.read_point_cloud(file_name1)
    target = o3d.io.read_point_cloud(file_name2)

    threshold = 0.2
    trans_init = np.eye(4, 4)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))

    T = reg_p2p.transformation

    # get source points
    points_a = source.points

    ones = np.ones((len(points_a), 1))
    points_a = np.append(points_a, ones, axis=1)
    result = points_a @ T

    result = np.delete(result, 3, axis=1)
    export_numpy_array_to_ply(result, colors_a, "open3d_transf.ply", rotate_columns=False)

def remove_empty_points_from_point_clouds(dir_path: str):
    import os
    os.chdir(dir_path)

    for filename in os.listdir(os.getcwd()):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            points_a, colors_a = import_pointcloud_from_ply(f.name)
            indices = points_a[:, 2] != 0
            points_a = points_a[indices]
            colors_a = colors_a[indices]
            export_numpy_array_to_ply(points_a, colors_a, f.name, rotate_columns=False)
            print(f.name)


if __name__ == "__main__":
    # test_removal_centroid("test_nou1.ply", cutoff=1)
    # test_removal_with_radius("test_nou1.ply")
    test_icp()
    # test_open3d("test_nou1.ply", "test_nou2.ply")





