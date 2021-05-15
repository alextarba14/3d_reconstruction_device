from icp_point_to_plane.icp_point_to_plane import icp
from input_output.ply import import_point_cloud_from_ply, export_numpy_array_to_ply
from processing.icp import icp_point_to_point
from processing.process import remove_points_far_away_from_centroid, remove_points_with_less_neighbours, \
    down_sample_point_cloud, remove_statistical_outliers
import numpy as np
import open3d as o3d


def test_removal_with_radius(file_name="test.ply", nb_neighbours=35, radius=0.05):
    points, colors = import_point_cloud_from_ply(file_name)
    indices = remove_points_with_less_neighbours(points, nb_neighbours, radius)
    indices = np.invert(indices)
    colors[indices] = (0, 0, 255)
    export_numpy_array_to_ply(points, colors,
                              f'removed_neighbours_less_than_{nb_neighbours}_radius_{radius}.ply',
                              rotate_columns=False)


def test_removal_centroid(file_name="test.ply", cutoff=1.0):
    points, colors = import_point_cloud_from_ply(file_name)
    indices = remove_points_far_away_from_centroid(points, cutoff)

    indices = np.invert(indices)
    colors[indices] = (0, 0, 255)
    export_numpy_array_to_ply(points, colors,
                              f'removed_with_cutoff_{cutoff}.ply',
                              rotate_columns=False)


def test_remove_outliers(file_name="outliers.ply", nb_neighbours=50, std_ratio=1):
    points_a, colors_a = import_point_cloud_from_ply(file_name)
    valid_indices = remove_statistical_outliers(points_a, nb_neighbours, std_ratio)
    invalid_indices = np.invert(valid_indices)
    colors_a[invalid_indices] = (0, 0, 255)
    export_numpy_array_to_ply(points_a[valid_indices], colors_a[valid_indices], f"removed_outliers_result.ply", rotate_columns=False)
    export_numpy_array_to_ply(points_a, colors_a, f"removed_outliers_result_color.ply", rotate_columns=False)


def test_icp(file_name1="test_nou1.ply", file_name2="test_nou2.ply"):
    # import point cloud a
    X_src, colors_src = import_point_cloud_from_ply(file_name1)
    # remove invalid points first
    indices = X_src[:, 2] != 0
    X_src = X_src[indices]
    colors_a = colors_src[indices]

    indices = remove_statistical_outliers(X_src, nb_neighbours=50, std_ratio=1)

    # keep only the valid points, without outliers
    points_a = X_src[indices]
    colors_a = colors_a[indices]

    # import point cloud b
    X_dst, colors_b = import_point_cloud_from_ply(file_name2)

    # remove invalid points first
    indices = X_dst[:, 2] != 0
    X_dst = X_dst[indices]
    indices = remove_statistical_outliers(X_dst, nb_neighbours=50, std_ratio=1)
    points_b = X_dst[indices]

    # remove some points from the bigger point cloud to have the same shape
    len_b = points_b.shape[0]
    len_a = points_a.shape[0]
    if len_b < len_a:
        points_a = points_a[:-(len_a - len_b)]
        colors_a = colors_a[:-(len_a - len_b)]
    else:
        points_b = points_b[:-(len_b - len_a)]

    initial_transformation = np.eye(4, 4)
    # perform icp on point clouds without outliers
    T = icp_point_to_point(points_a, points_b, initial_transformation)
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
    points_a, colors_a = import_point_cloud_from_ply(file_name1)
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
            points_a, colors_a = import_point_cloud_from_ply(f.name)
            indices = points_a[:, 2] != 0
            points_a = points_a[indices]
            colors_a = colors_a[indices]
            export_numpy_array_to_ply(points_a, colors_a, f.name, rotate_columns=False)
            print(f.name)


def test_icp_point_to_plane(file_name1="test_nou1.ply", file_name2="test_nou2.ply"):
    # Mapping source over destination using icp point_to_plane method
    # import source point cloud
    X_src, colors_src = import_point_cloud_from_ply(file_name1)
    valid_indices_src = X_src[:, 2] != 0
    X_src = X_src[valid_indices_src]

    # remove outliers from source
    indices = remove_statistical_outliers(X_src, nb_neighbours=50, std_ratio=1)
    X_src = X_src[indices]

    # import destination point cloud
    X_dst, colors_dst = import_point_cloud_from_ply(file_name2)
    valid_indices_dst = X_dst[:, 2] != 0
    X_dst = X_dst[valid_indices_dst]
    colors_dst = colors_dst[valid_indices_dst]

    # remove outliers from destination
    indices = remove_statistical_outliers(X_dst, nb_neighbours=50, std_ratio=1)
    X_dst = X_dst[indices]
    colors_dst = colors_dst[indices]

    # get transformation matrix that match destination over source
    Tr = icp(X_src, X_dst)

    ones = np.ones((len(X_dst), 1))
    X_dst = np.append(X_dst, ones, axis=1)
    X_result = Tr @ X_dst.T
    X_result = X_result.T
    X_result = np.delete(X_result, 3, axis=1)

    export_numpy_array_to_ply(X_result, colors_dst, "after_icp_point_without_outliers_to_plane.ply", rotate_columns=False)

def test_down_sampling_method(file_name="test_nou1.ply"):
    import time
    points_a, colors_a = import_point_cloud_from_ply(file_name)
    start = time.time()
    indices = down_sample_point_cloud(points_a)
    print("Downsampling took: ", (time.time() - start))
    export_numpy_array_to_ply(points_a[indices], colors_a[indices], f"downsampled_{file_name}", rotate_columns=False)

    inverted = np.invert(indices)
    export_numpy_array_to_ply(points_a[inverted], colors_a[inverted], f"what_removed_{file_name}", rotate_columns=False)


if __name__ == "__main__":
    # test_removal_centroid("outliers.ply", cutoff=1)
    # test_removal_with_radius("outliers.ply",radius=0.05)
    # test_icp()
    # test_open3d("test_nou1.ply", "test_nou2.ply")
    # test_remove_outliers(file_name="outliers.ply", nb_neighbours=30)

    # test_icp_point_to_plane()
    test_down_sampling_method()
