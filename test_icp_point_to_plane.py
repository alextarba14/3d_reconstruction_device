import time
import numpy as np

from icp_point_to_plane.icp_point_to_plane import icp
from input_output.ply import import_point_cloud_from_ply, export_numpy_array_to_ply
from processing.process import remove_statistical_outliers


def remove_outliers_from_point_clouds(vertices, colors):
    print("Removing outliers...")
    for i in range(len(vertices)):
        valid_indices = remove_statistical_outliers(vertices[i], nb_neighbours=50, std_ratio=1)
        vertices[i] = vertices[i][valid_indices]
        colors[i] = colors[i][valid_indices]
    print("Outliers removed.")
    return vertices, colors


if __name__ == "__main__":
    vertices = []
    colors = []
    for i in range(13):
        points, color = import_point_cloud_from_ply(f'./demo/raw{i}.ply')
        vertices.append(points)
        colors.append(color)

    start_time = time.time()
    # Remove outliers from each point cloud
    vertices, colors = remove_outliers_from_point_clouds(vertices, colors)
    # create main_color array to match main_point_cloud
    main_color = np.vstack(colors)

    transf_matrices = []
    transf_matrices.append(np.eye(4, 4))
    # get first information about first point cloud
    X_src = vertices[0].copy()
    for i in range(1, len(vertices)):
        X_dst = vertices[i].copy()
        # get transformation matrix that match destination over source
        Tr = icp(X_src, X_dst, correspondences=1000, neighbors=10, max_iterations=50)
        transf_matrices.append(Tr)

        # update the current source
        X_src = vertices[i].copy()

    print("Applying transformation matrices...")
    main_pc = vertices[0]
    index = 1
    length = len(vertices)
    # apply transformation to align each point cloud over the first point cloud
    while index < length:
        current_pc = vertices[index]

        # append a column of ones to transform the current point cloud
        # in homogeneous coordinates to do matrix multiplication
        ones = np.ones((len(current_pc), 1))
        current_pc = np.append(current_pc, ones, axis=1)

        # transpose current point cloud to allow matrix multiplication
        current_pc = current_pc.T

        # apply all transformation matrices to current point cloud
        j = 1
        while j <= index:
            current_transf = transf_matrices[j]
            current_pc = current_transf @ current_pc
            j = j + 1

        # transpose current point cloud back to have info in xyz coordinates
        current_pc = current_pc.T
        # remove column used to allow matrix multiplication
        current_pc = np.delete(current_pc, 3, axis=1)

        # append current point cloud to the main point cloud
        main_pc = np.vstack((main_pc, current_pc))
        # move to the next point cloud
        index = index + 1
    print("Process took: ", (time.time() - start_time))
    export_numpy_array_to_ply(main_pc, main_color,
                              f"./demo/result_1000_10nei_raw_time_{(time.time() - start_time)}.ply",
                              rotate_columns=False)
