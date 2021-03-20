import numpy as np
import time
from export.ply import export_numpy_array_to_ply


def apply_transform(pointcloud, matrix):
    i = 0
    length = len(pointcloud)
    while i < length - 230400:
        for j in range(i, i + 230400):
            pointcloud[j] = pointcloud[j] * matrix[i]
        i = i + 230400


def apply_transformations(pointclouds, transf_matrices, length, matrix_decrement):
    start_time_function = time.time()
    print("Started at: ", start_time_function)
    # number of transformation matrices: length + 1 - first one is empty
    # a matrix has 4 lines => no_lines=  4*(length+1)

    # initial pointcloud + length
    pointcloud_index = len(pointclouds)
    pointcloud_length = (int)(pointcloud_index / (length + 1))

    index = length
    while index > 0:
        start_time_index = time.time()
        print(f'Started index: {index} at: ', start_time_index)
        current_pointcloud = pointclouds[pointcloud_index - pointcloud_length:pointcloud_index]

        # append 1 at the end
        ones = np.ones((pointcloud_length, 1), dtype=np.float32)
        current_pointcloud = np.append(current_pointcloud, ones, axis=1)

        # apply transformations for each pointcloud
        j = index
        matrix_index = (index + 1) * matrix_decrement
        while j > 0:
            start_time = time.time()
            print(f'Started j: {j} at: ', start_time)
            current_transf_matrix = transf_matrices[matrix_index - matrix_decrement:matrix_index]
            inverse_transf_matrix = np.linalg.inv(current_transf_matrix)
            count = 0
            for i in range(len(current_pointcloud)):
                if current_pointcloud[i][0] != 0 and current_pointcloud[i][1] != 0 and current_pointcloud[i][2] != 0:
                    # avoid multiplication by zero
                    current_pointcloud[i] = current_pointcloud[i].dot(inverse_transf_matrix)

            j = j - 1
            matrix_index = matrix_index - matrix_decrement
            print(f'Stopped j: {j} after: ', time.time() - start_time)

        current_pointcloud = np.delete(current_pointcloud, 3, axis=1)
        # update the pointcloud in memory
        pointclouds[pointcloud_index - pointcloud_length:pointcloud_index] = current_pointcloud

        # save the transformed pointcloud
        file_name = f'transformed{index}.ply'
        export_numpy_array_to_ply(current_pointcloud, file_name=file_name)

        print(f'Stopped index: {index} after: ', time.time() - start_time_index)
        # decrement indexes
        index = index - 1
        matrix_index = matrix_index - matrix_decrement
        pointcloud_index = pointcloud_index - pointcloud_length

    print("Stopped after: ", time.time() - start_time_function)
    exit(1)
    return pointclouds
