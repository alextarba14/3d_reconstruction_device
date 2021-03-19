import numpy as np


def apply_transform(pointcloud, matrix):
    i = 0
    length = len(pointcloud)
    while i < length - 230400:
        for j in range(i, i + 230400):
            pointcloud[j] = pointcloud[j] * matrix[i]
        i = i + 230400


def apply_transformations(pointclouds, transf_matrices, length, matrix_decrement):
    # number of transformation matrices: length + 1 - first one is empty
    # a matrix has 4 lines => no_lines=  4*(length+1)
    matrix_index = len(transf_matrices)

    # initial pointcloud + length
    pointcloud_index = len(pointclouds)
    pointcloud_length = (int)(pointcloud_index / (length + 1))

    index = length
    while index > 0:
        current_transf_matrix = transf_matrices[matrix_index - matrix_decrement:matrix_index]
        inverse_transf_matrix = np.linalg.inv(current_transf_matrix)

        current_pointcloud = pointclouds[pointcloud_index - pointcloud_length:pointcloud_index]
        # previous_pointcloud = pointclouds[pointcloud_index - 2 * pointcloud_length:pointcloud_index - pointcloud_length]

        # append 1 at the end
        ones = np.ones((pointcloud_length, 1), dtype=np.float32)
        current_pointcloud = np.append(current_pointcloud, ones, axis=1)

        # decrement indexes
        index = index - 1
        matrix_index = matrix_index - matrix_decrement
        pointcloud_index = pointcloud_index - pointcloud_length
