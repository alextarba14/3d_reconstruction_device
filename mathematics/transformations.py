import numpy as np

def apply_transform(pointcloud, matrix):
    i = 0
    length = len(pointcloud)
    while i < length-230400:
        for j in range(i,i+230400):
            pointcloud[j]=pointcloud[j]*matrix[i]
        i = i + 230400


def apply_transformations(pointclouds, transf_matrices):
    index = len(transf_matrices) - 1
    while index > 0:
        current_transf_matrix = transf_matrices[index]
        inverse_transf_matrix = np.linalg.inv(current_transf_matrix)


        index = index - 1