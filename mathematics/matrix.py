import numpy as np
import math


def get_matrix_average(lines, columns, matrix):
    sum_vector = [0, 0, 0]
    for i in range(lines):
        for j in range(columns):
            sum_vector[j] = sum_vector[j] + matrix[i][j]

    for i in range(columns):
        sum_vector[i] = sum_vector[i] / lines

    return sum_vector


def create_rotation_matrix(gyro_data):
    mat = np.empty((3, 3), dtype=np.float32)
    # in realsense camera
    # x axis is used for pitch
    # y axis is used for yaw
    # z axis is used for roll

    # in rotation matrix
    # alpha is for yaw
    # beta is for pitch
    # gamma is for roll

    # Matrix computed based on: https://en.wikipedia.org/wiki/Rotation_matrix
    cos_alpha = math.cos(gyro_data[1])
    sin_alpha = math.sin(gyro_data[1])
    cos_beta = math.cos(gyro_data[0])
    sin_beta = math.sin(gyro_data[0])
    cos_gamma = math.cos(gyro_data[2])
    sin_gamma = math.sin(gyro_data[2])

    mat[0][0] = cos_alpha * cos_beta
    mat[0][1] = cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma
    mat[0][2] = cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma

    mat[1][0] = sin_alpha * cos_beta
    mat[1][1] = sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma
    mat[1][2] = sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma

    mat[2][0] = -sin_beta
    mat[2][1] = cos_beta * sin_gamma
    mat[2][2] = cos_beta * cos_gamma

    return mat


def create_transformation_matrix(rotation_matrix, accel_data):
    transf_mat = np.empty((4, 4), dtype=np.float32)
    # copy rotation matrix in transformation matrix
    for i in range(3):
        for j in range(3):
            transf_mat[i][j] = rotation_matrix[i][j]

    # append translation as the last column
    for i in range(3):
        transf_mat[i][3] = accel_data[i]

    # put 1 in the bottom right corner
    transf_mat[3][3] = 1

    return transf_mat
