import numpy as np
import math
from scipy.integrate import cumtrapz


def get_matrix_median(matrix):
    lines = len(matrix)
    columns = len(matrix[0])
    matrix_transposed = np.array(matrix).transpose()
    for j in range(columns):
        matrix_transposed[j].sort()

    matrix = matrix_transposed.transpose()

    return matrix[(int)(lines / 2)]


def get_matrix_average(matrix):
    lines = len(matrix)
    columns = len(matrix[0])
    sum_vector = get_matrix_sum_by_columns(matrix)
    for i in range(columns):
        sum_vector[i] = sum_vector[i] / lines

    return sum_vector


def get_matrix_sum_by_columns(matrix):
    lines = len(matrix)
    columns = len(matrix[0])
    sum_vector = [0, 0, 0]
    for i in range(lines):
        for j in range(columns):
            sum_vector[j] = sum_vector[j] + matrix[i][j]

    return sum_vector


def get_trapz_integral_by_time(matrix):
    """ Matrix has a number of 4 columns [x,y,z,timestamp]
        Timestamp is given in milliseconds, but the angular velocity is in Rad/seconds.
        So a division by 1000 is applied in order to convert milliseconds to seconds.
        Returns:
            A list containing the trapezoidal sum of the matrix by columns.
    """
    array = np.array(matrix)
    dx = array[:, 3] / 1000
    first_integral_vector = [np.trapz(array[:, 0], x=dx),
                             np.trapz(array[:, 1], x=dx),
                             np.trapz(array[:, 2], x=dx)]
    return first_integral_vector


def get_double_trapz_integral_by_time(matrix):
    """ Matrix has a number of 4 columns [x,y,z,timestamp]
        Timestamp is given in milliseconds, but the angular velocity is in Rad/seconds.
        So a division by 1000 is applied in order to convert miliseconds to seconds.
        It computes the double integral of acceleration to return position.
        Returns:
            A list containing the trapezoidal sum of the matrix by columns.
    """
    array = np.array(matrix)
    dx = array[:, 3] / 1000

    # compute the first integral array
    first_integral_x = cumtrapz(array[:, 0], dx)
    first_integral_y = cumtrapz(array[:, 1], dx)
    first_integral_z = cumtrapz(array[:, 2], dx)

    # remove last element from array
    dx = dx[:-1]

    # compute the double integral result
    double_integral_x = np.trapz(first_integral_x, x=dx)
    double_integral_y = np.trapz(first_integral_y, x=dx)
    double_integral_z = np.trapz(first_integral_z, x=dx)

    return [double_integral_x,
            double_integral_y,
            double_integral_z]


def create_rotation_matrix(gyro_data):
    mat = np.empty((3, 3), dtype=np.float32)
    # in realsense camera
    # x axis is used for pitch
    # y axis is used for yaw
    # z axis is used for roll

    # in rotation matrix
    # alpha is for roll
    # beta is for yaw
    # gamma is for pitch

    # Matrix computed based on: https://en.wikipedia.org/wiki/Rotation_matrix
    cos_alpha = math.cos(gyro_data[2])
    sin_alpha = math.sin(gyro_data[2])
    cos_beta = math.cos(gyro_data[1])
    sin_beta = math.sin(gyro_data[1])
    cos_gamma = math.cos(gyro_data[0])
    sin_gamma = math.sin(gyro_data[0])

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


def create_transformation_matrix(rotation_matrix, translation):
    # convert to numpy array
    rotation = np.array(rotation_matrix)
    translation = np.array(translation)

    # initialize transformation matrix as Identity
    transf_mat = np.eye(4, 4, dtype=np.float32)

    # copy rotation matrix in transformation matrix
    transf_mat[:3, :3] = rotation

    # copy the translation on the last column
    transf_mat[:3, 3] = translation

    return transf_mat


def get_indexes_of_valid_points(array):
    """
    Returns a tuple with indexes only for points that have valid depth (depth>0).
    First it transpose the nx3 array to get only the depth array.
    After that it will return the indexes based on the condition (depth>0).
    Args:
        array: (n,3) array containing coordinates of point in xyz.
    Returns:
        A tuple containing valid point indexes.
    """

    # keep only elements that have depth greater than 0
    valid_points = array[:, 2] != 0
    close_points = array[:, 2] <= 3
    return valid_points & close_points


def remove_gravity_from_accel_data(accel_data_array, gyro_data_array, accel_state):
    """
    It rotates the accelerometer state array with each gyro data to get the orientation.
    Then it subtracts the result from each accelerometer data.
    Args:
        accel_data_array: multiple accelerometer data from different frames.
        gyro_data_array: corresponding gyro data to the same accelerometer data array.
        accel_state: the previous state of the accelerometer.
    Returns:
        adjusted_accel_data: an array with the xyz values of accelerometer data and timestamp associated.
    """
    adjusted_accel_data = []
    # replace timestamp with 1 to be in homogeneous coordinates then convert to numpy array
    accel_state[3] = 1
    accel_state_array = np.array(accel_state)

    for i in range(len(accel_data_array)):
        # create the rotation matrix from a gyro data sample
        rotation_matrix = create_rotation_matrix(gyro_data_array[i])
        # no translation needed to compute the transformation matrix
        transf_matrix = create_transformation_matrix(rotation_matrix, [0, 0, 0])
        # rotate the accelerometer state by data provided from the gyroscope
        current_rotated = accel_state_array @ transf_matrix

        current_accel_data = np.array(accel_data_array[i])
        # remove the gravity from x,y,z axes but keep the timestamp
        current_accel_data[:3] -= current_rotated[:3]
        adjusted_accel_data.append(current_accel_data)

    return adjusted_accel_data
