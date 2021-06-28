import numpy as np
import math


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


def create_transformation_matrix_from_rotation(rotation_matrix):
    """
    It creates a transformation matrix from a rotation matrix.
    """
    # convert to numpy array
    rotation = np.array(rotation_matrix)

    # initialize transformation matrix as Identity
    transf_mat = np.eye(4, 4, dtype=np.float32)

    # copy rotation matrix in transformation matrix
    transf_mat[:3, :3] = rotation

    return transf_mat


def get_indexes_of_valid_points(point_cloud):
    """
    Returns a tuple with indexes only for points that have valid depth (depth>0)
    and are not too far away  (depth< 3) .
    Args:
        point_cloud: (n,3) array containing coordinates of point in xyz.
    Returns:
        A tuple containing valid point indexes.
    """

    # keep only elements that have depth greater than 0
    valid_points = point_cloud[:, 2] > 0
    # keep only elements that are not too far away
    close_points = point_cloud[:, 2] <= 3.5

    # result keep points in (0,2.5] meters
    return valid_points & close_points
