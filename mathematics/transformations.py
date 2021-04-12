import numpy as np
import time
from mathematics.kalman_filter import KalmanFilter


def apply_transformations(pointclouds, transf_matrices):
    """
    Apply transformation to each pointcloud by multiplying with their correspondent transformation matrix.
    P' = P*Tr_matrix
    Each pointcloud has a transformation matrix associated that represents the transformations that has been done
    until the current pointcloud. So there is no need to multiply the pointcloud with it's transformation matrix once more.
    It will multiply each pointcloud with the future transformations and update the pointclouds.
    """
    start_time_function = time.time()
    print("apply_transformations started at: ", start_time_function)

    length = len(pointclouds)
    index = 0
    while index < length:
        start_time_index = time.time()
        print(f'Started index: {index} at: ', start_time_index)
        current_pointcloud = pointclouds[index]
        pointclouds[index] = apply_next_transformations_to_current_pointcloud(current_pointcloud, index,
                                                                              transf_matrices, length)

        print(f'Stopped index: {index} after: ', time.time() - start_time_index)
        index = index + 1

    print("Ended after: ", time.time() - start_time_function)
    return pointclouds


def apply_next_transformations_to_current_pointcloud(current_pointcloud, index, transf_matrices, length):
    """
    Apply all future transformations that have happened after the current pointcloud.
    Multiply current_pointcloud[i] with all transformation matrices from [i+1, length).
    Each transformation matrix is in shape(4,4), but a point from current_pointcloud is in (1,3) shape.
    A column full of ones has been appended to the pointcloud to allow multiplication.
    current_pointcloud(1,4) *  transf_matrix(4,4) = pointcloud(1,4), the result will be in the (x,y,z,1) format
    after multiplications the end column will be removed.
    """
    # return current pointcloud when it is the last pointcloud
    if (index + 1) == length:
        return current_pointcloud

    # append a column full of ones at the end
    pc_length = len(current_pointcloud)
    ones = np.ones((pc_length, 1), dtype=np.float32)
    current_pointcloud = np.append(current_pointcloud, ones, axis=1)

    # apply transformations for each pointcloud
    j = index + 1
    while j < length:
        start_time = time.time()
        print(f'Started j: {j} at: ', start_time)
        current_transf_matrix = transf_matrices[j]

        # for i in range(pc_length):
        #     # multiply Tr*p = p' (obtaining points based referenced at previous system information)
        #     current_pointcloud[i] = current_transf_matrix.dot(current_pointcloud[i])
        # or
        # current_pointcloud = np.einsum("ij,kj->ik", current_pointcloud, current_transf_matrix)
        # or
        # current_pointcloud = current_pointcloud.dot(current_transf_matrix)
        current_pointcloud = current_pointcloud @ current_transf_matrix
        j = j + 1
        print(f'Stopped j: {j} after: ', time.time() - start_time)

    # removing the last column since it was added to perform dot product between vector[1x(3+1)] and transform matrix[4x4]
    current_pointcloud = np.delete(current_pointcloud, 3, axis=1)
    return current_pointcloud


def apply_transformations_in_reverse_order(pointclouds, transf_matrices):
    """
    Apply transformation to each pointcloud by multiplying with their correspondent transformation matrix.
    P' = P*Tr_matrix
    Each pointcloud has a transformation matrix associated that represents the transformations that has been done
    until from the previous pointcloud to the current pointcloud.
    It will multiply each pointcloud with his previous transformation matrix and update the pointcloud
    """
    start_time_function = time.time()
    print("apply_transformations started at: ", start_time_function)

    length = len(pointclouds)
    index = length - 1
    while index >= 0:
        start_time_index = time.time()
        print(f'Started index: {index} at: ', start_time_index)
        current_pointcloud = pointclouds[index]
        pointclouds[index] = apply_previous_transformations_to_current_pointcloud(current_pointcloud, index,
                                                                                  transf_matrices, length)

        print(f'Stopped index: {index} after: ', time.time() - start_time_index)
        index = index - 1

    print("Ended after: ", time.time() - start_time_function)
    return pointclouds


def apply_previous_transformations_to_current_pointcloud(current_pointcloud, index, transf_matrices, length):
    """
    Apply all future transformations that have happened after the current pointcloud.
    Multiply current_pointcloud[i] with all transformation matrices from [i+1, length).
    Each transformation matrix is in shape(4,4), but a point from current_pointcloud is in (1,3) shape.
    A column full of ones has been appended to the pointcloud to allow multiplication.
    current_pointcloud(1,4) *  transf_matrix(4,4) = pointcloud(1,4), the result will be in the (x,y,z,1) format
    after multiplications the end column will be removed.
    """
    if index == 0:
        return current_pointcloud
    # append a column full of ones at the end
    pc_length = len(current_pointcloud)
    ones = np.ones((pc_length, 1), dtype=np.float32)
    current_pointcloud = np.append(current_pointcloud, ones, axis=1)

    # apply transformations for each pointcloud
    j = index
    while j >= 0:
        start_time = time.time()
        print(f'Started j: {j} at: ', start_time)
        current_transf_matrix = transf_matrices[j]

        # for i in range(pc_length):
        #     # multiply Tr*p = p' (obtaining points based referenced at previous system information)
        #     current_pointcloud[i] = current_transf_matrix.dot(current_pointcloud[i])
        # or
        # current_pointcloud = np.einsum("ij,kj->ik", current_pointcloud, current_transf_matrix)
        # or
        # current_pointcloud = current_pointcloud.dot(current_transf_matrix)
        current_pointcloud = current_pointcloud @ current_transf_matrix
        j = j - 1
        print(f'Stopped j: {j} after: ', time.time() - start_time)

    # removing the last column since it was added to perform dot product between vector[1x(3+1)] and transform matrix[4x4]
    current_pointcloud = np.delete(current_pointcloud, 3, axis=1)
    return current_pointcloud


def remove_noise_from_data(array, sampling_rate):
    """
    It computes the FFT of the array and the Power Spectral Density.
    It removes all the items whose frequency is greater than the median value of PSD.
    Args:
        array: a 1 dimensional numpy array.
        sampling_rate: the sampling rate of the data acquisition.
    Returns:
        A new array without the most frequent items categorized as noise.
    """
    n = len(array)
    # Compute the one-dimensional discrete Fourier Transform for real input.
    fft = np.fft.rfft(array, n)
    # The Power Spectral Density.
    PSD = fft * np.conj(fft) / n
    # Discrete Fourier Transform sample frequencies
    # freq = np.fft.rfftfreq(n, d=(1. / sampling_rate))

    # removes all frequencies above the median value
    median_value = np.median(PSD)
    indices = PSD < median_value
    # PSD_attenuated = PSD * indices
    attenuated_fft = fft * indices

    # Compute the inverse of the n-point DFT for real input.
    attenuated_array = np.fft.irfft(attenuated_fft, n=n)

    return attenuated_array


def remove_noise_from_matrix(array, sampling_rate):
    """
    Removes noise from the list [x,y,z,timestamp] and returns the updated list.
    Args:
        array: a list with data from either gyroscope or accelerometer.
        sampling_rate: refresh rate for data acquisition.
    Returns:
        The list without noise.
    """
    # convert the array list to numpy array  first
    np_array = np.array(array)
    x = np_array[:, 0]
    y = np_array[:, 1]
    z = np_array[:, 2]

    # get updated data by median filter for each axis
    updated_x = remove_noise_from_data(x, sampling_rate)
    updated_y = remove_noise_from_data(y, sampling_rate)
    updated_z = remove_noise_from_data(z, sampling_rate)

    # add the timestamp to the result -> np_array[:,3]
    result = np.array([updated_x, updated_y, updated_z, np_array[:, 3]])
    return result.transpose().tolist()


def get_kalman_filtered_data(data_list, kalman_filter: KalmanFilter):
    """
    Filter the data on all 3 axis using the Kalman filter
    and return the data in the list format having the timestamp on the 4th column.
    """
    array = np.array(data_list)
    x_values = array[:, 0]
    y_values = array[:, 1]
    z_values = array[:, 2]

    filtered_x = kalman_filter.filter_data(x_values)
    filtered_y = kalman_filter.filter_data(y_values)
    filtered_z = kalman_filter.filter_data(z_values)

    # add the timestamp to the result -> np_array[:,3]
    result = np.array([filtered_x, filtered_y, filtered_z, array[:, 3]])
    return result.transpose().tolist()
