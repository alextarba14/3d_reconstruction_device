import numpy as np
import time

from plot.plot import plot_time_and_frequency


def apply_transform(pointcloud, matrix):
    i = 0
    length = len(pointcloud)
    while i < length - 230400:
        for j in range(i, i + 230400):
            pointcloud[j] = pointcloud[j] * matrix[i]
        i = i + 230400


def apply_transformations(pointclouds, transf_matrices):
    start_time_function = time.time()
    print("apply_transformations started at: ", start_time_function)
    # number of transformation matrices: length + 1 - first one is empty
    # a matrix has 4 lines => no_lines=  4*(length+1)

    index = len(pointclouds) - 1
    while index > 0:
        start_time_index = time.time()
        print(f'Started index: {index} at: ', start_time_index)
        current_pointcloud = pointclouds[index]
        pc_length = len(current_pointcloud)

        # append 1 at the end
        ones = np.ones((pc_length, 1), dtype=np.float32)
        current_pointcloud = np.append(current_pointcloud, ones, axis=1)

        # apply transformations for each pointcloud
        j = index
        while j > 0:
            start_time = time.time()
            print(f'Started j: {j} at: ', start_time)
            current_transf_matrix = transf_matrices[j]

            # for i in range(pc_length):
            #     # multiply Tr*p = p' (obtaining points based referenced at previous system information)
            #     current_pointcloud[i] = current_transf_matrix.dot(current_pointcloud[i])
            # or
            # current_pointcloud = np.einsum("ij,kj->ik", current_pointcloud, current_transf_matrix)
            current_pointcloud = current_pointcloud.dot(current_transf_matrix.T)

            j = j - 1
            print(f'Stopped j: {j} after: ', time.time() - start_time)

        # removing the last column since it was added to perform dot product between vector[1x(3+1)] and transform matrix[4x4]
        current_pointcloud = np.delete(current_pointcloud, 3, axis=1)
        # update the pointcloud in memory
        pointclouds[index] = current_pointcloud

        print(f'Stopped index: {index} after: ', time.time() - start_time_index)

        # decrement indexes
        index = index - 1

    print("Ended after: ", time.time() - start_time_function)
    return pointclouds


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
    freq = np.fft.rfftfreq(n, d=(1. / sampling_rate))

    # plot_time_and_frequency(array, PSD, freq)

    # removes all frequencies above the median value
    median_value = np.median(PSD)
    indices = PSD < median_value
    PSD_attenuated = PSD * indices
    attenuated_fft = fft * indices

    # Compute the inverse of the n-point DFT for real input.
    attenuated_array = np.fft.irfft(attenuated_fft, n=n)

    # plot_time_and_frequency(attenuated_array, PSD_attenuated, freq)

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


def filtered_kalman(measurement):
    """
        Compute the Kalman filter over the given measurement
        Args:
            measurement: a 1 dimensional numpy array or a list
        Returns:
            Filtered data
        ---------------------------------------------------------------
        Predict next state:
            updated[i] = F* updated[i-1] + B * u
        Predict next covariance:
            P[i] = F*P[i-1]*F_T + Q
        Compute the Kalman gain:
            K = P[i] * H_T/(H*P[i]*H_T + R)
        Update the state estimate:
            updated[i] = updated[i] + K*(measurement[i]-H*updated[i])
        Update covariance estimation:
            P[i] = (I-K*H)*P[i]
        -----------------------------------------------------------------
        F=1;
        B=0;    no control input
        H=1;    only one observable
        Q=1e-9; process noise covariance
        R;      observation noise covariance
        I=1;    identity
        -----------------------------------------------------------------
        updated[i] = updated[i-1]
        P[i] = P[i-1] + Q
        K = P[i]/(P[i] + R)
        updated[i] = updated[i] + K*(measurement[i]-updated[i])
        P[i] = (1-K)*P[i]
        -----------------------------------------------------------------
        K = P[i]/(P[i] + R)
        updated[i] = updated[i-1] + K*(measurement[i]-updated[i-1])
        P[i] = (1-K)*P[i] + Q
    """
    length = measurement.size
    updated = np.zeros(length)
    P = 1
    Q = 1e-9
    R = 1e-8
    previous = 0

    for i in range(0, length):
        K = P / (P + R)
        updated[i] = previous + K * (measurement[i] - previous)
        P = (1 - K) * P + Q
        previous = updated[i]

    return updated


def get_kalman_filtered_data(data_list):
    """
    Filter the data on all 3 axis using the Kalman filter
    and return the data in the list format having the timestamp on the 4th column.
    """
    array = np.array(data_list)
    x_values = array[:, 0]
    y_values = array[:, 1]
    z_values = array[:, 2]

    filtered_x = filtered_kalman(x_values)
    filtered_y = filtered_kalman(y_values)
    filtered_z = filtered_kalman(z_values)

    # add the timestamp to the result -> np_array[:,3]
    result = np.array([filtered_x, filtered_y, filtered_z, array[:, 3]])
    return result.transpose().tolist()
