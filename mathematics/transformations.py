import numpy as np
import time
from export.ply import export_numpy_array_to_ply
import matplotlib.pyplot as plt
from statistics import mode

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

    plot_time_and_frequency(array, PSD, freq)

    # removes all frequencies above the median value
    median_value = np.median(PSD)
    indices = PSD < median_value
    PSD_attenuated = PSD * indices
    attenuated_fft = fft * indices

    # Compute the inverse of the n-point DFT for real input.
    attenuated_array = np.fft.irfft(attenuated_fft, n=n)

    plot_time_and_frequency(attenuated_array, PSD_attenuated, freq)

    return attenuated_array
