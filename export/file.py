import numpy as np

from mathematics.matrix import get_double_trapz_integral_by_time
from mathematics.transformations import remove_noise_from_matrix, filtered_kalman
from plot.plot import plot_difference


def export_list_to_file(file_name, list):
    np_array = np.array(list)
    np.savetxt(file_name, np_array, delimiter=',')


def import_numpy_array_from_file(file_name):
    return np.genfromtxt(file_name, delimiter=',')


if __name__ == '__main__':
    ACCEL_RATE = 250
    GYRO_RATE = 200
    gyro_data = import_numpy_array_from_file("../Gyro_data_raw")
    accel_data = import_numpy_array_from_file("../Accel_data_raw")

    x_values = gyro_data[:, 0]
    y_values = gyro_data[:, 1]
    z_values = gyro_data[:, 2]
    timestamp = gyro_data[:, 3]

    # compute values on x axis
    updated = filtered_kalman(x_values)
    plot_difference(x_values, updated, timestamp, "X_difference")

    updated = filtered_kalman(y_values)
    plot_difference(y_values, updated, timestamp, "Y_difference")

    updated = filtered_kalman(z_values)
    plot_difference(z_values, updated, timestamp, "Z_difference")

    noiseless_gyro = remove_noise_from_matrix(gyro_data, GYRO_RATE)
    # noiseless_accel = remove_noise_from_matrix(accel_data, ACCEL_RATE)

    np_array_gyro = np.array(noiseless_gyro)
    plot_difference(x_values, np_array_gyro[:,0], timestamp, "X_FFT_difference")
    plot_difference(y_values, np_array_gyro[:,1], timestamp, "Y_FFT_difference")
    plot_difference(z_values, np_array_gyro[:,2], timestamp, "Z_FFT_difference")

    x_values = accel_data[:, 0]
    y_values = accel_data[:, 1]
    z_values = accel_data[:, 2]
    timestamp = accel_data[:, 3]

    # compute values on x axis
    updated = filtered_kalman(x_values)
    plot_difference(x_values, updated, timestamp, "X_accel_difference")

    updated = filtered_kalman(y_values)
    plot_difference(y_values, updated, timestamp, "Y_accel_difference")

    updated = filtered_kalman(z_values)
    plot_difference(z_values, updated, timestamp, "Z_accel_difference")

    noiseless_accel = remove_noise_from_matrix(accel_data, ACCEL_RATE)

    np_array_accel = np.array(noiseless_gyro)
    plot_difference(x_values, np_array_accel[:,0], timestamp, "X_FFT_accel_difference")
    plot_difference(y_values, np_array_accel[:,1], timestamp, "Y_FFT_accel_difference")
    plot_difference(z_values, np_array_accel[:,2], timestamp, "Z_FFT_accel_difference")