import numpy as np

from mathematics.matrix import get_double_trapz_integral_by_time
from mathematics.transformations import remove_noise_from_matrix


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

    noiseless_gyro = remove_noise_from_matrix(gyro_data, GYRO_RATE)
    noiseless_accel = remove_noise_from_matrix(accel_data, ACCEL_RATE)
    print("X:", get_double_trapz_integral_by_time(noiseless_accel))
