import matplotlib.pyplot as plt
import numpy as np


def save_data_as_plot_image(data_array, title="Plot"):
    array = np.array(data_array)
    x = array[:, 0]
    y = array[:, 1]
    z = array[:, 2]
    time = (array[:, 3] - array[0][3]) / 1000
    plt.plot(time, x)
    plt.plot(time, y)
    plt.plot(time, z)
    plt.legend(['x', 'y', 'z'])
    plt.title(title)
    plt.savefig(title + ".png")
    plt.close()
