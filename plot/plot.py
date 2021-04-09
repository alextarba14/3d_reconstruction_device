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


def plot_time_and_frequency(time_array, PSD, frequency_array):
    """
    Plot both the time array and the Power Spectral Density from the Discrete FFT.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(np.arange(0, len(time_array)), time_array, color='r', linewidth=1.5, linestyle='-.')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')

    ax2.plot(frequency_array, PSD, color='c', linewidth=2)
    ax2.set_xlabel('Frequency[Hz]')
    ax2.set_ylabel('Amplitude')

    # use tight_layout to add space between subplots
    fig.tight_layout()
    plt.show()
    plt.close()


def plot_difference(first_array, second_array, timestamp, first_label="Raw_data", second_label="Transformed_data", title="Difference"):
    time = (timestamp - timestamp[0]) / 1000
    plt.plot(time, first_array)
    plt.plot(time, second_array)
    plt.legend([first_label, second_label])
    plt.title(title)
    plt.savefig(title + ".png")
    plt.close()
