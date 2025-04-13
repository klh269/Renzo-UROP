"""
Testing different ways to account for covariance of errors in DTW.
"""
import numpy as np
import matplotlib.pyplot as plt
from utils_analysis.dtw_utils import dtw

num_samples = 100

def get_samples(x, num_samples=num_samples):
    """
    Create toy samples with sin(x) + ax for testing DTW.
    """
    x = np.tile(x, (num_samples, 1))
    a = np.random.normal(0, 0.5, size=num_samples)
    a = np.tile(a, (50, 1)).T
    samples = np.sin(2 * np.pi * x) + a * x

    return samples


def get_data(x):
    """
    Create toy data with triangular waves.
    """
    data = 2 * np.abs( (np.mod(x - 0.25, 1)) - 0.5 ) - 0.5
    return data


if __name__ == "__main__":
    x = np.linspace(0, 5, num_samples)
    Vobs = get_data(x)
    Vbar = np.sin(2 * np.pi * x)
    Vbar_samp = get_samples(x)

    # Plot the data and samples
    # fig, ax = plt.subplots()
    # ax.plot(x, Vobs, label='Data', color='black')
    # ax.plot(x, Vbar, label='Samples', color='blue')

    # fig.savefig("/mnt/users/koe/test.png", bbox_inches='tight', dpi=300)
    # plt.close(fig)
