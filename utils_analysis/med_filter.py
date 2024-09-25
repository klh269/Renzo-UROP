# (C) 2024 Enoch Ko.
"""
Simple median filter for fitting RCs (now tested in toy model).
"""
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

def med_filter(rad, arr, size:int, make_plots:bool=False, file_name:str=""):
    arr_filtered = median_filter(arr, size=size, mode='nearest', axes=2)
    residuals = arr - arr_filtered

    if make_plots:
        fig1, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax0.set_title("Simple median filter")
        ax0.set_ylabel("Normalized velocities")

        colours = [ 'red', 'k' ]
        labels = [ "Vbar", "Vobs" ]
        for j in range(2):
            ax0.scatter(rad, arr[0][j], color=colours[j], alpha=0.3)
            ax0.plot(rad, arr_filtered[0][j], color=colours[j], label=labels[j])

        ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
        ax0.grid()

        ax1.set_ylabel("Residuals")
        for j in range(2):
            ax1.scatter(rad[1:], residuals[0][j][1:], color=colours[j], alpha=0.3)
            ax1.plot(rad[1:], residuals[0][j][1:], color=colours[j], label=labels[j])

        ax1.grid()

        plt.subplots_adjust(hspace=0.05)
        fig1.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()

    return arr_filtered, residuals
