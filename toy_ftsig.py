#!/usr/bin/env python
# (C) 2024 Enoch Ko.
"""
Retreive numpy arrays of toy_model iterations over noise and sampling rate
and plot 2D histograms for feature classification with DTW and Pearson rho respectively.
"""
import numpy as np
import matplotlib.pyplot as plt

#  Determine the file location for array extraction
def fileloc(use_MF:bool, use_GP:bool):
    loc = "/mnt/users/koe/plots/toy_model/2Dsig/"
    if use_MF and use_GP: raise Exception("GP and MF cannot both be True!")
    elif use_MF: loc += "use_MF/"
    elif use_GP: loc += "use_GP/"
    return loc

# Analysis switches.
use_MF = True
use_GP = False
fname = fileloc(use_MF, use_GP)

# Array of sampling rates.
samp_rates = np.linspace(30, 300, 28, endpoint=True, dtype=int)

# Make 2D feature significance histograms.
signames = [ "dtw_ftsig", "rad_ftsig" ]
for sn in range(2):
    # Extract numpy arrays and combine into one big plottable 2D array.
    ft_significance = []
    for smp in range(11):
        num_samp = samp_rates[smp]
        ftsig = np.load(f"{fname}{signames[sn]}/num_samples={num_samp}.npy")
        ft_significance.append(ftsig)
    
    ft_significance = np.array(ft_significance)
    ft_significance = np.nan_to_num(ft_significance, nan=np.inf, posinf=np.inf)
    print("Array:")
    print(np.shape(ft_significance))

    if sn == 0:
        ft_significance = np.clip( ft_significance, 0., 10 )
    else:
        ft_significance = np.clip( ft_significance, 0., 100 )
        ft_significance = np.array(ft_significance) / 100
    
    plt.title("Feature significance")
    plt.imshow(ft_significance)
    plt.xlabel("Noise / ft height ratio")
    plt.ylabel("Sampling rate (# samples / unit radius)")
    plt.colorbar()
    plt.savefig(f"{fname}{signames[sn]}.png")
    plt.close()
