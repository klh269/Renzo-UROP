#!/usr/bin/env python
# (C) 2024 Enoch Ko.
"""
Retreive numpy arrays of toy_model iterations over noise and sampling rate
and plot 2D histograms for feature classification with DTW and Pearson rho respectively.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def get_significance(corr1, corr2, rhos:bool=True):
    if rhos == True:
        sigma1 = (corr1[0,:,2] - corr1[0,:,0]) / 2
        sigma2 = (corr2[0,:,2] - corr2[0,:,0]) / 2
        ftsig = abs(( corr2[0,:,1] - corr1[0,:,1] )) / np.sqrt(sigma1**2 + sigma2**2)
    else:
        sigma1 = (corr1[2] - corr1[0]) / 2
        sigma2 = (corr2[2] - corr2[0]) / 2
        ftsig = abs(( corr2[1] - corr1[1] )) / np.sqrt(sigma1**2 + sigma2**2)
    return ftsig

#  Determine the file location for array extraction
def fileloc(use_MF:bool, use_GP:bool):
    loc = "/mnt/users/koe/plots/toy_model/2Dsig/"
    if use_MF and use_GP: raise Exception("GP and MF cannot both be True!")
    elif use_MF: loc += "use_MF/"
    elif use_GP: loc += "use_GP/"
    return loc

def get_SPARC(ft_height:float = 0.2):
    # Get galaxy data from table1.
    file = "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt"

    SPARC_c = [ "Galaxy", "T", "D", "e_D", "f_D", "Inc",
            "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
            "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
    table = pd.read_fwf(file, skiprows=98, names=SPARC_c)
    galaxy_count = len(table["Galaxy"])

    columns = [ "Rad", "Vobs", "errV", "Vgas",
                "Vdisk", "Vbul", "SBdisk", "SBbul" ]
    
    gal_list, SPARC_rates, SPARC_noise = [], [], []
    for i in range(galaxy_count):
        g = table["Galaxy"][i]
        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)

        Vmax = max(data["Vobs"])
        noise_ratio = (np.mean(data["errV"]) / Vmax) / ft_height
        r = data["Rad"]
        sampling_rate = len(r) / max(r)

        if noise_ratio > 0.5:
            continue
        elif sampling_rate < 3.0:
            continue
        else:
            SPARC_noise.append( noise_ratio )
            SPARC_rates.append( sampling_rate )
            gal_list.append(g)
    
    return gal_list, SPARC_rates, SPARC_noise

# Analysis switches.
use_MF = False
use_GP = False
fname = fileloc(use_MF, use_GP)

# Array of sampling rates and noise.
noise_arr = np.linspace(0.0, 10.0, 51, endpoint=True) if use_GP else np.linspace(0.0, 10.0, 101, endpoint=True)
samp_rates = np.linspace(30, 200, 18, endpoint=True, dtype=int)

# Get numbers from SPARC dataset.
SPARC_ft = 0.2
gal_list, SPARC_rates, SPARC_noise = get_SPARC( ft_height=SPARC_ft )
print(f"Number of SPARC galaxies shown in the plot = {len(gal_list)} / 175")

# Make 2D feature significance histograms.
signames = [ "dtw_ftsig", "rad_ftsig" ]
titles = [ "DTW", r"Pearson $\rho$" ]
arrays = [ ["dtw_costs/", "Xft_costs/"], ["rad_pearsons/", "rad_Xft_pearsons/"] ]
for sn in range(2):
    # Extract numpy arrays and combine into one big plottable 2D array.
    ft_significance = []
    for smp in range(18):
        num_samp = samp_rates[smp]
        # ftsig = np.load(f"{fname}{signames[sn]}/num_samples={num_samp}.npy")
        # ft_significance.append(ftsig)

        corr1 = np.load(f"{fname}{arrays[sn][0]}num_samples={num_samp}.npy")
        corr2 = np.load(f"{fname}{arrays[sn][1]}num_samples={num_samp}.npy")
        ftsig = get_significance(corr1, corr2, rhos=sn)
        ft_significance.append(ftsig)
    
    ft_significance = np.array(ft_significance)
    ft_significance = np.nan_to_num(ft_significance, nan=np.inf, posinf=np.inf)
    ft_significance = gaussian_filter(ft_significance, 0.8)
    ft_significance = np.clip( ft_significance, 1.0, 10.0 )

    if sn == 1: noise_arr = noise_arr[1:]
    extent = [ 0.0, 0.5, 3.0, 20.0 ]
    
    plt.title("Feature significance: " + titles[sn])
    plt.imshow( ft_significance, interpolation='none', norm='linear', cmap='viridis',
                origin='lower', extent=extent, aspect='auto' )
    plt.xlabel("Noise / ft height ratio")
    plt.ylabel("Sampling rate (# samples / kpc)")
    plt.colorbar()
    contours = plt.contour( noise_arr/20.0, samp_rates/10.0, ft_significance, 9,
                            origin='lower', extent=extent, colors='black' )
    plt.clabel(contours, inline=1, fontsize=10)
    plt.scatter(SPARC_noise, SPARC_rates, color='red', label="SPARC galaxies (assuming ft height = 0.2)")
    for pt in range(len(gal_list)):
        if SPARC_noise[pt] < 0.2:
            plt.annotate(gal_list[pt], (SPARC_noise[pt], SPARC_rates[pt] + 0.2), color='red')
    plt.legend()
    plt.savefig(f"{fname}{signames[sn]}.png")
    plt.close()
