#!/usr/bin/env python
# (C) 2024 Enoch Ko.
"""
Retreive numpy arrays of toy_model iterations over noise and sampling rate
and plot 2D histograms for feature classification with DTW and Pearson rho respectively.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from math import floor
from scipy.ndimage import gaussian_filter

# from utils_analysis.little_things import get_things

plt.rcParams.update({'font.size': 13})

"""
Define functions.
"""
#  Determine the file location for array extraction
# def fileloc(bump_FWHM:float, use_MF:bool, use_GP:bool):
#     loc = f"/mnt/users/koe/plots/toy_model/2Dsig/FWHM={bump_FWHM}/"
#     if use_MF and use_GP: raise Exception("GP and MF cannot both be True!")
#     elif use_MF: loc += "use_MF/"
#     elif use_GP: loc += "use_GP/"
#     return loc

def get_SPARC(ft_height:float = 0.1):
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

def SPARC_Vobs_ft():
    SPARC_features = np.load("/mnt/users/koe/gp_fits/SPARC_features.npy", allow_pickle=True).item()
    galaxies = list(SPARC_features.keys())
    galaxy_count = len(galaxies)

    samp_rates, s2n_ratios = [], []

    for i in range(galaxy_count):
        g = galaxies[i]
        _, _, widths = SPARC_features[g]
        samp_rate = np.average(widths)

        g_dict = np.load(f"/mnt/users/koe/SPARC_fullGP/ft_windows/{g}.npy", allow_pickle=True).item()
        sig2noise = g_dict["sig2noise"]

        samp_rates.append( samp_rate )
        s2n_ratios.append( sig2noise )

    # print(f"Average sampling rate in SPARC = {np.mean(samp_rates):.2f}")    # 11.08
    # print(f"Average signal-to-noise ratio in SPARC = {np.mean(s2n_ratios):.2f}")    # 8.28

    return galaxies, samp_rates, s2n_ratios


"""
MAIN CODE.
"""
# Analysis switches.
use_window = False
scat_Vbar = True
# use_THINGS = True
# use_SPARC = True

# bump_FWHM = 15.0
# fname = fileloc(bump_FWHM, use_MF, use_GP)
bump_width = 0.3
if scat_Vbar: fname = f"/mnt/users/koe/plots/mock_data/width={bump_width}/"
else: fname = f"/mnt/users/koe/plots/mock_data_tests/width={bump_width}/"

# Array of sampling rates and noise.
height_arr = np.linspace(20.0, 2.0, 37, endpoint=True)
noise_arr = 20.0 / np.flip(height_arr)

# Define sampling rates (x-axis in 2D histogram, NB. samp_rate goes up to 300 in analyses).
if use_window: samp_rates = np.linspace(20, 200, num=37, endpoint=True, dtype=int) * 2
else: samp_rates = np.linspace(10, 200, num=39, endpoint=True, dtype=int) * 2


# Make 2D feature significance histograms.
if use_window: signames = [ "dtw_ftsig_window", "rad_ftsig_window" ]
else: signames = [ "dtw_ftsig", "rad_ftsig" ]
titles = [ "DTW", r"Pearson $\rho$" ]
arrays = [ ["dtw_costs/", "Xft_costs/"], ["rad_pearsons/", "rad_Xft_pearsons/"] ]
for sn in range(2):
    # Extract numpy arrays and combine into one big plottable 2D array.
    ft_significance = []
    for smp in range(len(samp_rates)):
        num_samp = int(samp_rates[smp] / 2)
        ftsig = np.load(f"{fname}{signames[sn]}/num_samples={num_samp}.npy")
        ft_significance.append(ftsig)
    
    ft_significance = np.fliplr(ft_significance)
    ft_significance = np.nan_to_num(ft_significance, nan=np.inf, posinf=np.inf)
    ft_significance = gaussian_filter(ft_significance, 1.0)

    # ft_significance = np.clip( ft_significance, 1.0, 6.0 )
    # ft_significance = ft_significance[:,:max_idx]

    # if sn == 1 and use_GP: height_arr = height_arr[:-1]
    extent = [ min(height_arr), max(height_arr), min(samp_rates)/10.0, max(samp_rates)/10.0 ]
    
    # plt.title(f"Feature significance: {titles[sn]} (ft width = {bump_FWHM/10})")
    plt.imshow( ft_significance, interpolation='none', norm='linear', cmap='viridis',
                origin='lower', extent=extent, aspect='auto' )
    plt.xlabel(r"Feature-to-noise ratio $h/\epsilon$")
    if sn == 1: 
        plt.ylabel(r"No. of points in feature $N$")
        plt.colorbar()
    else:
        plt.colorbar(label=r"Feature significance $S(N, h/\epsilon)$")

    # yticks = np.array([5, 10, 15, 20, 25, 30])
    # plt.yticks(ticks=yticks, labels=yticks*bump_FWHM/10)

    # if sn == 0 or use_window: c_levels = np.arange(floor(ft_significance.max()))
    # else: c_levels = np.arange(floor(ft_significance.max()), step=2)
    c_levels = [ 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16 ]
    contours = plt.contour( height_arr[::-1], samp_rates/10.0, ft_significance, levels=c_levels,
                            origin='lower', extent=extent, colors='black' )
    plt.clabel(contours, levels=c_levels, inline=1, fontsize=10)

    # plt.scatter(np.array(SPARC_noise), SPARC_rates, color='red', label="SPARC galaxies (assuming ft height = 0.1)")
    # for pt in range(len(gal_list)):
    #     if SPARC_noise[pt] < 0.2:
    #         plt.annotate(gal_list[pt], (SPARC_noise[pt], SPARC_rates[pt] + 0.2), color='red')

    # plt.scatter([4.84539013], [10], color='red', label="NGC 1560 (Sanders)")
    # plt.annotate(r"$v_{obs}$", (4.24539013, 10.7), color='red')
    plt.scatter([3.16058723], [10], marker="*", color='gold', label="NGC 1560 (Sanders)")
    # plt.annotate(r"$v_{bar}$", (2.56058723, 10.7), color='red')
    
    # Filter SPARC points within the boundaries specified by 'extent'
    SPARC_galaxies, SPARC_rates, SPARC_s2n = SPARC_Vobs_ft()

    # for i in range(len(SPARC_galaxies)):
    #     if SPARC_galaxies[i] == "UGC06787": print("For UGC06787, s2n = ", SPARC_s2n[i], " and rate = ", SPARC_rates[i])
    #     else: continue

    # raise Exception("Checked UGC02953 s2n and rate!")

    filtered_s2n, filtered_rates, filtered_gal = [], [], []
    for g, rate, s2n in zip(SPARC_galaxies, SPARC_rates, SPARC_s2n):
        if extent[0] <= s2n <= extent[1] and extent[2] <= rate <= extent[3]:
            filtered_s2n.append(s2n)
            filtered_rates.append(rate)
            filtered_gal.append(g)

    # Plot the filtered points
    plt.scatter(filtered_s2n, filtered_rates, color='tab:red', marker=".", label="SPARC galaxies (Vobs)")
    for pt in range(len(filtered_gal)):
        if filtered_gal[pt] in [ "NGC1003", "NGC2403", "UGC02953", "UGC06787" ]:
            plt.annotate(filtered_gal[pt], (filtered_s2n[pt] - 1.7, filtered_rates[pt] + 0.8), color='tab:red', fontsize=10)

    if sn == 1: plt.legend(loc='upper left')
    # plt.xscale("log")
    plt.savefig(f"{fname}{signames[sn]}.pdf", bbox_inches='tight')
    plt.close()


# Get required noise control from THINGS data for achieving 2 sigma / 5 sigma significance.
# if use_THINGS:
#     gal_things, r_things, Vobs_things, errV_things = get_things()

#     sigma_2, sigma_5 = [], []

#     for i in range(len(gal_things)):
#         r = r_things[i]
#         Vobs = Vobs_things[i]
#         # errV = errV_things[i]

#         sampling_rate = len(r) / (max(r) - min(r))
#         smp_idx = np.nonzero(sampling_rate < samp_rates/10.0)[0][0]
#         sig_idx2 = np.nonzero(ft_significance[smp_idx] >= 2.0)[0][0]
#         sig_idx5 = np.where(ft_significance[smp_idx] >= 5.0)[0][0]
#         sigma_2.append( 10.0 / noise_arr[sig_idx2] )
#         sigma_5.append( 10.0 / noise_arr[sig_idx5] )

#     sort_idx = np.argsort(sigma_2)
#     sigma_2 = np.array(sigma_2)[sort_idx]
#     sigma_5 = np.array(sigma_5)[sort_idx]
#     gal_things = np.array(gal_things)[sort_idx]

#     plt.title(r"Noise restriction on Little Things")
#     plt.xlabel("Things galaxies")
#     plt.ylabel("Feature / noise ratio")
#     plt.scatter(gal_things, np.round(sigma_2, 2), alpha=0.5, c="mediumblue", label=r"2 $\sigma$ significance")
#     plt.scatter(gal_things, np.round(sigma_5, 2), alpha=0.5, c="red", label=r"5 $\sigma$ significance")
#     plt.xticks([])

#     plt.legend()
#     plt.savefig(f"{fname}things_noise.pdf")
#     plt.close()

# if use_SPARC:
#     # Get galaxy data from table1.
#     file = "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt"

#     SPARC_c = [ "Galaxy", "T", "D", "e_D", "f_D", "Inc",
#             "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
#                 "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
#     table = pd.read_fwf(file, skiprows=98, names=SPARC_c)

#     columns = [ "Rad", "Vobs", "errV", "Vgas",
#                 "Vdisk", "Vbul", "SBdisk", "SBbul" ]

#     galaxy_count = len(table["Galaxy"])

#     sigma_2, sigma_5 = [], []

#     for i in range(galaxy_count):
#         g = table["Galaxy"][i]

#         file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
#         rawdata = np.loadtxt(file_path)
#         data = pd.DataFrame(rawdata, columns=columns)
#         r = data["Rad"]

#         sampling_rate = len(r) / (max(r) - min(r))
#         smp_idx = np.nonzero(sampling_rate < samp_rates/10.0)[0][0]
#         sig_idx2 = np.nonzero(ft_significance[smp_idx] >= 2.0)[0][0]
#         sig_idx5 = np.where(ft_significance[smp_idx] >= 5.0)[0][0]
#         sigma_2.append( 10.0 / noise_arr[sig_idx2] )
#         sigma_5.append( 10.0 / noise_arr[sig_idx5] )

#     sort_idx = np.argsort(sigma_2)
#     sigma_2 = np.array(sigma_2)[sort_idx]
#     sigma_5 = np.array(sigma_5)[sort_idx]
#     gal_sparc = np.array(table["Galaxy"])[sort_idx]

#     plt.title(r"Noise restriction on SPARC galaxies")
#     plt.xlabel("SPARC galaxies")
#     plt.ylabel("Feature / noise ratio")
#     plt.scatter(gal_sparc, np.round(sigma_2, 2), alpha=0.5, c="mediumblue", label=r"2 $\sigma$ significance")
#     plt.scatter(gal_sparc, np.round(sigma_5, 2), alpha=0.5, c="red", label=r"5 $\sigma$ significance")
#     plt.xticks([])

#     plt.legend()
#     plt.savefig(f"{fname}sparc_noise.pdf")
#     plt.close()
