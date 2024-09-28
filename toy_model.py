#!/usr/bin/env python
# (C) 2024 Enoch Ko.
"""
Generate toy models of galaxy RC using arctan curves + gaussian features,
then apply same analysis of correlation coefficients + DTW on residuals
to better understand the effect/restriction of feature sizes and noise/uncertainties.
"""
import math
import numpy as np
from scipy import stats, interpolate
import matplotlib
import matplotlib.pyplot as plt

import jax
import numpyro
import argparse

from resource import getrusage, RUSAGE_SELF
# from tqdm import tqdm

from utils_analysis.toy_gen import toy_gen
from utils_analysis.med_filter import med_filter
from utils_analysis.toy_GP import GP_fit, GP_residuals
from utils_analysis.dtw_utils import do_DTW
from utils_analysis.correlations import corr_radii, corr_window

matplotlib.use("Agg")
memory_usage = []   # Track memory usage throughout programme.


# Switches for running different parts of the analysis.
use_MF      = False
use_GP      = False
apply_DTW   = True
corr_rad    = True
corr_win    = True
make_plots  = True

# File names for different analysis methods.
if use_MF and use_GP:
    raise Exception("GP and median filter cannot be used simultaneously! To get 2D feature significance plots, run toy_ftsig.py.")
elif use_MF:
    fileloc = "/mnt/users/koe/plots/toy_model/use_MF/"
    MF_size = 20    # Define window size for median filter (if used).
elif use_GP:
    fileloc = "/mnt/users/koe/plots/toy_model/use_GP/"
else:
    fileloc = "/mnt/users/koe/plots/toy_model/"

if corr_rad or corr_win:
    colors = [ 'tab:red', 'k' ]
    labels = [ 'Vbar', 'Vobs' ]
    deriv_dir = [ "d0", "d1", "d2" ]
    color_bar = "orange"
    if corr_win: window_size = 11

# Print report of analyses used and some corresponding variables.
print("Running toy_model.py with the following methods:")
if use_MF:     print(f" - Median filter (window length = {MF_size})")
if use_GP:     print(" - Gaussian process")
if apply_DTW:  print(" - Dynamic time warping")
if corr_rad:   print(" - Correlation coefficients (increasing radii)")
if corr_win:   print(f" - Correlation coefficients (moving window, length = {window_size})")
if make_plots: print(" - Make plots")


# Parameters for Gaussian bump (fixed feature) and noise (varying amplitudes).
bump_size   = 20.0   # Defined in terms of percentage of max(Vbar)
bump_loc    = 5.0
bump_FWHM   = 0.5
bump_sigma  = bump_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# if use_GP:
#     noise_arr = np.linspace(0.0, bump_size/2, 51, endpoint=True)
#     num_iterations = 50
# else:
noise_arr = np.linspace(0.0, bump_size/2, 101, endpoint=True)
num_iterations = 200
num_noise = len(noise_arr)


# Initialize arrays for summary plots.
rad_spearmans = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
rad_pearsons  = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
rad_Xft_spearmans = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
rad_Xft_pearsons  = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]

win_spearmans = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
win_pearsons  = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
win_Xft_spearmans = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
win_Xft_pearsons  = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]

dtw_costs, Xft_costs = np.zeros((num_iterations, num_noise)), np.zeros((num_iterations, num_noise))
dtw_window, Xft_window = np.zeros((num_iterations, num_noise)), np.zeros((num_iterations, num_noise))


# Initialize args for GP (if used).
if use_GP:
    assert numpyro.__version__.startswith("0.15.0")
    numpyro.enable_x64()
    parser = argparse.ArgumentParser(description="Gaussian Process example") # To keep the inference from getting constant samples.
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--thinning", nargs="?", default=2, type=int)
    parser.add_argument("--num-data", nargs="?", default=25, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--init-strategy",
        default="median",
        type=str,
        choices=["median", "feasible", "uniform", "sample"],
    )
    parser.add_argument("--no-cholesky", dest="use_cholesky", action="store_false")
    parser.add_argument("--testing", default=False, type=bool)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)


"""
===================================
~~~~~~~~~~~~ MAIN LOOP ~~~~~~~~~~~~
===================================
"""
# for i in tqdm(range(num_noise)):
for i in range(num_noise):
    if i%10 == 0 or use_GP:
        if i == 0:
            print(f"\nRunning iteration {i}/{num_noise} (with {num_iterations} iterations per noise level)...")
        else:
            print(f"\nRunning iteration {i}/{num_noise}...")

    rad = np.linspace(10., 0., 100, endpoint=False)[::-1]   # Defined this way to exclude the starting point r=0.
    num_rad = len(rad)
    
    noise = noise_arr[i]
    bump, Vraw, velocities, Vraw_werr, v_werr, residuals, res_Xft = toy_gen(rad, bump_loc, bump_size, bump_sigma, noise, num_iterations)

    # Vobs (w/ feature) residuals if it's generated perfectly by MOND.
    MOND_res = (velocities[:,1,:] - Vraw[:,1,:])

    # Apply simple median filter.
    if use_MF:
        file_name = fileloc+f"MF_fits/ratio={round(noise/bump_size, 2)}.png"
        Xft_fname = fileloc+f"MF_fits/Xft/ratio={round(noise/bump_size, 2)}.png"
        
        if noise in noise_arr[::10]:
            _, residuals     = med_filter(rad, v_werr, size=MF_size, make_plots=make_plots, file_name=file_name)
            _, residuals_Xft = med_filter(rad, Vraw_werr, size=MF_size, make_plots=make_plots, file_name=Xft_fname)
        else:
            _, residuals     = med_filter(rad, v_werr, size=MF_size)
            _, residuals_Xft = med_filter(rad, Vraw_werr, size=MF_size)

        _, residuals_MOND = med_filter(rad, velocities, size=MF_size)

    # Apply GP regression.
    if use_GP:
        residuals, residuals_Xft, residuals_MOND = [], [], []
        file_names = [ fileloc+f"corner_plots/ratio={round(noise/bump_size, 2)}.png",
                               fileloc+f"GP_fits/ratio={round(noise/bump_size, 2)}.png" ]
        Xft_fnames = [ fileloc+f"corner_plots/Xft/ratio={round(noise/bump_size, 2)}.png",
                               fileloc+f"GP_fits/Xft/ratio={round(noise/bump_size, 2)}.png" ]

        pred_means, pred_bands = GP_fit(args, rad, v_werr[0], rad, make_plots=(make_plots and noise in noise_arr[::10]), file_name=file_names[0])
        Xft_means, Xft_bands = GP_fit(args, rad, Vraw_werr[0], rad, make_plots=(make_plots and noise in noise_arr[::10]), file_name=Xft_fnames[0])
        pred_means_MOND, pred_bands_MOND = GP_fit(args, rad, velocities[itr], rad)

        for itr in range(num_iterations):
            residuals.append( GP_residuals(rad, v_werr[itr], rad, pred_means, pred_bands,
                                           make_plots=(make_plots and itr==1 and noise in noise_arr[::10]), file_name=file_names[1]) )
            residuals_Xft.append( GP_residuals(rad, Vraw_werr[itr], rad, Xft_means, Xft_bands,
                                               make_plots=(make_plots and itr==1 and noise in noise_arr[::10]), file_name=Xft_fnames[1]) )
            residuals_MOND.append( GP_residuals(rad, velocities[itr], rad, pred_means_MOND, pred_bands_MOND) )

    # Transpose residuals arrays and extract required Xft residuals for DTW:
    # Transpose: 3D array of size num_iterations x 2 (vel) x 100 (rad) --> 2 x num_iterations x 100 (rad).
    res_dtw = np.transpose(residuals, (1, 0, 2))

    if use_MF or use_GP:
        res_Xft = np.transpose(residuals_Xft, (1, 0, 2))[1]
        MOND_res = np.transpose(residuals_MOND, (1, 0, 2))[1]


    # Interpolate residuals for potential use of derivatives in calculating naive correlation coefficients.
    if corr_rad or corr_win:
        # Interpolate the residuals with cubic Hermite spline splines.
        res_fits, res_Xft_fits = [], []
        for itr in range(num_iterations):
            v_d0, v_d1, v_d2 = [], [], []
            for v_comp in residuals[itr]:
                v_d0.append(interpolate.pchip_interpolate(rad, v_comp, rad))
                v_d1.append(interpolate.pchip_interpolate(rad, v_comp, rad, der=1))
                v_d2.append(interpolate.pchip_interpolate(rad, v_comp, rad, der=2))
            res_fits.append( [ v_d0, v_d1, v_d2 ] )

            v_d0, v_d1, v_d2 = [], [], []
            v_d0.append(interpolate.pchip_interpolate(rad, res_Xft[itr], rad))
            v_d1.append(interpolate.pchip_interpolate(rad, res_Xft[itr], rad, der=1))
            v_d2.append(interpolate.pchip_interpolate(rad, res_Xft[itr], rad, der=2))
            res_Xft_fits.append( [ v_d0, v_d1, v_d2 ] )


    """
    -----------------
    DTW on residuals.
    -----------------
    """
    if apply_DTW:
        # DTW analyses on full RCs.
        for itr in range(num_iterations):
            file_names = [ fileloc+f"Xft_matrix/ratio={round(noise/bump_size, 2)}.png",
                           fileloc+f"Xft_alignment/ratio={round(noise/bump_size, 2)}.png" ]
            Xft_cost = do_DTW(itr, num_rad, res_Xft, MOND_res, window=False, make_plots=(make_plots and itr==0 and noise in noise_arr[::10]), file_names=file_names)
            Xft_costs[itr][i] = Xft_cost

        for itr in range(num_iterations):
            file_names = [ fileloc+f"dtw_matrix/ratio={round(noise/bump_size, 2)}.png",
                           fileloc+f"dtw_alignment/ratio={round(noise/bump_size, 2)}.png" ]
            norm_cost = do_DTW(itr, num_rad, res_dtw[1], MOND_res, window=False, make_plots=(make_plots and itr==0 and noise in noise_arr[::10]), file_names=file_names)
            dtw_costs[itr][i] = norm_cost

        # DTW along a small window (length 1) around the feature.
        for itr in range(num_iterations):
            file_names = [ fileloc+f"dtw_window/Xft_matrix/ratio={round(noise/bump_size, 2)}.png",
                           fileloc+f"dtw_window/Xft_alignment/ratio={round(noise/bump_size, 2)}.png" ]
            win_cost = do_DTW(itr, window_size, res_Xft, MOND_res, window=True, make_plots=(make_plots and itr==0 and noise in noise_arr[::10]), file_names=file_names)
            Xft_window[itr][i] = win_cost

        for itr in range(num_iterations):
            file_names = [ fileloc+f"dtw_window/matrix/ratio={round(noise/bump_size, 2)}.png",
                           fileloc+f"dtw_window/alignment/ratio={round(noise/bump_size, 2)}.png" ]
            win_cost = do_DTW(itr, window_size, res_dtw[1], MOND_res, window=True, make_plots=(make_plots and itr==0 and noise in noise_arr[::10]), file_names=file_names)
            dtw_window[itr][i] = win_cost


    """
    -----------------------------------
    Calculate correlation coefficients.
    -----------------------------------
    """
    # Correlation plots using spheres of increasing radius
    if corr_rad and noise != 0.0:
        noise_ratio = round(noise/bump_size, 2)
        for der in range(3):
            rad_corr_perc = corr_radii( num_iterations, der, num_rad, res_fits, v_werr, (make_plots and noise in noise_arr[::10]),
                                        fileloc, noise_ratio, rad, bump, Vraw, residuals, noise )
            rad_spearmans[der][i-1] = rad_corr_perc[0]
            rad_pearsons[der][i-1]  = rad_corr_perc[1]

            res_fits_temp, v_werr_temp = [ np.squeeze(np.array(res_fits)[:,:,0]), np.squeeze(res_Xft_fits, axis=2) ], [ v_werr[:,0,:], Vraw_werr[:,1,:] ]
            res_fits_temp, v_werr_temp = np.transpose(res_fits_temp, (1, 2, 0, 3)), np.transpose(v_werr_temp, (1, 0, 2))
            rad_corr_perc = corr_radii( num_iterations, der, num_rad, res_fits_temp, v_werr_temp )
            rad_Xft_spearmans[der][i-1] = rad_corr_perc[0]
            rad_Xft_pearsons[der][i-1]  = rad_corr_perc[1]


    # Correlation plots using windows of length max{1 * Reff, 5 data points}.
    if corr_win and noise != 0.0:
        noise_ratio = round(noise/bump_size, 2)
        for der in range(3):
            win_corr_perc = corr_window( num_iterations, der, num_rad, res_fits, v_werr, window_size,
                                        (make_plots and noise in noise_arr[::10]), fileloc, noise_ratio, rad, bump, Vraw, residuals, noise )
            win_spearmans[der][i-1] = win_corr_perc[0]
            win_pearsons[der][i-1]  = win_corr_perc[1]

            res_fits_temp, v_werr_temp = [ np.squeeze(np.array(res_fits)[:,:,0]), np.squeeze(res_Xft_fits, axis=2) ], [ v_werr[:,0,:], Vraw_werr[:,1,:] ]
            res_fits_temp, v_werr_temp = np.transpose(res_fits_temp, (1, 2, 0, 3)), np.transpose(v_werr_temp, (1, 0, 2))
            win_corr_perc = corr_window( num_iterations, der, num_rad, res_fits_temp, v_werr_temp, window_size )
            win_Xft_spearmans[der][i-1] = win_corr_perc[0]
            win_Xft_pearsons[der][i-1]  = win_corr_perc[1]


    memory_usage.append(getrusage(RUSAGE_SELF).ru_maxrss)
    jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem.

print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)

plt.title("Memory usage for toy_model.py")
plt.xlabel("Number of iterations executed")
plt.ylabel("Maximum memory used (kb)")
plt.plot(range(num_noise), memory_usage)

plt.savefig(fileloc+"memory_usage.png", dpi=300, bbox_inches="tight")
plt.close()


bump_ratio = noise_arr / bump_size
half_noise = math.ceil( num_noise / 2 )

# Summary plots.
if apply_DTW:
    dtw_costs   = np.percentile( dtw_costs,  [16.0, 50.0, 84.0], axis=0 )
    Xft_costs   = np.percentile( Xft_costs,  [16.0, 50.0, 84.0], axis=0 )
    dtw_window  = np.percentile( dtw_window, [16.0, 50.0, 84.0], axis=0 )
    Xft_window  = np.percentile( Xft_window, [16.0, 50.0, 84.0], axis=0 )

    plt.title("Normalized DTW alignment costs")
    plt.ylabel("Normalized DTW costs")
    plt.xlabel("Noise / feature height")
    plt.plot(bump_ratio[:half_noise], dtw_costs[1][:half_noise], color='tab:blue', label="Costs w/ feature")
    plt.fill_between(bump_ratio[:half_noise], dtw_costs[0][:half_noise], dtw_costs[2][:half_noise], color='tab:blue', alpha=0.2)
    plt.plot(bump_ratio[:half_noise], Xft_costs[1][:half_noise], '--', color='red', label="Costs w/o feature")
    plt.fill_between(bump_ratio[:half_noise], Xft_costs[0][:half_noise], Xft_costs[2][:half_noise], color='red', alpha=0.2)

    plt.legend()
    plt.savefig(fileloc+"dtwVnoise.png", dpi=300, bbox_inches="tight")
    plt.close()


    plt.title("Normalized DTW alignment costs")
    plt.ylabel("Normalized DTW costs")
    plt.xlabel("Noise / feature height")
    plt.plot(bump_ratio, dtw_costs[1], color='tab:blue', label="Costs w/ feature")
    plt.fill_between(bump_ratio, dtw_costs[0], dtw_costs[2], color='tab:blue', alpha=0.2)
    plt.plot(bump_ratio, Xft_costs[1], '--', color='red', label="Costs w/o feature")
    plt.fill_between(bump_ratio, Xft_costs[0], Xft_costs[2], color='red', alpha=0.2)

    plt.legend()
    plt.savefig(fileloc+"dtwVnoise_FULL.png", dpi=300, bbox_inches="tight")
    plt.close()


    plt.title("Normalized DTW alignment costs (on window around feature)")
    plt.ylabel("Normalized DTW costs")
    plt.xlabel("Noise / feature height")
    plt.plot(bump_ratio[:half_noise], dtw_window[1][:half_noise], color='tab:blue', label="Costs w/ feature")
    plt.fill_between(bump_ratio[:half_noise], dtw_window[0][:half_noise], dtw_window[2][:half_noise], color='tab:blue', alpha=0.2)
    plt.plot(bump_ratio[:half_noise], Xft_window[1][:half_noise], '--', color='red', label="Costs w/o feature")
    plt.fill_between(bump_ratio[:half_noise], Xft_window[0][:half_noise], Xft_window[2][:half_noise], color='red', alpha=0.2)

    plt.legend()
    plt.savefig(fileloc+"dtw_window/dtwVnoise.png", dpi=300, bbox_inches="tight")
    plt.close()


    plt.title("Normalized DTW alignment costs (on window around feature)")
    plt.ylabel("Normalized DTW costs")
    plt.xlabel("Noise / feature height")
    plt.plot(bump_ratio, dtw_window[1], color='tab:blue', label="Costs w/ feature")
    plt.fill_between(bump_ratio, dtw_window[0], dtw_window[2], color='tab:blue', alpha=0.2)
    plt.plot(bump_ratio, Xft_window[1], '--', color='red', label="Costs w/o feature")
    plt.fill_between(bump_ratio, Xft_window[0], Xft_window[2], color='red', alpha=0.2)

    plt.legend()
    plt.savefig(fileloc+"dtw_window/dtwVnoise_FULL.png", dpi=300, bbox_inches="tight")
    plt.close()


c_corr = [ 'royalblue', 'midnightblue', 'red', 'darkred' ]

if corr_rad:
    rad_spearmans, rad_pearsons = np.array(rad_spearmans), np.array(rad_pearsons)
    rad_Xft_spearmans, rad_Xft_pearsons = np.array(rad_Xft_spearmans), np.array(rad_Xft_pearsons)
    for der in range(3):
        # Truncated (by half) correlation vs noise plots.
        plt.title("Correlation coefficients across entire RC")
        plt.ylabel("Correlation coefficients")
        plt.xlabel("Noise / feature height")

        plt.plot(bump_ratio[1:half_noise], rad_spearmans[der,:half_noise-1,1], color=c_corr[0], label=r"Spearman $\rho$ (w/ ft)")
        plt.fill_between(bump_ratio[1:half_noise], rad_spearmans[der,:half_noise-1,0], rad_spearmans[der,:half_noise-1,2], color=c_corr[0], alpha=0.15)
        plt.plot(bump_ratio[1:half_noise], rad_pearsons[der,:half_noise-1,1], '--', color=c_corr[1], label=r"Pearson $\rho$ (w/ ft)")
        plt.fill_between(bump_ratio[1:half_noise], rad_pearsons[der,:half_noise-1,0], rad_pearsons[der,:half_noise-1,2], color=c_corr[1], alpha=0.15)
        plt.plot(bump_ratio[1:half_noise], rad_Xft_spearmans[der,:half_noise-1,1], color=c_corr[2], label=r"Spearman $\rho$ (w/o ft)")
        plt.fill_between(bump_ratio[1:half_noise], rad_Xft_spearmans[der,:half_noise-1,0], rad_Xft_spearmans[der,:half_noise-1,2], color=c_corr[2], alpha=0.15)
        plt.plot(bump_ratio[1:half_noise], rad_Xft_pearsons[der,:half_noise-1,1], '--', color=c_corr[3], label=r"Pearson $\rho$ (w/o ft)")
        plt.fill_between(bump_ratio[1:half_noise], rad_Xft_pearsons[der,:half_noise-1,0], rad_Xft_pearsons[der,:half_noise-1,2], color=c_corr[3], alpha=0.15)
        
        plt.legend()
        plt.savefig(fileloc+f"correlations/radii_d{der}.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Full correlation vs noise plot.
        plt.title("Correlation coefficients across entire RC")
        plt.ylabel("Correlation coefficients")
        plt.xlabel("Noise / feature height")

        plt.plot(bump_ratio[1:], rad_spearmans[der,:,1], color=c_corr[0], label=r"Spearman $\rho$ (w/ ft)")
        plt.fill_between(bump_ratio[1:], rad_spearmans[der,:,0], rad_spearmans[der,:,2], color=c_corr[0], alpha=0.15)
        plt.plot(bump_ratio[1:], rad_pearsons[der,:,1], '--', color=c_corr[1], label=r"Pearson $\rho$ (w/ ft)")
        plt.fill_between(bump_ratio[1:], rad_pearsons[der,:,0], rad_pearsons[der,:,2], color=c_corr[1], alpha=0.15)
        plt.plot(bump_ratio[1:], rad_Xft_spearmans[der,:,1], color=c_corr[2], label=r"Spearman $\rho$ (w/o ft)")
        plt.fill_between(bump_ratio[1:], rad_Xft_spearmans[der,:,0], rad_Xft_spearmans[der,:,2], color=c_corr[2], alpha=0.15)
        plt.plot(bump_ratio[1:], rad_Xft_pearsons[der,:,1], '--', color=c_corr[3], label=r"Pearson $\rho$ (w/o ft)")
        plt.fill_between(bump_ratio[1:], rad_Xft_pearsons[der,:,0], rad_Xft_pearsons[der,:,2], color=c_corr[3], alpha=0.15)

        plt.legend()
        plt.savefig(fileloc+f"correlations/radii_d{der}_FULL.png", dpi=300, bbox_inches="tight")
        plt.close()


if corr_win:
    win_spearmans, win_pearsons = np.array(win_spearmans), np.array(win_pearsons)
    win_Xft_spearmans, win_Xft_pearsons = np.array(win_Xft_spearmans), np.array(win_Xft_pearsons)
    for der in range(3):
        # Truncated (by half) correlation vs noise plots.
        plt.title("Correlation coefficients on window around feature")
        plt.ylabel("Correlation coefficients")
        plt.xlabel("Noise / feature height")

        plt.plot(bump_ratio[1:half_noise], win_spearmans[der,:half_noise-1,1], color=c_corr[0], label=r"Spearman $\rho$ (w/ ft)")
        plt.fill_between(bump_ratio[1:half_noise], win_spearmans[der,:half_noise-1,0], win_spearmans[der,:half_noise-1,2], color=c_corr[0], alpha=0.15)
        plt.plot(bump_ratio[1:half_noise], win_pearsons[der,:half_noise-1,1], '--', color=c_corr[1], label=r"Pearson $\rho$ (w/ ft)")
        plt.fill_between(bump_ratio[1:half_noise], win_pearsons[der,:half_noise-1,0], win_pearsons[der,:half_noise-1,2], color=c_corr[1], alpha=0.15)
        plt.plot(bump_ratio[1:half_noise], win_Xft_spearmans[der,:half_noise-1,1], color=c_corr[2], label=r"Spearman $\rho$ (w/o ft)")
        plt.fill_between(bump_ratio[1:half_noise], win_Xft_spearmans[der,:half_noise-1,0], win_Xft_spearmans[der,:half_noise-1,2], color=c_corr[2], alpha=0.15)
        plt.plot(bump_ratio[1:half_noise], win_Xft_pearsons[der,:half_noise-1,1], '--', color=c_corr[3], label=r"Pearson $\rho$ (w/o ft)")
        plt.fill_between(bump_ratio[1:half_noise], win_Xft_pearsons[der,:half_noise-1,0], win_Xft_pearsons[der,:half_noise-1,2], color=c_corr[3], alpha=0.15)
        
        plt.legend()
        plt.savefig(fileloc+f"correlations/window_d{der}.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Full correlation vs noise plot.
        plt.title("Correlation coefficients on window around feature")
        plt.ylabel("Correlation coefficients")
        plt.xlabel("Noise / feature height")

        plt.plot(bump_ratio[1:], win_spearmans[der,:,1], color=c_corr[0], label=r"Spearman $\rho$ (w/ ft)")
        plt.fill_between(bump_ratio[1:], win_spearmans[der,:,0], win_spearmans[der,:,2], color=c_corr[0], alpha=0.15)
        plt.plot(bump_ratio[1:], win_pearsons[der,:,1], '--', color=c_corr[1], label=r"Pearson $\rho$ (w/ ft)")
        plt.fill_between(bump_ratio[1:], win_pearsons[der,:,0], win_pearsons[der,:,2], color=c_corr[1], alpha=0.15)
        plt.plot(bump_ratio[1:], win_Xft_spearmans[der,:,1], color=c_corr[2], label=r"Spearman $\rho$ (w/o ft)")
        plt.fill_between(bump_ratio[1:], win_Xft_spearmans[der,:,0], win_Xft_spearmans[der,:,2], color=c_corr[2], alpha=0.15)
        plt.plot(bump_ratio[1:], win_Xft_pearsons[der,:,1], '--', color=c_corr[3], label=r"Pearson $\rho$ (w/o ft)")
        plt.fill_between(bump_ratio[1:], win_Xft_pearsons[der,:,0], win_Xft_pearsons[der,:,2], color=c_corr[3], alpha=0.15)

        plt.legend()
        plt.savefig(fileloc+f"correlations/window_d{der}_FULL.png", dpi=300, bbox_inches="tight")
        plt.close()
