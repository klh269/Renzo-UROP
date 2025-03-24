#!/usr/bin/env python
# (C) 2024 Enoch Ko.
"""
Generate toy models of galaxy RC using arctan curves + gaussian features,
then apply same analysis of correlation coefficients + DTW on residuals
to better understand the effect/restriction of sampling rates, feature sizes and uncertainties.

To run this as a for-loop in bash:
for i in {0..29}; do addqueue -q cmb -c "1-3 days" -n 1 -m 4 mock_data.py --ft-width 0.25 --samp-idx $i; done
"""
import numpy as np
from math import floor, ceil
import matplotlib
import matplotlib.pyplot as plt

import jax
import numpyro
import argparse
from resource import getrusage, RUSAGE_SELF
from tqdm import tqdm

from utils_analysis.toy_gen import toy_gen, toy_scatter
# from utils_analysis.med_filter import med_filter
from utils_analysis.toy_GP import GP_fit, get_residuals
from utils_analysis.dtw_utils import do_DTW
from utils_analysis.correlations import corr_radii

matplotlib.use("Agg")
plt.rcParams.update({'font.size': 13})


"""
------------------------
Define useful functions.
------------------------
"""
def set_args( use_GP:bool=False ):
    """
    Initialize args, inc.
        - ft_width: feature width (in kpc) * 10;
        - samp_idx: sampling rate selection (0-29 for 1-30 samples per kpc); 
        - initialize args for GP (if used).
    """
    if use_GP:
        assert numpyro.__version__.startswith("0.15.0")
        numpyro.enable_x64()    # To keep the inference from getting constant samples.
        parser = argparse.ArgumentParser(description="Gaussian Process")
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
        parser.add_argument("--ft-width", default=10, type=float)
        parser.add_argument("--samp-idx", type=int)
        args = parser.parse_args()

        numpyro.set_platform(args.device)
        numpyro.set_host_device_count(args.num_chains)

    else:
        parser = argparse.ArgumentParser(description="Mock Data")
        parser.add_argument("--ft-width", default=10, type=float)
        parser.add_argument("--samp-idx", type=int)
        args = parser.parse_args()
    
    return args

def get_fname( width:float ):
    """Get file name for saving plots and arrays"""
    fname = f"/mnt/users/koe/plots/mock_data/width={width}/"
    # if use_GP: fname += "use_GP/"
    # else:      fname += "use_MF/"
    return fname


"""
------------------------------------------------------------
Initialize parameters for mock data generation and analysis.
------------------------------------------------------------
"""
# Switches for running different parts of the analysis.
use_GP    = True    # Note: A median filter is used if use_GP = False.
apply_DTW = True
corr_rad  = True
use_window = True

args = set_args( use_GP )
    
# Parameters for Gaussian bump (fixed feature) and noise (varying amplitudes).
bump_size = -4.0    # Bump size similar to that in Sanders' NGC 1560.
bump_loc  = 5.0

# file_FWHM  = args.ft_width
# bump_FWHM  = file_FWHM / 10
# bump_sigma = bump_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

bump_sigma = args.ft_width

# print(f"Correlating RCs with features of FWHM = {bump_FWHM}")

# Generate noise from 1 / uniform height array (fewer iterations for GP due to demanding runtime).
# if use_GP:
#     height_arr = np.linspace(20.0, 2.0, 37, endpoint=True)
#     noise_arr = - bump_size / height_arr
#     num_iterations = 50
# else:

# height_arr = np.linspace(5.0, 2.0, 7, endpoint=True)
height_arr = np.linspace(20.0, 2.0, 37, endpoint=True)
noise_arr = - bump_size / height_arr
num_iterations = 1000

num_noise = len(noise_arr)


# Initialize arrays for summary arrays.
# rad_spearmans     = [ [ [] for _ in range(num_noise) ] for _ in range(3) ]
# rad_pearsons      = [ [ [] for _ in range(num_noise) ] for _ in range(3) ]
# rad_Xft_spearmans = [ [ [] for _ in range(num_noise) ] for _ in range(3) ]
# rad_Xft_pearsons  = [ [ [] for _ in range(num_noise) ] for _ in range(3) ]

# Remove PCHIP interpolation and therefore higher derivatives.
MOND_spearmans = [ [] for _ in range(num_noise) ]
MOND_pearsons = [ [] for _ in range(num_noise) ]
LCDM_spearmans = [ [] for _ in range(num_noise) ]
LCDM_pearsons = [ [] for _ in range(num_noise) ]

MOND_costs, LCDM_costs  = np.zeros((num_iterations, num_noise)), np.zeros((num_iterations, num_noise))
# dtw_window, Xft_window = np.zeros((num_iterations, num_noise)), np.zeros((num_iterations, num_noise))


# Create array of sampling rates to sample from.
samp_idx    = args.samp_idx
samp_rate   = np.linspace(1, 30, 30, endpoint=True, dtype=int)[samp_idx]
num_samples = samp_rate * 10

# if not use_GP: MF_size = int(max( 5, bump_FWHM * samp_rate * 2 ))


# Print report of analyses used and some corresponding variables.
print(f"[samp_rate = {samp_rate}] Running mock_data.py with the following methods:")
if use_GP:    print(" - Gaussian process")
# else:         print(f" - Median filter (window length = {MF_size})")
if apply_DTW: print(" - Dynamic time warping")
if corr_rad:  print(" - Correlation coefficients (increasing radii)")


# Simple code for calculating feature significance by comparing costs/correlation coefficients
# between (Vobs w/ ft + Vbar w/ ft) and (Vobs W/O ft + Vbar w/ ft).
def get_significance(corr1, corr2):
    # if rhos == True:
    #     sigma1 = (corr1[0,:,2] - corr1[0,:,0]) / 2
    #     sigma2 = (corr2[0,:,2] - corr2[0,:,0]) / 2
    #     ftsig = abs(( corr2[0,:,1] - corr1[0,:,1] )) / np.sqrt(sigma1**2 + sigma2**2)
    # else:   # DTW dim = 3 perc x num_noise
    sigma1 = (corr1[2] - corr1[0]) / 2
    sigma2 = (corr2[2] - corr2[0]) / 2
    ftsig = abs(( corr2[1] - corr1[1] )) / np.sqrt(sigma1**2 + sigma2**2)
    return ftsig


"""
===================================
~~~~~~~~~~~~ MAIN LOOP ~~~~~~~~~~~~
===================================
"""
fileloc = get_fname( bump_sigma )

# Define galaxy radius (units ~kpc; excluding the point r=0).
rad = np.linspace(10., 0., num_samples, endpoint=False)[::-1]
num_rad = len(rad)

Vmax, bump, Vbar_raw, vel_MOND, vel_LCDM = toy_gen(rad, bump_loc, bump_size, bump_sigma, noise_arr[0])

# Apply GP regression.
if use_GP:
    print("\nApplying GPR...")
    res_Vbar, res_LCDM, res_MOND = [], [], []

    rad_Xft = np.delete(rad, np.s_[floor(4.5*samp_rate):ceil(5.5*samp_rate)])
    Vbar_Xft = np.delete(Vbar_raw/Vmax, np.s_[floor(4.5*samp_rate):ceil(5.5*samp_rate)])
    Vmond_Xft = np.delete(vel_MOND/Vmax, np.s_[floor(4.5*samp_rate):ceil(5.5*samp_rate)])
    Vcdm_Xft = np.delete(vel_LCDM/Vmax, np.s_[floor(4.5*samp_rate):ceil(5.5*samp_rate)])
    
    Vbar_means, Vbar_bands = GP_fit(args, rad_Xft, Vbar_Xft, rad)
    Vmond_means, Vmond_bands = GP_fit(args, rad_Xft, Vmond_Xft, rad)
    Vcdm_means, Vcdm_bands = GP_fit(args, rad_Xft, Vcdm_Xft, rad)

for i in range(num_noise):
    # if i%10 == 0:
        # if i == 0:
        #     print(f"\nRunning noise level {i+1}/{num_noise} (with {num_iterations} iterations per level)...")
        # else:
    print(f"\nRunning noise level {i+1}/{num_noise}...")

    noise = noise_arr[i]
    
    # Generate toy RCs with residuals (Vraw = w/o ft, Vraw_werr = w/ noise; velocities = w/ ft, v_werr = w/ noise);
    # dim = itr x 2 (Vbar, Vobs) x num_rad.
    # _, Vraw, velocities, Vraw_werr, v_werr, _, _ = toy_gen(rad, bump_loc, bump_size, bump_sigma, noise, num_iterations)
    Vbar, Vmond, Vcdm = toy_scatter(num_iterations, noise, Vmax, Vbar_raw, vel_MOND, vel_LCDM)

    # Vobs (w/ feature) residuals if it's generated perfectly by MOND.
    # MOND_res = (velocities[:,1,:] - Vraw[:,1,:])


    if use_GP:
        # print(f"Calculating residuals (with {num_iterations} iterations)...")
        for itr in tqdm(range(num_iterations), desc="Calculating residuals"):
            if use_window:
                res_Vbar.append( get_residuals(rad, Vbar[itr], rad, Vbar_means, Vbar_bands)[floor(4.0*samp_rate):ceil(6.0*samp_rate)] )
                res_MOND.append( get_residuals(rad, Vmond[itr], rad, Vmond_means, Vmond_bands)[floor(4.0*samp_rate):ceil(6.0*samp_rate)] )
                res_LCDM.append( get_residuals(rad, Vcdm[itr], rad, Vcdm_means, Vcdm_bands)[floor(4.0*samp_rate):ceil(6.0*samp_rate)] )
            else:
                res_Vbar.append( get_residuals(rad, Vbar[itr], rad, Vbar_means, Vbar_bands) )   # dim (after all appends) = itr x rad
                res_MOND.append( get_residuals(rad, Vmond[itr], rad, Vmond_means, Vmond_bands) )
                res_LCDM.append( get_residuals(rad, Vcdm[itr], rad, Vcdm_means, Vcdm_bands) )
            
        # Plot the generated mock data.
        if samp_rate == 10 and i == 32:
            print("Generating example plot...")
            fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2]})

            norm_noise = noise / Vmax
            ax0.errorbar(rad, Vbar[0], norm_noise, alpha=0.25, c='tab:red', fmt='o', capsize=2, label=r"$v_{\text{bar}}$")
            ax0.errorbar(rad, Vmond[0], norm_noise, alpha=0.25, c='mediumblue', fmt='o', capsize=2, label=r"$v_{\text{MOND}}$")
            ax0.errorbar(rad, Vcdm[0], norm_noise, alpha=0.25, c='tab:green', fmt='o', capsize=2, label=r"$v_{\Lambda CDM}$")

            ax0.plot(rad, Vbar_means, c='tab:red')
            ax0.plot(rad, Vmond_means, c='mediumblue')
            ax0.plot(rad, Vcdm_means, c='tab:green')

            ax0.set_ylabel(r"Velocities ($\times v_{\text{max}}$)")
            ax0.legend()
            ax0.grid()

            if use_window: r_plot = rad[floor(4.0*samp_rate):ceil(6.0*samp_rate)]
            else: r_plot = rad

            ax1.plot(r_plot, res_Vbar[0], c='tab:red', marker='o', alpha=0.3)
            ax1.plot(r_plot, res_MOND[0], c='mediumblue', marker='o', alpha=0.3)
            ax1.plot(r_plot, res_LCDM[0], c='tab:green', marker='o', alpha=0.3)
            ax1.grid()
            ax1.set_ylabel(r"Residuals ($\times v_{\text{max}}$)")

            ax1.set_xlabel("Radius (kpc)")
            plt.subplots_adjust(hspace=0.05)

            if use_window: fig.savefig(f"{fileloc}example_window.pdf")
            else: fig.savefig(f"{fileloc}example.pdf")
            plt.close()

            # raise ValueError("Test plot generated. Exiting...")
    
    # else:
    #     # Apply median filter.
    #     _, residuals     = med_filter(rad, v_werr, win_size=MF_size)
    #     _, residuals_Xft = med_filter(rad, Vraw_werr, win_size=MF_size)
    #     _, residuals_MOND = med_filter(rad, velocities, win_size=MF_size)


    # Transpose residuals arrays and extract required Xft residuals for DTW:
    # 3D array of size num_iterations x 2 (vel) x 100 (rad) --> 2 x num_iterations x 100 (rad).
    # res_dtw = np.transpose(residuals_Vbar, (1, 0, 2))
    # res_Xft = np.transpose(residuals_MOND, (1, 0, 2))[1]
    # MOND_res = np.transpose(residuals_MOND, (1, 0, 2))[1]

    if use_window: num_rad = ceil(5.5*samp_rate) - floor(4.5*samp_rate)

    """
    -----------------
    DTW on residuals.
    -----------------
    """
    if apply_DTW:
        print("Applying DTW...")
        for itr in range(num_iterations):
            MOND_cost = do_DTW(itr, num_rad, res_MOND, res_Vbar)
            MOND_costs[itr][i] = MOND_cost

        for itr in range(num_iterations):
            LCDM_cost = do_DTW(itr, num_rad, res_LCDM, res_Vbar)
            LCDM_costs[itr][i] = LCDM_cost


    """
    -----------------------------------
    Calculate correlation coefficients.
    -----------------------------------
    """
    # Interpolate residuals for potential use of derivatives in calculating naive correlation coefficients.
    if corr_rad:
        print("Calculating correlation coefficients...")
        v_werr = np.transpose( np.array([ Vbar, Vmond ]), (1, 0, 2) )   # dim = itr x 2 x rad
        if use_window: rad_corr_perc = corr_radii( num_iterations, num_rad, v_werr[:,:,floor(4.5*samp_rate):ceil(5.5*samp_rate)], use_window=use_window )
        else: rad_corr_perc = corr_radii( num_iterations, num_rad, v_werr )   # dim = 2 (Spearman, Pearson) x 3
        MOND_spearmans[i] = rad_corr_perc[0]
        MOND_pearsons[i]  = rad_corr_perc[1]

        Vnfw = np.transpose( np.array([ Vbar, Vcdm ]), (1, 0, 2) )
        if use_window: rad_corr_perc = corr_radii( num_iterations, num_rad, Vnfw[:,:,floor(4.5*samp_rate):ceil(5.5*samp_rate)], use_window=use_window )
        else: rad_corr_perc = corr_radii( num_iterations, num_rad, Vnfw )
        LCDM_spearmans[i] = rad_corr_perc[0]
        LCDM_pearsons[i]  = rad_corr_perc[1]

    jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem when GP is used.

"""
===============================
Calculate feature significance.
===============================
"""
if apply_DTW:
    MOND_perc = np.nanpercentile( MOND_costs,  [16.0, 50.0, 84.0], axis=0 )    # dim = 3 x num_noise
    LCDM_perc = np.nanpercentile( LCDM_costs,  [16.0, 50.0, 84.0], axis=0 )
    dtw_ftsig = get_significance(MOND_perc, LCDM_perc)
    if use_window: np.save(f"{fileloc}dtw_ftsig_window/num_samples={num_samples}", dtw_ftsig)
    else: np.save(f"{fileloc}dtw_ftsig/num_samples={num_samples}", dtw_ftsig)

if corr_rad:
    rad_ftsig = get_significance(np.transpose(MOND_pearsons), np.transpose(LCDM_pearsons))
    if use_window: np.save(f"{fileloc}rad_ftsig_window/num_samples={num_samples}", rad_ftsig)
    else: np.save(f"{fileloc}rad_ftsig/num_samples={num_samples}", rad_ftsig)

print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
