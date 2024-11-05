#!/usr/bin/env python
# (C) 2024 Enoch Ko.
"""
Generate toy models of galaxy RC using arctan curves + gaussian features,
then apply same analysis of correlation coefficients + DTW on residuals
to better understand the effect/restriction of sampling rates, feature sizes and uncertainties.
"""
import numpy as np
from scipy import interpolate
import matplotlib

import jax
import numpyro
import argparse
from resource import getrusage, RUSAGE_SELF
# from tqdm import tqdm

from utils_analysis.toy_gen import toy_gen
from utils_analysis.med_filter import med_filter
from utils_analysis.toy_GP import GP_fit, GP_residuals
from utils_analysis.dtw_utils import do_DTW
from utils_analysis.correlations import corr_radii

matplotlib.use("Agg")


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
        parser = argparse.ArgumentParser(description="Toy Model")
        parser.add_argument("--ft-width", default=10, type=float)
        parser.add_argument("--samp-idx", type=int)
        args = parser.parse_args()
    
    return args

def get_fname( file_FWHM:float, use_GP:bool=False ):
    """Get file name for saving plots and arrays"""
    fname = f"/mnt/users/koe/plots/toy_model/2Dsig/FWHM={file_FWHM}/"
    if use_GP: fname += "use_GP/"
    else:      fname += "use_MF/"

    print(fname)
    return fname


"""
------------------------------------------------------------
Initialize parameters for mock data generation and analysis.
------------------------------------------------------------
"""
# Switches for running different parts of the analysis.
use_GP    = False    # Note: A median filter is used if use_GP = False.
apply_DTW = True
corr_rad  = True

args = set_args( use_GP )
    
# Parameters for Gaussian bump (fixed feature) and noise (varying amplitudes).
bump_size  = 10.0   # Defined in terms of percentage of max(Vbar)
bump_size *= 2.0
bump_loc   = 5.0
file_FWHM  = args.ft_width
bump_FWHM  = file_FWHM / 10
bump_sigma = bump_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

print(f"Correlating RCs with features of FWHM = {bump_FWHM}")

# Generate noise from 1 / uniform height array (less samples for GP due to demanding runtime).
if use_GP:
    height_arr = np.linspace(100.0, 2.0, 49, endpoint=True)
    noise_arr = bump_size / height_arr
    num_iterations = 50
else:
    height_arr = np.linspace(100.0, 2.0, 99, endpoint=True)
    noise_arr = bump_size / height_arr
    num_iterations = 200

num_noise = len(noise_arr)


# Initialize arrays for summary arrays.
rad_spearmans     = [ [ [] for _ in range(num_noise) ] for _ in range(3) ]
rad_pearsons      = [ [ [] for _ in range(num_noise) ] for _ in range(3) ]
rad_Xft_spearmans = [ [ [] for _ in range(num_noise) ] for _ in range(3) ]
rad_Xft_pearsons  = [ [ [] for _ in range(num_noise) ] for _ in range(3) ]

dtw_costs,  Xft_costs  = np.zeros((num_iterations, num_noise)), np.zeros((num_iterations, num_noise))
dtw_window, Xft_window = np.zeros((num_iterations, num_noise)), np.zeros((num_iterations, num_noise))


# Create array of sampling rates to sample from.
# To run this as a loop in bash: for i in {0..29}; do addqueue -q cmb -c "1-3 days" -n 1 -m 8 mock_data.py --ft-width {?} --samp-idx $i; done
samp_idx    = args.samp_idx
samp_rate   = np.linspace(1, 30, 30, endpoint=True, dtype=int)[samp_idx]
num_samples = samp_rate * 10

if not use_GP: MF_size = int(max( 5, bump_FWHM * samp_rate * 2 ))


# Print report of analyses used and some corresponding variables.
print(f"[samp_rate = {samp_rate}] Running toy_model.py with the following methods:")
if use_GP:    print(" - Gaussian process")
else:         print(f" - Median filter (window length = {MF_size})")
if apply_DTW: print(" - Dynamic time warping")
if corr_rad:  print(" - Correlation coefficients (increasing radii)")


# Simple code for calculating feature significance by comparing costs/correlation coefficients
# between (Vobs w/ ft + Vbar w/ ft) and (Vobs W/O ft + Vbar w/ ft).
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


"""
===================================
~~~~~~~~~~~~ MAIN LOOP ~~~~~~~~~~~~
===================================
"""
fileloc = get_fname( file_FWHM, use_GP )

for i in range(num_noise):
    if i%10 == 0 or use_GP:
        if i == 0:
            print(f"\nRunning iteration {i+1}/{num_noise} (with {num_iterations} iterations per noise level)...")
        else:
            print(f"\nRunning iteration {i+1}/{num_noise}...")

    noise = noise_arr[i]

    # Define galaxy radius (units ~kpc; excluding the point r=0).
    rad = np.linspace(10., 0., num_samples, endpoint=False)[::-1]
    num_rad = len(rad)
    
    # Generate toy RCs with residuals (Vraw = w/o ft, Vraw_werr = w/ noise; velocitites = w/ ft, v_werr = w/ noise).
    bump, Vraw, velocities, Vraw_werr, v_werr, residuals, res_Xft = toy_gen(rad, bump_loc, bump_size, bump_sigma, noise, num_iterations)

    # Vobs (w/ feature) residuals if it's generated perfectly by MOND.
    MOND_res = (velocities[:,1,:] - Vraw[:,1,:])


    if use_GP:
        # Apply GP regression.
        residuals, residuals_Xft, residuals_MOND = [], [], []

        for itr in range(num_iterations):
            pred_means, pred_bands = GP_fit(args, rad, v_werr[itr], rad)
            Xft_means, Xft_bands = GP_fit(args, rad, Vraw_werr[itr], rad)
            pred_means_MOND, pred_bands_MOND = GP_fit(args, rad, velocities[itr], rad)

            residuals.append( GP_residuals(rad, v_werr[itr], rad, pred_means, pred_bands) )
            residuals_Xft.append( GP_residuals(rad, Vraw_werr[itr], rad, Xft_means, Xft_bands) )
            residuals_MOND.append( GP_residuals(rad, velocities[itr], rad, pred_means_MOND, pred_bands_MOND) )
    
    else:
        # Apply median filter.
        _, residuals     = med_filter(rad, v_werr, win_size=MF_size)
        _, residuals_Xft = med_filter(rad, Vraw_werr, win_size=MF_size)
        _, residuals_MOND = med_filter(rad, velocities, win_size=MF_size)


    # Transpose residuals arrays and extract required Xft residuals for DTW:
    # Transpose: 3D array of size num_iterations x 2 (vel) x 100 (rad) --> 2 x num_iterations x 100 (rad).
    res_dtw = np.transpose(residuals, (1, 0, 2))
    res_Xft = np.transpose(residuals_Xft, (1, 0, 2))[1]
    MOND_res = np.transpose(residuals_MOND, (1, 0, 2))[1]


    """
    -----------------
    DTW on residuals.
    -----------------
    """
    if apply_DTW:
        # DTW analyses on full RCs.
        for itr in range(num_iterations):
            Xft_cost = do_DTW(itr, num_rad, res_Xft, MOND_res, window=False)
            Xft_costs[itr][i] = Xft_cost

        for itr in range(num_iterations):
            norm_cost = do_DTW(itr, num_rad, res_dtw[1], MOND_res, window=False)
            dtw_costs[itr][i] = norm_cost


    """
    -----------------------------------
    Calculate correlation coefficients.
    -----------------------------------
    """
    # Interpolate residuals for potential use of derivatives in calculating naive correlation coefficients.
    if corr_rad:
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

        # Correlation plots using spheres of increasing radius.
        for der in range(3):
            rad_corr_perc = corr_radii( num_iterations, der, num_rad, res_fits, v_werr )
            rad_spearmans[der][i] = rad_corr_perc[0]
            rad_pearsons[der][i]  = rad_corr_perc[1]

            res_fits_temp, v_werr_temp = [ np.squeeze(np.array(res_fits)[:,:,0]), np.squeeze(res_Xft_fits, axis=2) ], [ v_werr[:,0,:], Vraw_werr[:,1,:] ]
            res_fits_temp, v_werr_temp = np.transpose(res_fits_temp, (1, 2, 0, 3)), np.transpose(v_werr_temp, (1, 0, 2))
            rad_corr_perc = corr_radii( num_iterations, der, num_rad, res_fits_temp, v_werr_temp )
            rad_Xft_spearmans[der][i] = rad_corr_perc[0]
            rad_Xft_pearsons[der][i]  = rad_corr_perc[1]

    jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem when GP is used.

"""
===============================
Calculate feature significance.
===============================
"""
if apply_DTW:
    dtw_costs   = np.nanpercentile( dtw_costs,  [16.0, 50.0, 84.0], axis=0 )
    Xft_costs   = np.nanpercentile( Xft_costs,  [16.0, 50.0, 84.0], axis=0 )
    dtw_ftsig = get_significance(dtw_costs, Xft_costs, rhos=False)
    np.save(f"{fileloc}dtw_ftsig/num_samples={num_samples}", dtw_ftsig)

if corr_rad:
    rad_spearmans, rad_pearsons = np.array(rad_spearmans), np.array(rad_pearsons)
    rad_Xft_spearmans, rad_Xft_pearsons = np.array(rad_Xft_spearmans), np.array(rad_Xft_pearsons)
    rad_ftsig = get_significance(rad_pearsons, rad_Xft_pearsons, rhos=True)
    np.save(f"{fileloc}rad_ftsig/num_samples={num_samples}", rad_ftsig)

print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
