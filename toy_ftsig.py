#!/usr/bin/env python
# (C) 2024 Enoch Ko.
"""
Generate toy models of galaxy RC using arctan curves + gaussian features,
then apply same analysis of correlation coefficients + DTW on residuals
and create 2D significance histograms to better identify features in real data.
"""
import numpy as np
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt

import jax
import numpyro
import argparse

from resource import getrusage, RUSAGE_SELF

from utils_analysis.toy_gen import toy_gen
from utils_analysis.med_filter import med_filter
from utils_analysis.toy_GP import GP_fit, GP_residuals
from utils_analysis.dtw_utils import do_DTW
from utils_analysis.correlations import corr_radii

from multiprocessing import Pool

matplotlib.use("Agg")


"""
===================================
~~~~~~~~~~~~ MAIN LOOP ~~~~~~~~~~~~
===================================
"""
def toy_model(num_samples):
    # Initialize arrays for summary plots.
    rad_spearmans     = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
    rad_pearsons      = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
    rad_Xft_spearmans = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
    rad_Xft_pearsons  = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]

    # win_spearmans     = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
    # win_pearsons      = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
    # win_Xft_spearmans = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]
    # win_Xft_pearsons  = [ [ [] for _ in range(num_noise-1) ] for _ in range(3) ]

    dtw_costs,  Xft_costs  = np.zeros((num_iterations, num_noise)), np.zeros((num_iterations, num_noise))
    # dtw_window, Xft_window = np.zeros((num_iterations, num_noise)), np.zeros((num_iterations, num_noise))


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


    for i in range(num_noise):
        # if i == 0:
        print(f"Analyzing toy model with {num_samples} data points and {num_iterations} iterations per noise level...")

        noise = noise_arr[i]

        # Define galaxy radius (units ~kpc; excluding the point r=0).
        rad = np.linspace(10., 0., num_samples, endpoint=False)[::-1]
        num_rad = len(rad)
        
        # Generate toy RCs with residuals (Vraw = w/o ft, Vraw_werr = w/ noise; velocitites = w/ ft, v_werr = w/ noise).
        _, Vraw, velocities, Vraw_werr, v_werr, residuals, res_Xft = toy_gen(rad, bump_loc, bump_size, bump_sigma, noise, num_iterations)

        # Vobs (w/ feature) residuals if it's generated perfectly by MOND.
        MOND_res = (velocities[:,1,:] - Vraw[:,1,:])


        # Apply simple median filter.
        if use_MF:            
            _, residuals     = med_filter(rad, v_werr, size=MF_size)
            _, residuals_Xft = med_filter(rad, Vraw_werr, size=MF_size)
            _, residuals_MOND = med_filter(rad, velocities, size=MF_size)

        # Apply GP regression.
        if use_GP:
            residuals, residuals_Xft, residuals_MOND = [], [], []

            # pred_means, pred_bands = GP_fit(args, rad, v_werr[0], rad)
            # Xft_means, Xft_bands = GP_fit(args, rad, Vraw_werr[0], rad)
            # pred_means_MOND, pred_bands_MOND = GP_fit(args, rad, velocities[0], rad)

            for itr in range(num_iterations):
                pred_means, pred_bands = GP_fit(args, rad, v_werr[itr], rad)
                Xft_means, Xft_bands = GP_fit(args, rad, Vraw_werr[itr], rad)
                pred_means_MOND, pred_bands_MOND = GP_fit(args, rad, velocities[itr], rad)

                residuals.append( GP_residuals(rad, v_werr[itr], rad, pred_means, pred_bands) )
                residuals_Xft.append( GP_residuals(rad, Vraw_werr[itr], rad, Xft_means, Xft_bands) )
                residuals_MOND.append( GP_residuals(rad, velocities[itr], rad, pred_means_MOND, pred_bands_MOND) )

        # Transpose residuals arrays and extract required Xft residuals for DTW:
        # Transpose: 3D array of size num_iterations x 2 (vel) x 100 (rad) --> 2 x num_iterations x 100 (rad).
        res_dtw = np.transpose(residuals, (1, 0, 2))

        if use_MF or use_GP:
            res_Xft = np.transpose(residuals_Xft, (1, 0, 2))[1]
            MOND_res = np.transpose(residuals_MOND, (1, 0, 2))[1]


        # Interpolate residuals for potential use of derivatives in calculating naive correlation coefficients.
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
        # DTW analyses on full RCs.
        for itr in range(num_iterations):
            Xft_cost = do_DTW(itr, num_rad, res_Xft, MOND_res, window=False)
            Xft_costs[itr][i] = Xft_cost
            norm_cost = do_DTW(itr, num_rad, res_dtw[1], MOND_res, window=False)
            dtw_costs[itr][i] = norm_cost

            # DTW along a small window (length 1) around the feature.
            # win_Xft_cost = do_DTW(itr, window_size, res_Xft, MOND_res, window=True)
            # Xft_window[itr][i] = win_Xft_cost
            # win_cost = do_DTW(itr, window_size, res_dtw[1], MOND_res, window=True)
            # dtw_window[itr][i] = win_cost


        """
        -----------------------------------
        Calculate correlation coefficients.
        -----------------------------------
        """
        # Correlation plots using spheres of increasing radius
        if noise != 0.0:
            for der in range(3):
                rad_corr_perc = corr_radii( num_iterations, der, num_rad, res_fits, v_werr )
                rad_spearmans[der][i-1] = rad_corr_perc[0]
                rad_pearsons[der][i-1]  = rad_corr_perc[1]

                res_fits_temp, v_werr_temp = [ np.squeeze(np.array(res_fits)[:,:,0]), np.squeeze(res_Xft_fits, axis=2) ], [ v_werr[:,0,:], Vraw_werr[:,1,:] ]
                res_fits_temp, v_werr_temp = np.transpose(res_fits_temp, (1, 2, 0, 3)), np.transpose(v_werr_temp, (1, 0, 2))
                rad_corr_perc = corr_radii( num_iterations, der, num_rad, res_fits_temp, v_werr_temp )
                rad_Xft_spearmans[der][i-1] = rad_corr_perc[0]
                rad_Xft_pearsons[der][i-1]  = rad_corr_perc[1]

        # Correlation plots using windows of length max{1 * Reff, 5 data points}.
        # if noise != 0.0:
        #     for der in range(3):
        #         win_corr_perc = corr_window( num_iterations, der, num_rad, res_fits, v_werr, window_size )
        #         win_spearmans[der][i-1] = win_corr_perc[0]
        #         win_pearsons[der][i-1]  = win_corr_perc[1]

        #         res_fits_temp, v_werr_temp = [ np.squeeze(np.array(res_fits)[:,:,0]), np.squeeze(res_Xft_fits, axis=2) ], [ v_werr[:,0,:], Vraw_werr[:,1,:] ]
        #         res_fits_temp, v_werr_temp = np.transpose(res_fits_temp, (1, 2, 0, 3)), np.transpose(v_werr_temp, (1, 0, 2))
        #         win_corr_perc = corr_window( num_iterations, der, num_rad, res_fits_temp, v_werr_temp, window_size )
        #         win_Xft_spearmans[der][i-1] = win_corr_perc[0]
        #         win_Xft_pearsons[der][i-1]  = win_corr_perc[1]

        jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem.

    print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)

    # Summary arrays.
    dtw_costs   = np.nanpercentile( dtw_costs,  [16.0, 50.0, 84.0], axis=0 )
    Xft_costs   = np.nanpercentile( Xft_costs,  [16.0, 50.0, 84.0], axis=0 )
    # dtw_window  = np.nanpercentile( dtw_window, [16.0, 50.0, 84.0], axis=0 )
    # Xft_window  = np.nanpercentile( Xft_window, [16.0, 50.0, 84.0], axis=0 )

    rad_spearmans, rad_pearsons = np.array(rad_spearmans), np.array(rad_pearsons)
    rad_Xft_spearmans, rad_Xft_pearsons = np.array(rad_Xft_spearmans), np.array(rad_Xft_pearsons)
    # win_spearmans, win_pearsons = np.array(win_spearmans), np.array(win_pearsons)
    # win_Xft_spearmans, win_Xft_pearsons = np.array(win_Xft_spearmans), np.array(win_Xft_pearsons)

    # Calculate feature significance.
    dtw_ftsig = abs( (dtw_costs[1] - Xft_costs[1]) / ((Xft_costs[2] - Xft_costs[0]) / 2) )
    rad_ftsig = abs( (rad_pearsons[1] - rad_Xft_pearsons[1]) / ((rad_Xft_pearsons[2] - rad_Xft_pearsons[0]) / 2) )

    # Return feature significance in one big array.
    return dtw_ftsig, rad_ftsig

    
if __name__ == '__main__':
    # Create array of sampling rates to sample from (change the index from 0 to 19 manually, or figure out how to use parallel processes...).
    samp_rates = np.linspace(3, 20, 18, endpoint=True, dtype=int)
    num_samples = samp_rates * 10

    # Switches for running different parts of the analysis.
    use_MF      = False
    use_GP      = False

    # File names for different analysis methods.
    fileloc = "/mnt/users/koe/plots/toy_model/2Dsig/"
    if use_MF and use_GP:
        raise Exception("GP and median filter cannot be used simultaneously!")
    elif use_MF:
        fileloc += "use_MF/"
        MF_size = 20    # Define window size for median filter (if used).
    elif use_GP:
        fileloc += "use_GP/"

    # window_size = 11

    # Print report of analyses used and some corresponding variables.
    if use_MF: print(f"\nComputing residuals with median filter (window length = {MF_size}).")
    if use_GP: print("\nComputing residuals with Gaussian process.")

    # Parameters for Gaussian bump (fixed feature) and noise (varying amplitudes).
    bump_size  = 20.0   # Defined in terms of percentage of max(Vbar)
    bump_loc   = 5.0
    bump_FWHM  = 0.5
    bump_sigma = bump_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    if use_GP:
        noise_arr = np.linspace(0.0, bump_size/2, 51, endpoint=True)
        num_iterations = 50
    else:
        noise_arr = np.linspace(0.0, bump_size/2, 101, endpoint=True)
        num_iterations = 200
    num_noise = len(noise_arr)

    print(f"\nCorrelating RCs with {num_noise} noise levels from 0.0 to 0.5 * ft height.")


    # Execute main loop as parallel processes.
    with Pool() as p:
        # Parallel process for-loop to obtain ft significance, each num_samples iteration returns: [ dtw_ftsig, rad_ftsig ].
        results = p.map(toy_model, num_samples)

    # Transpose results to dimensions: ftsig (2) x num_samples (20) x num_noise (51 or 101).
    ft_significance = np.transpose(results, (1, 0, 2))

    # Plot 2D histograms.
    ft_significance = np.split(ft_significance, 2)
    fig_names = [ "Feature significance: DTW", "Feature significance: Pearson correlation" ]
    fig_loc = [ "dtw_hist.png", "corr_hist.png" ]

    for idx, ftsig in enumerate(ft_significance):
        plt.title(fig_names[idx])
        plt.xlabel("Number of data points")
        plt.ylabel("Noise / ft height ratio")
        plt.hist2d(ftsig[:, 0], ftsig[0, :], bins=30, cmap="viridis")
        plt.colorbar(label="Ft significance")
        plt.savefig(fileloc+fig_loc[idx], dpi=300, bbox_inches="tight")
        plt.close()
