#!/usr/bin/env python
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
from tqdm import tqdm

from utils_analysis.toy_gen import toy_gen
from utils_analysis.toy_GP import GP_fit, GP_residuals
from utils_analysis.dtw_utils import do_DTW

matplotlib.use("Agg")
memory_usage = []   # Track memory usage throughout programme.

# Switches for running different parts of the analysis.
use_GP      = False
apply_DTW   = True
corr_radii  = False     # SET FALSE: Code to be fixed for noise iterations.
corr_window = False     # SET FALSE: Code to be fixed for noise iterations.
make_plots  = True


if use_GP:
    fileloc = "/mnt/users/koe/plots/toy_model/use_GP/"
else:
    fileloc = "/mnt/users/koe/plots/toy_model/"

colors = [ 'k', 'tab:red' ]
labels = [ 'Vobs', 'Vbar' ]
deriv_dir = [ "d0", "d1", "d2" ]
color_bar = "orange"


"""
Parameters for Gaussian bump (fixed feature) and noise (varying amplitudes).
"""
bump_size   = 20.0   # Defined in terms of percentage of max(Vbar)
bump_loc    = 5.0
bump_FWHM   = 0.5
bump_sigma  = bump_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

rad = np.linspace(10., 0., 100, endpoint=False)[::-1]   # Defined this way to exclude the starting point r=0.
num_rad = len(rad)

noise_arr = np.linspace(0.0, bump_size, 201, endpoint=True)
num_noise = len(noise_arr)
num_iterations = 500    # Iterations per noise level (for smoothing out DTW costs and correlations in final plots).


# Initialize arrays for summary plots.
win_spearmans = np.zeros((num_iterations, num_noise))
win_pearsons = np.copy(win_spearmans)
dtw_costs, Xft_costs = np.copy(win_spearmans), np.copy(win_spearmans)
dtw_window, Xft_window = np.copy(win_spearmans), np.copy(win_spearmans)


"""
Initialize args for GP (if used).
"""
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
MAIN LOOP.
"""
for i in range(num_noise):
    if i%10 == 0:
        print(f"Running iteration {i}/{num_noise}...")
    
    noise = noise_arr[i]
    bump, Vraw, Vraw_werr, v_werr, residuals, res_Xft, MOND_res = toy_gen(rad, bump_loc, bump_size, bump_sigma, noise, num_iterations)

    # Transpose residuals array for DTW:
    # 3D array of size num_iterations x 2 (vel) x 100 (rad) --> 2 x num_iterations x 100 (rad).
    res_dtw = np.transpose(residuals, (1, 0, 2))


    if use_GP:
        residuals, residuals_Xft = [], []
        file_names = [ fileloc+f"corner_plots/ratio={round(noise/bump_size, 2)}.png",
                               fileloc+f"GP_fits/ratio={round(noise/bump_size, 2)}.png" ]
        Xft_fnames = [ fileloc+f"corner_plots/Xft/ratio={round(noise/bump_size, 2)}.png",
                               fileloc+f"GP_fits/Xft/ratio={round(noise/bump_size, 2)}.png" ]
        
        if noise in noise_arr[::10]:
            pred_means, pred_bands = GP_fit(args, rad, v_werr[0], rad, make_plots=make_plots, file_name=file_names[0])
            Xft_means, Xft_bands = GP_fit(args, rad, Vraw_werr[0], rad, make_plots=make_plots, file_name=Xft_fnames[0])
        else:
            pred_means, pred_bands = GP_fit(args, rad, v_werr[0], rad)
            Xft_means, Xft_bands = GP_fit(args, rad, Vraw_werr[0], rad)

        for itr in range(num_iterations):
            if itr == 0 and noise in noise_arr[::10]:
                residuals.append( GP_residuals(rad, v_werr[itr], rad, pred_means, pred_bands,
                                            make_plots=make_plots, file_name=file_names[1]) )
                residuals_Xft.append( GP_residuals(rad, Vraw_werr[itr], rad, Xft_means, Xft_bands,
                                                make_plots=make_plots, file_name=Xft_fnames[1]) )
            else:
                residuals.append( GP_residuals(rad, v_werr[itr], rad, pred_means, pred_bands) )
                residuals_Xft.append( GP_residuals(rad, Vraw_werr[itr], rad, Xft_means, Xft_bands) )

        res_dtw = np.transpose(residuals, (1, 0, 2))
        residuals_Xft = np.transpose(residuals_Xft, (1, 0, 2))
        res_Xft = np.array(residuals_Xft[0])
        MOND_res = np.array(residuals_Xft[1])


    if corr_radii or corr_window:
        # Interpolate the residuals with cubic Hermite spline splines.
        res_fits = []
        for itr in range(num_iterations):
            v_d0, v_d1, v_d2 = [], [], []
            for v_comp in residuals[itr]:
                v_d0.append(interpolate.pchip_interpolate(rad, v_comp, rad))
                # v_d1.append(interpolate.pchip_interpolate(rad, v_comp, rad, der=1))
                # v_d2.append(interpolate.pchip_interpolate(rad, v_comp, rad, der=2))
            res_fits.append( v_d0 )


    """
    DTW on GP residuals.
    """
    if apply_DTW:
        # print("Warping time dynamically... or something like that...")

        """
        DTW analyses on full RCs.
        """
        for itr in range(num_iterations):
            if itr==0 and noise in noise_arr[::10]:
                file_names = [ fileloc+f"Xft_matrix/ratio={round(noise/bump_size, 2)}.png",
                               fileloc+f"Xft_alignment/ratio={round(noise/bump_size, 2)}.png" ]
                Xft_cost = do_DTW(itr, num_rad, res_Xft, MOND_res, window=False, make_plots=make_plots, file_names=file_names)
                Xft_costs[itr][i] = Xft_cost
            else:
                Xft_cost = do_DTW(itr, num_rad, res_Xft, MOND_res, window=False)
                Xft_costs[itr][i] = Xft_cost

        for itr in range(num_iterations):
            if itr==0 and noise in noise_arr[::10]:
                file_names = [ fileloc+f"dtw_matrix/ratio={round(noise/bump_size, 2)}.png",
                               fileloc+f"dtw_alignment/ratio={round(noise/bump_size, 2)}.png" ]
                norm_cost = do_DTW(itr, num_rad, res_dtw[0], MOND_res, window=False, make_plots=make_plots, file_names=file_names)
                dtw_costs[itr][i] = norm_cost
            else:
                norm_cost = do_DTW(itr, num_rad, res_dtw[0], MOND_res, window=False)
                dtw_costs[itr][i] = norm_cost


        """
        DTW on RCs within window (of length 1) around feature.
        """
        # DTW along a small window (length 1) around the feature.
        window_size = 11
        for itr in range(num_iterations):
            if itr==0 and noise in noise_arr[::10]:
                file_names = [ fileloc+f"dtw_window/Xft_matrix/ratio={round(noise/bump_size, 2)}.png",
                               fileloc+f"dtw_window/Xft_alignment/ratio={round(noise/bump_size, 2)}.png" ]
                win_cost = do_DTW(itr, window_size, res_Xft, MOND_res, window=True, make_plots=make_plots, file_names=file_names)
                Xft_window[itr][i] = win_cost
            else:
                win_cost = do_DTW(itr, window_size, res_Xft, MOND_res, window=True)
                Xft_window[itr][i] = win_cost

        for itr in range(num_iterations):
            if itr==0 and noise in noise_arr[::10]:
                file_names = [ fileloc+f"dtw_window/matrix/ratio={round(noise/bump_size, 2)}.png",
                               fileloc+f"dtw_window/alignment/ratio={round(noise/bump_size, 2)}.png" ]
                win_cost = do_DTW(itr, window_size, res_dtw[0], MOND_res, window=True, make_plots=make_plots, file_names=file_names)
                dtw_window[itr][i] = win_cost
            else:
                win_cost = do_DTW(itr, window_size, res_dtw[0], MOND_res, window=True)
                dtw_window[itr][i] = win_cost


    """
    ----------------------------------------------------
    Correlation plots using spheres of increasing radius
    ----------------------------------------------------
    """
    if corr_radii and noise != 0.0:
        # print("Computing correlation coefficients with increasing radii...")
        # Correlate Vobs and Vbar (d0, d1, d2) as a function of (maximum) radius, i.e. spheres of increasing r.
        rad_corr = [ [[], []], [[], []], [[], []] ]
        for der in range(1):
            for j in range(10, num_rad):
                rad_corr[der][0].append(stats.spearmanr(res_fits[der][0][:j], res_fits[der][1][:j])[0])
                rad_corr[der][1].append(stats.pearsonr(res_fits[der][0][:j], res_fits[der][1][:j])[0])

        # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
        bar_ratio = []
        for rd in range(num_rad):
            bar_ratio.append(sum(v_werr[1][:rd]/v_werr[0][:rd]) / (rd+1))

        rad_spearman = stats.spearmanr(rad_corr[der][0], bar_ratio[10:])[0]
        rad_pearson = stats.pearsonr(rad_corr[der][1], bar_ratio[10:])[0]


        """
        Plot GP fits, residuals (+ PCHIP) and correlations.
        """
        der = 0
        if make_plots and noise in noise_arr[::10]:
            fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
            fig.set_size_inches(7, 7)

            ax0.set_title("Correlation by increasing radii: Toy model")
            ax0.set_ylabel("Normalised velocities")
            ax2.set_xlabel("Radii (kpc)")

            for i in range(2):
                ax0.errorbar(rad, v_werr[i], noise/100.0, color=colors[i], alpha=0.3, capsize=3, fmt="o", ls="none")
                ax0.plot(rad, Vraw[i], color=colors[i], label=labels[i])
                if der == 0:
                    ax1.scatter(rad, residuals[i], color=colors[i], alpha=0.3)
                ax1.plot(rad, res_fits[der][i], color=colors[i], alpha=0.7, label=labels[i])

            ax0.plot(rad, bump, '--', label="Feature")
            ax0.legend(loc="upper left", bbox_to_anchor=(1,1))
            ax0.grid()

            ax1.legend(loc="upper left", bbox_to_anchor=(1,1))
            ax1.grid()


            # Plot correlations and Vbar/Vobs.
            ax2.plot(rad[10:], rad_corr[der][0], color='mediumblue', label=r"Spearman $\rho$")
            ax2.plot(rad[10:], rad_corr[der][1], ':', color='mediumblue', label=r"Pearson $\rho$")
            ax2.plot([], [], ' ', label=r"$\rho_s=$"+str(round(rad_spearman, 3))+r", $\rho_p=$"+str(round(rad_pearson, 3)))
        
            ax5 = ax2.twinx()
            ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
            ax5.plot(rad[10:], bar_ratio[10:], '--', color=color_bar, label="Vbar/Vobs")
            ax5.tick_params(axis='y', labelcolor=color_bar)
        
            ax2.legend(loc="upper left", bbox_to_anchor=(1.11, 1))
            ax2.grid()

            plt.subplots_adjust(hspace=0.05)
            fig.savefig(fileloc+f"radii_{deriv_dir[der]}/ratio={round(noise/bump_size, 2)}.png", dpi=300, bbox_inches="tight")
            plt.close()


    """
    -----------------------------------------------------------------------
    Correlation plots using windows of length max{1 * Reff, 5 data points}.
    -----------------------------------------------------------------------
    """
    if corr_window and noise != 0.0:
        # print("Computing correlation coefficients with moving window...")
        # Correlate Vobs and Vbar (d0, d1, d2) along a moving window of length 1 * kpc.
        wmax = num_rad - 5
        win_corr = [ [[], []], [[], []], [[], []] ]
        for der in range(1):
            for j in range(5, wmax):
                jmin, jmax = j - 5, j + 5
                win_corr[der][0].append(stats.spearmanr(res_fits[der][0][jmin:jmax], res_fits[der][1][jmin:jmax])[0])
                win_corr[der][1].append(stats.pearsonr(res_fits[der][0][jmin:jmax], res_fits[der][1][jmin:jmax])[0])

        mid_pt = math.floor( len(win_corr[der][0]) / 2 )
        win_spearmans.append(win_corr[der][0][mid_pt])
        win_pearsons.append(win_corr[der][1][mid_pt])
            
        # Compute average baryonic dominance (using Vobs from SPARC data) in moving window.
        wbar_ratio = []
        for j in range(5, wmax):
            wbar_ratio.append( sum( v_werr[1][j-5:j+5] / v_werr[0][j-5:j+5] ) / 11 )

        win_spearman = stats.spearmanr(win_corr[der][0], wbar_ratio)[0]
        win_pearson = stats.pearsonr(win_corr[der][1], wbar_ratio)[0]
        

        # Plot corrletaions as 1 main plot + 1 subplot, using only Vobs from data for Vbar/Vobs.
        if make_plots and noise in noise_arr[::10]:
            fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
            fig1.set_size_inches(7, 7)
            ax0.set_title("Moving window correlation: Toy model")
            ax0.set_ylabel("Normalised velocities")
            ax2.set_xlabel("Radii (kpc)")
            for i in range(2):
                ax0.errorbar(rad, v_werr[i], noise/100.0, color=colors[i], alpha=0.3, capsize=3, fmt="o", ls="none")
                ax0.plot(rad, Vraw[i], color=colors[i], label=labels[i])
                if der == 0:
                    ax1.scatter(rad, residuals[i], color=colors[i], alpha=0.3)
                ax1.plot(rad, res_fits[der][i], color=colors[i], alpha=0.7, label=labels[i])

            ax0.plot(rad, bump, '--', label="Feature")
            ax0.legend(loc="upper left", bbox_to_anchor=(1,1))
            ax0.grid()

            ax1.legend(loc="upper left", bbox_to_anchor=(1,1))
            ax1.grid()

            # ax2.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
            ax2.set_xlabel('Radius (kpc)')
            ax2.set_ylabel("Correlations")

            # Plot correlations and Vbar/Vobs.
            ax2.plot(rad[5:wmax], win_corr[der][0], color='mediumblue', label=r"Spearman $\rho$")
            ax2.plot(rad[5:wmax], win_corr[der][1], ':', color='mediumblue', label=r"Pearson $\rho$")
            ax2.plot([], [], ' ', label=r"$\rho_s=$"+str(round(win_spearman, 3))+r", $\rho_p=$"+str(round(win_pearson, 3)))
        
            ax5 = ax2.twinx()
            ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
            ax5.plot(rad[10:wmax], wbar_ratio[5:], '--', color=color_bar, label="Vbar/Vobs")
            ax5.tick_params(axis='y', labelcolor=color_bar)
            
            ax2.legend(loc="upper left", bbox_to_anchor=(1.11, 1))
            ax2.grid()

            plt.subplots_adjust(hspace=0.05)
            fig1.savefig(fileloc+f"window_{deriv_dir[der]}/ratio={round(noise/bump_size, 2)}.png", dpi=300, bbox_inches="tight")
            plt.close()
        
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

if apply_DTW:
    half_noise = math.ceil( num_noise / 2 )

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


    plt.title("Normalized DTW alignment costs")
    plt.ylabel("Normalized DTW costs")
    plt.xlabel("Noise / feature height")
    plt.plot(bump_ratio[:half_noise], dtw_window[1][:half_noise], color='tab:blue', label="Costs w/ feature")
    plt.fill_between(bump_ratio[:half_noise], dtw_window[0][:half_noise], dtw_window[2][:half_noise], color='tab:blue', alpha=0.2)
    plt.plot(bump_ratio[:half_noise], Xft_window[1][:half_noise], '--', color='red', label="Costs w/o feature")
    plt.fill_between(bump_ratio[:half_noise], Xft_window[0][:half_noise], Xft_window[2][:half_noise], color='red', alpha=0.2)

    plt.legend()
    plt.savefig(fileloc+"dtw_window/dtwVnoise.png", dpi=300, bbox_inches="tight")
    plt.close()


    plt.title("Normalized DTW alignment costs")
    plt.ylabel("Normalized DTW costs")
    plt.xlabel("Noise / feature height")
    plt.plot(bump_ratio, dtw_window[1], color='tab:blue', label="Costs w/ feature")
    plt.fill_between(bump_ratio, dtw_window[0], dtw_window[2], color='tab:blue', alpha=0.2)
    plt.plot(bump_ratio, Xft_window[1], '--', color='red', label="Costs w/o feature")
    plt.fill_between(bump_ratio, Xft_window[0], Xft_window[2], color='red', alpha=0.2)

    plt.legend()
    plt.savefig(fileloc+"dtw_window/dtwVnoise_FULL.png", dpi=300, bbox_inches="tight")
    plt.close()


if corr_window:
    plt.title("Correlation coefficients at peak of feature")
    plt.ylabel("Correlation coefficients")
    plt.xlabel("Noise / feature height")
    plt.plot(bump_ratio[1:101], win_spearmans[:100], color='mediumblue', label=r"Spearman $\rho$")
    plt.plot(bump_ratio[1:101], win_pearsons[:100], '--', color='mediumblue', label=r"Pearson $\rho$")

    plt.legend()
    plt.savefig(fileloc+"corrVnoise.png", dpi=300, bbox_inches="tight")
    plt.close()


    plt.title("Correlation coefficients at peak of feature")
    plt.ylabel("Correlation coefficients")
    plt.xlabel("Noise / feature height")
    plt.plot(bump_ratio[1::], win_spearmans, color='mediumblue', label=r"Spearman $\rho$")
    plt.plot(bump_ratio[1::], win_pearsons, '--', color='mediumblue', label=r"Pearson $\rho$")

    plt.legend()
    plt.savefig(fileloc+"corrVnoise_FULL.png", dpi=300, bbox_inches="tight")
    plt.close()
