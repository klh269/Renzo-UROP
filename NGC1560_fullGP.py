#!/usr/bin/env python
"""
Correlation coefficients + dynamic time warping on GP residuals of NGC1560_Stacy,
a dataset obtained from digitizing Stacy's NGC 1560 plot with Plot Digitizer;
taking into account uncertainties (Vbar) and Vobs scattering (errV).

This is a modified version of the original analyze_NGC1560.py code.
Here we fit a new GP for each sample (instead of only one for all 1000 samples).

The following paper (Broeils 1992) analyzes NGC 1560 in some detail, thus might be useful:
https://articles.adsabs.harvard.edu/pdf/1992A%26A...256...19B
"""
import pandas as pd
import argparse
from resource import getrusage, RUSAGE_SELF
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
from scipy import stats
import math

import jax
from jax import vmap
import jax.random as random
import corner
import numpyro

from utils_analysis.gp_utils import model, predict, run_inference
from utils_analysis.dtw_utils import dtw
from utils_analysis.Vobs_fits import Vbar_sq
from utils_analysis.mock_gen import Vobs_MCMC
# from utils_analysis.extract_ft import ft_check

matplotlib.use("Agg")
plt.rcParams.update({'font.size': 13})


plot_digitizer = True
use_window = False   # Use only a window around feature, [15:24] or [35:58], for analysis.

make_plots = True
do_DTW = True
do_correlations = True

# Directory for saving plots.
if use_window: floc = "/mnt/users/koe/plots/NGC1560_fullGP/window/"
else: floc = "/mnt/users/koe/plots/NGC1560_fullGP/"

if plot_digitizer: fileloc = floc + "plot_digitizer/"
else: fileloc = floc
print(f"Saving plots to {fileloc}")
fname_DTW = fileloc + "dtw/"

num_samples = 1000


# Main code to run.
def main(args, r_full, rad, v_data, v_mock, num_samples=num_samples, ls:float=4.5):
    """
    Do inference for Vbar with uniform prior for correlation length,
    then apply the resulted lengthscale to Vobs (both real and mock data).
    """
    data_labels = [ r"$V_{\text{bar}}$", r"$V_{\text{obs}}$" ]
    mock_labels = [ r"$V_{\text{MOND}}$", r"$V_{\Lambda CDM}$", "Vbar_MOND", "Vbar_LCDM" ]
    v_comps = data_labels + mock_labels
    colours = [ 'tab:red', 'k', 'mediumblue', 'tab:green' ]

    # "Raw" percentiles from uncertainties and scattering.
    raw_median = np.percentile(v_mock, 50.0, axis=1)                # dim = (4, r)
    raw_percentiles = np.percentile(v_mock, [16.0, 84.0], axis=1)   # dim = (2, 4, r)
    Vbar_errors = ( np.diff(raw_percentiles[:,0], axis=0) / 2 )[0]


    """ ------------
    GPR on data.
    ------------ """
    meanGP_data = []
    for j in range(2):
        print(f"Fitting function to {data_labels[j]} with ls = {ls} kpc...")
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = run_inference(model, args, rng_key, r_full, v_data[j], ls=ls)

        vmap_args = (
            random.split(rng_key_predict, samples["var"].shape[0]),
            samples["var"],
            samples["noise"],
        )

        if plot_digitizer:
            r_Xft = np.delete(r_full, np.s_[19:24], axis=0)
            data_Xft = np.delete(v_data[j], np.s_[19:24], axis=0)
            means, _ = vmap(
                lambda rng_key, var, noise: predict(
                    rng_key, r_Xft, data_Xft, rad, var, ls, noise, use_cholesky=args.use_cholesky
                )
            )(*vmap_args)
        else:
            r_Xft = np.delete(r_full, np.s_[37:58], axis=0)
            data_Xft = np.delete(v_data[j], np.s_[37:58], axis=0)
            means, _ = vmap(
                lambda rng_key, var, noise: predict(
                    rng_key, r_Xft, data_Xft, rad, var, ls, noise, use_cholesky=args.use_cholesky
                )
            )(*vmap_args)

        mean_pred = np.mean( means, axis=0 )  # [ Vbar, Vobs ]
        meanGP_data.append( mean_pred )
        
    # Compute residuals of data fits.
    res_Vbar_data, res_Vobs = [], []
    for k in range(len(r_full)):
        idx = (np.abs(rad - r_full[k])).argmin()
        res_Vbar_data.append(v_data[0][k] - meanGP_data[0][idx])
        res_Vobs.append(v_data[1][k] - meanGP_data[1][idx])
    res_data = np.array([ res_Vbar_data, res_Vobs ])    # dim = (2, len(r))


    """ --------------------
    GPR on mock samples.
    -------------------- """
    res_mock = []
    # GPR and analysis on individual mock samples.
    # for smp in tqdm(range(num_samples), desc="GPR on mock samples"):
    for smp in range(num_samples):
        if smp % 100 == 0: print(f"GPR on mock sample {smp} of {num_samples}...")
        meanGP_mock = []
    
        for j in range(4):
            rng_key, rng_key_predict = random.split(random.PRNGKey(0))
            samples = run_inference(model, args, rng_key, r_full, v_mock[j,smp], ls=ls)

            # do prediction
            vmap_args = (
                random.split(rng_key_predict, samples["var"].shape[0]),
                samples["var"],
                samples["noise"],
            )

            if plot_digitizer:
                r_Xft = np.delete(r_full, np.s_[19:24], axis=0)
                mock_Xft = np.delete(v_mock[j,smp], np.s_[19:24], axis=0)
                means, _ = vmap(
                    lambda rng_key, var, noise: predict(
                        rng_key, r_Xft, mock_Xft, rad, var, ls, noise, use_cholesky=args.use_cholesky
                    )
                )(*vmap_args)
            else:
                r_Xft = np.delete(r_full, np.s_[37:58], axis=0)
                mock_Xft = np.delete(v_mock[j,smp], np.s_[37:58], axis=0)
                means, _ = vmap(
                    lambda rng_key, var, noise: predict(
                        rng_key, r_Xft, mock_Xft, rad, var, ls, noise, use_cholesky=args.use_cholesky
                    )
                )(*vmap_args)

            mean_pred = np.mean(means, axis=0)
            meanGP_mock.append(mean_pred)   # [ Vbar_MOND, Vbar_LCDM, MOND, LCDM ]

            if smp == 0: plotGP_mock = np.array(mean_pred) / num_samples
            else: plotGP_mock += np.array(mean_pred) / num_samples

            jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem.

        # Compute residuals of fits.
        res_Vbar_MOND, res_Vbar_LCDM, res_MOND, res_LCDM = [], [] ,[], []
        for k in range(len(r_full)):
            idx = (np.abs(rad - r_full[k])).argmin()
            res_Vbar_MOND.append(v_mock[0,smp,k] - meanGP_mock[0][idx])
            res_Vbar_LCDM.append(v_mock[1,smp,k] - meanGP_mock[1][idx])
            res_MOND.append(v_mock[2,smp,k] - meanGP_mock[2][idx])
            res_LCDM.append(v_mock[3,smp,k] - meanGP_mock[3][idx])
        res_mock.append( np.array([ res_Vbar_MOND, res_Vbar_LCDM, res_MOND, res_LCDM ]) )   # dim = num_samples x (4, len(r))

    res_mock = np.transpose( res_mock, (1, 2, 0) )  # dim = (4, len(r), num_samples)
    res_perc = np.percentile( res_mock, [16.0, 50.0, 84.0], axis=2 )    # dim = (3, 4, len(r))
    # mock_err = np.diff( res_perc, axis=0 )     # dim = (2, 4, len(r))

    if plot_digitizer: np.save("/mnt/users/koe/mock_residuals/NGC1560_Sanders.npy", res_mock)
    else: np.save("/mnt/users/koe/mock_residuals/NGC1560_Gentile.npy", res_mock)
    print("\nMock residuals saved.")
    print("Memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)

    if use_window:
        if plot_digitizer:
            res_data = res_data[19:24]
            res_mock = res_mock[:,19:24]
            err_Vobs = v_data[2][19:24]
            Vbar_errors = Vbar_errors[19:24]
            r = r_full[19:24]
        else:
            res_data = res_data[37:58]
            res_mock = res_mock[:,37:58]
            err_Vobs = v_data[2][37:58]
            Vbar_errors = Vbar_errors[37:58]
            r = r_full[37:58]
    else:
        r = r_full
        err_Vobs = v_data[2]


    """ -------------------------
    Analysis of GP residuals.
    ------------------------- """
    if do_DTW:
        for smp in tqdm(range(num_samples), desc="DTW"):
            # Construct distance matrices.
            dist_data = np.zeros((len(r), len(r)))
            dist_MOND = np.copy(dist_data)
            dist_LCDM = np.copy(dist_data)
            
            for n in range(len(r)):
                for m in range(len(r)):
                    dist_data[n, m] = np.abs( res_data[0,n] - res_data[1,m] )
                    dist_MOND[n, m] = np.abs( res_mock[0,n,smp] - res_mock[2,m,smp] )
                    dist_LCDM[n, m] = np.abs( res_mock[1,n,smp] - res_mock[3,m,smp] )
            
            dist_mats = np.array([ dist_data, dist_MOND, dist_LCDM ])
            mats_dir = [ "data", "MOND", "LCDM" ]
            
            # DTW!
            for j in range(3):
                if j == 0 and smp >= 1:
                    dtw_cost[j].append(dtw_cost[j][0])
                else:
                    path, cost_mat = dtw(dist_mats[j])
                    x_path, y_path = zip(*path)
                    cost = cost_mat[ len(r)-1, len(r)-1 ]
                    dtw_cost[j].append(cost / (2 * len(r)))

                if make_plots and smp == 0:
                    # Plot distance matrix and cost matrix with optimal path.
                    if plot_digitizer:
                        plt.title("Dynamic time warping: NGC 1560 (Sanders 2007)")
                    else:
                        plt.title("Dynamic time warping: NGC 1560 (Gentile et al. 2010)")
                    plt.axis('off')

                    plt.subplot(121)
                    plt.title("Distance matrix")
                    plt.imshow(dist_mats[j], cmap=plt.cm.binary, interpolation="nearest", origin="lower")

                    plt.subplot(122)
                    plt.title("Cost matrix")
                    plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
                    plt.plot(x_path, y_path)

                    plt.savefig(fname_DTW+"matrix_"+mats_dir[j]+".png", dpi=300, bbox_inches="tight")
                    plt.close('all')

                    # Visualize DTW alignment.
                    plt.title("DTW alignment: NGC 1560")

                    if j == 0:
                        diff = abs(max(np.array(res_data[0])) - min(res_data[1]))
                        for x_i, y_j in path:
                            plt.plot([x_i, y_j], [res_data[1,x_i] + diff, res_data[0][y_j] - diff], c="C7", alpha=0.4)
                        plt.plot(np.arange(len(r)), res_data[1] + diff, c='k', label=v_comps[1])
                        plt.plot(np.arange(len(r)), res_data[0] - diff, c="tab:red", label=r'$V_{\text{bar}}$')

                    else: 
                        diff = abs(max(np.array(res_mock)[j-1,:,smp]) - min(np.array(res_mock)[j+1,:,smp]))
                        for x_i, y_j in path:
                            plt.plot([x_i, y_j], [res_mock[j+1][x_i][smp] + diff, res_mock[j-1][y_j][smp] - diff], c="C7", alpha=0.4)
                        plt.plot(np.arange(len(r)), np.array(res_mock)[j+1,:,smp] + diff, c=colours[j+1], label=v_comps[j+1])
                        plt.plot(np.arange(len(r)), np.array(res_mock)[j-1,:,smp] - diff, c='tab:red', label=r'$V_{\text{bar}}$')

                    plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(cost))
                    plt.plot([], [], c='w', label="Normalized cost = {:.4f}".format(cost/(len(r)*2)))

                    plt.axis("off")
                    plt.legend(bbox_to_anchor=(1,1))
                    plt.savefig(fname_DTW+"vis_"+mats_dir[j]+".png", dpi=300, bbox_inches="tight")
                    plt.close('all')

    if do_correlations:

        pearsonr_data = []
        for j in range(3, len(r)+1):
            pearsonr_data.append(stats.pearsonr(res_data[0,:j], res_data[1,:j])[0])
        pearson_data.append(pearsonr_data[-1])

        # Compute correlation coefficients for mock Vobs vs Vbar.
        radii_corr = []

        # for smp in range(num_samples):
        for smp in tqdm(range(num_samples), desc="Correlation by radii"):
            """
            ----------------------------------------------------
            Correlation plots using spheres of increasing radius
            ----------------------------------------------------
            """
            correlations_r = []
            for i in range(1, 3):
                pearsonr_mock = []
                for j in range(3, len(r)+1):
                    pearsonr_mock.append(stats.pearsonr(res_mock[i-1,:j,smp], res_mock[i+1,:j,smp])[0])
                correlations_r.append(pearsonr_mock)
            radii_corr.append(correlations_r)
        
        rcorr_percentiles = np.percentile(radii_corr, [16.0, 50.0, 84.0], axis=0)
        pearson_mock.append([ rcorr_percentiles[:,0,-1], rcorr_percentiles[:,1,-1] ])

        """
        Plot GP fits, residuals and correlations.
        """
        if make_plots:
            """Pearson correlations."""
            fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
            fig1.set_size_inches(7, 7)
            if plot_digitizer: ax0.set_ylabel("Velocities (km/s)")
            
            for j in range(4):
                if j == 1: ax0.errorbar(r_full, v_data[1], v_data[2], color='k', alpha=0.3, fmt='.', capsize=2.5, zorder=10)  # Vobs
                elif j == 0: ax0.errorbar(r_full, v_data[0], Vbar_errors, color='tab:red', alpha=0.3, fmt='.', capsize=2.5)   # Vbar
                # else: ax0.errorbar(r_full, raw_median[j], mock_err[:,j], color=colours[j], alpha=0.3, fmt='.', capsize=2.5)  # MOND, LCDM
                else: ax0.scatter(r_full, raw_median[j], c=colours[j], alpha=0.3, marker='.')   # MOND, LCDM
                # Plot mean prediction from GP.
                if j < 2: ax0.plot(rad, meanGP_data[j], color=colours[j], label=v_comps[j], zorder=10-j)
                else: ax0.plot(rad, plotGP_mock[j], color=colours[j], label=v_comps[j], zorder=10-j)

            if plot_digitizer: ax0.legend()
            if not use_window: ax0.set_ylim((0.0, 83.0))

            if plot_digitizer:
                ax0.axvline(r_full[19], c='k', ls='--', alpha=0.5, zorder=0, label="Feature")
                ax0.axvline(r_full[23], c='k', ls='--', alpha=0.5, zorder=0)
            else:
                ax0.axvline(r_full[37], c='k', ls='--', alpha=0.5, zorder=0, label="Feature")
                ax0.axvline(r_full[57], c='k', ls='--', alpha=0.5, zorder=0)

            if plot_digitizer: ax1.set_ylabel("Residuals")

            ax1.errorbar(r, res_data[1], err_Vobs, color='k', alpha=0.3, fmt='.', capsize=2.5, zorder=10)   # Vobs
            ax1.errorbar(r, res_data[0], Vbar_errors, color='tab:red', fmt='.', alpha=0.4, capsize=2.5, zorder=11)  # Vbar
            for j in range(2, 4):
                # ax1.errorbar(r, res_perc[1,j], mock_err[:,j], color=colours[j], fmt='.', alpha=0.3, capsize=2.5, zorder=10-j)   # MOND, LCDM
                ax1.scatter(r, res_perc[1,j], c=colours[j], marker='.', alpha=0.3, zorder=10-j)     # MOND, LCDM

            if not use_window: ax1.set_ylim((-8.5, 8.5))

            if plot_digitizer:
                ax0.axvline(r_full[19], c='k', ls='--', alpha=0.5, zorder=0, label="Feature")
                ax0.axvline(r_full[23], c='k', ls='--', alpha=0.5, zorder=0)
            else:
                ax0.axvline(r_full[37], c='k', ls='--', alpha=0.5, zorder=0, label="Feature")
                ax0.axvline(r_full[57], c='k', ls='--', alpha=0.5, zorder=0)

            ax2.set_xlabel("Radius (kpc)")
            if plot_digitizer: ax2.set_ylabel(r"$\rho_p$ w.r.t. $V_{\text{bar}}$")

            for j in range(2):
                if use_window:
                    ax2.plot(r[2:], rcorr_percentiles[1][j], c=colours[j+2], label=v_comps[j+2]+r": Pearson $\rho$")
                    ax2.fill_between(r[2:], rcorr_percentiles[0][j], rcorr_percentiles[2][j], color=colours[j+2], alpha=0.2)
                else:
                    ax2.plot(r[4:], rcorr_percentiles[1][j][2:], c=colours[j+2], label=v_comps[j+2]+r": Pearson $\rho$")
                    ax2.fill_between(r[4:], rcorr_percentiles[0][j][2:], rcorr_percentiles[2][j][2:], color=colours[j+2], alpha=0.2)

            if use_window: ax2.plot(r[2:], pearsonr_data, c='k', label=r"Data: Pearson $\rho$")
            else: ax2.plot(r[4:], pearsonr_data[2:], c='k', label=r"Data: Pearson $\rho$")

            if not use_window: ax2.set_ylim((0.0, 0.9))

            if plot_digitizer:
                ax0.axvline(r_full[19], c='k', ls='--', alpha=0.5, zorder=0, label="Feature")
                ax0.axvline(r_full[23], c='k', ls='--', alpha=0.5, zorder=0)
            else:
                ax0.axvline(r_full[37], c='k', ls='--', alpha=0.5, zorder=0, label="Feature")
                ax0.axvline(r_full[57], c='k', ls='--', alpha=0.5, zorder=0)

            if not plot_digitizer:
                ax0.set_yticklabels([])
                ax1.set_yticklabels([])
                ax2.set_yticklabels([])

            plt.subplots_adjust(hspace=0.05)
            fig1.savefig(fileloc+"pearson.pdf", dpi=300, bbox_inches="tight")
            plt.close()
    
    print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
    jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem.


def GP_args():
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

    return args


if __name__ == "__main__":
    # Initialize GP arguments.
    args = GP_args()
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    galaxy, pearson_mock, pearson_data = [], [], []
    dtw_cost = [ [], [], [] ]

    """
    Plotting galaxy rotation curves directly from data with variables:
    Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
    """
    if plot_digitizer:
        file_path = "/mnt/users/koe/data/NGC1560_Stacy.dat"
        columns = [ "Rad", "Vobs", "Vgas", "Vdisk_raw", "Vdisk", "errV" ]
    else:
        file_path = "/mnt/users/koe/data/NGC1560.dat"
        columns = [ "Rad", "Vobs", "errV", "Sdst", "Vdisk", "Sdgas", "Vgas", "Vgth" ]

    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)
    r = data["Rad"]
    bulged = False

    table = { "D":[2.99], "e_D":[0.1], "Inc":[82.0], "e_Inc":[1.0] }
    i_table = 0

    rad = np.linspace(min(r), max(r), 100)

    # Normalise velocities by Vmax = max(Vobs) from SPARC data.
    Vbar2 = Vbar_sq(data, bulged)
    Vbar = np.sqrt(Vbar2)

    print("Fitting for LCDM...")
    nfw_samples  = Vobs_MCMC(table, i_table, data, bulged, profile="NFW")    # Vobs_MCMC() runs MCMC with Vobs_fit() from Vobs_fits.py
    print("Fitting for MOND...")
    mond_samples = Vobs_MCMC(table, i_table, data, bulged, profile="MOND")

    # Select num_samples random samples from MCMC fits; dim = (num_samples, len(r)).
    rand_idx = np.random.choice( 20000, num_samples, replace=False )
    full_LCDM = nfw_samples["Vpred scattered"][rand_idx]
    full_Vbar_LCDM = nfw_samples["Vbar"][rand_idx]
    full_MOND = mond_samples["Vpred scattered"][rand_idx]
    full_Vbar_MOND = mond_samples["Vbar"][rand_idx]

    if make_plots:
        labels = [ "Distance", "Gas M/L", "Disk M/L", "inc", "logM200c", "logc" ]
        samples_arr = np.vstack([nfw_samples[label] for label in labels]).T
        fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
        fig.savefig(fileloc+"corner_NFW.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        labels = [ "Distance", "Gas M/L", "Disk M/L", "inc" ]
        samples_arr = np.vstack([mond_samples[label] for label in labels]).T
        fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
        fig.savefig(fileloc+"corner_MOND.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    v_data = np.array([ Vbar, data["Vobs"], data["errV"] ])
    v_mock = np.array([ full_Vbar_MOND, full_Vbar_LCDM, full_MOND, full_LCDM ])

    main(args, r.to_numpy(), rad, v_data, v_mock)


# Plot summary histograms.
if make_plots:
    galaxies = [ "NGC1560" ]
    galaxy_count = 1

    """
    Plot histogram of normalized DTW costs (in ascending order of costs for data).
    """
    if do_DTW:
        # dim = (3 x v_comps, galaxy_count, num_samples)
        dtw_cost = np.array(dtw_cost)

        # Arrays of shape (5 x percentiles, 3 x v_comps, galaxy_count).
        dtw_percentiles = np.percentile(dtw_cost, [5.0, 16.0, 50.0, 84.0, 95.0], axis=1)

        galaxies = [ "NGC1560" ]
        galaxy_count = 1
        
        fig, ax = plt.subplots()

        hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
        colours = [ 'k', 'mediumblue', 'tab:green' ]

        for j in range(3):
            mean_norm = np.nanmean(dtw_percentiles[2][j])
            low_err = dtw_percentiles[2][j] - dtw_percentiles[1][j]
            up_err = dtw_percentiles[3][j] - dtw_percentiles[2][j]

            if j == 0:
                print("\nDTW cost ("+hist_labels[j]+") = {:.4f}".format(mean_norm))
            else:
                print("\nMedian cost ("+hist_labels[j]+") = {:.4f}".format(dtw_percentiles[2][j]))
                print("Upper error = {:.4f}".format(up_err))
                print("Lower error = {:.4f}".format(low_err))

            if j == 0:
                ax.axhline(y=mean_norm, color='k', linestyle='dashed', label="Data: {:.4f}".format(mean_norm))
            else:
                if j == 1: trans = Affine2D().translate(-0.1, 0.0) + ax.transData
                else: trans = Affine2D().translate(+0.1, 0.0) + ax.transData
                ax.errorbar(galaxies, dtw_percentiles[2][j], [[low_err], [up_err]], fmt='.', ls='none',
                            capsize=2.5, color=colours[j], alpha=0.8, transform=trans, label=f"{hist_labels[j]}: Mean = {mean_norm:.4f}")
        
        if not use_window: ax.set_ylim(bottom=0.0)
        ax.legend()
        ax.set_xticks([])
        ax.set_ylabel("Normalized DTW cost")
        fig.savefig(fname_DTW+"histo1.pdf", dpi=300, bbox_inches="tight")
        plt.close()

print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)

