#!/usr/bin/env python
"""
Correlation coefficients + dynamic time warping on GP residuals of NGC1560_Stacy,
a dataset obtained from digitizing Stacy's NGC 1560 plot with Plot Digitizer;
taking into account uncertainties (Vbar) and Vobs scattering (errV).

To run this in queue:
addqueue -q cmb -c "5 mins" -n 1 -m 16 analyze_NGC1560.py

The following paper (Broeils 1992) analyzes NGC 1560 in some detail, thus might be useful:
https://articles.adsabs.harvard.edu/pdf/1992A%26A...256...19B
"""

import jax.experimental
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
from utils_analysis.mock_gen import Vobs_MCMC   # Vobs_scat, Vbar_sq_unc
from utils_analysis.extract_ft import ft_check

matplotlib.use("Agg")
plt.rcParams.update({'font.size': 13})


plot_digitizer = False
use_fits = True
use_window = False   # Use only a window around feature, [15:24] or [35:57], for analysis.

make_plots = True
do_DTW = True
do_correlations = True

DTW_power = 1.0     # Power for initializing DTW distance matrix (default = 1.0).

if use_window:
    floc = "/mnt/users/koe/plots/NGC1560/window/"
    # if not plot_digitizer: raise ValueError("Windowed analysis only suitable for Sanders's RCs (w/ features) atm.")
else: floc = "/mnt/users/koe/plots/NGC1560/"   # Directory for saving plots.

if plot_digitizer:
    floc += "plot_digitizer/"
if use_fits:
    fileloc = floc + "use_fits/"
else:
    fileloc = floc

# Options: cost wrt MOND: "dtw/"; cost wrt LCDM: "dtw/cost_vsLCDM/", original cose (MSE): "dtw/cost_vsVbar/".
if do_DTW:
    fname_DTW = fileloc + "dtw/"
    print(f"fname_DTW = {fname_DTW}")

num_samples = 1000


# Main code to run.
def main(args, r_full, rad, Y, v_data, v_mock, num_samples=num_samples, ls:float=4.5):
    """
    Do inference for Vbar with uniform prior for correlation length,
    then apply the resulted lengthscale to Vobs (both real and mock data).
    """
    v_comps = [ r"$V_{\text{bar}}$", r"$V_{\text{obs}}$", r"$V_{\text{MOND}}$", r"$V_{\Lambda CDM}$", "Vbar_MOND", "Vbar_LCDM" ]
    colours = [ 'tab:red', 'k', 'mediumblue', 'tab:green' ]
    mean_prediction = []
    percentiles = []

    # if not plot_digitizer: ls = 3.5
    
    for j in range(6):
        print(f"Fitting function to {v_comps[j]} with ls = {ls} kpc...")
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = run_inference(model, args, rng_key, r_full, Y[j], ls=ls)

        # do prediction
        vmap_args = (
            random.split(rng_key_predict, samples["var"].shape[0]),
            samples["var"],
            samples["noise"],
        )

        if plot_digitizer:
            r_Xft = np.delete(r_full, np.s_[19:23], axis=0)
            Vcomp_Xft = np.delete(Y[j], np.s_[19:23], axis=0)
            means, predictions = vmap(
                lambda rng_key, var, noise: predict(
                    rng_key, r_Xft, Vcomp_Xft, rad, var, ls, noise, use_cholesky=args.use_cholesky
                )
            )(*vmap_args)
        else:
            r_Xft = np.delete(r_full, np.s_[37:57], axis=0)
            r_Xft = np.delete(r_Xft, np.s_[3:8], axis=0)
            Vcomp_Xft = np.delete(Y[j], np.s_[37:57], axis=0)
            Vcomp_Xft = np.delete(Vcomp_Xft, np.s_[3:8], axis=0)
            means, predictions = vmap(
                lambda rng_key, var, noise: predict(
                    rng_key, r_Xft, Vcomp_Xft, rad, var, ls, noise, use_cholesky=args.use_cholesky
                )
            )(*vmap_args)
            # means, predictions = vmap(
            #     lambda rng_key, var, noise: predict(
            #         rng_key, r_full, Y[j], rad, var, ls, noise, use_cholesky=args.use_cholesky
            #     )
            # )(*vmap_args)

        mean_pred = np.mean(means, axis=0)
        mean_prediction.append(mean_pred)   # [ Vbar, Vobs, MOND, LCDM ]

        confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
        percentiles.append(confidence_band)

    # "Raw" percentiles from uncertainties and scattering.
    raw_median = np.percentile(v_mock, 50.0, axis=2)                # dim = (4, r)
    raw_percentiles = np.percentile(v_mock, [16.0, 84.0], axis=2)   # dim = (2, 4, r)
    raw_errors = np.abs( raw_percentiles - raw_median )             # dim = (2, 4, r)

    # Compute residuals of fits.
    res_Vbar_data, res_Vobs, res_Vbar_MOND, res_Vbar_LCDM, res_MOND, res_LCDM = [], [] ,[], [], [], []
    for k in range(len(r_full)):
        idx = (np.abs(rad - r_full[k])).argmin()
        
        res_Vbar_data.append(v_data[0][k] - mean_prediction[0][idx])
        res_Vobs.append(v_data[1][k] - mean_prediction[1][idx])

        res_Vbar_MOND.append(v_mock[0][k] - mean_prediction[4][idx])
        res_Vbar_LCDM.append(v_mock[1][k] - mean_prediction[5][idx])
        res_MOND.append(v_mock[2][k] - mean_prediction[2][idx])
        res_LCDM.append(v_mock[3][k] - mean_prediction[3][idx])

    # print(f"Normalized res_Vobs: {res_Vobs[19:23] / Y[6][19:23]}")
    # print(f"Normalized res_Vbar: {res_Vbar_data[19:23] / Y[6][19:23]}")
    # raise ValueError("Stop here.")

    res_data = np.array([ res_Vbar_data, res_Vobs ])            # dim = (2, len(r))
    res_mock = np.array([ res_Vbar_MOND, res_Vbar_LCDM, res_MOND, res_LCDM ])  # dim = (3, len(r), num_samples)

    if use_window:
        if plot_digitizer:
            res_data = res_data[:,15:25]
            res_mock = res_mock[:,15:25]
            r = r_full[15:25]
        else:
            res_data = res_data[:,37:57]
            res_mock = res_mock[:,37:57]
            r = r_full[37:57]
    else:
        r = r_full

    # Residual percentiles from uncertainties and scattering.
    res_median = np.median( res_mock, axis=2 )
    res_percentiles = np.percentile( res_mock, [16.0, 84.0], axis=2 )   # dim = (2, 4, r)
    res_errors = np.abs( res_percentiles - res_median )

    # Extract properties of feature(s) (if any).
    if not use_window:
        lb_ft, rb_ft, ft_widths = ft_check( np.array(res_data[0]), raw_errors[1,0] )
        print("Feature(s) in Vbar:")
        print(lb_ft)
        print(rb_ft)
        print(ft_widths)

        lb_ft, rb_ft, ft_widths = ft_check( np.array(res_data[1]), Y[6] )
        print("Feature(s) in Vobs:")
        print(lb_ft)
        print(rb_ft)
        print(ft_widths)

    """
    DTW on GP residuals.
    """
    if do_DTW:
        # for smp in range(num_samples):
        for smp in tqdm(range(num_samples), desc="DTW"):
            # Construct distance matrices.
            dist_data = np.zeros((len(r), len(r)))
            dist_MOND = np.copy(dist_data)
            dist_LCDM = np.copy(dist_data)
            
            for n in range(len(r)):
                for m in range(len(r)):
                    # Construct distance matrix such that cost = 0 if Vobs = MOND(Vbar).
                    # if fname_DTW == fileloc+"dtw/":
                    #     dist_data[n, m] = np.abs(res_data[1,n] - res_mock[1,m,smp])
                    #     dist_MOND[n, m] = np.abs(res_mock[1,n,smp] - res_mock[1,m,smp])
                    #     dist_LCDM[n, m] = np.abs(res_mock[2,n,smp] - res_mock[1,m,smp])

                    # # Alternative constructions:
                    # elif fname_DTW == fileloc+"dtw/cost_vsLCDM/":
                    #     dist_data[n, m] = np.abs(res_data[1,n] - res_mock[2,m,smp])
                    #     dist_MOND[n, m] = np.abs(res_mock[1,n,smp] - res_mock[2,m,smp])
                    #     dist_LCDM[n, m] = np.abs(res_mock[2,n,smp] - res_mock[2,m,smp])
                    # else:
                    dist_data[n, m] = np.abs(res_data[1,n] - res_data[0,m]) ** DTW_power
                    dist_MOND[n, m] = np.abs(res_mock[2,n,smp] - res_mock[0,m,smp]) ** DTW_power
                    dist_LCDM[n, m] = np.abs(res_mock[3,n,smp] - res_mock[1,m,smp]) ** DTW_power
                    # else:
                    #     dist_data[n, m] = np.abs(res_data[1,n] - res_mock[0,m,smp])
                    #     dist_MOND[n, m] = np.abs(res_mock[1,n,smp] - res_mock[0,m,smp])
                    #     dist_LCDM[n, m] = np.abs(res_mock[2,n,smp] - res_mock[0,m,smp])
            
            dist_mats = np.array([ dist_data, dist_MOND, dist_LCDM ])
            mats_dir = [ "data", "MOND", "LCDM" ]
            
            # DTW!
            for j in range(3):
                path, cost_mat = dtw(dist_mats[j])
                x_path, y_path = zip(*path)
                cost = cost_mat[ len(r)-1, len(r)-1 ]
                dtw_cost[j].append(cost)
                norm_cost[j].append(cost / (2 * len(r)))

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

                    # Settings for visualizing different DTW constructions.
                    # if fname_DTW == fileloc+"dtw/":
                    #     ref_curve = [ res_mock[1], "mediumblue", "MOND" ]
                    # elif fname_DTW == fileloc+"dtw/cost_vsLCDM/":
                    #     ref_curve = [ res_mock[2], "tab:green", r"$\Lambda$CDM" ]
                    # else:
                    #     ref_curve = [ res_data[0], "tab:red", "Vbar" ]

                    if j == 0:
                        # diff = abs(max(np.array(ref_curve[0])[:,smp]) - min(res_data[1]))
                        diff = abs(max(np.array(res_data[0])) - min(res_data[1]))
                        for x_i, y_j in path:
                            # plt.plot([x_i, y_j], [res_data[1,x_i] + diff, ref_curve[0][y_j][smp] - diff], c="C7", alpha=0.4)
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


    """
    Code for PCHIP + correlations on GP residuals.
    """
    if do_correlations:

        pearsonr_data = []
        for j in range(3, len(r)+1):
            pearsonr_data.append(stats.pearsonr(res_data[0][:j], res_data[1][:j])[0])
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
            # Correlate Vobs and Vbar (d0, d1, d2) as a function of (maximum) radius, i.e. spheres of increasing r.
            # correlations_r = rad_corr arrays with [ MOND, LCDM ], so 2 Vobs x 3 derivatives x 2 correlations each,
            # where rad_corr = [ [[Spearman d0], [Pearson d0]], [[S d1], [P d1]], [[S d2], [P d2]] ].
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
            c_temp = [ 'tab:red', 'mediumblue', 'tab:green' ]

            """Pearson correlations."""
            fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
            fig1.set_size_inches(7, 7)
            # if plot_digitizer:
            #     ax0.set_title("Pearson correlation: NGC 1560 (Sanders 2007)")
            # else:
            #     ax0.set_title("Pearson correlation: NGC 1560 (Gentile et al. 2010)")
            if not plot_digitizer: ax0.set_ylabel("Velocities (km/s)")
            
            for j in range(4):
                if j == 1: ax0.errorbar(r_full, v_data[1], Y[6], color='k', alpha=0.3, fmt='o', capsize=2)  # Vobs
                elif j == 0: ax0.scatter(r_full, v_data[0], color='tab:red', alpha=0.3)                     # Vbar
                else: ax0.errorbar(r_full, raw_median[j], raw_errors[:, j], c=c_temp[j-1], alpha=0.3, fmt='o', capsize=2)
                # Plot mean prediction from GP.
                ax0.plot(rad, mean_prediction[j], color=colours[j], label=v_comps[j])
                # Fill in 1-sigma (68%) confidence band of GP fit.
                # ax0.fill_between(rad, percentiles[j][0], percentiles[j][1], color=colours[j], alpha=0.2)

            # if plot_digitizer: ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
            if not plot_digitizer: ax0.legend()
            ax0.grid()

            if not plot_digitizer: ax1.set_ylabel("Residuals")
            for j in range(1, 4):
                # Plots for mock Vobs (sampled w/ uncertainties).
                if j == 3:  # Plot Vobs.
                    # if der == 0:
                    # ax1.errorbar(r[5:], res_data[1][5:], Y[6][5:], color='k', alpha=0.3, ls='none', fmt='o', capsize=2)
                    # ax1.plot(r[5:], res_data[1][5:], color='k', label=v_comps[j])
                    if use_window: 
                        if plot_digitizer:
                            ax1.errorbar(r, res_data[1], Y[6][15:25], color='k', alpha=0.3, fmt='o', capsize=2, label=v_comps[j])
                        else:
                            ax1.errorbar(r, res_data[1], Y[6][37:57], color='k', alpha=0.3, fmt='o', capsize=2, label=v_comps[j])
                    else: ax1.errorbar(r, res_data[1], Y[6], color='k', alpha=0.3, fmt='o', capsize=2, label=v_comps[j])
                    # ax1.plot(r, res_data[1], color='k', label=v_comps[j])
                else:
                    # if der == 0:
                    # ax1.scatter(r[5:], res_median[j+1][5:], c=c_temp[j], alpha=0.3)
                    ax1.errorbar(r, res_median[j+1], res_errors[:,j+1], c=c_temp[j], alpha=0.3, fmt='o', capsize=2, label=v_comps[j])
                    # ax1.plot(r[5:], res_median[j+1][5:], c=c_temp[j], label=v_comps[j])
                    # ax1.fill_between(r[5:], res_percentiles[0][j+1][5:], res_percentiles[1][j+1][5:], color=c_temp[j], alpha=0.15)
                    # ax1.plot(r, res_median[j+1], c=c_temp[j], label=v_comps[j])
                    # ax1.fill_between(r, res_percentiles[0][j+1], res_percentiles[1][j+1], color=c_temp[j], alpha=0.15)
            ax1.scatter(r, res_data[0], color='tab:red', alpha=0.4, label=v_comps[0])

            ax1.grid()

            ax2.set_xlabel("Radius (kpc)")
            if not plot_digitizer: ax2.set_ylabel(r"Correlations w.r.t. $V_{bar}$")

            for j in range(2):
                if use_window:
                    ax2.plot(r[2:], rcorr_percentiles[1][j], c=c_temp[j+1], label=v_comps[j+2]+r": Pearson $\rho$")
                    ax2.fill_between(r[2:], rcorr_percentiles[0][j], rcorr_percentiles[2][j], color=colours[j+2], alpha=0.2)
                else:
                    ax2.plot(r[4:], rcorr_percentiles[1][j][2:], c=c_temp[j+1], label=v_comps[j+2]+r": Pearson $\rho$")
                    ax2.fill_between(r[4:], rcorr_percentiles[0][j][2:], rcorr_percentiles[2][j][2:], color=colours[j+2], alpha=0.2)

            if use_window: ax2.plot(r[2:], pearsonr_data, c='k', label=r"Data: Pearson $\rho$")
            else: ax2.plot(r[4:], pearsonr_data[2:], c='k', label=r"Data: Pearson $\rho$")
            # ax2.plot([], [], ' ', label=r"$\rho_p=$"+str(round(np.nanmean(pearsonr_data), 3)))
            ax2.grid()

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

    
    galaxy, spearman_mock, pearson_mock, spearman_data, pearson_data = [], [], [], [], []
    dtw_cost = [ [], [], [] ]
    norm_cost = [ [], [], [] ]

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
    # r /= 1.3    # Scale length of NGC 1560 according to Broeils.
    bulged = False

    table = { "D":[2.99], "e_D":[0.1], "Inc":[82.0], "e_Inc":[1.0] }    # , "L":[23.8], "e_L":[1.8] }
    i_table = 0

    rad_count = math.ceil((max(r)-min(r))*100)
    rad = np.linspace(min(r), max(r), rad_count)
    # data["Vdisk"] /= np.sqrt(pdisk)   # Correction now implemented in data

    # Normalise velocities by Vmax = max(Vobs) from SPARC data.
    # Vbar2_unc = Vbar_sq_unc(table, i_table, data, bulged, num_samples)
    Vbar2 = Vbar_sq(data, bulged)
    Vbar = np.sqrt(Vbar2)

    # if make_plots and not use_window:
    #     # Plot Vobs and Vbar.
    #     # if plot_digitizer:
    #     #     plt.title("NGC 1560 (Sanders 2007) from manual digitization")
    #     # else:
    #     #     plt.title("NGC 1560 (Gentile et al. 2010) from S. McGaugh")
            
    #     plt.xlabel("Radius (kpc)")
    #     # plt.xlabel(" ")
    #     if plot_digitizer: plt.ylabel("Velocities (km/s)")
    #     else: plt.yticks(color='w')

    #     plt.errorbar(r, data["Vobs"], data["errV"], fmt=".", ls='none', capsize=2, c='k', label=r"$V_{\text{obs}}$")
    #     plt.plot(r, np.sqrt(Vbar2), c='k', linestyle='dashdot', label=r"$V_{\text{bar}}$")
    #     plt.plot(r, data["Vgas"], c='k', linestyle='dotted', label=r"$V_{\text{gas}}$")
    #     plt.plot(r, np.sqrt(0.5)*data["Vdisk"], c='k', linestyle='dashed', label=r"$V_{\text{disc}}$")

    #     plt.ylim((-6, 83))
    #     plt.legend()
    #     plt.savefig(floc+"raw_data.pdf")
    #     plt.close()

    if use_fits:
        nfw_samples  = Vobs_MCMC(table, i_table, data, bulged, profile="NFW")    # Vobs_MCMC() runs MCMC with Vobs_fit() from Vobs_fits.py
        mond_samples = Vobs_MCMC(table, i_table, data, bulged, profile="MOND")

        # v_LCDM = nfw_samples["Vpred"][np.argmax(nfw_samples["log_likelihood"])]
        # v_MOND = mond_samples["Vpred"][np.argmax(mond_samples["log_likelihood"])]
        # full_LCDM = Vobs_scat( np.array([v_LCDM] * num_samples).T, data["errV"], num_samples)   # Assume errV completely UNcorrelated
        # full_MOND = Vobs_scat( np.array([v_MOND] * num_samples).T, data["errV"], num_samples)

        # Select 1000 random samples from MCMC fits.
        rand_idx = np.random.choice( 20000, 1000, replace=False )
        full_LCDM = nfw_samples["Vpred scattered"][rand_idx].T
        full_Vbar_LCDM = nfw_samples["Vbar"][rand_idx].T
        full_MOND = mond_samples["Vpred scattered"][rand_idx].T
        full_Vbar_MOND = mond_samples["Vbar"][rand_idx].T
        
        v_LCDM = nfw_samples["Vpred"][np.argmax(nfw_samples["log_likelihood"])]
        Vbar_LCDM = nfw_samples["Vbar"][np.argmax(nfw_samples["log_likelihood"])]
        v_MOND = mond_samples["Vpred"][np.argmax(mond_samples["log_likelihood"])]
        Vbar_MOND = mond_samples["Vbar"][np.argmax(mond_samples["log_likelihood"])]

        if make_plots:
            # labels = ["Distance", "Rc", "rho0", "Disk M/L"]
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

    # else:
    #     # Get v_DM from Fedir's LCDM abundance matching.
    #     v_DM = np.load("/mnt/users/koe/v_DM/NGC1560.npy")

    #     def LCDM_unc(Vbar2_unc, num_samples=num_samples):
    #         vDM_unc = np.array([v_DM] * num_samples).T
    #         return np.sqrt(Vbar2_unc + vDM_unc**2)
        
    #     v_LCDM = np.sqrt(Vbar2 + np.array(v_DM)**2)
    #     v_MOND = np.sqrt(MOND_vsq(r, Vbar2))
    #     full_LCDM = Vobs_scat(LCDM_unc(Vbar2_unc), data["errV"], num_samples)
    #     full_MOND = Vobs_scat(MOND_unc(r, Vbar2_unc, num_samples), data["errV"], num_samples)
    
    v_components = np.array([ Vbar, data["Vobs"], v_MOND, v_LCDM, Vbar_MOND, Vbar_LCDM, data["errV"] ])  # For GP fits + feature extraction
    v_data = np.array([ np.sqrt(Vbar2), data["Vobs"] ])
    # v_mock = np.array([ np.sqrt(Vbar2_unc), np.sqrt(Vbar2_unc), full_MOND, full_LCDM ])
    v_mock = np.array([ full_Vbar_MOND, full_Vbar_LCDM, full_MOND, full_LCDM ])

    main(args, r.to_numpy(), rad, v_components, v_data, v_mock)


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
        norm_cost = np.array(norm_cost)

        # Arrays of shape (5 x percentiles, 3 x v_comps, galaxy_count).
        norm_percentiles = np.percentile(norm_cost, [5.0, 16.0, 50.0, 84.0, 95.0], axis=1)
        # dtw_percentiles = np.percentile(dtw_cost, [5.0, 16.0, 50.0, 84.0, 95.0], axis=1)

        galaxies = [ "NGC1560" ]
        galaxy_count = 1
        
        fig, ax = plt.subplots()

        # Plot histogram of normalized DTW alignment costs of all galaxies.
        # if fname_DTW == fileloc+"dtw/cost_vsLCDM/": plt.title(r"Normalized DTW alignment costs (relative to $\Lambda$CDM)")
        # elif fname_DTW == fileloc+"dtw/cost_vsVbar/": plt.title("Normalized DTW alignment costs (relative to Vbar)")
        # else: plt.title("Normalized DTW alignment costs (relative to MOND)")

        hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
        colours = [ 'k', 'mediumblue', 'tab:green' ]

        # if fname_DTW == fileloc+"dtw/cost_vsVbar/":
        #     plt.bar(galaxies, norm_percentiles[2][0], color=colours[0], alpha=0.3, label=hist_labels[0])

        for j in range(3):
            # if fname_DTW == fileloc+"dtw/":
            #     if j == 1: continue     # Only plot values for data and LCDM since cost(MOND) == 0.
            # elif fname_DTW == fileloc+"dtw/cost_vsLCDM/":
            #     if j == 2: continue     # Only plot values for data and MOND since cost(LCDM) == 0.
            mean_norm = np.nanmean(norm_percentiles[2][j])
            low_err = norm_percentiles[2][j] - norm_percentiles[1][j]
            up_err = norm_percentiles[3][j] - norm_percentiles[2][j]

            if j == 0:
                print("\nMean cost ("+hist_labels[j]+") = {:.4f}".format(mean_norm))
            else:
                print("\nMedian cost ("+hist_labels[j]+") = {:.4f}".format(norm_percentiles[2][j]))
                print("Upper error = {:.4f}".format(up_err))
                print("Lower error = {:.4f}".format(low_err))

            if j == 0:
                ax.axhline(y=mean_norm, color='k', linestyle='dashed', label="Data: {:.4f}".format(mean_norm))
            else:
                # low_norm1 = np.full(galaxy_count, np.nanmean(norm_percentiles[1][j]))
                # up_norm1 = np.full(galaxy_count, np.nanmean(norm_percentiles[3][j]))
                # plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j], alpha=0.25)

                # if not(fname_DTW == fileloc+"dtw/cost_vsVbar/" and j == 0):
                if j == 1: trans = Affine2D().translate(-0.1, 0.0) + ax.transData
                else: trans = Affine2D().translate(+0.1, 0.0) + ax.transData
                ax.errorbar(galaxies, norm_percentiles[2][j], [[low_err], [up_err]], fmt='.', ls='none',
                            capsize=2, color=colours[j], alpha=0.8, transform=trans, label=f"{hist_labels[j]}: Mean = {mean_norm:.4f}")
        
        ax.set_ylim(bottom=0.0)
        ax.legend()
        ax.set_xticks([])
        ax.set_ylabel("Normalized DTW cost")
        fig.savefig(fname_DTW+"histo1.pdf", dpi=300, bbox_inches="tight")
        # plt.savefig(fname_DTW+"corr_scat/histo1.pdf", dpi=300, bbox_inches="tight")
        plt.close()


        """
        Scatter plots of cost(mock) against cost(data).
        """
        # plotloc = [ "MOND", "LCDM" ]
        # for j in range(1, 3):
        #     plt.title("Scatter plot: cost("+hist_labels[j]+") vs cost(data)")
        #     low_err = norm_percentiles[2][j] - norm_percentiles[1][j]
        #     up_err = norm_percentiles[3][j] - norm_percentiles[2][j]

        #     plt.xlabel("Cost(Data)")
        #     plt.ylabel("Cost("+hist_labels[j]+")")
        #     plt.errorbar(norm_percentiles[2][0], norm_percentiles[2][j], [low_err, up_err], fmt='.', ls='none', capsize=2, color=colours[j], alpha=0.5, label=hist_labels[j])
        
        #     plt.legend()
        #     plt.savefig(fname_DTW+"scatter_"+plotloc[j-1]+".png", dpi=300, bbox_inches="tight")
        #     plt.close()


        """
        Plot histogram of differences in normalized DTW costs (mock - data, in ascending order of costs for MOND - data).
        """
        # # Rearrange galaxies into ascending order in cost_diff(MOND).
        # cost_diff = np.array([norm_cost[1] - norm_cost[0], norm_cost[2] - norm_cost[0]])

        # # Arrays of shape (5 x percentiles, 2 x v_comps, galaxy_count).
        # diff_perc = np.percentile(cost_diff, [5.0, 16.0, 50.0, 84.0, 95.0], axis=2)

        # # Sort by descending order in difference between (LCDM - data).
        # sort_args = np.argsort(diff_perc[2][1])[::-1]
        # diff_percentiles = diff_perc[:, :, sort_args]

        # # Plot histogram of normalized DTW alignment costs of all galaxies.
        # plt.title("Normalised cost differences (mock - real data)")
        # hist_labels = [ "MOND", r"$\Lambda$CDM" ]
        # colours = [ 'mediumblue', 'tab:green' ]

        # for j in range(2):          
        #     mean_diff = np.nanmean(diff_percentiles[2][j])
        #     low_err = diff_percentiles[2][j] - diff_percentiles[1][j]
        #     up_err = diff_percentiles[3][j] - diff_percentiles[2][j]

        #     low_norm1 = np.full(galaxy_count, np.nanmean(diff_percentiles[1][j]))
        #     # low_norm2 = np.full(galaxy_count, np.nanmean(diff_percentiles[0][j]))
        #     up_norm1 = np.full(galaxy_count, np.nanmean(diff_percentiles[3][j]))
        #     # up_norm2 = np.full(galaxy_count, np.nanmean(diff_percentiles[4][j]))

        #     plt.errorbar(galaxies, diff_percentiles[2][j], [low_err, up_err], fmt='.', ls='none', capsize=2, color=colours[j], alpha=0.5, label=hist_labels[j])
        #     plt.axhline(y=mean_diff, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_diff))
        #     plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j], alpha=0.25)
        #     # plt.fill_between(galaxies, low_norm2, up_norm2, color=colours[j], alpha=0.1)
        
        # plt.legend()
        # plt.xticks([])
        # plt.savefig(fname_DTW+"histo2.pdf", dpi=300, bbox_inches="tight")
        # # plt.savefig(fname_DTW+"corr_scat/histo2.pdf", dpi=300, bbox_inches="tight")
        # plt.close()

    """
    Plot histogram of Pearson coefficients across RC (in ascending order of coefficients for data).
    """
    if do_correlations:
        """Pearson histogram"""
        # Rearrange galaxies into ascending order in median of corr(MOND, Vbar).
        # dim = (# of galaxies, 2 x mock_vcomps, 3 x percentiles)
        mock_sorted = np.array(sorted(pearson_mock, key=lambda x: x[0][0]))

        plt.title("Pearson coefficients across RC")
        hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
        colours = [ 'k', 'mediumblue', 'tab:green' ]

        # mean_corr = np.nanmean(pearson_data)
        plt.bar(galaxies, sorted(pearson_data), color=colours[0], alpha=0.3, label=hist_labels[0])
        plt.axhline(y=pearson_data, color=colours[0], linestyle='dashed', label="Mean = {:.4f}".format(pearson_data[0]))

        print("\nPearson correlation (Data) = {:.4f}".format(pearson_data[0]))

        for j in range(2):
            # med_corr = np.nanmean(mock_sorted[:,j,1])
            low_err = mock_sorted[:,j,1] - mock_sorted[:,j,0]
            up_err = mock_sorted[:,j,2] - mock_sorted[:,j,1]

            print("\nMedian correlation ("+hist_labels[j+1]+") = {:.4f}".format(mock_sorted[0,j,1]))
            print("Upper correlation = {:.4f}".format(up_err[0]))
            print("Lower correlation = {:.4f}".format(low_err[0]))

            # low_norm1 = np.full(galaxy_count, np.nanmean(mock_sorted[:,j,2]))
            # up_norm1 = np.full(galaxy_count, np.nanmean(mock_sorted[:,j,0]))

            plt.errorbar(galaxies, mock_sorted[:,j,1], [low_err, up_err], fmt='.', ls='none', capsize=2, color=colours[j+1], alpha=0.5, label=hist_labels[j+1])
            # plt.axhline(y=med_corr, color=colours[j+1], linestyle='dashed', label="Mean = {:.4f}".format(med_corr))
            # plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j+1], alpha=0.25)
        
        plt.legend()
        plt.xticks([])
        plt.savefig(fileloc+"corr_radii/pearson/histo1.pdf", dpi=300, bbox_inches="tight")
        plt.close()

print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)

