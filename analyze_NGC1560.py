#!/usr/bin/env python
"""
Correlation coefficients + dynamic time warping on GP residuals of NGC1560_Stacy,
a dataset obtained from digitizing Stacy's NGC 1560 plot with Plot Digitizer;
taking into account uncertainties (Vbar) and Vobs scattering (errV).

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
from utils_analysis.Vobs_fits import Vbar_sq, MOND_vsq
from utils_analysis.mock_gen import Vobs_MCMC
from utils_analysis.mock_gen import Vbar_sq_unc, MOND_unc, Vobs_scat
from utils_analysis.extract_ft import ft_check

matplotlib.use("Agg")
plt.rcParams.update({'font.size': 13})


plot_digitizer = True
use_fits = True

make_plots = False
do_DTW = False
do_correlations = False

floc = "/mnt/users/koe/plots/NGC1560/"   # Directory for saving plots.
if plot_digitizer:
    floc += "plot_digitizer/"
if use_fits: 
    fileloc = floc + "use_fits/"
else:
    fileloc = floc

# Options: cost wrt MOND: "dtw/"; cost wrt LCDM: "dtw/cost_vsLCDM/", original cose (MSE): "dtw/cost_vsVbar/".
if do_DTW:
    fname_DTW = fileloc + "dtw/cost_vsVbar/"
    print(f"fname_DTW = {fname_DTW}")

num_samples = 1000


# Main code to run.
def main(args, r, rad, Y, v_data, v_mock, num_samples=num_samples, ls:float=4.5):
    """
    Do inference for Vbar with uniform prior for correlation length,
    then apply the resulted lengthscale to Vobs (both real and mock data).
    """
    v_comps = [ "Vbar (SPARC)", "Vobs (SPARC)", "Vobs (MOND)", "Vobs (LCDM)" ]
    colours = [ 'tab:red', 'k', 'mediumblue', 'tab:green' ]
    mean_prediction = []
    percentiles = []
    
    for j in range(4):
        print(f"Fitting function to {v_comps[j]} with ls = {ls} kpc...")
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = run_inference(model, args, rng_key, r, Y[j], ls=ls)

        # do prediction
        vmap_args = (
            random.split(rng_key_predict, samples["var"].shape[0]),
            samples["var"],
            samples["noise"],
        )

        r_Xft = np.delete(r, np.s_[19:23], axis=0)
        Vcomp_Xft = np.delete(Y[j], np.s_[19:23], axis=0)
        means, predictions = vmap(
            lambda rng_key, var, noise: predict(
                rng_key, r_Xft, Vcomp_Xft, rad, var, ls, noise, use_cholesky=args.use_cholesky
            )
        )(*vmap_args)
        # means, predictions = vmap(
        #     lambda rng_key, var, noise: predict(
        #         rng_key, r, Y[j], rad, var, ls, noise, use_cholesky=args.use_cholesky
        #     )
        # )(*vmap_args)

        mean_pred = np.mean(means, axis=0)
        mean_prediction.append(mean_pred)   # [ Vbar, Vobs, MOND, LCDM ]

        confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
        percentiles.append(confidence_band)

    # "Raw" percentiles from uncertainties and scattering.
    raw_median = np.percentile(v_mock, 50.0, axis=2)                # dim = (3, r)
    raw_percentiles = np.percentile(v_mock, [16.0, 84.0], axis=2)   # dim = (2, 3, r)
    raw_errors = np.abs( raw_percentiles - raw_median )             # dim = (2, 3, r)

    # Compute residuals of fits.
    res_Vbar_data, res_Vobs, res_Vbar_mock, res_MOND, res_LCDM = [], [] ,[], [], []
    for k in range(len(r)):
        idx = (np.abs(rad - r[k])).argmin()
        
        res_Vbar_data.append(v_data[0][k] - mean_prediction[0][idx])
        res_Vobs.append(v_data[1][k] - mean_prediction[1][idx])

        res_Vbar_mock.append(v_mock[0][k] - mean_prediction[0][idx])
        res_MOND.append(v_mock[1][k] - mean_prediction[2][idx])
        res_LCDM.append(v_mock[2][k] - mean_prediction[3][idx])

    # print(f"Normalized res_Vobs: {res_Vobs[19:23] / Y[4][19:23]}")
    # print(f"Normalized res_Vbar: {res_Vbar_data[19:23] / Y[4][19:23]}")
    # raise ValueError("Stop here.")

    res_data = np.array([ res_Vbar_data, res_Vobs ])            # dim = (2, len(r))
    res_mock = np.array([ res_Vbar_mock, res_MOND, res_LCDM ])  # dim = (3, len(r), num_samples)

    # Residual percentiles from uncertainties and scattering; dimensions = (3, 1 or 2, len(r)).
    res_median = np.percentile(res_mock, 50.0, axis=2)                  # dim = (3, r)
    res_percentiles = np.percentile(res_mock, [16.0, 84.0], axis=2)     # dim = (2, 3, r)
    # res_errors = np.abs( res_percentiles - res_median )               # dim = (2, 3, r)

    # Extract properties of feature(s) (if any).
    lb_ft, rb_ft, ft_widths = ft_check( np.array(res_data[0]), raw_errors[1,0] )
    print("Feature(s) in Vbar:")
    print(lb_ft)
    print(rb_ft)
    print(ft_widths)

    lb_ft, rb_ft, ft_widths = ft_check( np.array(res_data[1]), Y[4] )
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
                    if fname_DTW == fileloc+"dtw/":
                        dist_data[n, m] = np.abs(res_Vobs[n] - res_MOND[m][smp])
                        dist_MOND[n, m] = np.abs(res_MOND[n][smp] - res_MOND[m][smp])
                        dist_LCDM[n, m] = np.abs(res_LCDM[n][smp] - res_MOND[m][smp])

                    # Alternative constructions:
                    elif fname_DTW == fileloc+"dtw/cost_vsLCDM/":
                        dist_data[n, m] = np.abs(res_Vobs[n] - res_LCDM[m][smp])
                        dist_MOND[n, m] = np.abs(res_MOND[n][smp] - res_LCDM[m][smp])
                        dist_LCDM[n, m] = np.abs(res_LCDM[n][smp] - res_LCDM[m][smp])
                    else:
                        dist_data[n, m] = np.abs(res_Vobs[n] - res_Vbar_data[m])
                        dist_MOND[n, m] = np.abs(res_MOND[n][smp] - res_Vbar_data[m])
                        dist_LCDM[n, m] = np.abs(res_LCDM[n][smp] - res_Vbar_data[m])
                    # else:
                    #     dist_data[n, m] = np.abs(res_Vobs[n] - res_Vbar_mock[m][smp])
                    #     dist_MOND[n, m] = np.abs(res_MOND[n][smp] - res_Vbar_mock[m][smp])
                    #     dist_LCDM[n, m] = np.abs(res_LCDM[n][smp] - res_Vbar_mock[m][smp])
            
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
                    if fname_DTW == fileloc+"dtw/":
                        ref_curve = [ res_MOND, "mediumblue", "MOND" ]
                    elif fname_DTW == fileloc+"dtw/cost_vsLCDM/":
                        ref_curve = [ res_LCDM, "tab:green", r"$\Lambda$CDM" ]
                    else:
                        ref_curve = [ res_Vbar_data, "tab:red", "Vbar" ]

                    if j == 0:
                        # diff = abs(max(np.array(ref_curve[0])[:,smp]) - min(res_Vobs))
                        diff = abs(max(np.array(ref_curve[0])) - min(res_Vobs))
                        for x_i, y_j in path:
                            # plt.plot([x_i, y_j], [res_Vobs[x_i] + diff, ref_curve[0][y_j][smp] - diff], c="C7", alpha=0.4)
                            plt.plot([x_i, y_j], [res_Vobs[x_i] + diff, ref_curve[0][y_j] - diff], c="C7", alpha=0.4)
                        plt.plot(np.arange(len(r)), np.array(res_Vobs) + diff, c='k', label=v_comps[1])

                    else: 
                        diff = abs(max(np.array(ref_curve[0])) - min(np.array(res_mock)[j,:,smp]))
                        for x_i, y_j in path:
                            plt.plot([x_i, y_j], [res_mock[j][x_i][smp] + diff, ref_curve[0][y_j] - diff], c="C7", alpha=0.4)
                        plt.plot(np.arange(len(r)), np.array(res_mock)[j,:,smp] + diff, c=colours[j+1], label=v_comps[j+1])

                    plt.plot(np.arange(len(r)), np.array(ref_curve[0]) - diff, c=ref_curve[1], label=ref_curve[2])
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
            if smp % 5:
                continue

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
                    pearsonr_mock.append(stats.pearsonr(res_mock[0,:j,smp], res_mock[i,:j,smp])[0])
                correlations_r.append(pearsonr_mock)
            radii_corr.append(correlations_r)
        
        res_mock_percentiles = np.percentile(res_mock, [16.0, 50.0, 84.0], axis=2)
        rcorr_percentiles = np.percentile(radii_corr, [16.0, 50.0, 84.0], axis=0)
        pearson_mock.append([ rcorr_percentiles[:,0,-1], rcorr_percentiles[:,1,-1] ])

        """
        Plot GP fits, residuals and correlations.
        """
        if make_plots:
            c_temp = [ 'tab:red', 'mediumblue', 'tab:green' ]
            labels_temp = [ "Vbar (SPARC)", "Vobs (SPARC)", "Vobs (MOND)", r"Vobs ($\Lambda$CDM)" ]

            """Pearson correlations."""
            fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
            fig1.set_size_inches(7, 7)
            # if plot_digitizer:
            #     ax0.set_title("Pearson correlation: NGC 1560 (Sanders 2007)")
            # else:
            #     ax0.set_title("Pearson correlation: NGC 1560 (Gentile et al. 2010)")
            if not plot_digitizer: ax0.set_ylabel("Velocities (km/s)")
            
            for j in range(4):
                if j == 1:  # Plot Vobs.
                    ax0.errorbar(r, v_data[1], Y[4], color='k', alpha=0.3, fmt='o', capsize=2)
                elif j == 0: # Plot Vbar.
                    ax0.errorbar(r, raw_median[0], raw_errors[:, 0], color='tab:red', alpha=0.3, fmt='o', capsize=2)
                else:
                    ax0.errorbar(r, raw_median[j-1], raw_errors[:, j-1], c=c_temp[j-1], alpha=0.3, fmt='o', capsize=2)
                # Plot mean prediction from GP.
                ax0.plot(rad, mean_prediction[j], color=colours[j], label=labels_temp[j])
                # Fill in 1-sigma (68%) confidence band of GP fit.
                ax0.fill_between(rad, percentiles[j][0], percentiles[j][1], color=colours[j], alpha=0.2)

            if plot_digitizer: ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
            ax0.grid()

            if not plot_digitizer: ax1.set_ylabel("Residuals (km/s)")
            for j in range(4):
                # Plots for mock Vobs + Vbar (sampled w/ uncertainties).
                if j == 3:  # Plot Vobs.
                    # if der == 0:
                    # ax1.errorbar(r[5:], res_data[1][5:], Y[4][5:], color='k', alpha=0.3, ls='none', fmt='o', capsize=2)
                    # ax1.plot(r[5:], res_data[1][5:], color='k', label=labels_temp[j])
                    ax1.errorbar(r, res_data[1], Y[4], color='k', alpha=0.3, ls='none', fmt='o', capsize=2)
                    ax1.plot(r, res_data[1], color='k', label=labels_temp[j])
                else:
                    # if der == 0:
                    # ax1.scatter(r[5:], res_median[j][5:], c=c_temp[j], alpha=0.3)
                    ax1.scatter(r, res_median[j], c=c_temp[j], alpha=0.3)
                    # ax1.plot(r[5:], res_mock_percentiles[1][j][5:], c=c_temp[j], label=labels_temp[j])
                    # ax1.fill_between(r[5:], res_mock_percentiles[0][j][5:], res_mock_percentiles[2][j][5:], color=c_temp[j], alpha=0.15)
                    ax1.plot(r, res_mock_percentiles[1][j], c=c_temp[j], label=labels_temp[j])
                    ax1.fill_between(r, res_mock_percentiles[0][j], res_mock_percentiles[2][j], color=c_temp[j], alpha=0.15)

            ax1.grid()

            ax2.set_xlabel("Radius (kpc)")
            if not plot_digitizer: ax2.set_ylabel(r"Correlations w.r.t. $v_{bar}$")

            for j in range(2):
                ax2.plot(r[4:], rcorr_percentiles[1][j][2:], c=c_temp[j+1], label=labels_temp[j+2]+r": Pearson $\rho$")
                ax2.fill_between(r[4:], rcorr_percentiles[0][j][2:], rcorr_percentiles[2][j][2:], color=colours[j+2], alpha=0.2)

            ax2.plot(r[4:], pearsonr_data[2:], c='k', label=r"Data: Pearson $\rho$")
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

    table = { "D":[2.99], "e_D":[0.1], "Inc":[82.0], "e_Inc":[1.0], "L":[23.8], "e_L":[1.8] }
    i_table = 0

    rad_count = math.ceil((max(r)-min(r))*100)
    rad = np.linspace(min(r), max(r), rad_count)
    # data["Vdisk"] /= np.sqrt(pdisk)   # Correction now implemented in data

    # Normalise velocities by Vmax = max(Vobs) from SPARC data.
    Vbar2_unc = Vbar_sq_unc(table, i_table, data, bulged, num_samples)
    Vbar2 = Vbar_sq(data, bulged)

    # if make_plots:
    #     # Plot Vobs and Vbar.
    #     # if plot_digitizer:
    #     #     plt.title("NGC 1560 (Sanders 2007) from manual digitization")
    #     # else:
    #     #     plt.title("NGC 1560 (Gentile et al. 2010) from S. McGaugh")
            
    #     plt.xlabel("Radius (kpc)")
    #     # plt.xlabel(" ")
    #     if plot_digitizer: plt.ylabel("Velocities (km/s)")
    #     else: plt.yticks(color='w')

    #     plt.errorbar(r, data["Vobs"], data["errV"], fmt=".", ls='none', capsize=2, c='k', label=r"$v_{\text{obs}}$")
    #     plt.plot(r, np.sqrt(Vbar2), c='k', linestyle='dashdot', label=r"$v_{\text{bar}}$")
    #     plt.plot(r, data["Vgas"], c='k', linestyle='dotted', label=r"$V_{\text{gas}}$")
    #     plt.plot(r, np.sqrt(0.5)*data["Vdisk"], c='k', linestyle='dashed', label=r"$V_{\text{disc}}$")

    #     plt.ylim((-6, 83))
    #     plt.legend()
    #     plt.savefig(floc+"raw_data.pdf")
    #     plt.close()

    if use_fits:
        nfw_samples = Vobs_MCMC(table, i_table, data, bulged, profile="NFW")    # Vobs_MCMC() runs MCMC with Vobs_fit() from Vobs_fits.py
        mond_samples = Vobs_MCMC(table, i_table, data, bulged, profile="MOND")
        v_LCDM = np.median(nfw_samples["Vpred"], axis=0)
        v_MOND = np.median(mond_samples["Vpred"], axis=0)
        full_LCDM = Vobs_scat( np.array([v_LCDM] * num_samples).T, data["errV"], num_samples)   # Assume errV completely UNcorrelated
        full_MOND = Vobs_scat( np.array([v_MOND] * num_samples).T, data["errV"], num_samples)
        # full_LCDM = Vobs_scat( np.array(nfw_samples["Vpred"]).T, data["errV"], num_samples=20000 )
        # full_MOND = Vobs_scat( np.array(mond_samples["Vpred"]).T, data["errV"], num_samples=20000 )
        # v_LCDM = np.median(full_LCDM, axis=1)
        # v_MOND = np.median(full_MOND, axis=1)

        if make_plots:
            # labels = ["Distance", "Rc", "rho0", "Disk M/L"]
            labels = [ "Distance", "Disk M/L", "L", "inc", "logM200c", "logc" ]
            samples_arr = np.vstack([nfw_samples[label] for label in labels]).T
            fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
            fig.savefig(fileloc+"corner_NFW.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            labels = [ "Distance", "Disk M/L", "L", "inc" ]
            samples_arr = np.vstack([mond_samples[label] for label in labels]).T
            fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
            fig.savefig(fileloc+"corner_MOND.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    else:
        # Get v_DM from Fedir's LCDM abundance matching.
        v_DM = np.load("/mnt/users/koe/v_DM/NGC1560.npy")

        def LCDM_unc(Vbar2_unc, num_samples=num_samples):
            vDM_unc = np.array([v_DM] * num_samples).T
            return np.sqrt(Vbar2_unc + vDM_unc**2)
        
        v_LCDM = np.sqrt(Vbar2 + np.array(v_DM)**2)
        v_MOND = np.sqrt(MOND_vsq(r, Vbar2))
        full_LCDM = Vobs_scat(LCDM_unc(Vbar2_unc), data["errV"], num_samples)
        full_MOND = Vobs_scat(MOND_unc(r, Vbar2_unc, num_samples), data["errV"], num_samples)
    
    v_components = np.array([ np.sqrt(Vbar2), data["Vobs"], v_MOND, v_LCDM, data["errV"]])  # For GP fits + feature extraction
    v_data = np.array([ np.sqrt(Vbar2), data["Vobs"] ])
    v_mock = np.array([ np.sqrt(Vbar2_unc), full_MOND, full_LCDM ])

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
        
        # Plot histogram of normalized DTW alignment costs of all galaxies.
        if fname_DTW == fileloc+"dtw/cost_vsLCDM/": plt.title(r"Normalized DTW alignment costs (relative to $\Lambda$CDM)")
        elif fname_DTW == fileloc+"dtw/cost_vsVbar/": plt.title("Normalized DTW alignment costs (relative to Vbar)")
        else: plt.title("Normalized DTW alignment costs (relative to MOND)")

        hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
        colours = [ 'k', 'mediumblue', 'tab:green' ]

        if fname_DTW == fileloc+"dtw/cost_vsVbar/":
            plt.bar(galaxies, norm_percentiles[2][0], color=colours[0], alpha=0.3, label=hist_labels[0])

        for j in range(3):
            if fname_DTW == fileloc+"dtw/":
                if j == 1: continue     # Only plot values for data and LCDM since cost(MOND) == 0.
            elif fname_DTW == fileloc+"dtw/cost_vsLCDM/":
                if j == 2: continue     # Only plot values for data and MOND since cost(LCDM) == 0.
            mean_norm = np.nanmean(norm_percentiles[2][j])
            low_err = norm_percentiles[2][j] - norm_percentiles[1][j]
            up_err = norm_percentiles[3][j] - norm_percentiles[2][j]

            print("Mean cost ("+hist_labels[j]+") = {:.4f}".format(mean_norm))
            print("Upper error = {:.4f}".format(np.nanmean(up_err)))
            print("Lower error = {:.4f}".format(np.nanmean(low_err)))

            plt.axhline(y=mean_norm, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_norm))
            if j != 0:
                low_norm1 = np.full(galaxy_count, np.nanmean(norm_percentiles[1][j]))
                up_norm1 = np.full(galaxy_count, np.nanmean(norm_percentiles[3][j]))
                plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j], alpha=0.25)

            if not(fname_DTW == fileloc+"dtw/cost_vsVbar/" and j == 0):
                plt.errorbar(galaxies, norm_percentiles[2][j], [low_err, up_err], fmt='.', ls='none',
                            capsize=2, color=colours[j], alpha=0.5, label=hist_labels[j])
        
        plt.legend()
        plt.xticks([])
        plt.savefig(fname_DTW+"histo1.pdf", dpi=300, bbox_inches="tight")
        # plt.savefig(fname_DTW+"corr_scat/histo1.png", dpi=300, bbox_inches="tight")
        plt.close()


        """
        Scatter plots of cost(mock) against cost(data).
        """
        plotloc = [ "MOND", "LCDM" ]
        for j in range(1, 3):
            plt.title("Scatter plot: cost("+hist_labels[j]+") vs cost(data)")
            low_err = norm_percentiles[2][j] - norm_percentiles[1][j]
            up_err = norm_percentiles[3][j] - norm_percentiles[2][j]

            plt.xlabel("Cost(Data)")
            plt.ylabel("Cost("+hist_labels[j]+")")
            plt.errorbar(norm_percentiles[2][0], norm_percentiles[2][j], [low_err, up_err], fmt='.', ls='none', capsize=2, color=colours[j], alpha=0.5, label=hist_labels[j])
        
            plt.legend()
            plt.savefig(fname_DTW+"scatter_"+plotloc[j-1]+".png", dpi=300, bbox_inches="tight")
            plt.close()


        """
        Plot histogram of differences in normalized DTW costs (mock - data, in ascending order of costs for MOND - data).
        """
        # Rearrange galaxies into ascending order in cost_diff(MOND).
        cost_diff = np.array([norm_cost[1] - norm_cost[0], norm_cost[2] - norm_cost[0]])

        # Arrays of shape (5 x percentiles, 2 x v_comps, galaxy_count).
        diff_perc = np.percentile(cost_diff, [5.0, 16.0, 50.0, 84.0, 95.0], axis=2)

        # Sort by descending order in difference between (LCDM - data).
        sort_args = np.argsort(diff_perc[2][1])[::-1]
        diff_percentiles = diff_perc[:, :, sort_args]

        # Plot histogram of normalized DTW alignment costs of all galaxies.
        plt.title("Normalised cost differences (mock - real data)")
        hist_labels = [ "MOND", r"$\Lambda$CDM" ]
        colours = [ 'mediumblue', 'tab:green' ]

        for j in range(2):          
            mean_diff = np.nanmean(diff_percentiles[2][j])
            low_err = diff_percentiles[2][j] - diff_percentiles[1][j]
            up_err = diff_percentiles[3][j] - diff_percentiles[2][j]

            low_norm1 = np.full(galaxy_count, np.nanmean(diff_percentiles[1][j]))
            # low_norm2 = np.full(galaxy_count, np.nanmean(diff_percentiles[0][j]))
            up_norm1 = np.full(galaxy_count, np.nanmean(diff_percentiles[3][j]))
            # up_norm2 = np.full(galaxy_count, np.nanmean(diff_percentiles[4][j]))

            plt.errorbar(galaxies, diff_percentiles[2][j], [low_err, up_err], fmt='.', ls='none', capsize=2, color=colours[j], alpha=0.5, label=hist_labels[j])
            plt.axhline(y=mean_diff, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_diff))
            plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j], alpha=0.25)
            # plt.fill_between(galaxies, low_norm2, up_norm2, color=colours[j], alpha=0.1)
        
        plt.legend()
        plt.xticks([])
        plt.savefig(fname_DTW+"histo2.png", dpi=300, bbox_inches="tight")
        # plt.savefig(fname_DTW+"corr_scat/histo2.png", dpi=300, bbox_inches="tight")
        plt.close()

    """
    Plot histogram of Spearman coefficients across RC (in ascending order of coefficients for data).
    """
    if do_correlations:
        """Pearson histogram"""
        # Rearrange galaxies into ascending order in median of corr(MOND, Vbar).
        # dim = (# of galaxies, 2 x mock_vcomps, 3 x percentiles)
        mock_sorted = np.array(sorted(pearson_mock, key=lambda x: x[0][0]))

        plt.title("Pearson coefficients across RC")
        hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
        colours = [ 'k', 'mediumblue', 'tab:green' ]

        mean_corr = np.nanmean(pearson_data)
        plt.bar(galaxies, sorted(pearson_data), color=colours[0], alpha=0.3, label=hist_labels[0])
        plt.axhline(y=mean_corr, color=colours[0], linestyle='dashed', label="Mean = {:.4f}".format(mean_corr))

        for j in range(2):
            med_corr = np.nanmean(mock_sorted[:,j,1])
            low_err = mock_sorted[:,j,1] - mock_sorted[:,j,0]
            up_err = mock_sorted[:,j,2] - mock_sorted[:,j,1]

            low_norm1 = np.full(galaxy_count, np.nanmean(mock_sorted[:,j,2]))
            up_norm1 = np.full(galaxy_count, np.nanmean(mock_sorted[:,j,0]))

            plt.errorbar(galaxies, mock_sorted[:,j,1], [low_err, up_err], fmt='.', ls='none', capsize=2, color=colours[j+1], alpha=0.5, label=hist_labels[j+1])
            plt.axhline(y=med_corr, color=colours[j+1], linestyle='dashed', label="Mean = {:.4f}".format(med_corr))
            plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j+1], alpha=0.25)
        
        plt.legend()
        plt.xticks([])
        plt.savefig(fileloc+"corr_radii/pearson/histo1.png", dpi=300, bbox_inches="tight")
        plt.close()

print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)

