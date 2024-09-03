#!/usr/bin/env python
"""
Correlation coefficients + dynamic time warping on GP residuals of FAKE1560,
a fabricated dataset which looks more like the plot people quote (e.g. Stacy);
taking into account uncertainties (Vbar) and Vobs scattering (errV).

The following paper (Broeils 1992) analyzes NGC 1560 in some detail, thus might be useful:
https://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?bibcode=1992A%26A...256...19B&db_key=AST&page_ind=0&plate_select=NO&data_type=GIF&type=SCREEN_GIF&classic=YES
"""

import jax.experimental
import pandas as pd
import argparse
from resource import getrusage, RUSAGE_SELF
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, interpolate, stats
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
# from utils_analysis.mock_gen import Vbar_sq_unc, MOND_unc

matplotlib.use("Agg")  # noqa: E402


make_plots = True
do_DTW = True
do_correlations = True

fileloc = "/mnt/users/koe/plots/FAKE1560/"   # Directory for saving plots.


# Main code to run.
def main(args, g, r, Y, rad):
    """
    Do inference for Vbar with uniform prior for correlation length,
    then apply the resulted lengthscale to Vobs (both real and mock data).
    """
    v_comps = [ "Vobs (SPARC)", "Vobs (MOND)", "Vobs (LCDM)", "Vbar (SPARC)" ]
    colours = [ 'k', 'mediumblue', 'tab:green', 'tab:red' ]
    mean_prediction = []
    percentiles = []

    # GP on Vbar with uniform prior on length.
    print("Fitting function to " + v_comps[0] + "...")
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, r, Y[0])

    # do prediction
    vmap_args = (
        random.split(rng_key_predict, samples["var"].shape[0]),
        samples["var"],
        samples["length"],
        samples["noise"],
    )
    means, predictions = vmap(
        lambda rng_key, var, length, noise: predict(
            rng_key, r, Y[0], rad, var, length, noise, use_cholesky=args.use_cholesky
        )
    )(*vmap_args)

    mean_pred = np.mean(means, axis=0)
    mean_prediction.append(mean_pred)

    confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
    percentiles.append(confidence_band)

    if make_plots:
        labels = ["length", "var", "noise"]
        samples_arr = np.vstack([samples[label] for label in labels]).T
        fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
        fig.savefig(fileloc+"corner_Vobs.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # GP on Vbar and mock Vobs with fixed lengthscale from Vobs (data).
    vr = np.median(samples["var"])
    ls = np.median(samples["length"])
    ns = np.median(samples["noise"])
    for j in range(1, 4):
        print("\nFitting function to " + v_comps[j] + " with length = " + str(round(ls, 2)) + "...")
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))

        # do prediction
        vmap_args = (
            random.split(rng_key_predict, samples["var"].shape[0]),
        )
        means, predictions = vmap(
            lambda rng_key: predict(
                rng_key, r, Y[j], rad, vr, ls, ns, use_cholesky=args.use_cholesky
            )
        )(*vmap_args)

        mean_pred = np.mean(means, axis=0)
        mean_prediction.append(mean_pred)

        confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
        percentiles.append(confidence_band)

    
    # Compute residuals of fits.
    res_Vobs, res_MOND, res_LCDM, res_Vbar = [], [] ,[], []
    for k in range(len(r)):
        idx = (np.abs(rad - r[k])).argmin()
        res_Vobs.append(Y[0][k] - mean_prediction[0][idx])
        res_MOND.append(Y[1][k] - mean_prediction[1][idx])
        res_LCDM.append(Y[2][k] - mean_prediction[2][idx])
        res_Vbar.append(Y[3][k] - mean_prediction[3][idx])
    residuals = np.array([ res_Vobs, res_MOND, res_LCDM, res_Vbar ])


    """
    DTW on GP residuals.
    """
    if do_DTW:
        print("Warping time dynamically... or something like that...")        
        # Construct distance matrices.
        dist_data = np.zeros((len(r), len(r)))
        dist_MOND = np.copy(dist_data)
        dist_LCDM = np.copy(dist_data)
        for n in range(len(r)):
            for m in range(len(r)):
                dist_data[n, m] = abs(res_Vobs[n] - res_Vbar[m])
                dist_MOND[n, m] = abs(res_MOND[n] - res_Vbar[m])
                dist_LCDM[n, m] = abs(res_LCDM[n] - res_Vbar[m])
        
        dist_mats = np.array([ dist_data, dist_MOND, dist_LCDM ])
        mats_dir = [ "data", "MOND", "LCDM" ]
        
        # DTW!
        for j in range(3):
            path, cost_mat = dtw(dist_mats[j])
            x_path, y_path = zip(*path)
            cost = cost_mat[ len(r)-1, len(r)-1 ]
            dtw_cost[j].append(cost)
            norm_cost[j].append(cost / (2 * len(r)))

            if make_plots:
                # Plot distance matrix and cost matrix with optimal path.
                plt.title("Dynamic time warping: "+g)
                plt.axis('off')

                plt.subplot(121)
                plt.title("Distance matrix")
                plt.imshow(dist_mats[j], cmap=plt.cm.binary, interpolation="nearest", origin="lower")

                plt.subplot(122)
                plt.title("Cost matrix")
                plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
                plt.plot(x_path, y_path)

                plt.savefig(fileloc+"dtw/matrix_"+mats_dir[j]+".png", dpi=300, bbox_inches="tight")
                plt.close('all')

                # Visualize DTW alignment.
                plt.title("DTW alignment: "+g)

                diff = abs(max(res_Vbar) - min(residuals[j]))
                for x_i, y_j in path:
                    plt.plot([x_i, y_j], [residuals[j][x_i] + diff, res_Vbar[y_j] - diff], c="C7", alpha=0.4)
                plt.plot(np.arange(len(r)), np.array(residuals[j]) + diff, c=colours[j], label=v_comps[j])
                plt.plot(np.arange(len(r)), np.array(res_Vbar) - diff, c="red", label="Vbar")
                plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(cost))
                plt.plot([], [], c='w', label="Normalized cost = {:.4f}".format(cost/(len(r)*2)))

                plt.axis("off")
                plt.legend(bbox_to_anchor=(1,1))
                plt.savefig(fileloc+"dtw/vis_"+mats_dir[j]+".png", dpi=300, bbox_inches="tight")
                plt.close('all')
    
    """
    Code for PCHIP on GP residuals.
    """
    if do_correlations:
        # Interpolate the residuals with cubic Hermite spline splines.
        v_d0, v_d1, v_d2 = [], [], []
        for v_comp in residuals:
            v_d0.append(interpolate.pchip_interpolate(r, v_comp, rad))
            v_d1.append(interpolate.pchip_interpolate(r, v_comp, rad, der=1))
            v_d2.append(interpolate.pchip_interpolate(r, v_comp, rad, der=2))
                
        res_fits = [ v_d0, v_d1, v_d2 ]


        """
        ---------------------------------------------------
        Correlation plots using sphers of increasing radius
        ---------------------------------------------------
        """
        print("Computing correlation coefficients with increasing radii...")
        # Correlate Vobs and Vbar (d0, d1, d2) as a function of (maximum) radius, i.e. spheres of increasing r.
        # correlations_r = rad_corr arrays with [ data, MOND, LCDM ], so 3 Vobs x 3 derivatives x 2 correlations each,
        # where rad_corr = [ [[Spearman d0], [Pearson d0]], [[S d1], [P d1]], [[S d2], [P d2]] ].
        correlations_r = []
        for i in range(3):
            rad_corr = [ [[], []], [[], []], [[], []] ]
            for k in range(3):
                for j in range(10, len(rad)):
                    rad_corr[k][0].append(stats.spearmanr(res_fits[k][3][:j], res_fits[k][i][:j])[0])
                    rad_corr[k][1].append(stats.pearsonr(res_fits[k][3][:j], res_fits[k][i][:j])[0])
            correlations_r.append(rad_corr)


        """
        Plot GP fits, residuals (+ PCHIP) and correlations.
        """
        color_bar = "orange"
        deriv_dir = [ "d0", "d1", "d2" ]

        # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
        bar_ratio = []
        for rd in tqdm(range(len(rad))):
            bar_ratio.append(sum(mean_prediction[3][:rd]/mean_prediction[0][:rd]) / (rd+1))

        if make_plots:
            # Plot corrletaions as 1 main plot + 1 subplot, using only Vobs from data for Vbar/Vobs.
            der_axis = [ "Residuals (km/s)", "1st derivative", "2nd derivative" ]
            for der in range(3):
                fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
                fig1.set_size_inches(7, 7)
                ax0.set_title("Residuals correlation: "+g)
                ax0.set_ylabel("Velocities (km/s)")

                for j in range(4):
                    if j == 0:
                        ax0.errorbar(r, Y[j], data["errV"], color=colours[j], alpha=0.3, ls='none', fmt='o', capsize=2.5)
                    else:
                        # Scatter plot for data/mock data points.
                        ax0.scatter(r, Y[j], color=colours[j], alpha=0.3)
                    # Plot mean prediction from GP.
                    ax0.plot(rad, mean_prediction[j], color=colours[j], label=v_comps[j])

                    # if j == 2:
                    #     ylow, yhigh = np.percentile(nfw_samples["Vpred"]/Vmax, [16.0, 84.0], axis=0)
                    #     ax0.fill_between(r, ylow, yhigh, color=colours[j], alpha=0.3)

                    # Fill in 1-sigma (68%) confidence band of GP fit.
                    ax0.fill_between(rad, percentiles[j][0, :], percentiles[j][1, :], color=colours[j], alpha=0.2)

                ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
                ax0.grid()

                ax1.set_ylabel(der_axis[der])
                for j in range(4):
                    if der == 0:
                        ax1.scatter(r, residuals[j], color=colours[j], alpha=0.3)
                    ax1.plot(rad, res_fits[der][j], color=colours[j], label=v_comps[j])

                ax1.grid()

                # ax2.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
                ax2.set_xlabel('Radius (kpc)')
                ax2.set_ylabel("Correlations")
                
                vel_comps = [ "Data", "MOND", r"$\Lambda$CDM" ]

                for j in range(3):
                    # Plot correlations and Vbar/Vobs.
                    ax2.plot(rad[10:], correlations_r[j][der][0], color=colours[j], label=vel_comps[j]+r": Spearman $\rho$")
                    ax2.plot(rad[10:], correlations_r[j][der][1], ':', color=colours[j], label=vel_comps[j]+r": Pearson $\rho$")
                    # ax2.plot([], [], ' ', label=vel_comps[j]+r": $\rho_s=$"+str(round(correlations[j][der][0], 3))+r", $\rho_p=$"+str(round(correlations[j][der][1], 3)))
                    ax2.plot([], [], ' ', label=vel_comps[j]+r": $\rho_s=$"+str(round(stats.spearmanr(correlations_r[j][der][0], bar_ratio[10:])[0], 3))+r", $\rho_p=$"+str(round(stats.pearsonr(correlations_r[j][der][1], bar_ratio[10:])[0], 3)))

                ax5 = ax2.twinx()
                ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
                ax5.plot(rad[10:], bar_ratio[10:], '--', color=color_bar, label="Vbar/Vobs")
                ax5.tick_params(axis='y', labelcolor=color_bar)
                
                ax2.legend(bbox_to_anchor=(1.64, 1.3))
                ax2.grid()

                plt.subplots_adjust(hspace=0.05)
                fig1.savefig(fileloc+"radii_"+deriv_dir[der]+".png", dpi=300, bbox_inches="tight")
                plt.close()


        """
        -----------------------------------------------------------------------
        Correlation plots using windows of length max{1 * Reff, 5 data points}.
        -----------------------------------------------------------------------
        """
        print("Computing correlation coefficients with moving window...")
        if len(rad) > 100:
            # Correlate Vobs and Vbar (d0, d1, d2) along a moving window of length 1 * Reff.
            # correlations_w = win_corr arrays with [ data, MOND, LCDM ], so 3 Vobs x 3 derivatives x 2 correlations each,
            # where win_corr = [ [[Spearman d0], [Pearson d0]], [[S d1], [P d1]], [[S d2], [P d2]] ].
            wmax = len(rad) - 50
            correlations_w = []
            
            for vc in range(3):
                win_corr = [ [[], []], [[], []], [[], []] ]
                for der in range(3):
                    for j in range(50, wmax):

                        idx = (np.abs(r - rad[j])).argmin()
                        X_jmin, X_jmax = math.ceil(r[max(0, idx-2)] * 100), math.ceil(r[min(len(r)-1, idx+2)] * 100)
                        
                        if X_jmax - X_jmin > 100:
                            win_corr[der][0].append(stats.spearmanr(res_fits[der][3][X_jmin:X_jmax], res_fits[der][vc][X_jmin:X_jmax])[0])
                            win_corr[der][1].append(stats.pearsonr(res_fits[der][3][X_jmin:X_jmax], res_fits[der][vc][X_jmin:X_jmax])[0])
                        else:
                            jmin, jmax = j - 50, j + 50
                            win_corr[der][0].append(stats.spearmanr(res_fits[der][3][jmin:jmax], res_fits[der][vc][jmin:jmax])[0])
                            win_corr[der][1].append(stats.pearsonr(res_fits[der][3][jmin:jmax], res_fits[der][vc][jmin:jmax])[0])

                    # Apply SG filter to smooth out correlation curves for better visualisation.
                    win_corr[der][0] = signal.savgol_filter(win_corr[der][0], 50, 2)
                    win_corr[der][1] = signal.savgol_filter(win_corr[der][1], 50, 2)

                correlations_w.append(win_corr)

            # Compute average baryonic dominance (using Vobs from SPARC data) in moving window.
            wbar_ratio = []
            for j in tqdm(range(50, wmax)):
                wbar_ratio.append( sum( mean_prediction[3][j-50:j+50] / mean_prediction[0][j-50:j+50] ) / 101 )


            """
            1. Plot GP fits, residuals (+ PCHIP) and correlations.
            """
            if make_plots:
                # Plot corrletaions as 1 main plot + 1 subplot, using only Vobs from data for Vbar/Vobs.
                for der in range(3):
                    fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
                    fig1.set_size_inches(7, 7)
                    ax0.set_title("Moving window correlation: "+g)
                    ax0.set_ylabel("Velocities (km/s)")

                    for j in range(4):
                        # Scatter plot for data/mock data points.
                        ax0.scatter(r, Y[j], color=colours[j], alpha=0.3)
                        # Plot mean prediction from GP.
                        ax0.plot(rad, mean_prediction[j], color=colours[j], label=v_comps[j])
                        # Fill in 1-sigma (68%) confidence band of GP fit.
                        ax0.fill_between(rad, percentiles[j][0, :], percentiles[j][1, :], color=colours[j], alpha=0.2)

                    ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
                    ax0.grid()

                    ax1.set_ylabel(der_axis[der])
                    for j in range(4):
                        if der == 0:
                            ax1.scatter(r, residuals[j], color=colours[j], alpha=0.3)
                        ax1.plot(rad, res_fits[der][j], color=colours[j], label=v_comps[j])

                    ax1.grid()

                    # ax2.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
                    ax2.set_xlabel('Radius (kpc)')
                    ax2.set_ylabel("Correlations")

                    for j in range(3):
                        # Plot correlations and Vbar/Vobs.
                        ax2.plot(rad[50:wmax], correlations_w[j][der][0], color=colours[j], label=vel_comps[j]+r": Spearman $\rho$")
                        ax2.plot(rad[50:wmax], correlations_w[j][der][1], ':', color=colours[j], label=vel_comps[j]+r": Pearson $\rho$")
                        ax2.plot([], [], ' ', label=vel_comps[j]+r": $\rho_s=$"+str(round(stats.spearmanr(correlations_w[j][der][0], wbar_ratio)[0], 3))+r", $\rho_p=$"+str(round(stats.pearsonr(correlations_w[j][der][1], wbar_ratio)[0], 3)))

                    ax5 = ax2.twinx()
                    ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
                    ax5.plot(rad[50:wmax], wbar_ratio, '--', color=color_bar, label="Vbar/Vobs")
                    ax5.tick_params(axis='y', labelcolor=color_bar)
                    
                    ax2.legend(bbox_to_anchor=(1.64, 1.3))
                    ax2.grid()

                    plt.subplots_adjust(hspace=0.05)
                    fig1.savefig(fileloc+"window_"+deriv_dir[der]+".png", dpi=300, bbox_inches="tight")
                    plt.close()
    
    print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
    jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem.


if __name__ == "__main__":
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
    parser.add_argument("--testing", default=True, type=bool)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    
    galaxy, correlations_ALL = [], []
    dtw_cost = [ [], [], [] ]
    norm_cost = [ [], [], [] ]

    """
    Plotting galaxy rotation curves directly from data with variables:
    Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
    """
    file_path = "/mnt/users/koe/data/FAKE1560.dat"
    rawdata = np.loadtxt(file_path)
    columns = [ "Rad", "Vobs", "errV", "Sdst",
                "Vdisk", "Sdgas", "Vgas", "Vgth" ]
    data = pd.DataFrame(rawdata, columns=columns)
    r = data["Rad"]
    # r /= 1.3    # Scale length of NGC 1560 according to Broeils.
    bulged = False

    table = { "D":[2.99], "e_D":[0.1], "Inc":[82.0], "e_Inc":[1.0] }
    i_table = 0

    # Normalise velocities by Vmax = max(Vobs) from SPARC data.
    Vbar_squared = Vbar_sq(data, bulged)
    nfw_samples = Vobs_MCMC(table, i_table, data, bulged, profile="NFW")
    mond_samples = Vobs_MCMC(table, i_table, data, bulged, profile="MOND")
    v_LCDM = np.median(nfw_samples["Vpred"], axis=0)
    v_MOND = np.median(mond_samples["Vpred"], axis=0)

    if make_plots:
        # labels = ["Distance", "Rc", "rho0", "Disk M/L"]
        labels = [ "Distance", "Disk M/L", "logM200c", "logc" ]
        samples_arr = np.vstack([nfw_samples[label] for label in labels]).T
        fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
        fig.savefig(fileloc+"corner_NFW.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        labels = [ "Distance", "Disk M/L" ]
        samples_arr = np.vstack([mond_samples[label] for label in labels]).T
        fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
        fig.savefig(fileloc+"corner_MOND.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    v_components = np.array([ data["Vobs"], v_MOND, v_LCDM, np.sqrt(Vbar_squared) ])
    Vmax = max(v_components[0])
    data["errV"] /= Vmax
    v_components /= Vmax

    rad_count = math.ceil((max(r)-min(r))*100)
    rad = np.linspace(min(r), max(r), rad_count)

    main(args, "FAKE1560", r.to_numpy(), v_components, rad)
