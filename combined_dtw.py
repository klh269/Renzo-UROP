#!/usr/bin/env python
"""
# NOTE: Use 200GB RAM (-m 200) when queueing or job will get stopped (max > 180 GB).
No longer needed!! Way too many points were used for the GP fit, code now stays safely below 1 GB.

Gaussian process + dynamic time warping analysis,
combining all SPARC data with MOND and LCDM mock data.

Procedure:
For each galaxy, fit a smooth curve through Vobs and Vbar with GP,
then align their residuals using dynamic time warping.
Compare the normalised alginment cost to get a handle of correlation.

**Yet to account for errors/uncertainties, crucial for meaningful comparison/interpretation!
"""
import jax.experimental
import pandas as pd
import argparse
from resource import getrusage, RUSAGE_SELF

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

from tqdm import tqdm
from utils_analysis.gp_utils import model, predict, run_inference
from utils_analysis.dtw_utils import dtw

matplotlib.use("Agg")  # noqa: E402


testing = False      # Only analyze NGC0024.
test_multiple = False   # Loops over the first handful of galaxies instead of just the fist one (DDO161).
make_plots = True
do_DTW = False
do_correlations = False

fileloc = "/mnt/users/koe/plots/combined_dtw/"


# Main code to run.
def main(args, g, X, Y, X_test, ls): 
    """
    Do inference for Vbar with uniform prior for correlation length,
    then apply the resulted lengthscale to Vobs (both real and mock data).
    """
    v_comps = [ "Vbar (SPARC)", "Vobs (SPARC)", "Vobs (MOND)", "Vobs (LCDM)" ]
    colours = [ 'tab:red', 'k', 'mediumblue', 'tab:green' ]
    corner_dir = [ "Vbar/", "Vobs_data/", "Vobs_MOND/", "Vobs_LCDM/" ]

    # print(f"Fitting function to {v_comps[1]} with correlation length = {ls}...")
    # rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    # samples = run_inference(model, args, rng_key, X, Y[1])

    # # do prediction
    # vmap_args = (
    #     random.split(rng_key_predict, samples["var"].shape[0]),
    #     samples["var"],
    #     samples["noise"],
    # )
    # means, predictions = vmap(
    #     lambda rng_key, var, noise: predict(
    #         rng_key, X, Y[1], X_test, var, ls, noise, use_cholesky=args.use_cholesky
    #     )
    # )(*vmap_args)

    # mean_pred = np.mean(means, axis=0)
    # gp_predictions[1] = mean_pred

    # confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
    # gp_16percent[1] = confidence_band[0]
    # gp_84percent[1] = confidence_band[1]

    # if make_plots:
    #     labels = ["var", "noise"]
    #     samples_arr = np.vstack([samples[label] for label in labels]).T
    #     fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
    #     fig.savefig(fileloc+"corner_plots/"+corner_dir[1]+g+".png", dpi=300, bbox_inches="tight")
    #     plt.close(fig)

    # GP on Vobs with fixed hyperparameters from Vbar.
    # vr = stats.mode(np.round(samples["var"], 0))[0]
    # vr = np.median(samples["var"])
    # print(f"var MAP = {vr}")
    # ls = stats.mode(np.round(samples["length"], 2))[0]
    # ls = np.median(samples["length"])
    # ls_ALL.append(ls)
    # print(f"length MAP = {ls}")
    # ns = stats.mode(np.round(samples["noise"], 1))[0]
    # ns = np.median(samples["noise"])
    # print(f"noise MAP = {ns}")

    # print(f"\nFitting function to each RC with ls = {ls}...")
    for j in range(4):
        # print("\nFitting function to " + v_comps[j] + " with length = " + str(round(ls, 2)) + "...")
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = run_inference(model, args, rng_key, X, Y[j], ls=ls)

        # do prediction
        vmap_args = (
            random.split(rng_key_predict, samples["var"].shape[0]),
            samples["var"],
            samples["noise"],
        )
        means, predictions = vmap(
            lambda rng_key, var, noise: predict(
                rng_key, X, Y[j], X_test, var, ls, noise, use_cholesky=args.use_cholesky
            )
        )(*vmap_args)

        mean_pred = np.mean(means, axis=0)
        gp_predictions[j] = mean_pred

        confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
        gp_16percent[j] = confidence_band[0]
        gp_84percent[j] = confidence_band[1]

        if make_plots:
            labels = ["var", "noise"]
            samples_arr = np.vstack([samples[label] for label in labels]).T
            fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".5f", quantiles=[0.16, 0.5, 0.84], smooth=1)
            fig.savefig(fileloc+"corner_plots/"+corner_dir[j]+g+".png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    
    # Compute residuals of fits.
    if make_plots or do_DTW or do_correlations:
        res_Vbar, res_Vobs, res_MOND, res_LCDM = [], [] ,[], []
        for k in range(len(X)):
            idx = (np.abs(X_test - X[k])).argmin()
            res_Vbar.append(Y[0][k] - gp_predictions[0][idx])
            res_Vobs.append(Y[1][k] - gp_predictions[1][idx])
            res_MOND.append(Y[2][k] - gp_predictions[2][idx])
            res_LCDM.append(Y[3][k] - gp_predictions[3][idx])
        residuals = np.array([ res_Vbar, res_Vobs, res_MOND, res_LCDM ])

        if make_plots:
            # Plot GP fits as 1 main plot + 1 subplot (residuals).
            fig1, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2]})
            ax0.set_title("Residuals correlation: "+g)
            ax0.set_ylabel("Velocities (km/s)")

            for j in range(4):
                # Scatter plot for data/mock data points.
                ax0.scatter(X, Y[j], color=colours[j], alpha=0.3, label=v_comps[j])
                # Plot mean prediction from GP.
                ax0.plot(X_test, gp_predictions[j], color=colours[j])
                # Fill in 1-sigma (68%) confidence band of GP fit.
                ax0.fill_between(X_test, gp_16percent[j], gp_84percent[j], color=colours[j], alpha=0.2)

            ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
            ax0.grid()

            ax1.set_ylabel("Residuals (km/s)")
            for j in range(4):
                ax1.plot(r, residuals[j], 'o-', color=colours[j], alpha=0.45)
            ax1.grid()

            plt.subplots_adjust(hspace=0.05)
            fig1.savefig(fileloc+"ls_fits/"+g+".png", dpi=300, bbox_inches="tight")
            plt.close()


    """
    DTW on GP residuals.
    """
    if do_DTW:
        if testing:
            print("\nRunning DTW on GP residuals...")
        
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
        mats_dir = [ "data/", "MOND/", "LCDM/" ]
        
        # DTW!
        for j in range(3):
            path, cost_mat = dtw(dist_mats[j])
            x_path, y_path = zip(*path)
            cost = cost_mat[ len(r)-1, len(r)-1 ]
            dtw_cost[j].append(cost)
            norm_cost[j].append(cost / (2 * len(X)))

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

                plt.savefig(fileloc+"dtw/matrix_"+mats_dir[j]+g+".png", dpi=300, bbox_inches="tight")
                plt.close('all')

                # Visualize DTW alignment.
                plt.title("DTW alignment: "+g)

                diff = abs(max(res_Vbar) - min(residuals[j+1]))
                for x_i, y_j in path:
                    plt.plot([x_i, y_j], [residuals[j+1][x_i] + diff, res_Vbar[y_j] - diff], c="C7", alpha=0.4)
                plt.plot(np.arange(len(r)), np.array(residuals[j+1]) + diff, c=colours[j+1], label=v_comps[j+1])
                plt.plot(np.arange(len(r)), np.array(res_Vbar) - diff, c="red", label="Vbar")
                plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(cost))
                plt.plot([], [], c='w', label="Normalized cost = {:.4f}".format(cost/(len(r)*2)))

                plt.axis("off")
                plt.legend(bbox_to_anchor=(1,1))
                plt.savefig(fileloc+"dtw/vis_"+mats_dir[j]+g+".png", dpi=300, bbox_inches="tight")
                plt.close('all')
    
    """
    Code for PCHIP on GP residuals.
    """
    if do_correlations:
        if testing:
            print("Computing correlation coefficients...")

        # Interpolate the residuals with cubic Hermite spline splines.
        v_d0, v_d1, v_d2 = [], [], []
        for v_comp in residuals:
            v_d0.append(interpolate.pchip_interpolate(r, v_comp, X_test))
            v_d1.append(interpolate.pchip_interpolate(r, v_comp, X_test, der=1))
            v_d2.append(interpolate.pchip_interpolate(r, v_comp, X_test, der=2))

        # # Apply SG filter to interpolated residuals with window size = 0.2*Reff and extract first and second derivatives.
        # d0_sg, d1_sg, d2_sg = [], [], []
        # for v_comp0 in v_d0:
        #     d0_sg.append(signal.savgol_filter(v_comp0, 50, 2))
        # for v_comp1 in v_d1:
        #     d1_sg.append(signal.savgol_filter(v_comp1, 50, 2))
        # for v_comp2 in v_d2:
        #     d2_sg.append(signal.savgol_filter(v_comp2, 50, 2))
        
        res_fits = [ v_d0, v_d1, v_d2 ]


        """
        ---------------------------------------------------
        Correlation plots using sphers of increasing radius
        ---------------------------------------------------
        """
        if testing:
            print("Correlating coefficients by max radii...")

        # Correlate Vobs and Vbar (d0, d1, d2) as a function of (maximum) radius, i.e. spheres of increasing r.
        # correlations_r = rad_corr arrays with [ data, MOND, LCDM ], so 3 Vobs x 3 derivatives x 2 correlations each,
        # where rad_corr = [ [[Spearman d0], [Pearson d0]], [[S d1], [P d1]], [[S d2], [P d2]] ].
        correlations_r = []
        for i in range(1, 4):
            rad_corr = [ [[], []], [[], []], [[], []] ]
            for k in range(3):
                for j in range(10, len(X_test)):
                    rad_corr[k][0].append(stats.spearmanr(res_fits[k][0][:j], res_fits[k][i][:j])[0])
                    rad_corr[k][1].append(stats.pearsonr(res_fits[k][0][:j], res_fits[k][i][:j])[0])
            correlations_r.append(rad_corr)


        """
        1. Plot the residual splines and 1st + 2nd derivatives for Vbar and all Vobs, and their
        Spearman/Pearson correlation alongside the galaxy's (average) baryonic ratio in 3 separate subplots.
        """
        subdir = "correlations/radii/sep_subplots/"

        color_bar = "orange"
        deriv_dir = [ "d0/", "d1/", "d2/" ]
        # correlations = [ [], [], [] ]   # correlations = list of component lists: [ [S d0, P d0], [S d1, P d1], [S d2, P d2] ].

        # if make_plots:
        #     # Plot corrletaions as 1 main plot + 3 subplots, each with a different corresponding Vbar/Vobs.
        #     for der in range(3):
        #         fig0, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
        #         fig0.set_size_inches(18.5, 10.5)
        #         axes = [ ax2, ax3, ax4 ]
        #         ax1.set_title("Residuals correlation: "+g)
        #         ax1.set_ylabel("Velocities (km/s)")

        #         for j in range(4):
        #             ax1.scatter(r, residuals[j], color=colours[j], alpha=0.3)
        #             ax1.plot(X_test, res_fits[der][j], color=colours[j], label=v_comps[j])
        #         ax1.legend(bbox_to_anchor=(1.35,1))

        #         for j in range(3):
        #             # Compute baryonic dominance, i.e. average Vbar/Vobs from centre to some max radius.
        #             bar_ratio = []
        #             for rd in range(len(X_test)):
        #                 bar_ratio.append(sum(gp_predictions[0][:rd]/gp_predictions[j+1][:rd]) / (rd+1))

        #             # Compute correlation between rs or rp and the baryonic ratio, using rs for rs-bar and rp for rp-bar.
        #             correlations_comp = []
        #             correlations_comp.append(stats.spearmanr(correlations_r[j][der][0], bar_ratio[10:])[0])
        #             correlations_comp.append(stats.pearsonr(correlations_r[j][der][1], bar_ratio[10:])[0])
        #             correlations[der].append(correlations_comp)

        #             # Plot correlations and Vbar/Vobs.
        #             if j > 0:
        #                 axes[j].set_xlabel(r'Normalised radius ($\times R_{eff}$)')
        #             axes[j].set_ylabel("Correlation: "+v_comps[j+1])
        #             axes[j].plot(X_test[10:], correlations_r[j][der][0], color=colours[j+1], label=r"Spearman $\rho$")
        #             axes[j].plot(X_test[10:], correlations_r[j][der][1], ':', color=colours[j+1], label=r"Pearson $\rho$")
        #             axes[j].tick_params(axis='y', labelcolor=colours[j+1])

        #             ax5 = axes[j].twinx()
        #             ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
        #             ax5.plot(X_test[10:], bar_ratio[10:], '--', color=color_bar, label="Vbar/Vobs")
        #             ax5.tick_params(axis='y', labelcolor=color_bar)

        #             axes[j].plot([], [], ' ', label=r"$\rho_s=$"+str(round(correlations_comp[0], 3))+r", $\rho_p=$"+str(round(correlations_comp[1], 3)))
        #             axes[j].legend(bbox_to_anchor=(1.5, 1))

        #         plt.subplots_adjust(hspace=0.05, wspace=0.6)
        #         fig0.savefig(fileloc+subdir+deriv_dir[der]+g+".png", dpi=300, bbox_inches="tight")
        #         plt.close()

        # correlations_ALL.append(correlations)


        """
        2. Plot GP fits, residuals (+ PCHIP) and correlations.
        """
        subdir = "correlations/radii/"

        # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
        bar_ratio = []
        for rd in range(len(X_test)):
            bar_ratio.append(sum(gp_predictions[0][:rd]/gp_predictions[1][:rd]) / (rd+1))

        if make_plots:
            # Plot corrletaions as 1 main plot + 1 subplot, using only Vobs from data for Vbar/Vobs.
            der_axis = [ "Residuals (km/s)", "1st derivative", "2nd derivative" ]
            for der in range(3):
                fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
                fig1.set_size_inches(7, 7)
                ax0.set_title("Residuals correlation: "+g)
                ax0.set_ylabel("Velocities (km/s)")

                for j in range(4):
                    # Scatter plot for data/mock data points.
                    ax0.scatter(X, Y[j], color=colours[j], alpha=0.3)
                    # Plot mean prediction from GP.
                    ax0.plot(X_test, gp_predictions[j], color=colours[j], label=v_comps[j])
                    # Fill in 1-sigma (68%) confidence band of GP fit.
                    ax0.fill_between(X_test, gp_16percent[j], gp_84percent[j], color=colours[j], alpha=0.2)

                ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
                ax0.grid()

                ax1.set_ylabel(der_axis[der])
                for j in range(4):
                    if der == 0:
                        ax1.scatter(r, residuals[j], color=colours[j], alpha=0.3)
                    ax1.plot(X_test, res_fits[der][j], color=colours[j], label=v_comps[j])

                ax1.grid()

                ax2.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
                ax2.set_ylabel("Correlations")
                
                vel_comps = [ "Data", "MOND", r"$\Lambda$CDM" ]

                for j in range(3):
                    # Plot correlations and Vbar/Vobs.
                    ax2.plot(X_test[10:], correlations_r[j][der][0], color=colours[j+1], label=vel_comps[j]+r": Spearman $\rho$")
                    ax2.plot(X_test[10:], correlations_r[j][der][1], ':', color=colours[j+1], label=vel_comps[j]+r": Pearson $\rho$")
                    # ax2.plot([], [], ' ', label=vel_comps[j]+r": $\rho_s=$"+str(round(correlations[j][der][0], 3))+r", $\rho_p=$"+str(round(correlations[j][der][1], 3)))
                    ax2.plot([], [], ' ', label=vel_comps[j]+r": $\rho_s=$"+str(round(stats.spearmanr(correlations_r[j][der][0], bar_ratio[10:])[0], 3))+r", $\rho_p=$"+str(round(stats.pearsonr(correlations_r[j][der][1], bar_ratio[10:])[0], 3)))

                ax5 = ax2.twinx()
                ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
                ax5.plot(X_test[10:], bar_ratio[10:], '--', color=color_bar, label="Vbar/Vobs")
                ax5.tick_params(axis='y', labelcolor=color_bar)
                
                ax2.legend(bbox_to_anchor=(1.64, 1.3))
                ax2.grid()

                plt.subplots_adjust(hspace=0.05)
                fig1.savefig(fileloc+subdir+deriv_dir[der]+g+".png", dpi=300, bbox_inches="tight")
                plt.close()


        """
        -----------------------------------------------------------------------
        Correlation plots using windows of length max{1 * Reff, 5 data points}.
        (Only for galaxies with Rmax > 1 * Reff)
        -----------------------------------------------------------------------
        """
        if testing:
            print("Correlating coefficients by moving window...")

        if len(X_test) > 100:
            # Correlate Vobs and Vbar (d0, d1, d2) along a moving window of length 1 * Reff.
            # correlations_w = win_corr arrays with [ data, MOND, LCDM ], so 3 Vobs x 3 derivatives x 2 correlations each,
            # where win_corr = [ [[Spearman d0], [Pearson d0]], [[S d1], [P d1]], [[S d2], [P d2]] ].
            wmax = len(X_test) - 50
            correlations_w = []
            
            for vc in range(1, 4):
                win_corr = [ [[], []], [[], []], [[], []] ]
                for der in range(3):
                    for j in range(50, wmax):

                        idx = (np.abs(X - X_test[j])).argmin()
                        X_jmin, X_jmax = math.ceil(X[max(0, idx-2)] * 100), math.ceil(X[min(len(X)-1, idx+2)] * 100)
                        
                        if X_jmax - X_jmin > 100:
                            win_corr[der][0].append(stats.spearmanr(res_fits[der][0][X_jmin:X_jmax], res_fits[der][vc][X_jmin:X_jmax])[0])
                            win_corr[der][1].append(stats.pearsonr(res_fits[der][0][X_jmin:X_jmax], res_fits[der][vc][X_jmin:X_jmax])[0])
                        else:
                            jmin, jmax = j - 50, j + 50
                            win_corr[der][0].append(stats.spearmanr(res_fits[der][0][jmin:jmax], res_fits[der][vc][jmin:jmax])[0])
                            win_corr[der][1].append(stats.pearsonr(res_fits[der][0][jmin:jmax], res_fits[der][vc][jmin:jmax])[0])

                    # Apply SG filter to smooth out correlation curves for better visualisation.
                    win_corr[der][0] = signal.savgol_filter(win_corr[der][0], 50, 2)
                    win_corr[der][1] = signal.savgol_filter(win_corr[der][1], 50, 2)

                correlations_w.append(win_corr)

            # Compute average baryonic dominance (using Vobs from SPARC data) in moving window.
            wbar_ratio = []
            for j in range(50, wmax):
                wbar_ratio.append( sum( gp_predictions[0][j-50:j+50] / gp_predictions[1][j-50:j+50] ) / 101 )


            """
            1. Plot GP fits, residuals (+ PCHIP) and correlations.
            """
            subdir = "correlations/window/"

            if make_plots:
                # Plot corrletaions as 1 main plot + 1 subplot, using only Vobs from data for Vbar/Vobs.
                for der in range(3):
                    fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
                    fig1.set_size_inches(7, 7)
                    ax0.set_title("Moving window correlation: "+g)
                    ax0.set_ylabel("Velocities (km/s)")

                    for j in range(4):
                        # Scatter plot for data/mock data points.
                        ax0.scatter(X, Y[j], color=colours[j], alpha=0.3)
                        # Plot mean prediction from GP.
                        ax0.plot(X_test, gp_predictions[j], color=colours[j], label=v_comps[j])
                        # Fill in 1-sigma (68%) confidence band of GP fit.
                        ax0.fill_between(X_test, gp_16percent[j], gp_84percent[j], color=colours[j], alpha=0.2)

                    ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
                    ax0.grid()

                    ax1.set_ylabel(der_axis[der])
                    for j in range(4):
                        if der == 0:
                            ax1.scatter(r, residuals[j], color=colours[j], alpha=0.3)
                        ax1.plot(X_test, res_fits[der][j], color=colours[j], label=v_comps[j])

                    ax1.grid()

                    ax2.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
                    ax2.set_ylabel("Correlations")
                    
                    vel_comps = [ "Data", "MOND", r"$\Lambda$CDM" ]

                    for j in range(3):
                        # Plot correlations and Vbar/Vobs.
                        ax2.plot(X_test[50:wmax], correlations_w[j][der][0], color=colours[j+1], label=vel_comps[j]+r": Spearman $\rho$")
                        ax2.plot(X_test[50:wmax], correlations_w[j][der][1], ':', color=colours[j+1], label=vel_comps[j]+r": Pearson $\rho$")
                        ax2.plot([], [], ' ', label=vel_comps[j]+r": $\rho_s=$"+str(round(stats.spearmanr(correlations_w[j][der][0], wbar_ratio)[0], 3))+r", $\rho_p=$"+str(round(stats.pearsonr(correlations_w[j][der][1], wbar_ratio)[0], 3)))

                    ax5 = ax2.twinx()
                    ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
                    ax5.plot(X_test[50:wmax], wbar_ratio, '--', color=color_bar, label="Vbar/Vobs")
                    ax5.tick_params(axis='y', labelcolor=color_bar)
                    
                    ax2.legend(bbox_to_anchor=(1.64, 1.3))
                    ax2.grid()

                    plt.subplots_adjust(hspace=0.05)
                    fig1.savefig(fileloc+subdir+deriv_dir[der]+g+".png", dpi=300, bbox_inches="tight")
                    plt.close()


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
        choices=["median", "feasible", "value", "uniform", "sample"],
    )
    parser.add_argument("--no-cholesky", dest="use_cholesky", action="store_false")
    parser.add_argument("--testing", default=testing, type=bool)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    return args

if __name__ == "__main__":
    args = GP_args()

    # Get galaxy data from table1.
    file = "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt"
    
    # Get v_DM from Fedir's LCDM abundance matching (V_LCDM.txt).
    DMtable_str = []
    v_DM = []
    with open('/mnt/users/koe/V_LCDM.txt') as f_DM:
        list_DM = f_DM.read().splitlines()
        for line in list_DM:
            DMtable_str.append(line.split(", "))

    for line in DMtable_str:
        del line[0]
        v_DM.append([float(num) for num in line])

    SPARC_c = [ "Galaxy", "T", "D", "e_D", "f_D", "Inc",
            "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
                "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
    table = pd.read_fwf(file, skiprows=98, names=SPARC_c)

    columns = [ "Rad", "Vobs", "errV", "Vgas",
                "Vdisk", "Vbul", "SBdisk", "SBbul" ]

    # Define constants
    G = 4.300e-6    # Gravitational constant G (kpc (km/s)^2 solarM^(-1))
    pdisk = 0.5
    pbul = 0.7
    a0 = 1.2e-10 / 3.24e-14     # Scale acceleration for MOND in pc/yr^2

    # Calculate baryonic matter from data of individual galaxies.
    def Vbar(arr):
        v = np.sqrt( arr["Vgas"]**2
                    + (arr["Vdisk"]**2 * pdisk)
                    + (arr["Vbul"]**2 * pbul) )
        return v
    
    def MOND_Vobs(arr, a0=a0):
        # Quadratic solution from MOND simple interpolating function.
        acc = Vbar(arr)**2 / r
        y = acc / a0
        nu = 1 + np.sqrt((1 + 4/y))
        nu /= 2
        return np.sqrt(acc * nu * r)

    galaxy_count = len(table["Galaxy"])
    skips = 0
    if testing:
        if test_multiple:
            galaxy_count = 13   # First 2 galaxies.
        else:
            galaxy_count = 1    # Test on NGC 6946.
    bulged_count = 0
    xbulge_count = 0
    
    galaxy, correlations_ALL, ls_ALL = [], [], []
    dtw_cost = [ [], [], [] ]
    norm_cost = [ [], [], [] ]

    for i in tqdm(range(galaxy_count), desc="SPARC galaxies"):
    # for i in range(galaxy_count):
        if testing and not test_multiple:
            # i = 91  # NGC6946
            i = 32  # NGC0024

        g = table["Galaxy"][i]
            
        # if g=="D512-2" or g=="D564-8" or g=="D631-7" or g=="NGC4138" or g=="NGC5907" or g=="UGC06818":
        #     skips += 1
        #     continue

        """
        Plotting galaxy rotation curves directly from data with variables:
        Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
        """
        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
        r = data["Rad"]
        # r = data["Rad"] / table["Rdisk"][i] # Normalised radius (Rdisk = scale length of stellar disk).

        # Reject galaxies with less than 20 data points.
        if len(r) < 20:
            continue

        Rmax = max(np.diff(r)) # Maximum difference in r of data points (to be used as length scale for GP kernel)
        # Rmin = min(np.diff(r)) # Minimum difference in r (used for upsampling the data)

        # Normalise velocities by Vmax = max(Vobs) from SPARC data.
        v_LCDM = np.sqrt(Vbar(data)**2 + np.array(v_DM[i])**2)
        v_components = np.array([ Vbar(data), data["Vobs"], MOND_Vobs(data), v_LCDM ])
        # Vmax = max(v_components[1])
        # v_components /= Vmax

        if bulged:
            bulged_count += 1
        else:
            xbulge_count += 1
        
        # rad_count = math.ceil((max(r)-min(r))*100)
        rad = np.linspace(min(r), max(r), 100)

        X, X_test = r.to_numpy(), rad
        
        # print("")
        # print("==================================")
        # print("Analyzing galaxy "+g+" ("+str(i+1)+"/175)")
        # print("==================================")

        gp_predictions = [ [], [], [], [] ]
        gp_16percent = [ [], [], [], [] ]
        gp_84percent = [ [], [], [], [] ]

        ls_dict = np.load("/mnt/users/koe/gp_fits/ls_dict.npy", allow_pickle=True).item()
        ls = ls_dict[g]

        main(args, g, X, v_components, X_test, ls)

        # print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
        jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem.

        # Save GP fits to CSV for later use (for incorporating uncertainties/errors).
        # One array per galaxy, each containing 13 lists:
        # radii, mean (x4), 16th percentile (x4), 84th percentile (x4).
        gp_fits = np.array([rad, *gp_predictions, *gp_16percent, *gp_84percent])
        np.save("/mnt/users/koe/gp_fits/"+g, gp_fits)
        # print("\nGP results successfully saved as /mnt/users/koe/gp_fits/"+g+".npy.")

        galaxy.append(g)
    
    # if not testing:
    #     np.save("/mnt/users/koe/gp_fits/galaxy", galaxy)
    #     print("\nList of analyzed galaxies now saved as /mnt/users/koe/gp_fits/galaxy.npy.")
    #     np.save("/mnt/users/koe/gp_fits/ls_ALL", ls_ALL)
    #     print("\nList of length scales now saved as /mnt/users/koe/gp_fits/ls_ALL.npy.")

    if make_plots and do_DTW:
        """
        Plot histogram of normalized DTW costs (in ascending order of costs for data).
        """
        # Rearrange galaxies into ascending order in norm_cost.
        sort_args = np.argsort(norm_cost[0])
        norm_cost = np.array(norm_cost)
        costs_sorted = []   # [ [data], [MOND], [LCDM] ]
        for j in range(3):
            costs_sorted.append(norm_cost[j][sort_args])

        # Plot histogram of normalized DTW alignment costs of all galaxies.
        plt.title("Normalized DTW alignment costs")
        hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
        colours = [ 'k', 'mediumblue', 'tab:green' ]

        for j in range(3):
            mean_norm = np.mean(norm_cost[j])
            plt.bar(galaxy, costs_sorted[j], color=colours[j], alpha=0.3, label=hist_labels[j])
            plt.axhline(y=mean_norm, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_norm))
        
        plt.legend()
        plt.xticks([])
        plt.savefig(fileloc+"dtw/histo1.png", dpi=300, bbox_inches="tight")
        plt.close()
        

        """
        Plot histogram of differences in normalized DTW costs (mock - data, in ascending order of costs for MOND - data).
        """
        # Rearrange galaxies into ascending order in cost_diff(MOND).
        cost_diff = np.array([norm_cost[1] - norm_cost[0], norm_cost[2] - norm_cost[0]])
        sort_args = np.argsort(cost_diff[0])
        diff_sorted = []   # [ [MOND], [LCDM] ]
        for j in range(2):
            diff_sorted.append(cost_diff[j][sort_args])

        # Plot histogram of normalized DTW alignment costs of all galaxies.
        plt.title("Normalised cost differences (mock - real data)")
        hist_labels = [ "MOND", r"$\Lambda$CDM" ]
        colours = [ 'mediumblue', 'red' ]

        for j in range(2):
            mean_diff = np.mean(cost_diff[j])
            plt.bar(galaxy, diff_sorted[j], color=colours[j], alpha=0.4, label=hist_labels[j])
            plt.axhline(y=mean_diff, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_diff))
        
        plt.legend()
        plt.xticks([])
        plt.savefig(fileloc+"dtw/histo2.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("\nMax memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
