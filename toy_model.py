#!/usr/bin/env python
"""
Generate toy models of galaxy RC using arctan curves + gaussian features,
then apply same analysis of correlation coefficients + DTW on residuals
to better understand the effect/restriction of feature sizes and noise/uncertainties.
"""
import math
import numpy as np
from scipy import stats, interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils_analysis.dtw_utils import dtw
from utils_analysis.Vobs_fits import MOND_vsq


# Switches for running different parts of the analysis.
do_DTW      = True
corr_radii  = True
corr_window = True
make_plots  = True

fileloc = "/mnt/users/koe/plots/toy_model/"
colors = [ 'k', 'tab:red' ]
labels = [ 'Vobs', 'Vbar' ]
deriv_dir = [ "d0", "d1", "d2" ]
color_bar = "orange"

bump_size   = 20.0   # Defined in terms of percentage of max(Vbar)
bump_loc    = 5.0
bump_FWHM   = 0.5
bump_sigma  = bump_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

rad = np.linspace(10., 0., 100, endpoint=False)[::-1]   # Defined this way to exclude the starting point r=0.
noise_arr = np.linspace(0.0, bump_size/2, 101, endpoint=True)
win_spearmans, win_pearsons = [], []
dtw_costs = []


for noise in tqdm(noise_arr):
    # Generate Vbar with Gaussian bump.
    Vbar_raw = np.arctan(rad)
    Vbar_raw *= 100.0   # Multiplication factor for getting sensible MOND RC.

    bump = bump_size * bump_sigma * np.sqrt(2*np.pi) * stats.norm.pdf(rad, bump_loc, bump_sigma)
    Vbar = Vbar_raw + bump
    bump /= 100.0
    
    # Generate Vobs from Vbar using MOND function.
    vel_MOND = np.sqrt(MOND_vsq(rad, Vbar**2))
    Vmax = max(vel_MOND)
    velocities = np.array([ vel_MOND, Vbar ])

    # Scatter RCs with Gaussian noise.
    v_werr = np.random.normal(velocities, noise) / Vmax
    
    # Generate perfect GP fit using original functions (arctan w/o bump) and calculate residuals.
    Vobs_raw = np.sqrt(MOND_vsq(rad, Vbar_raw**2))
    Vraw = np.array([ Vobs_raw, Vbar_raw ]) / Vmax
    residuals = v_werr - Vraw

    # Interpolate the residuals with cubic Hermite spline splines.
    v_d0, v_d1, v_d2 = [], [], []
    for v_comp in residuals:
        v_d0.append(interpolate.pchip_interpolate(rad, v_comp, rad))
        # v_d1.append(interpolate.pchip_interpolate(rad, v_comp, rad, der=1))
        # v_d2.append(interpolate.pchip_interpolate(rad, v_comp, rad, der=2))
    # res_fits = [ v_d0, v_d1, v_d2 ]
    res_fits = [ v_d0 ]


    """
    DTW on GP residuals.
    """
    if do_DTW:
        # print("Warping time dynamically... or something like that...")        
        # Construct distance matrices.
        dist_data = np.zeros((len(rad), len(rad)))
        for n in range(len(rad)):
            for m in range(len(rad)):
                # Define new distance matrix construction s.t. cost = 0 for MOND (w/o noise).
                MOND_res = (vel_MOND - Vobs_raw) / Vmax
                dist_data[n, m] = abs(residuals[0][n] - MOND_res[m])
                # dist_data[n, m] = abs(residuals[0][n] - residuals[1][m])

        # DTW!
        path, cost_mat = dtw(dist_data)
        x_path, y_path = zip(*path)
        cost = cost_mat[ len(rad)-1, len(rad)-1 ]
        dtw_costs.append(cost)

        if make_plots and noise in noise_arr[::10]:
            # Plot distance matrix and cost matrix with optimal path.
            plt.title("Dynamic time warping: Toy model")
            plt.axis('off')

            plt.subplot(121)
            plt.title("Distance matrix")
            plt.imshow(dist_data, cmap=plt.cm.binary, interpolation="nearest", origin="lower")

            plt.subplot(122)
            plt.title("Cost matrix")
            plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
            plt.plot(x_path, y_path)

            plt.savefig(fileloc+f"dtw_matrix/ratio={round(noise/bump_size, 2)}.png", dpi=300, bbox_inches="tight")
            plt.close()

            # Visualize DTW alignment.
            plt.title("DTW alignment: Toy model")

            diff = abs(max(MOND_res) - min(residuals[0]))
            for x_i, y_j in path:
                plt.plot([x_i, y_j], [residuals[0][x_i] + diff, MOND_res[y_j] - diff], c="C7", alpha=0.4)
            plt.plot(np.arange(len(rad)), np.array(residuals[0]) + diff, c='k', label="Vobs")
            plt.plot(np.arange(len(rad)), np.array(MOND_res) - diff, c="red", label="Vbar")
            plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(cost))
            plt.plot([], [], c='w', label="Normalized cost = {:.4f}".format(cost/(len(rad)*2)))

            plt.axis("off")
            plt.legend(bbox_to_anchor=(1,1))
            plt.savefig(fileloc+f"dtw_alignment/ratio={round(noise/bump_size, 2)}.png", dpi=300, bbox_inches="tight")
            plt.close()


    """
    ---------------------------------------------------
    Correlation plots using sphers of increasing radius
    ---------------------------------------------------
    """
    if corr_radii and noise != 0.0:
        # print("Computing correlation coefficients with increasing radii...")
        # Correlate Vobs and Vbar (d0, d1, d2) as a function of (maximum) radius, i.e. spheres of increasing r.
        rad_corr = [ [[], []], [[], []], [[], []] ]
        for der in range(1):
            for j in range(10, len(rad)):
                rad_corr[der][0].append(stats.spearmanr(res_fits[der][0][:j], res_fits[der][1][:j])[0])
                rad_corr[der][1].append(stats.pearsonr(res_fits[der][0][:j], res_fits[der][1][:j])[0])

        # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
        bar_ratio = []
        for rd in range(len(rad)):
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
        wmax = len(rad) - 5
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


bump_ratio = noise_arr / bump_size

if do_DTW:
    plt.title("DTW alignment costs")
    plt.ylabel("DTW cost")
    plt.xlabel("Noise / feature height")
    plt.plot(bump_ratio, dtw_costs, color='k')

    plt.savefig(fileloc+"dtwVnoise.png", dpi=300, bbox_inches="tight")
    plt.close()

if corr_window:
    plt.title("Correlation coefficients at peak of feature")
    plt.ylabel("Correlation coefficients")
    plt.xlabel("Noise / feature height")
    plt.plot(bump_ratio[1::], win_spearmans, color='mediumblue', label=r"Spearman $\rho$")
    plt.plot(bump_ratio[1::], win_pearsons, '--', color='mediumblue', label=r"Pearson $\rho$")

    plt.legend()
    plt.savefig(fileloc+"corrVnoise.png", dpi=300, bbox_inches="tight")
    plt.close()
