#!/usr/bin/env python
"""
Generate toy models of galaxy RC using arctan curves + gaussian features,
then apply same analysis of correlation coefficients + DTW on residuals
to better understand the effect/restriction of feature sizes and noise/uncertainties.
"""
import numpy as np
from scipy import stats, interpolate
import matplotlib.pyplot as plt
# from tqdm import tqdm

from utils_analysis.dtw_utils import dtw

# Switches for running different parts of the analysis.
do_DTW      = True
corr_radii  = True
corr_window = True

fileloc = "/mnt/users/koe/plots/toy_model/"

# Generate Gaussian featurs.
bump_loc    = 5.0
bump_FWHM   = 0.5
bump_size   = 0.1
noise       = 0.02
bar_ratio   = 0.7

rad = np.linspace(0., 10., 100)

Vobs = np.arctan(rad)
Vobs /= max(Vobs)
Vbar = Vobs * bar_ratio
Vraw = np.array([ Vobs, Vbar ])

bump_FWHM /= 2.0
bump = bump_size * bump_FWHM * np.sqrt(2*np.pi) * stats.norm.pdf(rad, bump_loc, bump_FWHM)
features = np.array([ 1.0 * bump, 1.0 * bar_ratio * bump ])
velocities = Vraw + features

v_werr = np.random.normal(velocities, noise)
residuals = v_werr - (velocities - features)


# Interpolate the residuals with cubic Hermite spline splines.
v_d0, v_d1, v_d2 = [], [], []
for v_comp in residuals:
    v_d0.append(interpolate.pchip_interpolate(rad, v_comp, rad))
    v_d1.append(interpolate.pchip_interpolate(rad, v_comp, rad, der=1))
    v_d2.append(interpolate.pchip_interpolate(rad, v_comp, rad, der=2))
res_fits = [ v_d0, v_d1, v_d2 ]


"""
DTW on GP residuals.
"""
if do_DTW:
    print("Warping time dynamically... or something like that...")        
    # Construct distance matrices.
    dist_data = np.zeros((len(rad), len(rad)))
    for n in range(len(rad)):
        for m in range(len(rad)):
            dist_data[n, m] = abs(residuals[0][n] - residuals[1][m])

    # DTW!
    path, cost_mat = dtw(dist_data)
    x_path, y_path = zip(*path)
    cost = cost_mat[ len(rad)-1, len(rad)-1 ]

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

    plt.savefig(fileloc+"dtw_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Visualize DTW alignment.
    plt.title("DTW alignment: Toy model")

    diff = abs(max(residuals[1]) - min(residuals[0]))
    for x_i, y_j in path:
        plt.plot([x_i, y_j], [residuals[0][x_i] + diff, residuals[1][y_j] - diff], c="C7", alpha=0.4)
    plt.plot(np.arange(len(rad)), np.array(residuals[0]) + diff, c='k', label="Vobs")
    plt.plot(np.arange(len(rad)), np.array(residuals[1]) - diff, c="red", label="Vbar")
    plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(cost))
    plt.plot([], [], c='w', label="Normalized cost = {:.4f}".format(cost/(len(rad)*2)))

    plt.axis("off")
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(fileloc+"dtw_alignment.png", dpi=300, bbox_inches="tight")
    plt.close()


"""
---------------------------------------------------
Correlation plots using sphers of increasing radius
---------------------------------------------------
"""
if corr_radii:
    print("Computing correlation coefficients with increasing radii...")
    # Correlate Vobs and Vbar (d0, d1, d2) as a function of (maximum) radius, i.e. spheres of increasing r.
    rad_corr = [ [[], []], [[], []], [[], []] ]
    for der in range(1):
        for j in range(10, len(rad)):
            rad_corr[der][0].append(stats.spearmanr(res_fits[der][0][:j], res_fits[der][1][:j])[0])
            rad_corr[der][1].append(stats.pearsonr(res_fits[der][0][:j], res_fits[der][1][:j])[0])

    # # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
    # bar_ratio = []
    # for rd in tqdm(range(len(rad))):
    #     bar_ratio.append(sum(Vraw[1][:rd]/Vraw[0][:rd]) / (rd+1))


    """
    Plot GP fits, residuals (+ PCHIP) and correlations.
    """
    colors = [ 'k', 'tab:red' ]
    labels = [ 'Vobs', 'Vbar' ]
    deriv_dir = [ "d0", "d1", "d2" ]
    # color_bar = "orange"

    for der in range(1):
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
        fig.set_size_inches(7, 7)

        ax0.set_title("Correlation by increasing radii: Toy model")
        ax0.set_ylabel("Normalised velocities")
        ax2.set_xlabel("Radii (kpc)")

        for i in range(2):
            ax0.errorbar(rad, v_werr[i], noise, color=colors[i], alpha=0.3, capsize=3, fmt="o", ls="none")
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
        # ax2.plot([], [], ' ', label=r": $\rho_s=$"+str(round(stats.spearmanr(rad_corr[der][0], bar_ratio[10:])[0], 3))+r", $\rho_p=$"+str(round(stats.pearsonr(rad_corr[der][1], bar_ratio[10:])[0], 3)))

        # ax5 = ax2.twinx()
        # ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
        # ax5.plot(rad[10:], bar_ratio[10:], '--', color=color_bar, label="Vbar/Vobs")
        # ax5.tick_params(axis='y', labelcolor=color_bar)

        ax2.legend(loc="upper left", bbox_to_anchor=(1,1))
        ax2.grid()

        plt.subplots_adjust(hspace=0.05)
        fig.savefig(fileloc+"radii_"+deriv_dir[der]+".png", dpi=300, bbox_inches="tight")
        plt.close()


"""
-----------------------------------------------------------------------
Correlation plots using windows of length max{1 * Reff, 5 data points}.
-----------------------------------------------------------------------
"""
if corr_window:
    print("Computing correlation coefficients with moving window...")
    # Correlate Vobs and Vbar (d0, d1, d2) along a moving window of length 1 * kpc.
    wmax = len(rad) - 5
    win_corr = [ [[], []], [[], []], [[], []] ]
    for der in range(1):
        for j in range(5, wmax):
            jmin, jmax = j - 5, j + 5
            win_corr[der][0].append(stats.spearmanr(res_fits[der][0][jmin:jmax], res_fits[der][1][jmin:jmax])[0])
            win_corr[der][1].append(stats.pearsonr(res_fits[der][0][jmin:jmax], res_fits[der][1][jmin:jmax])[0])

        # Apply SG filter to smooth out correlation curves for better visualisation.
        # win_corr[der][0] = signal.savgol_filter(win_corr[der][0], 5, 2)
        # win_corr[der][1] = signal.savgol_filter(win_corr[der][1], 5, 2)
        

    # Plot corrletaions as 1 main plot + 1 subplot, using only Vobs from data for Vbar/Vobs.
    for der in range(1):
        fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
        fig1.set_size_inches(7, 7)
        ax0.set_title("Moving window correlation: Toy model")
        ax0.set_ylabel("Normalised velocities")
        ax2.set_xlabel("Radii (kpc)")
        for i in range(2):
            ax0.errorbar(rad, v_werr[i], noise, color=colors[i], alpha=0.3, capsize=3, fmt="o", ls="none")
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
        # ax2.plot([], [], ' ', label=r": $\rho_s=$"+str(round(stats.spearmanr(correlations_w[j][der][0], wbar_ratio)[0], 3))+r", $\rho_p=$"+str(round(stats.pearsonr(correlations_w[j][der][1], wbar_ratio)[0], 3)))

        # ax5 = ax2.twinx()
        # ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
        # ax5.plot(rad[50:wmax], wbar_ratio, '--', color=color_bar, label="Vbar/Vobs")
        # ax5.tick_params(axis='y', labelcolor=color_bar)
        
        ax2.legend(loc="upper left", bbox_to_anchor=(1,1))
        ax2.grid()

        plt.subplots_adjust(hspace=0.05)
        fig1.savefig(fileloc+"window_"+deriv_dir[der]+".png", dpi=300, bbox_inches="tight")
        plt.close()
