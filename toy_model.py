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

from resource import getrusage, RUSAGE_SELF
# from tqdm import tqdm

from utils_analysis.dtw_utils import dtw
from utils_analysis.Vobs_fits import MOND_vsq

memory_usage = []


# Switches for running different parts of the analysis.
do_DTW      = True
corr_radii  = False     # Code to be fixed for noise iterations.
corr_window = False     # Code to be fixed for noise iterations.
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
num_rad = len(rad)

noise_arr = np.linspace(0.0, bump_size, 201, endpoint=True)
num_noise = len(noise_arr)
num_iterations = 100     # Iterations per noise level (for smoothing out DTW costs and correlations in final plots).
max_index = num_iterations - 1  # Useful index for later plots.

# Initialize arrays for summary plots.
win_spearmans = np.zeros((num_iterations, num_noise))
win_pearsons = np.copy(win_spearmans)
dtw_costs, Xft_costs = np.copy(win_spearmans), np.copy(win_spearmans)
dtw_window, Xft_window = np.copy(win_spearmans), np.copy(win_spearmans)


for i in range(num_noise):
    if i%10 == 0:
        print(f"Running iteration {i}/{num_noise}...")
    
    noise = noise_arr[i]

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
    velocities = np.array( [velocities] * num_iterations )

    # Scatter RCs with Gaussian noise.
    v_werr = np.random.normal(velocities, noise) / Vmax
    
    # Generate perfect GP fit using original functions (arctan w/o bump) and calculate residuals.
    Vobs_raw = np.sqrt(MOND_vsq(rad, Vbar_raw**2))
    Vraw = np.array([ Vobs_raw, Vbar_raw ]) / Vmax
    Vraw = np.array( [Vraw] * num_iterations )
    residuals = v_werr - Vraw

    # Vobs residuals due to pure noise, i.e. smooth RC without feature.
    Vobs_raw = np.array( [Vobs_raw] * num_iterations )
    res_Xft = np.random.normal(Vobs_raw, noise) - Vobs_raw
    res_Xft /= Vmax


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
    if do_DTW:
        # print("Warping time dynamically... or something like that...")   
        MOND_res = (vel_MOND - Vobs_raw) / Vmax

        """
        DTW analyses on full RCs.
        """
        for itr in range(num_iterations):
            dist_Xft = np.zeros((num_rad, num_rad))
            dist_Xft_rev = np.copy(dist_Xft)
            for n in range(num_rad):
                for m in range(num_rad):
                    dist_Xft[n, m] = abs(res_Xft[itr][n] - MOND_res[itr][m])
                    dist_Xft_rev[n, m] = abs(res_Xft[itr][num_rad-n-1] - MOND_res[itr][num_rad-m-1])
            
            # DTW!
            path, cost_mat = dtw(dist_Xft)
            x_path, y_path = zip(*path)
            Xft_cost = cost_mat[ num_rad-1, num_rad-1 ]
            Xft_cost /= (num_rad * 2)
            Xft_costs[itr][i] = Xft_cost

        if make_plots and noise in noise_arr[::10]:
            # Plot distance matrix and cost matrix with optimal path.
            plt.title("Dynamic time warping: Toy model (w/o feature)")
            plt.axis('off')

            plt.subplot(121)
            plt.title("Distance matrix")
            plt.imshow(dist_Xft, cmap=plt.cm.binary, interpolation="nearest", origin="lower")

            plt.subplot(122)
            plt.title("Cost matrix")
            plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
            plt.plot(x_path, y_path)

            plt.savefig(fileloc+f"Xft_matrix/ratio={round(noise/bump_size, 2)}.png", dpi=300, bbox_inches="tight")
            plt.close()

            # Visualize DTW alignment.
            plt.title("DTW alignment: Toy model (w/o feature)")

            diff = abs(max(MOND_res[max_index]) - min(res_Xft[max_index]))
            for x_i, y_j in path:
                plt.plot([x_i, y_j], [res_Xft[max_index][x_i] + diff, MOND_res[max_index][y_j] - diff], c="C7", alpha=0.4)
            plt.plot(np.arange(num_rad), res_Xft[max_index] + diff, c='k', label="Vobs")
            plt.plot(np.arange(num_rad), MOND_res[max_index] - diff, c="red", label="Vbar")
            plt.plot([], [], c='w', label="Normalized cost = {:.4f}".format(Xft_cost))

            plt.axis("off")
            plt.legend(bbox_to_anchor=(1,1))
            plt.savefig(fileloc+f"Xft_alignment/ratio={round(noise/bump_size, 2)}.png", dpi=300, bbox_inches="tight")
            plt.close()


        # Construct distance matrices.
        for itr in range(num_iterations):
            dist_data = np.zeros((num_rad, num_rad))
            dist_data_rev = np.copy(dist_data)
            for n in range(num_rad):
                for m in range(num_rad):
                    # Define new distance matrix construction s.t. cost = 0 for MOND (w/o noise).
                    # Take the average of forward and backward cost for more accurate value.
                    dist_data[n, m] = abs(residuals[itr][0][n] - MOND_res[itr][m])
                    dist_data_rev[n, m] = abs(residuals[itr][0][num_rad-n-1] - MOND_res[itr][num_rad-m-1])
                    # dist_data[n, m] = abs(residuals[0][n] - residuals[1][m])

            # DTW!
            path, cost_mat = dtw(dist_data)
            path_rev, cost_mat_rev = dtw(dist_data_rev)
            x_path, y_path = zip(*path)
            xrev_path, yrev_path = zip(*path_rev)

            cost_fwd = cost_mat[ num_rad-1, num_rad-1 ]
            cost_rev = cost_mat_rev[ num_rad-1, num_rad-1 ]
            cost = ( cost_fwd + cost_rev ) / 2
            norm_cost = cost / (num_rad * 2)
            dtw_costs[itr][i] = norm_cost

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

            diff = abs(max(MOND_res[max_index]) - min(residuals[max_index][0]))
            for x_i, y_j in path:
                plt.plot([x_i, y_j], [residuals[max_index][0][x_i] + diff, MOND_res[max_index][y_j] - diff], c="C7", alpha=0.4)
            plt.plot(np.arange(num_rad), np.array(residuals[max_index][0]) + diff, c='darkblue', label="Vobs")
            plt.plot(np.arange(num_rad), np.array(MOND_res[max_index]) - diff, c="red", label="Vbar")
            plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(cost))
            plt.plot([], [], c='w', label="Normalized cost = {:.4f}".format(norm_cost))

            plt.axis("off")
            plt.legend(bbox_to_anchor=(1,1))
            plt.savefig(fileloc+f"dtw_alignment/ratio={round(noise/bump_size, 2)}.png", dpi=300, bbox_inches="tight")
            plt.close()


        """
        DTW on RCs within window (of length 1) around feature.
        """
        # DTW along a small window (length 1) around the feature.
        window_size = 11
        for itr in range(num_iterations):
            dist_data = np.zeros((window_size, window_size))
            dist_data_rev = np.copy(dist_data)
            for n in range(window_size):
                for m in range(window_size):
                    # Define new distance matrix construction s.t. cost = 0 for MOND (w/o noise).
                    # Take the average of forward and backward cost for more accurate value.
                    dist_data[n, m] = abs(res_Xft[itr][44+n] - MOND_res[itr][44+m])
                    dist_data_rev[n, m] = abs(res_Xft[itr][num_rad-44-n] - MOND_res[itr][num_rad-44-m])
            
            path, cost_mat = dtw(dist_data)
            path_rev, cost_mat_rev = dtw(dist_data_rev)
            x_path, y_path = zip(*path)
            xrev_path, yrev_path = zip(*path_rev)

            cost_fwd = cost_mat[ window_size-1, window_size-1 ]
            cost_rev = cost_mat_rev[ window_size-1, window_size-1 ]
            win_cost = ( cost_fwd + cost_rev ) / 2
            win_cost /= (window_size * 2)
            Xft_window[itr][i] = win_cost

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

            plt.savefig(fileloc+f"dtw_window/Xft_matrix/ratio={round(noise/bump_size, 2)}.png", dpi=300, bbox_inches="tight")
            plt.close()

            # Visualize DTW alignment.
            plt.title("DTW alignment: Toy model")

            diff = abs(max(MOND_res[max_index]) - min(res_Xft[max_index]))
            for x_i, y_j in path:
                plt.plot([x_i, y_j], [res_Xft[max_index][44+x_i] + diff, MOND_res[max_index][44+y_j] - diff], c="C7", alpha=0.4)
            plt.plot(np.arange(window_size), np.array(res_Xft[max_index][44:55]) + diff, c='darkblue', label="Vobs")
            plt.plot(np.arange(window_size), np.array(MOND_res[max_index][44:55]) - diff, c="red", label="Vbar")
            plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(win_cost))

            plt.axis("off")
            plt.legend(bbox_to_anchor=(1,1))
            plt.savefig(fileloc+f"dtw_window/Xft_alignment/ratio={round(noise/bump_size, 2)}.png", dpi=300, bbox_inches="tight")
            plt.close()


        # DTW along a small window (length 1) around the feature.
        for itr in range(num_iterations):
            dist_data = np.zeros((window_size, window_size))
            dist_data_rev = np.copy(dist_data)
            for n in range(window_size):
                for m in range(window_size):
                    # Define new distance matrix construction s.t. cost = 0 for MOND (w/o noise).
                    # Take the average of forward and backward cost for more accurate value.
                    dist_data[n, m] = abs(residuals[itr][0][44+n] - MOND_res[itr][44+m])
                    dist_data_rev[n, m] = abs(residuals[itr][0][num_rad-44-n] - MOND_res[itr][num_rad-44-m])
            
            path, cost_mat = dtw(dist_data)
            path_rev, cost_mat_rev = dtw(dist_data_rev)
            x_path, y_path = zip(*path)
            xrev_path, yrev_path = zip(*path_rev)

            cost_fwd = cost_mat[ window_size-1, window_size-1 ]
            cost_rev = cost_mat_rev[ window_size-1, window_size-1 ]
            win_cost = ( cost_fwd + cost_rev ) / 2
            win_cost /= (window_size * 2)
            dtw_window[itr][i] = win_cost

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

            plt.savefig(fileloc+f"dtw_window/matrix/ratio={round(noise/bump_size, 2)}.png", dpi=300, bbox_inches="tight")
            plt.close()

            # Visualize DTW alignment.
            plt.title("DTW alignment: Toy model")

            diff = abs(max(MOND_res[max_index]) - min(residuals[max_index][0]))
            for x_i, y_j in path:
                plt.plot([x_i, y_j], [residuals[max_index][0][44+x_i] + diff, MOND_res[max_index][44+y_j] - diff], c="C7", alpha=0.4)
            plt.plot(np.arange(window_size), np.array(residuals[max_index][0][44:55]) + diff, c='darkblue', label="Vobs")
            plt.plot(np.arange(window_size), np.array(MOND_res[max_index][44:55]) - diff, c="red", label="Vbar")
            plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(win_cost))

            plt.axis("off")
            plt.legend(bbox_to_anchor=(1,1))
            plt.savefig(fileloc+f"dtw_window/alignment/ratio={round(noise/bump_size, 2)}.png", dpi=300, bbox_inches="tight")
            plt.close()


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

print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)

plt.title("Memory usage for toy_model.py")
plt.xlabel("Number of iterations executed")
plt.ylabel("Maximum memory used (kb)")
plt.plot(range(num_noise), memory_usage)

plt.savefig(fileloc+"memory_usage.png", dpi=300, bbox_inches="tight")
plt.close()


bump_ratio = noise_arr / bump_size

if do_DTW:
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
