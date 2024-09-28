# (C) 2024 Enoch Ko.
"""
Functions for calculating and plotting correlation coefficients.
"""
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

colors = [ 'tab:red', 'k' ]
c_corr = [ 'mediumblue', 'tab:green' ]
labels = [ 'Vbar', 'Vobs' ]
deriv_dir = [ "d0", "d1", "d2" ]
color_bar = "orange"


def corr_radii(num_iterations:int, der:int, num_rad:int, res_fits, v_werr, make_plots:bool=False,
               fileloc="", noise_ratio=0.0, rad=[], bump=[], Vraw=[], residuals=[], noise=0.):
    """
    res_fits: used in calculating correlation coefficients,
    v_werr:   used in calculating bar_ratio ([Vbar, Vobs]);
    both must have dimensions of itr x der x comp x rad.
    """
    if der not in [0, 1, 2]:
        raise ValueError("Only 0th, 1st and 2nd derivatives are supported.")
    
    bar_ratios   = [ [] for _ in range(num_iterations) ]
    radii_corr   = [ [] for _ in range(num_iterations) ]
    rad_spearman = np.zeros(num_iterations)
    rad_pearson  = np.zeros(num_iterations)

    for itr in range(num_iterations):
        # Correlate Vobs and Vbar (d0, d1, d2) as a function of (maximum) radius, i.e. spheres of increasing r.
        rad_corr = [[], []]
        for j in range(10, num_rad):
            rad_corr[0].append(stats.spearmanr(res_fits[itr][der][0][:j], res_fits[itr][der][1][:j])[0])
            rad_corr[1].append(stats.pearsonr(res_fits[itr][der][0][:j], res_fits[itr][der][1][:j])[0])
        radii_corr[itr] = rad_corr

        # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
        bar_ratio = []
        for rd in range(num_rad):
            bar_ratio.append(sum(v_werr[itr][1][:rd]/v_werr[itr][0][:rd]) / (rd+1))
        bar_ratios[itr] = bar_ratio

        # Correlate baryonic ratio with correlation coefficients.
        rad_spearman[itr] = stats.spearmanr(rad_corr[0], bar_ratio[10:])[0]
        rad_pearson[itr]  = stats.pearsonr(rad_corr[1], bar_ratio[10:])[0]
    
    # Extract 1-sigma percentiles and means from iterations.
    bar_percentiles = np.percentile( bar_ratios,   [16.0, 50.0, 84.0], axis=0 )
    corr_perc       = np.percentile( radii_corr,   [16.0, 50.0, 84.0], axis=0 )
    spearman_perc   = np.percentile( rad_spearman, [16.0, 50.0, 84.0], axis=0 )
    pearson_perc    = np.percentile( rad_pearson,  [16.0, 50.0, 84.0], axis=0 )

    if make_plots:
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
        fig.set_size_inches(7, 7)

        ax0.set_title("Correlation by increasing radii: Toy model")
        ax0.set_ylabel("Normalised velocities")
        ax2.set_xlabel("Radii (kpc)")

        for i in range(2):
            ax0.errorbar(rad, v_werr[0][i], noise/100.0, color=colors[i], alpha=0.3, capsize=3, fmt="o", ls="none")
            ax0.plot(rad, Vraw[0][i], color=colors[i], label=labels[i])
            if der == 0:
                ax1.scatter(rad, residuals[0][i], color=colors[i], alpha=0.3)
            ax1.plot(rad, res_fits[0][der][i], color=colors[i], alpha=0.7, label=labels[i])

        ax0.plot(rad, bump, '--', label="Feature")
        ax0.legend(loc="upper left", bbox_to_anchor=(1,1))
        ax0.grid()

        ax1.legend(loc="upper left", bbox_to_anchor=(1,1))
        ax1.grid()

        # Plot correlations and Vbar/Vobs.
        ax2.plot(rad[10:], corr_perc[1][0], color=c_corr[0], label=r"Spearman $\rho$")
        ax2.plot(rad[10:], corr_perc[1][1], color=c_corr[1], label=r"Pearson $\rho$")
        ax2.fill_between(rad[10:], corr_perc[0][0], corr_perc[2][0], color=c_corr[0], alpha=0.2)
        ax2.fill_between(rad[10:], corr_perc[0][1], corr_perc[2][1], color=c_corr[1], alpha=0.2)

        sigma_spearman = max(spearman_perc - spearman_perc[1])
        sigma_pearson = max(pearson_perc - pearson_perc[1])
        ax2.plot([], [], ' ', label=r"$\rho_s=$"+f"{round(spearman_perc[1], 3)} \u00B1 {round(sigma_spearman, 3)}"
                                    +r", $\rho_p=$"+f"{round(rad_pearson[1], 3)} \u00B1 {round(sigma_pearson, 3)}")
    
        ax5 = ax2.twinx()
        ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
        ax5.plot(rad[10:], bar_percentiles[1][10:], '--', color=color_bar, label="Vbar/Vobs")
        ax5.fill_between(rad[10:], bar_percentiles[0][10:], bar_percentiles[2][10:], color=color_bar, alpha=0.2)
        ax5.tick_params(axis='y', labelcolor=color_bar)
    
        ax2.legend(loc="upper left", bbox_to_anchor=(1.11, 1))
        ax2.grid()

        plt.subplots_adjust(hspace=0.05)
        fig.savefig(fileloc+f"correlations/radii_{deriv_dir[der]}/ratio={noise_ratio}.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # Return the 1-sigma percentiles of correlation coefficients across the whole RC (i.e. at r = rmax).
    full_corr = np.array(radii_corr)[:,:,-1]
    full_corr_perc = np.percentile(full_corr, [16.0, 50.0, 84.0], axis=0)
    return full_corr_perc.T


def corr_window(num_iterations:int, der:int, num_rad:int, res_fits, v_werr, win_size:int=11, make_plots:bool=False,
                fileloc="", noise_ratio=0.0, rad=[], bump=[], Vraw=[], residuals=[], noise=0.):
    """
    res_fits: used in calculating correlation coefficients,
    v_werr:   used in calculating bar_ratio ([Vbar, Vobs]);
    both must have dimensions of itr x der x comp x rad.
    """
    if der not in [0, 1, 2]:
        raise ValueError("Only 0th, 1st and 2nd derivatives are supported.")
    
    if win_size % 2 != 1:
        raise ValueError(f"Please use an odd integer for window size (you've entered win_size = {win_size}).")
    
    bar_ratios  = [ [] for _ in range(num_iterations) ]
    window_corr  = [ [] for _ in range(num_iterations) ]
    win_spearman = np.zeros(num_iterations)
    win_pearson  = np.zeros(num_iterations)

    wmin = int((win_size - 1) / 2)
    wmax = num_rad - wmin

    for itr in range(num_iterations):
        # Correlate Vobs and Vbar (d0, d1, d2) along a moving window of length 1 * kpc.
        win_corr = [[], []]
        for j in range(wmin, wmax):
            jmin, jmax = j - 5, j + 5
            win_corr[0].append(stats.spearmanr(res_fits[itr][der][0][jmin:jmax], res_fits[itr][der][1][jmin:jmax])[0])
            win_corr[1].append(stats.pearsonr(res_fits[itr][der][0][jmin:jmax], res_fits[itr][der][1][jmin:jmax])[0])
        window_corr[itr] = win_corr

        # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
        bar_ratio = []
        for j in range(wmin, wmax):
            bar_ratio.append( sum( v_werr[itr][1][j-wmin:j+wmin] / v_werr[itr][0][j-wmin:j+wmin] ) / 11 )
        bar_ratios[itr] = bar_ratio

        # Correlate baryonic ratio with correlation coefficients.
        win_spearman[itr] = stats.spearmanr(win_corr[0], bar_ratio)[0]
        win_pearson[itr]  = stats.pearsonr(win_corr[1], bar_ratio)[0]
    
    # Extract 1-sigma percentiles and means from iterations.
    bar_percentiles = np.percentile( bar_ratios,   [16.0, 50.0, 84.0], axis=0 )
    corr_perc       = np.percentile( window_corr,  [16.0, 50.0, 84.0], axis=0 )
    spearman_perc   = np.percentile( win_spearman, [16.0, 50.0, 84.0], axis=0 )
    pearson_perc    = np.percentile( win_pearson,  [16.0, 50.0, 84.0], axis=0 )

    if make_plots:
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
        fig.set_size_inches(7, 7)

        ax0.set_title("Correlation on moving window: Toy model")
        ax0.set_ylabel("Normalised velocities")
        ax2.set_xlabel("Radii (kpc)")

        for i in range(2):
            ax0.errorbar(rad, v_werr[0][i], noise/100.0, color=colors[i], alpha=0.3, capsize=3, fmt="o", ls="none")
            ax0.plot(rad, Vraw[0][i], color=colors[i], label=labels[i])
            if der == 0:
                ax1.scatter(rad, residuals[0][i], color=colors[i], alpha=0.3)
            ax1.plot(rad, res_fits[0][der][i], color=colors[i], alpha=0.7, label=labels[i])

        ax0.plot(rad, bump, '--', label="Feature")
        ax0.legend(loc="upper left", bbox_to_anchor=(1,1))
        ax0.grid()

        ax1.legend(loc="upper left", bbox_to_anchor=(1,1))
        ax1.grid()

        # Plot correlations and Vbar/Vobs.
        ax2.plot(rad[wmin:wmax], corr_perc[1][0], color=c_corr[0], label=r"Spearman $\rho$")
        ax2.plot(rad[wmin:wmax], corr_perc[1][1], color=c_corr[1], label=r"Pearson $\rho$")
        ax2.fill_between(rad[wmin:wmax], corr_perc[0][0], corr_perc[2][0], color=c_corr[0], alpha=0.2)
        ax2.fill_between(rad[wmin:wmax], corr_perc[0][1], corr_perc[2][1], color=c_corr[1], alpha=0.2)

        sigma_spearman = max(spearman_perc - spearman_perc[1])
        sigma_pearson = max(pearson_perc - pearson_perc[1])
        ax2.plot([], [], ' ', label=r"$\rho_s=$"+f"{round(spearman_perc[1], 3)} \u00B1 {round(sigma_spearman, 3)}"
                                    +r", $\rho_p=$"+f"{round(win_pearson[1], 3)} \u00B1 {round(sigma_pearson, 3)}")
    
        ax5 = ax2.twinx()
        ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
        ax5.plot(rad[wmin:wmax], bar_percentiles[1], '--', color=color_bar, label="Vbar/Vobs")
        ax5.fill_between(rad[wmin:wmax], bar_percentiles[0], bar_percentiles[2], color=color_bar, alpha=0.2)
        ax5.tick_params(axis='y', labelcolor=color_bar)
    
        ax2.legend(loc="upper left", bbox_to_anchor=(1.11, 1))
        ax2.grid()

        plt.subplots_adjust(hspace=0.05)
        fig.savefig(fileloc+f"correlations/window_{deriv_dir[der]}/ratio={noise_ratio}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Extract correlation coefficient from window centred on feature.
    window_corr = np.array(window_corr)
    mid_pt = math.floor( len(bar_ratio) / 2 )
    centre_corr = window_corr[:,:,mid_pt]
    centre_corr_perc = np.percentile(centre_corr, [16.0, 50.0, 84.0], axis=0)
    return centre_corr_perc.T
