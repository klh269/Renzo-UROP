# (C) 2024 Enoch Ko.
"""
Functions for calculating and plotting correlation coefficients.
"""
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficient is not defined.")

colors = [ 'tab:red', 'k' ]
c_corr = [ 'mediumblue', 'tab:green' ]
labels = [ 'Vbar', 'Vobs' ]
deriv_dir = [ "d0", "d1", "d2" ]
color_bar = "orange"


def get_covariance(x_samples, y_samples, len:int):
    """
    Compute the empirical covariance matrix of x and y jointly
    from samples for calculating the generalized Pearson correlation.
    """
    # Stack into a single 2n x N_samples array
    X_stack = np.vstack( [ x_samples, y_samples ] )  # shape: (2*len, num_samples)

    # Compute empirical covariance across samples (axis=1)
    cov_matrix = np.cov( X_stack )  # shape: (2n, 2n)

    # Extract the n x n cross-covariance blocks
    Sigma_x = cov_matrix[:len, :len]
    Sigma_y = cov_matrix[len:, len:]
    # Sigma_xy = cov_matrix[:len, len:]

    # Output: Averaged covariance matrix of dimensions n x n
    return 0.5 * (Sigma_x + Sigma_y)


def generalized_pearson(x, y, cov):
    """
    Computes the generalized Pearson correlation coefficient, which takes into account the covariance of errors (e.g., in Vbar).

    Parameters:
        x, y: 1D numpy arrays of the same length.
        Sigma: 2D numpy array (n x n) covariance matrix representing
               covariance among the data points.
    
    Returns:
        Generalized Pearson correlation coefficient (float)
    """
    if x.shape != y.shape: raise ValueError("x and y must have the same shape.")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]: raise ValueError("cov must be a square matrix.")

    # Ensure column vector shape
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Invert the covariance matrix
    Sigma_inv = np.linalg.inv(cov)

    # Compute the numerator and denominator
    numerator = float(x.T @ Sigma_inv @ y)
    denominator = np.sqrt(float(x.T @ Sigma_inv @ x) * float(y.T @ Sigma_inv @ y))

    return numerator / denominator



def corr_radii(num_iterations:int, num_rad:int, v_werr, use_window:bool=False, make_plots:bool=False,
               fileloc="", noise_ratio=0.0, rad=[], bump=[], Vraw=[], residuals=[], noise=0.):
    """
    v_werr must have dimensions of itr x 2 (Vbar, Vobs) x rad.
    """
    # bar_ratios   = [ [] for _ in range(num_iterations) ]
    radii_corr   = [ [] for _ in range(num_iterations) ]
    # rad_spearman = np.zeros(num_iterations)
    # rad_pearson  = np.zeros(num_iterations)

    for itr in range(num_iterations):
        # Correlate Vobs and Vbar (d0) as a function of (maximum) radius, i.e. spheres of increasing r.
        rad_corr = [[], []]

        if use_window:
            for j in range(2, num_rad):
                rad_corr[0].append(stats.spearmanr(v_werr[itr][0][:j], v_werr[itr][1][:j])[0])
                rad_corr[1].append(stats.pearsonr(v_werr[itr][0][:j], v_werr[itr][1][:j])[0])
        else:
            for j in range(5, num_rad):
                rad_corr[0].append(stats.spearmanr(v_werr[itr][0][:j], v_werr[itr][1][:j])[0])
                rad_corr[1].append(stats.pearsonr(v_werr[itr][0][:j], v_werr[itr][1][:j])[0])

        radii_corr[itr] = rad_corr

        # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
        # bar_ratio = []
        # for rd in range(num_rad):
        #     bar_ratio.append( sum(v_werr[itr][0][:rd] / v_werr[itr][1][:rd]) / (rd + 1) )
        # bar_ratios[itr] = bar_ratio

        # # Correlate baryonic ratio with correlation coefficients.
        # if use_window:
        #     rad_spearman[itr] = stats.spearmanr(rad_corr[0], bar_ratio[2:])[0]
        #     rad_pearson[itr]  = stats.pearsonr(rad_corr[1], bar_ratio[2:])[0]
        # else:
        #     rad_spearman[itr] = stats.spearmanr(rad_corr[0], bar_ratio[5:])[0]
        #     rad_pearson[itr]  = stats.pearsonr(rad_corr[1], bar_ratio[5:])[0]
    
    # Extract 1-sigma percentiles and means from iterations.
    # bar_percentiles = np.nanpercentile( bar_ratios,   [16.0, 50.0, 84.0], axis=0 )
    corr_perc       = np.nanpercentile( radii_corr,   [16.0, 50.0, 84.0], axis=0 )
    # spearman_perc   = np.nanpercentile( rad_spearman, [16.0, 50.0, 84.0], axis=0 )
    # pearson_perc    = np.nanpercentile( rad_pearson,  [16.0, 50.0, 84.0], axis=0 )

    if make_plots:
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
        fig.set_size_inches(7, 7)

        ax0.set_title("Correlation by increasing radii")
        ax0.set_ylabel("Normalised velocities")
        ax2.set_xlabel("Radii (kpc)")

        for i in range(2):
            ax0.errorbar(rad, v_werr[0][i], noise/100.0, color=colors[i], alpha=0.3, capsize=3, fmt="o", ls="none")
            ax0.plot(rad, Vraw[0][i], color=colors[i], label=labels[i])
            ax1.scatter(rad, residuals[0][i], color=colors[i], alpha=0.3)
            ax1.plot(rad, v_werr[0][i], color=colors[i], alpha=0.7, label=labels[i])

        ax0.plot(rad, bump, '--', label="Feature")
        ax0.legend(loc="upper left", bbox_to_anchor=(1,1))
        ax0.grid()

        ax1.legend(loc="upper left", bbox_to_anchor=(1,1))
        ax1.grid()

        # Plot correlations and Vbar/Vobs.
        ax2.plot(rad[5:], corr_perc[1][0], color=c_corr[0], label=r"Spearman $\rho$")
        ax2.plot(rad[5:], corr_perc[1][1], color=c_corr[1], label=r"Pearson $\rho$")
        ax2.fill_between(rad[5:], corr_perc[0][0], corr_perc[2][0], color=c_corr[0], alpha=0.2)
        ax2.fill_between(rad[5:], corr_perc[0][1], corr_perc[2][1], color=c_corr[1], alpha=0.2)

        # sigma_spearman = max(spearman_perc - spearman_perc[1])
        # sigma_pearson = max(pearson_perc - pearson_perc[1])
        # ax2.plot([], [], ' ', label=r"$\rho_s=$"+f"{round(spearman_perc[1], 3)} \u00B1 {round(sigma_spearman, 3)}"
        #                             +r", $\rho_p=$"+f"{round(rad_pearson[1], 3)} \u00B1 {round(sigma_pearson, 3)}")
    
        # ax5 = ax2.twinx()
        # ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
        # ax5.plot(rad[5:], bar_percentiles[1][5:], '--', color=color_bar, label="Vbar/Vobs")
        # ax5.fill_between(rad[5:], bar_percentiles[0][5:], bar_percentiles[2][5:], color=color_bar, alpha=0.2)
        # ax5.tick_params(axis='y', labelcolor=color_bar)
    
        # ax2.legend(loc="upper left", bbox_to_anchor=(1.11, 1))
        ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax2.grid()

        plt.subplots_adjust(hspace=0.05)
        fig.savefig(fileloc+f"correlations/radii_d0/ratio={noise_ratio}.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # Return the 1-sigma percentiles of correlation coefficients across the whole RC (i.e. at r = rmax).
    full_corr = np.array(radii_corr)[:,:,-1]
    full_corr_perc = np.nanpercentile(full_corr, [16.0, 50.0, 84.0], axis=0)    # dim = 3 x 2 (Spearman, Pearson)

    return full_corr_perc.T   # dim = 2 x 3


def corr_radii_der(num_iterations:int, der:int, num_rad:int, res_fits, v_werr, make_plots:bool=False,
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
        for j in range(5, num_rad):
            rad_corr[0].append(stats.spearmanr(res_fits[itr][der][0][:j], res_fits[itr][der][1][:j])[0])
            rad_corr[1].append(stats.pearsonr(res_fits[itr][der][0][:j], res_fits[itr][der][1][:j])[0])
        radii_corr[itr] = rad_corr

        # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
        bar_ratio = []
        for rd in range(num_rad):
            bar_ratio.append(sum(v_werr[itr][0][:rd]/v_werr[itr][1][:rd]) / (rd+1))
        bar_ratios[itr] = bar_ratio

        # Correlate baryonic ratio with correlation coefficients.
        rad_spearman[itr] = stats.spearmanr(rad_corr[0], bar_ratio[5:])[0]
        rad_pearson[itr]  = stats.pearsonr(rad_corr[1], bar_ratio[5:])[0]
    
    # Extract 1-sigma percentiles and means from iterations.
    bar_percentiles = np.nanpercentile( bar_ratios,   [16.0, 50.0, 84.0], axis=0 )
    corr_perc       = np.nanpercentile( radii_corr,   [16.0, 50.0, 84.0], axis=0 )
    spearman_perc   = np.nanpercentile( rad_spearman, [16.0, 50.0, 84.0], axis=0 )
    pearson_perc    = np.nanpercentile( rad_pearson,  [16.0, 50.0, 84.0], axis=0 )

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
        ax2.plot(rad[5:], corr_perc[1][0], color=c_corr[0], label=r"Spearman $\rho$")
        ax2.plot(rad[5:], corr_perc[1][1], color=c_corr[1], label=r"Pearson $\rho$")
        ax2.fill_between(rad[5:], corr_perc[0][0], corr_perc[2][0], color=c_corr[0], alpha=0.2)
        ax2.fill_between(rad[5:], corr_perc[0][1], corr_perc[2][1], color=c_corr[1], alpha=0.2)

        sigma_spearman = max(spearman_perc - spearman_perc[1])
        sigma_pearson = max(pearson_perc - pearson_perc[1])
        ax2.plot([], [], ' ', label=r"$\rho_s=$"+f"{round(spearman_perc[1], 3)} \u00B1 {round(sigma_spearman, 3)}"
                                    +r", $\rho_p=$"+f"{round(rad_pearson[1], 3)} \u00B1 {round(sigma_pearson, 3)}")
    
        ax5 = ax2.twinx()
        ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
        ax5.plot(rad[5:], bar_percentiles[1][5:], '--', color=color_bar, label="Vbar/Vobs")
        ax5.fill_between(rad[5:], bar_percentiles[0][5:], bar_percentiles[2][5:], color=color_bar, alpha=0.2)
        ax5.tick_params(axis='y', labelcolor=color_bar)
    
        ax2.legend(loc="upper left", bbox_to_anchor=(1.11, 1))
        ax2.grid()

        plt.subplots_adjust(hspace=0.05)
        fig.savefig(fileloc+f"correlations/radii_{deriv_dir[der]}/ratio={noise_ratio}.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # Return the 1-sigma percentiles of correlation coefficients across the whole RC (i.e. at r = rmax).
    full_corr = np.array(radii_corr)[:,:,-1]
    full_corr_perc = np.nanpercentile(full_corr, [16.0, 50.0, 84.0], axis=0)
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
            bar_ratio.append( sum( v_werr[itr][0][j-wmin:j+wmin] / v_werr[itr][1][j-wmin:j+wmin] ) / 11 )
        bar_ratios[itr] = bar_ratio

        # Correlate baryonic ratio with correlation coefficients.
        win_spearman[itr] = stats.spearmanr(win_corr[0], bar_ratio)[0]
        win_pearson[itr]  = stats.pearsonr(win_corr[1], bar_ratio)[0]
    
    # Extract 1-sigma percentiles and means from iterations.
    bar_percentiles = np.nanpercentile( bar_ratios,   [16.0, 50.0, 84.0], axis=0 )
    corr_perc       = np.nanpercentile( window_corr,  [16.0, 50.0, 84.0], axis=0 )
    spearman_perc   = np.nanpercentile( win_spearman, [16.0, 50.0, 84.0], axis=0 )
    pearson_perc    = np.nanpercentile( win_pearson,  [16.0, 50.0, 84.0], axis=0 )

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
    centre_corr_perc = np.nanpercentile(centre_corr, [16.0, 50.0, 84.0], axis=0)
    return centre_corr_perc.T
