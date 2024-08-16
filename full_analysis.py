#!/usr/bin/env python
"""
Correlation coefficients + dynamic time warping on GP residuals,
taking into account uncertainties (Vbar) and Vobs scattering (errV).

GP fits taken from combined_dtw.py output, which are saved in /mnt/users/koe/gp_fits/.
"""
import pandas as pd
from resource import getrusage, RUSAGE_SELF

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, interpolate, stats
import math

from tqdm import tqdm


testing = True
test_multiple = False   # Loops over the first handful of galaxies instead of just the fist one (DDO161).
make_plots = False
do_DTW = False
do_correlations = False

fileloc = "/mnt/users/koe/plots/full_analysis/"


# Dynamic programming code for DTW, see dtw.py for details.
def dp(dist_mat):
    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)


# Main code to run.
def main(g, r, vel):
    # Load in GP results from combined_dtw.py
    gp_fits = np.load("/mnt/users/koe/gp_fits/"+g+".npy")
    rad = gp_fits[0]
    mean_prediction = [ gp_fits[1], gp_fits[2], gp_fits[3], gp_fits[4] ]    # Mean predictions from GP for [ Vbar, Vobs, MOND, LCDM ]
    lower_percentile = [ gp_fits[5], gp_fits[6], gp_fits[7], gp_fits[8] ]   # 16t percentiles from GP
    upper_percentile = [ gp_fits[9], gp_fits[10], gp_fits[11], gp_fits[12] ]    # 84th percentiles from GP

    # "Raw" percentiles from uncertainties and scattering.
    raw_median = np.percentile(vel, 50.0, axis=2)               # dim = (4, r)
    raw_percentiles = np.percentile(vel, [16.0, 84.0], axis=2)  # dim = (2, 4, r)
    raw_errors = np.abs( raw_percentiles - raw_median )         # dim = (2, 4, r)

    # Compute residuals of fits.
    res_Vbar, res_Vobs, res_MOND, res_LCDM = [], [] ,[], []
    for k in range(len(r)):
        idx = (np.abs(rad - r[k])).argmin()
        res_Vbar.append(vel[0][k] - mean_prediction[0][idx])
        res_Vobs.append(vel[1][k] - mean_prediction[1][idx])
        res_MOND.append(vel[2][k] - mean_prediction[2][idx])
        res_LCDM.append(vel[3][k] - mean_prediction[3][idx])
    residuals = np.array([ res_Vbar, res_Vobs, res_MOND, res_LCDM ])    # dimensions = (4, len(r), num_samples)

    # Residual percentiles from uncertainties and scattering; dimensions = (4, 1 or 2, len(r)).
    res_median = np.percentile(residuals, 50.0, axis=2)                 # dim = (4, r)
    res_percentiles = np.percentile(residuals, [16.0, 84.0], axis=2)    # dim = (2, 4, r)
    res_errors = np.abs( res_percentiles - res_median )                 # dim = (2, 4, r)

    # Labels and colours for plots.
    v_comps = [ "Vbar (SPARC)", "Vobs (SPARC)", "Vobs (MOND)", "Vobs (LCDM)" ]
    colours = [ 'tab:red', 'k', 'mediumblue', 'tab:green' ]


    for j in range(4):
        if j == 1:
            plt.scatter(r, raw_median[j], color=colours[j], alpha=0.3)
        else:
            plt.errorbar(r, raw_median[j], raw_errors[:, j], color=colours[j], alpha=0.6, fmt='o', capsize=3)
        # Plot mean prediction from GP.
        plt.plot(rad, mean_prediction[j], color=colours[j], label=v_comps[j])
        # Fill in 1-sigma (68%) confidence band of GP fit.
        plt.fill_between(rad, lower_percentile[j], upper_percentile[j], color=colours[j], alpha=0.2)

    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.grid()
    plt.savefig(fileloc+"test.png", dpi=300, bbox_inches="tight")

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
            path, cost_mat = dp(dist_mats[j])
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
            v_d0.append(interpolate.pchip_interpolate(r, v_comp, rad))
            v_d1.append(interpolate.pchip_interpolate(r, v_comp, rad, der=1))
            v_d2.append(interpolate.pchip_interpolate(r, v_comp, rad, der=2))
        
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
                for j in range(10, len(rad)):
                    rad_corr[k][0].append(stats.spearmanr(res_fits[k][0][:j], res_fits[k][i][:j])[0])
                    rad_corr[k][1].append(stats.pearsonr(res_fits[k][0][:j], res_fits[k][i][:j])[0])
            correlations_r.append(rad_corr)


        """
        Plot GP fits, residuals (+ PCHIP) and correlations.
        """
        if make_plots:
            subdir = "correlations/radii/"
            color_bar = "orange"
            deriv_dir = [ "d0/", "d1/", "d2/" ]

            # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
            bar_ratio = []
            for rd in range(len(rad)):
                bar_ratio.append(sum(mean_prediction[0][:rd]/mean_prediction[1][:rd]) / (rd+1))

            # Plot corrletaions as 1 main plot + 1 subplot, using only Vobs from data for Vbar/Vobs.
            der_axis = [ "Residuals (km/s)", "1st derivative", "2nd derivative" ]
            for der in range(3):
                fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
                fig1.set_size_inches(7, 7)
                ax0.set_title("Residuals correlation: "+g)
                ax0.set_ylabel("Velocities (km/s)")

                for j in range(4):
                    # Scatter plot for data/mock data points.
                    ax0.scatter(r, vel[j], color=colours[j], alpha=0.3)
                    # Plot mean prediction from GP.
                    ax0.plot(rad, mean_prediction[j], color=colours[j], label=v_comps[j])
                    # Fill in 1-sigma (68%) confidence band of GP fit.
                    ax0.fill_between(rad, lower_percentile[j], upper_percentile[j], color=colours[j], alpha=0.2)

                ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
                ax0.grid()

                ax1.set_ylabel(der_axis[der])
                for j in range(4):
                    if der == 0:
                        ax1.scatter(r, residuals[j], color=colours[j], alpha=0.3)
                    ax1.plot(rad, res_fits[der][j], color=colours[j], label=v_comps[j])

                ax1.grid()

                ax2.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
                ax2.set_ylabel("Correlations")
                
                vel_comps = [ "Data", "MOND", r"$\Lambda$CDM" ]

                for j in range(3):
                    # Plot correlations and Vbar/Vobs.
                    ax2.plot(rad[10:], correlations_r[j][der][0], color=colours[j+1], label=vel_comps[j]+r": Spearman $\rho$")
                    ax2.plot(rad[10:], correlations_r[j][der][1], ':', color=colours[j+1], label=vel_comps[j]+r": Pearson $\rho$")
                    ax2.plot([], [], ' ', label=vel_comps[j]+r": $\rho_s=$"+str(round(stats.spearmanr(correlations_r[j][der][0], bar_ratio[10:])[0], 3))+r", $\rho_p=$"+str(round(stats.pearsonr(correlations_r[j][der][1], bar_ratio[10:])[0], 3)))

                ax5 = ax2.twinx()
                ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
                ax5.plot(rad[10:], bar_ratio[10:], '--', color=color_bar, label="Vbar/Vobs")
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

        if len(rad) > 100:
            # Correlate Vobs and Vbar (d0, d1, d2) along a moving window of length 1 * Reff.
            # correlations_w = win_corr arrays with [ data, MOND, LCDM ], so 3 Vobs x 3 derivatives x 2 correlations each,
            # where win_corr = [ [[Spearman d0], [Pearson d0]], [[S d1], [P d1]], [[S d2], [P d2]] ].
            wmax = len(rad) - 50
            correlations_w = []
            
            for vc in range(1, 4):
                win_corr = [ [[], []], [[], []], [[], []] ]
                for der in range(3):
                    for j in range(50, wmax):

                        idx = (np.abs(r - rad[j])).argmin()
                        X_jmin, X_jmax = math.ceil(r[max(0, idx-2)] * 100), math.ceil(r[min(len(r)-1, idx+2)] * 100)
                        
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
                wbar_ratio.append( sum( mean_prediction[0][j-50:j+50] / mean_prediction[1][j-50:j+50] ) / 101 )


            """
            Plot GP fits, residuals (+ PCHIP) and correlations.
            """
            if make_plots:
                subdir = "correlations/window/"

                # Plot corrletaions as 1 main plot + 1 subplot, using only Vobs from data for Vbar/Vobs.
                for der in range(3):
                    fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
                    fig1.set_size_inches(7, 7)
                    ax0.set_title("Moving window correlation: "+g)
                    ax0.set_ylabel("Velocities (km/s)")

                    for j in range(4):
                        # Scatter plot for data/mock data points.
                        ax0.scatter(r, vel[j], color=colours[j], alpha=0.3)
                        # Plot mean prediction from GP.
                        ax0.plot(rad, mean_prediction[j], color=colours[j], label=v_comps[j])
                        # Fill in 1-sigma (68%) confidence band of GP fit.
                        ax0.fill_between(rad, lower_percentile[j], upper_percentile[j], color=colours[j], alpha=0.2)

                    ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
                    ax0.grid()

                    ax1.set_ylabel(der_axis[der])
                    for j in range(4):
                        if der == 0:
                            ax1.scatter(r, residuals[j], color=colours[j], alpha=0.3)
                        ax1.plot(rad, res_fits[der][j], color=colours[j], label=v_comps[j])

                    ax1.grid()

                    ax2.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
                    ax2.set_ylabel("Correlations")
                    
                    vel_comps = [ "Data", "MOND", r"$\Lambda$CDM" ]

                    for j in range(3):
                        # Plot correlations and Vbar/Vobs.
                        ax2.plot(rad[50:wmax], correlations_w[j][der][0], color=colours[j+1], label=vel_comps[j]+r": Spearman $\rho$")
                        ax2.plot(rad[50:wmax], correlations_w[j][der][1], ':', color=colours[j+1], label=vel_comps[j]+r": Pearson $\rho$")
                        ax2.plot([], [], ' ', label=vel_comps[j]+r": $\rho_s=$"+str(round(stats.spearmanr(correlations_w[j][der][0], wbar_ratio)[0], 3))+r", $\rho_p=$"+str(round(stats.pearsonr(correlations_w[j][der][1], wbar_ratio)[0], 3)))

                    ax5 = ax2.twinx()
                    ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
                    ax5.plot(rad[50:wmax], wbar_ratio, '--', color=color_bar, label="Vbar/Vobs")
                    ax5.tick_params(axis='y', labelcolor=color_bar)
                    
                    ax2.legend(bbox_to_anchor=(1.64, 1.3))
                    ax2.grid()

                    plt.subplots_adjust(hspace=0.05)
                    fig1.savefig(fileloc+subdir+deriv_dir[der]+g+".png", dpi=300, bbox_inches="tight")
                    plt.close()


if __name__ == "__main__":
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
    num_samples = 100


    # Sample Vbar squared with uncertainties in M/L ratios, luminosities and distances.
    def Vbar_sq_unc(table, i_table, data, bulged, num_samples=num_samples):
        # Sample mass-to-light ratios
        dist_pdisk = np.random.normal(pdisk, 0.125, size=num_samples)
        dist_pgas = np.random.normal(1., 0.04, size=num_samples)
        if bulged:
            dist_pbul = np.random.normal(pbul, 0.175, size=num_samples)
        else:
            dist_pbul = np.zeros(num_samples)

        # Sample luminosity
        L36 = stats.truncnorm.rvs(-table["L"][i_table] / table["e_L"][i_table], np.inf, table["L"][i_table], table["e_L"][i_table], size=num_samples)
        dist_pdisk *= L36 / table["L"][i_table]
        dist_pbul *= L36 / table["L"][i_table]

        # Sample distance to the galaxy
        galdist = stats.truncnorm.rvs(-table["D"][i_table] / table["e_D"][i_table], np.inf, table["D"][i_table], table["e_D"][i_table], size=num_samples)
        dist_scale = galdist / table["D"][i_table]
        dist_scaling = np.full((len(data["Vdisk"]), num_samples), dist_scale)

        dist_pdisk = np.array([dist_pdisk] * len(data["Vdisk"]))
        dist_pbul = np.array([dist_pbul] * len(data["Vbul"]))
        dist_pgas = np.array([dist_pgas] * len(data["Vgas"]))

        Vdisk = np.array([data["Vdisk"]] * num_samples).T
        Vbul = np.array([data["Vbul"]] * num_samples).T
        Vgas = np.array([data["Vgas"]] * num_samples).T

        Vbar_squared = (dist_pdisk * Vdisk**2
                        + dist_pbul * Vbul**2
                        + dist_pgas * Vgas**2)
        Vbar_squared *= dist_scaling

        return Vbar_squared

    
    def MOND_unc(Vbar2_unc, num_samples=num_samples):
        r_unc = np.array([r] * num_samples).T
        acc = Vbar2_unc / r_unc
        y = acc / a0
        nu = 1 + np.sqrt((1 + 4/y))
        nu /= 2

        return np.sqrt(acc * nu * r_unc)
    
    def LCDM_unc(Vbar2_unc, i_table, num_samples=num_samples):
        vDM_unc = np.array([v_DM[i_table]] * num_samples).T
        return np.sqrt(Vbar2_unc + vDM_unc**2)

    # Scatter a Vobs array with Gaussian noise of width data["errV"].
    def Vobs_scat(Vobs, errV, num_samples=num_samples):
        errV_copies = np.array([errV] * num_samples).T
        return np.random.normal(Vobs, errV_copies)


    galaxies = np.load("/mnt/users/koe/gp_fits/galaxy.npy")
    galaxy_count = len(galaxies)

    if testing:
        if test_multiple:
            galaxy_count = 3
        else:
            galaxy_count = 31
    bulged_count = 0
    xbulge_count = 0
    
    correlations_ALL = []
    dtw_cost = [ [], [], [] ]
    norm_cost = [ [], [], [] ]

    for i in tqdm(range(galaxy_count)):
        if i < 30:
            continue

        g = galaxies[i]
        i_tab = np.where(table["Galaxy"] == g)[0][0]

        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
        r = data["Rad"] / table["Rdisk"][i_tab] # Normalised radius (Rdisk = scale length of stellar disk).

        Vbar_squared = Vbar_sq_unc(table, i_tab, data, bulged)
        data_copies = np.array([data["Vobs"]] * num_samples).T
        full_MOND = Vobs_scat(MOND_unc(Vbar_squared), data["errV"])
        full_LCDM = Vobs_scat(LCDM_unc(Vbar_squared, i_tab), data["errV"])

        velocities = np.array([ np.sqrt(Vbar_squared), data_copies, full_MOND, full_LCDM ])
        # Vmax = max(velocities[1])
        # velocities /= Vmax

        if bulged:
            bulged_count += 1
        else:
            xbulge_count += 1

        main(g, r.to_numpy(), velocities)


    """
    Plot histogram of normalized DTW costs (in ascending order of costs for data).
    """
    if make_plots:
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
            plt.bar(galaxies, costs_sorted[j], color=colours[j], alpha=0.3, label=hist_labels[j])
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
            plt.bar(galaxies, diff_sorted[j], color=colours[j], alpha=0.4, label=hist_labels[j])
            plt.axhline(y=mean_diff, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_diff))
        
        plt.legend()
        plt.xticks([])
        plt.savefig(fileloc+"dtw/histo2.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
