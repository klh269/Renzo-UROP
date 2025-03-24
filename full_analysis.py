#!/usr/bin/env python
"""
Correlation coefficients + dynamic time warping on GP residuals,
taking into account uncertainties (Vbar) and Vobs scattering (errV).

GP fits taken from combined_dtw.py output, which are saved in /mnt/users/koe/gp_fits/.
"""
import pandas as pd
from resource import getrusage, RUSAGE_SELF
import jax

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utils_analysis.dtw_utils import dtw
from utils_analysis.mock_gen import Vbar_sq_unc, MOND_unc, Vobs_scat
from utils_analysis.extract_ft import ft_check
from tqdm import tqdm

plt.rcParams.update({'font.size': 13})


testing = False
test_multiple = False   # Loops over the first handful of galaxies instead of just the fist one (DDO161).
make_plots = True
use_DTW = True
do_correlations = True
use_MSE = False

fileloc = "/mnt/users/koe/plots/SPARC_fixedls/"
# Options: cost wrt MOND: "dtw/"; cost wrt LCDM: "dtw/cost_vsLCDM/", original cost: "dtw/cost_vsVbar/".
if use_DTW:
    # fname_DTW = fileloc + "dtw/cost_vsVbar/"
    fname_DTW = fileloc + "dtw/"
    # print(f"fname_DTW = {fname_DTW}")
num_samples = 1000   # No. of iterations sampling for uncertainties + errors.


# Main code to run.
def main(g, r, v_data, v_mock, num_samples=num_samples):
    # Load in GP results from combined_dtw.py
    gp_fits = np.load("/mnt/users/koe/gp_fits/"+g+".npy")
    rad = gp_fits[0]
    mean_prediction = [ gp_fits[1], gp_fits[3], gp_fits[4], gp_fits[2] ]    # Mean predictions from GP for [ Vbar, MOND, LCDM, Vobs ]
    lower_percentile = [ gp_fits[5], gp_fits[7], gp_fits[8], gp_fits[6] ]   # 16t percentiles from GP
    upper_percentile = [ gp_fits[9], gp_fits[11], gp_fits[12], gp_fits[10] ]    # 84th percentiles from GP

    # "Raw" percentiles from uncertainties and scattering.
    raw_median = np.percentile(v_mock, 50.0, axis=2)                # dim = (4, r)
    raw_percentiles = np.percentile(v_mock, [16.0, 84.0], axis=2)   # dim = (2, 4, r)
    raw_errors = np.abs( raw_percentiles - raw_median )             # dim = (2, 4, r)

    # Compute residuals of fits.
    res_Vbar_data, res_Vobs, res_Vbar_mock, res_MOND, res_LCDM = [], [] ,[], [], []
    for k in range(len(r)):
        idx = (np.abs(rad - r[k])).argmin()
        
        res_Vbar_data.append(v_data[0][k] - mean_prediction[0][idx])
        res_Vobs.append(v_data[1][k] - mean_prediction[3][idx])

        res_Vbar_mock.append(v_mock[0][k] - mean_prediction[0][idx])
        res_MOND.append(v_mock[1][k] - mean_prediction[1][idx])
        res_LCDM.append(v_mock[2][k] - mean_prediction[2][idx])

    res_data = np.array([ res_Vbar_data, res_Vobs ])            # dim = (2, len(r))
    res_mock = np.array([ res_Vbar_mock, res_MOND, res_LCDM ])  # dim = (3, len(r), num_samples)

    # Residual percentiles from uncertainties and scattering; dimensions = (3, 1 or 2, len(r)).
    res_median = np.percentile(res_mock, 50.0, axis=2)                  # dim = (3, r)
    res_percentiles = np.percentile(res_mock, [16.0, 84.0], axis=2)     # dim = (2, 3, r)
    res_errors = np.abs( res_percentiles - res_median )                 # dim = (2, 3, r)

    # Labels and colours for plots.
    v_comps = [ "Vbar (SPARC)", "Vobs (MOND)", "Vobs (LCDM)", "Vobs (SPARC)" ]
    colours = [ 'tab:red', 'mediumblue', 'tab:green', 'k' ]

    lb, rb, widths = ft_check(res_data[1][5:], v_data[2][5:])
    if len(lb)>0:
        print(f"\nFeature found in Vobs of {g}")
        print(f"Properties: lb={[x+5 for x in lb]}, rb={[x+5 for x in rb]}, widths={widths}")

    lb, rb, widths = ft_check(res_data[0][5:], res_errors[1,0][5:])
    if len(lb)>0:
        print(f"\nFeature found in Vbar of {g}")
        print(f"Properties: lb={[x+5 for x in lb]}, rb={[x+5 for x in rb]}, widths={widths}")


    """
    Simple mean-squared error (MSE) relative to GP fits.
    """
    def meansq_err( r, y_true, y_pred ):
        if len(y_true) != len(y_pred):
            raise RuntimeError(f"Length of y_true ({len(y_true)}) does not equal length of y_pred ({len(y_pred)})!")
        return np.sum( (y_pred - y_true)**2 / len(r) )

    if use_MSE:
        res_ref = np.zeros(len(r))
        mse_data = meansq_err( r, res_Vobs, res_ref )
        mse_MOND, mse_LCDM = [], []
        for smp in range(num_samples):
            mse_MOND.append( meansq_err(r, np.array(res_MOND)[:,smp], res_ref) )
            mse_LCDM.append( meansq_err(r, np.array(res_LCDM)[:,smp], res_ref) )
        mse_perc[0].append( mse_data )
        mse_perc[1].append( np.percentile(mse_MOND, [16.0, 50.0, 84.0]) )
        mse_perc[2].append( np.percentile(mse_LCDM, [16.0, 50.0, 84.0]) )


    """
    DTW on GP residuals.
    """
    if use_DTW:
        dtw_cost_smp = [ [], [], [] ]
        norm_cost_smp = [ [], [], [] ]

        for smp in range(num_samples):
        # for smp in tqdm(range(num_samples), desc="DTW"):
            # Construct distance matrices.
            dist_data = np.zeros((len(r), len(r)))
            dist_MOND = np.copy(dist_data)
            dist_LCDM = np.copy(dist_data)
            
            for n in range(len(r)):
                for m in range(len(r)):
                    # Construct distance matrix such that cost = 0 if Vobs = MOND(Vbar).
                    # if fname_DTW ==fileloc+"dtw/":
                    #     dist_data[n, m] = np.abs(res_Vobs[n] - res_MOND[m][smp])
                    #     dist_MOND[n, m] = np.abs(res_MOND[n][smp] - res_MOND[m][smp])
                    #     dist_LCDM[n, m] = np.abs(res_LCDM[n][smp] - res_MOND[m][smp])

                    # # Alternative constructions:
                    # elif fname_DTW == fileloc+"dtw/cost_vsLCDM/":
                    #     dist_data[n, m] = np.abs(res_Vobs[n] - res_LCDM[m][smp])
                    #     dist_MOND[n, m] = np.abs(res_MOND[n][smp] - res_LCDM[m][smp])
                    #     dist_LCDM[n, m] = np.abs(res_LCDM[n][smp] - res_LCDM[m][smp])
                    # else:
                    dist_data[n, m] = np.abs(res_Vobs[n] - res_Vbar_data[m])
                    dist_MOND[n, m] = np.abs(res_MOND[n][smp] - res_Vbar_data[m])
                    dist_LCDM[n, m] = np.abs(res_LCDM[n][smp] - res_Vbar_data[m])
            
            dist_mats = np.array([ dist_data, dist_MOND, dist_LCDM ])
            mats_dir = [ "data/", "MOND/", "LCDM/" ]
            
            # DTW!
            for j in range(3):
                path, cost_mat = dtw(dist_mats[j])
                x_path, y_path = zip(*path)
                cost = cost_mat[ len(r)-1, len(r)-1 ]
                dtw_cost_smp[j].append(cost)
                norm_cost_smp[j].append(cost / (2 * len(r)))

                if make_plots and smp == 0:
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

                    plt.savefig(fname_DTW+"matrix_"+mats_dir[j]+g+".png", dpi=300, bbox_inches="tight")
                    plt.close('all')

                    # Visualize DTW alignment.
                    plt.title("DTW alignment: "+g)

                    # Settings for visualizing different DTW constructions.
                    # if fname_DTW == fileloc+"dtw/":
                    #     ref_curve = [ res_MOND, "mediumblue", "MOND" ]
                    # elif fname_DTW == fileloc+"dtw/cost_vsLCDM/":
                    #     ref_curve = [ res_LCDM, "tab:green", r"$\Lambda$CDM" ]
                    # else:
                    ref_curve = [ res_Vbar_data, "tab:red", "Vbar" ]

                    if j == 0:
                        diff = abs(max(np.array(ref_curve[0])) - min(res_Vobs))
                        for x_i, y_j in path:
                            plt.plot([x_i, y_j], [res_Vobs[x_i] + diff, ref_curve[0][y_j] - diff], c="C7", alpha=0.4)
                        plt.plot(np.arange(len(r)), np.array(res_Vobs) + diff, c='k', label=v_comps[3])

                    else: 
                        diff = abs(max(np.array(ref_curve[0])) - min(np.array(res_mock)[j,:,smp]))
                        for x_i, y_j in path:
                            plt.plot([x_i, y_j], [res_mock[j][x_i][smp] + diff, ref_curve[0][y_j] - diff], c="C7", alpha=0.4)
                        plt.plot(np.arange(len(r)), np.array(res_mock)[j,:,smp] + diff, c=colours[j], label=v_comps[j])

                    plt.plot(np.arange(len(r)), np.array(ref_curve[0]) - diff, c=ref_curve[1], label=ref_curve[2])
                    plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(cost))
                    plt.plot([], [], c='w', label="Normalized cost = {:.4f}".format(cost/(len(r)*2)))

                    plt.axis("off")
                    plt.legend(bbox_to_anchor=(1,1))
                    plt.savefig(fname_DTW+"vis_"+mats_dir[j]+g+".png", dpi=300, bbox_inches="tight")
                    plt.close('all')
        
        for j in range(3):
            dtw_cost[j].append(dtw_cost_smp[j])
            norm_cost[j].append(norm_cost_smp[j])


    """
    Code for calculating correlation coefficients on GP residuals.
    """
    if do_correlations:

        # Interpolate the residuals with cubic Hermite spline splines.
        # v_d0, v_d1, v_d2 = [], [], []
        # for v_comp in res_data:
        #     v_d0.append(interpolate.pchip_interpolate(r, v_comp, rad))
        #     v_d1.append(interpolate.pchip_interpolate(r, v_comp, rad, der=1))
        #     v_d2.append(interpolate.pchip_interpolate(r, v_comp, rad, der=2))
        
        # res_fits_data = [ v_d0, v_d1, v_d2 ]

        # Compute correlation coefficients for data Vobs vs Vbar.
        # rcorr_data = [ [[], []], [[], []], [[], []] ]
        pearsonr_data = []
        # for k in range(3):
        for j in range(3, len(r)+1):
            # rcorr_data_d0[0].append(stats.spearmanr(res_data[0][:j], res_data[1][:j])[0])
            pearsonr_data.append(stats.pearsonr(res_data[0][:j], res_data[1][:j])[0])

        # spearman_data.append(rcorr_data_d0[0][-1])
        pearson_data.append(pearsonr_data[-1])

        # Compute correlation coefficients for mock Vobs vs Vbar.
        radii_corr = []     # dim = (num_samples/10, 2 x mock_vcomps, 3 x der, 2 x rho, rad)
        # res_fits_mock = []

        for smp in range(num_samples):
        # for smp in tqdm(range(num_samples), desc="Correlation by radii"):
            if smp % 5:
                continue

            # Interpolate the residuals with cubic Hermite spline splines.
            # v_d0, v_d1, v_d2 = [], [], []
            # for v_comp in res_mock[:, :, smp]:
            #     v_d0.append(interpolate.pchip_interpolate(r, v_comp, rad))
            #     v_d1.append(interpolate.pchip_interpolate(r, v_comp, rad, der=1))
            #     v_d2.append(interpolate.pchip_interpolate(r, v_comp, rad, der=2))
            
            # res_fits = [ v_d0, v_d1, v_d2 ]
            # res_fits_mock.append(res_fits)

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
                # rad_corr = [ [[], []], [[], []], [[], []] ]
                pearsonr_mock = []
                # for k in range(3):
                for j in range(3, len(r)+1):
                    # rcorr_mock_d0[0].append(stats.spearmanr(res_mock[0,:j,smp], res_mock[i,:j,smp])[0])
                    pearsonr_mock.append(stats.pearsonr(res_Vbar_data[:j], res_mock[i,:j,smp])[0])
                correlations_r.append(pearsonr_mock)
            
            radii_corr.append(correlations_r)
        
        res_mock_percentiles = np.percentile(res_mock, [16.0, 50.0, 84.0], axis=2)
        rcorr_percentiles = np.percentile(radii_corr, [16.0, 50.0, 84.0], axis=0)
        # spearman_mock.append([ rcorr_percentiles[:,0,0,0,-1], rcorr_percentiles[:,1,0,0,-1] ])
        pearson_mock.append([ rcorr_percentiles[:,0,-1], rcorr_percentiles[:,1,-1] ])


        """
        Plot GP fits, residuals and correlations.
        """
        if make_plots:
            # subdir = "correlations/radii/"
            # color_bar = "orange"
            # deriv_dir = [ "d0/", "d1/", "d2/" ]
            c_temp = [ 'tab:red', 'mediumblue', 'tab:green' ]
            labels_temp = [ "Vbar (SPARC)", "Vobs (MOND)", r"Vobs ($\Lambda$CDM)", "Vobs (SPARC)" ]

            # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
            # bar_ratio = []
            # for rd in range(len(rad)):
            #     bar_ratio.append(sum(mean_prediction[0][:rd]/mean_prediction[3][:rd]) / (rd+1))

            # Plot corrletaions as 1 main plot (+ residuals) + 1 subplot, using only Vobs from data for Vbar/Vobs.
            # der_axis = [ "Residuals (km/s)", "1st derivative", "2nd derivative" ]

            # """Spearman correlations"""
            # for der in range(1):
            #     fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
            #     fig1.set_size_inches(7, 7)
            #     ax0.set_title("Residuals correlation: "+g)
            #     ax0.set_ylabel("Velocities (km/s)")
                
            #     for j in range(4):
            #         if j == 3:
            #             ax0.errorbar(r, v_data[1], data["errV"], color='k', alpha=0.3, fmt='o', capsize=2)
            #         else:
            #             ax0.errorbar(r, raw_median[j], raw_errors[:, j], c=c_temp[j], alpha=0.3, fmt='o', capsize=2)
            #         # Plot mean prediction from GP.
            #         ax0.plot(rad, mean_prediction[j], color=colours[j], label=labels_temp[j])
            #         # Fill in 1-sigma (68%) confidence band of GP fit.
            #         ax0.fill_between(rad, lower_percentile[j], upper_percentile[j], color=colours[j], alpha=0.2)

            #     ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
            #     ax0.grid()

            #     ax1.set_ylabel(der_axis[der])
            #     for j in range(4):
            #         # Plots for mock Vobs + Vbar (sampled w/ uncertainties).
            #         if j == 3:
            #             if der == 0:
            #                 ax1.errorbar(r, res_data[1], v_data[2], color='k', alpha=0.3, ls='none', fmt='o', capsize=2)
            #             ax1.plot(rad, res_data[1], color='k', label=labels_temp[j])
            #         else:
            #             if der == 0:
            #                 ax1.scatter(r, res_median[j], c=c_temp[j], alpha=0.3)
            #                 # ax1.errorbar(r, res_median[j], res_errors[:, j], color=colours[j], alpha=0.3, ls='none', fmt='o', capsize=2)
            #             ax1.plot(rad, res_mock_percentiles[1][j], c=c_temp[j], label=labels_temp[j])
            #             ax1.fill_between(rad, res_mock_percentiles[0][j], res_mock_percentiles[2][j], color=c_temp[j], alpha=0.15)

            #     ax1.grid()

            #     ax2.set_xlabel("Radii (kpc)")
            #     ax2.set_ylabel("Correlations w/ Vbar")
                
            #     vel_comps = [ "MOND", r"$\Lambda$CDM", "Data" ]

            #     for j in range(2):
            #         mean_spearmanr = 0.
            #         mean_pearsonr = 0.

            #         ax2.plot(rad[10:], rcorr_percentiles[1][j][der][0], c=c_temp[j+1], label=vel_comps[j]+r": Spearman $\rho$")
            #         ax2.fill_between(rad[10:], rcorr_percentiles[0][j][der][0], rcorr_percentiles[2][j][der][0], color=colours[j+1], alpha=0.2)
                    
            #         for smp in range(len(radii_corr)):
            #             mean_spearmanr += stats.spearmanr(radii_corr[smp][j][der][0], bar_ratio[10:])[0] / len(radii_corr)
            #             mean_pearsonr += stats.pearsonr(radii_corr[smp][j][der][1], bar_ratio[10:])[0] / len(radii_corr)
            #         ax2.plot([], [], ' ', label=r": $\rho_s=$"+str(round(mean_spearmanr, 3)))

            #     ax2.plot(rad[10:], rcorr_data[der][0], c='k', label=vel_comps[2]+r": Spearman $\rho$")
            #     ax2.plot([], [], ' ', label=r": $\rho_s=$"+str(round(np.nanmean(rcorr_data[der][0]), 3)))

            #     ax5 = ax2.twinx()
            #     ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
            #     ax5.plot(rad[10:], bar_ratio[10:], '--', color=color_bar, label="Vbar/Vobs")
            #     ax5.tick_params(axis='y', labelcolor=color_bar)
                
            #     # ax2.legend(bbox_to_anchor=(1.64, 1.3))
            #     ax2.grid()

            #     plt.subplots_adjust(hspace=0.05)
            #     fig1.savefig(fileloc+subdir+deriv_dir[der]+g+".png", dpi=300, bbox_inches="tight")
            #     plt.close()

            """Pearson correlations."""
            # for der in range(1):
            fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
            fig1.set_size_inches(7, 7)
            ax0.set_title("Residuals correlation: "+g)
            ax0.set_ylabel("Velocities (km/s)")
            
            for j in range(4):
                if j == 3:
                    ax0.errorbar(r, v_data[1], data["errV"], color='k', alpha=0.3, fmt='o', capsize=2)
                else:
                    ax0.errorbar(r, raw_median[j], raw_errors[:, j], c=c_temp[j], alpha=0.3, fmt='o', capsize=2)
                # Plot mean prediction from GP.
                ax0.plot(rad, mean_prediction[j], color=colours[j], label=labels_temp[j])
                # Fill in 1-sigma (68%) confidence band of GP fit.
                ax0.fill_between(rad, lower_percentile[j], upper_percentile[j], color=colours[j], alpha=0.2)

            ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
            ax0.grid()

            ax1.set_ylabel("Residuals (km/s)")
            for j in range(4):
                # Plots for mock Vobs + Vbar (sampled w/ uncertainties).
                if j == 3:
                    # if der == 0:
                    ax1.errorbar(r[5:], res_data[1][5:], v_data[2][5:], color='k', alpha=0.3, ls='none', fmt='o', capsize=2)
                    ax1.plot(r[5:], res_data[1][5:], color='k', label=labels_temp[j])
                else:
                    # if der == 0:
                    ax1.scatter(r[5:], res_median[j][5:], c=c_temp[j], alpha=0.3)
                    # ax1.errorbar(r, res_median[j], res_errors[:, j], color=colours[j], alpha=0.3, ls='none', fmt='o', capsize=2)
                    ax1.plot(r[5:], res_mock_percentiles[1][j][5:], c=c_temp[j], label=labels_temp[j])
                    ax1.fill_between(r[5:], res_mock_percentiles[0][j][5:], res_mock_percentiles[2][j][5:], color=c_temp[j], alpha=0.15)

            ax1.grid()

            ax2.set_xlabel("Radii (kpc)")
            ax2.set_ylabel("Correlations w/ Vbar")
            
            vel_comps = [ "MOND", r"$\Lambda$CDM", "Data" ]

            for j in range(2):
                # mean_spearmanr = 0.
                # mean_pearsonr = 0.

                ax2.plot(r[2:], rcorr_percentiles[1][j], c=c_temp[j+1], label=vel_comps[j]+r": Pearson $\rho$")
                ax2.fill_between(r[2:], rcorr_percentiles[0][j], rcorr_percentiles[2][j], color=colours[j+1], alpha=0.2)
                
                # for smp in range(len(radii_corr)):
                    # mean_spearmanr += stats.spearmanr(radii_corr[smp][j][der][0], bar_ratio[10:])[0] / len(radii_corr)
                    # mean_pearsonr += stats.pearsonr(radii_corr[smp][j][der][1], bar_ratio[10:])[0] / len(radii_corr)
                # ax2.plot([], [], ' ', label=r"$\rho_p=$"+str(round(mean_pearsonr, 3)))

            ax2.plot(r[2:], pearsonr_data, c='k', label=r"Data: Pearson $\rho$")
            ax2.plot([], [], ' ', label=r"$\rho_p=$"+str(round(np.nanmean(pearsonr_data), 3)))

            # ax5 = ax2.twinx()
            # ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
            # ax5.plot(rad[10:], bar_ratio[10:], '--', color=color_bar, label="Vbar/Vobs")
            # ax5.tick_params(axis='y', labelcolor=color_bar)
            
            # ax2.legend(bbox_to_anchor=(1.64, 1.3))
            ax2.grid()

            plt.subplots_adjust(hspace=0.05)
            fig1.savefig(fileloc+"correlations/"+g+".png", dpi=300, bbox_inches="tight")
            plt.close()
    
    # print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
    jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem.


    #     """
    #     -----------------------------------------------------------------------
    #     Correlation plots using windows of length max{1 * Reff, 5 data points}.
    #     (Only for galaxies with Rmax > 1 * Reff)
    #     -----------------------------------------------------------------------
    #     """
    #     if len(rad) > 100:
    #         # Correlate Vobs and Vbar (d0, d1, d2) along a moving window of length 1 * Reff.
    #         # correlations_w = win_corr arrays with [ data, MOND, LCDM ], so 3 Vobs x 3 derivatives x 2 correlations each,
    #         # where win_corr = [ [[Spearman d0], [Pearson d0]], [[S d1], [P d1]], [[S d2], [P d2]] ].
    #         wmax = len(rad) - 50
    #         correlations_w = []
            
    #         for vc in range(1, 4):
    #             win_corr = [ [[], []], [[], []], [[], []] ]
    #             for der in range(3):
    #                 for j in range(50, wmax):

    #                     idx = (np.abs(r - rad[j])).argmin()
    #                     X_jmin, X_jmax = math.ceil(r[max(0, idx-2)] * 100), math.ceil(r[min(len(r)-1, idx+2)] * 100)
                        
    #                     if X_jmax - X_jmin > 100:
    #                         win_corr[der][0].append(stats.spearmanr(res_fits[der][0][X_jmin:X_jmax], res_fits[der][vc][X_jmin:X_jmax])[0])
    #                         win_corr[der][1].append(stats.pearsonr(res_fits[der][0][X_jmin:X_jmax], res_fits[der][vc][X_jmin:X_jmax])[0])
    #                     else:
    #                         jmin, jmax = j - 50, j + 50
    #                         win_corr[der][0].append(stats.spearmanr(res_fits[der][0][jmin:jmax], res_fits[der][vc][jmin:jmax])[0])
    #                         win_corr[der][1].append(stats.pearsonr(res_fits[der][0][jmin:jmax], res_fits[der][vc][jmin:jmax])[0])

    #                 # Apply SG filter to smooth out correlation curves for better visualisation.
    #                 win_corr[der][0] = signal.savgol_filter(win_corr[der][0], 50, 2)
    #                 win_corr[der][1] = signal.savgol_filter(win_corr[der][1], 50, 2)

    #             correlations_w.append(win_corr)

    #         # Compute average baryonic dominance (using Vobs from SPARC data) in moving window.
    #         wbar_ratio = []
    #         for j in range(50, wmax):
    #             wbar_ratio.append( sum( mean_prediction[0][j-50:j+50] / mean_prediction[1][j-50:j+50] ) / 101 )


    #         """
    #         Plot GP fits, residuals (+ PCHIP) and correlations.
    #         """
    #         if make_plots:
    #             subdir = "correlations/window/"

    #             # Plot corrletaions as 1 main plot + 1 subplot, using only Vobs from data for Vbar/Vobs.
    #             for der in range(3):
    #                 fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
    #                 fig1.set_size_inches(7, 7)
    #                 ax0.set_title("Moving window correlation: "+g)
    #                 ax0.set_ylabel("Velocities (km/s)")

    #                 for j in range(4):
    #                     # Scatter plot for data/mock data points.
    #                     ax0.scatter(r, vel[j], color=colours[j], alpha=0.3)
    #                     # Plot mean prediction from GP.
    #                     ax0.plot(rad, mean_prediction[j], color=colours[j], label=v_comps[j])
    #                     # Fill in 1-sigma (68%) confidence band of GP fit.
    #                     ax0.fill_between(rad, lower_percentile[j], upper_percentile[j], color=colours[j], alpha=0.2)

    #                 ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
    #                 ax0.grid()

    #                 ax1.set_ylabel(der_axis[der])
    #                 for j in range(4):
    #                     if der == 0:
    #                         ax1.scatter(r, residuals[j], color=colours[j], alpha=0.3)
    #                     ax1.plot(rad, res_fits[der][j], color=colours[j], label=v_comps[j])

    #                 ax1.grid()

    #                 ax2.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
    #                 ax2.set_ylabel("Correlations")
                    
    #                 vel_comps = [ "Data", "MOND", r"$\Lambda$CDM" ]

    #                 for j in range(3):
    #                     # Plot correlations and Vbar/Vobs.
    #                     ax2.plot(rad[50:wmax], correlations_w[j][der][0], color=colours[j+1], label=vel_comps[j]+r": Spearman $\rho$")
    #                     ax2.plot(rad[50:wmax], correlations_w[j][der][1], ':', color=colours[j+1], label=vel_comps[j]+r": Pearson $\rho$")
    #                     ax2.plot([], [], ' ', label=vel_comps[j]+r": $\rho_s=$"+str(round(stats.spearmanr(correlations_w[j][der][0], wbar_ratio)[0], 3))+r", $\rho_p=$"+str(round(stats.pearsonr(correlations_w[j][der][1], wbar_ratio)[0], 3)))

    #                 ax5 = ax2.twinx()
    #                 ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
    #                 ax5.plot(rad[50:wmax], wbar_ratio, '--', color=color_bar, label="Vbar/Vobs")
    #                 ax5.tick_params(axis='y', labelcolor=color_bar)
                    
    #                 ax2.legend(bbox_to_anchor=(1.64, 1.3))
    #                 ax2.grid()

    #                 plt.subplots_adjust(hspace=0.05)
    #                 fig1.savefig(fileloc+subdir+deriv_dir[der]+g+".png", dpi=300, bbox_inches="tight")
    #                 plt.close()


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


    # Calculate baryonic matter from data of individual galaxies.
    def Vbar(arr):
        v = np.sqrt( arr["Vgas"]**2
                    + (arr["Vdisk"]**2 * pdisk)
                    + (arr["Vbul"]**2 * pbul) )
        return v
    
    def LCDM_unc(Vbar2_unc, i_table, num_samples=num_samples):
        vDM_unc = np.array([v_DM[i_table]] * num_samples).T
        return np.sqrt(Vbar2_unc + vDM_unc**2)

    galaxies = np.load("/mnt/users/koe/gp_fits/galaxy.npy")
    galaxy_count = len(galaxies)

    if testing:
        galaxy_count = 1
        galaxies = ['NGC6946']
    elif test_multiple:
        galaxy_count = 5
        galaxies = galaxies[:galaxy_count]
    bulged_count = 0
    xbulge_count = 0
    
    # spearman_data, spearman_mock = [], []
    pearson_data, pearson_mock = [], []
    dtw_cost = [ [], [], [] ]
    norm_cost = [ [], [], [] ]
    mse_perc = [ [], [], [] ]

    for i in range(galaxy_count):
    # for i in tqdm(range(galaxy_count)):
        g = galaxies[i]
        i_tab = np.where(table["Galaxy"] == g)[0][0]

        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
        r = data["Rad"].to_numpy()

        # data_copies = np.array([data["Vobs"]] * num_samples).T

        Vbar_squared = Vbar_sq_unc(table, i_tab, data, bulged, num_samples)
        # Vbar_squared = np.array([Vbar(data)] * num_samples).T

        # full_MOND = MOND_unc(Vbar_squared)
        # full_LCDM = LCDM_unc(Vbar_squared, i_tab)
        full_MOND = Vobs_scat(MOND_unc(r, Vbar_squared, num_samples), data["errV"], num_samples)     # Assume errV completely UNcorrelated
        full_LCDM = Vobs_scat(LCDM_unc(Vbar_squared, i_tab), data["errV"], num_samples)      # Assume errV completely UNcorrelated
        # full_MOND = Vobs_scat_corr(MOND_unc(Vbar_squared), data["errV"])      # Assume errV completely correlated
        # full_LCDM = Vobs_scat_corr(LCDM_unc(Vbar_squared, i_tab), data["errV"])   # Assume errV completely correlated

        v_data = np.array([ Vbar(data), data["Vobs"], data["errV"] ])
        v_mock = np.array([ np.sqrt(Vbar_squared), full_MOND, full_LCDM ])
        # Vmax = max(velocities[1])
        # velocities /= Vmax

        if bulged:
            bulged_count += 1
        else:
            xbulge_count += 1

        # print("\nAnalyzing galaxy "+g+" ("+str(i+1)+"/60)")
        main(g, r, v_data, v_mock)

    print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)


    """
    --------------
    SUMMARY PLOTS.
    --------------
    """
    if make_plots and not testing:
        """
        Plot histogram of MSE (in ascending order of data).
        """
        if use_MSE:
            mse_argsort = np.argsort( mse_perc[0] )
            print(f"Galaxies in ascending order of mse(data): {np.array(galaxies)[mse_argsort]}")

            plt.title("Mean-squared errors relative to GP fits")
            plt.ylabel("Mean-squared errors")
            plt.xlabel("Galaxies")

            mse_MOND_perc, mse_LCDM_perc = np.array(mse_perc[1])[mse_argsort], np.array(mse_perc[2])[mse_argsort]
            mse_MOND_errors = [ mse_MOND_perc[:,2] - mse_MOND_perc[:,1], mse_MOND_perc[:,1] - mse_MOND_perc[:,0] ]
            mse_LCDM_errors = [ mse_LCDM_perc[:,2] - mse_LCDM_perc[:,1], mse_LCDM_perc[:,1] - mse_LCDM_perc[:,0] ]

            plt.bar(galaxies[mse_argsort], np.array(mse_perc[0])[mse_argsort], color='k', alpha=0.3, label="Data")
            plt.errorbar(galaxies[mse_argsort], mse_MOND_perc[:,1], mse_MOND_errors, fmt='.', ls='none',
                         capsize=2, color='mediumblue', alpha=0.5, label="MOND")
            plt.errorbar(galaxies[mse_argsort], mse_LCDM_perc[:,1], mse_LCDM_errors, fmt='.', ls='none',
                         capsize=2, color='tab:green', alpha=0.5, label=r"$\Lambda$CDM")

            plt.legend()
            plt.xticks([])
            # plt.yscale('log')
            plt.savefig(fileloc+"MSE.png", dpi=300, bbox_inches="tight")
            plt.close()


        """
        Plot histogram of normalized DTW costs (in ascending order of costs for data).
        """
        if use_DTW:
            # dim = (3 x v_comps, galaxy_count, num_samples)
            dtw_cost = np.array(dtw_cost)
            norm_cost = np.array(norm_cost)

            # Arrays of shape (5 x percentiles, 3 x v_comps, galaxy_count).
            norm_percentiles = np.percentile(norm_cost, [5.0, 16.0, 50.0, 84.0, 95.0], axis=2)
            dtw_percentiles = np.percentile(dtw_cost, [5.0, 16.0, 50.0, 84.0, 95.0], axis=2)

            # Rearrange galaxies into ascending order in median of data normalised costs.
            sort_args = np.argsort(norm_percentiles[2][0])
            norm_percentiles = norm_percentiles[:, :, sort_args]

            print(f"Galaxies in ascending order of cost(data): {np.array(galaxies)[sort_args]}")

            # Plot histogram of normalized DTW alignment costs of all galaxies.
            # if fname_DTW == fileloc+"dtw/cost_vsLCDM/": plt.title(r"Normalized DTW alignment costs (relative to $\Lambda$CDM)")
            # elif fname_DTW == fileloc+"dtw/cost_vsVbar/": plt.title("Normalized DTW alignment costs (relative to Vbar)")
            # else: plt.title("Normalized DTW alignment costs (relative to MOND)")

            hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
            colours = [ 'k', 'mediumblue', 'tab:green' ]

            # if fname_DTW == fileloc+"dtw/cost_vsVbar/":
            plt.bar(galaxies, norm_percentiles[2][0], color=colours[0], alpha=0.3, label=hist_labels[0])

            for j in range(3):
                # if fname_DTW == fileloc+"dtw/":
                #     if j == 1: continue     # Only plot values for data and LCDM since cost(MOND) == 0.
                # elif fname_DTW == fileloc+"dtw/cost_vsLCDM/":
                #     if j == 2: continue     # Only plot values for data and MOND since cost(LCDM) == 0.
                mean_norm = np.nanmean(norm_percentiles[2][j])
                low_err = norm_percentiles[2][j] - norm_percentiles[1][j]
                up_err = norm_percentiles[3][j] - norm_percentiles[2][j]

                if j != 0:
                    low_norm1 = np.full(galaxy_count, np.nanmean(norm_percentiles[1][j]))
                    # low_norm2 = np.full(galaxy_count, np.nanmean(norm_percentiles[0][j]))
                    up_norm1 = np.full(galaxy_count, np.nanmean(norm_percentiles[3][j]))
                    # up_norm2 = np.full(galaxy_count, np.nanmean(norm_percentiles[4][j]))
                    plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j], alpha=0.25)
                    # plt.fill_between(galaxies, low_norm2, up_norm2, color=colours[j], alpha=0.1)

                plt.axhline(y=mean_norm, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_norm))
                # if not(fname_DTW == fileloc+"dtw/cost_vsVbar/" and j == 0):
                if j != 0:
                    plt.errorbar(galaxies, norm_percentiles[2][j], [low_err, up_err], fmt='.', ls='none',
                                capsize=2, color=colours[j], alpha=0.5, label=hist_labels[j])
                            
            plt.legend()
            plt.xticks([])
            plt.savefig(fname_DTW+"histo1.png", dpi=300, bbox_inches="tight")
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

            print(f"Galaxies in descending order of cost(LCDM) - cost(data): {np.array(galaxies)[sort_args]}")

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
            plt.close()

        """
        Plot histogram of Spearman coefficients across RC (in ascending order of coefficients for data).
        """
        if do_correlations:
            # """Spearman histogram"""
            # # Rearrange galaxies into ascending order in median of corr(MOND, Vbar).
            # # dim = (# of galaxies, 2 x mock_vcomps, 3 x percentiles)
            # mock_sorted = np.array(sorted(spearman_mock, key=lambda x: x[0][0]))

            # plt.title("Spearman coefficients across RC")
            # hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
            # colours = [ 'k', 'mediumblue', 'tab:green' ]

            # mean_corr = np.nanmean(spearman_data)
            # plt.bar(galaxies, sorted(spearman_data), color=colours[0], alpha=0.3, label=hist_labels[0])
            # plt.axhline(y=mean_corr, color=colours[0], linestyle='dashed', label="Mean = {:.4f}".format(mean_corr))

            # for j in range(2):
            #     med_corr = np.nanmean(mock_sorted[:,j,1])
            #     low_err = mock_sorted[:,j,1] - mock_sorted[:,j,0]
            #     up_err = mock_sorted[:,j,2] - mock_sorted[:,j,1]

            #     low_norm1 = np.full(galaxy_count, np.nanmean(mock_sorted[:,j,2]))
            #     up_norm1 = np.full(galaxy_count, np.nanmean(mock_sorted[:,j,0]))

            #     plt.errorbar(galaxies, mock_sorted[:,j,1], [low_err, up_err], fmt='.', ls='none', capsize=2, color=colours[j+1], alpha=0.5, label=hist_labels[j+1])
            #     plt.axhline(y=med_corr, color=colours[j+1], linestyle='dashed', label="Mean = {:.4f}".format(med_corr))
            #     plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j+1], alpha=0.25)
            
            # plt.legend()
            # plt.xticks([])
            # plt.savefig(fileloc+"correlations/radii/histo1.png", dpi=300, bbox_inches="tight")
            # plt.close()

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
            plt.savefig(fileloc+"correlations/histo1.png", dpi=300, bbox_inches="tight")
            plt.close()

    print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
