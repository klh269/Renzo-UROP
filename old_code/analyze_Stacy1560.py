#!/usr/bin/env python
"""
Correlation coefficients + dynamic time warping on GP residuals of NGC1560_Stacy,
a dataset obtained from digitizing Stacy's NGC 1560 plot with Plot Digitizer;
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
from scipy import interpolate, stats
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
from utils_analysis.mock_gen import Vbar_sq_unc, MOND_unc, Vobs_scat
from utils_analysis.extract_ft import ft_check

matplotlib.use("Agg")


make_plots = True
do_DTW = False
do_correlations = True

fileloc = "/mnt/users/koe/plots/NGC1560/plot_digitizer/"   # Directory for saving plots.
num_samples = 500


# Main code to run.
def main(args, g, r, rad, Y, v_data, v_mock, num_samples=num_samples):
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

    # "Raw" percentiles from uncertainties and scattering.
    raw_median = np.percentile(v_mock, 50.0, axis=2)                # dim = (3, r)
    raw_percentiles = np.percentile(v_mock, [16.0, 84.0], axis=2)   # dim = (2, 3, r)
    raw_errors = np.abs( raw_percentiles - raw_median )             # dim = (2, 3, r)

    # Compute residuals of fits.
    res_Vbar_data, res_Vobs, res_Vbar_mock, res_MOND, res_LCDM = [], [] ,[], [], []
    for k in range(len(r)):
        idx = (np.abs(rad - r[k])).argmin()
        
        res_Vbar_data.append(v_data[1][k] - mean_prediction[0][idx])
        res_Vobs.append(v_data[0][k] - mean_prediction[3][idx])

        res_Vbar_mock.append(v_mock[0][k] - mean_prediction[3][idx])
        res_MOND.append(v_mock[1][k] - mean_prediction[1][idx])
        res_LCDM.append(v_mock[2][k] - mean_prediction[2][idx])

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
        dtw_cost_smp = [ [], [], [] ]
        norm_cost_smp = [ [], [], [] ]

        for smp in tqdm(range(num_samples), desc="DTW"):
            # Construct distance matrices.
            dist_data = np.zeros((len(r), len(r)))
            dist_MOND = np.copy(dist_data)
            dist_LCDM = np.copy(dist_data)
            
            for n in range(len(r)):
                for m in range(len(r)):
                    # Construct distance matrix such that cost = 0 if Vobs = MOND(Vbar).
                    dist_data[n, m] = np.abs(res_Vobs[n] - res_MOND[m][smp])
                    dist_MOND[n, m] = np.abs(res_MOND[n][smp] - res_MOND[m][smp])
                    dist_LCDM[n, m] = np.abs(res_LCDM[n][smp] - res_MOND[m][smp])
            
            dist_mats = np.array([ dist_data, dist_MOND, dist_LCDM ])
            # mats_dir = [ "data/", "MOND/", "LCDM/" ]
            
            # DTW!
            for j in range(3):
                path, cost_mat = dtw(dist_mats[j])
                # x_path, y_path = zip(*path)
                cost = cost_mat[ len(r)-1, len(r)-1 ]
                dtw_cost_smp[j].append(cost)
                norm_cost_smp[j].append(cost / (2 * len(r)))
        
        for j in range(3):
            dtw_cost[j].append(dtw_cost_smp[j])
            norm_cost[j].append(norm_cost_smp[j])


    """
    Code for PCHIP + correlations on GP residuals.
    """
    if do_correlations:

        # Interpolate the residuals with cubic Hermite spline splines.
        v_d0, v_d1, v_d2 = [], [], []
        for v_comp in res_data:
            v_d0.append(interpolate.pchip_interpolate(r, v_comp, rad))
            v_d1.append(interpolate.pchip_interpolate(r, v_comp, rad, der=1))
            v_d2.append(interpolate.pchip_interpolate(r, v_comp, rad, der=2))
        
        res_fits_data = [ v_d0, v_d1, v_d2 ]

        # Compute correlation coefficients for data Vobs vs Vbar.
        rcorr_data = [ [[], []], [[], []], [[], []] ]
        for k in range(3):
            for j in range(10, len(rad)):
                rcorr_data[k][0].append(stats.spearmanr(res_fits_data[k][0][:j], res_fits_data[k][1][:j])[0])
                rcorr_data[k][1].append(stats.pearsonr(res_fits_data[k][0][:j], res_fits_data[k][1][:j])[0])
        
        spearman_data.append(rcorr_data[0][0][-1])
        pearson_data.append(rcorr_data[0][1][-1])

        # Compute correlation coefficients for mock Vobs vs Vbar.
        radii_corr = []     # dim = (num_samples/10, 2 x mock_vcomps, 3 x der, 2 x rho, rad)
        res_fits_mock = []

        # for smp in range(num_samples):
        for smp in tqdm(range(num_samples), desc="Correlation by radii"):
            if smp % 10:
                continue

            # Interpolate the residuals with cubic Hermite spline splines.
            v_d0, v_d1, v_d2 = [], [], []
            for v_comp in res_mock[:, :, smp]:
                v_d0.append(interpolate.pchip_interpolate(r, v_comp, rad))
                v_d1.append(interpolate.pchip_interpolate(r, v_comp, rad, der=1))
                v_d2.append(interpolate.pchip_interpolate(r, v_comp, rad, der=2))
            
            res_fits = [ v_d0, v_d1, v_d2 ]
            res_fits_mock.append(res_fits)

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
                rad_corr = [ [[], []], [[], []], [[], []] ]
                for k in range(3):
                    for j in range(10, len(rad)):
                        rad_corr[k][0].append(stats.spearmanr(res_fits[k][0][:j], res_fits[k][i][:j])[0])
                        rad_corr[k][1].append(stats.pearsonr(res_fits[k][0][:j], res_fits[k][i][:j])[0])
                correlations_r.append(rad_corr)
            
            radii_corr.append(correlations_r)
        
        res_fits_percentiles = np.percentile(res_fits_mock, [16.0, 50.0, 84.0], axis=0)
        rcorr_percentiles = np.percentile(radii_corr, [16.0, 50.0, 84.0], axis=0)
        spearman_mock.append([ rcorr_percentiles[:,0,0,0,-1], rcorr_percentiles[:,1,0,0,-1] ])
        pearson_mock.append([ rcorr_percentiles[:,0,0,1,-1], rcorr_percentiles[:,1,0,1,-1] ])

        """
        Plot GP fits, residuals (+ PCHIP) and correlations.
        """
        if make_plots:
            subdir = "corr_radii/"
            color_bar = "orange"
            deriv_dir = [ "d0", "d1", "d2" ]
            c_temp = [ 'tab:red', 'mediumblue', 'tab:green' ]
            labels_temp = [ "Vbar (SPARC)", "Vobs (MOND)", r"Vobs ($\Lambda$CDM)", "Vobs (SPARC)" ]

            # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
            bar_ratio = []
            for rd in range(len(rad)):
                bar_ratio.append(sum(mean_prediction[0][:rd]/mean_prediction[3][:rd]) / (rd+1))

            # Plot corrletaions as 1 main plot (+ residuals) + 1 subplot, using only Vobs from data for Vbar/Vobs.
            der_axis = [ "Residuals (km/s)", "1st derivative", "2nd derivative" ]

            """Spearman correlations"""
            for der in range(3):
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
                    ax0.fill_between(rad, percentiles[j][0], percentiles[j][1], color=colours[j], alpha=0.2)

                ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
                ax0.grid()

                ax1.set_ylabel(der_axis[der])
                for j in range(4):
                    # Plots for mock Vobs + Vbar (sampled w/ uncertainties).
                    if j == 3:
                        if der == 0:
                            ax1.errorbar(r, res_data[1], Y[4], color='k', alpha=0.3, ls='none', fmt='o', capsize=2)
                        ax1.plot(rad, res_fits_data[der][1], color='k', label=labels_temp[j])
                    else:
                        if der == 0:
                            ax1.scatter(r, res_median[j], c=c_temp[j], alpha=0.3)
                            # ax1.errorbar(r, res_median[j], res_errors[:, j], color=colours[j], alpha=0.3, ls='none', fmt='o', capsize=2)
                        ax1.plot(rad, res_fits_percentiles[1][der][j], c=c_temp[j], label=labels_temp[j])
                        ax1.fill_between(rad, res_fits_percentiles[0][der][j], res_fits_percentiles[2][der][j], color=c_temp[j], alpha=0.15)

                ax1.grid()

                ax2.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
                ax2.set_ylabel("Correlations w/ Vbar")
                
                vel_comps = [ "MOND", r"$\Lambda$CDM", "Data" ]

                for j in range(2):
                    mean_spearmanr = 0.
                    mean_pearsonr = 0.

                    ax2.plot(rad[10:], rcorr_percentiles[1][j][der][0], c=c_temp[j+1], label=vel_comps[j]+r": Spearman $\rho$")
                    ax2.fill_between(rad[10:], rcorr_percentiles[0][j][der][0], rcorr_percentiles[2][j][der][0], color=colours[j+1], alpha=0.2)
                    
                    for smp in range(len(radii_corr)):
                        mean_spearmanr += stats.spearmanr(radii_corr[smp][j][der][0], bar_ratio[10:])[0] / len(radii_corr)
                        mean_pearsonr += stats.pearsonr(radii_corr[smp][j][der][1], bar_ratio[10:])[0] / len(radii_corr)
                    ax2.plot([], [], ' ', label=r": $\rho_s=$"+str(round(mean_spearmanr, 3)))

                ax2.plot(rad[10:], rcorr_data[der][0], c='k', label=vel_comps[2]+r": Spearman $\rho$")
                ax2.plot([], [], ' ', label=r": $\rho_s=$"+str(round(np.nanmean(rcorr_data[der][0]), 3)))

                ax5 = ax2.twinx()
                ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
                ax5.plot(rad[10:], bar_ratio[10:], '--', color=color_bar, label="Vbar/Vobs")
                ax5.tick_params(axis='y', labelcolor=color_bar)
                
                ax2.legend(bbox_to_anchor=(1.64, 1.3))
                ax2.grid()

                plt.subplots_adjust(hspace=0.05)
                fig1.savefig(fileloc+subdir+deriv_dir[der]+".png", dpi=300, bbox_inches="tight")
                plt.close()

            """Pearson correlations."""
            for der in range(3):
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
                    ax0.fill_between(rad, percentiles[j][0], percentiles[j][1], color=colours[j], alpha=0.2)

                ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
                ax0.grid()

                ax1.set_ylabel(der_axis[der])
                for j in range(4):
                    # Plots for mock Vobs + Vbar (sampled w/ uncertainties).
                    if j == 3:
                        if der == 0:
                            ax1.errorbar(r, res_data[1], Y[4], color='k', alpha=0.3, ls='none', fmt='o', capsize=2)
                        ax1.plot(rad, res_fits_data[der][1], color='k', label=labels_temp[j])
                    else:
                        if der == 0:
                            ax1.scatter(r, res_median[j], c=c_temp[j], alpha=0.3)
                            # ax1.errorbar(r, res_median[j], res_errors[:, j], color=colours[j], alpha=0.3, ls='none', fmt='o', capsize=2)
                        ax1.plot(rad, res_fits_percentiles[1][der][j], c=c_temp[j], label=labels_temp[j])
                        ax1.fill_between(rad, res_fits_percentiles[0][der][j], res_fits_percentiles[2][der][j], color=c_temp[j], alpha=0.15)

                ax1.grid()

                ax2.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
                ax2.set_ylabel("Correlations w/ Vbar")
                
                vel_comps = [ "MOND", r"$\Lambda$CDM", "Data" ]

                for j in range(2):
                    mean_spearmanr = 0.
                    mean_pearsonr = 0.

                    ax2.plot(rad[10:], rcorr_percentiles[1][j][der][1], c=c_temp[j+1], label=vel_comps[j]+r": Pearson $\rho$")
                    ax2.fill_between(rad[10:], rcorr_percentiles[0][j][der][1], rcorr_percentiles[2][j][der][0], color=colours[j+1], alpha=0.2)
                    
                    for smp in range(len(radii_corr)):
                        mean_spearmanr += stats.spearmanr(radii_corr[smp][j][der][0], bar_ratio[10:])[0] / len(radii_corr)
                        mean_pearsonr += stats.pearsonr(radii_corr[smp][j][der][1], bar_ratio[10:])[0] / len(radii_corr)
                    ax2.plot([], [], ' ', label=r": $\rho_s=$"+str(round(mean_spearmanr, 3))+r", $\rho_p=$"+str(round(mean_pearsonr, 3)))

                ax2.plot(rad[10:], rcorr_data[der][1], c='k', label=vel_comps[2]+r": Pearson $\rho$")
                ax2.plot([], [], ' ', label=r": $\rho_s=$"+str(round(np.nanmean(rcorr_data[der][0]), 3))+r", $\rho_p=$"+str(round(np.nanmean(rcorr_data[der][1]), 3)))

                ax5 = ax2.twinx()
                ax5.set_ylabel(r'Average $v_{bar}/v_{obs}$')
                ax5.plot(rad[10:], bar_ratio[10:], '--', color=color_bar, label="Vbar/Vobs")
                ax5.tick_params(axis='y', labelcolor=color_bar)
                
                ax2.legend(bbox_to_anchor=(1.64, 1.3))
                ax2.grid()

                plt.subplots_adjust(hspace=0.05)
                fig1.savefig(fileloc+subdir+"pearson/"+deriv_dir[der]+".png", dpi=300, bbox_inches="tight")
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

    
    galaxy, spearman_mock, pearson_mock, spearman_data, pearson_data = [], [], [], [], []
    dtw_cost = [ [], [], [] ]
    norm_cost = [ [], [], [] ]

    """
    Plotting galaxy rotation curves directly from data with variables:
    Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
    """
    file_path = "/mnt/users/koe/data/NGC1560_Stacy.dat"
    rawdata = np.loadtxt(file_path)
    columns = [ "Rad", "Vobs", "Vgas", "Vdisk", "errV" ]

    data = pd.DataFrame(rawdata, columns=columns)
    r = data["Rad"]
    # r /= 1.3    # Scale length of NGC 1560 according to Broeils.
    bulged = False

    table = { "D":[2.99], "e_D":[0.1], "Inc":[82.0], "e_Inc":[1.0], "L":[23.8], "e_L":[1.8] }
    i_table = 0

    # Normalise velocities by Vmax = max(Vobs) from SPARC data.
    Vbar_squared = Vbar_sq_unc(table, i_table, data, bulged, num_samples)
    nfw_samples = Vobs_MCMC(table, i_table, data, bulged, profile="NFW")
    mond_samples = Vobs_MCMC(table, i_table, data, bulged, profile="MOND")
    # v_LCDM = np.median(nfw_samples["Vpred"], axis=0)
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

    # Get v_DM from Fedir's LCDM abundance matching.
    v_DM = np.load("/mnt/users/koe/v_DM/NGC1560.npy")

    def LCDM_unc(Vbar2_unc, num_samples=num_samples):
        vDM_unc = np.array([v_DM] * num_samples).T
        return np.sqrt(Vbar2_unc + vDM_unc**2)
    
    Vbar2 = Vbar_sq(data, bulged)
    v_LCDM = np.sqrt(Vbar2 + np.array(v_DM)**2)
    
    v_components = np.array([ data["Vobs"], v_MOND, v_LCDM, np.sqrt(Vbar2), data["errV"]])
    # Vmax = max(v_components[0])
    # data["errV"] /= Vmax
    # v_components /= Vmax

    rad_count = math.ceil((max(r)-min(r))*100)
    rad = np.linspace(min(r), max(r), rad_count)

    full_MOND = Vobs_scat(MOND_unc(r, Vbar_squared, num_samples), data["errV"], num_samples)    # Assume errV completely UNcorrelated
    full_LCDM = Vobs_scat(LCDM_unc(Vbar_squared), data["errV"], num_samples)                # Assume errV completely UNcorrelated

    v_data = np.array([ np.sqrt(Vbar2), data["Vobs"] ])
    v_mock = np.array([ np.sqrt(Vbar_squared), full_MOND, full_LCDM ])

    main(args, "NGC1560 (Stacy)", r.to_numpy(), rad, v_components, v_data, v_mock)


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
        norm_percentiles = np.percentile(norm_cost, [5.0, 16.0, 50.0, 84.0, 95.0], axis=2)
        dtw_percentiles = np.percentile(dtw_cost, [5.0, 16.0, 50.0, 84.0, 95.0], axis=2)

        # Rearrange galaxies into ascending order in median of data normalised costs.
        sort_args = np.argsort(norm_percentiles[2][0])
        norm_percentiles = norm_percentiles[:, :, sort_args]

        galaxies = [ "NGC1560" ]
        galaxy_count = 1
        
        # Plot histogram of normalized DTW alignment costs of all galaxies.
        plt.title("Normalized DTW alignment costs (relative to MOND)")
        hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
        colours = [ 'k', 'mediumblue', 'tab:green' ]

        # plt.scatter(galaxies, norm_cost[0], c=colours[0], alpha=0.5, label=hist_labels[0])
        mean_norm = np.nanmean(norm_percentiles[2][0])
        plt.bar(galaxies, norm_percentiles[2][0], color=colours[0], alpha=0.3, label=hist_labels[0])
        plt.axhline(y=mean_norm, color=colours[0], linestyle='dashed', label="Mean = {:.4f}".format(mean_norm))

        # for j in range(1, 3):
        jj = 2  # Only plot values for LCDM since cost(MOND) == 0.
        mean_norm = np.nanmean(norm_percentiles[2][jj])
        low_err = norm_percentiles[2][jj] - norm_percentiles[1][jj]
        up_err = norm_percentiles[3][jj] - norm_percentiles[2][jj]

        low_norm1 = np.full(galaxy_count, np.nanmean(norm_percentiles[1][jj]))
        # low_norm2 = np.full(galaxy_count, np.nanmean(norm_percentiles[0][jj]))
        up_norm1 = np.full(galaxy_count, np.nanmean(norm_percentiles[3][jj]))
        # up_norm2 = np.full(galaxy_count, np.nanmean(norm_percentiles[4][jj]))

        plt.errorbar(galaxies, norm_percentiles[2][jj], [low_err, up_err], fmt='.', ls='none',
                        capsize=2, color=colours[jj], alpha=0.5, label=hist_labels[jj])
        plt.axhline(y=mean_norm, color=colours[jj], linestyle='dashed', label="Mean = {:.4f}".format(mean_norm))
        plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[jj], alpha=0.25)
        # plt.fill_between(galaxies, low_norm2, up_norm2, color=colours[jj], alpha=0.1)
        
        plt.legend()
        plt.xticks([])
        plt.savefig(fileloc+"dtw/histo1.png", dpi=300, bbox_inches="tight")
        # plt.savefig(fileloc+"dtw/corr_scat/histo1.png", dpi=300, bbox_inches="tight")
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
            plt.savefig(fileloc+"dtw/scatter_"+plotloc[j-1]+".png", dpi=300, bbox_inches="tight")
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
        plt.savefig(fileloc+"dtw/histo2.png", dpi=300, bbox_inches="tight")
        # plt.savefig(fileloc+"dtw/corr_scat/histo2.png", dpi=300, bbox_inches="tight")
        plt.close()

    """
    Plot histogram of Spearman coefficients across RC (in ascending order of coefficients for data).
    """
    if do_correlations:
        """Spearman histogram"""
        # Rearrange galaxies into ascending order in median of corr(MOND, Vbar).
        # dim = (# of galaxies, 2 x mock_vcomps, 3 x percentiles)
        mock_sorted = np.array(sorted(spearman_mock, key=lambda x: x[0][0]))

        plt.title("Spearman coefficients across RC")
        hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
        colours = [ 'k', 'mediumblue', 'tab:green' ]

        mean_corr = np.nanmean(spearman_data)
        plt.bar(galaxies, sorted(spearman_data), color=colours[0], alpha=0.3, label=hist_labels[0])
        plt.axhline(y=mean_corr, color=colours[0], linestyle='dashed', label="Mean = {:.4f}".format(mean_corr))

        for j in range(2):
            med_corr = np.nanmean(mock_sorted[:,j,1])
            low_err = mock_sorted[:,j,1] - mock_sorted[:,j,0]
            up_err = mock_sorted[:,j,2] - mock_sorted[:,j,1]

            low_norm1 = np.full(galaxy_count, np.nanmean(mock_sorted[:,j,2]))
            up_norm1 = np.full(galaxy_count, np.nanmean(mock_sorted[:,j,0]))

            plt.errorbar(galaxies, mock_sorted[0,j,1], [low_err, up_err], fmt='.', ls='none', capsize=2, color=colours[j+1], alpha=0.5, label=hist_labels[j+1])
            plt.axhline(y=med_corr, color=colours[j+1], linestyle='dashed', label="Mean = {:.4f}".format(med_corr))
            plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j], alpha=0.25)
        
        plt.legend()
        plt.xticks([])
        plt.savefig(fileloc+"corr_radii/histo1.png", dpi=300, bbox_inches="tight")
        plt.close()

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
            plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j], alpha=0.25)
        
        plt.legend()
        plt.xticks([])
        plt.savefig(fileloc+"corr_radii/pearson/histo1.png", dpi=300, bbox_inches="tight")
        plt.close()

print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)

