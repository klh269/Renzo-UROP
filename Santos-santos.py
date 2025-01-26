#!/usr/bin/env python
"""
Gaussian process + correlation statistics on simulated galaxies (Santos-Santos),
with MOND generated using Milgrom's formula w/ simple IF.
NOTE: Full run with GP fits + analyses requires upwards of 230 GB!
GP fits are stored in /mnt/users/koe/GP_fits/Santos-Santos/.
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
from numpyro.infer import MCMC, NUTS, init_to_median

import corner
import numpyro

from utils_analysis.gp_utils import model, predict, run_inference
from utils_analysis.dtw_utils import dtw
from utils_analysis.Vobs_fits import MOND_vsq, NFW_fit, BIC_from_samples

matplotlib.use("Agg")  # noqa: E402


testing = True
make_plots = True
do_DTW = True
do_correlations = True

fileloc = "/mnt/users/koe/plots/Santos-Santos/"


# Main code to run.
def main(args, g, X, Y, X_test): 
    """
    Do inference for Vbar with uniform prior for correlation length,
    then apply the resulted lengthscale to Vobs (both real and mock data).
    """
    v_comps = [ r"$v_{bar}$", r"$v_{obs}$", r"$v_{MOND}$", r"$V_{\Lambda CDM}$" ]
    colours = [ 'tab:red', 'k', 'mediumblue', 'tab:green' ]
    corner_dir = [ "Vbar/", "Vobs_data/", "Vobs_MOND/", "Vobs_LCDM/" ]
    mean_prediction = []
    percentiles = []

    # GP on Vbar with uniform prior on length.
    print("Fitting function to " + v_comps[0] + "...")
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    if testing:
        # Remove feature from Vobs before fitting GP.
        X_Xft, Vobs_Xft = np.delete(X, np.s_[10:15], axis=0), np.delete(Y[0], np.s_[10:15], axis=0)
        samples = run_inference(model, args, rng_key, X_Xft, Vobs_Xft)
    else:
        samples = run_inference(model, args, rng_key, X, Y[0])

    # do prediction
    vmap_args = (
        random.split(rng_key_predict, samples["var"].shape[0]),
        samples["var"],
        samples["length"],
        samples["noise"],
    )
    if testing:
        means, predictions = vmap(
            lambda rng_key, var, length, noise: predict(
                rng_key, X_Xft, Vobs_Xft, X_test, var, length, noise, use_cholesky=args.use_cholesky
            )
        )(*vmap_args)
    else:
        means, predictions = vmap(
            lambda rng_key, var, length, noise: predict(
                rng_key, X, Y[0], X_test, var, length, noise, use_cholesky=args.use_cholesky
            )
        )(*vmap_args)

    mean_pred = np.mean(means, axis=0)
    mean_prediction.append(mean_pred)
    gp_predictions[0] = mean_pred

    confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
    percentiles.append(confidence_band)
    gp_16percent[0] = confidence_band[0]
    gp_84percent[0] = confidence_band[1]

    if make_plots:
        labels = ["length", "var", "noise"]
        samples_arr = np.vstack([samples[label] for label in labels]).T
        fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
        fig.savefig(fileloc+"corner_plots/"+corner_dir[0]+g+".png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # GP on Vobs with fixed hyperparameters from Vbar.
    vr = stats.mode(np.round(samples["var"], 0))[0]
    print(f"var MAP = {vr}")
    ls = stats.mode(np.round(samples["length"], 2))[0]
    print(f"length MAP = {ls}")
    ns = stats.mode(np.round(samples["noise"], 1))[0]
    print(f"noise MAP = {ns}")

    for j in range(1, 4):
        print("\nFitting function to " + v_comps[j] + " with length = " + str(round(ls, 2)) + "...")
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))

        # do prediction
        vmap_args = (
            random.split(rng_key_predict, samples["var"].shape[0]),
        )
        if testing:
            Vcomp_Xft = np.delete(Y[j], np.s_[10:15], axis=0)
            means, predictions = vmap(
                lambda rng_key: predict(
                    rng_key, X_Xft, Vcomp_Xft, X_test, vr, ls, ns, use_cholesky=args.use_cholesky
                )
            )(*vmap_args)
        else:
            means, predictions = vmap(
                lambda rng_key: predict(
                    rng_key, X, Y[j], X_test, vr, ls, ns, use_cholesky=args.use_cholesky
                )
            )(*vmap_args)

        mean_pred = np.mean(means, axis=0)
        mean_prediction.append(mean_pred)
        gp_predictions[j] = mean_pred

        confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
        percentiles.append(confidence_band)
        gp_16percent[j] = confidence_band[0]
        gp_84percent[j] = confidence_band[1]

        if make_plots:
            labels = ["var", "noise"]
            samples_arr = np.vstack([samples[label] for label in labels]).T
            fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".5f", quantiles=[0.16, 0.5, 0.84], smooth=1)
            fig.savefig(fileloc+"corner_plots/"+corner_dir[j]+g+".png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    
    # Compute residuals of fits.
    res_Vbar, res_Vobs, res_MOND, res_LCDM = [], [] ,[], []
    for k in range(len(X)):
        idx = (np.abs(X_test - X[k])).argmin()
        res_Vbar.append(Y[0][k] - mean_prediction[0][idx])
        res_Vobs.append(Y[1][k] - mean_prediction[1][idx])
        res_MOND.append(Y[2][k] - mean_prediction[2][idx])
        res_LCDM.append(Y[3][k] - mean_prediction[3][idx])
    residuals = np.array([ res_Vbar, res_Vobs, res_MOND, res_LCDM ])


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
        
        res_fits = [ v_d0, v_d1, v_d2 ]

        """
        ---------------------------------------------------
        Correlation plots using sphers of increasing radius
        ---------------------------------------------------
        """
        if testing:
            print("Correlating coefficients by max radii...")

        # Correlate Vobs and Vbar (d0, d1, d2) as a function of (maximum) radius, i.e. spheres of increasing r.
        # correlations_r = rad_corr arrays with [ data, MOND ], so 3 Vobs x 3 derivatives x 2 correlations each,
        # where rad_corr = [ [[Spearman d0], [Pearson d0]], [[S d1], [P d1]], [[S d2], [P d2]] ].
        correlations_r = []
        for i in range(1, 4):
            rad_corr = [ [[], []], [[], []], [[], []] ]
            for k in range(3):
                for j in range(10, len(X_test)):
                    rad_corr[k][0].append(stats.spearmanr(res_fits[k][0][:j], res_fits[k][i][:j])[0])
                    rad_corr[k][1].append(stats.pearsonr(res_fits[k][0][:j], res_fits[k][i][:j])[0])
            correlations_r.append(rad_corr)

        spearman_data.append(np.array(correlations_r)[:,0,0,-1])
        pearson_data.append(np.array(correlations_r)[:,0,1,-1])

        """
        Plot GP fits, residuals (+ PCHIP) and correlations.
        """
        color_bar = "orange"
        deriv_dir = [ "d0/", "d1/", "d2/" ]
        subdir = "correlations/radii/"

        # Compute baryonic dominance, i.e. average Vbar/Vobs(data) from centre to some max radius.
        bar_ratio = []
        for rd in range(len(X_test)):
            bar_ratio.append(sum(mean_prediction[0][:rd]/mean_prediction[1][:rd]) / (rd+1))

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
                    ax0.plot(X_test, mean_prediction[j], color=colours[j], label=v_comps[j])
                    # Fill in 1-sigma (68%) confidence band of GP fit.
                    ax0.fill_between(X_test, percentiles[j][0, :], percentiles[j][1, :], color=colours[j], alpha=0.2)

                ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
                ax0.grid()

                ax1.set_ylabel(der_axis[der])
                for j in range(4):
                    if der == 0:
                        ax1.scatter(r, residuals[j], color=colours[j], alpha=0.3)
                    ax1.plot(X_test, res_fits[der][j], color=colours[j], label=v_comps[j])

                ax1.grid()

                ax2.set_xlabel('Radii (kpc)')
                ax2.set_ylabel("Correlations")
                
                vel_comps = [ "Data", "MOND", r"$\Lambda$CDM" ]

                for j in range(3):
                    # Plot correlations and Vbar/Vobs.
                    ax2.plot(X_test[10:], correlations_r[j][der][0], color=colours[j+1], label=vel_comps[j]+r": Spearman $\rho$")
                    ax2.plot(X_test[10:], correlations_r[j][der][1], ':', color=colours[j+1], label=vel_comps[j]+r": Pearson $\rho$")
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
            # correlations_w = win_corr arrays with [ data, MOND ], so 2 Vobs x 3 derivatives x 2 correlations each,
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

            # Compute average baryonic dominance (using Vobs from data) in moving window.
            wbar_ratio = []
            for j in range(50, wmax):
                wbar_ratio.append( sum( mean_prediction[0][j-50:j+50] / mean_prediction[1][j-50:j+50] ) / 101 )


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
                        ax0.plot(X_test, mean_prediction[j], color=colours[j], label=v_comps[j])
                        # Fill in 1-sigma (68%) confidence band of GP fit.
                        ax0.fill_between(X_test, percentiles[j][0, :], percentiles[j][1, :], color=colours[j], alpha=0.2)

                    ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
                    ax0.grid()

                    ax1.set_ylabel(der_axis[der])
                    for j in range(4):
                        if der == 0:
                            ax1.scatter(r, residuals[j], color=colours[j], alpha=0.3)
                        ax1.plot(X_test, res_fits[der][j], color=colours[j], label=v_comps[j])

                    ax1.grid()

                    ax2.set_xlabel('Radii (kpc)')
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
        choices=["median", "feasible", "value", "uniform", "sample"],
    )
    parser.add_argument("--no-cholesky", dest="use_cholesky", action="store_false")
    parser.add_argument("--testing", default=testing, type=bool)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)


    # Define constants
    G = 4.300e-6    # Gravitational constant G (kpc (km/s)^2 solarM^(-1))
    
    spearman_data, pearson_data = [], []
    dtw_cost = [ [], [], [] ]
    norm_cost = [ [], [], [] ]

    galaxies = [ "g1536_Irr", "g1536_MW", "g5664_Irr", "g5664_MW", "g7124_Irr", "g7124_MW",
                 "g15784_Irr", "g15784_MW", "g15807_Irr", "g21647_Irr", "g21647_MW", "g22437_Irr" ]
    galaxy_count = 1 if testing else len(galaxies)
    columns = [ "Rad", "Vobs", "Vbar" ]

    for i in range(galaxy_count):
        if testing: g = "g15807_Irr"
        else: g = galaxies[i]

        print("")
        print("==================================")
        print(f"Analyzing galaxy {g} ({i+1}/{galaxy_count})")
        print("==================================")

        file_path = "/mnt/users/koe/data/Santos-sims/"+g+".dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)

        r = data["Rad"].to_numpy()
        Vbar = data["Vbar"].to_numpy()
        data["errV"] = np.full( len(r), 0.01*max(data["Vobs"]) )

        # Fit for v_LCDM.
        nuts_kernel = NUTS(NFW_fit, init_strategy=init_to_median(num_samples=100))
        mcmc = MCMC(nuts_kernel, num_warmup=10000, num_samples=20000, progress_bar=testing)
        mcmc.run(random.PRNGKey(0), data)
        mcmc.print_summary()
        samples = mcmc.get_samples()
        log_likelihood = samples.pop("log_likelihood")
        print(f"BIC: {BIC_from_samples(samples, log_likelihood)}")
        
        v_LCDM = np.median(samples["Vpred"], axis=0)
        v_MOND = np.sqrt(MOND_vsq(r, Vbar**2))
        v_components = np.array([ Vbar, data["Vobs"], v_MOND, v_LCDM ])
        
        rad_count = math.ceil((max(r)-min(r))*100)
        rad = np.linspace(min(r), max(r), rad_count)

        X, X_test = r, rad

        gp_predictions = [ [], [], [], [] ]
        gp_16percent = [ [], [], [], [] ]
        gp_84percent = [ [], [], [], [] ]

        main(args, g, X, v_components, X_test)

        # Save GP fits to CSV for later use (for incorporating uncertainties/errors).
        # One array per galaxy, each containing 13 lists:
        # radii, mean (x4), 16th percentile (x4), 84th percentile (x4).
        gp_fits = np.array([rad, *gp_predictions, *gp_16percent, *gp_84percent])
        np.save("/mnt/users/koe/gp_fits/Santos-Santos/"+g, gp_fits)
        print("\nGP results successfully saved as /mnt/users/koe/gp_fits/Santos-Santos/"+g+".npy.")
    
    if not testing:
        np.save("/mnt/users/koe/gp_fits/Santos-Santos/galaxies", galaxies)
        print("\nList of analyzed galaxies now saved as /mnt/users/koe/gp_fits/Santos-Santos/galaxies.npy.")

    if make_plots:
        if do_DTW:
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
            # plt.title("Normalized DTW alignment costs")
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
        

            # """
            # Plot histogram of differences in normalized DTW costs (mock - data, in ascending order of costs for MOND - data).
            # """
            # # Rearrange galaxies into ascending order in cost_diff(MOND).
            # cost_diff = np.array([norm_cost[1] - norm_cost[0], norm_cost[2] - norm_cost[0]])
            # sort_args = np.argsort(cost_diff[0])
            # diff_sorted = cost_diff[0][sort_args]   # [ MOND ]

            # # Plot histogram of normalized DTW alignment costs of all galaxies.
            # plt.title("Normalised cost differences (mock - real data)")
            # hist_labels = [ "MOND" ]
            # colours = [ 'mediumblue', 'red' ]

            # for j in range(2):
            #     mean_diff = np.mean(cost_diff[j])
            #     plt.bar(galaxies, diff_sorted[j], color=colours[j], alpha=0.4, label=hist_labels[j])
            #     plt.axhline(y=mean_diff, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_diff))
            
            # plt.legend()
            # plt.xticks([])
            # plt.savefig(fileloc+"dtw/histo2.png", dpi=300, bbox_inches="tight")
            # plt.close()


        """
        Plot histogram of Spearman coefficients across RC (in ascending order of coefficients for data).
        """
        if do_correlations:
            """
            Plot histogram of Spearman correlations (in ascending order for data).
            """
            spearman_data = np.transpose(spearman_data)
            sort_args = np.argsort(spearman_data[0])
            spearman_sorted = []   # [ [data], [MOND] ]
            for j in range(3):
                spearman_sorted.append(spearman_data[j][sort_args])

            # Plot histogram of Spearman correlations for all galaxies.
            # plt.title("Spearman correlation coefficients")
            hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
            colours = [ 'k', 'mediumblue', 'tab:green' ]

            for j in range(3):
                mean_corr = np.mean(spearman_sorted[j])
                plt.bar(galaxies, spearman_sorted[j], color=colours[j], alpha=0.3, label=hist_labels[j])
                plt.axhline(y=mean_corr, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_corr))
            
            plt.legend()
            plt.xticks([])
            plt.savefig(fileloc+"correlations/radii/d0/spearman_histo1.png", dpi=300, bbox_inches="tight")
            plt.close()

            """
            Plot histogram of Pearson correlations (in ascending order for data).
            """
            pearson_data = np.transpose(pearson_data)
            sort_args = np.argsort(pearson_data[0])
            pearson_sorted = []   # [ [data], [MOND], [LCDM] ]
            for j in range(3):
                pearson_sorted.append(pearson_data[j][sort_args])

            # Plot histogram of Pearson correlations for all galaxies.
            # plt.title("pearson correlation coefficients")
            hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
            colours = [ 'k', 'mediumblue', 'tab:green' ]

            for j in range(3):
                mean_corr = np.mean(pearson_sorted[j])
                plt.bar(galaxies, pearson_sorted[j], color=colours[j], alpha=0.3, label=hist_labels[j])
                plt.axhline(y=mean_corr, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_corr))
            
            plt.legend()
            plt.xticks([])
            plt.savefig(fileloc+"correlations/radii/d0/pearson_histo1.png", dpi=300, bbox_inches="tight")
            plt.close()

    print("\nMax memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
