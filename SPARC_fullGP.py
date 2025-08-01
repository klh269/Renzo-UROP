#!/usr/bin/env python
"""
Correlation coefficients + dynamic time warping on SPARC GP residuals,
taking into account uncertainties (Vbar) and Vobs scattering (errV).

This version fits a new GPR to each mock sample, so (1000 x 3 + 2) x 60 = 180,120 GPRs in total.
Galaxies should be fitted in parallel using SPARC_fullGP/run_analyses.py.
"""
import pandas as pd
import argparse
from resource import getrusage, RUSAGE_SELF

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import jax
from jax import vmap
import jax.random as random
import numpyro

from utils_analysis.gp_utils import model, predict, run_inference
from utils_analysis.dtw_utils import dtw
from utils_analysis.Vobs_fits import Vbar_sq
from utils_analysis.mock_gen import Vobs_scat, MOND_unc, Vbar_sq_unc
# from utils_analysis.extract_ft import ft_check

matplotlib.use("Agg")
plt.rcParams.update({'font.size': 13})

# Directory for saving plots.
fileloc = "/mnt/users/koe/plots/SPARC_fullGP/"
make_plots = False

num_samples = 1000


# Main code to run.
def main(args, g, r_full, rad, v_data, v_mock, ls:float, num_samples=num_samples,
         redo_GPR:bool=False, print_progress:bool=False, ft_windows:bool=False):
    """
    Do inference for Vbar with uniform prior for correlation length,
    then apply the resulted lengthscale to Vobs (both real and mock data).
    """
    data_labels = [ r"$V_{\text{bar}}$", r"$V_{\text{obs}}$" ]
    mock_labels = [ r"$V_{\text{MOND}}$", r"$V_{\Lambda CDM}$" ]
    v_comps = data_labels + mock_labels
    colours = [ 'tab:red', 'k', 'mediumblue', 'tab:green' ]

    raw_median = np.median(v_mock, axis=1)    # dim = (4, r)


    """ ------------
    GPR on data.
    ------------ """
    meanGP_data = []
    for j in range(2):
        if print_progress: print(f"Fitting function to {data_labels[j]} with ls = {ls} kpc...")
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = run_inference(model, args, rng_key, r_full, v_data[j], ls=ls)

        vmap_args = (
            random.split(rng_key_predict, samples["var"].shape[0]),
            samples["var"],
            samples["noise"],
        )

        means, _ = vmap(
            lambda rng_key, var, noise: predict(
                rng_key, r_full, v_data[j], rad, var, ls, noise, use_cholesky=args.use_cholesky
            )
        )(*vmap_args)

        mean_pred = np.mean(means, axis=0)  # [ Vbar, Vobs ]
        meanGP_data.append(mean_pred)

    # Compute residuals of data fits.
    res_Vbar_data, res_Vobs = [], []
    for k in range(len(r_full)):
        idx = (np.abs(rad - r_full[k])).argmin()
        res_Vbar_data.append(v_data[0][k] - meanGP_data[0][idx])
        res_Vobs.append(v_data[1][k] - meanGP_data[1][idx])
    res_data = np.array([ res_Vbar_data, res_Vobs ])    # dim = (2, len(r))


    if redo_GPR:
        """ --------------------
        GPR on mock samples.
        -------------------- """
        res_mock = []
        plotGP_mock = [ [], [], [] ]
        # GPR and analysis on individual mock samples.
        for smp in range(num_samples):
            if print_progress and smp % max( 1, num_samples/10 ) == 0: print(f"GPR on mock sample {smp+1} of {num_samples}...")
            meanGP_mock = []
        
            for j in range(3):
                rng_key, rng_key_predict = random.split(random.PRNGKey(0))
                samples = run_inference(model, args, rng_key, r_full, v_mock[j,smp], ls=ls)

                # do prediction
                vmap_args = (
                    random.split(rng_key_predict, samples["var"].shape[0]),
                    samples["var"],
                    samples["noise"],
                )
                means, _ = vmap(
                    lambda rng_key, var, noise: predict(
                        rng_key, r_full, v_mock[j,smp], rad, var, ls, noise, use_cholesky=args.use_cholesky
                    )
                )(*vmap_args)

                mean_pred = np.mean(means, axis=0)
                meanGP_mock.append(mean_pred)   # [ MOND, LCDM ]

                if smp == 0: plotGP_mock[j] = np.array(mean_pred) / num_samples
                else: plotGP_mock[j] += np.array(mean_pred) / num_samples

                jax.clear_caches()  # DO NOT DELETE THIS LINE! Reduces memory usage from > 100 GB to < 1 GB!

            # Compute residuals of fits.
            res_Vbar_mock, res_MOND, res_LCDM = [], [] ,[]
            for k in range(len(r_full)):
                idx = (np.abs(rad - r_full[k])).argmin()
                res_Vbar_mock.append(v_mock[0,smp,k] - meanGP_mock[0][idx])
                res_MOND.append(v_mock[1,smp,k] - meanGP_mock[1][idx])
                res_LCDM.append(v_mock[2,smp,k] - meanGP_mock[2][idx])
            res_mock.append( np.array([ res_Vbar_mock, res_Vbar_mock, res_MOND, res_LCDM ]) )   # dim = num_samples x (4, len(r))

        res_mock = np.transpose( res_mock, (1, 2, 0) )  # dim = (4, len(r), num_samples)

        np.save(f"/mnt/users/koe/mock_residuals/{g}.npy", res_mock)
        np.save(f"/mnt/users/koe/mock_residuals/plotGP/{g}.npy", plotGP_mock)

        if print_progress:
            print("\nMock residuals and GP means saved.")
            print("Memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)

    else:
        """ --------------------
        Load mock samples.
        -------------------- """
        res_mock = np.load( f"/mnt/users/koe/mock_residuals/{g}.npy" )  # dim = (4, len(r), num_samples)
        plotGP_mock = np.load( f"/mnt/users/koe/mock_residuals/plotGP/{g}.npy" )    # dim = (4, len(r))
        if print_progress: print("Mock residuals and GP means loaded.")
        
    res_mock_percentiles = np.percentile(res_mock, [16.0, 50.0, 84.0], axis=2)  # dim = (3, 4, len(r))


    """ -------------------------
    Analysis of GP residuals.
    ------------------------- """
    # Only analyze regions with features in Vobs.
    if ft_windows:
        SPARC_features = np.load("/mnt/users/koe/gp_fits/SPARC_features.npy", allow_pickle=True).item()
        lb, rb, _ = SPARC_features[g]
        if len(lb) in [1, 2]:
            slices = [slice(lb[i], rb[i]) for i in range(len(lb))]
            res_data = np.concatenate([res_data[:, s] for s in slices], axis=1)
            res_mock = np.concatenate([res_mock[:, s, :] for s in slices], axis=1)
            r = np.concatenate([r_full[s] for s in slices], axis=0)
            err_Vobs = np.concatenate([v_data[2][s] for s in slices], axis=0)
        else:
            raise ValueError(f"Galaxy {g} has more than 2 features ({len(lb)}) in Vbar!?")
    
    dtw_cost = [ [], [], [] ]

    """ Check for features in Vbar and Vobs. """
    # if print_progress: print(f"\nChecking for features...")
    # Vbar_percentiles = np.percentile(res_mock[0], [16.0, 50.0, 84.0], axis=1)
    # Vbar_err = ( Vbar_percentiles[2] - Vbar_percentiles[0] ) / 2.0

    # lb, rb, widths = ft_check(res_data[0], Vbar_err)
    # if print_progress and len(lb) > 0:
    #     print(f"\nFeature found in Vbar of {g}")
    #     print(f"Properties: lb={lb}, rb={rb}, widths={widths}")
    # Vbar_features = { "lb": lb, "rb": rb, "widths": widths }

    # lb, rb, widths = ft_check(res_data[1], v_data[2])
    # if print_progress and len(lb) > 0:
    #     print(f"\nFeature found in Vobs of {g}")
    #     print(f"Properties: lb={lb}, rb={rb}, widths={widths}")
    # Vobs_features = { "lb": lb, "rb": rb, "widths": widths }

    """ DTW. """
    if print_progress: print(f"\nComputing DTW...")
    for smp in range(num_samples):
        # Construct distance matrices.
        dist_data = np.zeros((len(r), len(r)))
        dist_MOND = np.copy(dist_data)
        dist_LCDM = np.copy(dist_data)
        
        for n in range(len(r)):
            for m in range(len(r)):
                dist_data[n, m] = np.abs( res_data[0,n] - res_data[1,m] )
                dist_MOND[n, m] = np.abs( res_mock[0,n,smp] - res_mock[2,m,smp] )
                dist_LCDM[n, m] = np.abs( res_mock[1,n,smp] - res_mock[3,m,smp] )
        
        dist_mats = np.array([ dist_data, dist_MOND, dist_LCDM ])
        mats_dir = [ "data", "MOND", "LCDM" ]
        
        # DTW!
        for j in range(3):
            if j == 0 and smp >= 1:
                dtw_cost[j].append(dtw_cost[j][0])
            else:
                path, cost_mat = dtw(dist_mats[j])
                x_path, y_path = zip(*path)
                cost = cost_mat[ len(r)-1, len(r)-1 ]
                dtw_cost[j].append(cost / (2 * len(r)))

            if make_plots and smp == 0:
                fname_DTW = fileloc + "dtw/"

                # Plot distance matrix and cost matrix with optimal path.
                plt.axis('off')
                plt.subplot(121)
                plt.title("Distance matrix")
                plt.imshow(dist_mats[j], cmap=plt.cm.binary, interpolation="nearest", origin="lower")

                plt.subplot(122)
                plt.title("Cost matrix")
                plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
                plt.plot(x_path, y_path)

                plt.savefig(fname_DTW+"matrix_"+mats_dir[j]+".png", dpi=300, bbox_inches="tight")
                plt.close('all')

                # Visualize DTW alignment.
                plt.title("DTW alignment: NGC 1560")

                if j == 0:
                    diff = abs(max(np.array(res_data[0])) - min(res_data[1]))
                    for x_i, y_j in path:
                        plt.plot([x_i, y_j], [res_data[1,x_i] + diff, res_data[0,y_j] - diff], c="C7", alpha=0.4)
                    plt.plot(np.arange(len(r)), res_data[1] + diff, c='k', label=v_comps[1])
                    plt.plot(np.arange(len(r)), res_data[0] - diff, c="tab:red", label=r'$V_{\text{bar}}$')

                else: 
                    diff = abs(max(np.array(res_mock)[j-1,:,smp]) - min(np.array(res_mock)[j+1,:,smp]))
                    for x_i, y_j in path:
                        plt.plot([x_i, y_j], [res_mock[j+1,x_i,smp] + diff, res_mock[j-1,y_j,smp] - diff], c="C7", alpha=0.4)
                    plt.plot(np.arange(len(r)), np.array(res_mock)[j+1,:,smp] + diff, c=colours[j+1], label=v_comps[j+1])
                    plt.plot(np.arange(len(r)), np.array(res_mock)[j-1,:,smp] - diff, c='tab:red', label=r'$V_{\text{bar}}$')

                plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(cost))
                plt.plot([], [], c='w', label="Normalized cost = {:.4f}".format(cost/(len(r)*2)))

                plt.axis("off")
                plt.legend(bbox_to_anchor=(1,1))
                plt.savefig(fname_DTW+"vis_"+mats_dir[j]+".png", dpi=300, bbox_inches="tight")
                plt.close('all')

    """ Pearson correlation coefficients. """
    if print_progress: print(f"\nComputing Pearson correlation coefficients...")
    pearsonr_data = []
    for j in range(3, len(r)+1):
        pearsonr_data.append(stats.pearsonr(res_data[0,:j], res_data[1,:j])[0])
    pearson_data = pearsonr_data[-1]

    radii_corr = []
    for smp in range(num_samples):
        correlations_r = []
        for i in range(1, 3):
            pearsonr_mock = []
            for j in range(3, len(r)+1):
                # pearsonr_mock.append(stats.pearsonr(res_mock[i-1,:j,smp], res_mock[i+1,:j,smp])[0])
                pearsonr_mock.append(stats.pearsonr(res_data[0,:j], res_mock[i+1,:j,smp])[0])
            correlations_r.append(pearsonr_mock)
        radii_corr.append(correlations_r)
        
    rcorr_percentiles = np.percentile(radii_corr, [16.0, 50.0, 84.0], axis=0)
    pearson_mock = [ rcorr_percentiles[:,0,-1], rcorr_percentiles[:,1,-1] ]

    """ Plot GP fits, residuals and correlations. """
    if make_plots:
        """Pearson correlations."""
        fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
        fig1.set_size_inches(7, 7)
        ax0.set_title("Residuals correlation: "+g)
        ax0.set_ylabel("Velocities (km/s)")
        
        for j in range(4):
            if j < 2:
                if j == 0: ax0.scatter(r, v_data[0], color='tab:red', alpha=0.3)    # Vbar
                else: ax0.errorbar(r, v_data[1], v_data[2], color='k', alpha=0.3, fmt='o', capsize=2)   # Vobs
                ax0.plot(rad, meanGP_data[j], color=colours[j], label=v_comps[j], zorder=10-j)
            else: 
                ax0.scatter(r, raw_median[j-1], c=colours[j], alpha=0.3)
                ax0.plot(rad, plotGP_mock[j-1], color=colours[j], label=v_comps[j], zorder=10-j)

        ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
        ax0.grid()

        ax1.set_ylabel("Residuals (km/s)")
        for j in range(4):
            # Plots for mock Vobs + Vbar (sampled w/ uncertainties).
            if j < 2:
                ax1.errorbar(r[2:], res_data[1,2:], v_data[2,2:], color='k', alpha=0.3, ls='none', fmt='o', capsize=2)
                ax1.scatter(r[2:], res_data[0,2:], c='tab:red', alpha=0.3)
                ax1.plot(r[2:], res_data[j,2:], c=colours[j])
            else:
                ax1.scatter(r[2:], res_mock_percentiles[1,j,2:], c=colours[j], alpha=0.3)
                ax1.plot(r[2:], res_mock_percentiles[1,j,2:], c=colours[j])
                ax1.fill_between(r[2:], res_mock_percentiles[0,j,2:], res_mock_percentiles[2,j,2:], color=colours[j], alpha=0.15)

        ax1.grid()

        ax2.set_xlabel("Radii (kpc)")
        ax2.set_ylabel("Correlations w/ Vbar")

        ax2.plot(r[2:], pearsonr_data, c='k')
        for j in range(2):
            ax2.plot(r[2:], rcorr_percentiles[1,j], c=colours[j+2])
            ax2.fill_between(r[2:], rcorr_percentiles[0,j], rcorr_percentiles[2,j], color=colours[j+2], alpha=0.2)

        ax2.grid()
        plt.subplots_adjust(hspace=0.05)
        fig1.savefig(fileloc+"pearson/"+g+".png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # for smp in range(num_samples):
    #     radii_corr.append( [ stats.pearsonr(res_mock[0,:,smp], res_mock[2,:,smp])[0],
    #                         stats.pearsonr(res_mock[1,:,smp], res_mock[3,:,smp])[0] ] )     # [ MOND, LCDM ]
    
    # pearson_mock = np.percentile(radii_corr, [16.0, 50.0, 84.0], axis=0).T

    g_dict = { "pearson_data" : pearson_data, "pearson_mock" : pearson_mock, "dtw_cost" : dtw_cost }
    if ft_windows:
        g_dict["sig2noise"] = np.max( np.abs(res_data[1]) / err_Vobs )
        np.save(f"/mnt/users/koe/SPARC_fullGP/ft_windows/{g}.npy", g_dict)
    else: np.save(f"/mnt/users/koe/SPARC_fullGP/{g}.npy", g_dict)

    if print_progress: print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
    jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem.


def get_args():
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
    parser.add_argument("--testing", default=False, type=bool)
    parser.add_argument("--galaxy", default="", type=str)
    parser.add_argument("--redo-GPR", default=False, type=bool)
    parser.add_argument("--ft-windows", default=False, type=bool,
                        help="Analyze only regions with features in Vobs.")
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    return args


def LCDM_unc(Vbar2_unc, i_table, num_samples):
    vDM_unc = np.array([v_DM[i_table]] * num_samples).T
    return np.sqrt(Vbar2_unc + vDM_unc**2)


if __name__ == "__main__":
    # Initialize GP arguments.
    args = get_args()
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

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
    
    # Analyze only features instead.
    ft_windows = args.ft_windows
    if ft_windows:
        SPARC_features = np.load("/mnt/users/koe/gp_fits/SPARC_features.npy", allow_pickle=True).item()
        galaxies = list(SPARC_features.keys())
    else:
        galaxies = np.load("/mnt/users/koe/gp_fits/galaxy.npy")

    if args.galaxy != "":
        galaxy_count = 1
        print_progress = True
    else:
        galaxy_count = len(galaxies)
        print_progress = False

    for i in range(galaxy_count):
        if args.galaxy != "": g = args.galaxy
        else: g = galaxies[i]
        i_tab = np.where(table["Galaxy"] == g)[0][0]

        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        bulged = np.any( data["Vbul"] > 0 )     # Check whether galaxy has bulge.

        r = data["Rad"].to_numpy()
        rad = np.linspace(min(r), max(r), 100)

        Vbar_squared = Vbar_sq_unc(table, i_tab, data, bulged, num_samples)
        
        # Assume errV completely UNcorrelated.
        full_MOND = Vobs_scat(MOND_unc(r, Vbar_squared, num_samples), data["errV"], num_samples)
        full_LCDM = Vobs_scat(LCDM_unc(Vbar_squared, i_tab, num_samples), data["errV"], num_samples)

        v_data = np.array([ np.sqrt(Vbar_sq(data)), data["Vobs"], data["errV"] ])
        v_mock = np.array([ np.sqrt(Vbar_squared).T, full_MOND.T, full_LCDM.T ])    # dim = (3, num_samples, len(r))

        ls_dict = np.load("/mnt/users/koe/gp_fits/ls_dict.npy", allow_pickle=True).item()
        ls = ls_dict[g]

        print(f"Analyzing {g} ({i+1}/{galaxy_count})...")
        main( args, g, r, rad, v_data, v_mock, ls, num_samples=num_samples,
             redo_GPR=args.redo_GPR, print_progress=print_progress, ft_windows=ft_windows )

    print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
