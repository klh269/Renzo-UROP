#!/usr/bin/env python
"""
Correlation coefficients + dynamic time warping on Santos-Santos residuals,
taking into account uncertainties (Vbar) and Vobs scattering (errV).

GP fits taken from Santos-Santos.py output, which are saved in /mnt/users/koe/gp_fits/Santos-Santos/.
"""
import pandas as pd
from resource import getrusage, RUSAGE_SELF
import jax

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
from scipy import stats

from utils_analysis.dtw_utils import dtw
from utils_analysis.mock_gen import Vobs_scat
# from utils_analysis.extract_ft import ft_check
# from tqdm import tqdm

plt.rcParams.update({'font.size': 14})


use_features = True
testing = False
make_plots = True
use_DTW = True
do_correlations = True

fileloc = "/mnt/users/koe/plots/Santos-Santos/"
if use_features: fileloc += "ft_windows/"

fileloc += "NGC1560_errV/"    # Set to: /SPARC_errV/ or /NGC1560_errV/.
if use_DTW: fname_DTW = fileloc + "dtw/"

num_samples = 1000
avg_errV = 0.02   # Average error (x Vmax) in Vobs data (SPARC: 0.05, Sanders NGC 1560: 0.02).


# Main code to run.
def main(g, r_full, velocities, errV, num_samples=num_samples):
    v_data = velocities[:2]
    v_mock = velocities[2:]

    # Load in GP results from combined_dtw.py
    gp_fits = np.load(f"/mnt/users/koe/gp_fits/Santos-Santos/{g}.npy")
    rad = gp_fits[0]
    mean_prediction = [ gp_fits[1], gp_fits[2], gp_fits[3], gp_fits[4] ]    # Mean predictions from GP for [ Vbar, Vobs, MOND, LCDM ]
    lower_percentile = [ gp_fits[5], gp_fits[6], gp_fits[7], gp_fits[8] ]   # 16t percentiles from GP
    upper_percentile = [ gp_fits[9], gp_fits[10], gp_fits[11], gp_fits[12] ]    # 84th percentiles from GP

    # "Raw" percentiles from uncertainties and scattering.
    raw_median = np.percentile(velocities, 50.0, axis=2)    # dim = (4, r)
    raw_percentiles = np.percentile(velocities, [16.0, 84.0], axis=2)   # dim = (3, 4, r)
    raw_errors = np.abs( raw_percentiles - raw_median )     # dim = (2, 4, r)

    # Compute residuals of fits (v_data = [Vbar, Vobs], v_mock = [MOND, LCDM]).
    res_Vbar, res_Vobs, res_MOND, res_LCDM = [], [] ,[], []
    for k in range(len(r_full)):
        idx = (np.abs(rad - r_full[k])).argmin()
        
        res_Vbar.append(v_data[0,k] - mean_prediction[0][idx])
        res_Vobs.append(v_data[1,k] - mean_prediction[1][idx])

        res_MOND.append(v_mock[0,k] - mean_prediction[2][idx])
        res_LCDM.append(v_mock[1,k] - mean_prediction[3][idx])

    res_Vbar = np.array( res_Vbar )
    res_Vobs = np.array( res_Vobs )
    res_data = np.array([ res_Vbar, res_Vobs ])     # dim = (2, len(r), num_samples)
    res_mock = np.array([ res_MOND, res_LCDM ])     # dim = (2, len(r), num_samples)

    if use_features:
        ft_dict = np.load("/mnt/users/koe/Santos-analysis/ft_properties.npy", allow_pickle=True).item()
        ft_properties = ft_dict[g]
        lb, rb = ft_properties[0][0], ft_properties[1][0]
        res_Vbar = res_Vbar[lb:rb+1]
        res_Vobs = res_Vobs[lb:rb+1]
        res_data = res_data[:,lb:rb+1,:]
        res_mock = res_mock[:,lb:rb+1,:]
        r = r_full[lb:rb+1]
    else:
        r = r_full

    # Residual percentiles from uncertainties and scattering; dimensions = (3, 1 or 2, len(r)).
    res_data_median = np.percentile(res_data, 50.0, axis=2)                 # dim = (3, r)
    res_data_percentiles = np.percentile(res_data, [16.0, 84.0], axis=2)    # dim = (2 (perc), 2 (v_comp), r)
    res_mock_median = np.percentile(res_mock, 50.0, axis=2)                 # dim = (3, r)
    res_mock_percentiles = np.percentile(res_mock, [16.0, 84.0], axis=2)    # dim = (2 (perc), 2 (v_comp), r)

    # Labels and colours for plots.
    v_comps = [ r"$V_{bar}$", r"$V_{obs}$", r"$V_{MOND}$", r"$V_{\Lambda CDM}$" ]
    colours = [ 'tab:red', 'k', 'mediumblue', 'tab:green' ]


    """
    DTW on GP residuals.
    """
    if use_DTW:
        dtw_cost_smp = [ [], [], [] ]
        norm_cost_smp = [ [], [], [] ]

        print("Warping time dynamically, kinda...")
        for smp in range(num_samples):
        # for smp in tqdm(range(num_samples), desc="DTW"):
            # Construct distance matrices.
            dist_data = np.zeros((len(r), len(r)))
            dist_MOND = np.copy(dist_data)
            dist_LCDM = np.copy(dist_data)
            
            for n in range(len(r)):
                for m in range(len(r)):
                    dist_data[n, m] = np.abs(res_Vobs[n, smp] - res_Vbar[m, smp])
                    dist_MOND[n, m] = np.abs(res_MOND[n][smp] - res_Vbar[m, smp])
                    dist_LCDM[n, m] = np.abs(res_LCDM[n][smp] - res_Vbar[m, smp])
            
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
                    ref_curve = [ res_Vbar[:,smp], "tab:red", "Vbar" ]

                    if j == 0:
                        diff = abs(max(np.array(ref_curve[0])) - min(res_Vobs[:, smp]))
                        for x_i, y_j in path:
                            plt.plot([x_i, y_j], [res_Vobs[x_i, smp] + diff, ref_curve[0][y_j] - diff], c="C7", alpha=0.4)
                        plt.plot(np.arange(len(r)), np.array(res_Vobs[:, smp]) + diff, c='k', label=v_comps[3])

                    else: 
                        diff = abs(max(np.array(ref_curve[0])) - min(np.array(res_mock)[j-1,:,smp]))
                        for x_i, y_j in path:
                            plt.plot([x_i, y_j], [res_mock[j-1][x_i][smp] + diff, ref_curve[0][y_j] - diff], c="C7", alpha=0.4)
                        plt.plot(np.arange(len(r)), np.array(res_mock)[j-1,:,smp] + diff, c=colours[j+1], label=v_comps[j])

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
        # Compute correlation coefficients for mock Vobs vs Vbar.
        corr_data, corr_mock = [], []   # dim = (num_samples/10, 2 x mock_vcomps, 2 x rho, rad)

        print("Correlating by radii...")
        for smp in range(num_samples):
        # for smp in tqdm(range(num_samples), desc="Correlation by radii"):
            correlations_data= []
            for j in range(3, len(r)+1):
                correlations_data.append(stats.pearsonr(res_Vbar[:j,smp], res_Vobs[:j,smp])[0])
            corr_data.append(correlations_data)

            correlations_mock = []
            for i in range(2):
                pearsonr_mock = []
                for j in range(3, len(r)+1):
                    pearsonr_mock.append(stats.pearsonr(res_Vbar[:j,smp], res_mock[i,:j,smp])[0])
                correlations_mock.append(pearsonr_mock)
            
            corr_mock.append(correlations_mock)

        res_data_percentiles = np.percentile(res_data, [16.0, 50.0, 84.0], axis=2)
        corr_data_percentiles = np.percentile(corr_data, [16.0, 50.0, 84.0], axis=0)
        pearson_data.append( corr_data_percentiles[:,-1] )
        
        res_mock_percentiles = np.percentile(res_mock, [16.0, 50.0, 84.0], axis=2)
        corr_mock_percentiles = np.percentile(corr_mock, [16.0, 50.0, 84.0], axis=0)
        pearson_mock.append([ corr_mock_percentiles[:,0,-1], corr_mock_percentiles[:,1,-1] ])


        """
        Plot GP fits, residuals and correlations.
        """
        if make_plots:
            c_data = [ 'tab:red', 'k' ]
            c_mock = [ 'mediumblue', 'tab:green' ]

            """Pearson correlations."""
            # for der in range(1):
            fig1, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [5, 2, 3]})
            fig1.set_size_inches(7, 7)
            ax0.set_title("Residuals correlation: "+g)
            ax0.set_ylabel("Velocities (km/s)")

            # ax0.scatter(r, v_data[0], c='tab:red', alpha=0.3)   # Vbar
            # ax0.errorbar(r, v_data[1], errV, color='k', alpha=0.3, fmt='o', capsize=2)   # Vobs
            for j in range(4): ax0.errorbar(r_full, raw_median[j], raw_errors[:, j], c=colours[j], alpha=0.3, fmt='o', capsize=2) # Vmock
            
            for j in range(4):
                ax0.plot(rad, mean_prediction[j], color=colours[j], label=v_comps[j])
                ax0.fill_between(rad, lower_percentile[j], upper_percentile[j], color=colours[j], alpha=0.2)

            ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
            ax0.grid()

            # ax1.scatter(r, res_data[0], c='tab:red', alpha=0.3)
            # ax1.errorbar(r, res_data[1], errV, color='k', alpha=0.3, ls='none', fmt='o', capsize=2)
            for j in range(2):
                ax1.scatter(r, res_data_median[j], c=c_data[j], alpha=0.3)
                ax1.plot(r, res_data_percentiles[1][j], c=c_data[j])
                ax1.fill_between(r, res_data_percentiles[0][j], res_data_percentiles[2][j], color=c_data[j], alpha=0.15)

                ax1.scatter(r, res_mock_median[j], c=c_mock[j], alpha=0.3)
                ax1.plot(r, res_mock_percentiles[1][j], c=c_mock[j])
                ax1.fill_between(r, res_mock_percentiles[0][j], res_mock_percentiles[2][j], color=c_mock[j], alpha=0.15)

            ax1.set_ylabel("Residuals (km/s)")
            ax1.grid()

            ax2.set_xlabel("Radii (kpc)")
            ax2.set_ylabel("Correlations w/ Vbar")
            
            vel_comps = [ "MOND", r"$\Lambda$CDM" ]

            for j in range(2):
                ax2.plot(r[2:], corr_mock_percentiles[1][j], c=c_mock[j], label=vel_comps[j]+r": Pearson $\rho$")
                ax2.fill_between(r[2:], corr_mock_percentiles[0][j], corr_mock_percentiles[2][j], color=c_mock[j], alpha=0.2)

            # ax2.plot(r[2:], pearsonr_data, c='k', label=r"Data: Pearson $\rho$")
            # ax2.plot([], [], ' ', label=r"$\rho_p=$"+str(round(np.nanmean(pearsonr_data), 3)))
            ax2.plot(r[2:], corr_data_percentiles[1], c='k', label=r"Data: Pearson $\rho$")
            ax2.fill_between(r[2:], corr_data_percentiles[0], corr_data_percentiles[2], color='k', alpha=0.2)

            ax2.legend()
            ax2.grid()

            plt.subplots_adjust(hspace=0.05)
            fig1.savefig(fileloc+"correlations/"+g+".png", dpi=300, bbox_inches="tight")
            plt.close()
    
    jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem.


if __name__ == "__main__":
    dtw_cost, norm_cost = [ [], [], [] ], [ [], [], [] ]

    if use_features: galaxies = np.load("/mnt/users/koe/Santos-analysis/ft_galaxies.npy")
    else: galaxies = np.load("/mnt/users/koe/gp_fits/Santos-Santos/galaxies.npy")
    galaxy_count = len(galaxies)
    columns = [ "Rad", "Vobs", "Vbar" ]

    if testing:
        galaxy_count = 2
        galaxies = [ 'g1536_Irr', 'g15807_Irr' ]
    
    pearson_data, pearson_mock = [], []
    dtw_cost = [ [], [], [] ]
    norm_cost = [ [], [], [] ]

    for i in range(galaxy_count):
        g = galaxies[i]

        file_path = f"/mnt/users/koe/data/Santos-sims/{g}.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        r = data["Rad"].to_numpy()
        errV = np.full(len(r), avg_errV * max(data["Vobs"]))

        MCMC_fits = np.load(f"/mnt/users/koe/MCMC_fits/Santos-Santos/{g}.npy")
        Vbar = np.array([data["Vbar"]] * num_samples).T
        Vobs = np.array([data["Vobs"]] * num_samples).T
        v_MOND = np.array([MCMC_fits[0]] * num_samples).T
        v_LCDM = np.array([MCMC_fits[1]] * num_samples).T

        # full_Vbar = Vobs_scat(Vbar, errV, num_samples)
        full_Vobs = Vobs_scat(Vobs, errV, num_samples)
        full_MOND = Vobs_scat(v_MOND, errV, num_samples)
        full_LCDM = Vobs_scat(v_LCDM, errV, num_samples)

        velocities = np.array([ Vbar, full_Vobs, full_MOND, full_LCDM ])

        print(f"\nAnalyzing galaxy: {g} ({i+1}/{galaxy_count})")
        main(g, r, velocities, errV)

    print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)


    """
    --------------
    SUMMARY PLOTS.
    --------------
    """
    if make_plots and not testing:
        g_features = [ "g15807_Irr", "g15784_Irr", "g1536_MW", "g5664_MW", "C1", "C5", "C6", "C7", "C8" ]
        g_CLUES = [ "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10" ]

        analysis_dir = "/mnt/users/koe/Santos-analysis/"
        if use_features: analysis_dir += "ft_windows/"

        """
        Plot histogram of normalized DTW costs (in ascending order of costs for data).
        """
        if use_DTW:
            print("\nGenerating DTW histograms...")

            # dim = (3 x v_comps, galaxy_count, num_samples)
            dtw_cost = np.array(dtw_cost)
            norm_cost = np.array(norm_cost)

            # Arrays of shape (5 x percentiles, 3 x v_comps, galaxy_count).
            norm_percentiles = np.percentile(norm_cost, [16.0, 50.0, 84.0], axis=2)
            dtw_percentiles = np.percentile(dtw_cost, [16.0, 50.0, 84.0], axis=2)

            # # Rearrange galaxies into ascending order in median of data normalised costs.
            # sort_args = np.argsort(norm_percentiles[1][0])
            # norm_percentiles = norm_percentiles[:, :, sort_args]

            # Load sorted arrays and indices from Santos-Santos.py (original analysis w/o errors).
            sort_args = np.load(f"{analysis_dir}dtw_args.npy")
            sort_args = np.flip(sort_args)
            costs_sorted = np.load(f"{analysis_dir}dtw.npy")
            costs_sorted = np.flip(costs_sorted, axis=1)

            norm_percentiles = norm_percentiles[:,:,sort_args]

            print(f"Galaxies in ascending order of cost(data): {np.array(galaxies)[sort_args]}")

            hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
            colours = [ 'k', 'mediumblue', 'tab:green' ]

            fig, ax = plt.subplots()
            # ax.bar(galaxies, norm_percentiles[1][0], color=colours[0], alpha=0.3, label=hist_labels[0])
            for j in range(3): ax.plot(galaxies[sort_args], costs_sorted[j], color=colours[j], label=hist_labels[j])
            
            ax1 = ax.twinx()
            for j in range(3):
                # mean_norm = np.nanmean(norm_percentiles[1][j])
                low_err = norm_percentiles[1][j] - norm_percentiles[0][j]
                up_err = norm_percentiles[2][j] - norm_percentiles[1][j]

                # if j != 0:
                #     low_norm1 = np.full(galaxy_count, np.nanmean(norm_percentiles[0][j]))
                #     up_norm1 = np.full(galaxy_count, np.nanmean(norm_percentiles[2][j]))
                    # ax.fill_between(galaxies, low_norm1, up_norm1, color=colours[j], alpha=0.25)

                # ax.axhline(y=mean_norm, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_norm))
                if j == 0: trans = Affine2D().translate(-0.15, 0.0) + ax1.transData
                elif j == 2: trans = Affine2D().translate(+0.15, 0.0) + ax1.transData
                else: trans = ax1.transData
                ax1.errorbar(galaxies[sort_args], norm_percentiles[1][j], [low_err, up_err], fmt='.', ls='none',
                             capsize=2, color=colours[j], alpha=0.7, transform=trans)
                            
            # ax.legend()
            # ax.set_xticks([])

            # Bold galaxies with features identified by authors + label CLUES galaxies in red.
            for ele in ax.get_xticklabels():
                if ele.get_text() in g_CLUES: ele.set_color('tab:red')
                if ele.get_text() in g_features: ele.set_fontweight('bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

            ax.set_xlabel("Galaxies")
            ax.set_ylabel("DTW cost (no noise, solid lines)")
            ax1.set_ylabel("DTW cost w/ noise (error bars)")

            fig.savefig(fname_DTW+"histo1.pdf", dpi=300, bbox_inches="tight")
            plt.close()


            """
            Scatter plots of cost(mock) against cost(data).
            """
            # plotloc = [ "MOND", "LCDM" ]
            # for j in range(1, 3):
            #     plt.title("Scatter plot: cost("+hist_labels[j]+") vs cost(data)")
            #     low_err = norm_percentiles[2][j] - norm_percentiles[1][j]
            #     up_err = norm_percentiles[3][j] - norm_percentiles[2][j]

            #     plt.xlabel("Cost(Data)")
            #     plt.ylabel("Cost("+hist_labels[j]+")")
            #     plt.errorbar(norm_percentiles[2][0], norm_percentiles[2][j], [low_err, up_err], fmt='.', ls='none', capsize=2, color=colours[j], alpha=0.5, label=hist_labels[j])
            
            #     plt.legend()
            #     plt.savefig(fname_DTW+"scatter_"+plotloc[j-1]+".png", dpi=300, bbox_inches="tight")
            #     plt.close()


            """
            Plot histogram of differences in normalized DTW costs (mock - data, in ascending order of costs for MOND - data).
            """
            # # Rearrange galaxies into ascending order in cost_diff(MOND).
            # cost_diff = np.array([norm_cost[1] - norm_cost[0], norm_cost[2] - norm_cost[0]])

            # # Arrays of shape (5 x percentiles, 2 x v_comps, galaxy_count).
            # diff_perc = np.percentile(cost_diff, [5.0, 16.0, 50.0, 84.0, 95.0], axis=2)

            # # Sort by descending order in difference between (LCDM - data).
            # sort_args = np.argsort(diff_perc[2][1])[::-1]
            # diff_percentiles = diff_perc[:, :, sort_args]

            # print(f"Galaxies in descending order of cost(LCDM) - cost(data): {np.array(galaxies)[sort_args]}")

            # # Plot histogram of normalized DTW alignment costs of all galaxies.
            # plt.title("Normalised cost differences (mock - real data)")
            # hist_labels = [ "MOND", r"$\Lambda$CDM" ]
            # colours = [ 'mediumblue', 'tab:green' ]

            # for j in range(2):          
            #     mean_diff = np.nanmean(diff_percentiles[2][j])
            #     low_err = diff_percentiles[2][j] - diff_percentiles[1][j]
            #     up_err = diff_percentiles[3][j] - diff_percentiles[2][j]

            #     low_norm1 = np.full(galaxy_count, np.nanmean(diff_percentiles[1][j]))
            #     up_norm1 = np.full(galaxy_count, np.nanmean(diff_percentiles[3][j]))

            #     plt.errorbar(galaxies, diff_percentiles[2][j], [low_err, up_err], fmt='.', ls='none', capsize=2, color=colours[j], alpha=0.5, label=hist_labels[j])
            #     plt.axhline(y=mean_diff, color=colours[j], linestyle='dashed', label="Mean = {:.4f}".format(mean_diff))
            #     plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j], alpha=0.25)
            
            # plt.legend()
            # plt.xticks([])
            # plt.savefig(fname_DTW+"histo2.png", dpi=300, bbox_inches="tight")
            # plt.close()

        """
        Plot histogram of Pearson coefficients across RC (in ascending order of coefficients for data).
        """
        if do_correlations:
            print("\nGenerating correlation histograms...")

            """Pearson histogram"""
            # Load sorted arrays and indices from Santos-Santos.py (original analysis w/o errors).
            sort_args = np.load(f"{analysis_dir}pearson_args.npy")
            pearson_sorted = np.load(f"{analysis_dir}pearson.npy")

            # Rearrange galaxies into ascending order in median of corr(MOND, Vbar).
            # dim = (# of galaxies, 2 x mock_vcomps, 3 x percentiles)
            # mock_sorted = np.array(sorted(pearson_mock, key=lambda x: x[0][0]))
            mock_sorted = np.array(pearson_mock)[sort_args]
            data_sorted = np.array(pearson_data)[sort_args]

            # plt.title("Pearson coefficients across RC")
            hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
            colours = [ 'k', 'mediumblue', 'tab:green' ]
            
            fig, ax = plt.subplots()

            # mean_corr = np.nanmean(pearson_data)
            # ax.bar(galaxies, sorted(pearson_data), color=colours[0], alpha=0.3, label=hist_labels[0])
            # ax.axhline(y=mean_corr, color=colours[0], linestyle='dashed', label="Mean = {:.4f}".format(mean_corr))

            for j in range(3):
                mean_corr = np.mean(pearson_sorted[j])
                ax.plot(galaxies[sort_args], pearson_sorted[j], color=colours[j], label=hist_labels[j])

            trans = Affine2D().translate(-0.15, 0.0) + ax.transData
            ax.errorbar(galaxies[sort_args], data_sorted[:,1], [data_sorted[:,1] - data_sorted[:,0], data_sorted[:,2] - data_sorted[:,1]],
                        fmt='.', ls='none', capsize=2, color=colours[0], alpha=0.7, transform=trans)

            for j in range(2):
                # med_corr = np.nanmean(mock_sorted[:,j,1])
                low_err = mock_sorted[:,j,1] - mock_sorted[:,j,0]
                up_err = mock_sorted[:,j,2] - mock_sorted[:,j,1]

                # low_norm1 = np.full(galaxy_count, np.nanmean(mock_sorted[:,j,2]))
                # up_norm1 = np.full(galaxy_count, np.nanmean(mock_sorted[:,j,0]))

                if j == 1: trans = Affine2D().translate(+0.15, 0.0) + ax.transData
                else: trans = ax.transData
                ax.errorbar(galaxies[sort_args], mock_sorted[:,j,1], [low_err, up_err], fmt='.', ls='none', capsize=2, color=colours[j+1], alpha=0.7, transform=trans)
                # plt.axhline(y=med_corr, color=colours[j+1], linestyle='dashed', label="Mean = {:.4f}".format(med_corr))
                # plt.fill_between(galaxies, low_norm1, up_norm1, color=colours[j+1], alpha=0.25)
            
            ax.legend()
            # ax.set_xticks([])

            # Bold galaxies with features identified by authors + label CLUES galaxies in red.
            for ele in ax.get_xticklabels():
                if ele.get_text() in g_CLUES: ele.set_color('tab:red')
                if ele.get_text() in g_features: ele.set_fontweight('bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

            ax.set_xlabel("Galaxies")
            ax.set_ylabel("Pearson coefficient")
            fig.savefig(fileloc+"correlations/histo1.pdf", dpi=300, bbox_inches="tight")
            plt.close()

    print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
