import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from resource import getrusage, RUSAGE_SELF


print_features = False
fileloc = "/mnt/users/koe/plots/SPARC_fullGP/"
fname_DTW = fileloc+"dtw/"

def get_data(g:str):
    g_dict = np.load(f"/mnt/users/koe/SPARC_fullGP/{g}.npy", allow_pickle=True).item()

    Vbar_features = g_dict["Vbar_features"]
    Vobs_features = g_dict["Vobs_features"]
    pearson_data = g_dict["pearson_data"]
    pearson_mock = g_dict["pearson_mock"]
    dtw_cost = g_dict["dtw_cost"]

    return Vbar_features, Vobs_features, pearson_data, pearson_mock, dtw_cost


if __name__ == "__main__":
    # Load data
    galaxies = np.load("/mnt/users/koe/gp_fits/galaxy.npy")
    Vbar_ft_count, Vobs_ft_count = 0, 0
    pearson_data, pearson_mock = [], []
    dtw_cost = []

    for g in galaxies:
        Vbar_features, Vobs_features, g_pearson_data, g_pearson_mock, g_dtw_cost = get_data(g)
        pearson_data.append(g_pearson_data)
        pearson_mock.append(g_pearson_mock)
        dtw_cost.append(g_dtw_cost)

        # Check for features in Vbar and Vobs.
        if len(Vbar_features["lb"]) > 0:
            if print_features: print(f"Galaxy {g} has features in Vbar: {Vbar_features}")
            Vbar_ft_count += 1
        if len(Vobs_features["lb"]) > 0:
            if print_features: print(f"Galaxy {g} has features in Vobs: {Vobs_features}")
            Vobs_ft_count += 1
        if print_features: print("")

    print(f"No. of galaxies with features in Vbar: {Vbar_ft_count}")
    print(f"No. of galaxies with features in Vobs: {Vobs_ft_count}")

    """ Plot DTW costs (in ascending order of costs for data). """
    # Arrays of shape (3 x percentiles, 3 x v_comps, galaxy_count).
    dtw_percentiles = np.percentile(dtw_cost, [16.0, 50.0, 84.0], axis=2)
    dtw_percentiles = np.transpose(dtw_percentiles, (0, 2, 1))
    print(np.shape(dtw_percentiles))

    # Rearrange galaxies into ascending order in median of data normalised costs.
    sort_args = np.argsort(dtw_percentiles[1][0])
    sort_args = np.flip(sort_args)
    dtw_percentiles = dtw_percentiles[:, :, sort_args]

    # print(f"Galaxies in descending order of cost(data): {np.array(galaxies)[sort_args]}")

    hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
    colours = [ 'k', 'mediumblue', 'tab:green' ]

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 4)

    # if fname_DTW == fileloc+"dtw/cost_vsVbar/":
    ax.plot(galaxies[sort_args], dtw_percentiles[1][0], color='k', label="Data")

    for j in range(1, 3):
        low_err = dtw_percentiles[1][j] - dtw_percentiles[0][j]
        up_err = dtw_percentiles[2][j] - dtw_percentiles[1][j]

        if j == 1: trans = Affine2D().translate(-0.1, 0.0) + ax.transData
        else: trans = Affine2D().translate(+0.1, 0.0) + ax.transData
        ax.errorbar(galaxies[sort_args], dtw_percentiles[1][j], [low_err, up_err], fmt='.', ls='none',
                    capsize=2, color=colours[j], alpha=0.7, transform=trans, label=hist_labels[j])

    # Only label galaxy with at least one ft in Vobs (bold if 2x ft).
    # new_labels = []
    # for ele in ax.get_xticklabels():
    #     if ele.get_text() in g_doubleft:
    #         ele.set_fontweight('bold')
    #         new_labels.append(ele.get_text())
    #     elif ele.get_text() in g_features: new_labels.append(ele.get_text())
    #     else: new_labels.append("")
    # ax.set_xticklabels(new_labels, rotation=45, ha='right', fontsize=7)

    ax.set_xlabel("Galaxies")
    ax.set_ylabel("Normalized DTW cost")
    fig.savefig(fname_DTW+"dtw_summary.pdf", dpi=300, bbox_inches="tight")
    plt.close()


    """Pearson histogram"""
    # Rearrange galaxies into ascending order in median of corr(MOND, Vbar).
    # dim = (# of galaxies, 2 x mock_vcomps, 3 x percentiles)
    # mock_sorted = np.array(sorted(pearson_mock, key=lambda x: x[0][0]))
    sort_args = np.argsort(pearson_data)
    pearson_data = np.array(pearson_data)[sort_args]
    mock_sorted = np.array(pearson_mock)[sort_args]

    # plt.title("Pearson coefficients across RC")
    hist_labels = [ "Data", "MOND", r"$\Lambda$CDM" ]
    colours = [ 'k', 'mediumblue', 'tab:green' ]

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 4)

    ax.plot(galaxies[sort_args], pearson_data, color='k', label="Data")

    for j in range(2):
        low_err = mock_sorted[:,j,1] - mock_sorted[:,j,0]
        up_err = mock_sorted[:,j,2] - mock_sorted[:,j,1]

        if j == 1: trans = Affine2D().translate(-0.1, 0.0) + ax.transData
        else: trans = Affine2D().translate(+0.1, 0.0) + ax.transData
        ax.errorbar(galaxies[sort_args], mock_sorted[:,j,1], [low_err, up_err], fmt='.', transform=trans,
                    ls='none', capsize=2, color=colours[j+1], alpha=0.7, label=hist_labels[j+1])

    ax.legend()

    # Only label galaxy with at least one ft in Vobs (bold if 2x ft).
    # new_labels = []
    # for ele in ax.get_xticklabels():
    #     if ele.get_text() in g_doubleft:
    #         ele.set_fontweight('bold')
    #         new_labels.append(ele.get_text())
    #     elif ele.get_text() in g_features: new_labels.append(ele.get_text())
    #     else: new_labels.append("")
    # ax.set_xticklabels(new_labels, rotation=45, ha='right', fontsize=7)

    # ax.set_xlabel("Galaxies")
    ax.set_ylabel("Pearson coefficient")
    fig.savefig(fileloc+"pearson_summary.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)