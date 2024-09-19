# (C) 2024 Enoch Ko.
"""
Utilities and functions for correlating curves with Dynamic Time Warping.
"""
import numpy as np
import matplotlib.pyplot as plt

# Dynamic programming code for DTW.
def dtw(dist_mat):
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


def do_DTW(itr:int, length:int, arr1, arr2, max_index:int, window:bool, make_plots=False, file_names=""):
    # arr1 and arr2 are 2D array-like objects of size (iterations x length);
    # this function is to be called within a loop over said iterations.
    # file_names should be a list of strings in the order: [ matrix, alignment ].
    dist_mat_fwd = np.zeros((length, length))
    dist_mat_rev = np.zeros((length, length))
    if window:
        for n in range(length):
            for m in range(length):
                dist_mat_fwd[n, m] = abs(arr1[itr][44+n] - arr2[itr][44+m])
                dist_mat_rev[n, m] = abs(arr1[itr][length-44-n] - arr2[itr][length-44-m])
    else:
        for n in range(length):
            for m in range(length):
                dist_mat_fwd[n, m] = abs(arr1[itr][n] - arr2[itr][m])
                dist_mat_rev[n, m] = abs(arr1[itr][length-n-1] - arr2[itr][length-m-1])
    
    # DTW!
    path_fwd, cost_mat_fwd = dtw(dist_mat_fwd)
    _, cost_mat_rev = dtw(dist_mat_rev)
    cost_fwd = cost_mat_fwd[ length-1, length-1 ]
    cost_rev = cost_mat_rev[ length-1, length-1 ]

    cost = (cost_fwd + cost_rev) / 2
    norm_cost = cost / (length * 2)

    # Plot distance matrix and cost matrix with optimal path.
    if make_plots:
        x_path, y_path = zip(*path_fwd)

        plt.title("Dynamic time warping: Toy model")
        plt.axis('off')

        plt.subplot(121)
        plt.title("Distance matrix")
        plt.imshow(dist_mat_fwd, cmap=plt.cm.binary, interpolation="nearest", origin="lower")

        plt.subplot(122)
        plt.title("Cost matrix")
        plt.imshow(cost_mat_fwd, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
        plt.plot(x_path, y_path)

        plt.savefig(file_names[0], dpi=300, bbox_inches="tight")
        plt.close()

        # Visualize DTW alignment.
        plt.title("DTW alignment: Toy model")

        diff = abs(max(arr2[max_index]) - min(arr1[max_index]))

        if window:
            for x_i, y_j in path_fwd:
                plt.plot([x_i, y_j], [arr1[max_index][44+x_i] + diff, arr2[max_index][44+y_j] - diff], c="C7", alpha=0.4)
                plt.plot(np.arange(length), np.array(arr1[max_index][44:55]) + diff, c='darkblue', label="Vobs")
                plt.plot(np.arange(length), np.array(arr2[max_index][44:55]) - diff, c="red", label="Vbar")
        else:
            for x_i, y_j in path_fwd:
                plt.plot([x_i, y_j], [arr1[max_index][x_i] + diff, arr2[max_index][y_j] - diff], c="C7", alpha=0.4)
                plt.plot(np.arange(length), arr1[max_index] + diff, c='k', label="Vobs")
                plt.plot(np.arange(length), arr2[max_index] - diff, c="red", label="Vbar")

        plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(cost))
        plt.plot([], [], c='w', label="Normalized cost = {:.4f}".format(norm_cost))

        plt.axis("off")
        plt.legend(bbox_to_anchor=(1,1))
        plt.savefig(file_names[1], dpi=300, bbox_inches="tight")
        plt.close()

    return norm_cost
