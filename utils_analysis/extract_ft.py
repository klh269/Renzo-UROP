# (C) 2024 Enoch Ko.
"""
Function for feature extraction.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import mahalanobis


def normalize_residuals(data, errors):
    if errors.ndim == 1:
        # Uncorrelated noise
        return data / errors
    elif errors.ndim == 2:
        # Correlated noise: errors is the full covariance matrix, apply Cholesky whitening to get uncorrelated errors
        # (L is the lower triangular matrix from Cholesky decomposition)
        L, L_T = cho_factor(errors, lower=True)
        return cho_solve((L, L_T), data)  # Equivalent to L^{-1} * data
    else:
        raise ValueError("Errors must be 1D (uncorrelated errors) or 2D (covariance matrix)")


def split_signs(arr):
    """
    Split array into positive and negative segments and extract their indices.
    E.g. Input = [1, 2, -1, 2, 3, 4, -3, -4, 5, 6];
    Returns [[0, 1], [2], [3, 4, 5], [6, 7], [8, 9]].
    """
    split_idx = []
    idx_pos, idx_neg = [], []

    for i in range(len(arr)):
        if arr[i] > 0:
            if len(idx_neg) > 0:
                split_idx.append(idx_neg)
                idx_neg = []
            idx_pos.append(i)
        elif arr[i] < 0:
            if len(idx_pos) > 0:
                split_idx.append(idx_pos)
                idx_pos = []
            idx_neg.append(i)

    # Append remaining segment.
    if len(idx_pos) > 0:
        split_idx.append(idx_pos)
    if len(idx_neg) > 0:
        split_idx.append(idx_neg)

    return split_idx


def ft_check(arr, errV, min_height:float=2.0):
    arr_normalized = normalize_residuals(arr, errV)
    split_idx = split_signs( arr_normalized )

    lb_ft, rb_ft = [], []   # Lists to store left and right boundaries of features.
    for segment in split_idx:
        condition = np.abs(np.array(arr_normalized[segment])) > min_height
        if np.count_nonzero(condition) >= 3:
            lb_ft.append(segment[0])
            rb_ft.append(segment[-1])

    if len(lb_ft) == 0:
        return [], [], []

    lb_features, rb_features = [], []
    i = 0
    while i < len(lb_ft):
        lb_features.append(lb_ft[i])
        if i + 1 < len(lb_ft) and lb_ft[i+1] - rb_ft[i] <= 1:
            rb_features.append(rb_ft[i+1])
            i += 2
        else:
            rb_features.append(rb_ft[i])
            i += 1

    return np.array(lb_features), np.array(rb_features), np.array([rb_features[i] - lb_features[i] for i in range(len(lb_features))])


def get_mahal_dist(arr, cov_inv, lb, rb):
    arr = arr[lb:rb]
    zeros = np.zeros_like(arr)

    mahal_dist = mahalanobis(zeros, arr, cov_inv[lb:rb, lb:rb])
    return mahal_dist


def ft_check_mahanalobis(arr, cov, min_sig:float=6.0):
    dists, left_bases, right_bases = [], [], []
    cov_inv = np.linalg.inv(cov)

    for i in range(1, len(arr)-2):
        mahal_dist = get_mahal_dist(arr, cov_inv, i-1, i+2)
        if mahal_dist > min_sig:
            dists.append(mahal_dist)
            left_bases.append(i-1)
            right_bases.append(i+2)

    return dists, left_bases, right_bases
