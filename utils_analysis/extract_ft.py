# (C) 2024 Enoch Ko.
"""
Function for feature extraction.
"""
import numpy as np


# def find_consecutive(arr, stepsize=1):
    # """Split array into groups of consecutive elements."""
    # return np.split(arr, np.where(np.diff(arr) != stepsize)[0]+1)

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
    arr_normalized = arr / errV
    split_idx = split_signs( arr_normalized )

    lb_ft, rb_ft = [], []   # Lists to store left and right boundaries of features.
    for segment in split_idx:
        condition = np.abs(np.array(arr_normalized[segment])) > min_height
        # if np.count_nonzero(condition) >= 1 and len(segment) >= 3:
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

    return lb_features, rb_features, [rb_features[i] - lb_features[i] for i in range(len(lb_features))]


    # lb_peaks, rb_peaks = [], []
    # for segment in idx_peaks:
    #     if np.count_nonzero( np.array(arr_normalized[segment]) > min_height ) >= 3: # and len(segment) >= 3:
    #         lb_peaks.append( segment[0] )
    #         rb_peaks.append( segment[-1] )

    # lb_troughs, rb_troughs = [], []
    # for segment in idx_troughs:
    #     if np.count_nonzero( np.array(arr_normalized[segment]) < - min_height ) >= 3: # and len(segment) >= 3:
    #         lb_troughs.append( segment[0] )
    #         rb_troughs.append( segment[-1] )
            
    # lb_ft, rb_ft = [], []
    # if len(lb_ft) >= 1:
    #     # Check if peaks and troughs can be joined together to form a larger feature (wiggle).
    #     bases = list(itertools.chain(lb_ft, rb_ft))
    #     bases.sort()
    #     ft_bases = []
    #     i = 0
    #     while i < len(bases)-1:
    #         if bases[i+1] - bases[i] != 1:
    #             ft_bases.append(bases[i])
    #             i += 1
    #         else:
    #             i += 2
    #     ft_bases.append(bases[-1])

    #     for ftb in ft_bases:
    #         lb_ft.append(ftb) if ftb in lb_peaks or ftb in lb_troughs else rb_ft.append(ftb)

    # elif len(lb_troughs) >= 1:
    #     lb_ft = lb_troughs
    #     rb_ft = rb_troughs
    # elif len(lb_peaks) >= 1:
    #     lb_ft = lb_peaks
    #     rb_ft = rb_peaks

    # lb_features, rb_features, widths = [], [], []
    # for ib in range(len(lb_ft)):
    #     if rb_ft[ib] - lb_ft[ib] >= 4:
    #         lb_features.append(lb_ft[ib])
    #         rb_features.append(rb_ft[ib])
    #         widths.append(rb_ft[ib] - lb_ft[ib])

    # return lb_features, rb_features, widths


# def ft_check(arr, errV):
#     min_width = 2
#     min_height = 3.0 * errV
#     peaks, prop1 = find_peaks( arr, height=min_height, width=min_width, rel_height=0.5 )
#     troughs, prop2 = find_peaks( -arr, height=min_height, width=min_width, rel_height=0.5 )
#     prop2["peak_heights"]  = - prop2["peak_heights"]
#     prop2["width_heights"] = - prop2["width_heights"]

#     # Merge dictionaries of properties from both peaks and troughs
#     props = [ prop1, prop2 ]
#     properties = {}
#     for key in prop1.keys():
#         properties[key] = np.concatenate( list(properties[key] for properties in props) )

#     return np.concatenate( (peaks, troughs) ), properties
