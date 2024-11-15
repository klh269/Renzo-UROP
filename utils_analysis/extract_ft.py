# (C) 2024 Enoch Ko.
"""
Function for feature extraction.
"""
import numpy as np
import itertools
# from scipy.signal import find_peaks


def find_consecutive(arr, stepsize=1):
    """Split array into groups of consecutive elements."""
    return np.split(arr, np.where(np.diff(arr) != stepsize)[0]+1)

def ft_check(arr, errV, min_height:float=2.0):
    idx_pos = np.where( arr > 0.0 )[0]
    idx_peaks = find_consecutive( idx_pos )
    idx_neg = np.where( arr < 0.0 )[0]
    idx_troughs = find_consecutive( idx_neg )

    lb_peaks, rb_peaks = [], []
    for segment in idx_peaks:
        if np.count_nonzero( np.array(arr[segment]) > min_height * np.array(errV[segment]) ) >= 1 and len(segment) >= 3:
            lb_peaks.append( segment[0] )
            rb_peaks.append( segment[-1] )

    lb_troughs, rb_troughs = [], []
    for segment in idx_troughs:
        if np.count_nonzero( np.array(arr[segment]) < - min_height * np.array(errV[segment]) ) >= 1 and len(segment) >= 3:
            lb_troughs.append( segment[0] )
            rb_troughs.append( segment[-1] )
            
    lb_ft, rb_ft = [], []
    if len(lb_troughs) >= 1 and len(lb_peaks) >= 1:
        # Check if peaks and troughs can be joined together to form a larger feature (wiggle).
        bases = list(itertools.chain(lb_peaks, rb_peaks, lb_troughs, rb_troughs))
        bases.sort()
        ft_bases = []
        i = 0
        while i < len(bases)-1:
            if bases[i+1] - bases[i] != 1:
                ft_bases.append(bases[i])
                i += 1
            else:
                i += 2
        ft_bases.append(bases[-1])

        for ftb in ft_bases:
            lb_ft.append(ftb) if ftb in lb_peaks or ftb in lb_troughs else rb_ft.append(ftb)

    elif len(lb_troughs) >= 1:
        lb_ft = lb_troughs
        rb_ft = rb_troughs
    elif len(lb_peaks) >= 1:
        lb_ft = lb_peaks
        rb_ft = rb_peaks

    lb_features, rb_features, widths = [], [], []
    for ib in range(len(lb_ft)):
        if rb_ft[ib] - lb_ft[ib] >= 4:
            lb_features.append(lb_ft[ib])
            rb_features.append(rb_ft[ib])
            widths.append(rb_ft[ib] - lb_ft[ib])

    return lb_features, rb_features, widths


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
