#!/usr/bin/env python
# (C) 2024 Enoch Ko.
"""
Summary statistics for NGC 1560 (from output of analyze_NGC1560.py).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


fig, ax = plt.subplots( 1, 2, sharey=True )
fig.set_size_inches( 10, 3 )

names = [ "Gentile et al. (window)", "Sanders (window)", "Gentile et al.", "Sanders" ]
colours = [ "mediumblue", "tab:green" ]
labels = [ "MOND", r"$\Lambda$CDM" ]

pearson_data = [ 0.82, 0.96, 0.62, 0.83 ]
pearson_med = np.array([ [0.22, 0.10], [0.81, 0.49], [0.44, 0.35], [0.70, 0.43] ])
pearson_err = np.array([ [ [0.25, 0.23], [0.25, 0.22] ], [ [0.13, 0.09], [0.31, 0.25] ],
                         [ [0.13, 0.11], [0.14, 0.12] ], [ [0.09, 0.08], [0.13, 0.12] ] ])

# Calculate significances for Pearson
pearson_avg_err = np.mean(pearson_err, axis=2)  # Average error for each pair
sig_MOND = (pearson_data - pearson_med[:,0]) / pearson_avg_err[:,0]
sig_LCDM = (pearson_data - pearson_med[:,1]) / pearson_avg_err[:,1]
print("MOND significances (Pearson):\n", np.round(sig_MOND, 2))
print("LCDM significances (Pearson):\n", np.round(sig_LCDM, 2))

ax[0].scatter( pearson_data, names, c="tab:red", marker='x', label="Data" )
for i in range(2): 
    if i == 0: trans = Affine2D().translate(0.0, -0.09) + ax[0].transData
    else: trans = Affine2D().translate(0.0, +0.09) + ax[0].transData
    ax[0].errorbar( pearson_med[:,i], names, xerr=np.flip(np.transpose(pearson_err[:,i]), axis=0),
                    c=colours[i], fmt='o', ms=4, ls='none', capsize=4, label=labels[i], transform=trans )
    
ax[0].set_xlabel("Pearson coefficient")
ax[0].set_xlim(right=1.0)
# ax[0].grid(axis='y')
ax[0].grid(axis='x', alpha=0.5)

dtw_data = [ 0.35, 0.65, 0.30, 0.62 ]
dtw_med = np.array([ [1.07, 1.00], [0.64, 0.67], [0.99, 0.95], [0.71, 0.70] ])  # [MOND, LCDM]
dtw_err = np.array([ [ [0.22, 0.27], [0.20, 0.22] ], [ [0.17, 0.23], [0.18, 0.25] ],
                     [ [0.11, 0.13], [0.12, 0.11] ], [ [0.13, 0.16], [0.13, 0.14] ] ])  # [low_err, up_err]

# Calculate significances for DTW
dtw_avg_err = np.mean(dtw_err, axis=2)  # Average error for each pair
sig_MOND_dtw = (dtw_data - dtw_med[:,0]) / dtw_avg_err[:,0]
sig_LCDM_dtw = (dtw_data - dtw_med[:,1]) / dtw_avg_err[:,1]
print("MOND significances (DTW):\n", np.round(sig_MOND_dtw, 2))
print("LCDM significances (DTW):\n", np.round(sig_LCDM_dtw, 2))

ax[1].scatter( dtw_data, names, c="tab:red", marker='x', label="Data" )
for i in range(2):
    if i == 0: trans = Affine2D().translate(0.0, -0.09) + ax[1].transData
    else: trans = Affine2D().translate(0.0, +0.09) + ax[1]. transData
    ax[1].errorbar( dtw_med[:,i], names, xerr=np.flip(np.transpose(dtw_err[:,i]), axis=0),
                    c=colours[i], fmt='o', ms=4, ls='none', capsize=4, label=labels[i], transform=trans )

ax[1].set_xlabel("DTW cost")
ax[1].set_xlim(left=1.5, right=0.0)
# ax[1].grid(axis='y')
ax[1].grid(axis='x', alpha=0.5)

ax[0].legend()

plt.subplots_adjust(wspace=0.05)
fig.savefig("/mnt/users/koe/plots/NGC1560_fullGP/summary.pdf", dpi=300, bbox_inches="tight")
plt.close()
