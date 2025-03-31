#!/usr/bin/env python
# (C) 2024 Enoch Ko.
"""
Summary statistics for NGC 1560 (from output of analyze_NGC1560.py).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D


fig, ax = plt.subplots( 1, 2, sharey=True )
fig.set_size_inches( 10, 4 )

names = [ "Gentile et al. (window)", "Santos (window)", "Gentile et al.", "Santos" ]
colours = [ "mediumblue", "tab:green" ]
labels = [ "MOND", r"$\Lambda$CDM" ]

pearson_data = [ 0.89, 0.96, 0.61, 0.81 ]
pearson_med = np.array([ [0.74, 0.30], [0.79, 0.36], [0.50, 0.21], [0.54, 0.22] ])
pearson_err = np.array([ [ [0.08, 0.11], [0.21, 0.23] ], [ [0.10, 0.14], [0.27, 0.29] ],
                         [ [0.09, 0.10], [0.12, 0.13] ], [ [0.10, 0.12], [0.17, 0.22] ] ])

ax[0].scatter( pearson_data, names, c="tab:red", marker='x' )
for i in range(2): 
    # if i == 0: trans = Affine2D().translate(0.0, -0.05) + ax[0].transData
    # else: trans = Affine2D().translate(0.0, +0.05) + ax[0]. transData
    ax[0].errorbar( pearson_med[:,i], names, xerr=np.flip(np.transpose(pearson_err[:,i]), axis=0),
                    c=colours[i], fmt='o', ls='none', capsize=4)    # , transform=trans )
    
ax[0].set_xlabel("Pearson coefficient")
ax[0].set_xlim(right=1.0)
ax[0].grid(axis='y')

dtw_data = [ 0.31, 0.47, 0.31, 0.57 ]
dtw_med = np.array([ [0.98, 2.05], [0.59, 2.64], [1.02, 1.95], [0.77, 1.98] ])  # [MOND, LCDM]
dtw_err = np.array([ [ [0.19, 0.18], [0.86, 0.73] ], [ [0.15, 0.15], [0.52, 0.56] ],
                     [ [0.13, 0.11], [0.80, 0.64] ], [ [0.15, 0.12], [0.47, 0.47] ] ])

ax[1].scatter( dtw_data, names, c="tab:red", marker='x', label="Data" )
for i in range(2):
    # if i == 0: trans = Affine2D().translate(0.0, -0.05) + ax[1].transData
    # else: trans = Affine2D().translate(0.0, +0.05) + ax[1]. transData
    ax[1].errorbar( dtw_med[:,i], names, xerr=np.flip(np.transpose(dtw_err[:,i]), axis=0),
                    c=colours[i], fmt='o', ls='none', capsize=4, label=labels[i]) #, transform=trans )

ax[1].set_xlabel("DTW cost (axis reversed)")
ax[1].set_xlim(left=3.45, right=0.0)
ax[1].grid(axis='y')
ax[1].legend()

plt.subplots_adjust(wspace=0.05)
fig.savefig("/mnt/users/koe/plots/NGC1560/DTW_summary.pdf", dpi=300, bbox_inches="tight")
plt.close()
