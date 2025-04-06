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

pearson_data = [ 0.89, 0.98, 0.60, 0.82 ]
pearson_med = np.array([ [0.74, 0.30], [0.81, 0.43], [0.50, 0.18], [0.56, 0.23] ])
pearson_err = np.array([ [ [0.08, 0.11], [0.21, 0.23] ], [ [0.13, 0.22], [0.36, 0.55] ],
                         [ [0.09, 0.10], [0.12, 0.12] ], [ [0.10, 0.12], [0.19, 0.23] ] ])

ax[0].scatter( pearson_data, names, c="tab:red", marker='x' )
for i in range(2): 
    if i == 0: trans = Affine2D().translate(0.0, -0.05) + ax[0].transData
    else: trans = Affine2D().translate(0.0, +0.05) + ax[0].transData
    ax[0].errorbar( pearson_med[:,i], names, xerr=np.flip(np.transpose(pearson_err[:,i]), axis=0),
                    c=colours[i], fmt='o', ms=4, ls='none', capsize=4, transform=trans )
    
ax[0].set_xlabel("Pearson coefficient")
ax[0].set_xlim(right=1.0)
# ax[0].grid(axis='y')
ax[0].grid(axis='x', alpha=0.5)

dtw_data = [ 0.31, 0.91, 0.31, 0.57 ]
dtw_med = np.array([ [0.98, 2.05], [0.54, 3.06], [1.02, 1.95], [0.77, 1.98] ])  # [MOND, LCDM]
dtw_err = np.array([ [ [0.19, 0.18], [0.86, 0.73] ], [ [0.22, 0.18], [0.59, 0.56] ],
                     [ [0.13, 0.11], [0.80, 0.64] ], [ [0.15, 0.12], [0.47, 0.47] ] ])

ax[1].scatter( dtw_data, names, c="tab:red", marker='x', label="Data" )
for i in range(2):
    if i == 0: trans = Affine2D().translate(0.0, -0.05) + ax[1].transData
    else: trans = Affine2D().translate(0.0, +0.05) + ax[1]. transData
    ax[1].errorbar( dtw_med[:,i], names, xerr=np.flip(np.transpose(dtw_err[:,i]), axis=0),
                    c=colours[i], fmt='o', ms=4, ls='none', capsize=4, label=labels[i], transform=trans )

ax[1].set_xlabel("DTW cost")
ax[1].set_xlim(left=3.8, right=0.0)
# ax[1].grid(axis='y')
ax[1].grid(axis='x', alpha=0.5)
ax[1].legend()

plt.subplots_adjust(wspace=0.05)
fig.savefig("/mnt/users/koe/plots/NGC1560/summary.pdf", dpi=300, bbox_inches="tight")
plt.close()
