#!/usr/bin/env python
# (C) 2024 Enoch Ko.
"""
Testing feature extraction algorithm on both versions of NGC 1560.
"""
import numpy as np
import matplotlib.pyplot as plt

fileloc = "/mnt/users/koe/ftalg_tests/"
Sanders_Vbar = np.transpose(np.load(f"{fileloc}Sanders_Vbar.npy"))
Sanders_Vobs = np.transpose(np.load(f"{fileloc}Sanders_Vobs.npy"))
Gentile_Vbar = np.transpose(np.load(f"{fileloc}Gentile_Vbar.npy"))
Gentile_Vobs = np.transpose(np.load(f"{fileloc}Gentile_Vobs.npy"))

sig_ALL = [ Sanders_Vbar, Sanders_Vobs, Gentile_Vbar, Gentile_Vobs ]
# num_segments = [ 6, 4, 7, 8 ]
feature_idx = [ 3, 2, 4, 5 ]

fig, ax = plt.subplots()
powers = np.linspace( 1.0, 2.0, 11, endpoint=True )
labels = [ "Sanders (Vbar)", "Sanders (Vobs)", "Gentile (Vbar)", "Gentile (Vobs)" ]

for i in range(4):
    for j, sums in enumerate(sig_ALL[i]):
        if j == feature_idx[i]: ax.plot( powers, sums, label=labels[i] )
        else: ax.plot( powers, sums, c='k', alpha=0.7 )

ax.legend()
ax.set_xlabel("Powers")
ax.set_ylabel("Segment significance")
fig.savefig(f"{fileloc}NGC1560.png", dpi=300, bbox_inches="tight")
