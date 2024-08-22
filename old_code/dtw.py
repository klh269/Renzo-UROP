#!/usr/bin/env python
"""
Compare Vbar and Vobs using dynamic time warping.
Dynamic programming function (dp) taken from Herman Kamper:
https://github.com/kamperh/lecture_dtw_notebook/blob/main/dtw.ipynb

He also has a very helpful YouTube tutorial on DTW:
https://www.youtube.com/playlist?list=PLmZlBIcArwhMJoGk5zpiRlkaHUqy5dLzL 
"""

from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.spatial.distance as dist
from scipy import interpolate, signal
import pandas as pd
from tqdm import tqdm

# Trigger for code testing; "testing = True" runs the main for-loop with only the first galaxy.
testing = False
test_galaxy = "DDO064"
makeplots = True

directory = "/mnt/users/koe/plots/dtw/"
file = "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt"

SPARC_c = [ "Galaxy", "T", "D", "e_D", "f_D", "Inc",
           "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
            "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
table = pd.read_fwf(file, skiprows=98, names=SPARC_c)

columns = [ "Rad", "Vobs", "errV", "Vgas",
            "Vdisk", "Vbul", "SBdisk", "SBbul" ]

# Define constants
G = 4.300e-6    # Gravitational constant G (kpc (km/s)^2 solarM^(-1))
pdisk = 0.5
pbul = 0.7

def Vbar(arr):
    v = np.sqrt( arr["Vgas"]**2
                + (arr["Vdisk"]**2 * pdisk)
                + (arr["Vbul"]**2 * pbul) )
    return v

galaxy_count = len(table["Galaxy"])
skips = 0
if testing:
    galaxy_count = 1
bulged_count = 0
xbulge_count = 0

def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

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

for i in tqdm(range(galaxy_count)):
    g = table["Galaxy"][i]
    if testing:
        g = test_galaxy

    if g=="D512-2" or g=="D564-8" or g=="D631-7" or g=="NGC4138" or g=="NGC5907" or g=="UGC06818":
        skips += 1
        continue

    file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)
    bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
    r = data["Rad"] / table["Reff"][i] # Normalised radius (Reff = scale length of stellar disk).
    
    # Reject galaxies with less than 10 data points (correlations are not meaningful).
    if len(r) < 10:
        skips += 1
        continue
    
    # Normalise velocities by Vmax = max(Vobs).
    Vmax = max(data["Vobs"])
    v_components = [ data["Vobs"], Vbar(data), data["Vgas"], data["Vdisk"]*np.sqrt(pdisk) ]
    if bulged:
        v_components.append( data["Vbul"]*np.sqrt(pbul) )
    v_components = np.array(v_components)
    v_components /= Vmax
    nerrV = data["errV"] / Vmax
    
    if bulged:
        bulged_count += 1
    else:
        xbulge_count += 1

    # Interpolate the data with cubic Hermite spline splines.
    rad_count = math.ceil((max(r)-min(r))*100)
    rad = np.linspace(min(r), max(r), rad_count)
    v_d0, v_d1, v_d2 = [], [], []
    for v_comp in v_components:
        v_d0.append(interpolate.pchip_interpolate(r, v_comp, rad))
        # v_d1.append(interpolate.pchip_interpolate(r, v_comp, rad, der=1))
        # v_d2.append(interpolate.pchip_interpolate(r, v_comp, rad, der=2))

    # Apply SG filter to interpolated data and derivatives with window size = 0.2*Reff.
    d0_sg, d1_sg, d2_sg = [], [], []
    for v_comp0 in v_d0:
        d0_sg.append(signal.savgol_filter(v_comp0, 50, 2))
    # for v_comp1 in v_d1:
    #     d1_sg.append(signal.savgol_filter(v_comp1, 50, 2))
    # for v_comp2 in v_d2:
    #     d2_sg.append(signal.savgol_filter(v_comp2, 50, 2))

    """
    Set up and run dynamic time warping.
    """
    # Construct distance matrix.
    dist_mat0 = np.zeros((rad_count, rad_count))
    # dist_mat1 = np.zeros((rad_count, rad_count))
    # dist_mat2 = np.zeros((rad_count, rad_count))
    for n in range(rad_count):
        for m in range(rad_count):
            dist_mat0[n, m] = abs(d0_sg[0][n] - d0_sg[1][m])
            # dist_mat1[n, m] = abs(d1_sg[0][n] - d1_sg[1][m])
            # dist_mat2[n, m] = abs(d2_sg[0][n] - d2_sg[1][m])

    # DTW.
    path, cost_mat = dp(dist_mat0)
    x_path, y_path = zip(*path)
    # print("\nGalaxy "+g+" ("+str(i+1)+"/175):")
    # print("Alignment cost: {:.4f}".format(cost_mat[ rad_count-1, rad_count-1 ]))
    # print("Normalized alignment cost: {:.4f}".format(cost_mat[ rad_count-1, rad_count-1 ]/(rad_count*2)))
    
    if makeplots:
        # Plot distance matrix and cost matrix with optimal path.
        plt.title("Dynamic time warping: "+g)
        plt.figure(figsize=(6, 4))
        plt.subplot(121)
        plt.title("Distance matrix")
        plt.imshow(dist_mat0, cmap=plt.cm.binary, interpolation="nearest", origin="lower")

        plt.subplot(122)
        plt.title("Cost matrix")
        plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
        plt.plot(x_path, y_path)

        plt.savefig(directory+"d0/cost_matrix/"+g+".png", dpi=300, bbox_inches="tight")
        plt.close()

        # Visualize DTW alignment.
        colors = ["k", "red"]
        labels = ["Vobs", "Vbar"]
        plt.figure()
        plt.title("DTW alignment: "+g)

        for x_i, y_j in path:
            if x_i%(math.ceil(rad_count/100))==0 or y_j%(math.ceil(rad_count/100))==0:
                plt.plot([x_i, y_j], [d0_sg[0][x_i] + 0.5, d0_sg[1][y_j] - 0.5], c="C7", alpha=0.4)
        plt.plot(np.arange(rad_count), d0_sg[0] + 0.5, c=colors[0], label=labels[0])
        plt.plot(np.arange(rad_count), d0_sg[1] - 0.5, c=colors[1], label=labels[1])

        plt.axis("off")
        plt.legend()
        plt.savefig(directory+"d0/"+g+".png", dpi=300)
        plt.close()

        # Plot (fitered) spline fits for Vbar and Vobs, and their transformed curves under DTW.
        # fig0, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        # ax1.set_title("Dynamic time warping: "+g)
        # ax1.set_ylabel("Normalised velocities")

        # ax1.scatter(r, v_components[0], color=colors[0], alpha=0.3)
        # ln1 = ax1.plot(rad, d0_sg[0], color=colors[0], label=labels[0])
        # ax1.scatter(r, v_components[1], color=colors[1], alpha=0.3)
        # ln2 = ax1.plot(rad, d0_sg[1], color=colors[1], label=labels[1])
        
        # lns = ln1 + ln2
        # labels = [l.get_label() for l in lns]
        # ax1.legend(lns, labels)

        # ax2.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
        # ax2.set_ylabel("Normalised velocities")
        # ln3 = ax2.plot(rad[x_path], d0_sg[0][x_path], color=colors[0], label=labels[0])
        # ln4 = ax2.plot(rad[y_path], d0_sg[1][y_path], color=colors[1], label=labels[1])

        # lns = ln3 + ln4
        # labels = [l.get_label() for l in lns]
        # ax2.legend(lns, labels)

        # plt.subplots_adjust(hspace=0.05)
        # fig0.savefig(directory+"d0/"+g+".png", dpi=300, bbox_inches="tight")
        # plt.close()
