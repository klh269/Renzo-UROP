import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy import linalg

from tqdm import tqdm

testing = True

file = "C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/SPARC_Lelli2016c.mrt.txt"

SPARC_c = [ "Galaxy", "T", "D", "e_D", "f_D", "Inc",
           "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
            "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
table = pd.read_fwf(file, skiprows=98, names=SPARC_c)

columns = [ "Rad", "Vobs", "errV", "Vgas",
            "Vdisk", "Vbul", "SBdisk", "SBbul" ]

# Define constants
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

galaxy = []

def lowess(x, y, f, iter=5):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

for i in tqdm(range(galaxy_count)):
    g = table["Galaxy"][i]
    g = "NGC5055"

    if g=="D512-2" or g=="D564-8" or g=="D631-7" or g=="NGC4138" or g=="NGC5907" or g=="UGC06818":
        skips += 1
        continue
    
    """
    Plotting galaxy rotation curves directly from data with variables:
    Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
    """
    file_path = "C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/data/"+g+"_rotmod.dat"
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)
    bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
    r = data["Rad"] / table["Rdisk"][i] # Normalised radius (Rdisk = scale length of stellar disk).
    
    # Reject galaxies with less than 5 data points (quartic splines do not work).
    if len(r) < 5:
        skips += 1
        continue
    
    # Normalise velocities by Vmax = max(Vobs).
    Vmax = max(data["Vobs"])
    nVobs = data["Vobs"] / Vmax
    nerrV = data["errV"] / Vmax
    nVbar = Vbar(data) / Vmax
    nVgas = data["Vgas"] / Vmax
    nVdisk = data["Vdisk"] / Vmax
    nVbul = data["Vbul"] / Vmax
    
    if bulged:
        bulged_count += 1
    else:
        xbulge_count += 1

    rad = np.linspace(r[0], r[len(r)-1], num=10000)

    plt.title("SG filter: "+g)
    plt.xlabel(r'Normalised radius ($\times R_{eff}$)')
    plt.ylabel('Normalised velocities')
    
    plt.errorbar(r, nVobs, yerr=data["errV"]/Vmax, color='k', fmt="o", capsize=3, alpha=0.3)
    # plt.scatter(r, nVbar, color='red', alpha=0.3)
    # plt.scatter(r, nVgas, color="green", alpha=0.3)
    # plt.scatter(r, nVdisk*np.sqrt(pdisk), color="blue", alpha=0.3)
    
    Vobs_lowess = lowess(r.to_numpy(), nVobs.to_numpy(), 1/3)

    plt.plot(r, Vobs_lowess, color='k', label="Vobs")
    # plt.plot(rad, Vbar_spline(rad), color='red', label="Vbar")
    # plt.plot(rad, Vgas_spline(rad), color='green', label="Vgas")
    # plt.plot(rad, Vdisk_spline(rad)*np.sqrt(pdisk), color='blue', label="Vdisk")

    # if bulged:
    #     plt.scatter(r, nVbul*np.sqrt(pbul), color="darkorange", alpha=0.3)
    #     plt.plot(rad, Vbul_spline(rad)*np.sqrt(pbul), color='darkorange', label="Vbul")

    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/lowess/"+g+".png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
