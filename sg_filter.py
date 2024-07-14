import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate

from tqdm import tqdm

testing = False

file = "C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/SPARC_Lelli2016c.mrt.txt"

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

galaxy = []

for i in tqdm(range(galaxy_count)):
    g = table["Galaxy"][i]

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

    """
    Apply SG filter to data.
    """
    Vobs_sg = signal.savgol_filter(nVobs, 5, 2, deriv=0)
    Vbar_sg = signal.savgol_filter(nVbar, 5, 2, deriv=0)
    Vgas_sg = signal.savgol_filter(nVgas, 5, 2, deriv=0)
    Vdisk_sg = signal.savgol_filter(nVdisk, 5, 2, deriv=0)
    if bulged:
        Vbul_sg = signal.savgol_filter(nVbul, 5, 2, deriv=0)

    """
    Fit (quartic) splines through data and Vbar (reconstructed).
    """
    # Vobs_spline = interpolate.UnivariateSpline(r, Vobs_sg, k=4, s=0)
    # Vbar_spline = interpolate.UnivariateSpline(r, Vbar_sg, k=4, s=0)
    # Vgas_spline = interpolate.UnivariateSpline(r, Vgas_sg, k=4, s=0)
    # Vdisk_spline = interpolate.UnivariateSpline(r, Vdisk_sg, k=4, s=0)
    # if bulged:
    #     Vbul_spline = interpolate.UnivariateSpline(r, Vbul_sg, k=4, s=0)

    rad = np.linspace(r[0], r[len(r)-1], num=10000)

    plt.title("SG filter: "+g)
    plt.xlabel(r'Normalised radius ($\times R_{eff}$)')
    plt.ylabel('Normalised velocities')
    
    plt.errorbar(r, nVobs, yerr=data["errV"]/Vmax, color='k', fmt="o", capsize=3, alpha=0.3)
    plt.scatter(r, nVbar, color='red', alpha=0.3)
    plt.scatter(r, nVgas, color="green", alpha=0.3)
    plt.scatter(r, nVdisk*np.sqrt(pdisk), color="blue", alpha=0.3)

    plt.plot(r, Vobs_sg, color='k', label="Vobs")
    plt.plot(r, Vbar_sg, color='red', label="Vbar")
    plt.plot(r, Vgas_sg, color='green', label="Vgas")
    plt.plot(r, Vdisk_sg*np.sqrt(pdisk), color='blue', label="Vdisk")
    
    # plt.plot(rad, Vobs_spline(rad), color='k', label="Vobs")
    # plt.plot(rad, Vbar_spline(rad), color='red', label="Vbar")
    # plt.plot(rad, Vgas_spline(rad), color='green', label="Vgas")
    # plt.plot(rad, Vdisk_spline(rad)*np.sqrt(pdisk), color='blue', label="Vdisk")

    if bulged:
        plt.scatter(r, nVbul*np.sqrt(pbul), color="darkorange", alpha=0.3)
        plt.plot(r, Vbul_sg*np.sqrt(pbul), color='darkorange', label="Vbul")
        # plt.plot(rad, Vbul_spline(rad)*np.sqrt(pbul), color='darkorange', label="Vbul")

    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/sg_filter/"+g+".png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
