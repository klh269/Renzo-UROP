import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

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

for i in range(len(table["Galaxy"])):
    g = table["Galaxy"][i]
    
    # Reject galaxies with inclination < 45 degrees and
    # only consider galaxies with quality = 1 (i.e. High).
    if table["Inc"][i] < 45 or table["Q"][i] > 1:
        continue
    
    """
    Plotting galaxy rotation curves directly from data with variables:
    Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
    """
    file_path = "C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/data/"+g+"_rotmod.dat"
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)
    bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
    r = data["Rad"]
    # print("Number of data points for "+g+" = "+str(len(r)))
    
    # Reject galaxies with <10 data points.
    if len(r) < 15:
        # print("(rejected)")
        continue
    
    # Cubic spline fits
    Vobs = interpolate.UnivariateSpline(r, data["Vobs"])
    Vbar_fit = interpolate.UnivariateSpline(r, Vbar(data))
    Vgas = interpolate.UnivariateSpline(r, data["Vgas"])
    Vdisk = interpolate.UnivariateSpline(r, data["Vdisk"])
    if bulged:
        Vbul = interpolate.UnivariateSpline(r, data["Vbul"])
    
    plt.title(g)
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocities (km/s)')
    
    plt.errorbar(r, data["Vobs"], yerr=data["errV"], color='k', fmt="o", capsize=3, alpha=0.3)
    plt.scatter(r, Vbar(data), color='red', alpha=0.3)
    plt.scatter(r, data["Vgas"], color="green", alpha=0.3)
    plt.scatter(r, data["Vdisk"]*np.sqrt(pdisk), color="blue", alpha=0.3)
    
    plt.plot(r, Vobs(r), color='k', label="Vobs")
    plt.plot(r, Vbar_fit(r), color='red', label="Vbar")
    plt.plot(r, Vgas(r), color="green", label="Vgas")
    plt.plot(r, Vdisk(r)*np.sqrt(pdisk), color="blue", label="Vdisk")
    
    if bulged:
        plt.scatter(r, data["Vbul"]*np.sqrt(pbul), color="darkorange", alpha=0.3)
        plt.plot(r, Vbul(r)*np.sqrt(pbul), color="darkorange", label="Vbul")
    
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/"+g+"_d0.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    # # Plotting the cubic splines.    
    # fig1, ax1 = plt.subplots()
    # ax1.set_title(g)
    
    # color = "tab:red"
    # ax1.set_xlabel("Radius (kpc)")
    # ax1.set_ylabel("Velocities (km/s)", color=color)
    
    # ax1.scatter(r, Vbar(data), color=color, alpha=0.3)
    # ln1 = ax1.plot(r, Vbar_fit(r), color=color, label="Baryonic curve - Vbar")
    # ax1.tick_params(axis='y', labelcolor=color)
    
    # ax2 = ax1.twinx()
    # color = "tab:blue"
    # ax2.set_ylabel("Velocities (km/s)", color=color)
    # ax2.scatter(r, data["Vobs"], color='k', alpha=0.3)
    # ln2 = ax2.plot(r, Vobs(r), color=color, label="Total curve - Vobs")
    # ax2.tick_params(axis='y', labelcolor=color)
    
    # lns = ln1 + ln2
    # labels = [l.get_label() for l in lns]

    # plt.legend(lns, labels)    
    # fig1.tight_layout()
    # plt.show()
    
    # fig1.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/"+g+"_d0.png", dpi=300, bbox_inches="tight")
    # plt.close()
    
    # Plotting first derivatives of the splines.
    dVbar = Vbar_fit.derivative(1)
    dVobs = Vobs.derivative(1)
    
    fig2, ax1 = plt.subplots()
    ax1.set_title("First derivative: "+g)
    
    color = "tab:red"
    ax1.set_xlabel("Radius (kpc)")
    ax1.set_ylabel("Velocities (km/s)", color=color)

    ln1 = ax1.plot(r, dVbar(r), color=color, label="Baryonic curve - Vbar")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = "black"
    ax2.set_ylabel("Velocities (km/s)", color=color)
    ln2 = ax2.plot(r, dVobs(r), color=color, label="Total curve - Vobs")
    ax2.tick_params(axis='y', labelcolor=color)
    
    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]

    plt.legend(lns, labels) 
    fig2.tight_layout()
    plt.show()
    
    fig2.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/"+g+"_d1.png", dpi=300, bbox_inches="tight")
    plt.close()
