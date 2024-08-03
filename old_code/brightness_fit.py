import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# galaxy = [ "NGC5055", "NGC5585",
#             "NGC6015", "NGC6946", "NGC7331" ]
galaxy = [ "NGC6946" ]

columns = [ "Rad", "Vobs", "errV", "Vgas",
            "Vdisk", "Vbul", "SBdisk", "SBbul" ]

# units_list = [ "kpc", "km/s", "km/s", "km/s",
#          "km/s", "km/s", "L/pc^2", "L/pc^2" ]

# Define constants
G = 4.300e-6    # Gravitational constant G (kpc (km/s)^2 solarM^(-1))
pdisk = np.sqrt(0.5)
pbul = np.sqrt(0.7)

for g in galaxy:
    """
    Plotting galaxy rotation curves directly from data with variables:
    Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
    """
    file_path = "C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/data/"+g+"_rotmod.dat"
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)
    r = data["Rad"]
    print(len(r))
    
    # Cubic spline fits
    SBfit = interpolate.UnivariateSpline(r, data["SBdisk"])
    Vobsfit = interpolate.UnivariateSpline(r, data["Vobs"])
    
    # Plotting the cubic splines.    
    fig1, ax1 = plt.subplots()
    ax1.set_title(g)
    
    color = "tab:red"
    ax1.set_xlabel("Radius (kpc)")
    ax1.set_ylabel(r"Disk surface brightness ($L/pc^2$)", color=color)

    # ax1.scatter(r, data["SBbul"], alpha=0.3, label="Bulge")
    ax1.scatter(r, data["SBdisk"], alpha=0.3)
    ln1 = ax1.plot(r, SBfit(r), color=color, label="Disc surface brightness")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Velocities (km/s)", color=color)
    ax2.scatter(r, data["Vobs"], color='k', alpha=0.3)
    ln2 = ax2.plot(r, Vobsfit(r), color=color, label="Total curve - Vobs")
    ax2.tick_params(axis='y', labelcolor=color)
    
    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]

    plt.legend(lns, labels, loc="center")    
    fig1.tight_layout()
    plt.show()
    
    fig1.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/surface_brightness/spline_"+g+".png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plotting first derivatives of the splines.
    d_SBfit = SBfit.derivative(1)
    d_Vobsfit = Vobsfit.derivative(1)
    
    fig2, ax1 = plt.subplots()
    ax1.set_title("First derivative: "+g)
    
    color = "tab:red"
    ax1.set_xlabel("Radius (kpc)")
    ax1.set_ylabel(r"Disk surface brightness ($L/pc^2$)", color=color)

    ln1 = ax1.plot(r, d_SBfit(r), color=color, label="Disc surface brightness")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Velocities (km/s)", color=color)
    ln2 = ax2.plot(r, d_Vobsfit(r), color=color, label="Total curve - Vobs")
    ax2.tick_params(axis='y', labelcolor=color)
    
    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]

    plt.legend(lns, labels, loc="lower right") 
    fig2.tight_layout()
    plt.show()
    
    fig2.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/surface_brightness/d1_"+g+".png", dpi=300, bbox_inches="tight")
    plt.close()
    