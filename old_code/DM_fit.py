# import pygrc as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# galaxy = [ "NGC5055", "NGC5585",
#             "NGC6015", "NGC6946", "NGC7331" ]
galaxy = [ "NGC2403" ]

columns = [ "Rad", "Vobs", "errV", "Vgas",
            "Vdisk", "Vbul", "SBdisk", "SBbul" ]

# units_list = [ "kpc", "km/s", "km/s", "km/s",
#          "km/s", "km/s", "L/pc^2", "L/pc^2" ]

# Define constants
G = 4.300e-6    # Gravitational constant G (kpc (km/s)^2 solarM^(-1))
pdisk = 0.5
pbul = 0.7

# Equation for dark matter halo velocity
def halo_v(r, rho0, rc):
    # v = np.sqrt(4*np.pi*G*rho0*rc**2*(1 - rc/r * np.arctan(r/rc))) # Pseudo-isothermal profile
    v = np.sqrt(4*np.pi*G*rho0*rc**3*(np.log((rc + r)/rc) - r/(rc + r))/r) # NFW profile
    v = v.fillna(0)
    return v

# Total velocity WITHOUT dark matter halo
def total_v_noDM(arr):
    v = np.sqrt( arr["Vgas"]**2
                + (arr["Vdisk"]**2 * pdisk)
                + (arr["Vbul"]**2 * pbul) )
    return v

for g in galaxy:
    """
    Plotting galaxy rotation curves directly from data with variables:
    Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
    """
    file_path = "C:/Users/admin/Desktop/Other/Oxford UROP 2024/data/"+g+"_rotmod.dat"
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)
    r = data["Rad"]
    print(len(r))
    
    # Total velocity WITH dark matter halo component
    def total_v(r, rho0, rc):
        v = np.sqrt( data["Vgas"]**2
                    + (data["Vdisk"]**2 * pdisk)
                    + (data["Vbul"]**2 * pbul) 
                    + halo_v(r, rho0, rc)**2 )
        return v
    
    parameters, cov = curve_fit(total_v, r, data["Vobs"], sigma=data["errV"], p0=[2e+7, 1.4])
    rho0 = parameters[0]
    rc = parameters[1]
    perr = np.sqrt(np.diag(cov))/np.sqrt(len(r))
    print(parameters)
    print(cov)
    
    # Plotting rotation curves.
    plt.title(g)
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocities (km/s)")
 
    plt.errorbar(r, data["Vobs"], yerr=data["errV"], fmt=".", capsize=3, label="Total curve - observed")
    plt.plot(r, data["Vgas"], label="Gas")
    plt.plot(r, data["Vdisk"]*np.sqrt(pdisk), label="Stellar disc")
    # plt.plot(r, data["Vbul"]*np.sqrt(pbul), label="Bulge")
    plt.plot(r, total_v_noDM(data), linestyle="dashed", color="grey", label="Total curve W/O DM")
    plt.plot(r, halo_v(r, rho0, rc), label="Dark matter halo - best fit")
    plt.plot(r, total_v(r, rho0, rc), color="black", label="Total curve WITH DM (fit)")
    plt.fill_between(r, total_v(r, *(parameters-perr)), total_v(r, *(parameters+perr)), facecolor="yellow", alpha=0.5, label=r"$1\sigma$ confidence spread")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    
    plt.savefig("C:/Users/admin/Desktop/Other/Oxford UROP 2024/plots/DM_NFW/"+g+".png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
