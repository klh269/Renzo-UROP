import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

galaxy = [ "NGC5055", "NGC5585",
            "NGC6015", "NGC6946", "NGC7331" ]
# galaxy = [ "NGC6946" ]

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
    
    # Plotting rotation curves.
    plt.title(g)
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocities (km/s)")

    plt.scatter(r, data["SBbul"], label="Bulge")
    plt.scatter(r, data["SBdisk"], label="Stellar disc")
    plt.legend()
    
    plt.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/surface_brightness/"+g+".png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

