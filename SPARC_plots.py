# import pygrc as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

galaxy = [ "NGC4217", "NGC5055", "NGC5585",
            "NGC6015", "NGC6946", "NGC7331" ]
# galaxy = [ "NGC6946" ]

# df_all = []

columns = [ "Rad", "Vobs", "errV", "Vgas",
            "Vdisk", "Vbul", "SBdisk", "SBbul" ]

units_list = [ "kpc", "km/s", "km/s", "km/s",
         "km/s", "km/s", "L/pc^2", "L/pc^2" ]

c_photo = [ "radius", "mu", "kill", "error" ]

units = {}
for i in range(8):
    units[columns[i]] = units_list[i]

for g in galaxy:
    file_path = "C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/data/"+g+"_rotmod.dat"
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)
    # df_all.append(data)
    # photo_path = "C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/photometric profiles/"+g+".sfb"
    # photo_raw = np.loadtxt(photo_path, skiprows=1)
    # photo_data = pd.DataFrame(photo_raw, columns=c_photo)
    # print(photo_data)
    plt.title(g)
    # plt.ylim(0, 400)
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocities (km/s)")
    plt.errorbar(data["Rad"], data["Vobs"], yerr=data["errV"], capsize=1.5, label="Vobs")
    plt.plot(data["Rad"], data["Vgas"], label="Vgas")
    plt.plot(data["Rad"], data["Vdisk"], label="Vdisk")
    plt.plot(data["Rad"], data["Vbul"], label="Vbul")
    # plt.plot(data["Rad"], np.sqrt(data["Vgas"]**2+data["Vdisk"]**2+data["Vbul"]**2), label="Vtot")
    plt.legend()
    # plt.plot(data["Rad"], data["Vgas"], label="Vobs")
    # plt.plot(photo_data["radius"], photo_data["mu"])
    # print(data)
    # plt.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/test_plots/"+g)
    plt.show()
    plt.close()
    
    plt.title("Sufrace Brightness: "+g)
    plt.xlabel("Radius (kpc)")
    plt.ylabel(r"Luminosity ($L_\odot /pc^2$)")
    # plt.plot(data["Rad"], data["Vobs"], label="Vobs")
    plt.plot(data["Rad"], data["SBdisk"], label="SBdisk")
    plt.plot(data["Rad"], data["SBbul"], label="SBbul")
    plt.legend()
    plt.show()
    plt.close()
