# import pygrc as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm

# galaxy = [ "NGC5055", "NGC5585",
#             "NGC6015", "NGC6946", "NGC7331" ]
galaxy = [ "NGC6946" ]

columns = [ "Rad", "Vobs", "errV", "Vgas",
            "Vdisk", "Vbul", "SBdisk", "SBbul" ]

# units_list = [ "kpc", "km/s", "km/s", "km/s",
#          "km/s", "km/s", "L/pc^2", "L/pc^2" ]

# Define constants
G = 4.300e-6    # Gravitational constant G (kpc (km/s)^2 solarM^(-1))

# Equation for dark matter halo velocity
def halo_v(r, rho0, rc):
    # v = np.sqrt(4*np.pi*G*rho0*rc**2*(1 - rc/r * np.arctan(r/rc))) # Pseudo-isothermal profile
    v = np.sqrt(4*np.pi*G*rho0*rc**3*(np.log((rc + r)/rc) - r/(rc + r))/r) # NFW profile
    # v = v.fillna(0)
    return v

# Total velocity WITHOUT dark matter halo
def total_v_noDM(arr, pdisk, pbul):
    v = np.sqrt( arr["Vgas"]**2
                + (arr["Vdisk"]**2 * pdisk)
                + (arr["Vbul"]**2 * pbul) )
    return v

for g in galaxy:
    """
    Plotting galaxy rotation curves directly from data with variables:
    Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
    """
    file_path = "C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/data/"+g+"_rotmod.dat"
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)
    r = data["Rad"]
    
    # Total velocity WITH dark matter halo component
    def total_v(r, rho0, rc, pdisk, pbul):
        v = np.sqrt( data["Vgas"]**2
                    + (data["Vdisk"]**2 * pdisk)
                    + (data["Vbul"]**2 * pbul)
                    + halo_v(r, rho0, rc)**2 )
        return v
    
    model = lm.Model(total_v)
    params = model.make_params()
    params.add('rho0', value=1e+7, min=0)
    params.add('rc', value=15, min=0.1)
    params.add('pdisk', value=0.5, min=0, max=10)
    params.add('pbul', value=1, min=0, max=10, vary=True)
    
    fit = model.fit(data["Vobs"], params, r=r)
    
    print(fit.fit_report())

    rho0 = fit.params['rho0'].value
    rc = fit.params['rc'].value
    pdisk = fit.params['pdisk'].value
    pbul = fit.params['pbul'].value
    
    fit_params = np.array([ rho0, rc, pdisk, pbul ])
    fit_err = np.array([ fit.params['rho0'].stderr, fit.params['rc'].stderr,
                        fit.params['pdisk'].stderr, fit.params['pbul'].stderr ]) / np.sqrt(len(r))
    
    # Plotting rotation curves.
    plt.title(g)
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocities (km/s)")
 
    plt.errorbar(r, data["Vobs"], yerr=data["errV"], fmt=".", capsize=3, label="Total curve - observed")
    plt.plot(r, data["Vgas"], label="Gas")
    plt.plot(r, data["Vdisk"]*pdisk, label="Stellar disc")
    plt.plot(r, data["Vbul"]*pbul, label="Bulge")
    plt.plot(r, total_v_noDM(data, pdisk, pbul), linestyle="dashed", color="grey", label="Total curve W/O DM")
    plt.plot(r, halo_v(r, rho0, rc), label="Dark matter halo - best fit")
    plt.plot(r, total_v(r, *fit_params), color="black", label="Total curve WITH DM (fit)")
    plt.fill_between(r, total_v(r, *(fit_params-fit_err)), total_v(r, *(fit_params+fit_err)), facecolor="yellow", alpha=0.5, label=r"$1 \sigma$ range from DM fit")
    
    plt.plot([], [], ' ', label="- Disc prefactor = "+str(round(pdisk,2)))
    plt.plot([], [], ' ', label="- Bulge prefactor = "+str(round(pbul,2)))
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    
    # plt.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/DM_NFW/lmfit/"+g+"_lmfit.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
