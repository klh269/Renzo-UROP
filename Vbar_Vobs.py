import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

"""
Triggers for different sections of code. Set variable to True for
    spline_fit:     (Cubic) spline fit through data and Vbar (reconstructed).
    plot_splines:   Produce plots of spline fits.
    plot_d1:        Plot first derivative of spline fits.
"""
finite_diff = True
spline_fit = True
plot_splines = True
plot_d1 = True
plot_d2 = True
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
if testing:
    galaxy_count = 1
bulged_count = 0
xbulge_count = 0

fd_corr = {}
spline_corr = {}
bulged_corr = {}
xbulge_corr = {}
d2_corr = {}
bulged_corr2 = {}
xbulge_corr2 = {}

for i in range(galaxy_count):
    g = table["Galaxy"][i]
    
    """
    Plotting galaxy rotation curves directly from data with variables:
    Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
    """
    file_path = "C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/data/"+g+"_rotmod.dat"
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)
    bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
    r = data["Rad"]
    
    if bulged:
        bulged_count += 1
    else:
        xbulge_count += 1
    
    """
    Compute finite differences directly from discrete data points.
    """
    if finite_diff:
        dVobs_dr = np.diff(data["Vobs"]) / np.diff(r)
        dVbar_dr = np.diff(Vbar(data)) / np.diff(r)
        corr = np.corrcoef(dVobs_dr, dVbar_dr)[0,1]
        fd_corr[g] = corr
    
    """
    Fit (cubic) splines through data and Vbar (reconstructed).
    """
    if spline_fit:
        Vobs = interpolate.UnivariateSpline(r, data["Vobs"])
        Vbar_fit = interpolate.UnivariateSpline(r, Vbar(data))
        Vgas = interpolate.UnivariateSpline(r, data["Vgas"])
        Vdisk = interpolate.UnivariateSpline(r, data["Vdisk"])
        if bulged:
            Vbul = interpolate.UnivariateSpline(r, data["Vbul"])
        
        # First derivative of splines (change 1 to n for nth derivative).
        dVbar = Vbar_fit.derivative(1)
        dVobs = Vobs.derivative(1)
        
        # Compute correlation by taking finite differences between nearby points.
        rad = np.linspace(0., r[len(r)-1], num=10000)
        dVobs_dr = dVobs(rad)
        dVbar_dr = dVbar(rad)
        corr = np.corrcoef(dVobs_dr, dVbar_dr)[0,1]
        spline_corr[g] = corr
        if bulged:
            bulged_corr[g] = corr
        else:
            xbulge_corr[g] = corr
        
        # Second derivative of splines.
        d2Vbar = Vbar_fit.derivative(2)
        d2Vobs = Vobs.derivative(2)
        
        # Compute correlation by taking finite differences between nearby points.
        d2Vobs_dr2 = d2Vobs(rad)
        d2Vbar_dr2 = d2Vbar(rad)
        corr = np.corrcoef(d2Vobs_dr2, d2Vbar_dr2)[0,1]
        d2_corr[g] = corr
        if bulged:
            bulged_corr2[g] = corr
        else:
            xbulge_corr2[g] = corr
    
    """
    Plot spline fits for data;
    Trigger by setting plot_splines = True in top of code.
    """
    if plot_splines:
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
        plt.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/d0/"+g+".png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
    
    """
    Plot first derivatives of the splines;
    Trigger by setting plot_d1 = True in top of code.
    """
    if plot_d1:
        fig1, ax1 = plt.subplots()
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
        fig1.tight_layout()
        plt.show()
        
        fig1.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/d1/"+g+".png", dpi=300, bbox_inches="tight")
        plt.close()
        
    """
    Plot first derivatives of the splines;
    Trigger by setting plot_d1 = True in top of code.
    """
    if plot_d2:
        fig2, ax1 = plt.subplots()
        ax1.set_title("Second derivative: "+g)
        
        color = "tab:red"
        ax1.set_xlabel("Radius (kpc)")
        ax1.set_ylabel("Velocities (km/s)", color=color)
    
        ln1 = ax1.plot(r, d2Vbar(r), color=color, label="Baryonic curve - Vbar")
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = "black"
        ax2.set_ylabel("Velocities (km/s)", color=color)
        ln2 = ax2.plot(r, d2Vobs(r), color=color, label="Total curve - Vobs")
        ax2.tick_params(axis='y', labelcolor=color)
        
        lns = ln1 + ln2
        labels = [l.get_label() for l in lns]
    
        plt.legend(lns, labels) 
        fig2.tight_layout()
        plt.show()
        
        fig2.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/d2/"+g+".png", dpi=300, bbox_inches="tight")
        plt.close()

print("Using finite difference:")
corr_pos = [c for c in fd_corr.values() if c > 0]
pos_rate = len(corr_pos) / galaxy_count
print("Total number of galaxies =", galaxy_count)
print("Number of positive correlations =", len(corr_pos), ", i.e.", round(pos_rate*100, 1), "% of all galaxies.")

print()
print("Using (cubic) spline fits:")
corr_pos = [c for c in spline_corr.values() if c > 0]
pos_rate = len(corr_pos) / galaxy_count
print("No. of positive 1st derivative correlations =", len(corr_pos), ", i.e.", round(pos_rate*100, 1), "% of all galaxies.")
bulged_pos = [c for c in bulged_corr.values() if c > 0]
xbulge_pos = [c for c in xbulge_corr.values() if c > 0]
print("Out of these, there are {} galaxies with bulge ({}%) and {} without ({}%)."
      .format(len(bulged_pos), round(len(bulged_pos)/bulged_count*100, 1), len(xbulge_pos), round(len(xbulge_pos)/xbulge_count*100, 1)))

corr2_pos = [c for c in d2_corr.values() if c > 0]
pos_rate = len(corr2_pos) / galaxy_count
print("No. of positive 2nd derivative correlations =", len(corr2_pos), ", i.e.", round(pos_rate*100, 1), "% of all galaxies.")
bulged_pos = [c for c in bulged_corr2.values() if c > 0]
xbulge_pos = [c for c in xbulge_corr2.values() if c > 0]
print("Out of these, there are {} galaxies with bulge ({}%) and {} without ({}%)."
      .format(len(bulged_pos), round(len(bulged_pos)/bulged_count*100, 1), len(xbulge_pos), round(len(xbulge_pos)/xbulge_count*100, 1)))
