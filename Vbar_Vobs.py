import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.stats
from tabulate import tabulate

"""
Triggers for different sections of code. Set variable to True for
    spline_fit:     (Quartic) spline fit through data and Vbar (reconstructed).
    plot_splines:   Produce plots of spline fits.
    plot_d1:        Plot first derivative of the splines of ONLY Vobs and Vbar.
    plot_d1_ALL:    Plot first derivative of ALL splines.
    plot_d2:        Plot second derivative of the splines of ONLY Vobs and Vbar.
    plot_d2_ALL:    Plot second derivative of ALL splines.
    testing:        Run the for-loop through only the first few galaxies and
                    stop printing correlation statistics and table (as txt files).
"""
finite_diff = True
spline_fit = True
plot_splines = True
plot_d1 = True
plot_d1_ALL = True
plot_d2 = True
plot_d2_ALL = True
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
    galaxy_count = 5
bulged_count = 0
xbulge_count = 0

galaxy = []
fd_corr = []
spline_corr = []
bulged_corr = []
xbulge_corr = []
d2_corr = []
bulged_corr2 = []
xbulge_corr2 = []

for i in range(galaxy_count):
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
    nVbar = Vbar(data) / Vmax
    nVgas = data["Vgas"] / Vmax
    nVdisk = data["Vdisk"] / Vmax
    nVbul = data["Vbul"] / Vmax
    
    if bulged:
        bulged_count += 1
    else:
        xbulge_count += 1
    
    """
    Compute finite differences directly from discrete data points.
    """
    if finite_diff:
        dVobs_dr = np.diff(nVobs) / np.diff(r)
        dVbar_dr = np.diff(nVbar) / np.diff(r)
        corr = scipy.stats.spearmanr(dVobs_dr, dVbar_dr)[0]
        fd_corr.append(corr)
    
    """
    Fit (quartic) splines through data and Vbar (reconstructed).
    """
    if spline_fit:
        Vobs = interpolate.UnivariateSpline(r, nVobs, k=4)
        Vbar_fit = interpolate.UnivariateSpline(r, nVbar, k=4)
        Vgas = interpolate.UnivariateSpline(r, nVgas, k=4)
        Vdisk = interpolate.UnivariateSpline(r, nVdisk, k=4)
        if bulged:
            Vbul = interpolate.UnivariateSpline(r, nVbul, k=4)
        
        # First derivative of splines (change 1 to n for nth derivative).
        dVbar = Vbar_fit.derivative(1)
        dVobs = Vobs.derivative(1)
        dVgas = Vgas.derivative(1)
        dVdisk = Vdisk.derivative(1)
        if bulged:
            dVbul = Vbul.derivative(1)
        
        # Compute correlation of Vbar vs Vobs by taking
        # finite differences between nearby points.
        rad = np.linspace(0., r[len(r)-1], num=10000)
        dVobs_dr = dVobs(rad)
        dVbar_dr = dVbar(rad)
        corr = scipy.stats.spearmanr(dVobs_dr, dVbar_dr)[0] # Spearman's rho
        spline_corr.append(corr)
        if bulged:
            bulged_corr.append(corr)
            xbulge_corr.append(-np.inf)
        else:
            bulged_corr.append(-np.inf)
            xbulge_corr.append(corr)
        
        # Second derivative of splines.
        d2Vbar = Vbar_fit.derivative(2)
        d2Vobs = Vobs.derivative(2)
        d2Vgas = Vgas.derivative(2)
        d2Vdisk = Vdisk.derivative(2)
        if bulged:
            d2Vbul = Vbul.derivative(2)
        
        # Compute correlation by taking finite differences between nearby points.
        d2Vobs_dr2 = d2Vobs(rad)
        d2Vbar_dr2 = d2Vbar(rad)
        corr = scipy.stats.spearmanr(d2Vobs_dr2, d2Vbar_dr2)[0]
        d2_corr.append(corr)
        if bulged:
            bulged_corr2.append(corr)
            xbulge_corr2.append(-np.inf)
        else:
            bulged_corr2.append(-np.inf)
            xbulge_corr2.append(corr)
    
    """
    Plot spline fits for data;
    Trigger by setting plot_splines = True in top of code.
    """
    if plot_splines:
        plt.title(g)
        plt.xlabel(r'Normalised radius ($\times$Reff)')
        plt.ylabel('Normalised velocities')
        
        plt.errorbar(r, nVobs, yerr=data["errV"]/Vmax, color='k', fmt="o", capsize=3, alpha=0.3)
        plt.scatter(r, nVbar, color='red', alpha=0.3)
        plt.scatter(r, nVgas, color="green", alpha=0.3)
        plt.scatter(r, nVdisk*np.sqrt(pdisk), color="blue", alpha=0.3)
        
        plt.plot(rad, Vobs(rad), color='k', label="Vobs")
        plt.plot(rad, Vbar_fit(rad), color='red', label="Vbar")
        plt.plot(rad, Vgas(rad), color="green", label="Vgas")
        plt.plot(rad, Vdisk(rad)*np.sqrt(pdisk), color="blue", label="Vdisk")
        
        if bulged:
            plt.scatter(r, nVbul*np.sqrt(pbul), color="darkorange", alpha=0.3)
            plt.plot(rad, Vbul(rad)*np.sqrt(pbul), color="darkorange", label="Vbul")
        
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/d0/"+g+".png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
    
    """
    Plot first derivatives of the splines for Vobs and Vbar;
    Trigger by setting plot_d1 = True in top of code.
    """
    if plot_d1:
        fig1, ax1 = plt.subplots()
        ax1.set_title("First derivative: "+g)
        
        color = "tab:red"
        ax1.set_xlabel(r"Normalised radius ($\times$Reff)")
        ax1.set_ylabel("Normalised velocities", color=color)
    
        ln1 = ax1.plot(rad, dVbar(rad), color=color, label="Baryonic curve - Vbar")
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = "black"
        ax2.set_ylabel("Velocities (km/s)", color=color)
        ln2 = ax2.plot(rad, dVobs(rad), color=color, label="Total curve - Vobs")
        ax2.tick_params(axis='y', labelcolor=color)
        
        lns = ln1 + ln2
        labels = [l.get_label() for l in lns]
    
        plt.legend(lns, labels) 
        fig1.tight_layout()
        plt.show()
        
        fig1.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/d1/"+g+".png", dpi=300, bbox_inches="tight")
        plt.close()
    
    """
    Plot first derivatives of ALL the splines;
    Trigger by setting plot_d1_ALL = True in top of code.
    """
    if plot_d1_ALL:
        plt.title("First derivative: "+g)
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Velocities (km/s)')
        
        plt.plot(rad, dVobs(rad), color='k', label="Vobs")
        plt.plot(rad, dVbar(rad), color='red', label="Vbar")
        plt.plot(rad, dVgas(rad), color="green", label="Vgas")
        plt.plot(rad, dVdisk(rad)*np.sqrt(pdisk), color="blue", label="Vdisk")
        
        if bulged:
            plt.plot(rad, dVbul(rad)*np.sqrt(pbul), color="darkorange", label="Vbul")
        
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/d1_ALL/"+g+".png", dpi=300, bbox_inches="tight")
        plt.show()
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
    
        ln1 = ax1.plot(rad, d2Vbar(rad), color=color, label="Baryonic curve - Vbar")
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = "black"
        ax2.set_ylabel("Velocities (km/s)", color=color)
        ln2 = ax2.plot(rad, d2Vobs(rad), color=color, label="Total curve - Vobs")
        ax2.tick_params(axis='y', labelcolor=color)
        
        lns = ln1 + ln2
        labels = [l.get_label() for l in lns]
    
        plt.legend(lns, labels) 
        fig2.tight_layout()
        plt.show()
        
        fig2.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/d2/"+g+".png", dpi=300, bbox_inches="tight")
        plt.close()
        
        """
        Plot first derivatives of ALL the splines;
        Trigger by setting plot_d1_ALL = True in top of code.
        """
        if plot_d2_ALL:
            plt.title("Second derivative: "+g)
            plt.xlabel('Radius (kpc)')
            plt.ylabel('Velocities (km/s)')
            
            plt.plot(rad, d2Vobs(rad), color='k', label="Vobs")
            plt.plot(rad, d2Vbar(rad), color='red', label="Vbar")
            plt.plot(rad, d2Vgas(rad), color="green", label="Vgas")
            plt.plot(rad, d2Vdisk(rad)*np.sqrt(pdisk), color="blue", label="Vdisk")
            
            if bulged:
                plt.plot(rad, d2Vbul(rad)*np.sqrt(pbul), color="darkorange", label="Vbul")
            
            plt.legend(bbox_to_anchor=(1,1), loc="upper left")
            plt.savefig("C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/d2_ALL/"+g+".png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()
     
    # Make a list of the used galaxies.
    galaxy.append(g)

if not testing:
    # Write all correlations into a table and save to txt file (correlations.txt, same directory as the plots)
    corr_arrays = np.array([galaxy, fd_corr, spline_corr, bulged_corr, xbulge_corr, d2_corr, bulged_corr2, xbulge_corr2])
    corr_arrays = np.transpose(corr_arrays)
    corr_arrays = corr_arrays.reshape(galaxy_count,8)
    header = [ "Galaxy", "Finite difference", "Spline: 1st derivative",
                "w/ bulge","w/o bulge", "Spline: 2nd derivative", "w/ bulge", "w/o bulge" ]
    corr_table = pd.DataFrame(corr_arrays, columns=header)
    
    corr_table.to_csv('C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/correlations.txt', sep='\t', index=False)
    with open('C:/Users/admin/OneDrive/Desktop/Other/Oxford UROP 2024/plots/Vbar_Vobs/corr_table.txt', 'w') as f:
        f.write(tabulate(corr_table, headers=header))
    
    # Some useful info regarding percentages of positive correlations.
    galaxy_count = len(galaxy)
    print("Using finite difference:")
    corr_pos = [c for c in fd_corr if c > 0]
    pos_rate = len(corr_pos) / galaxy_count
    print("Total number of galaxies =", galaxy_count)
    print("Number of positive correlations =", len(corr_pos), ", i.e.", round(pos_rate*100, 1), "% of all galaxies.")
    
    print()
    print("Using (cubic) spline fits:")
    corr_pos = [c for c in spline_corr if c > 0]
    pos_rate = len(corr_pos) / galaxy_count
    print("No. of positive 1st derivative correlations =", len(corr_pos), ", i.e.", round(pos_rate*100, 1), "% of all galaxies.")
    bulged_pos = [c for c in bulged_corr if c > 0]
    xbulge_pos = [c for c in xbulge_corr if c > 0]
    print("Out of these, there are {} galaxies with bulge ({}%) and {} without ({}%)."
          .format(len(bulged_pos), round(len(bulged_pos)/bulged_count*100, 1), len(xbulge_pos), round(len(xbulge_pos)/xbulge_count*100, 1)))
    
    corr2_pos = [c for c in d2_corr if c > 0]
    pos_rate = len(corr2_pos) / galaxy_count
    print("No. of positive 2nd derivative correlations =", len(corr2_pos), ", i.e.", round(pos_rate*100, 1), "% of all galaxies.")
    bulged_pos = [c for c in bulged_corr2 if c > 0]
    xbulge_pos = [c for c in xbulge_corr2 if c > 0]
    print("Out of these, there are {} galaxies with bulge ({}%) and {} without ({}%)."
          .format(len(bulged_pos), round(len(bulged_pos)/bulged_count*100, 1), len(xbulge_pos), round(len(xbulge_pos)/xbulge_count*100, 1)))
