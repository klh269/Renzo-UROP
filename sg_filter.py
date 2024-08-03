#!/usr/bin/env python
"""
Interpolates data with piecewise cubic Hermite spline (PCHIP),
filters noise (of length scale < 0.2*Reff) with SG filter, and
correlates the splines, 1st and 2nd derivatives of Vobs vs Vbar.
"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate, stats

from tqdm import tqdm

# Trigger for code testing; "testing = True" runs the main for-loop with only the first galaxy.
testing = False
test_galaxy = "DDO064"
makeplots = True

directory = "/mnt/users/koe/plots/pchip_sg/"
file = "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt"

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
correlations_sd0 , correlations_sd1, correlations_sd2 = [], [], []
correlations_pd0, correlations_pd1, correlations_pd2 = [], [], []

"""
Main for-loop, same interpolation + filter for all galaxies.
"""
for i in tqdm(range(galaxy_count)):
    g = table["Galaxy"][i]
    if testing:
        g = test_galaxy

    if g=="D512-2" or g=="D564-8" or g=="D631-7" or g=="NGC4138" or g=="NGC5907" or g=="UGC06818":
        skips += 1
        continue

    file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)
    bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
    r = data["Rad"] / table["Reff"][i] # Normalised radius (Reff = scale length of stellar disk).
    
    # Reject galaxies with less than 10 data points (correlations are not meaningful).
    if len(r) < 10:
        skips += 1
        continue
    
    # Normalise velocities by Vmax = max(Vobs).
    Vmax = max(data["Vobs"])
    v_components = [ data["Vobs"], Vbar(data), data["Vgas"], data["Vdisk"]*np.sqrt(pdisk) ]
    if bulged:
        v_components.append( data["Vbul"]*np.sqrt(pbul) )
    v_components = np.array(v_components)
    v_components /= Vmax
    nerrV = data["errV"] / Vmax
    
    if bulged:
        bulged_count += 1
    else:
        xbulge_count += 1

    # Interpolate the data with cubic Hermite spline splines.
    rad_count = math.ceil((max(r)-min(r))*100)
    rad = np.linspace(min(r), max(r), rad_count)
    v_d0, v_d1, v_d2 = [], [], []
    for v_comp in v_components:
        v_d0.append(interpolate.pchip_interpolate(r, v_comp, rad))
        v_d1.append(interpolate.pchip_interpolate(r, v_comp, rad, der=1))
        v_d2.append(interpolate.pchip_interpolate(r, v_comp, rad, der=2))

    # Apply SG filter to interpolated data with window size = 0.2*Reff and extract first and second derivatives.
    d0_sg, d1_sg, d2_sg = [], [], []
    for v_comp0 in v_d0:
        d0_sg.append(signal.savgol_filter(v_comp0, 50, 2))
    for v_comp1 in v_d1:
        d1_sg.append(signal.savgol_filter(v_comp1, 50, 2))
    for v_comp2 in v_d2:
        d2_sg.append(signal.savgol_filter(v_comp2, 50, 2))

    # Correlate Vobs and Vbar (d0, d1, d2) as a function of (maximum) radius, i.e. spheres of increasing r.
    # rs = Spearman rho (correlates ranks), rp = Pearson rho.
    rs_d0, rs_d1, rs_d2 = [], [], []
    rp_d0, rp_d1, rp_d2 = [], [], []
    for j in range(10, len(rad)):
        # Spearman rho.
        rs_d0.append(stats.spearmanr(d0_sg[0][:j], d0_sg[1][:j])[0])
        rs_d1.append(stats.spearmanr(d1_sg[0][:j], d1_sg[1][:j])[0])
        rs_d2.append(stats.spearmanr(d2_sg[0][:j], d2_sg[1][:j])[0])
        # Pearson rho.
        rp_d0.append(stats.pearsonr(d0_sg[0][:j], d0_sg[1][:j])[0])
        rp_d1.append(stats.pearsonr(d1_sg[0][:j], d1_sg[1][:j])[0])
        rp_d2.append(stats.pearsonr(d2_sg[0][:j], d2_sg[1][:j])[0])
    
    # Computes ratio arr1/arr2, averaged across length of each segmented array (increasing R).
    def mean_ratio(arr1, arr2):
        ratio = []
        for j in range(len(arr1)):
            ratio.append(sum(arr1[:j]/arr2[:j]) / (j+1))
        return ratio

    # Compute baryonic dominance, i.e. average Vbar/Vobs from centre to some max radius.
    bar_ratio = mean_ratio(v_d0[1], v_d0[0])

    # Compute correlation between rs or rp and the baryonic ratio, using rs for rs-bar and rp for rp-bar.
    scorr_d0 = stats.spearmanr(rs_d0, bar_ratio[10:])[0]
    scorr_d1 = stats.spearmanr(rs_d1, bar_ratio[10:])[0]
    scorr_d2 = stats.spearmanr(rs_d2, bar_ratio[10:])[0]
    pcorr_d0 = stats.pearsonr(rp_d0, bar_ratio[10:])[0]
    pcorr_d1 = stats.pearsonr(rp_d1, bar_ratio[10:])[0]
    pcorr_d2 = stats.pearsonr(rp_d2, bar_ratio[10:])[0]

    correlations_sd0.append(scorr_d0)
    correlations_sd1.append(scorr_d1)
    correlations_sd2.append(scorr_d2)
    correlations_pd0.append(pcorr_d0)
    correlations_pd1.append(pcorr_d1)
    correlations_pd2.append(pcorr_d2)

    if makeplots:
        """
        Plot (all) SG-filtered spline fits.
        """
        fig0 = plt.figure(1)
        frame1 = fig0.add_axes((.1,.3,.8,.6))
        plt.title("PCHIP + SG filter: "+g)
        plt.ylabel('Normalised velocities')
        
        labels = [ "Vobs", "Vbar", "Vgas", "Vdisk", "Vbul" ]
        colors = [ "k", "red", "green", "blue", "darkorange" ]

        for j in range(len(v_components)):
            plt.scatter(r, v_components[j], color=colors[j], alpha=0.3)
            plt.plot(rad, d0_sg[j], color=colors[j], label=labels[j])

        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.grid()
        
        # Compute and plot residuals.
        frame2 = fig0.add_axes((.1,.1,.8,.2))
        plt.xlabel(r'Normalised radius ($\times R_{eff}$)')
        plt.ylabel("Residuals")

        for j in range(2):
            res_comp = []
            for k in range(len(r)):
                idx = (np.abs(rad - r[k])).argmin()
                res_comp.append(v_components[j][k] - d0_sg[j][idx])
            plt.scatter(r, res_comp, color=colors[j], alpha=0.3, label=labels[j])

        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.grid()

        fig0.savefig(directory+g+".png", dpi=300, bbox_inches="tight")
        plt.close()

        """
        Plot the spline fits again, but now only for Vobs and Vbar (top plot),
        and their Spearman correlation alongside the galaxy's (average) baryonic ratio (bottom plot).
        """
        fig0, (ax1, ax3) = plt.subplots(2, 1, sharex=True)
        ax1.set_title("PCHIP + SG filter: "+g)
        ax1.set_ylabel("Normalised velocities")

        ax1.scatter(r, v_components[0], color=colors[0], alpha=0.3)
        ln1 = ax1.plot(rad, d0_sg[0], color=colors[0], label=labels[0])
        ax1.scatter(r, v_components[1], color=colors[1], alpha=0.3)
        ln2 = ax1.plot(rad, d0_sg[1], color=colors[1], label=labels[1])
        
        lns = ln1 + ln2
        labels = [l.get_label() for l in lns]
        ax1.legend(lns, labels, bbox_to_anchor=(1.35,1))

        color = "tab:blue"
        ax3.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
        ax3.set_ylabel("Correlation")
        ln3 = ax3.plot(rad[10:], rs_d0, color=color, label=r"Spearman $\rho$")
        ln4 = ax3.plot(rad[10:], rp_d0, ':', color=color, label=r"Pearson $\rho$")
        ax3.tick_params(axis='y', labelcolor=color)

        ax4 = ax3.twinx()
        color = "orange"
        ax4.set_ylabel(r'Average $v_{bar}/v_{obs}$')
        ln5 = ax4.plot(rad[10:], bar_ratio[10:], '--', color=color, label="Vbar/Vobs")
        ax4.tick_params(axis='y', labelcolor=color)

        ln6 = ax4.plot([], [], ' ', label=r"$\rho_s=$"+str(round(scorr_d0, 3))+r", $\rho_p=$"+str(round(pcorr_d0, 3)))
        lns = ln3 + ln4 + ln5 + ln6
        labels = [l.get_label() for l in lns]
        ax3.legend(lns, labels, bbox_to_anchor=(1.57,1))

        plt.subplots_adjust(hspace=0.05)
        fig0.savefig(directory+"d0/"+g+".png", dpi=300, bbox_inches="tight")
        plt.close()

        """
        Plot first derivatives of the splines for Vobs and Vbar (top plot),
        and their Spearman correlation alongside the galaxy's (average) baryonic ratio (bottom plot).
        """
        fig1, (ax1, ax3) = plt.subplots(2, 1, sharex=True)
        ax1.set_title("First derivative: "+g)
        
        color = "tab:red"
        ax1.set_ylabel(r'$dv_{bar}/dr$', color=color)

        ln1 = ax1.plot(rad, d1_sg[1], color=color, label="Vbar")
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = "black"
        ax2.set_ylabel(r'$dv_{obs}/dr$', color=color)
        ln2 = ax2.plot(rad, d1_sg[0], color=color, label="Vobs")
        ax2.tick_params(axis='y', labelcolor=color)
        
        lns = ln1 + ln2
        labels = [l.get_label() for l in lns]
        ax1.legend(lns, labels, bbox_to_anchor=(1.35,1))

        color = "tab:blue"
        ax3.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
        ax3.set_ylabel("Correlation")
        ln3 = ax3.plot(rad[10:], rs_d1, color=color, label=r"Spearman $\rho$")
        ln4 = ax3.plot(rad[10:], rp_d1, ':', color=color, label=r"Pearson $\rho$")
        ax3.tick_params(axis='y', labelcolor=color)

        ax4 = ax3.twinx()
        color = "orange"
        ax4.set_ylabel(r'Average $v_{bar}/v_{obs}$')
        ln5 = ax4.plot(rad[10:], bar_ratio[10:], '--', color=color, label="Vbar/Vobs")
        ax4.tick_params(axis='y', labelcolor=color)

        ln6 = ax4.plot([], [], ' ', label=r"$\rho_s=$"+str(round(scorr_d1, 3))+r", $\rho_p=$"+str(round(pcorr_d1, 3)))
        lns = ln3 + ln4 + ln5 + ln6
        labels = [l.get_label() for l in lns]
        ax3.legend(lns, labels, bbox_to_anchor=(1.57,1))

        plt.subplots_adjust(hspace=0.05)
        fig1.savefig(directory+"d1/"+g+".png", dpi=300, bbox_inches="tight")
        plt.close()

        """
        Plot second derivatives of splines for Vobs and Vbar (top plot),
        and their Spearman correlation alongside the galaxy's (average) baryonic ratio (bottom plot).
        """
        fig2, (ax1, ax3) = plt.subplots(2, 1, sharex=True)
        ax1.set_title("Second derivative: "+g)
        
        color = "tab:red"
        ax1.set_ylabel(r'$d^2v_{bar}/dr^2$', color=color)

        ln1 = ax1.plot(rad, d2_sg[1], color=color, label="Vbar")
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = "black"
        ax2.set_ylabel(r'$d^2v_{obs}/dr^2$', color=color)
        ln2 = ax2.plot(rad, d2_sg[0], color=color, label="Vobs")
        ax2.tick_params(axis='y', labelcolor=color)
        
        lns = ln1 + ln2
        labels = [l.get_label() for l in lns]
        ax1.legend(lns, labels, bbox_to_anchor=(1.35,1))

        color = "tab:blue"
        ax3.set_xlabel(r'Normalised radius ($\times R_{eff}$)')
        ax3.set_ylabel("Correlation")
        ln3 = ax3.plot(rad[10:], rs_d2, color=color, label=r"Spearman $\rho$")
        ln4 = ax3.plot(rad[10:], rp_d2, ':', color=color, label=r"Pearson $\rho$")
        ax3.tick_params(axis='y', labelcolor=color)

        ax4 = ax3.twinx()
        color = "orange"
        ax4.set_ylabel(r'Average $v_{bar}/v_{obs}$')
        ln5 = ax4.plot(rad[10:], bar_ratio[10:], '--', color=color, label="Vbar/Vobs")
        ax4.tick_params(axis='y', labelcolor=color)

        ln6 = ax4.plot([], [], ' ', label=r"$\rho_s=$"+str(round(scorr_d2, 3))+r", $\rho_p=$"+str(round(pcorr_d2, 3)))
        lns = ln3 + ln4 + ln5 + ln6
        labels = [l.get_label() for l in lns]
        ax3.legend(lns, labels, bbox_to_anchor=(1.57,1))

        plt.subplots_adjust(hspace=0.05)
        fig2.savefig(directory+"d2/"+g+".png", dpi=300, bbox_inches="tight")
        plt.close()

print("Mean correlations:")
print("rs_d0 = ", np.nanmean(correlations_sd0))
print("rs_d1 = ", np.nanmean(correlations_sd1))
print("rs_d2 = ", np.nanmean(correlations_sd2))
print("rp_d0 = ", np.nanmean(correlations_pd0))
print("rp_d1 = ", np.nanmean(correlations_pd1))
print("rp_d2 = ", np.nanmean(correlations_pd2))
