# (C) 2024 Enoch Ko.
"""
Check and extract features (both peaks and troughs)
from RC residuals using scipy.signal.find_peaks.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy import interpolate
from scipy.signal import find_peaks

from utils_analysis.toy_gen import toy_gen
from utils_analysis.Vobs_fits import Vbar_sq
from utils_analysis.med_filter import med_filter


# Switches for extracting features from different RCs.
testing     = True
use_toy     = False
use_SPARC   = False


def ft_check(rmax, arr, errV):
    min_width = 3*rmax/len(arr)
    avg_errV = np.mean(errV)
    peaks, prop1 = find_peaks( arr, height=avg_errV, width=min_width, rel_height=0.5 )
    troughs, prop2 = find_peaks( -arr, height=avg_errV, width=min_width, rel_height=0.5 )
    prop2["peak_heights"]  = - prop2["peak_heights"]
    prop2["width_heights"] = - prop2["width_heights"]

    # Merge dictionaries of properties from both peaks and troughs
    props = [ prop1, prop2 ]
    properties = {}
    for key in prop1.keys():
        properties[key] = np.concatenate( list(properties[key] for properties in props) )

    return np.concatenate( (peaks, troughs) ), properties


if not (use_toy or use_SPARC):
    rad = np.linspace(0.0, 4.0*np.pi, 50)
    vel = np.sin(rad)
    v_werr = np.random.normal(vel, 0.1)
    peaks, properties = ft_check( max(rad), v_werr, 0.1 )
    print(peaks, properties)

    plt.title("Residuals ft_check test")
    plt.plot(rad, vel, color='k', alpha=0.5)
    plt.scatter(rad, v_werr, color='tab:blue', alpha=0.5)

    for ft in range(len(peaks)):
        lb = properties["left_bases"][ft] + 1
        rb = properties["right_bases"][ft] + 1
        plt.plot(rad[lb:rb], v_werr[lb:rb], color='red', alpha=0.5)
        plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"]*4.0*np.pi/50,
                   xmax=properties["right_ips"]*4.0*np.pi/50, color = "C1")
    
    plt.savefig("/mnt/users/koe/test.png")
    plt.close()
    

if use_toy:
    # Parameters for Gaussian bump (fixed feature) and noise (varying amplitudes).
    bump_size  = 20.0   # Defined in terms of percentage of max(Vbar)
    bump_loc   = 5.0
    bump_FWHM  = 0.5
    bump_sigma = bump_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Define galaxy radius (units ~kpc; excluding the point r=0).
    num_samples = 100
    rad = np.linspace(10., 0., num_samples, endpoint=False)[::-1]
    num_rad = len(rad)

    # Generate toy RCs with residuals (Vraw = w/o ft, Vraw_werr = w/ noise; velocitites = w/ ft, v_werr = w/ noise).
    noise = 0.05
    num_iterations = 1
    bump, Vraw, velocities, Vraw_werr, v_werr, residuals, res_Xft = toy_gen(rad, bump_loc, bump_size, bump_sigma, noise, num_iterations)

    # Vobs (w/ feature) residuals if it's generated perfectly by MOND.
    MOND_res = (velocities[:,1,:] - Vraw[:,1,:])

    peaks, properties = ft_check( max(rad), residuals[0][1], [noise] )
    print( peaks, properties )

    if testing:
        lb = properties["left_bases"][0] + 1
        rb = properties["right_bases"][0] + 1
        # lb = int(properties["left_ips"][0])
        # rb = int(properties["right_ips"][0])

        plt.title("Residuals ft_check test")
        plt.plot(rad, residuals[0][1], alpha=0.5)
        plt.plot(rad[lb:rb], residuals[0][1][lb:rb], color='red', alpha=0.5)
        plt.hlines(y=properties["width_heights"], xmin=(properties["left_ips"]+1)/10,
                xmax=(properties["right_ips"]+1)/10, color = "C1")
        plt.savefig("/mnt/users/koe/test.png")
        plt.close()


if use_SPARC:
    # Get galaxy data from table1.
    file = "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt"

    SPARC_c = [ "Galaxy", "T", "D", "e_D", "f_D", "Inc",
            "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
                "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
    table = pd.read_fwf(file, skiprows=98, names=SPARC_c)

    columns = [ "Rad", "Vobs", "errV", "Vgas",
                "Vdisk", "Vbul", "SBdisk", "SBbul" ]

    galaxy_count = 1 if testing else len(table["Galaxy"])
    pgals = []
    pVbar, pVobs = galaxy_count, galaxy_count

    for i in tqdm(range(galaxy_count)):
        g = "NGC6946" if testing else table["Galaxy"][i]

        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
        r = data["Rad"]
        rad = np.linspace( min(r), max(r), len(r)*20 )

        Vbar2 = Vbar_sq(data, bulged)
        v_components = np.array([ np.sqrt(Vbar2), data["Vobs"] ])
        have_peaks = True

        for res in range(2):
            # v_d0 = interpolate.pchip_interpolate(r, v_components[res], rad)
            # _, residuals = med_filter( rad, v_d0, axes=0 )
            _, residuals = med_filter( r, v_components[res], axes=0 )

            peaks, properties = ft_check( max(r), np.array(residuals), np.array(data["errV"]) )
            # print(f"Residual: {res}, Peaks Found: {peaks}, Number of Peaks: {len(peaks)}")  # Debugging line
    
            if len(peaks) == 0:
                have_peaks = False
                if res == 0: pVbar -= 1
                else: pVobs -= 1

            if testing and res == 1:
                print(peaks, properties)

                plt.title("Residuals ft_check test")
                plt.errorbar(r, residuals, data["errV"], alpha=0.5, ls='none')

                for ft in range(len(peaks)):
                    lb = properties["left_bases"][ft]
                    rb = properties["right_bases"][ft]
                    plt.plot(rad[lb:rb], residuals[lb:rb], color='red', alpha=0.5)
                    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                            xmax=properties["right_ips"], color = "C1")
                
                plt.savefig("/mnt/users/koe/test.png")
                plt.close()
            
        # print(f"Final Have Peaks: {have_peaks}")  # Check the status before printing g

        if have_peaks:
            pgals.append(g)

    print(f"Number of galaxies with features in both Vobs and Vbar: {len(pgals)}")
    print(f"Number of galaxies with features in ONLY Vobs: {pVobs}")
    print(f"Number of galaxies with features in ONLY Vbar: {pVbar}")
    print(pgals)
