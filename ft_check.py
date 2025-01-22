# (C) 2024 Enoch Ko.
"""
Check and extract features (both peaks and troughs)
from RC residuals using scipy.signal.find_peaks.
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpyro
import argparse

from utils_analysis.toy_gen import toy_gen
from utils_analysis.Vobs_fits import Vbar_sq
# from utils_analysis.med_filter import med_filter
from utils_analysis.little_things import get_things_res
from utils_analysis.toy_GP import GP_fit, GP_residuals
from utils_analysis.extract_ft import ft_check



# Switches for extracting features from different RCs.
testing    = False
use_toy    = False
use_1560   = False
use_SPARC  = False
use_THINGS = True


if not (use_toy or use_1560 or use_SPARC or use_THINGS):
    rad = np.linspace(0.0, 4.0*np.pi, 50)
    vel = np.sin(rad)
    v_werr = np.random.normal(vel, 0.1)
    peaks, properties = ft_check( v_werr, 0.1 )
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

    peaks, properties = ft_check( residuals[0][1], [noise] )
    print( peaks, properties )

    if testing:
        plt.title("Residuals ft_check test")
        plt.plot(rad, residuals[0][1], alpha=0.5)

        for ft in range(len(peaks)):
            lb = properties["left_bases"][ft] + 1
            rb = properties["right_bases"][ft] + 1
            plt.plot(rad[lb:rb], residuals[0][1][lb:rb], color='red', alpha=0.5)
            plt.hlines(y=properties["width_heights"], xmin=(properties["left_ips"]+1)/10,
                       xmax=(properties["right_ips"]+1)/10, color = "C1")
            
        plt.savefig("/mnt/users/koe/test.png")
        plt.close()


if use_1560:
    # Get galaxy data from digitized plot.
    file = "/mnt/users/koe/data/NGC1560_Stacy.dat"
    columns = [ "Rad", "Vobs", "Vgas", "Vdisk", "errV" ]
    rawdata = np.loadtxt( file )
    data = pd.DataFrame(rawdata, columns=columns)

    plots_loc = "/mnt/users/koe/plots/NGC1560/plot_digitizer/"

    r = data["Rad"].to_numpy()
    Vobs = data["Vobs"].to_numpy()
    Vbar = np.sqrt(Vbar_sq(data))
    errV = data["errV"].to_numpy()


    # Initialize args for GP and sampling rate.
    assert numpyro.__version__.startswith("0.15.0")
    numpyro.enable_x64()    # To keep the inference from getting constant samples.
    parser = argparse.ArgumentParser(description="Gaussian Process")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--thinning", nargs="?", default=2, type=int)
    parser.add_argument("--num-data", nargs="?", default=25, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--init-strategy",
        default="median",
        type=str,
        choices=["median", "feasible", "uniform", "sample"],
    )
    parser.add_argument("--no-cholesky", dest="use_cholesky", action="store_false")
    parser.add_argument("--testing", default=False, type=bool)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)


    # Run GP on Vbar and Vobs
    rad = np.linspace(0., max(r), 100)
    pred_means, pred_bands = GP_fit( args, r, [Vbar, Vobs], rad, summary=True )

    file_name = f"{plots_loc}data_plot.png"
    residuals = GP_residuals( r, [Vbar, Vobs], rad, pred_means, pred_bands, make_plots=True, file_name=file_name )

    # Print out properties of features, if any. (There better be one...)
    peaks, properties = ft_check( np.array(residuals[0]), np.array(errV) )
    if len(peaks) > 0:
        print("Feature(s) found in Vobs of NGC 1560.")
        print(f"Feature properties: {properties}")
    else:
        print("No features found in Vobs of NGC 1560.")

    # Overlay data points on original plot for comparison and fine tuning.
    # img = plt.imread("/mnt/users/koe/plots/NGC1560/plot_digitizer/ref_plot.png")
    fig, ax = plt.subplots()
    # ax.imshow(img)
    plt.errorbar(r, Vobs, errV, capsize=2.5, fmt='o', ls='none', color='k', label="Vobs")
    plt.scatter(r, Vbar, color='red', label="Vbar")

    for ft in range(len(peaks)):
        lb = properties["left_bases"][ft]
        rb = properties["right_bases"][ft]
        plt.plot(r[lb:rb], Vobs[lb:rb], color='blue')

    plt.legend()
    fig.savefig(f"{plots_loc}ref_compare.png", dpi=300, bbox_inches="tight")
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

    pgals, pgals_Vobs, pgals_Vbar = [], [], []
    galaxies = np.load("/mnt/users/koe/gp_fits/galaxy.npy")
    galaxy_count = 1 if testing else len(galaxies)

    noise_arr = np.linspace(0.1, 10.0, 100)
    SPARC_noise_threshold = []

    # for i in range(galaxy_count):
    for i in tqdm(range(galaxy_count), desc="SPARC galaxies"):
        g = "ESO563-G021" if testing else galaxies[i]

        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
        r = data["Rad"] / table["Rdisk"][i]    # Normalize radius by disk scale length.

        Vbar2 = Vbar_sq(data, bulged)
        v_components = np.array([ np.sqrt(Vbar2), data["Vobs"] ])

        # Load in GP results from combined_dtw.py
        gp_fits = np.load("/mnt/users/koe/gp_fits/"+g+".npy")
        rad = gp_fits[0]
        mean_prediction = [ gp_fits[1], gp_fits[3], gp_fits[4], gp_fits[2] ]    # Mean predictions from GP for [ Vbar, MOND, LCDM, Vobs ]
        # lower_percentile = [ gp_fits[5], gp_fits[7], gp_fits[8], gp_fits[6] ]   # 16t percentiles from GP
        # upper_percentile = [ gp_fits[9], gp_fits[11], gp_fits[12], gp_fits[10] ]    # 84th percentiles from GP

        # Compute residuals of fits.
        # res_Vbar_data, res_Vobs = [], []
        res_Vobs = []
        for k in range(len(r)):
            idx = (np.abs(rad - r[k])).argmin()
            # res_Vbar_data.append(v_components[0][k] - mean_prediction[0][idx])
            res_Vobs.append(v_components[1][k] - mean_prediction[3][idx])

        # res_data = np.array([ res_Vbar_data, res_Vobs ])    # dim = (2, len(r))
        # have_peaks = True

        # for res in range(2):
        #     label = "Vbar" if res == 0 else "Vobs"
            # _, residuals = med_filter( r, v_components[res], axes=0 )
            # residuals = res_data[res]

        # print(res_Vobs)
        # print(np.array(data["errV"]))
        # print(np.array(res_Vobs) / np.array(data["errV"]))

        for noise in np.flip(noise_arr):
            _, _, widths = ft_check( np.array(res_Vobs), np.array(data["errV"]), noise )
            if len(widths) > 0:
                if noise == noise_arr[-1]: print(g)
                SPARC_noise_threshold.append(noise)
                break
            
        #     lb, rb, widths = ft_check( np.array(residuals), np.array(data["errV"]) )
        #     # print(f"Residual: {res}, Peaks Found: {peaks}, Number of Peaks: {len(peaks)}")  # Debugging line
    
        #     if len(lb) == 0:
        #         have_peaks = False

        #     if testing and res == 1:
        #         print(lb, rb, widths)

        #         plt.title("Residuals ft_check test")
        #         plt.errorbar(r, residuals, data["errV"], alpha=0.5, color='k')

        #         for ft in range(len(lb)):
        #             plt.plot(r[lb[ft]:rb[ft]], residuals[lb[ft]:rb[ft]], color='red', alpha=0.5)
        #             plt.hlines(y=0., xmin=r[lb[ft]], xmax=r[rb[ft]], color = "C1")
                
        #         plt.savefig("/mnt/users/koe/test.png")
        #         plt.close()

        #     if len(lb) > 0:
        #         if res == 0: pgals_Vbar.append(g)
        #         if res == 1: pgals_Vobs.append(g)

        # if have_peaks:
        #     pgals.append(g)

    # print(f"\nNumber of galaxies with features in Vobs: {len(pgals_Vobs)}")
    # print(pgals_Vobs)
    # print(f"\nNumber of galaxies with features in Vbar: {len(pgals_Vbar)}")
    # print(pgals_Vbar)
    # print(f"\nNumber of galaxies with features in both Vobs and Vbar: {len(pgals)}")
    # print(pgals)


if use_THINGS:
    pgals = []
    galaxies, rad, errV, residuals = get_things_res()

    noise_arr = np.linspace(0.0, 1.9, 20)
    THINGS_noise_threshold = []

    for i in tqdm(range(18), desc="THINGS galaxies"):
        for noise in np.flip(noise_arr):
            _, _, widths = ft_check( np.array(residuals[i]), np.array(errV[i]), noise )
            if len(widths) > 0:
                THINGS_noise_threshold.append(noise)
                # print(f"ft found in {galaxies[i].upper()} with height {noise}*noise")
                break

    # for i in tqdm(range(18)):
    #     lb, rb, widths = ft_check( np.array(residuals[i]), np.array(errV[i]) )
    #     if len(widths) > 0:
    #         pgals.append(galaxies[i].upper())
    #         print(f"Feature(s) found in galaxy {galaxies[i].upper()}:")
    #         print(f"Feature properties: {lb, rb, widths}")
    
    # print(f"Number of Little Things galaxies with features: {len(pgals)}")
    # print(pgals)


# print(f"SPARC noise threshold (Vobs): {SPARC_noise_threshold}")
print(f"THINGS noise threshold: {THINGS_noise_threshold}")

# plt.hist(SPARC_noise_threshold[1], bins=50, alpha=0.5, label="SPARC")
plt.hist(THINGS_noise_threshold, bins=20, alpha=0.5, label="THINGS")
plt.xlabel("Noise threshold")
plt.ylabel("Number of galaxies")

# plt.yscale("log")
plt.legend()
plt.savefig("/mnt/users/koe/plots/ft_check.png")