#!/usr/bin/env python
import jax.experimental
import pandas as pd
import argparse
from resource import getrusage, RUSAGE_SELF

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import jax
from jax import vmap
import jax.random as random

import corner
import numpyro
from utils_analysis.gp_utils import model, predict, run_inference

matplotlib.use("Agg")  # noqa: E402


testing = False      # Only analyze NGC 6946.
test_multiple = False   # Loops over the first handful of galaxies instead of just the fist one (DDO161).
make_plots = True

fileloc = "/mnt/users/koe/plots/length_tests/"


# Main code to run.
def main(args, g, X, Y, X_test): 
    """
    Do inference for Vbar with uniform prior for correlation length,
    then apply the resulted lengthscale to Vobs (both real and mock data).
    """
    corner_dir = [ "Vbar/", "Vobs_data/", "Vobs_MOND/", "Vobs_LCDM/" ]

    for j in range(4):
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = run_inference(model, args, rng_key, X, Y[j])

        # do prediction
        vmap_args = (
            random.split(rng_key_predict, samples["var"].shape[0]),
            samples["var"],
            samples["length"],
            samples["noise"],
        )
        _, _ = vmap(
            lambda rng_key, var, length, noise: predict(
                rng_key, X, Y[j], X_test, var, length, noise, use_cholesky=args.use_cholesky
            )
        )(*vmap_args)

        ls = stats.mode(np.round(samples["length"], 2))[0]
        lengths[j].append(ls)

        if make_plots:
            labels = ["length", "var", "noise"]
            samples_arr = np.vstack([samples[label] for label in labels]).T
            fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
            fig.savefig(fileloc+"corner_plots/"+corner_dir[j]+g+".png", dpi=300, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.15.0")
    numpyro.enable_x64()
    parser = argparse.ArgumentParser(description="Gaussian Process example") # To keep the inference from getting constant samples.
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
        choices=["median", "feasible", "value", "uniform", "sample"],
    )
    parser.add_argument("--no-cholesky", dest="use_cholesky", action="store_false")
    parser.add_argument("--testing", default=testing, type=bool)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)


    # Get galaxy data from table1.
    file = "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt"
    
    # Get v_DM from Fedir's LCDM abundance matching (V_LCDM.txt).
    DMtable_str = []
    v_DM = []
    with open('/mnt/users/koe/V_LCDM.txt') as f_DM:
        list_DM = f_DM.read().splitlines()
        for line in list_DM:
            DMtable_str.append(line.split(", "))

    for line in DMtable_str:
        del line[0]
        v_DM.append([float(num) for num in line])

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
    a0 = 1.2e-10 / 3.24e-14     # Scale acceleration for MOND in pc/yr^2

    # Calculate baryonic matter from data of individual galaxies.
    def Vbar(arr):
        v = np.sqrt( arr["Vgas"]**2
                    + (arr["Vdisk"]**2 * pdisk)
                    + (arr["Vbul"]**2 * pbul) )
        return v
    
    def MOND_Vobs(arr, a0=a0):
        # Quadratic solution from MOND simple interpolating function.
        acc = Vbar(arr)**2 / r
        y = acc / a0
        nu = 1 + np.sqrt((1 + 4/y))
        nu /= 2
        return np.sqrt(acc * nu * r)

    galaxy_count = len(table["Galaxy"])
    if testing:
        if test_multiple:
            galaxy_count = 13   # First 2 galaxies.
        else:
            galaxy_count = 1    # Test on NGC 6946.

    lengths = [ [], [], [], [] ]
    comparisons = [ [], [], [], [] ]    # [ Raw, Rmax, Rdisk, Reff ]

    # for i in tqdm(range(galaxy_count)):
    for i in range(galaxy_count):
        if testing and not test_multiple:
            i = 91  # NGC 6946.

        g = table["Galaxy"][i]

        """
        Plotting galaxy rotation curves directly from data with variables:
        Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
        """
        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
        r = data["Rad"]
        # r = data["Rad"] / table["Rdisk"][i] # Normalised radius (Rdisk = scale length of stellar disk).

        # Reject galaxies with less than 20 data points.
        if len(r) < 20:
            continue

        comparisons[0].append(1.0)
        comparisons[1].append(max(r))
        comparisons[2].append(table["Rdisk"][i])
        comparisons[3].append(table["Reff"][i])

        Rmax = max(np.diff(r)) # Maximum difference in r of data points (to be used as length scale for GP kernel)
        # Rmin = min(np.diff(r)) # Minimum difference in r (used for upsampling the data)

        # Normalise velocities by Vmax = max(Vobs) from SPARC data.
        v_LCDM = np.sqrt(Vbar(data)**2 + np.array(v_DM[i])**2)
        v_components = np.array([ Vbar(data), data["Vobs"], MOND_Vobs(data), v_LCDM ])
        rad = np.linspace(min(r), max(r), 100)

        X, X_test = r.to_numpy(), rad
        
        print("")
        print("==================================")
        print("Analyzing galaxy "+g+" ("+str(i+1)+"/175)")
        print("==================================")

        main(args, g, X, v_components, X_test)
        print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
        jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem.

    if make_plots:
        hist_labels = [ "Vbar", "Vobs_data", "Vobs_MOND", "Vobs_LCDM" ]
        comp_labels = [ ["kpc", "Rmax"], ["Rdisk", "Reff"] ]

        lengths = np.array(lengths)
        comparisons = np.array(comparisons)

        for j in range(4):
            fig, ax = plt.subplots(2, 2)
            for row in range(2):
                for col in range(2):
                    ax[row, col].hist(lengths[j]/comparisons[2*row+col], bins=20)
                    ax[row, col].set_xlabel(f"Lengthscale ({comp_labels[row][col]})")
                    
            # Share y_axis labels.
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.ylabel("Frequencies")
            plt.tight_layout()

            plt.savefig(f"{fileloc}{hist_labels[j]}.png", dpi=300, bbox_inches="tight")
            plt.close()

    print("\nTest complete.")
    print("Max memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
