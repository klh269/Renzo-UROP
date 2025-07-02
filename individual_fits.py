#!/usr/bin/env python
"""
Fit GPs with hand-picked correlation lengthscales on SPARC galaxies.
This overwrites the saved arrays of GP fits from combined_dtw.py.
"""
import pandas as pd
import argparse
from resource import getrusage, RUSAGE_SELF

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import vmap
import jax.random as random

import numpyro

from utils_analysis.gp_utils import model, predict, run_inference
from utils_analysis.Vobs_fits import Vbar_sq, MOND_vsq

matplotlib.use("Agg")  # noqa: E402

# Main code to run.
def main(args, g, X, Y, X_test, ls_Vbar, ls_Vobs):
    """
    Do inference for Vbar with uniform prior for correlation length,
    then apply the resulted lengthscale to Vobs (both real and mock data).
    """
    mean_prediction, percentiles = [], []

    # GP on Vbar with uniform prior on length.
    print(f"Fitting function to Vbar with length = {ls_Vbar} kpc (Rmax = {max(r)} kpc)...")
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, X, Y[0], ls=ls_Vbar)

    # do prediction
    vmap_args = (
        random.split(rng_key_predict, samples["var"].shape[0]),
        samples["var"],
        samples["noise"],
    )
    means, predictions = vmap(
        lambda rng_key, var, noise: predict(
            rng_key, X, Y[0], X_test, var, ls_Vbar, noise, use_cholesky=args.use_cholesky
        )
    )(*vmap_args)

    mean_pred = np.mean(means, axis=0)
    mean_prediction.append(mean_pred)
    gp_predictions[0] = mean_pred

    confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
    percentiles.append(confidence_band)
    gp_16percent[0] = confidence_band[0]
    gp_84percent[0] = confidence_band[1]


    print(f"Fitting function to all Vobs with length = {ls_Vobs} kpc...")
    for j in range(1, 4):
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = run_inference(model, args, rng_key, X, Y[0], ls=ls_Vobs)

        # do prediction
        vmap_args = (
            random.split(rng_key_predict, samples["var"].shape[0]),
            samples["var"],
            samples["noise"],
        )
        means, predictions = vmap(
            lambda rng_key, var, noise: predict(
                rng_key, X, Y[j], X_test, var, ls_Vobs, noise, use_cholesky=args.use_cholesky
            )
        )(*vmap_args)

        mean_pred = np.mean(means, axis=0)
        mean_prediction.append(mean_pred)
        gp_predictions[j] = mean_pred

        confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
        percentiles.append(confidence_band)
        gp_16percent[j] = confidence_band[0]
        gp_84percent[j] = confidence_band[1]

    # Plot results.
    v_labels = [ "Vbar", "Vobs", "MOND", "LCDM" ]
    colors = [ "r", "k", "mediumblue", "tab:green" ]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for j in range(4):
        ax.scatter(X, Y[j], color=colors[j], label=v_labels[j], alpha=0.5)
        ax.plot(X_test, mean_prediction[j], color=colors[j], label=f"Mean prediction ({v_labels[j]})")
        ax.fill_between(X_test, percentiles[j][0], percentiles[j][1], color=colors[j], alpha=0.2)
    ax.set_xlabel("Radii (kpc)")
    ax.set_ylabel("Velocities (km/s)")
    ax.set_title(f"{g} (lbar = {ls_Vbar}, lobs = {ls_Vobs})")
    ax.legend()
    fig.savefig(f"/mnt/users/koe/plots/individual_fits/{g}.png", dpi=300, bbox_inches="tight")


def GP_args():
    """
    Initialize GP args.
    """
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
    parser.add_argument("--testing", default=True, type=bool)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    
    return args

def get_vDM():
    """
    Get v_DM from Fedir's LCDM abundance matching (V_LCDM.txt).
    """
    DMtable_str = []
    v_DM = []
    with open('/mnt/users/koe/V_LCDM.txt') as f_DM:
        list_DM = f_DM.read().splitlines()
        for line in list_DM:
            DMtable_str.append(line.split(", "))

    for line in DMtable_str:
        del line[0]
        v_DM.append([float(num) for num in line])
    
    return v_DM

if __name__ == "__main__":
    args = GP_args()

    columns = [ "Rad", "Vobs", "errV", "Vgas",
                "Vdisk", "Vbul", "SBdisk", "SBbul" ]
    
    correlations_ALL = []
    dtw_cost = [ [], [], [] ]
    norm_cost = [ [], [], [] ]

    galaxy_to_fit = [ "NGC1003", "NGC3521", "UGC02953", "UGC06787" ]
    DM_idx = [ 40, 54, 109, 135 ]
    galaxy_count = len(galaxy_to_fit)
    ls_Vbar = [ 10.0, 12.0, 10.0, 10.0 ]
    ls_Vobs = [ 15.0, 12.0, 10.0, 20.0 ]

    v_DM = get_vDM()

    # for i in tqdm(range(galaxy_count)):
    for i in range(galaxy_count):
        g = galaxy_to_fit[i]

        """
        Plotting galaxy rotation curves directly from data with variables:
        Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
        """
        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
        r = data["Rad"]

        Vbar2 = Vbar_sq(data, bulged)
        v_LCDM = np.sqrt(Vbar2 + np.array(v_DM[DM_idx[i]])**2)
        v_components = np.array([ np.sqrt(Vbar2), data["Vobs"], np.sqrt(MOND_vsq(r, Vbar2)), v_LCDM ])
        rad = np.linspace(min(r), max(r), 100)

        X, X_test = r.to_numpy(), rad
        
        print("")
        print("==================================")
        print(f"Analyzing galaxy {g} ({str(i+1)}/{galaxy_count})")
        print("==================================")

        gp_predictions = [ [], [], [], [] ]
        gp_16percent = [ [], [], [], [] ]
        gp_84percent = [ [], [], [], [] ]

        main(args, g, X, v_components, X_test, ls_Vbar[i], ls_Vobs[i])

        # Save GP fits to CSV for later use (for incorporating uncertainties/errors).
        # One array per galaxy, each containing 13 lists:
        # radii, mean (x4), 16th percentile (x4), 84th percentile (x4).
        gp_fits = np.array([rad, *gp_predictions, *gp_16percent, *gp_84percent])
        np.save("/mnt/users/koe/gp_fits/"+g, gp_fits)
        print("\nGP results successfully saved as /mnt/users/koe/gp_fits/"+g+".npy")

        print("\nMemory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
        jax.clear_caches()    # One-line attempt to solve the JIT memory allocation problem.
