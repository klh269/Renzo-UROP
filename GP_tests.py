#!/usr/bin/env python
"""
Test GPs of different correlation lengthscale on SPARC galaxies and NGC 1560.
We wish to find an optimal minimum that accommodates features of all sizes.
"""
import pandas as pd
import argparse
from resource import getrusage, RUSAGE_SELF

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# import jax
from jax import vmap
import jax.random as random
import numpyro

from tqdm import tqdm
from utils_analysis.Vobs_fits import Vbar_sq
from utils_analysis.gp_utils import model, run_inference, predict

matplotlib.use("Agg")  # noqa: E402


if __name__ == "__main__":
    # Initialize GP parameters.
    print("Initializing GP parameters.")
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
    parser.add_argument("--testing", default=False, type=bool)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)


    # Test values for correlation lengthscale.
    # correlation_lengths = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
    # correlation_lengths = [ "test" ]

    # Save data and GP predictions for final plots.
    r_data, radii, Vobs_ALL, errV_ALL, Vbar_ALL = [], [], [], [], []

    # Load data.
    set_id = "set01"
    print(f"Testing galaxies in {set_id}.")

    if set_id == "set01": test_galaxies = [ "NGC1003", "UGC02953", "UGC06787", "NGC1560" ]      # Set 01 (default set in fixed_ls)
    elif set_id == "set02": test_galaxies = [ "NGC0289", "NGC2903", "NGC2915", "NGC3521" ]      # Set 02
    elif set_id == "set03": test_galaxies = [ "NGC5985", "NGC6503", "UGC02916", "UGC05721" ]    # Set 03

    for g in test_galaxies:
        if g == "NGC1560":
            file_path = "/mnt/users/koe/data/NGC1560_Stacy.dat"
            columns = [ "Rad", "Vobs", "Vgas", "Vdisk_raw", "Vdisk", "errV" ]
        else:
            file_path = f"/mnt/users/koe/data/{g}_rotmod.dat"
            columns = [ "Rad", "Vobs", "errV", "Vgas",
                        "Vdisk", "Vbul", "SBdisk", "SBbul" ]
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        r = data["Rad"].to_numpy()
        rad = np.linspace(min(r), max(r), 100)
        Vobs = data["Vobs"].to_numpy()
        errV = data["errV"].to_numpy()

        # Calculate Vbar.
        if g == "NGC1560":
            Vbar = np.sqrt(Vbar_sq(data, False).to_numpy())
        else:
            bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
            Vbar = np.sqrt(Vbar_sq(data, bulged).to_numpy())

        # Save data.
        r_data.append(r)
        radii.append(rad)
        Vobs_ALL.append(Vobs)
        errV_ALL.append(errV)
        Vbar_ALL.append(Vbar)

    # for cl0 in correlation_lengths:
    mean_pred, confidence_band = [ [], [] ], [ [], [] ] # [ Vbar, Vobs]
    # print(f"Testing galaxies with correlation lengthscale = {cl} x Rmax.")

    for i in tqdm(range(4)):
        # ls = cl * max(r_data[i])
        # Run GP fit on Vbar.
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        # samples = run_inference(model, args, rng_key, r_data[i], Vbar_ALL[i], min_length=cl)

        # # Fix lengthscale to 0.5 * max(r) for UGC 2953.
        # if test_galaxies[i] == "UGC02953":
        #     ls = 29.60
        #     samples = run_inference(model, args, rng_key, r_data[i], Vobs_ALL[i], ls=ls)

        #     vmap_args = (
        #         random.split(rng_key_predict, samples["var"].shape[0]),
        #         samples["var"],
        #         samples["noise"],
        #     )
        #     means, predictions = vmap(
        #         lambda rng_key, var, noise: predict(
        #             rng_key, r_data[i], Vobs_ALL[i], radii[i], var, ls, noise, use_cholesky=args.use_cholesky
        #         )
        #     )(*vmap_args)

        # else:
        ls_dict = np.load("/mnt/users/koe/gp_fits/ls_dict.npy", allow_pickle=True).item()
        ls = ls_dict[test_galaxies[i]]
        to_fit = [ Vobs_ALL[i], Vbar_ALL[i] ]
        for j in range(2):
            samples = run_inference(model, args, rng_key, r_data[i], to_fit[j], ls=ls)

            vmap_args = (
                random.split(rng_key_predict, samples["var"].shape[0]),
                samples["var"],
                samples["noise"],
            )
            means, predictions = vmap(
                lambda rng_key, var, noise: predict(
                    rng_key, r_data[i], to_fit[j], radii[i], var, ls, noise, use_cholesky=args.use_cholesky
                )
            )(*vmap_args)

            # Save results for Vbar.
            mean_pred[j].append(np.mean(means, axis=0))
            confidence_band[j].append(np.percentile(predictions, [16.0, 84.0], axis=0))

        # # Extract MAP values from Vobs fit.
        # vr = stats.mode(np.round(samples["var"], 0))[0]
        # print(f"var MAP = {vr}")
        
        # if test_galaxies[i] == "UGC02953": ls = 29.60
        # else: ls = stats.mode(np.round(samples["length"], 2))[0]
        # print(f"length MAP = {ls}")

        # ns = stats.mode(np.round(samples["noise"], 1))[0]
        # print(f"noise MAP = {ns}")

        # # Run GP fit on Vbar using hyperparameters from Vobs.
        # samples = run_inference(model, args, rng_key, r_data[i], Vbar_ALL[i], ls=ls)

        # vmap_args = (
        #     random.split(rng_key_predict, samples["var"].shape[0]),
        #     # samples["var"],
        #     # samples["noise"],
        # )
        # means, predictions = vmap(
        #     lambda rng_key: predict(
        #         rng_key, r_data[i], Vbar_ALL[i], radii[i], vr, ls, ns, use_cholesky=args.use_cholesky
        #     )
        # )(*vmap_args)

        # # Save results for Vobs.
        # mean_pred[1].append(np.mean(means, axis=0))
        # confidence_band[1].append(np.percentile(predictions, [16.0, 84.0], axis=0))

    # Plot results.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, ax in enumerate(axes.flatten()):
        ax.errorbar(r_data[i], Vobs_ALL[i], yerr=errV_ALL[i], fmt="o", label="Data")
        ax.scatter(r_data[i], Vbar_ALL[i], c="k", label="Vbar")
        ax.plot(radii[i], mean_pred[1][i], "r", label="Mean prediction (Vobs)")
        ax.plot(radii[i], mean_pred[0][i], "b", label="Mean prediction (Vbar)")
        ax.fill_between(radii[i], confidence_band[1][i][0], confidence_band[1][i][1], alpha=0.3, label=r"$1\sigma$ confidence (Vobs)")
        ax.fill_between(radii[i], confidence_band[0][i][0], confidence_band[0][i][1], alpha=0.3, label=r"$1\sigma$ confidence (Vbar)")

        ax.set_title(test_galaxies[i])
        ax.set_xlabel("Radii (kpc)")
        ax.set_ylabel("Velocities (km/s)")
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.savefig(f"/mnt/users/koe/plots/gp_tests/fixed_ls/{set_id}.png")
    plt.close()

    print("\nMax memory usage: %s (kb)" %getrusage(RUSAGE_SELF).ru_maxrss)
