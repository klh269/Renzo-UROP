#!/usr/bin/env python
"""
NOTE: Use 30GB RAM (-n 30) when queueing or job might get stopped.

Gaussian process + dynamic time warping analysis,
combining all SPARC data with MOND and LCDM mock data.

Procedure:
For each galaxy, fit a smooth curve through Vobs and Vbar with GP,
then align their residuals using dynamic time warping.
Compare the normalised alginment cost to get a handle of correlation.

**Yet to account for errors/uncertainties, crucial for meaningful comparison/interpretation!
"""
import pandas as pd
import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, interpolate
import math

import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as random

import corner
from tqdm import tqdm

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)

matplotlib.use("Agg")  # noqa: E402


testing = False # Runs only one galaxy (test_galaxy)
test_galaxy = "NGC2403"
progress_bar = testing # Progress bar for each MCMC.

fileloc = "/mnt/users/koe/plots/combined_dtw/"


# Squared exponential kernel with diagonal noise term
def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k


def model(X, Y, ls=0):
    # set uninformative log-normal priors on our three kernel hyperparameters
    var = numpyro.sample("var", dist.LogNormal(0.0, 1.0))
    noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
    if ls == 0:
        length = numpyro.sample("length", dist.Uniform(0., max(X)))
        k = kernel(X, X, var, length, noise)
    else:
        k = kernel(X, X, var, ls, noise)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
        obs=Y,
    )


# helper function for doing hmc inference
def run_inference(model, args, rng_key, X, Y, ls=0):
    start = time.time()
    # demonstrate how to use different HMC initialization strategies
    if args.init_strategy == "value":
        init_strategy = init_to_value(
            values={"var": 1.0, "noise": 0.05, "length": 0.5}
        )
        if ls != 0:
            init_strategy = init_to_value(
                values={"var": 1.0, "noise": 0.05, "length": ls}
            )
    elif args.init_strategy == "median":
        init_strategy = init_to_median(num_samples=100)
    elif args.init_strategy == "feasible":
        init_strategy = init_to_feasible()
    elif args.init_strategy == "sample":
        init_strategy = init_to_sample()
    elif args.init_strategy == "uniform":
        init_strategy = init_to_uniform(radius=1)
    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, X, Y, ls)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for Gaussian process predictions
def predict(rng_key, X, Y, X_test, var, length, noise, use_cholesky=True):
    # compute kernels between train and test data, etc.
    k_pp = kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)

    # since K_xx is symmetric positive-definite, we can use the more efficient and
    # stable Cholesky decomposition instead of matrix inversion
    if use_cholesky:
        K_xx_cho = jax.scipy.linalg.cho_factor(k_XX)
        K = k_pp - jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, k_pX.T))
        mean = jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, Y))
    else:
        K_xx_inv = jnp.linalg.inv(k_XX)
        K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))

    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
        rng_key, X_test.shape[:1]
    )

    # Return both the mean function and a sample from the 
    # posterior predictive for the given set of hyperparameters
    return mean, mean + sigma_noise


# Dynamic programming code for DTW, see dtw.py for details.
def dp(dist_mat):
    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)


# Main code to run.
def main(args, g, X, Y, X_test, bulged): 
    """
    Do inference for Vbar with uniform prior for correlation length,
    then apply the resulted lengthscale to Vobs (both real and mock data).
    """
    v_comps = [ "Vbar (SPARC)", "Vobs (SPARC)", "Vobs (MOND)", "Vobs (LCDM)" ]
    corner_dir = [ "Vbar/", "Vobs_data/", "Vobs_MOND/", "Vobs_LCDM/" ]
    mean_prediction = []
    percentiles = []

    # GP on Vbar with uniform prior on length.
    print("Fitting function to " + v_comps[0] + "...")
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, X, Y[0])

    # do prediction
    vmap_args = (
        random.split(rng_key_predict, samples["var"].shape[0]),
        samples["var"],
        samples["length"],
        samples["noise"],
    )
    means, predictions = vmap(
        lambda rng_key, var, length, noise: predict(
            rng_key, X, Y[0], X_test, var, length, noise, use_cholesky=args.use_cholesky
        )
    )(*vmap_args)

    mean_prediction.append(np.mean(means, axis=0))
    percentiles.append(np.percentile(predictions, [16.0, 84.0], axis=0))

    labels = ["length", "var", "noise"]
    samples_arr = np.vstack([samples[label] for label in labels]).T
    fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
    fig.savefig(fileloc+"corner_plots/"+corner_dir[0]+g+".png", dpi=300, bbox_inches="tight")
    plt.close()

    # GP on Vobs with fixed lengthscale from Vbar.
    ls = np.median(samples["length"])
    for j in range(1, 3):
        print("\nFitting function to " + v_comps[j] + " with length = " + str(round(ls, 2)) + "...")
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = run_inference(model, args, rng_key, X, Y[j], ls=ls)

        # do prediction
        vmap_args = (
            random.split(rng_key_predict, samples["var"].shape[0]),
            samples["var"],
            samples["noise"],
        )
        means, predictions = vmap(
            lambda rng_key, var, noise: predict(
                rng_key, X, Y[j], X_test, var, ls, noise, use_cholesky=args.use_cholesky
            )
        )(*vmap_args)

        mean_prediction.append(np.mean(means, axis=0))
        percentiles.append(np.percentile(predictions, [16.0, 84.0], axis=0))

        labels = ["var", "noise"]
        samples_arr = np.vstack([samples[label] for label in labels]).T
        fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
        fig.savefig(fileloc+"corner_plots/"+corner_dir[j]+g+".png", dpi=300, bbox_inches="tight")
        plt.close()


    """
    Make plots.
    """
    colours = [ 'tab:red', 'k', 'mediumblue', 'tab:green' ]
    fig0 = plt.figure(1)
    frame1 = fig0.add_axes((.1,.3,.8,.6))
    plt.title("Gaussian process: "+g)
    plt.ylabel("Normalised velocities")

    for j in range(3):
        # Scatter plot for data/mock data points.
        plt.scatter(X, Y[j], color=colours[j], alpha=0.3)
        # Plot mean prediction from GP.
        plt.plot(X_test, mean_prediction[j], color=colours[j], label=v_comps[j])
        # Fill in 1-sigma (68%) confidence band of GP fit.
        plt.fill_between(X_test, percentiles[j][0, :], percentiles[j][1, :], color=colours[j], alpha=0.2)

    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.grid()
    
    # Compute residuals of fits.
    res_Vbar, res_Vobs, res_MOND, res_LCDM = [], [] ,[], []
    for k in range(len(X)):
        idx = (np.abs(X_test - X[k])).argmin()
        res_Vbar.append(Y[0][k] - mean_prediction[0][idx])
        res_Vobs.append(Y[1][k] - mean_prediction[1][idx])
        res_MOND.append(Y[2][k] - mean_prediction[2][idx])
        # res_LCDM.append(Y[3][k] - mean_prediction[3][idx])
    residuals = np.array([ res_Vbar, res_Vobs, res_MOND ])

    frame2 = fig0.add_axes((.1,.1,.8,.2))
    plt.xlabel(r'Normalised radius ($\times R_{eff}$)')
    plt.ylabel("Residuals")

    for j in range(3):
        plt.scatter(r, residuals[j], color=colours[j], alpha=0.3, label=v_comps[j])
        plt.plot(r, residuals[j], color=colours[j], alpha=0.3)

    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.grid()

    fig0.savefig(fileloc+g+".png", dpi=300, bbox_inches="tight")
    plt.close()


    """
    DTW for residuals.
    """
    # Construct distance matrices.
    dist_data = np.zeros((len(r), len(r)))
    dist_MOND = np.copy(dist_data)
    # dist_LCDM = np.copy(dist_data)
    for n in range(len(r)):
        for m in range(len(r)):
            dist_data[n, m] = abs(res_Vobs[n] - res_Vbar[m])
            dist_MOND[n, m] = abs(res_MOND[n] - res_Vbar[m])
            # dist_LCDM[n, m] = abs(res_LCDM[n] - res_Vbar[m])
    
    dist_mats = np.array([ dist_data, dist_MOND ])
    mats_dir = [ "data/", "MOND/", "LCDM/" ]
    
    # DTW!
    for j in range(2):
        path, cost_mat = dp(dist_mats[j])
        x_path, y_path = zip(*path)
        cost = cost_mat[ len(r)-1, len(r)-1 ]
        dtw_cost.append(cost)

        # Plot distance matrix and cost matrix with optimal path.
        plt.title("Dynamic time warping: "+g)
        plt.figure(figsize=(6, 4))
        plt.subplot(121)
        plt.title("Distance matrix")
        plt.imshow(dist_mats[j], cmap=plt.cm.binary, interpolation="nearest", origin="lower")

        plt.subplot(122)
        plt.title("Cost matrix")
        plt.imshow(cost_mat, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
        plt.plot(x_path, y_path)

        plt.savefig(fileloc+"dtw/matrix_"+mats_dir[j]+g+".png", dpi=300, bbox_inches="tight")
        plt.close()

        # Visualize DTW alignment.
        plt.figure()
        plt.title("DTW alignment: "+g)

        diff = abs(max(res_Vbar) - min(residuals[j+1]))
        for x_i, y_j in path:
            plt.plot([x_i, y_j], [residuals[j+1][x_i] + diff, res_Vbar[y_j] - diff], c="C7", alpha=0.4)
        plt.plot(np.arange(len(r)), np.array(residuals[j+1]) + diff, c=colours[j+1], label=v_comps[j+1])
        plt.plot(np.arange(len(r)), np.array(res_Vbar) - diff, c="red", label="Vbar")
        plt.plot([], [], c='w', label="Alignment cost = {:.4f}".format(cost))
        plt.plot([], [], c='w', label="Normalized cost = {:.4f}".format(cost/(len(r)*2)))

        plt.axis("off")
        plt.legend(bbox_to_anchor=(1,1))
        plt.savefig(fileloc+"dtw/vis_"+mats_dir[j]+g+".png", dpi=300, bbox_inches="tight")
        plt.close()


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
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)


    # Get galaxy data from table1.
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
    a0 = 1.2e-10 / 3.24e-14     # Scale acceleration for MOND in pc/yr^2

    # Calculate baryonic matter from data of individual galaxies.
    def Vbar(arr):
        v = np.sqrt( arr["Vgas"]**2
                    + (arr["Vdisk"]**2 * pdisk)
                    + (arr["Vbul"]**2 * pbul) )
        return v
    
    def MOND_Vobs(arr, a0=a0):
        # Quadratic solution from MOND standard interpolating function.
        acc = Vbar(arr)**2 / r
        v4 = (acc * r)**2 + np.sqrt((acc * r)**4 + 4*(acc * a0 * r**2)**2)
        v4 /= 2
        return v4**0.25

    galaxy_count = len(table["Galaxy"])
    skips = 0
    if testing:
        galaxy_count = 1
    bulged_count = 0
    xbulge_count = 0
    
    galaxy = []
    dtw_cost = []

    # for i in tqdm(range(galaxy_count)):
    for i in range(galaxy_count):
        g = table["Galaxy"][i]

        if testing:
            g = test_galaxy
            
        if g=="D512-2" or g=="D564-8" or g=="D631-7" or g=="NGC4138" or g=="NGC5907" or g=="UGC06818":
            skips += 1
            continue

        """
        Plotting galaxy rotation curves directly from data with variables:
        Vobs (overall observed, corrected for inclination), Vgas, Vdisk, Vbul.
        """
        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
        r = data["Rad"] / table["Rdisk"][i] # Normalised radius (Rdisk = scale length of stellar disk).

        # Reject galaxies with less than 20 data points.
        if len(r) < 20:
            continue

        Rmax = max(np.diff(r)) # Maximum difference in r of data points (to be used as length scale for GP kernel)
        # Rmin = min(np.diff(r)) # Minimum difference in r (used for upsampling the data)

        # Normalise velocities by Vmax = max(Vobs) from SPARC data.
        v_components = np.array([ Vbar(data), data["Vobs"], MOND_Vobs(data) ])
        Vmax = max(v_components[1])
        v_components /= Vmax

        if bulged:
            bulged_count += 1
        else:
            xbulge_count += 1
        
        rad = np.linspace(r[0], r[len(r)-1], num=1000)

        X, X_test = r.to_numpy(), rad
        
        print("")
        print("==================================")
        print("Analyzing galaxy "+g+" ("+str(i+1)+"/175)")
        print("==================================")

        main(args, g, X, v_components, X_test, bulged)

        galaxy.append(g)

    
    mean_cost = np.mean(dtw_cost)
    norm_cost = np.array(dtw_cost) / (len(r) * 2)
    mean_norm = np.mean(norm_cost)

    print("\nMean alignment cost = {:.4f}".format(mean_cost))
    print("Mean normalized alignment cost = {:.4f}".format(mean_norm))

    # # Plot histogram of normalized DTW alignment costs of all galaxies.
    # plt.title("Normalized DTW alignment cost")

    # plt.bar(galaxy, sorted(norm_cost))
    # plt.plot([], [], ' ', label="Mean alginment cost = {:.4f}".format(mean_cost))
    # plt.axhline(y=mean_norm, c='r', linestyle='dashed', label="Normalized mean = {:.4f}".format(mean_norm))
    # plt.legend()
    # plt.xticks([])

    # plt.savefig(fileloc+"dtw/histo1.png", dpi=300, bbox_inches="tight")
    # plt.close()
