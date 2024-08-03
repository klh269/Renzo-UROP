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

testing = True
test_galaxy = "DDO064"
fileloc = "/mnt/users/koe/plots/gp_RQker/"
progress_bar = True

# Rational quadratic kernel
def kernel(X, Z, var, length, alpha):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = var * jnp.power((1 + 0.5 * deltaXsq / alpha), -alpha)
    return k


def model(X, Y):
    var = numpyro.sample("var", dist.LogNormal(0.0, 1.0))
    alpha = numpyro.sample("alpha", dist.Uniform(0.0, 10.0))
    length = numpyro.sample("length", dist.Uniform(0., max(X)))

    # compute kernel
    k = kernel(X, X, var, length, alpha)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
        obs=Y,
    )


# helper function for doing hmc inference
def run_inference(model, args, rng_key, X, Y):
    start = time.time()
    # demonstrate how to use different HMC initialization strategies
    if args.init_strategy == "value":
        init_strategy = init_to_value(
            values={"var": 1.0, "alpha": 2.0, "length": 0.5}
        )
    elif args.init_strategy == "median":
        init_strategy = init_to_median(num_samples=10)
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
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for Gaussian process predictions
def predict(rng_key, X, Y, X_test, var, length, alpha, use_cholesky=True):
    # compute kernels between train and test data, etc.
    k_pp = kernel(X_test, X_test, var, length, alpha)
    k_pX = kernel(X_test, X, var, length, alpha)
    k_XX = kernel(X, X, var, length, alpha)

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

    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise

def main(args, g, X, Y, X_test, bulged):
        """
        Do inference for each velocity compenent separately.
        """
        v_list = [ "Vobs", "Vbar", "Vgas", "Vdisk", "Vbul" ]
        mean_prediction = []
        percentiles = []
        for i in range(len(Y)):
            if i==4 and not bulged:
                continue
            
            print("Fitting function to " + v_list[i])
            rng_key, rng_key_predict = random.split(random.PRNGKey(0))
            samples = run_inference(model, args, rng_key, X, np.array(Y[i]))

            # do prediction
            vmap_args = (
                random.split(rng_key_predict, samples["var"].shape[0]),
                samples["var"],
                samples["length"],
                samples["alpha"],
            )
            means, predictions = vmap(
                lambda rng_key, var, length, alpha: predict(
                    rng_key, X, np.array(Y[i]), X_test, var, length, alpha, use_cholesky=args.use_cholesky
                )
            )(*vmap_args)

            mean_prediction.append(np.mean(means, axis=0))
            percentiles.append(np.percentile(predictions, [16.0, 84.0], axis=0))

            # labels = ["length", "var", "alpha"]
            # samples_arr = np.vstack([samples[label] for label in labels]).T
            # fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
            # if i < 2:
            #     fig.savefig(fileloc+"corner_"+v_list[i]+"/"+g+".png", dpi=300, bbox_inches="tight")
            # plt.close(fig)

        """
        Make plots.
        """
        fig0 = plt.figure(1)
        frame1 = fig0.add_axes((.1,.3,.8,.6))
        plt.title("Gaussian process: "+g)
        # plt.xlabel(xlabel="Normalised radius "+r"($\times R_{eff}$)")
        plt.ylabel("Normalised velocities")

        plt.scatter(X, Y[0], color="k", alpha=0.3) # Vobs
        plt.scatter(X, Y[1], color="red", alpha=0.3) # Vbar
        plt.scatter(X, Y[2], color="green", alpha=0.3) # Vgas
        plt.scatter(X, Y[3]*np.sqrt(pdisk), color="blue", alpha=0.3) # Vdisk

        # Plot splines from pchip_interpolate.
        plt.plot(X, Y[0], '--', color="k", label="Vobs_spline", alpha=0.5)
        plt.plot(X, Y[1], '--', color="red", label="Vbar_spline", alpha=0.5)
        
        # plot mean predictions.
        plt.plot(X_test, mean_prediction[0], color="k", label="Vobs")
        plt.plot(X_test, mean_prediction[1], color="red", label="Vbar")
        plt.plot(X_test, mean_prediction[2], color="green", label="Vgas")
        plt.plot(X_test, mean_prediction[3]*np.sqrt(pdisk), color="blue", label="Vdisk")

        # plot 68% (1 sigma) confidence level of predictions for Vobs and Vbar.
        plt.fill_between(X_test, percentiles[0][0, :], percentiles[0][1, :], color="k", alpha=0.2)
        plt.fill_between(X_test, percentiles[1][0, :], percentiles[1][1, :], color="red", alpha=0.2)
        plt.fill_between(X_test, percentiles[2][0, :], percentiles[2][1, :], color="green", alpha=0.2)
        plt.fill_between(X_test, percentiles[3][0, :]*np.sqrt(pdisk), percentiles[3][1, :]*np.sqrt(pdisk), color="blue", alpha=0.2)

        # Same thing for galaxies w/ bulge.
        if bulged:
            plt.scatter(X, Y[4]*np.sqrt(pbul), color="darkorange", alpha=0.3)
            plt.plot(X_test, mean_prediction[4]*np.sqrt(pbul), color="darkorange", label="Vbul")
            plt.fill_between(X_test, percentiles[4][0, :]*np.sqrt(pbul), percentiles[4][1, :]*np.sqrt(pbul), color="darkorange", alpha=0.2)
        
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.grid()
        
        # Compute residuals of fits. Maybe change this to spline-prediction for smoother residual plots?
        res_Vobs = []
        res_Vbar = []
        for k in range(len(X)):
            idx = (np.abs(X_test - X[k])).argmin()
            res_Vobs.append(Y[0][k] - mean_prediction[0][idx])
            res_Vbar.append(Y[1][k] - mean_prediction[1][idx])

        frame2 = fig0.add_axes((.1,.1,.8,.2))
        plt.xlabel(r'Normalised radius ($\times R_{eff}$)')
        plt.ylabel("Residuals")
        plt.scatter(r, res_Vobs, color='k', alpha=0.3, label="Vobs")
        plt.scatter(r, res_Vbar, color='red', alpha=0.3, label="Vbar")
        plt.plot(r, res_Vobs, color='k', alpha=0.5)
        plt.plot(r, res_Vbar, color='red', alpha=0.5)
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        plt.grid()

        fig0.savefig(fileloc+g+".png", dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.15.0")
    numpyro.enable_x64()
    parser = argparse.ArgumentParser(description="Gaussian Process example") # To keep the inference from getting constant samples.
    parser.add_argument("-n", "--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=2000, type=int)
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

    # Calculate baryonic matter from data of individual galaxies.
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
    fd_corr = []
    spline_corr = []
    bulged_corr = []
    xbulge_corr = []
    d2_corr = []
    bulged_corr2 = []
    xbulge_corr2 = []
    
    # for i in tqdm(range(galaxy_count)):
    for i in range(galaxy_count):
        # Galaxies which fail to produce corner plots:
        # DDO161 (7/175), F563-1 (15), F568-3 (21), F583-1 (28)
        # if i < 6:
        #     continue

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

        # Reject galaxies with less than 10 data points.
        if len(r) < 10:
            continue

        Rmax = max(np.diff(r)) # Maximum difference in r of data points (to be used as length scale for GP kernel)
        # Rmin = min(np.diff(r)) # Minimum difference in r (used for upsampling the data)

        # Normalise velocities by Vmax = max(Vobs).
        Vmax = max(data["Vobs"])
        nVobs = data["Vobs"] / Vmax
        nerrV = data["errV"] / Vmax
        nVbar = Vbar(data) / Vmax
        nVgas = data["Vgas"] / Vmax
        nVdisk = data["Vdisk"] / Vmax
        nVbul = data["Vbul"] / Vmax

        if bulged:
            bulged_count += 1
        else:
            xbulge_count += 1
        
        rad = np.linspace(r[0], r[len(r)-1], num=1000)

        X, X_test = r.to_numpy(), rad
        Y = np.array([ nVobs, nVbar, nVgas, nVdisk, nVbul ])
        # Y = np.array([ nVobs, nVbar ])
              
        print("")
        print("==================================")
        print("Analyzing galaxy "+g+" ("+str(i+1)+"/175)")
        print("==================================")
        # print("Rmax =", Rmax)

        main(args, g, X, Y, X_test, bulged)
