# (C) 2024 Enoch Ko.
"""
Check and extract features in RC residuals.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpyro
import argparse
import jax
from jax import vmap
import jax.random as random

from utils_analysis.gp_utils import model, predict, run_inference
from utils_analysis.toy_gen import toy_gen
from utils_analysis.Vobs_fits import Vbar_sq
from utils_analysis.mock_gen import Vbar_sq_unc, Vobs_MCMC
from utils_analysis.little_things import get_things_res
from utils_analysis.extract_ft import ft_check  # ft_check_mahanalobis

plt.rcParams.update({'font.size': 13})


def get_args():
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

    return args

def get_NGC1560_res(use_Sanders:bool=True, ls:float=4.5):
    # Get galaxy data from digitized plot.
    if use_Sanders:
        file = "/mnt/users/koe/data/NGC1560_Stacy.dat"
        columns = [ "Rad", "Vobs", "Vgas", "Vdisk_raw", "Vdisk", "errV" ]
    else:
        file = "/mnt/users/koe/data/NGC1560.dat"
        columns = [ "Rad", "Vobs", "errV", "Sdst", "Vdisk", "Sdgas", "Vgas", "Vgth" ]
    rawdata = np.loadtxt( file )
    data = pd.DataFrame(rawdata, columns=columns)

    r = data["Rad"].to_numpy()
    rad = np.linspace(0., max(r), 100)
    Vobs = data["Vobs"].to_numpy()
    Vbar = np.sqrt(Vbar_sq(data))
    errV = data["errV"].to_numpy()

    print(f"Running GPR on NGC 1560.")
    args = get_args()
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    v_data = np.array([Vbar, Vobs])
    mean_preds = []

    for j in range(2):
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = run_inference(model, args, rng_key, r, v_data[j], ls=ls)

        vmap_args = (
            random.split(rng_key_predict, samples["var"].shape[0]),
            samples["var"],
            samples["noise"],
        )

        # r_Xft = np.delete(r, np.s_[19:24], axis=0)
        # data_Xft = np.delete(v_data[j], np.s_[19:24], axis=0)
        means, _ = vmap(
            lambda rng_key, var, noise: predict(
                rng_key, r, v_data[j], rad, var, ls, noise, use_cholesky=args.use_cholesky
            )
        )(*vmap_args)

        mean_prediction = np.mean(means, axis=0)
        mean_preds.append(mean_prediction)

    mean_preds = np.array(mean_preds)

    residuals = [ [], [] ]
    for j in range(2):
        for k in range(len(r)):
            idx = (np.abs(rad - r[k])).argmin()
            residuals[j].append(v_data[j,k] - mean_preds[j,idx])

    return np.array(residuals), errV, Vbar, Vobs

def get_NGC1560_cov_Vbar(args, use_Sanders:bool=True, ls:float=4.5, num_samples:int=1000, redo_fits:bool=False):
    """
    Estimate errors in Vbar by MC sampling:
    1. Sample 1000 realisations of Vbar using prior uncertainties in galaxy parameters.
    2. For each realisation, fit a GP and compute residuals.
    3. Compute covariance matrix of residuals (assume uncorrelated errors for our algorithm).
    """
    if use_Sanders:
        file = "/mnt/users/koe/data/NGC1560_Stacy.dat"
        columns = [ "Rad", "Vobs", "Vgas", "Vdisk_raw", "Vdisk", "errV" ]
    else:
        file = "/mnt/users/koe/data/NGC1560.dat"
        columns = [ "Rad", "Vobs", "errV", "Sdst", "Vdisk", "Sdgas", "Vgas", "Vgth" ]
    rawdata = np.loadtxt( file )
    data = pd.DataFrame(rawdata, columns=columns)

    table = { "D":[2.99], "e_D":[0.1], "Inc":[82.0], "e_Inc":[1.0] }
    i_table = 0
    bulged = False
    Vbar2_unc = Vbar_sq_unc(table, i_table, data, bulged, num_samples)
    Vbar_samples = np.sqrt(Vbar2_unc).T   # dim = (num_samples, len(r))

    r = data["Rad"].to_numpy()
    rad = np.linspace(0., max(r), 100)

    if redo_fits:
        residuals = []

        for smp in range(num_samples):
            # if smp % max( 1, num_samples/50 ) == 0: 
            print(f"GPR on mock sample {smp+1} of {num_samples}...")
        
            rng_key, rng_key_predict = random.split(random.PRNGKey(0))
            samples = run_inference(model, args, rng_key, r, Vbar_samples[smp], ls=ls)

            # do prediction
            vmap_args = (
                random.split(rng_key_predict, samples["var"].shape[0]),
                samples["var"],
                samples["noise"],
            )
            means, _ = vmap(
                lambda rng_key, var, noise: predict(
                    rng_key, r, Vbar_samples[smp], rad, var, ls, noise, use_cholesky=args.use_cholesky
                )
            )(*vmap_args)

            mean_pred = np.mean(means, axis=0)

            # Compute residuals of fits.
            res_Vbar = []
            for k in range(len(r)):
                idx = (np.abs(rad - r[k])).argmin()
                res_Vbar.append(Vbar_samples[smp,k] - mean_pred[idx])
            residuals.append(res_Vbar)

            jax.clear_caches()  # DO NOT DELETE THIS LINE! Reduces memory usage from > 100 GB to < 1 GB!x

        residuals = np.array(residuals).T    # dim = (len(r), num_samples)
        np.save(f"/mnt/users/koe/mock_residuals/NGC1560_Sanders_Vbar.npy", residuals)

    else:
        residuals = np.load(f"/mnt/users/koe/mock_residuals/NGC1560_Sanders_Vbar.npy")

    # Compute covariance matrix of residuals.
    cov_Vbar = np.cov(residuals, rowvar=True)   # dim = (len(r), len(r))

    return cov_Vbar

def sine_test():
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

def toy_ft(testing:bool=False):
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

def NGC1560_ft_tests(num_samples:int=1000):
    file_path = "/mnt/users/koe/data/NGC1560_Stacy.dat"
    columns = [ "Rad", "Vobs", "Vgas", "Vdisk_raw", "Vdisk", "errV" ]
    rawdata = np.loadtxt(file_path)
    data = pd.DataFrame(rawdata, columns=columns)

    table = { "D":[2.99], "e_D":[0.1], "Inc":[82.0], "e_Inc":[1.0] }
    i_table = 0

    nfw_samples  = Vobs_MCMC(table, i_table, data, bulged=False, profile="NFW")    # Vobs_MCMC() runs MCMC with Vobs_fit() from Vobs_fits.py
    rand_idx = np.random.choice( 20000, num_samples, replace=False )
    Vobs_LCDM = nfw_samples["Vpred scattered"]
    Vbar_LCDM = nfw_samples["Vbar"]
    cov_Vbar = np.cov( Vbar_LCDM, rowvar=False )

    res_data, errV, _, _ = get_NGC1560_res()
    cov_Vobs = np.diag(errV**2)   # Covariance matrix for Vobs.
    cov_Vobs = np.cov( Vobs_LCDM, rowvar=False )

    res_mock = np.load("/mnt/users/koe/mock_residuals/NGC1560_Sanders.npy")     # dim = (4, len(r), 1000)
    err_Vbar = np.sqrt(Vbar_sq(data, bulged=False)) * 0.001   # Assuming additional 0.1% uncorrelated scatter on Vbar.
    res_Vbar_scattered = np.random.normal(res_mock[1], np.array([err_Vbar] * 1000).T)    # dim = (len(r), 1000)
    cov_Vbar = np.cov( res_mock[1], rowvar=True )   # + np.diag(err_Vbar**2)    # Covariance matrix for Vbar.
    cov_Vobs = np.cov( res_mock[3], rowvar=True )

    fig, ax = plt.subplots(2, 1)
    for smp in range(10):
        ax[0].plot( data["Rad"], Vbar_LCDM[smp], label="Vbar", alpha=0.4 )
        ax[1].plot( data["Rad"], res_mock[1,:,smp], label="Vbar residuals", alpha=0.4 )
    fig.savefig("/mnt/users/koe/plots/NGC1560_Vbar_check.png", dpi=300, bbox_inches="tight")

    print(f"\nShape of covariance matrix: {cov_Vobs.shape}")
    print(f"\ncov_Vbar around feature: \n{cov_Vbar[19:24, 19:24]}")
    print(f"\ncov_Vobs around feature: \n{cov_Vobs[19:24, 19:24]}")

    print(f"\nDeterminant of cov_Vbar: {np.linalg.det(cov_Vbar.astype(np.float64)).astype(np.float64)}")
    print(f"\nDeterminant positive? {np.linalg.det(cov_Vbar) > 0}")
    print(f"\nDeterminant negative? {np.linalg.det(cov_Vbar) < 0}")
    print(f"\nPositive-definite check: {np.all(np.linalg.eigvals(cov_Vbar) > 0)}")

    print(f"\nEigenvalues of cov_Vbar: \n{np.linalg.eigvals(cov_Vbar)}")
    print(f"\nEigenvalues of cov_Vobs: \n{np.linalg.eigvals(cov_Vobs)}")

    cov_Vbar_inv = np.linalg.inv(cov_Vbar)
    print(f"\nInverted matrix for Vbar: \n{cov_Vbar_inv[19:24, 19:24]}")

    fig, ax = plt.subplots()
    
    # Print out properties of features, if any. (There better be one...)
    errors = [ cov_Vbar, cov_Vobs ]
    labels = [ "Vbar", "Vobs" ]
    colors = [ "tab:red", "k" ]
    for j in range(2):
        dists, lb, rb = ft_check( res_data[j], errors[j], min_sig=0.0 )
        if len(dists) > 0:
            # print(f"Feature(s) found in {labels[j]} of NGC 1560 (Sanders).")
            # print(f"Feature properties: {dists}, {lb}, {rb}")
            print(f"\nMahalanobis distances ({labels[j]}): \n{dists}")
            mean_dist = np.mean(dists)
            std_dist = np.std(dists)
            normalized_dists = (dists - mean_dist) / std_dist
            print(f"\nNormalized distances ({labels[j]}): \n{normalized_dists}")
            ax.hist(normalized_dists, bins=50, alpha=0.5, color=colors[j], label=labels[j])
        else:
            print(f"No features found in {labels[j]} of NGC 1560 (Sanders)!??")
    
    ax.set_xlabel("Mahalanobis distance")
    ax.set_ylabel("Number of samples")
    ax.legend()
    ax.set_title("Mahalanobis distance of residuals (Sanders NGC 1560)")

    fig.savefig("/mnt/users/koe/plots/NGC1560_ft_check.png", dpi=300, bbox_inches="tight")

def NGC1560_ft(use_Sanders:bool=True, num_samples:int=1000, redo_fits:bool=False):
    residuals, err_Vobs, _, _ = get_NGC1560_res(use_Sanders=use_Sanders)
    # cov_Vbar = get_NGC1560_cov_Vbar(get_args(), use_Sanders=use_Sanders, num_samples=num_samples, redo_fits=redo_fits)
    # err_Vbar = np.sqrt(np.diag(cov_Vbar))

    # lb_Vbar, rb_Vbar, widths_Vbar = ft_check(residuals[0], err_Vbar)
    lb_Vobs, rb_Vobs, widths_Vobs = ft_check(residuals[1], err_Vobs)

    # print(f"\nFeatures in Vbar: {lb_Vbar}, {rb_Vbar}, {widths_Vbar}")
    print(f"\nFeatures in Vobs: {lb_Vobs}, {rb_Vobs}, {widths_Vobs}")

def SPARC_ft(testing:bool=False, use_Vbar:bool=False):
    if use_Vbar:
        SPARC_c = [ "Galaxy", "T", "D", "e_D", "f_D", "Inc",
            "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
            "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
        table = pd.read_fwf( "/mnt/users/koe/SPARC_Lelli2016c.mrt.txt", skiprows=98, names=SPARC_c)

    columns = [ "Rad", "Vobs", "errV", "Vgas",
                "Vdisk", "Vbul", "SBdisk", "SBbul" ]

    galaxies = np.load("/mnt/users/koe/gp_fits/galaxy.npy")
    galaxy_count = 1 if testing else len(galaxies)

    noise_arr = np.linspace(0.1, 10.0, 100)
    SPARC_noise_threshold = []
    SPARC_features = {}
    ft_count = 0

    for i in range(galaxy_count):
    # for i in tqdm(range(galaxy_count), desc="SPARC galaxies"):
        g = "ESO563-G021" if testing else galaxies[i]

        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        bulged = np.any(data["Vbul"]>0) # Check whether galaxy has bulge.
        r = data["Rad"].to_numpy()

        Vbar2 = Vbar_sq(data, bulged)
        v_components = np.array([ np.sqrt(Vbar2), data["Vobs"] ])

        # Load in GP results from combined_dtw.py
        gp_fits = np.load("/mnt/users/koe/gp_fits/"+g+".npy")
        rad = gp_fits[0]
        mean_prediction = [ gp_fits[1], gp_fits[3], gp_fits[4], gp_fits[2] ]    # Mean predictions from GP for [ Vbar, MOND, LCDM, Vobs ]

        # Compute residuals of fits.
        res_Vobs = []
        for k in range(len(r)):
            idx = (np.abs(rad - r[k])).argmin()
            res_Vobs.append(v_components[1][k] - mean_prediction[3][idx])

        if use_Vbar:
            res_Vbar = []
            for k in range(len(r)):
                idx = (np.abs(rad - r[k])).argmin()
                res_Vbar.append(v_components[0][k] - mean_prediction[0][idx])

            Vbar2_unc = Vbar_sq_unc( table, np.where(table["Galaxy"] == g)[0][0], data, bulged, 1000 )
            Vbar_bands = np.sqrt( np.percentile(Vbar2_unc, [16.0, 84.0], axis=1) )
            Vbar_err = (Vbar_bands[1] - Vbar_bands[0]) / 2.0

            # lb, rb, widths = ft_check( np.array(res_Vbar)[5:], Vbar_err[5:], 2.0 )
            # if len(widths) > 0: print(f" - Feature found in Vbar(!) of {g}: lb = {lb+5}; rb = {rb+5}; widths = {widths}")
            lb, rb, widths = ft_check( np.array(res_Vbar), Vbar_err, 2.0 )
            if len(widths) > 0: print(f" - Feature found in Vbar(!) of {g}: lb = {lb}; rb = {rb}; widths = {widths}")
            

        for noise in np.flip(noise_arr):
            # _, _, widths = ft_check( np.array(res_Vobs)[5:], np.array(data["errV"])[5:], noise )
            _, _, widths = ft_check( np.array(res_Vobs), np.array(data["errV"]), noise )
            if len(widths) > 0:
                if noise >= 2.0:
                    # lb, rb, widths = ft_check( np.array(res_Vobs)[5:], np.array(data["errV"])[5:], 2.0 )
                    # print(f"Feature found in Vobs of {g}: lb = {lb+5}; rb = {rb+5}; widths = {widths}")
                    lb, rb, widths = ft_check( np.array(res_Vobs), np.array(data["errV"]), 2.0 )
                    print(f"Feature found in Vobs of {g}: lb = {lb}; rb = {rb}; widths = {widths}")
                    SPARC_features.update({g: [lb, rb, widths]})
                    ft_count += 1
                SPARC_noise_threshold.append(noise)
                break
    
    print(f"\nA total of {ft_count} galaxies with features (in Vobs).")

    np.save("/mnt/users/koe/gp_fits/SPARC_features.npy", SPARC_features)
    print(f"\nSPARC ft properties saved to /mnt/users/koe/gp_fits/SPARC_features.npy")

    return SPARC_noise_threshold

def SPARC_error_model(num_samples:int):
    # Sample from errV and apply the same ft idenfitication procedure;
    # we suspect that the lack of features is due to an overestimation of errors in SPARC.
    columns = [ "Rad", "Vobs", "errV", "Vgas",
                "Vdisk", "Vbul", "SBdisk", "SBbul" ]

    galaxies = np.load("/mnt/users/koe/gp_fits/galaxy.npy")
    galaxy_count = len(galaxies)

    noise_arr = np.linspace(0.1, 10.0, 101, endpoint=True)
    SPARC_err_thresholds = []

    for i in tqdm(range(galaxy_count), desc="SPARC error model"):
        g = galaxies[i]

        file_path = "/mnt/users/koe/data/"+g+"_rotmod.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        errV = data["errV"].to_numpy()

        errV_copies = np.tile(errV, (num_samples, 1))    # dim = (num_samples, len(errV))
        errV_zeros = np.zeros_like(errV_copies)
        residuals = np.random.normal(errV_zeros, errV_copies)

        for smp in range(num_samples):
            for noise in np.flip(noise_arr):
                _, _, widths = ft_check( np.array(residuals[smp]), errV, noise )
                if len(widths) > 0:
                    SPARC_err_thresholds.append(noise)
                    break

    return SPARC_err_thresholds

def THINGS_ft():
    _, _, errV, residuals = get_things_res()

    noise_arr = np.linspace(0.0, 2.5, 25)
    THINGS_noise_thresholds = []

    for i in tqdm(range(18), desc="THINGS galaxies"):
        for noise in np.flip(noise_arr):
            _, _, widths = ft_check( np.array(residuals[i]), np.array(errV[i]), noise )
            if len(widths) > 0:
                THINGS_noise_thresholds.append(noise)
                # print(f"ft found in {galaxies[i].upper()} with height {noise}*noise")
                break

    return THINGS_noise_thresholds

def THINGS_error_model(num_samples:int):
    # Sample from errV and apply the same ft idenfitication procedure;
    # we suspect that the lack of features is due to an overestimation of errors in THINGS.
    _, _, errV, _ = get_things_res()

    noise_arr = np.linspace(0.0, 2.5, 25)
    THINGS_err_thresholds = []

    for i in tqdm(range(18), desc="THINGS error model"):
        errV_copies = np.tile(errV[i], (num_samples, 1))    # dim = (num_samples, len(errV))
        errV_zeros = np.zeros_like(errV_copies)
        residuals = np.random.normal(errV_zeros, errV_copies)

        for smp in range(num_samples):
            for noise in np.flip(noise_arr):
                _, _, widths = ft_check( np.array(residuals[smp]), np.array(errV[i]), noise )
                if len(widths) > 0:
                    THINGS_err_thresholds.append(noise)
                    break

    return THINGS_err_thresholds

def Santos_ft(avg_errV:float=0.02):
    galaxies = np.load("/mnt/users/koe/gp_fits/Santos-Santos/galaxies.npy")
    galaxy_count = len(galaxies)
    columns = [ "Rad", "Vobs", "Vbar" ]

    ft_gal = []
    Santos_features = {}

    for i in range(galaxy_count):
        g = galaxies[i]

        file_path = f"/mnt/users/koe/data/Santos-sims/{g}.dat"
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        r = data["Rad"].to_numpy()
        errV = np.full(len(r), avg_errV * max(data["Vobs"]))

        gp_fits = np.load(f"/mnt/users/koe/gp_fits/Santos-Santos/{g}.npy")
        rad = gp_fits[0]
        mean_prediction = gp_fits[2]    # Mean predictions from GP for Vobs

        # Compute residuals.
        res_Vobs = []
        for k in range(len(r)):
            idx = (np.abs(rad - r[k])).argmin()
            res_Vobs.append(data["Vobs"][k] - mean_prediction[idx])
        res_Vobs = np.array(res_Vobs)

        lb, rb, widths = ft_check( res_Vobs, errV )
        if len(lb) > 0:
            ft_gal.append(g)
            Santos_features.update({g: [lb, rb, widths]})
            print(f"Feature(s) found in Vobs of {g}: lb = {lb}; rb = {rb}; widths = {widths}")

    return ft_gal, Santos_features


if __name__ == "__main__":
    # num_samples = 10
    # NGC1560_ft(use_Sanders=True)
    ft_galaxies, Santos_features = Santos_ft(avg_errV=0.01)
    np.save("/mnt/users/koe/Santos-analysis/ft_galaxies.npy", ft_galaxies)
    np.save("/mnt/users/koe/Santos-analysis/ft_properties.npy", Santos_features)

    # _ = SPARC_ft()
    # SPARC_features = np.load("/mnt/users/koe/gp_fits/SPARC_features.npy", allow_pickle=True).item()
    # print(f"Number of galaxies with features = {len(SPARC_features)}")
    # print(f"SPARC features: {SPARC_features}")

    """Histogram for THINGS."""
    # THINGS_err_thresholds = THINGS_error_model(num_samples)
    # THINGS_noise_thresholds = THINGS_ft()

    # plt.hist(THINGS_err_thresholds, bins=np.arange(0.0, 2.5, 0.1), weights=np.ones(np.shape(THINGS_err_thresholds))/num_samples, alpha=0.4, color="k", label="Expected distribution (MC sampling)")
    # plt.hist(THINGS_noise_thresholds, bins=np.arange(0.0, 2.5, 0.1), alpha=0.5, color="tab:blue", label="Features extracted from data")
    # # plt.hist(SPARC_noise_threshold[1], bins=50, alpha=0.5, label="SPARC")

    # plt.xlabel(r"Noise threshold ($T_{max}$)")
    # plt.xlim(left=0.0)
    # plt.ylabel("Number of galaxies")
    # plt.legend()
    # plt.savefig("/mnt/users/koe/plots/THINGS_ft_check.pdf", dpi=300, bbox_inches="tight")
    # plt.close()

    """Histogram for SPARC."""
    # SPARC_err_thresholds = SPARC_error_model(num_samples)
    # SPARC_noise_thresholds = SPARC_ft( use_Vbar=True )

    # plt.hist(SPARC_err_thresholds, bins=np.arange(0.0, 10.0, 0.1), weights=np.ones(np.shape(SPARC_err_thresholds))/num_samples, alpha=0.4, color="k", label="Expected distribution (MC sampling)")
    # plt.hist(SPARC_noise_thresholds, bins=np.arange(0.0, 10.0, 0.1), alpha=0.5, color="tab:blue", label="Features extracted from data")

    # plt.xlabel("Noise threshold")
    # plt.ylabel("Number of galaxies")
    # plt.legend()
    # plt.savefig("/mnt/users/koe/plots/SPARC_ft_check.png", dpi=300, bbox_inches="tight")
    # plt.close()
