# (C) 2024 Enoch Ko.
"""
Generate toy model using arctan RC with gaussian bump.
"""
import numpy as np
from scipy import stats

import numpyro
import argparse
import jax.random as random
from numpyro.infer import MCMC, NUTS, init_to_median

from utils_analysis.Vobs_fits import MOND_vsq, NFW_fit


def GP_args():
    """
    Initialize args, inc.
        - ft_width: feature width (in kpc) * 10;
        - samp_idx: sampling rate selection (0-29 for 1-30 samples per kpc); 
        - initialize args for GP (if used).
    """
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
    parser.add_argument("--ft-width", default=10, type=float)
    parser.add_argument("--samp-idx", type=int)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    
    return args

def toy_gen(rad, bump_loc, bump_size, bump_sigma, noise):
    """
    Returns the following constructed arrays:
        - bump: Pure Gaussian bump (w/o RC)
        - Vbar: Baryonic velocities (arctan curves + bump)
        - vel_MOND: MOND RC w/o uncertainties
        - vel_LCDM: LCDM velocities (from MCMC fit) w/o uncertainties
    """
    # Generate Vbar with Gaussian bump.
    Vbar_raw = np.arctan(rad)
    Vbar_raw *= 35.0 * 2 / np.pi   # Multiplication factor for getting sensible MOND RC.

    bump = bump_size * stats.norm.pdf(rad, bump_loc, bump_sigma)
    Vbar = Vbar_raw + bump
    # bump /= 35.0 * 2 / np.pi
    
    # Generate Vmond from Vbar using MOND function.
    vel_MOND = np.sqrt(MOND_vsq(rad, Vbar**2))
    Vmax = max(vel_MOND)

    # Fit (once only) for Vcdm.
    errV = np.array( [noise] * len(rad) )
    nuts_kernel = NUTS(NFW_fit, init_strategy=init_to_median(num_samples=100))
    mcmc = MCMC(nuts_kernel, num_warmup=10000, num_samples=20000, progress_bar=False)
    mcmc.run(random.PRNGKey(0), data={"Vobs": vel_MOND, "errV": errV, "Vbar": Vbar, "Rad": rad})
    samples = mcmc.get_samples()
    vel_LCDM = samples["Vpred"][np.argmax(samples["log_likelihood"])]

    return Vmax, bump, Vbar, vel_MOND, vel_LCDM


def toy_scatter(num_iterations:int, noise:float, Vmax:float, Vbar, vel_MOND, vel_LCDM):
    """
    Returns the following constructed arrays of dimensions (num_iterations x 2 (Vbar, Vobs) x len(rad)):
        - Vbar_werr: Vbar with Gaussian uncertainties
        - Vmond: MOND velocities with Gaussian uncertainties
        - Vcdm: LCDM velocities with Gaussian uncertainties
    """
    Vbar_copies = np.array( [Vbar] * num_iterations )
    Vbar_werr = np.random.normal(Vbar_copies, noise) / Vmax
    Vmond_copies = np.array( [vel_MOND] * num_iterations )
    Vmond = np.random.normal(Vmond_copies, noise) / Vmax
    Vcdm_copies = np.array( [vel_LCDM] * num_iterations )
    Vcdm = np.random.normal(Vcdm_copies, noise) / Vmax

    return Vbar_copies / Vmax, Vbar_werr, Vmond, Vcdm


def toy_gen_Xft(rad, bump_loc, bump_size, bump_sigma, noise, num_iterations):
    """
    Returns the following constructed arrays of dimensions (num_iterations x 2 (Vbar, Vobs) x len(rad)):
        - bump: Pure Gaussian bump (w/o RC)
        - Vraw: RC (arctan curves) w/o feature
        - velocities: RC WITH added feature (i.e. Vraw + bump)
        - Vraw_werr: Vraw with Gaussian uncertainties
        - v_werr: velocities with Gaussian uncertainties
        - residuals: residuals of v_werr from perfect arctan fits
        - res_Xft: residuals of Vraw_werr from perfect arctan fits
    """
    # Generate Vbar with Gaussian bump.
    Vbar_raw = np.arctan(rad)
    Vbar_raw *= 100.0   # Multiplication factor for getting sensible MOND RC.

    bump = bump_size * bump_sigma * np.sqrt(2*np.pi) * stats.norm.pdf(rad, bump_loc, bump_sigma)
    Vbar = Vbar_raw + bump
    bump /= 100.0
    
    # Generate Vobs from Vbar using MOND function.
    vel_MOND = np.sqrt(MOND_vsq(rad, Vbar**2))
    Vmax = max(vel_MOND)
    velocities = np.array([ Vbar, vel_MOND ])
    velocities = np.array( [velocities] * num_iterations )

    # Scatter RCs with Gaussian noise.
    v_werr = np.random.normal(velocities, noise) / Vmax
    
    # Generate perfect GP fit using original functions (arctan w/o bump) and calculate residuals.
    Vobs_raw = np.sqrt(MOND_vsq(rad, Vbar_raw**2))
    Vraw = np.array([ Vbar_raw, Vobs_raw ])
    Vraw = np.array( [Vraw] * num_iterations )
    Vraw_werr = np.random.normal(Vraw, noise) / Vmax  # Scatter Vraw for testing GP fits.
    Vraw /= Vmax
    residuals = v_werr - Vraw

    # Vobs residuals due to pure noise, i.e. smooth RC without feature.
    Vobs_raw = np.array( [Vobs_raw] * num_iterations )
    res_Xft = np.random.normal(Vobs_raw, noise) - Vobs_raw
    res_Xft /= Vmax

    velocities /= Vmax

    return bump, Vraw, velocities, Vraw_werr, v_werr, residuals, res_Xft
