# (C) 2024 Enoch Ko.
"""
Functions for sampling Vbar for uncertainties and observational errors,
propagated into the generation of mock data (MOND + LCDM from MCMC fits).
"""
import numpy as np
from scipy import stats
import jax.random as random
from numpyro.infer import MCMC, NUTS, init_to_median

from .params import pbul, pdisk, a0, num_samples
from .Vobs_fits import Vobs_fit, BIC_from_samples


# Sample Vbar squared with uncertainties in M/L ratios, luminosities and distances.
def Vbar_sq_unc(table, i_table, data, bulged=False, num_samples=num_samples):
    # Sample mass-to-light ratios
    dist_pdisk = np.random.normal(pdisk, 0.125, size=num_samples)
    dist_pgas = np.random.normal(1., 0.04, size=num_samples)
    dist_pbul = np.random.normal(pbul, 0.175, size=num_samples)

    # # Sample inclination (remember to convert from degrees to radians!).
    # inc_mean, inc_err = table["Inc"][i_table]*np.pi/180, table["e_Inc"][i_table]*np.pi/180
    # inc_min, inc_max = 0.0, 150 * np.pi / 180
    # inc = stats.truncnorm.rvs( (inc_min-inc_mean) / inc_err, (inc_max-inc_mean) / inc_err, loc=inc_mean, scale=inc_err, size=num_samples )
    # inc_scaling = np.sin(inc_mean) / np.sin(inc)

    # # Scale Vobs and errV according to sampled inclication.
    # Vobs = np.array(data["Vobs"]) * inc_scaling
    # e_Vobs = np.array(data["errV"]) * inc_scaling

    # Sample luminosity
    L36 = stats.truncnorm.rvs(-table["L"][i_table] / table["e_L"][i_table], np.inf, loc=table["L"][i_table], scale=table["e_L"][i_table], size=num_samples)
    dist_pdisk *= L36 / table["L"][i_table]
    dist_pbul *= L36 / table["L"][i_table]

    # Sample distance to the galaxy
    galdist = stats.truncnorm.rvs(-table["D"][i_table] / table["e_D"][i_table], np.inf, loc=table["D"][i_table], scale=table["e_D"][i_table], size=num_samples)
    dist_scale = galdist / table["D"][i_table]
    dist_scaling = np.full((len(data["Vdisk"]), num_samples), dist_scale)

    dist_pdisk = np.array([dist_pdisk] * len(data["Vdisk"]))
    dist_pbul = np.array([dist_pbul] * len(data["Vbul"])) if bulged else 0.
    dist_pgas = np.array([dist_pgas] * len(data["Vgas"]))

    Vdisk = np.array([data["Vdisk"]] * num_samples).T
    Vbul = np.array([data["Vbul"]] * num_samples).T if bulged else 0.
    Vgas = np.array([data["Vgas"]] * num_samples).T

    Vbar_squared = (dist_pdisk * Vdisk**2
                    + dist_pbul * Vbul**2
                    + dist_pgas * Vgas**2)
    Vbar_squared *= dist_scaling

    return Vbar_squared

def MOND_unc(r, Vbar2_unc, num_samples=num_samples):
    r_unc = np.array([r] * num_samples).T
    acc = Vbar2_unc / r_unc
    y = acc / a0
    mu = 1 + np.sqrt((1 + 4/y))
    mu /= 2
    return np.sqrt(acc * mu * r_unc)

# Scatter a Vobs array with Gaussian noise of width data["errV"].
def Vobs_scat(Vobs, errV, num_samples=num_samples):
    errV_copies = np.array([errV] * num_samples).T
    return np.random.normal(Vobs, errV_copies)

# Scatter a Vobs array with CORRELATED Gaussian noise of width data["errV"].
def Vobs_scat_corr(Vobs, errV, num_samples=num_samples):
    gaussian_corr = np.abs(np.random.normal(0., 1., size=num_samples))
    errV_copies = np.array([errV] * num_samples).T
    errV_copies *= gaussian_corr
    return np.random.normal(Vobs, errV_copies)


# Fit LCDM (NFW) or MOND mock data to Vobs array.
def Vobs_MCMC(table, i_table, data, bulged, profile):
    nuts_kernel = NUTS(Vobs_fit, init_strategy=init_to_median(num_samples=num_samples))
    mcmc = MCMC(nuts_kernel, num_warmup=10000, num_samples=20000, progress_bar=False)
    mcmc.run(random.PRNGKey(0), table, i_table, data, bulged, profile=profile)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    log_likelihood = samples.pop("log_likelihood")

    print(f"BIC: {BIC_from_samples(samples, log_likelihood)}")
    return samples
