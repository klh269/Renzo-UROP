# (C) 2024 Enoch Ko.
"""
Dark matter halo fits using different models in LCDM.
"""
import numpy as np
from scipy.integrate import quad

from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro import deterministic, sample, factor

from .params import G, pdisk, pbul, a0


# Calculate baryonic matter from data of individual galaxies.
def Vbar_sq(arr, bulged):
    if bulged:
        v_sq = arr["Vgas"]**2 + arr["Vdisk"]**2 * pdisk + arr["Vbul"]**2 * pbul
    else:
        v_sq = arr["Vgas"]**2 + arr["Vdisk"]**2 * pdisk
    return v_sq

# Pseudo-isothermal profile.
def ISOhalo_vsq(r, rho0, rc):
    v_sq = 4*jnp.pi*G*rho0*rc**2*(1 - rc/r * jnp.arctan(r/rc))
    return v_sq

# Navarro-Frenk-White (NFW) profile.
def NFWhalo_vsq(r, rho0, rc):
    x = r / rc
    v_sq = 4*jnp.pi*G*rho0*rc**3*(jnp.log(1+x) - x/(1+x))/r
    return v_sq

# Generalised NFW profile.
def gNFWhalo_vsq(r, rho0, rc, alpha):
    x = r / rc
    integrand = rho0 / (x**alpha * (1+x)**(3-alpha))
    Mr = quad(integrand, 0, x, args=(x))[0]
    v_sq = G * Mr / r
    return v_sq

def MOND_Vobs(r, Vbar2, a0=a0):
    # Quadratic solution from MOND simple interpolating function.
    acc = Vbar2 / r
    y = acc / a0
    nu = 1 + np.sqrt((1 + 4/y))
    nu /= 2
    return np.sqrt(acc * nu * r)


# Main code for sampling Vbar uncertainties and fit to Vobs.
def Vobs_fit(table, i_table, data, bulged, profile="NFW"):
    """
    Code for fitting halo profile to observed Vobs + Vbar.
    A major part of this references model.py from Richard Stiskalek:
    https://github.com/Richard-Sti/RCfit/blob/master/rcfit/model.py
    """
    # Sample halo parameters.
    if profile in ["Iso", "NFW", "gNFW"]:
        Rmax = max(data["Rad"])
        rho0 = sample("rho0", dist.Uniform(0., 1.0e5))
        rc = sample("Rc", dist.Uniform(0., 100*Rmax))
    
    # Sample mass-to-light ratios.
    smp_pdisk = sample("Disk M/L", dist.Normal(pdisk, 0.125))
    if bulged:
        smp_pbul = sample("Bulge M/L", dist.Normal(pbul, 0.175))        
    else:
        smp_pbul = deterministic("Bulge M/L", jnp.array(0.))

    # # Sample inclination and scale Vobs
    # inc_min, inc_max = 15 * jnp.pi / 180, 150 * jnp.pi / 180
    # inc = sample("inc",dist.TruncatedNormal(table["Inc"][i_table], table["e_Inc"][i_table], low=inc_min, high=inc_max))  # noqa
    # inc_scaling = jnp.sin(table["Inc"][i_table]) / jnp.sin(inc)
    # Vobs = deterministic("Vobs", jnp.array(data["Vobs"]) * inc_scaling)
    # e_Vobs = deterministic("e_Vobs", jnp.array(data["errV"]) * inc_scaling)

    # # Sample luminosity.
    # L = sample("L", dist.TruncatedNormal(table["L"][i_table], table["e_L"][i_table], low=0.))
    # Ups_disk *= L / table["L"][i_table]
    # Ups_bulge *= L / table["L"][i_table]

    Vobs = deterministic("Vobs", jnp.array(data["Vobs"]))
    e_Vobs = deterministic("e_Vobs", jnp.array(data["errV"]))

    # Sample distance to the galaxy.
    d = sample("Distance", dist.TruncatedNormal(table["D"][i_table], table["e_D"][i_table], low=0.))
    dist_scaling = d / table["D"][i_table]

    Vbar_squared = jnp.array(data["Vgas"]**2) + jnp.array(data["Vdisk"]**2) * smp_pdisk   # + jnp.array(data["Vbul"]**2) * smp_pbul
    Vbar_squared *= dist_scaling

    # Calculate the predicted velocity.
    r = deterministic("r", jnp.array(data["Rad"]) * dist_scaling)
    
    if profile == "Iso":
        Viso_squared = ISOhalo_vsq(r, rho0, rc)
        Vpred = deterministic("Vpred", jnp.sqrt(Viso_squared + Vbar_squared))
    elif profile == "NFW":
        Vnfw_squared = NFWhalo_vsq(r, rho0, rc)
        Vpred = deterministic("Vpred", jnp.sqrt(jnp.array(Vnfw_squared + Vbar_squared)))
    elif profile == "gNFW":
        alpha = sample("alpha", dist.Uniform(0., 10.))
        Vgnfw_squared = gNFWhalo_vsq(r, rho0, rc, alpha)
        Vpred = deterministic("Vpred", jnp.sqrt(Vgnfw_squared + Vbar_squared))
    elif profile == "MOND":
        Vpred = deterministic("Vpred", MOND_Vobs(r, Vbar_squared))
    else:
        raise ValueError(f"Unknown profile: '{profile}'.")
    
    ll = jnp.sum(dist.Normal(Vpred, e_Vobs).log_prob(Vobs))
    # We want to keep track of the log likelihood for BIC/AIC calculations.
    deterministic("log_likelihood", ll)
    factor("ll", ll)


def BIC_from_samples(samples, log_likelihood,
                     skip_keys=["Vobs", "Vpred", "e_Vobs", "log_likelihood", "r"]):  # noqa
    """
    Get the BIC from HMC samples of a Numpyro model.

    Parameters
    ----------
    samples: dict
        Dictionary of samples from the Numpyro MCMC object.
    log_likelihood: numpy array
        Log likelihood values of the samples.
    skip_keys: list
        List of keys to skip when counting the number of parameters

    Returns
    -------
    BIC, AIC: floats
    """
    ndata = samples["Vobs"].shape[1]
    kmax = np.argmax(log_likelihood)

    # How many parameters?
    nparam = 0
    for key, val in samples.items():
        if key in skip_keys or val.std() == 0:
            continue

        if val.ndim == 1:
            nparam += 1
        elif val.ndim == 2:
            nparam += val.shape[-1]
        else:
            raise ValueError("Invalid dimensionality of samples to count the number of parameters.")  # noqa

    BIC = nparam * np.log(ndata) - 2 * log_likelihood[kmax]

    return float(BIC)
