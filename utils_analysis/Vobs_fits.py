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
    v_sq = 4*np.pi*G*rho0*rc**2*(1 - rc/r * np.arctan(r/rc))
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
        rho0 = sample("rho0", dist.Uniform(0., 1.0e7))
        rc = sample("rc", dist.Uniform(0., 1.0e2))
    
    # Sample mass-to-light ratios.
    smp_pdisk = sample("smp_pdisk", dist.Normal(pdisk, 0.125))
    if bulged:
        smp_pbul = sample("smp_pbul", dist.Normal(pbul, 0.175))        
    else:
        smp_pbul = deterministic("smp_pbul", jnp.array(0.))

    # Sample inclination and scale Vobs.
    inc_min, inc_max = 15 * jnp.pi / 180, 150 * jnp.pi / 180
    inc = sample("Inc", dist.TruncatedNormal(table["Inc"][i_table], table["e_Inc"][i_table], low=inc_min, high=inc_max))
    inc_scaling = jnp.sin(table["Inc"][i_table]) / jnp.sin(inc)
    Vobs = deterministic("Vobs", jnp.array(data["Vobs"]) * inc_scaling)
    e_Vobs = deterministic("e_Vobs", jnp.array(data["errV"]) * inc_scaling)

    # Sample luminosity.
    # L = sample("L", dist.TruncatedNormal(table["L"][i_table], table["e_L"][i_table], low=0.))
    # Ups_disk *= L / table["L"][i_table]
    # Ups_bulge *= L / table["L"][i_table]

    # Sample distance to the galaxy.
    d = sample("dist", dist.TruncatedNormal(table["D"][i_table], table["e_D"][i_table], low=0.))
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
        alpha = sample("alpha", dist.Uniform(0., jnp.inf))
        Vgnfw_squared = gNFWhalo_vsq(r, rho0, rc, alpha)
        Vpred = deterministic("Vpred", jnp.sqrt(Vgnfw_squared + Vbar_squared))
    elif profile == "MOND":
        Vpred = deterministic("Vpred", MOND_Vobs(r, Vbar_squared))
    else:
        raise ValueError(f"Unknown profile: '{profile}'.")

    # ll = jnp.sum(dist.Normal(Vpred, e_Vobs).log_prob(Vobs))
    # deterministic("log_likelihood", ll)
    # factor("ll", ll)
