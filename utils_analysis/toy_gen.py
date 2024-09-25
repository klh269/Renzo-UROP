# (C) 2024 Enoch Ko.
"""
Generate toy model using arctan RC with gaussian bump.
"""
import numpy as np
from scipy import stats
from utils_analysis.Vobs_fits import MOND_vsq

def toy_gen(rad, bump_loc, bump_size, bump_sigma, noise, num_iterations):
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
