# (C) 2024 Enoch Ko.
"""
Parameters defined for analysis.
See SPARC paper (Lelli et al 2016) for details:
https://iopscience.iop.org/article/10.3847/0004-6256/152/6/157/pdf
"""

G = 4.300e-6    # Gravitational constant G (kpc (km/s)^2 solarM^(-1))
pdisk, pbul = 0.5, 0.7  # Mean mass-to-light ratios for disk and bulge.
a0 = 1.2e-10 / 3.24e-14     # Scale acceleration for MOND [pc/yr^2].
num_samples = 1000   # Number of Vbar (hence mock Vobs) samples to generate.

# Parameters from Richard's code:
GNEWTON = 4.30091727e-06    # km^2 kpc Msun^(-1) s^-2
RHO200C = 200 * 277.54      # Msun / kpc^3  (h = 1)
LITTLE_H = 0.7
