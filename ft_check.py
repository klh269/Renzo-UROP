# (C) 2024 Enoch Ko.
"""
Check and extract features (both peaks and troughs)
from RC residuals using scipy.signal.find_peaks.
"""
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from utils_analysis.toy_gen import toy_gen

def ft_check(rmax, arr, errV):
    avg_error = np.mean( errV )
    min_width = 3*rmax/len(arr)
    peaks, prop1 = find_peaks( arr, height=avg_error, width=min_width, rel_height=0.5 )
    troughs, prop2 = find_peaks( -arr, height=avg_error, width=min_width, rel_height=0.5 )
    prop2["peak_heights"]  = - prop2["peak_heights"]
    prop2["width_heights"] = - prop2["width_heights"]

    # Merge dictionaries of properties from both peaks and troughs
    props = [ prop1, prop2 ]
    properties = {}
    for key in prop1.keys():
        properties[key] = np.concatenate( list(properties[key] for properties in props) )

    return np.concatenate( (peaks, troughs) ), properties


# Parameters for Gaussian bump (fixed feature) and noise (varying amplitudes).
bump_size  = -20.0   # Defined in terms of percentage of max(Vbar)
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

# Vobs (w/ feature) residuals if it's generated perfectly by MOND.
MOND_res = (velocities[:,1,:] - Vraw[:,1,:])


"""
TESTING.
"""
peaks, properties = ft_check( max(rad), residuals[0][1], [noise] )
print( peaks, properties )

lb = properties["left_bases"][0]
rb = properties["right_bases"][0]
# lb = int(properties["left_ips"][0])
# rb = int(properties["right_ips"][0])

plt.title("Residuals ft_check test")
plt.plot(rad, residuals[0][1], alpha=0.5)
plt.plot(rad[lb:rb], residuals[0][1][lb:rb], color='red', alpha=0.5)
plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"]/10,
           xmax=properties["right_ips"]/10, color = "C1")
plt.savefig("/mnt/users/koe/test.png")
plt.close()
