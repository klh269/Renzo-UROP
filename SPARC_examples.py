#!/usr/bin/env python
"""
Generate example plots of Gaussian Process fits to SPARC galaxies,
where fits are obtained from combined_dtw.py.
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils_analysis.Vobs_fits import Vbar_sq
from utils_analysis.mock_gen import Vbar_sq_unc     #, Vobs_MCMC

matplotlib.use("Agg")  # noqa: E402
plt.rcParams.update({'font.size': 13})

# use_fits = False  # Use MCMC fits to get Vbar and Vobs.

if __name__ == "__main__":
    # Save data and GP predictions for final plots.
    r_data, radii, Vobs_ALL, errV_ALL, Vbar_ALL = [], [], [], [], []
    mean_pred, lower_perc, upper_perc = [], [], []

    # Load data.
    test_galaxies = [ "NGC1003", "NGC2403", "UGC02953", "UGC06787" ]

    SPARC_c = [ "Galaxy", "T", "D", "e_D", "f_D", "Inc",
            "e_Inc", "L", "e_L", "Reff", "SBeff", "Rdisk",
            "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "Ref."]
    table = pd.read_fwf("/mnt/users/koe/SPARC_Lelli2016c.mrt.txt", skiprows=98, names=SPARC_c)

    for g in test_galaxies:
        file_path = f"/mnt/users/koe/data/{g}_rotmod.dat"
        columns = [ "Rad", "Vobs", "errV", "Vgas",
                        "Vdisk", "Vbul", "SBdisk", "SBbul" ]
        rawdata = np.loadtxt(file_path)
        data = pd.DataFrame(rawdata, columns=columns)
        i_tab = np.where(table["Galaxy"] == g)[0][0]  # Get index of galaxy in table.

        # Get data.
        r = data["Rad"].to_numpy()
        Vobs = data["Vobs"].to_numpy()
        errV = data["errV"].to_numpy()

        bulged = np.any(data["Vbul"] > 0) # Check whether galaxy has bulge.
        Vbar = np.sqrt(Vbar_sq(data, bulged).to_numpy())

        # if use_fits:
        #     nfw_samples = Vobs_MCMC(table, i_tab, data, bulged, profile="NFW")    # Vobs_MCMC() runs MCMC with Vobs_fit() from Vobs_fits.py
        #     v_LCDM = nfw_samples["Vpred"][np.argmax(nfw_samples["log_likelihood"])]
        #     Vbar_LCDM = nfw_samples["Vbar"][np.argmax(nfw_samples["log_likelihood"])]

        #     # Select 1000 random samples from MCMC fits.
        #     rand_idx = np.random.choice( 20000, 1000, replace=False )
        #     full_LCDM = nfw_samples["Vpred scattered"][rand_idx].T
        #     Vbar_full = nfw_samples["Vbar"][rand_idx].T

        # else:
        Vbar_full = np.sqrt(Vbar_sq_unc(table, i_tab, data, bulged, num_samples=1000))
            
        Vbar_perc = np.percentile(Vbar_full, [16, 50, 84], axis=1)

        # Save data.
        r_data.append(r)
        Vobs_ALL.append(Vobs)
        errV_ALL.append(errV)
        Vbar_ALL.append(Vbar_perc)

        # Load in GP results from combined_dtw.py.
        gp_fits = np.load("/mnt/users/koe/gp_fits/"+g+".npy")
        rad = gp_fits[0]
        radii.append(rad)
        mean_pred.append([ gp_fits[1], gp_fits[2] ])    # Mean predictions from GP for [ Vbar, Vobs ]
        # lower_perc.append([ gp_fits[5], gp_fits[6] ])   # 16t percentiles from GP
        # upper_perc.append([ gp_fits[9], gp_fits[10] ])  # 84th percentiles from GP

    # Plot results.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, ax in enumerate(axes.flatten()):
        ax.errorbar(r_data[i], Vobs_ALL[i], yerr=errV_ALL[i], fmt=".", capsize=2, c="k", label=r"$V_\text{obs}$ (data)", zorder=2)
        ax.errorbar(r_data[i], Vbar_ALL[i][1], yerr=np.diff(Vbar_ALL[i], axis=0), fmt=".", c="red", capsize=2, label=r"$V_\text{bar}$ (data)", alpha=0.3, zorder=1)
        ax.scatter(r_data[i], Vbar_ALL[i][1], marker='.', c="red", alpha=0.5, zorder=1)
        ax.plot(radii[i], mean_pred[i][1], label=r"$V_\text{obs}$ (GPR)", zorder=10)
        ax.plot(radii[i], mean_pred[i][0], label=r"$V_\text{bar}$ (GPR)", zorder=10)
        # ax.fill_between(radii[i], lower_perc[i][1], upper_perc[i][1], alpha=0.3)    #, label=r"$1\sigma$ confidence ($V_\text{obs}$)")
        # ax.fill_between(radii[i], lower_perc[i][0], upper_perc[i][0], alpha=0.3)    #, label=r"$1\sigma$ confidence ($V_\text{bar}$)")

        ax.set_title(test_galaxies[i])
        fig.supxlabel("Radii (kpc)")
        fig.supylabel("Velocities (km/s)")
        if i == 0: ax.legend()

    plt.tight_layout()
    # if use_fits: figname = "/mnt/users/koe/plots/SPARC_example_MCMC.pdf"
    # else: 
    figname = "/mnt/users/koe/plots/SPARC_example.pdf"
    fig.savefig(figname, dpi=300, bbox_inches="tight")
    plt.close()
