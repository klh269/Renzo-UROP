#!/usr/bin/env python
# (C) 2024 Enoch Ko.
"""
Analyze data from Little Things, in search for more prominent Renzo-esque features.
Also for testing out if features can be extracted using ft_check.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpyro
import argparse

from utils_analysis.toy_GP import GP_fit

def get_things():
    galaxies = [ "cvidwa", "ddo47", "ddo50", "ddo52", "ddo53", "ddo87", "ddo101", "ddo126", "ddo133",
                "ddo154", "ddo168", "ddo210", "ddo216", "ddo216b", "ngc1569", "ngc2366", "ugc8508", "wlm" ]
    columns = [ 'R["]', "R[kpc]", "Vr[km/s]", "err_Vr", "Va[km/s]", "err_Va",
                "Vc[km/s]", "err_Vc", "Vd[km/s]", "err_Vd", "Sdens", "err_Sdens" ]
    
    radii, Vobs_ALL, errV_ALL = [], [], []
    
    for i in range(18):
    # for i in tqdm(range(18)):
        g = galaxies[i]
        data_loc = f"/mnt/users/koe/data/little_things/{g}_onlinetab.txt"

        rawdata = np.loadtxt(data_loc)
        data = pd.DataFrame(rawdata, columns=columns)
        r = data["R[kpc]"].to_numpy()
        Vobs = data["Vr[km/s]"].to_numpy()

        Vmax = max(Vobs)
        Vobs /= Vmax
        errV = data["err_Vr"].to_numpy() / Vmax

        radii.append(r)
        Vobs_ALL.append(Vobs)
        errV_ALL.append(errV)

    return galaxies, radii, Vobs_ALL, errV_ALL


def get_things_res( make_plots:bool=False, res_fits:bool=False ):
    galaxies, radii, Vobs_ALL, errV_ALL = get_things()
    
    if res_fits:
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

        numpyro.set_platform(args.device)
        numpyro.set_host_device_count(args.num_chains)

    residuals = []

    for i in range(18):
    # for i in tqdm(range(18)):
        g = galaxies[i]
        r = radii[i]
        Vobs = Vobs_ALL[i]
        errV = errV_ALL[i]

        rad = np.linspace(0., max(r), 100)

        if res_fits:
            pred_mean, pred_band = GP_fit( args, r, [Vobs, Vobs], rad, summary=False )

            # Compute residuals of fits.
            res_Vobs = []
            for k in range(len(r)):
                idx = (np.abs(rad - r[k])).argmin()
                res_Vobs.append(Vobs[k] - pred_mean[0][idx])
            np.save(f"/mnt/users/koe/gp_fits/little_things/{g}_residuals.npy", res_Vobs)
        else:
            res_Vobs = np.load(f"/mnt/users/koe/gp_fits/little_things/{g}_residuals.npy")

        residuals.append(res_Vobs)

        if make_plots:
            file_name = f"/mnt/users/koe/plots/little_things/{g}.png"

            fig1, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            ax0.set_title(f"GP fit: {g.upper()}")
            ax0.set_ylabel("Radial velocity [km/s]")

            ax0.errorbar( r, Vobs, errV, capsize=2.5, ls='none', color='tab:blue', label="Vobs -- data" )
            # Plot mean prediction from GP.
            ax0.plot(rad, pred_mean[0], color='k', label="Vobs -- GP fit")

            # Fill in 1-sigma (68%) confidence band of GP fit.
            ax0.fill_between(rad, pred_band[0][0, :], pred_band[0][1, :], color='k', alpha=0.2)

            ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
            ax0.grid()

            ax1.set_ylabel("Residuals")
            ax1.set_xlabel("Radii [kpc]")
            ax1.plot(r, res_Vobs, color='k', marker='o', label="Vobs")

            ax1.legend(bbox_to_anchor=(1, 1), loc="upper left")
            ax1.grid()

            plt.subplots_adjust(hspace=0.05)
            fig1.savefig(file_name, dpi=300, bbox_inches="tight")
            plt.close()
        
    return galaxies, radii, errV_ALL, residuals

