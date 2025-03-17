# (C) 2024 Enoch Ko.
"""
GP regression for Vobs in toy model (from Vbar with arctan RC + gaussian bump).
"""
import numpy as np
import matplotlib.pyplot as plt

import jax.experimental

import jax
from jax import vmap
import jax.random as random
import corner

from utils_analysis.gp_utils import model, predict, run_inference


def GP_fit(args, r, vel, rad, make_plots:bool=False, file_name:str="", summary:bool=False):
    # Fixed lengthscale to 5.0 for GP fit.
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, r, vel, summary=summary)

    # Do prediction for Vbar (Y[0]; to obtain GP hyperparameters).
    vmap_args = (
        random.split(rng_key_predict, samples["var"].shape[0]),
        samples["var"],
        samples["noise"],
    )
    means, predictions = vmap(
        lambda rng_key, var, noise: predict(
            rng_key, r, vel, rad, var, 0.5*max(r), noise, use_cholesky=args.use_cholesky
        )
    )(*vmap_args)

    mean_pred = np.mean(means, axis=0)
    confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)

    if make_plots:
        labels = ["var", "noise"]
        samples_arr = np.vstack([samples[label] for label in labels]).T
        fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
        fig.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Returns the list of predicted mean and 1-sigma confidence band
    return mean_pred, confidence_band


# def GP_fit_old(args, r, Y, rad, make_plots:bool=False, file_name:str="", summary:bool=False):
#     # Initialize lists of means and 1-sigma bands for GP predictions;
#     # List has to be of the form [ Vbar (to fit), Vobs (fixed params) ].
#     pred_means, pred_bands = [], []

#     rng_key, rng_key_predict = random.split(random.PRNGKey(0))
#     samples = run_inference(model, args, rng_key, r, Y[0], summary=summary)

#     # Do prediction for Vbar (Y[0]; to obtain GP hyperparameters).
#     vmap_args = (
#         random.split(rng_key_predict, samples["var"].shape[0]),
#         samples["var"],
#         samples["length"],
#         samples["noise"],
#     )
#     means, predictions = vmap(
#         lambda rng_key, var, length, noise: predict(
#             rng_key, r, Y[0], rad, var, length, noise, use_cholesky=args.use_cholesky
#         )
#     )(*vmap_args)

#     mean_pred = np.mean(means, axis=0)
#     pred_means.append(mean_pred)

#     confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
#     pred_bands.append(confidence_band)

#     if make_plots:
#         labels = ["length", "var", "noise"]
#         samples_arr = np.vstack([samples[label] for label in labels]).T
#         fig = corner.corner(samples_arr, show_titles=True, labels=labels, title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], smooth=1)
#         fig.savefig(file_name, dpi=300, bbox_inches="tight")
#         plt.close(fig)

#     # GP on Vbar and mock Vobs with fixed lengthscale from Vobs (data).
#     vr = np.median(samples["var"])
#     ls = np.median(samples["length"])
#     ns = np.median(samples["noise"])

#     rng_key, rng_key_predict = random.split(random.PRNGKey(0))

#     # Apply GP with fixed hyperparameters to Vobs (Y[1]).
#     vmap_args = (
#         random.split(rng_key_predict, samples["var"].shape[0]),
#     )
#     means, predictions = vmap(
#         lambda rng_key: predict(
#             rng_key, r, Y[1], rad, vr, ls, ns, use_cholesky=args.use_cholesky
#         )
#     )(*vmap_args)

#     mean_pred = np.mean(means, axis=0)
#     pred_means.append(mean_pred)

#     confidence_band = np.percentile(predictions, [16.0, 84.0], axis=0)
#     pred_bands.append(confidence_band)

#     # Returns the list of predicted means and 1-sigma confidence bands
#     return pred_means, pred_bands


def get_residuals(r, Y, rad, pred_means, pred_bands, make_plots:bool=False, file_name:str="", Vbar:bool=False):
    # Compute residuals of fits.
    res_Vbar, res_Vobs = [], []
    for k in range(len(r)):
        idx = (np.abs(rad - r[k])).argmin()
        if Vbar: 
            res_Vbar.append(Y[0][k] - pred_means[0][idx])
            res_Vobs.append(Y[1][k] - pred_means[1][idx])
        else:
            res_Vobs.append(Y[k] - pred_means[idx])
    if Vbar: residuals = np.array([ res_Vbar, res_Vobs ])
    else: residuals = np.array(res_Vobs)

    if make_plots:
        fig1, (ax0, ax1) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax0.set_title("GP fit")
        ax0.set_ylabel("Normalized velocities")

        colours = [ 'red', 'k' ]
        labels = [ "Vbar", "Vobs" ]
        for j in range(2):
            ax0.scatter(r, Y[j], color=colours[j], alpha=0.3)
            # Plot mean prediction from GP.
            ax0.plot(rad, pred_means[j], color=colours[j], label=labels[j])

            # Fill in 1-sigma (68%) confidence band of GP fit.
            ax0.fill_between(rad, pred_bands[j][0, :], pred_bands[j][1, :], color=colours[j], alpha=0.2)

        ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")
        ax0.grid()

        ax1.set_ylabel("Residuals")
        ax1.set_xlabel("Radii (kpc)")
        for j in range(2):
            ax1.plot(r, residuals[j], color=colours[j], marker="o", alpha=0.5, label=labels[j])

        ax1.grid()

        plt.subplots_adjust(hspace=0.05)
        fig1.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()

    # Returns the list of residuals ([ [res_Vbar, res_Vobs] x r ] or [ res_Vobs x r ]).
    return residuals
