# (C) 2024 Enoch Ko.
"""
Utilities and functions for non-linear regression with Gaussian Process.
"""
import jax.experimental
import time
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
)


# Squared exponential kernel with diagonal noise term
def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k


def model(X, Y, vr=0, ls=0, ns=0):
    # set uninformative log-normal priors on our three kernel hyperparameters
    if vr == 0:
        var = numpyro.sample("var", dist.LogNormal(0.0, 1.0))
    else:
        var = vr

    if ls == 0:
        length = numpyro.sample("length", dist.Uniform(max(X)/2., max(X)))
    else:
        length = ls
    
    if ns == 0:
        noise = numpyro.sample("noise", dist.LogNormal(0.0, 1.0))
    else:
        noise = ns

    k = kernel(X, X, var, length, noise)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
        obs=Y,
    )


# helper function for doing hmc inference
def run_inference(model, args, rng_key, X, Y, vr=0, ls=0, ns=0, summary:bool=True):
    start = time.time()
    # demonstrate how to use different HMC initialization strategies
    if args.init_strategy == "median":
        init_strategy = init_to_median(num_samples=100)
    elif args.init_strategy == "feasible":
        init_strategy = init_to_feasible()
    elif args.init_strategy == "sample":
        init_strategy = init_to_sample()
    elif args.init_strategy == "uniform":
        init_strategy = init_to_uniform(radius=1)
    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning,
        progress_bar=args.testing,
    )
    mcmc.run(rng_key, X, Y, vr, ls, ns)
    if (vr, ls, ns) == (0, 0, 0) and summary:
        mcmc.print_summary()
        print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for Gaussian process predictions
def predict(rng_key, X, Y, rad, var, length, noise, use_cholesky=True):
    # compute kernels between train and test data, etc.
    k_pp = kernel(rad, rad, var, length, noise, include_noise=True)
    k_pX = kernel(rad, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)

    # since K_xx is symmetric positive-definite, we can use the more efficient and
    # stable Cholesky decomposition instead of matrix inversion
    if use_cholesky:
        K_xx_cho = jax.scipy.linalg.cho_factor(k_XX)
        K = k_pp - jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, k_pX.T))
        mean = jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, Y))
    else:
        K_xx_inv = jnp.linalg.inv(k_XX)
        K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))

    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
        rng_key, rad.shape[:1]
    )

    # Return both the mean function and a sample from the 
    # posterior predictive for the given set of hyperparameters
    return mean, mean + sigma_noise
