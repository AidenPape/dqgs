"""
Automatic parameter tuning demo for the differentiable JAX MAOOAM model.
========================================================================

This script demonstrates the headline capability of :mod:`qgs_jax`: recovering
unknown *physical parameters* of the MAOOAM model by gradient descent through
the model integrator -- i.e. the model is usable for automatic parameter tuning.

Experiment
----------
1.  Pick a set of "true" physical parameters (the reference MAOOAM values).
2.  Generate a *truth* trajectory on the model attractor.
3.  Corrupt a few parameters (here ``kd``, ``kdp`` and ``Lpa``) by +-30%.
4.  Define a loss = mean-squared trajectory error over several short windows,
    as a differentiable function of those parameters.
5.  Minimise it with Adam (reverse-mode autodiff supplies exact gradients),
    and watch the parameters return to their true values.

Run::

    cd help_aiden
    python tune_demo.py
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "qgs"))

from qgs_jax.spec import extract_spec
from qgs_jax import model, assemble


# Parameters we pretend not to know and will recover.
TUNED = ["kd", "kdp", "Lpa"]
# Multiplicative corruption applied to the initial guess.
CORRUPT = {"kd": 1.30, "kdp": 0.70, "Lpa": 1.25}

DT = 0.1
WINDOW = 40           # steps per assimilation window
N_WINDOWS = 6         # number of short windows used in the loss
SPINUP_STEPS = 40000  # transient to reach the attractor
N_ITERS = 300
LR = 0.05             # Adam learning rate (in log-parameter space)


def adam(grad_fn, z0, n_iters, lr, verbose_every=25):
    """Minimal Adam optimiser. ``grad_fn(z) -> (loss, grad)``."""
    z = z0
    m = jnp.zeros_like(z)
    v = jnp.zeros_like(z)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []
    for t in range(1, n_iters + 1):
        loss, g = grad_fn(z)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g ** 2)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        z = z - lr * mhat / (jnp.sqrt(vhat) + eps)
        history.append(float(loss))
        if verbose_every and (t % verbose_every == 0 or t == 1):
            print(f"  iter {t:4d}   loss = {float(loss):.6e}")
    return z, history


def main():
    print("Building reference MAOOAM model and extracting differentiable assembly ...")
    spec = extract_spec()
    geom_jnp, theta0, tensor_of_theta = assemble.make_tensor_fn(spec)
    ndim = spec.ndim

    # True parameters and true tensor.
    theta_true = {k: jnp.asarray(float(spec.theta[k])) for k in assemble.THETA_KEYS}
    T_true = tensor_of_theta(theta_true)

    # --- generate an on-attractor initial condition and truth windows --------
    print(f"Spinning up onto the attractor ({SPINUP_STEPS} steps) ...")
    rng = np.random.default_rng(2024)
    x = jnp.asarray(rng.standard_normal(ndim) * 0.01)
    x = model.integrate_final(T_true, x, DT, SPINUP_STEPS, method="rk4")

    # Collect N_WINDOWS starting states spaced along a long truth run, and the
    # truth trajectory for each window.
    starts = []
    truth_windows = []
    for _ in range(N_WINDOWS):
        starts.append(x)
        win = model.integrate(T_true, x, DT, WINDOW, write_every=1, method="rk4")
        truth_windows.append(win)
        # advance well past this window to decorrelate the next start
        x = model.integrate_final(T_true, win[-1], DT, 2000, method="rk4")
    starts = jnp.stack(starts)              # (N_WINDOWS, ndim)
    truth_windows = jnp.stack(truth_windows)  # (N_WINDOWS, WINDOW, ndim)

    # Normalisation scale for the loss (per-variable std over all truth data).
    scale = jnp.std(truth_windows.reshape(-1, ndim), axis=0) + 1e-12

    # --- parameterise the tuned params multiplicatively ----------------------
    # theta_k = theta_true_k * exp(z_k);  z = 0 recovers the truth.
    base = jnp.asarray([float(spec.theta[k]) for k in TUNED])

    def make_theta(z):
        theta = dict(theta_true)
        factors = jnp.exp(z)
        for i, k in enumerate(TUNED):
            theta[k] = base[i] * factors[i]
        return theta

    def loss_of_z(z):
        theta = make_theta(z)
        T = tensor_of_theta(theta)

        def one_window(x0, truth):
            pred = model.integrate(T, x0, DT, WINDOW, write_every=1, method="rk4")
            return jnp.mean(((pred - truth) / scale) ** 2)

        return jnp.mean(jax.vmap(one_window)(starts, truth_windows))

    grad_fn = jax.jit(jax.value_and_grad(loss_of_z))

    # initial guess: corrupt the tuned parameters
    z0 = jnp.asarray([np.log(CORRUPT[k]) for k in TUNED])

    print("\nInitial guess (corrupted) vs truth:")
    for i, k in enumerate(TUNED):
        guess = float(base[i]) * CORRUPT[k]
        print(f"  {k:5s}: guess = {guess:.6e}   true = {float(base[i]):.6e}   "
              f"(off by {100*(CORRUPT[k]-1):+.0f}%)")
    print(f"\nInitial loss = {float(loss_of_z(z0)):.6e}")

    print(f"\nOptimising {len(TUNED)} parameters with Adam ({N_ITERS} iters) ...")
    z_opt, hist = adam(grad_fn, z0, N_ITERS, LR)

    print("\nRecovered parameters vs truth:")
    max_rel = 0.0
    for i, k in enumerate(TUNED):
        rec = float(base[i]) * float(jnp.exp(z_opt[i]))
        rel = abs(rec - float(base[i])) / abs(float(base[i]))
        max_rel = max(max_rel, rel)
        print(f"  {k:5s}: recovered = {rec:.6e}   true = {float(base[i]):.6e}   "
              f"rel.err = {rel:.2e}")

    print(f"\nFinal loss = {float(loss_of_z(z_opt)):.6e}")
    print(f"Max relative parameter error after tuning: {max_rel:.2e}")
    if max_rel < 1e-2:
        print("\nSUCCESS: parameters recovered to better than 1% by gradient descent.")
    else:
        print("\nNOTE: parameters not fully recovered to 1% (try more iters / windows).")
    return max_rel


if __name__ == "__main__":
    main()
