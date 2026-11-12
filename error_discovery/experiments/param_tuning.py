"""
error_discovery.experiments.param_tuning
========================================

The warm-up: a model with the *right structure* but the wrong constants.

Three physical parameters -- the two friction coefficients ``kd`` and ``kdp``, and the
atmospheric radiative-coupling scale ``Lpa`` -- are corrupted by 25-30%, and Adam walks
them back to truth by differentiating a forecast-mismatch loss through the RK4
integrator. It recovers them to ~1e-8, and that is the point: given a differentiable
model and a model class that contains the truth, parameter estimation is not the hard
problem.

Which is why this study sits with the diagnostics rather than in the argument. Every
other study here is about the case this one excludes -- an error whose *functional
form* is unknown, and which no amount of tuning ``kd`` can absorb. This is the contrast
that makes the rest of the package worth doing, and it is Lecture 1's figure.

It is also the one study that optimises over ``qgs_jax.assemble``'s *physical*
parameters rather than over injected-error coefficients, so it uses neither
:mod:`~error_discovery.models` nor :mod:`~error_discovery.discovery`. The tuned
parameters enter through ``tensor_of_theta``, which rebuilds the entire tendency tensor
-- four matrix inverses included -- as a traced JAX function.

``run`` stores the whole optimiser history (``z``, the loss, the gradient, and the
effective Adam step), not only the parameter error that gets plotted, so questions
about *how* the optimiser converged can be put to the cached ``.npz`` instead of
costing another run.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp

from qgs_jax import assemble, model
from qgs_jax.spec import extract_spec

from .. import figstyle
from ..config import DT, DATA_DIR, FIGURES_DIR

# The tuned parameters, and the factor each is multiplied by before tuning starts.
# Two frictions and one radiative coupling: enough to be a genuine three-parameter
# fit, few enough to read off a single panel.
TUNED = ("kd", "kdp", "Lpa")
CORRUPT = (1.30, 0.70, 1.25)

_NPZ = os.path.join(DATA_DIR, "param_tuning.npz")


def run(window=40, n_windows=6, spinup=40_000, gap=2_000, n_iters=300,
        lr=0.05, beta1=0.9, beta2=0.999, seed=2024, verbose=True):
    """Corrupt three parameters, then tune them back by Adam through the integrator.

    The truth is ``n_windows`` short windows of ``window`` steps, spaced ``gap`` steps
    apart along one trajectory of the uncorrupted model, so the loss averages over
    several points of the attractor rather than betting on one. Each window is scored
    against the per-component standard deviation of the truth, which puts the 36 modes
    -- whose amplitudes differ by orders of magnitude -- on a common footing.

    Tuning happens in log space, ``theta_k = theta_k_true * exp(z_k)``: it keeps every
    parameter positive, and it makes ``exp(z)`` the relative error directly.
    """
    spec = extract_spec()
    _geom, _theta0, tensor_of_theta = assemble.make_tensor_fn(spec)
    theta_true = {k: jnp.asarray(float(spec.theta[k])) for k in assemble.THETA_KEYS}
    T_true = tensor_of_theta(theta_true)
    base = jnp.asarray([float(spec.theta[k]) for k in TUNED])

    # Spin onto the attractor, then cut n_windows truth segments out of one trajectory.
    # This study does not use the shared cached IC: it integrates the *tuned-parameter*
    # tensor, not the twin's base tensor, so attractor.get_attractor_ic does not apply.
    rng = np.random.default_rng(seed)
    x = model.integrate_final(
        T_true, jnp.asarray(rng.standard_normal(spec.ndim) * 0.01), DT, spinup)
    starts, truth = [], []
    for _ in range(n_windows):
        starts.append(x)
        win = model.integrate(T_true, x, DT, window, write_every=1)
        truth.append(win)
        x = model.integrate_final(T_true, win[-1], DT, gap)
    starts, truth = jnp.stack(starts), jnp.stack(truth)
    scale = jnp.std(truth.reshape(-1, spec.ndim), axis=0) + 1e-12

    def loss_of_z(z):
        theta = dict(theta_true)
        for i, k in enumerate(TUNED):
            theta[k] = base[i] * jnp.exp(z[i])
        T = tensor_of_theta(theta)
        f = lambda x0, tr: jnp.mean(
            ((model.integrate(T, x0, DT, window, write_every=1) - tr) / scale) ** 2)
        return jnp.mean(jax.vmap(f)(starts, truth))

    vg = jax.jit(jax.value_and_grad(loss_of_z))

    # Adam, written out rather than pulled from optax: it is nine lines, and the
    # bias-corrected moments are what the stored history is *of*.
    z = jnp.asarray([np.log(c) for c in CORRUPT])
    m = jnp.zeros_like(z)
    v = jnp.zeros_like(z)
    z_hist, loss_hist, grad_hist, step_hist = [np.asarray(z)], [], [], []
    for t in range(1, n_iters + 1):
        loss, g = vg(z)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g ** 2
        step = (m / (1 - beta1 ** t)) / (jnp.sqrt(v / (1 - beta2 ** t)) + 1e-8)
        z = z - lr * step
        z_hist.append(np.asarray(z))
        loss_hist.append(float(loss))
        grad_hist.append(np.asarray(g))
        step_hist.append(np.asarray(step))

    z_hist = np.array(z_hist)
    relerr = np.abs(np.exp(z_hist) - 1.0)          # exp(z) = theta / theta_true

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(_NPZ, z=z_hist, loss=np.array(loss_hist), grad=np.array(grad_hist),
             step=np.array(step_hist), tuned=np.array(TUNED), corrupt=np.array(CORRUPT),
             window=window, n_windows=n_windows, spinup=spinup, gap=gap,
             n_iters=n_iters, lr=lr, beta1=beta1, beta2=beta2, seed=seed, dt=DT)
    if verbose:
        print("[param_tuning] final relative errors: "
              + "  ".join(f"{k}={e:.2e}" for k, e in zip(TUNED, relerr[-1])))
        print(f"[param_tuning] wrote {_NPZ}")
    return {"z": z_hist, "relerr": relerr, "loss": np.array(loss_hist)}


def figure():
    """Relative parameter error against Adam iteration, one curve per parameter.

    Iteration 0 is the corrupted starting point, so every curve begins at exactly the
    offset its legend entry names. The script this study replaced recorded the history
    only *after* each update, which put its first plotted point one step in and started
    the +30% curve near 0.2 -- a quarter of the way down the first decade, and visibly
    at odds with its own label.
    """
    plt = figstyle.pyplot()
    d = np.load(_NPZ)
    tuned = [str(t) for t in d["tuned"]]
    corrupt = np.asarray(d["corrupt"], dtype=float)
    relerr = np.abs(np.exp(np.asarray(d["z"])) - 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, k in enumerate(tuned):
        ax.plot(relerr[:, i], label=f"{k}  (start {100 * (corrupt[i] - 1):+.0f}%)")
    ax.set_yscale("log")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("relative parameter error")
    ax.set_title("Differentiable parameter tuning: gradients recover the truth")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, "fig_param_tuning.png")
    fig.savefig(out, dpi=figstyle.DPI)
    plt.close(fig)
    return [out]
