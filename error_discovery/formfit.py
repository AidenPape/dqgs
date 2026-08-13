"""
error_discovery.formfit
=======================

The shared vocabulary of the *unknown-form* experiments: build the twin, sample a
distribution of states, evaluate the injected error tendency on it, and score every
candidate error form against it.

Where :mod:`error_discovery.discovery` recovers coefficients when the form is
*known* (Levenberg--Marquardt through the integrator), this module answers the
prior question — which functional form is it? — by least squares over a candidate
library. The injected error is

    m(x) = eps * g(x) * (E x)

with E constant, so a candidate form g_j enters *linearly*: the best coefficient
for each candidate has a closed form, and no optimizer is involved. The verdict is
the **margin** — the best wrong form's relative residual minus the true form's.
Positive means the truth wins; :data:`config.THRESH` is where it counts as
identifiable.

Every study in the family imports its trajectories and its scoring from here, so
they are all measuring the same thing on the same distributions.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .models import TwinModels
from .discovery import rollout
from .config import DT, EPS


def twin():
    """The canonical twin: error driven by the ocean temperature mode, g = x[28]."""
    return TwinModels(eps=EPS, g_coeffs=(0.0, 0.0, 1.0, 0.0))


# --------------------------------------------------------------------------- #
# Sampling distributions. Every study compares "on the attractor" against some
# form of off-attractor excitation, so the three builders live together.
# --------------------------------------------------------------------------- #
def attractor_traj(tw, n):
    """The base model's own climatology, from the canonical cached IC."""
    from .attractor import get_attractor_ic
    from .config import ExperimentConfig
    xa = get_attractor_ic(tw, ExperimentConfig(), verbose=False)   # provenance-checked
    return np.asarray(rollout(lambda x: tw.f_base(x), xa, DT, n))


def transient_traj(tw, n, seed):
    """Off-attractor excitation: the base model released from a random state."""
    x0 = jnp.asarray(np.random.default_rng(seed).standard_normal(tw.ndim) * 0.5)
    return np.asarray(rollout(lambda x: tw.f_base(x), x0, DT, n))


def true_transient(tw, n, seed):
    """An excited trajectory of the TRUE (perturbed) model — used where the error
    must manifest dynamically (E3, E5)."""
    x0 = jnp.asarray(np.random.default_rng(seed).standard_normal(tw.ndim) * 0.5)
    return np.asarray(rollout(lambda x: tw.f_true(x), x0, DT, n))


# --------------------------------------------------------------------------- #
# The candidate library and the closed-form fit.
# --------------------------------------------------------------------------- #
def library(X, tw):
    """Over-complete predictor library: const + 36 linear + 4 quadratics."""
    vr = tw.spec.variables_range
    cols = [np.ones(len(X))] + [X[:, k] for k in range(tw.ndim)]
    names = ["const"] + [f"x{k}" for k in range(tw.ndim)]
    for a, nm in [(0, "psi_a1^2"), (tw.idx_thetaa1, "th_a1^2"),
                  (tw.idx_To1, "T_o1^2"), (vr[1], "psi_o1^2")]:
        cols.append(X[:, a] ** 2)
        names.append(nm)
    return np.stack(cols, 1), names


def true_term(tw):
    """The library name of the form actually injected."""
    return f"x{tw.idx_To1}"


def err_tendency(tw, X):
    """The injected error tendency m(x) = f_true(x) - f_base(x), row by row."""
    g = jax.vmap(lambda x: tw.f_true(x) - tw.f_base(x))
    return np.asarray(g(jnp.asarray(X)))


def fit(tw, X, m):
    """Best single-term coefficient c_j for every library predictor j (vectorised)."""
    Phi, names = library(X, tw)
    EX = X @ np.asarray(tw.E_full).T
    den = (Phi ** 2).T @ (EX * EX).sum(1)
    num = Phi.T @ (EX * m).sum(1)
    c = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return c, names


def resid(tw, c, X, m):
    """Relative residual ‖c_j phi_j(X)(E X) − m‖/‖m‖ for each predictor j (fixed c_j)."""
    Phi, _ = library(X, tw)
    EX = X @ np.asarray(tw.E_full).T
    aa = (Phi ** 2).T @ (EX * EX).sum(1)
    am = Phi.T @ (EX * m).sum(1)
    m2 = float((m * m).sum())
    return np.sqrt(np.maximum(c ** 2 * aa - 2 * c * am + m2, 0.0)) / np.sqrt(m2)


def margin_named(R, names, true_name):
    """Best wrong-form residual minus the named form's residual (>0 = that form wins).

    Returns ``(margin, is_argmin)``; the second value is 1 when the named form is
    also the outright best fit, which the margin alone does not imply once it goes
    negative. Pass ``true_term(tw)`` to score the form actually injected, or any
    other library name to score a form the truth does not contain.
    """
    t = names.index(true_name)
    others = np.delete(R, t)
    return float(others.min() - R[t]), int(np.argmin(R) == t)


# --------------------------------------------------------------------------- #
# Tendency estimators: recovering m(x) from an observed trajectory rather than
# evaluating it from the twin. Used wherever the error must be inferred from data.
# --------------------------------------------------------------------------- #
def fd_tendency(tw, Y):
    """Naive finite-difference error tendency:  (y' - y)/dt - f_base(y)."""
    fbase = np.asarray(jax.vmap(lambda x: tw.f_base(x))(jnp.asarray(Y[:-1])))
    return (Y[1:] - Y[:-1]) / DT - fbase


def forcing_recover(tw, Y):
    """Form-agnostic per-step forcing: the free delta_n with
    RK4(f_base + delta_n, y_n) = y_{n+1}. Removes the base-dynamics truncation
    that swamps the tiny error in a naive finite difference; delta_n approximates
    m(y_n) to first order in the RK4 stages (the true error varies within a step)."""
    from qgs_jax.model import rk4_step
    Yn = jnp.asarray(Y[:-1])
    step0 = jax.vmap(lambda x: rk4_step(lambda z: tw.f_base(z), x, DT))
    M = jax.vmap(lambda x: jax.jacfwd(lambda d: rk4_step(lambda z: tw.f_base(z) + d, x, DT))
                 (jnp.zeros(tw.ndim)))(Yn)
    r = jnp.asarray(Y[1:]) - step0(Yn)
    return np.asarray(jax.vmap(jnp.linalg.solve)(M, r))
