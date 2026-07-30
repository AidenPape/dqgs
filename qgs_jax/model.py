"""
qgs_jax.model
=============

JAX tendencies and integrators for the MAOOAM model.

The model is fully specified by its rank-3 tendency tensor ``T`` of shape
``(ndim+1, ndim+1, ndim+1)``::

    f(x)_i = sum_{j,k} T[i, j, k] * xbar[j] * xbar[k]      with  xbar = (1, x)

Given ``T`` (from :mod:`qgs_jax.spec` extraction, or from the differentiable
:mod:`qgs_jax.assemble`), this module provides:

* ``tendencies(T, x)`` -- the vector field ``f(x)``,
* ``jacobian(T, x)``   -- the analytic Jacobian ``Df(x)`` (closed form),
* RK2 / RK4 fixed-step integrators built on ``jax.lax.scan`` so that whole
  trajectories are differentiable and ``jit``-compilable.

Everything is written so the *same* code path differentiates with respect to the
state ``x`` (giving the tangent-linear / adjoint model for free) **and** with
respect to the tensor ``T`` -- which, composed with :mod:`qgs_jax.assemble`,
yields gradients with respect to the physical parameters for automatic tuning.
"""

import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)


def augment(x):
    """Return ``xbar = (1, x_1, ..., x_n)``."""
    return jnp.concatenate([jnp.ones((1,), dtype=x.dtype), x])


def tendencies(T, x):
    """MAOOAM tendencies ``f(x)`` from the rank-3 tensor ``T``.

    Parameters
    ----------
    T : (n+1, n+1, n+1) array
    x : (n,) array

    Returns
    -------
    (n,) array
        ``dx/dt``.
    """
    xb = augment(x)
    # full = sum_{j,k} T[i,j,k] xb_j xb_k  -> shape (n+1,)
    full = jnp.einsum("ijk,j,k->i", T, xb, xb)
    return full[1:]


def jacobian(T, x):
    """Analytic Jacobian ``Df(x)`` of the tendencies.

    Because ``f_i = sum_{j,k} T_ijk xbar_j xbar_k`` with ``xbar = (1, x)``,
    ``d f_i / d x_m = sum_k (T[i, m+1, k] + T[i, k, m+1]) xbar_k``.
    This matches the original ``qgs`` Jacobian tensor construction
    (``J = T + T.swapaxes(1, 2)``) contracted with ``xbar``.
    """
    xb = augment(x)
    J = T + jnp.swapaxes(T, 1, 2)            # (n+1, n+1, n+1)
    full = jnp.einsum("ijk,k->ij", J, xb)    # (n+1, n+1)
    return full[1:, 1:]


# ``jax.jacfwd``/``jacrev`` of ``tendencies`` give the same matrix; the closed
# form above avoids a second trace and is what the test-suite cross-checks
# against autodiff.
def jacobian_autodiff(T, x):
    """Jacobian via forward-mode autodiff (cross-check of :func:`jacobian`)."""
    return jax.jacfwd(lambda y: tendencies(T, y))(x)


# --------------------------------------------------------------------------- #
# Fixed-step integrators (differentiable, jit-friendly)
# --------------------------------------------------------------------------- #
def rk4_step(f, x, dt):
    """One classical Runge-Kutta 4 step of an autonomous field ``f(x)``."""
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk2_step(f, x, dt):
    """One Heun / RK2 step (matches qgs default ``RungeKuttaIntegrator`` order 2)."""
    k1 = f(x)
    k2 = f(x + dt * k1)
    return x + 0.5 * dt * (k1 + k2)


@partial(jax.jit, static_argnums=(3, 4, 5))
def integrate(T, x0, dt, n_steps, write_every=1, method="rk4"):
    """Integrate the model and return the sub-sampled trajectory.

    Parameters
    ----------
    T : tendency tensor
    x0 : (n,) initial condition
    dt : float, time step
    n_steps : int, number of integration steps
    write_every : int, store the state every ``write_every`` steps
    method : {"rk4", "rk2"}

    Returns
    -------
    (n_out, n) array
        Stored states (excluding the initial condition), where
        ``n_out = n_steps // write_every``.
    """
    f = lambda x: tendencies(T, x)
    step = rk4_step if method == "rk4" else rk2_step

    def body(x, i):
        x = step(f, x, dt)
        return x, x

    xs = jnp.arange(n_steps)
    xf, traj = jax.lax.scan(body, x0, xs)
    if write_every > 1:
        traj = traj[write_every - 1 :: write_every]
    return traj


@partial(jax.jit, static_argnums=(3, 4))
def integrate_final(T, x0, dt, n_steps, method="rk4"):
    """Integrate and return only the final state (cheap, for transients)."""
    f = lambda x: tendencies(T, x)
    step = rk4_step if method == "rk4" else rk2_step

    def body(x, _):
        return step(f, x, dt), None

    xf, _ = jax.lax.scan(body, x0, None, length=n_steps)
    return xf
