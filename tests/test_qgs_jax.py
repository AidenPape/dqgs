"""
Test-suite verifying that the JAX MAOOAM implementation (:mod:`qgs_jax`) is
operational and 1:1 with the original numpy/numba ``qgs`` code.

Run with::

    pytest test_qgs_jax.py -v

The suite checks, to machine precision:

1.  Tendencies ``f(x)``      : JAX vs original qgs.
2.  Jacobian ``Df(x)``       : JAX-analytic vs JAX-autodiff vs original qgs.
3.  Full RK4 trajectories    : JAX vs original qgs RungeKuttaIntegrator.
4.  Differentiable assembly  : pure-JAX tensor == original qgs tensor.
5.  Parameter gradients      : autodiff vs finite differences (incl. params that
                               flow through matrix inverses, sig0 and G).
6.  End-to-end tunability    : gradient of a trajectory-based loss w.r.t. a
                               physical parameter is finite and finite-diff-correct.
"""

import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from qgs_jax.spec import extract_spec                 # noqa: E402
from qgs_jax import model, assemble                   # noqa: E402


# ---- shared fixtures ------------------------------------------------------- #
@pytest.fixture(scope="module")
def spec():
    return extract_spec()


@pytest.fixture(scope="module")
def T_ref(spec):
    return jnp.asarray(spec.dense_tensor)


@pytest.fixture(scope="module")
def assembled(spec):
    geom_jnp, theta0, tensor_of_theta = assemble.make_tensor_fn(spec)
    return geom_jnp, theta0, tensor_of_theta


def _random_states(ndim, n=8, scale=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(ndim) * scale for _ in range(n)]


# ---- 1. tendencies --------------------------------------------------------- #
def test_tendencies_match(spec, T_ref):
    for x in _random_states(spec.ndim, seed=1):
        f_np = spec.f_numpy(0.0, x)
        f_jx = np.asarray(model.tendencies(T_ref, jnp.asarray(x)))
        assert np.allclose(f_np, f_jx, atol=1e-12, rtol=0), np.max(np.abs(f_np - f_jx))


# ---- 2. Jacobian ----------------------------------------------------------- #
def test_jacobian_analytic_matches_numpy(spec, T_ref):
    for x in _random_states(spec.ndim, seed=2):
        D_np = spec.Df_numpy(0.0, x)
        D_jx = np.asarray(model.jacobian(T_ref, jnp.asarray(x)))
        assert np.allclose(D_np, D_jx, atol=1e-11, rtol=0), np.max(np.abs(D_np - D_jx))


def test_jacobian_autodiff_matches_analytic(spec, T_ref):
    for x in _random_states(spec.ndim, seed=3):
        D_an = np.asarray(model.jacobian(T_ref, jnp.asarray(x)))
        D_ad = np.asarray(model.jacobian_autodiff(T_ref, jnp.asarray(x)))
        assert np.allclose(D_an, D_ad, atol=1e-12, rtol=0)


# ---- 3. trajectory --------------------------------------------------------- #
def test_trajectory_matches_numpy(spec, T_ref):
    from qgs.integrators.integrator import RungeKuttaIntegrator

    rng = np.random.default_rng(7)
    x0 = rng.standard_normal(spec.ndim) * 0.01
    dt, nsteps = 0.1, 1500

    integ = RungeKuttaIntegrator()
    integ.set_func(spec.f_numpy)
    integ.integrate(0.0, nsteps * dt, dt, ic=x0.copy(), write_steps=1)
    _, traj_np = integ.get_trajectories()
    traj_np = np.asarray(traj_np).T            # (nt+1, ndim), includes IC

    traj_jx = np.asarray(
        model.integrate(T_ref, jnp.asarray(x0), dt, nsteps, write_every=1, method="rk4")
    )
    # numpy stores the IC at index 0; jax stores steps only.
    err = np.max(np.abs(traj_np[1:] - traj_jx))
    assert err < 1e-10, err


# ---- 4. differentiable assembly is bit-exact ------------------------------- #
def test_assembly_matches_reference_tensor(spec, assembled):
    _, theta0, tensor_of_theta = assembled
    T_asm = np.asarray(tensor_of_theta(theta0))
    T_ref = spec.dense_tensor
    # The qgs tensor is stored upper-triangularised over axes (1,2); compare the
    # symmetrised tensors, which are invariant under that simplification.
    S_asm = T_asm + np.swapaxes(T_asm, 1, 2)
    S_ref = T_ref + np.swapaxes(T_ref, 1, 2)
    assert np.max(np.abs(S_asm - S_ref)) < 1e-12


def test_assembly_tendencies_match(spec, assembled):
    _, theta0, tensor_of_theta = assembled
    T_asm = tensor_of_theta(theta0)
    for x in _random_states(spec.ndim, seed=4):
        f_np = spec.f_numpy(0.0, x)
        f_asm = np.asarray(model.tendencies(T_asm, jnp.asarray(x)))
        assert np.allclose(f_np, f_asm, atol=1e-12, rtol=0)


# ---- 5. parameter gradients ------------------------------------------------ #
@pytest.mark.parametrize("key", ["kd", "sig0", "G", "Lpa", "beta", "Cpa0", "od", "Lpgo"])
def test_param_gradient_finite_difference(spec, assembled, key):
    _, theta0, tensor_of_theta = assembled
    x = jnp.asarray(np.random.default_rng(5).standard_normal(spec.ndim) * 0.1)

    def loss(theta):
        T = tensor_of_theta(theta)
        f = model.tendencies(T, x)
        return jnp.sum(f ** 2)

    g = jax.grad(loss)(theta0)
    eps = max(abs(float(theta0[key])), 1.0) * 1e-6
    tp = dict(theta0); tm = dict(theta0)
    tp[key] = theta0[key] + eps
    tm[key] = theta0[key] - eps
    fd = (float(loss(tp)) - float(loss(tm))) / (2 * eps)
    ad = float(g[key])
    # Central finite differences are the *less* accurate side here; for
    # near-cancellation gradients (e.g. od ~ 1e-8) FD truncation noise dominates,
    # so 1e-3 relative agreement is the realistic bar.  Autodiff is exact.
    denom = abs(fd) + 1e-12
    assert abs(ad - fd) / denom < 1e-3, (key, ad, fd)


# ---- 6. end-to-end tunability through the integrator ----------------------- #
def test_trajectory_loss_gradient(spec, assembled):
    _, theta0, tensor_of_theta = assembled
    x0 = jnp.asarray(np.random.default_rng(9).standard_normal(spec.ndim) * 0.01)
    dt, nsteps = 0.1, 50

    def traj_loss(kd_value):
        theta = dict(theta0)
        theta["kd"] = kd_value
        T = tensor_of_theta(theta)
        traj = model.integrate(T, x0, dt, nsteps, write_every=1, method="rk4")
        return jnp.sum(traj[-1] ** 2)

    ad = float(jax.grad(traj_loss)(theta0["kd"]))
    eps = 1e-7
    fd = (float(traj_loss(theta0["kd"] + eps)) - float(traj_loss(theta0["kd"] - eps))) / (2 * eps)
    assert np.isfinite(ad)
    assert abs(ad - fd) / (abs(fd) + 1e-12) < 1e-4, (ad, fd)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
