"""Tests for the Benettin/QR Lyapunov spectrum and its reductions (discovery.py).

Numerical checks on systems with known answers: Lorenz 63, a linear system whose
exponents are its eigenvalues, and the divergence-sum identity on the real MAOOAM
tendency. The rest pin down the pure reductions."""

import numpy as np
import jax.numpy as jnp
import pytest

from error_discovery.discovery import (lyapunov_spectrum, running_exponents, blocking_error,
                                       kolmogorov_entropy, kaplan_yorke_dim)

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def l63(x):
    return jnp.array([SIGMA * (x[1] - x[0]),
                      RHO * x[0] - x[1] - x[0] * x[2],
                      x[0] * x[1] - BETA * x[2]])


@pytest.fixture(scope="module")
def l63_spectrum():
    """2000 time units of Lorenz 63 from a point near the attractor, first 10% dropped."""
    dt, block = 0.01, 10
    logs, _ = lyapunov_spectrum(l63, np.array([1.0, 1.0, 20.0]), dt, block=block,
                                n_blocks=20000, seed=0)
    return logs[2000:].mean(axis=0) / (block * dt)


def test_lorenz63_spectrum(l63_spectrum):
    """Known spectrum (0.906, 0, -14.572); sum fixed at -(sigma + 1 + beta)."""
    lam = l63_spectrum
    assert abs(lam[0] - 0.906) < 0.02 * 0.906
    assert abs(lam[1]) < 0.02
    assert abs(kaplan_yorke_dim(lam) - 2.062) < 0.01 * 2.062
    # The exponents are those of the RK4 map, which differ from the flow's at O(dt^4):
    # at dt = 0.01 the sum sits 7.5e-6 relative from -(sigma + 1 + beta), hence 1e-4.
    assert abs(lam.sum() + (SIGMA + 1.0 + BETA)) < 1e-4 * (SIGMA + 1.0 + BETA)


def test_linear_system_exponents_are_eigenvalues():
    """xdot = A x, A diagonalisable with shuffled real eigenvalues: after the frame has
    converged, every block's stretch is exact, so the spectrum is the sorted eigenvalues.
    Catches QR sign and ordering errors."""
    eig = np.array([-1.0, 0.5, -2.0, 0.1, -0.3])
    V = np.random.default_rng(3).standard_normal((5, 5)) + 3.0 * np.eye(5)
    A = jnp.asarray(V @ np.diag(eig) @ np.linalg.inv(V))
    dt, block = 0.01, 100
    logs, _ = lyapunov_spectrum(lambda x: A @ x, np.zeros(5), dt, block=block, n_blocks=400,
                                seed=1)
    lam = logs[200:].mean(axis=0) / (block * dt)
    np.testing.assert_allclose(lam, np.sort(eig)[::-1], rtol=0, atol=1e-8)


@pytest.fixture(scope="module")
def maooam():
    from qgs_jax.spec import extract_spec
    from qgs_jax import model as jm
    from error_discovery.config import IC_CACHE
    T = jnp.asarray(extract_spec().dense_tensor)
    return T, jnp.asarray(np.load(IC_CACHE)["x"])


def test_divergence_sum_identity_maooam(maooam):
    """Sum of all 36 exponents = time average of trace(Df), which is constant for MAOOAM.

    The exponents are those of the RK4 map, whose volume contraction departs from the
    trace at high order in dt, by an amount that depends on the stretch of trajectory:
    6e-7 relative on this one at dt = 0.1, 9e-5 on one 2000 time units later. Over two
    halvings of dt the gap fell 51x and 1900x on those two. So the check is a loose bound
    at dt = 0.1, which catches a broken QR or a sign error, plus the collapse of the gap
    at dt / 4."""
    from qgs_jax import model as jm
    T, x0 = maooam
    f, jac = (lambda x: jm.tendencies(T, x)), (lambda x: jm.jacobian(T, x))
    tr = float(jnp.trace(jac(x0)))
    gaps = []
    for dt in (0.1, 0.025):
        block = int(round(10.0 / dt))
        logs, _ = lyapunov_spectrum(f, x0, dt, block=block, n_blocks=100, jac=jac)
        gaps.append(abs(logs.sum(axis=1).mean() / 10.0 - tr) / abs(tr))
    assert gaps[0] < 1e-3
    assert gaps[1] < gaps[0] / 10.0


def test_autodiff_and_closed_form_jacobians_agree(maooam):
    """The fast path (closed-form Jacobian) gives the autodiff path's log-stretches."""
    from qgs_jax import model as jm
    T, x0 = maooam
    f = lambda x: jm.tendencies(T, x)
    a, _ = lyapunov_spectrum(f, x0, 0.1, block=50, n_blocks=5)
    b, _ = lyapunov_spectrum(f, x0, 0.1, block=50, n_blocks=5, jac=lambda x: jm.jacobian(T, x))
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)


def test_single_vector_is_first_column_of_full_frame():
    dt, block = 0.01, 10
    full, _ = lyapunov_spectrum(l63, np.array([1.0, 1.0, 20.0]), dt, block=block,
                                n_blocks=500, seed=4)
    one, _ = lyapunov_spectrum(l63, np.array([1.0, 1.0, 20.0]), dt, block=block,
                               n_blocks=500, n_vec=1, seed=4)
    assert one.shape == (500, 1)
    assert abs(one.mean() - full[:, 0].mean()) / (block * dt) < 1e-10


def test_kaplan_yorke_and_kolmogorov_by_hand():
    s = np.array([-2.0, 1.0, 0.0])                   # unsorted on purpose
    assert kolmogorov_entropy(s) == 1.0
    assert kaplan_yorke_dim(s) == pytest.approx(2.5)  # k = 2, 2 + (1 + 0) / |-2|
    assert kaplan_yorke_dim([-0.1, -1.0]) == 0.0
    assert kaplan_yorke_dim([0.3, 0.1]) == 2.0


def test_reductions_are_post_hoc():
    logs = np.random.default_rng(0).standard_normal((1000, 3)) + [0.5, 0.0, -1.0]
    run = running_exponents(logs, t_block=2.0)
    np.testing.assert_allclose(run[-1], logs.mean(axis=0) / 2.0, rtol=0, atol=1e-12)
    L, se = blocking_error(logs, t_block=2.0)
    assert L[0] == 1 and 1000 // L[-1] >= 32
    np.testing.assert_allclose(se[0], logs.std(axis=0, ddof=1) / np.sqrt(1000) / 2.0,
                               rtol=1e-12)
