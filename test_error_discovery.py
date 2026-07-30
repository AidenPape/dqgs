"""
test_error_discovery.py
=======================

Tests for the differentiable error-discovery experiment.

Coverage:
  1. The JAX truth model matches the validated numpy perturbed model to machine
     precision, for several g(x) choices and configs.
  2. The discovery model reduces exactly to truth at c_true and to base at c=0.
  3. Predictor indices are derived correctly from the config.
  4. The autodiff residual Jacobian is finite (gradients flow).
  5. Levenberg-Marquardt never increases the cost.
  6. End-to-end: noiseless recovery at an intermediate window recovers c_true.
  7. Short windows are under-determined (cond(J^T J) is huge).
  8. Observation noise degrades recovery.

Run:  pytest test_error_discovery.py -v
"""

import sys

import numpy as np
import jax.numpy as jnp
import pytest

from error_discovery.models import TwinModels, obs_mask
from error_discovery import discovery as D
from error_discovery.experiment import get_attractor_ic
from error_discovery.config import ExperimentConfig


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tw():
    return TwinModels(eps=0.1, g_coeffs=(0.0, 0.0, 1.0, 0.0))


@pytest.fixture(scope="module")
def attractor(tw):
    # Use the canonical config so the shared IC cache is loaded, not rebuilt/clobbered.
    return get_attractor_ic(tw, ExperimentConfig(), verbose=False)


def _states(ndim, n=12, scale=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(ndim) * scale for _ in range(n)]


# ---- 1. truth model matches the numpy perturbed implementation ------------ #
@pytest.mark.parametrize("g_coeffs", [(0, 0, 1, 0), (0.5, 0.3, 1.0, 0.7), (1, 0, 0, 0)])
def test_truth_matches_numpy(g_coeffs):
    twg = TwinModels(eps=0.1, g_coeffs=g_coeffs)
    for x in _states(twg.ndim, seed=1):
        f_jax = np.asarray(twg.f_true(jnp.asarray(x)))
        f_np = twg.f_pert_np(0.0, x)
        assert np.allclose(f_jax, f_np, atol=1e-12, rtol=0), np.max(np.abs(f_jax - f_np))


# ---- 2. discovery model reduces correctly --------------------------------- #
def test_model_reduces(tw):
    for x in _states(tw.ndim, seed=2):
        xj = jnp.asarray(x)
        assert float(jnp.max(jnp.abs(tw.f_model(xj, tw.c_true) - tw.f_true(xj)))) == 0.0
        assert float(jnp.max(jnp.abs(tw.f_model(xj, jnp.zeros(4)) - tw.f_base(xj)))) == 0.0


# ---- 3. predictor indices ------------------------------------------------- #
def test_predictor_indices(tw):
    vr = tw.spec.variables_range
    assert (tw.idx_psia1, tw.idx_thetaa1, tw.idx_To1) == (0, int(vr[0]), int(vr[2]))


# ---- 4. autodiff Jacobian is finite --------------------------------------- #
def test_residual_jacobian_finite(tw, attractor):
    import jax
    Y = np.asarray(D.rollout(lambda x: tw.f_true(x), attractor, 0.1, 320))
    mask = obs_mask(tw.ndim, tw.spec.variables_range, "full")
    resid = D.make_residual(tw, attractor, Y, 0.1, D.obs_indices(mask))
    J = np.asarray(jax.jacfwd(resid)(jnp.zeros(4)))
    assert J.shape[1] == 4
    assert np.all(np.isfinite(J))


# ---- 5. LM does not increase the cost ------------------------------------- #
def test_lm_monotone(tw, attractor):
    Y = np.asarray(D.rollout(lambda x: tw.f_true(x), attractor, 0.1, 1280))
    mask = obs_mask(tw.ndim, tw.spec.variables_range, "full")
    resid = D.make_residual(tw, attractor, Y, 0.1, D.obs_indices(mask))
    _, _, hist = D.lm_fit(resid, jnp.zeros(4), n_iters=8)
    assert np.all(np.diff(hist) <= 1e-12)


# ---- 6. end-to-end noiseless recovery at an intermediate window ----------- #
def test_recovery_intermediate_window(tw, attractor):
    Y = np.asarray(D.rollout(lambda x: tw.f_true(x), attractor, 0.1, 1280))  # ~14 d
    mask = obs_mask(tw.ndim, tw.spec.variables_range, "full")
    r = D.recover(tw, attractor, Y, 0.1, mask, eta=0.0, n_iters=12)
    assert r["rel_err"] < 0.05, r
    # the active coefficient (T_o,1) is recovered; spurious ones stay small
    assert abs(r["c_hat"][2] - 0.1) < 0.02
    assert np.all(np.abs(r["c_hat"][[0, 1, 3]]) < 0.02)


# ---- 7. short window is under-determined ---------------------------------- #
def test_short_window_illconditioned(tw, attractor):
    Y = np.asarray(D.rollout(lambda x: tw.f_true(x), attractor, 0.1, 40))    # ~0.45 d
    mask = obs_mask(tw.ndim, tw.spec.variables_range, "full")
    r = D.recover(tw, attractor, Y, 0.1, mask, eta=0.0, n_iters=8)
    assert r["cond"] > 1e10, r["cond"]


# ---- 8. observation noise degrades recovery ------------------------------- #
def test_noise_degrades(tw, attractor):
    Y = np.asarray(D.rollout(lambda x: tw.f_true(x), attractor, 0.1, 1280))
    mask = obs_mask(tw.ndim, tw.spec.variables_range, "full")
    clean = D.recover(tw, attractor, Y, 0.1, mask, eta=0.0, n_iters=12)
    noisy = D.recover(tw, attractor, Y, 0.1, mask, eta=1e-3, n_iters=12, noise_seed=3)
    assert noisy["rel_err"] > clean["rel_err"]


# ---- 9. observation operators --------------------------------------------- #
def test_obs_masks(tw):
    vr = tw.spec.variables_range
    import numpy as _np
    full = _np.asarray(obs_mask(tw.ndim, vr, "full"))
    atmos = _np.asarray(obs_mask(tw.ndim, vr, "atmos"))
    ocean = _np.asarray(obs_mask(tw.ndim, vr, "ocean"))
    temp = _np.asarray(obs_mask(tw.ndim, vr, "temp"))
    assert full.sum() == tw.ndim
    assert atmos.sum() == vr[1] and atmos[: vr[1]].all() and not atmos[vr[1]:].any()
    assert ocean.sum() == tw.ndim - vr[1] and ocean[vr[1]:].all() and not ocean[: vr[1]].any()
    # temperatures only = theta_a (vr0:vr1) + T_o (vr2:vr3)
    assert temp.sum() == (vr[1] - vr[0]) + (vr[3] - vr[2])
    assert temp[vr[0]:vr[1]].all() and temp[vr[2]:vr[3]].all()
    assert not temp[: vr[0]].any() and not temp[vr[1]:vr[2]].any()


# ---- 10. noiseless recovery works from a partial (atmosphere-only) view ----- #
def test_partial_obs_recovery_noiseless(tw, attractor):
    # the error couples ocean temperature into the atmosphere; noiselessly it is
    # recoverable from atmosphere-only observations via the dynamics.
    Y = np.asarray(D.rollout(lambda x: tw.f_true(x), attractor, 0.1, 1280))
    mask = obs_mask(tw.ndim, tw.spec.variables_range, "atmos")
    r = D.recover(tw, attractor, Y, 0.1, mask, eta=0.0, n_iters=12)
    assert r["rel_err"] < 0.05, r
    assert abs(r["c_hat"][2] - 0.1) < 0.02


# ---- 11. unknown-form experiments (E1-E5) ---------------------------------- #
from error_discovery import experiment_form as Fm


def test_form_fit_recovers_truth_on_excited_data():
    tw = Fm.twin()
    X = Fm.true_transient(tw, 200, seed=1)[:-1]
    m = Fm.err_tendency(tw, X)
    c, nm = Fm._fit(tw, X, m)
    R = Fm._resid(tw, c, X, m)
    assert nm[int(np.argmin(R))] == Fm.true_term(tw)
    assert abs(c[nm.index(Fm.true_term(tw))] - 0.1) < 0.02


def test_forcing_recovers_tendency_and_form():
    tw = Fm.twin()
    Xt = Fm.true_transient(tw, 201, seed=2)
    m = Fm.err_tendency(tw, Xt[:-1])
    dhat = Fm.forcing_recover(tw, Xt)
    assert np.linalg.norm(dhat - m) / np.linalg.norm(m) < 0.3          # E5
    c, nm = Fm._fit(tw, Xt[:-1], dhat)
    assert nm[int(np.argmin(Fm._resid(tw, c, Xt[:-1], dhat)))] == Fm.true_term(tw)


def test_naive_fd_does_not_recover_form():
    """E3 contrast: the tiny error is below finite-difference truncation."""
    tw = Fm.twin()
    Xt = Fm.true_transient(tw, 201, seed=3)
    c, nm = Fm._fit(tw, Xt[:-1], Fm.fd_tendency(tw, Xt))
    assert nm[int(np.argmin(Fm._resid(tw, c, Xt[:-1], Fm.fd_tendency(tw, Xt))))] != Fm.true_term(tw)


def test_excitation_restores_identifiability():
    """E1/E2: attractor-only is ambiguous; a transient test set is decisive."""
    d = Fm.e1(noises=(0.0,), reps=3)
    assert np.median(d["insample"][0]) < 0.1
    assert np.median(d["generalise"][0]) > 0.5


def test_reanalysis_climatology_not_far_enough():
    """Sampling the true climatology (even nudged) doesn't identify the form;
    only off-attractor (transient) states do."""
    r = Fm.reanalysis(eps_list=(0.1,), taus=(20.0,), n=1500, n_spin=10000)
    assert r["biased"][0] < 0.1 and r["rows"][0]["true"][0] < 0.1
    assert r["transient"][0] > 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
