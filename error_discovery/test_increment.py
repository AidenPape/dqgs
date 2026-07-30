"""
test_increment.py
=================

Focused tests for the DA-increment vs climatology form-discovery experiment
(`experiment_increment.py`). The experiment re-uses the validated form-margin of
`experiment_form`; these tests check the increment-specific pieces.

Coverage:
  1. da_forecasts(): launches the BIASED model from TRUE states (so segment
     starts equal the true trajectory's analysis times, and the dynamics are the
     biased model's, not the true model's).
  2. _margin_exc(): returns (margin, slow std, fast std) consistent with the
     underlying experiment_form margin, on a real distribution.
  3. The headline ordering holds: an off-attractor transient excites the slow
     predictor T_o1 far more than a (biased) climatology, and clears the
     identifiability threshold while the climatology does not.

Run:  pytest error_discovery/test_increment.py -v
"""

import numpy as np
import jax.numpy as jnp
import pytest

from error_discovery import experiment_increment as E
from error_discovery.discovery import rollout


@pytest.fixture(scope="module")
def tw():
    return E.twin()


@pytest.fixture(scope="module")
def xa(tw):
    return E._attractor_ic(tw)


# ---- 1. DA forecasts use biased dynamics launched from true states -------- #
def test_da_forecasts_launch_from_true_with_biased_dynamics(tw, xa):
    window, n, spin = 25, 200, 2000
    Xtrue, _ = E.true_climatology(tw, xa, n, spin)   # deterministic == da_forecasts' internal copy
    X = E.da_forecasts(tw, xa, n, window=window, spin=spin)
    nseg = n // window
    assert X.shape == (nseg * window, tw.ndim)
    # segment k is a BIASED rollout launched from the true analysis state Xtrue[k*window]
    for k in range(nseg):
        seg = np.asarray(rollout(lambda z: tw.f_base(z),
                                 jnp.asarray(Xtrue[k * window]), E.DT, window))
        assert np.allclose(X[k * window:(k + 1) * window], seg, atol=1e-10)
    # the biased forecast genuinely diverges from the true model over the window
    seg0_true = np.asarray(rollout(lambda z: tw.f_true(z), jnp.asarray(Xtrue[0]), E.DT, window))
    assert not np.allclose(X[window - 1], seg0_true[-1], atol=1e-8)


# ---- 2. _margin_exc returns the form-margin + slow/fast excitation -------- #
def test_margin_exc_matches_form_margin(tw, xa):
    from error_discovery import experiment_form as F
    X = np.asarray(rollout(lambda z: tw.f_base(z), xa, E.DT, 400))
    mg, slo, fas = E._margin_exc(tw, X)
    mg_ref, slo_ref = F._margin_of(tw, X)
    assert np.isclose(mg, mg_ref, rtol=1e-6, atol=1e-9)   # same validated margin
    assert np.isclose(slo, slo_ref, rtol=1e-6)            # slow = T_o1 std
    assert np.isclose(fas, float(X[:, 0].std()), rtol=1e-6)  # fast = psi_a1 std


# ---- 3. excitation orders the distributions as claimed -------------------- #
def test_transient_excites_slow_predictor_more_than_climatology(tw, xa):
    Xclim = np.asarray(rollout(lambda z: tw.f_base(z), xa, E.DT, 1500))
    Xtran = np.asarray(rollout(lambda z: tw.f_base(z),
                               jnp.asarray(np.random.default_rng(0)
                                           .standard_normal(tw.ndim) * 0.5), E.DT, 1500))
    mg_c, slo_c, _ = E._margin_exc(tw, Xclim)
    mg_t, slo_t, _ = E._margin_exc(tw, Xtran)
    assert slo_t > 10 * slo_c          # transient swings T_o1 through its range
    assert mg_c < 0.1                  # climatology: form NOT identifiable
    assert mg_t > mg_c                 # transient strictly more identifiable
