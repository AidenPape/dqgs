"""
test_meanstd.py
===============

Tests for the corrected form-identifiability result (`experiment_meanstd.py`):
form discovery is library-dependent, and against a complete quadratic library a
single near-constant mode aliases every error form, so the controller is the
maximum |mean|/std over all modes (not the driver's own). These pin the verified
result, including the assertion whose failure would falsify it (full-state
excitation, where no mode is near-constant, must restore identifiability).

Run:  pytest error_discovery/test_meanstd.py -v
"""

import numpy as np
import pytest

from error_discovery.experiments import obstruction as M
from error_discovery.formfit import twin, attractor_traj


@pytest.fixture(scope="module")
def setup():
    tw = twin()
    X = attractor_traj(tw, 4000)                                   # biased climatology
    Xbroad = np.random.default_rng(1).standard_normal((4000, tw.ndim))  # full-state excitation
    return tw, X, Xbroad


def test_truth_form_exactly_represented(setup):
    """The true driver g=x[k] is in the library, so its own residual is ~0 and the
    margin is the best wrong form's residual."""
    tw, X, _ = setup
    for k in (tw.idx_psia1, tw.idx_thetaa1, tw.idx_To1, 21):
        assert M.margin_driver(tw, X, k, "full")[1] < 1e-6


def test_near_constant_mode_obstructs_full_library(setup):
    """On the attractor a restricted (no cross-term) library makes many drivers look
    identifiable, but the full quadratic library identifies none: a near-constant
    mode times any driver aliases that driver."""
    tw, X, _ = setup
    assert M.max_mean_over_std(X) > 100.0
    nr, _ = M.identifiable_count(tw, X, "restricted")
    nf, _ = M.identifiable_count(tw, X, "full")
    assert nr >= 10, f"restricted library identified only {nr}"
    assert nf == 0, f"full library unexpectedly identified {nf}"


def test_alias_residual_tracks_inverse_max_meanstd(setup):
    """The mechanism: best wrong-form residual under the full library is about
    1 / max|mean|/std (the most near-constant mode sets the strongest alias)."""
    tw, X, _ = setup
    _, margins = M.identifiable_count(tw, X, "full")
    assert np.isclose(np.median(margins), 1.0 / M.max_mean_over_std(X), rtol=0.5)


def test_full_state_excitation_restores_identifiability(setup):
    """The falsifiable claim: on a distribution where every mode crosses zero
    (no near-constant mode, max|mean|/std ~ O(1)), the full library identifies
    nearly every form. If this failed, the near-constant-mode mechanism would be wrong."""
    tw, _, Xbroad = setup
    assert M.max_mean_over_std(Xbroad) < 1.0
    nf, _ = M.identifiable_count(tw, Xbroad, "full")
    assert nf >= tw.ndim - 2, f"full-state excitation identified only {nf}/{tw.ndim}"


def test_scaling_the_error_does_not_help(setup):
    """Turning the injected error up or down does not restore identifiability: the
    attractor's max|mean|/std is eps-invariant (the error self-limits), so the form
    stays unidentifiable under both libraries across a wide eps range. Short spin-up
    here for speed; the claim is the invariance, not the absolute value."""
    rows = M.eps_sweep(eps_list=(0.1, 1.0, 10.0), n=1500, n_spin=4000)
    mm = np.array([r["max_meanstd"] for r in rows])
    assert mm.min() > 50.0                                  # a strongly near-constant mode persists
    assert (mm.max() - mm.min()) / mm.mean() < 0.05         # eps-invariant attractor geometry
    assert all(r["margin_full"] < 0.1 for r in rows)        # unidentifiable at every eps
    assert all(r["margin_restricted"] < 0.1 for r in rows)
