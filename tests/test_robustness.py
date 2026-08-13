"""
test_robustness.py
==================

Tests for the robustness experiments (`experiment_robustness.py`) that stress the
mean/std identifiability argument: the screening relation is approximate, and a
complexity prior cannot rescue an offset-dominated driver because the constant
alias caps the margin at 1/(driver's |mean|/std).

Run:  pytest error_discovery/test_robustness.py -v
"""

import numpy as np
import pytest

from error_discovery.experiments import robustness as R
from error_discovery.formfit import twin, attractor_traj, EPS
from error_discovery.screening import max_mean_over_std


@pytest.fixture(scope="module")
def floor():
    return R.parsimony_floor(n=4000)


def test_const_margin_equals_inverse_driver_meanstd(floor):
    """The truth's margin over the constant alias is about 1/(the driver's own
    |mean|/std): the offset-dominated drivers are aliased by the constant."""
    ms, cm, named = floor["meanstd"], floor["const_margin"], floor["named"]
    for nm, k in named.items():
        assert np.isclose(cm[k], 1.0 / ms[k], rtol=0.1), f"{nm}: {cm[k]:.4f} vs {1/ms[k]:.4f}"


def test_offset_drivers_are_parsimony_proof(floor):
    """An offset-dominated driver (|mean|/std > 10) stays below the 0.1 threshold
    against the constant, so no complexity prior can make it identifiable."""
    ms, cm = floor["meanstd"], floor["const_margin"]
    offset = ms > 10.0
    assert offset.sum() >= 8
    assert np.all(cm[offset] < 0.1)
    # and several low-|mean|/std drivers clear it against the constant
    assert (cm[ms < 1.0] > 0.1).mean() > 0.8


def test_raw_margin_is_a_flat_cross_product_floor(floor):
    """The raw best alias is the product of the driver with the single most
    near-constant mode, so the raw margin is a flat floor near 1/max(|mean|/std over
    modes), the SAME for every driver (the full library aliases all 36), and far below
    the constant ceiling. A complexity prior lifts the margin from this floor toward
    the constant ceiling, but no further."""
    raw, cm, ms = floor["raw_margin"], floor["const_margin"], floor["meanstd"]
    max_over_modes = float(ms.max())
    assert 0.3 < np.median(raw) * max_over_modes < 3.0          # floor at ~1/max-over-modes
    assert (np.abs(raw - np.median(raw)) < 0.01).mean() > 0.6   # flat across drivers
    offset = ms > 10.0
    assert np.median(raw[offset]) < 0.5 * np.median(cm[offset])  # floor well below ceiling


def test_screening_relation_is_approximate():
    """margin ~ 1/max(|mean|/std) holds only within a factor of a few in the
    obstructed regime, and saturates near 1 when no mode is near-constant."""
    tw = twin()
    Xa = attractor_traj(tw, 4000)
    Xb = np.random.default_rng(1).standard_normal((4000, tw.ndim))
    for X, regime in ((Xa, "obstructed"), (Xb, "excited")):
        G, names, order, EX, sq, den = R.library_matrix(tw, X)
        margins = []
        for k in range(tw.ndim):
            m = EPS * X[:, k][:, None] * EX
            res = R.residuals(G, EX, sq, den, m)
            margins.append(float(np.delete(res, names.index(f"x{k}")).min() - res[names.index(f"x{k}")]))
        med = float(np.median(margins))
        mm = max_mean_over_std(X)
        if regime == "obstructed":
            assert mm > 100.0
            assert 0.25 < med * mm < 4.0          # within a factor of a few of 1, not exact
        else:
            assert mm < 1.0
            assert med > 0.9                       # saturated, no near-constant mode
