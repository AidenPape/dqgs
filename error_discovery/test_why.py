"""Tests for experiment_why: the mechanism behind the transient-only asymmetry.

Encodes the three claims:
  1. form-identifiability margin is governed by the driver's own |mean|/std
     (collapses onto ~1/r; attractor unidentifiable, off-attractor identifiable);
  2. the recovered form is global: fit off-attractor it predicts the attractor error,
     while the best on-attractor alias fails off-attractor;
  3. predictability (1/|lambda|) greatly exceeds the days-to-weeks discovery window.
Small reps/sizes keep the suite fast; uses qgs_jax rollouts (single process)."""

import numpy as np
from error_discovery import experiment_why as W


def test_margin_governed_by_driver_meanstd():
    d = W.governing_law(n=800, reps=4, seed=0, save=False)
    r, mg = np.median(d["r"], 1), np.median(d["margin"], 1)
    # attractor end (frac 0): driver near-constant, form unidentifiable
    assert r[0] > 10.0 and mg[0] < 0.1
    # off-attractor end (frac 1): driver excited, form identifiable
    assert r[-1] < 2.0 and mg[-1] > 0.5
    # the margin rises substantially as the driver's |mean|/std falls off-attractor
    assert mg[-1] > 5 * mg[0]
    # collapse onto 1/r in the obstructed regime: attractor margin ~ 1/r within ~3x
    assert 1.0 / 3 < mg[0] * r[0] < 3.0


def test_form_is_global_alias_is_local():
    s = W.global_form_validation(n=800, reps=4, seed=0, save=False)
    true_oos = s["true_recOff_testAtt"][0]     # off-attractor-recovered model -> attractor
    true_is = s["true_recOff_testOff"][0]      # ...-> held-out off-attractor
    alias_is = s["alias_fitAtt_testAtt"][0]    # alias matches where fit
    alias_oos = s["alias_fitAtt_testOff"][0]   # alias fails off-attractor
    # the off-attractor-recovered model carries a small, real error (not machine zero)...
    assert 1e-4 < true_oos < 0.05
    # ...and the SAME error in both regimes, because the true form's coefficient is global
    assert abs(true_oos - true_is) < 0.02
    assert alias_is < 0.1                       # the alias matches on the attractor (it is an alias)
    assert alias_oos > 0.5                      # but is ~100% wrong out of its regime, unlike the truth


def test_predictability_exceeds_discovery_window():
    d = W.lyapunov_ics(n_ic=2, n_blocks=200, drop=40, save=False)
    # predictability time (~1/|lambda|) is far longer than the ~22-day discovery window
    assert d["predictability_days"] > 40.0
    assert abs(d["lam_mean"]) < 5e-3
