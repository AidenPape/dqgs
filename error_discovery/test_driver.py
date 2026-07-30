"""Tests for experiment_driver: which mode drives the error sets on-attractor
identifiability, and the controller is the driver's own |mean|/std (not
ocean-vs-atmosphere as a category).

Claims encoded:
  1. a fast, zero-centred atmospheric driver (psi_a1, |mean|/std ~ 4) is
     identifiable ON the attractor (margin > threshold), no transient needed;
  2. the slow ocean driver (T_o1, |mean|/std ~ 40) is NOT identifiable there;
  3. on-attractor margin falls monotonically as the driver's |mean|/std rises and
     tracks the 1/r prediction within a factor of a few across both fluids.
Small reps/sizes keep the suite fast; uses qgs_jax rollouts (single process)."""

import numpy as np
from error_discovery import experiment_driver as D


def _by_name(rows, key):
    return {r["name"].split("\n")[0]: r[key] for r in rows}


def test_atmospheric_driver_identifiable_on_attractor():
    rows = D.driver_scan(n=2000, reps=4, seed=0, save=False)
    r_att = _by_name(rows, "r_att")
    marg = _by_name(rows, "margin_att")
    # the fast atmospheric streamfunction is near zero-centred and identifiable on-attractor
    assert r_att["psi_a1"] < 10.0
    assert marg["psi_a1"] > 0.1
    # the slow ocean-temperature driver is pinned near-constant and is NOT
    assert r_att["T_o1"] > 20.0
    assert marg["T_o1"] < 0.1


def test_margin_tracks_one_over_meanstd_across_fluids():
    rows = D.driver_scan(n=2000, reps=4, seed=0, save=False)
    finite = [r for r in rows if np.isfinite(r["r_att"])]
    # margin decreases as the driver's |mean|/std increases (psi_a1 > theta_a1 > T_o1)
    order = sorted(finite, key=lambda r: r["r_att"])
    margins = [r["margin_att"] for r in order]
    assert margins == sorted(margins, reverse=True)
    # and each tracks 1/r within a factor of ~3
    for r in finite:
        assert 1.0 / 3 < r["margin_att"] * r["r_att"] < 3.0
    # the constant driver (|mean|/std = inf) is never identifiable
    const = [r for r in rows if not np.isfinite(r["r_att"])][0]
    assert const["margin_att"] < 0.1


def test_atmosphere_opens_up_ocean_stays_pinned_with_record_length():
    d = D.record_length_scan(Ns=(2000, 8000, 20000), save=False)
    m = d["margin"]                       # rows: psi_a1, theta_a1, T_o1
    psi, To = m[0], m[2]
    # atmospheric circulation: margin rises with record length and crosses threshold
    assert psi[-1] > psi[0]
    assert psi[-1] > 0.1
    # ocean temperature: never crosses the threshold at any record length
    assert (To < 0.1).all()
    # and its |mean|/std stays high while the atmosphere's falls
    r = d["meanstd"]
    assert (r[2] > 15).all()              # ocean temperature pinned at every length
    assert r[0][-1] < 10                  # atmospheric circulation excited by the long record


def test_full_error_circulation_recovered_temperature_conditioning():
    d = D.full_error_decomposition(noises=(0.01, 0.1), n=4000, reps=6, seed=0,
                                   record=20000, save=False)
    chat = d["chat"]
    # the circulation coefficient is recovered near eps with small spread
    assert abs(np.median(chat["psi_a1"][0]) - d["eps"]) < 0.03
    psi_spread = chat["psi_a1"].std(1)
    To_spread = chat["T_o1"].std(1)
    # the offset-dominated ocean-temperature coefficient is the less determined of the two
    assert (To_spread > psi_spread).all()
    # and its spread grows with noise (conditioning bottleneck)
    assert To_spread[-1] > To_spread[0]


def test_alias_collinear_on_attractor_breaks_under_excitation():
    d = D.alias_data(n=2000, seed=0)
    # a near-constant mode times a driver is collinear with the driver on the attractor
    assert d["att"]["m_meanstd"] > 10           # the multiplier is near-constant
    assert d["att"]["corr"] > 0.95              # alias indistinguishable from the driver
    # under broad excitation the multiplier varies and the alias breaks
    assert abs(d["broad"]["corr"]) < 0.2
