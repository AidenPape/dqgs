"""Tests for the sampling standard (sampling.py, attractor.segments_from_states).

Each numeric piece is checked against a case with a known answer, and the base-run I/O
against the integrator it replaces: a resume that drifts or a chunk that does not continue its
predecessor would silently corrupt every starting state drawn from the record."""

import os
import shutil

import numpy as np
import jax.numpy as jnp
import pytest

from error_discovery import sampling as sm
from error_discovery.config import DT, SAMPLING_SOURCE, baserun_path, regimes_path


# --- names ------------------------------------------------------------------ #
def test_file_names_follow_the_convention():
    assert os.path.basename(baserun_path("default", 1_000_000_000, 0, 1000)) == \
        "baserun_default_sp1000000000_yr00000-01000.npz"
    assert os.path.basename(regimes_path("default", 1_000_000_000, 0, 10_000)) == \
        "regimes_default_sp1000000000_yr00000-10000.npz"


# --- pure numerics ---------------------------------------------------------- #
def test_running_std_matches_brute_force():
    x = np.random.default_rng(0).standard_normal(500).cumsum() + 7.0
    w = 31
    got = sm.running_std(x, w)
    h = w // 2
    ref = np.array([x[i - h:i + h + 1].std() for i in range(h, len(x) - h)])
    assert np.all(np.isnan(got[:h])) and np.all(np.isnan(got[-h:]))
    assert np.max(np.abs(got[h:-h] - ref)) < 1e-10


def test_highpass_std_recovers_white_noise_under_a_slow_sine():
    rng = np.random.default_rng(1)
    t = np.arange(200_000)
    slow = 5.0 * np.sin(2 * np.pi * t / 20_000)          # period >> the 31-sample window
    both = np.column_stack([slow + 0.3 * rng.standard_normal(len(t)), slow])
    hf = sm.highpass_std(both, 31)
    assert abs(hf[0] - 0.3 * np.sqrt(30 / 31)) < 0.005     # the window mean takes 1/31 of it
    assert hf[1] < 1e-3


def test_tau_int_of_ar1_within_its_error_bar():
    phi, n = 0.9, 400_000
    rng = np.random.default_rng(2)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    tau, err, M, found = sm.tau_int(x)
    exact = (1 + phi) / (2 * (1 - phi))                   # 1/2 + sum_t phi^t = 9.5
    assert found
    assert M >= sm.TAU_C * tau
    assert abs(tau - exact) < 3 * err


def test_mixture_recovers_two_known_components():
    rng = np.random.default_rng(3)
    v = np.concatenate([rng.normal(-2.0, 0.3, 30_000), rng.normal(1.0, 0.5, 70_000), [np.nan]])
    means, stds, weights, model = sm.fit_mixture(v)
    assert np.allclose(means, [-2.0, 1.0], atol=0.02)
    assert np.allclose(stds, [0.3, 0.5], atol=0.02)
    assert np.allclose(weights, [0.3, 0.7], atol=0.01)
    p = sm.mixture_prob_high(model, np.array([-2.0, 1.0, np.nan]))
    assert p[0] < 1e-6 and p[1] > 1 - 1e-6 and np.isnan(p[2])
    # the 50/50 point of the fitted components, found by bisection, matches the closed form
    lo, hi = -2.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if sm.mixture_prob_high(model, np.array([mid]))[0] < 0.5 else (lo, mid)
    def logpdf(x, m, s, w):
        return np.log(w) - np.log(s) - 0.5 * ((x - m) / s) ** 2
    assert abs(logpdf(lo, means[0], stds[0], weights[0])
               - logpdf(lo, means[1], stds[1], weights[1])) < 1e-6


def test_labels_episodes_and_onsets():
    p = np.array([0.95] * 10 + [0.5] * 3 + [0.02] * 2 + [0.5] * 2 + [0.96] * 5 + [0.03] * 8
                 + [np.nan] * 2 + [0.01] * 4 + [0.97] * 6)
    lab = sm.regime_labels(p, 0.9)
    assert lab[0] == 1 and lab[10] == 0 and lab[13] == -1 and lab[30] == 0
    # the 2-long passive run is a flicker at min_len 3: the active runs either side merge
    assert sm.episodes(lab, 3) == [(1, 0, 22), (-1, 22, 36), (1, 36, 42)]
    assert sm.onsets(lab, -1, 3) == [22]
    assert sm.onsets(lab, 1, 3) == [36]          # the first episode has no onset
    # at min_len 2 the flicker is an episode of its own
    assert sm.onsets(lab, -1, 2) == [13, 22]


def test_pick_spaced_is_greedy_spaced_and_one_per_group():
    assert sm.pick_spaced([0, 3, 5, 9, 10, 20], 5) == [0, 5, 10, 20]
    group = {0: 0, 5: 0, 10: 1, 20: 1, 30: 2}
    assert sm.pick_spaced([0, 5, 10, 20, 30], 5, group) == [0, 10, 30]


def test_odd_window_and_chunk_size():
    assert sm.odd_window(30) % 2 == 1
    assert abs(sm.odd_window(30) * sm.snapshot_days() - 30) <= sm.snapshot_days()
    yrs = sm.snapshots_per_chunk() * sm.snapshot_days() / 365.25
    assert abs(yrs - 1000) < 0.01


# --- the base run ----------------------------------------------------------- #
CH = 1                                   # one-year chunks keep the run to ~65k steps


@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("base"))
    x = sm.base_run(n_years=2 * CH, chunk_years=CH, directory=d, verbose=False)
    return d, np.asarray(x)


def _chunk(d, c):
    cfg, sp = SAMPLING_SOURCE
    return np.load(baserun_path(cfg, sp, c * CH, (c + 1) * CH, d))


def test_base_run_hops_match_integrate_final(straight):
    """Each snapshot is SNAPSHOT_EVERY RK4 steps after the last, chunks join exactly, and the
    record starts at the spun-up state."""
    from qgs_jax import model as jm
    from qgs_jax.spec import extract_spec, build_reference_params
    from error_discovery.attractor import load_spun_up_ic
    d, x_end = straight
    T = jnp.asarray(extract_spec(build_reference_params("default")).dense_tensor)
    a, b = _chunk(d, 0), _chunk(d, 1)
    assert np.array_equal(a["snapshots"][0], np.asarray(load_spun_up_ic(*SAMPLING_SOURCE)))
    assert np.array_equal(a["x_next"], b["snapshots"][0])
    assert np.array_equal(b["x_next"], x_end)
    for snaps, j in ((a["snapshots"], 0), (a["snapshots"], 200), (b["snapshots"], 300)):
        ref = np.asarray(jm.integrate_final(T, jnp.asarray(snaps[j]), DT, sm.SNAPSHOT_EVERY))
        assert np.max(np.abs(snaps[j + 1] - ref)) < 1e-13
    last = np.asarray(jm.integrate_final(T, jnp.asarray(a["snapshots"][-1]), DT,
                                         sm.SNAPSHOT_EVERY))
    assert np.max(np.abs(a["x_next"] - last)) < 1e-13


def test_base_run_resumes_bit_identically_and_never_rewrites(straight, tmp_path):
    d, x_end = straight
    cfg, sp = SAMPLING_SOURCE
    first = baserun_path(cfg, sp, 0, CH, str(tmp_path))
    shutil.copy(baserun_path(cfg, sp, 0, CH, d), first)
    mtime = os.path.getmtime(first)
    x = sm.base_run(n_years=2 * CH, chunk_years=CH, directory=str(tmp_path), verbose=False)
    assert os.path.getmtime(first) == mtime
    assert np.array_equal(np.asarray(x), x_end)
    assert np.array_equal(np.load(baserun_path(cfg, sp, CH, 2 * CH, str(tmp_path)))["snapshots"],
                          _chunk(d, 1)["snapshots"])
    rec = sm.load_base_run(2 * CH, chunk_years=CH, directory=str(tmp_path))
    assert rec.shape == (2 * sm.snapshots_per_chunk(CH), 36)
    idx = [5, sm.snapshots_per_chunk(CH) + 7]
    assert np.array_equal(sm.load_snapshots(idx, chunk_years=CH, directory=str(tmp_path)),
                          rec[idx])


def test_base_run_refuses_a_chunk_from_another_recipe(straight, tmp_path):
    d, _ = straight
    cfg, sp = SAMPLING_SOURCE
    a = dict(_chunk(d, 0))
    a["snapshot_every"] = 90
    np.savez(baserun_path(cfg, sp, 0, CH, str(tmp_path)), **a)
    with pytest.raises(ValueError, match="snapshot_every"):
        sm.base_run(n_years=CH, chunk_years=CH, directory=str(tmp_path), verbose=False)


# --- segments from given states --------------------------------------------- #
def test_segments_from_states_is_an_f_true_rollout():
    from error_discovery.models import TwinModels
    from error_discovery.attractor import segments_from_states, load_spun_up_ic
    from error_discovery.discovery import rollout
    tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0))
    x0 = load_spun_up_ic(*SAMPLING_SOURCE)
    ics, trajs = segments_from_states(tw, [x0], 50, DT)
    ref = np.asarray(rollout(lambda y: tw.f_true(y), x0, DT, 50))
    assert np.array_equal(np.asarray(ics[0]), np.asarray(x0))
    assert np.array_equal(trajs[0], ref)
    ics, trajs = segments_from_states(tw, [x0], 50, DT, discard=30)
    ref = np.asarray(rollout(lambda y: tw.f_true(y), x0, DT, 80))
    assert np.array_equal(np.asarray(ics[0]), ref[29])
    assert np.max(np.abs(trajs[0] - ref[30:])) < 1e-13


# --- the cross-check diagnostic --------------------------------------------- #
def test_geopotential_difference_weights_at_n_15():
    """At x = pi/n the sin(n x) modes vanish and the y-symmetric ones cancel, leaving
    2 psi_a1 - 4 psi_a5 + 4 psi_a9; the qgs class itself would evaluate the basis at n = 1.3."""
    from error_discovery.experiments.sampling_checks import geopotential_difference_weights
    w, metres = geopotential_difference_weights()
    assert np.allclose(w, [2, 0, 0, 0, -4, 0, 0, 0, 4, 0], atol=1e-12)
    assert abs(metres - 2749.987) < 0.01


# --- noise ------------------------------------------------------------------ #
def test_shared_noise_is_common_across_lengths():
    a = sm.shared_noise(123, 1, 100, 36)
    b = sm.shared_noise(123, 1, 40, 36)
    assert np.array_equal(a[:40], b)
    assert not np.array_equal(a, sm.shared_noise(123, 2, 100, 36))
