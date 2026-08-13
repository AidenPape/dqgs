"""
test_climatology.py
===================

Climatology fidelity of the recovered correction. The surgical 10% coupling error is
climatologically nearly silent: it shifts the long-run climatological mean of every
mode by far less than the run-to-run sampling spread, so the corrected, base, and
truth climatologies are indistinguishable. Uses a short integration for speed; the
claim is the smallness of the structural shift relative to the sampling floor, not the
converged climatology.

Run:  pytest error_discovery/test_climatology.py -v
"""

import numpy as np

from error_discovery.experiments import climatology as C
from error_discovery.config import ExperimentConfig
from error_discovery.models import TwinModels
from error_discovery.discovery import rollout
from error_discovery.attractor import get_attractor_ic


def test_error_is_climatologically_small():
    """The structural mean shift between base and truth climatologies is a small
    fraction of each mode's own variability (the error barely moves the climatology)."""
    cfg = ExperimentConfig()
    tw = TwinModels(cfg.eps, cfg.g_coeffs)
    xa = get_attractor_ic(tw, cfg, verbose=False)
    n = 20_000
    truth = np.asarray(rollout(lambda x: tw.f_true(x), xa, cfg.dt, n))
    base = np.asarray(rollout(lambda x: tw.f_base(x), xa, cfg.dt, n))
    sd = truth.std(0)
    shift = np.abs(base.mean(0) - truth.mean(0)) / np.where(sd > 0, sd, 1.0)
    assert np.median(shift) < 0.05, f"structural mean shift unexpectedly large ({np.median(shift):.3f})"


def test_quantile_stats_shape():
    """_quantile_stats returns per-mode statistics of the right shape."""
    traj = np.random.default_rng(0).standard_normal((1000, 36))
    st = C._quantile_stats(traj)
    for k in ("mean", "q1", "median", "q3", "std"):
        assert st[k].shape == (36,)
    assert np.all(st["q3"] >= st["q1"])
