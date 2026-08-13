"""
error_discovery.attractor
=========================

The spun-up attractor state every study starts from, and the truth segments the
lead-time sweep slices out of it.

The spin-up is the most expensive step in the project (800k steps), so the result
is cached to ``results/attractor_ic.npz`` along with the recipe that produced it.
Every figure's comparability rests on that one initial condition: **do not clobber
the cache, and do not pass a non-default spin-up.**
"""

import os
import time

import numpy as np
import jax.numpy as jnp

from .config import RESULTS_DIR, IC_CACHE, IC_SEED, IC_SCALE
from .discovery import rollout


def _cache_matches(d, tw, cfg):
    """True if a cached IC was built with the current spin-up recipe."""
    try:
        return (int(d["ndim"]) == tw.ndim and int(d["spinup_steps"]) == cfg.spinup_steps
                and int(d["ic_seed"]) == IC_SEED and float(d["dt"]) == cfg.dt)
    except KeyError:
        return False


def get_attractor_ic(tw, cfg, verbose=True):
    """Load the cached attractor IC (if it matches the recipe), else spin one up.

    The cache stores its provenance (ndim, spinup_steps, seed, dt) so a stale or
    mismatched IC is rebuilt rather than silently reused.
    """
    if os.path.exists(IC_CACHE):
        d = np.load(IC_CACHE)
        if _cache_matches(d, tw, cfg):
            if verbose:
                print(f"[ic]    loaded cached attractor IC ({IC_CACHE})")
            return jnp.asarray(d["x"])
        if verbose:
            print("[ic]    cached IC does not match current recipe; rebuilding")
    from qgs_jax import model as jm
    rng = np.random.default_rng(IC_SEED)
    x0 = jnp.asarray(rng.standard_normal(tw.ndim) * IC_SCALE)
    t = time.time()
    xa = np.asarray(jm.integrate_final(tw.T_base, x0, cfg.dt, cfg.spinup_steps))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(IC_CACHE, x=xa, ndim=tw.ndim, spinup_steps=cfg.spinup_steps,
             ic_seed=IC_SEED, dt=cfg.dt)
    if verbose:
        print(f"[ic]    spun up {cfg.spinup_steps} steps & cached "
              f"({time.time() - t:.0f}s, |x|={np.linalg.norm(xa):.3f})")
    return jnp.asarray(xa)


def build_segments(tw, cfg, x_attr):
    """Return (segment_ics, truth_trajs) with each truth_traj of length max_window."""
    max_w = max(cfg.window_steps)
    total = (cfg.n_ic - 1) * cfg.ic_spacing_steps + max_w
    truth = np.asarray(rollout(lambda y: tw.f_true(y), x_attr, cfg.dt, total))
    ics, trajs = [], []
    for k in range(cfg.n_ic):
        s = k * cfg.ic_spacing_steps
        ics.append(x_attr if s == 0 else jnp.asarray(truth[s - 1]))
        trajs.append(truth[s:s + max_w])
    return ics, trajs
