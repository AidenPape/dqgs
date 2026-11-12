"""
error_discovery.attractor
=========================

The spun-up attractor state every study starts from, the truth segments the
lead-time sweep slices out of it, and the long spin-up.

The 800k-step spin-up is cached to ``initial_conditions/attractor_ic_default_sp800000.npz`` along
with the recipe that produced it. Every figure's comparability rests on that one initial
condition: **do not clobber the cache, and do not pass a non-default spin-up.**

``spin_up`` builds a longer state of any named parameter set (1e9 steps by default), with
checkpoints, and ``load_spun_up_ic`` reads it back after checking its recipe. File names
follow ``config.ic_path``; the convention is in ``error_discovery/README.md``.
"""

import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from .config import (IC_DIR, IC_CACHE, IC_SEED, IC_SCALE, DT, STEPS_PER_DAY,
                     DAYS_PER_YEAR, ic_path)
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
    os.makedirs(IC_DIR, exist_ok=True)
    np.savez(IC_CACHE, x=xa, ndim=tw.ndim, spinup_steps=cfg.spinup_steps,
             ic_seed=IC_SEED, dt=cfg.dt)
    if verbose:
        print(f"[ic]    spun up {cfg.spinup_steps} steps & cached "
              f"({time.time() - t:.0f}s, |x|={np.linalg.norm(xa):.3f})")
    return jnp.asarray(xa)


def build_segments(tw, cfg, x_attr):
    """Return (segment_ics, truth_trajs) with each truth_traj of length max_window.

    Segments are spaced by twice the longest window, so no two share a step at any window in
    the sweep and one full window of trajectory is discarded between them. The spacing is
    *derived* from ``max(window_steps)`` rather than configured: a fixed spacing silently
    overlaps once the sweep grows past it, which is what had happened -- adjacent segments
    shared 22% at 3840 steps, 41% at 5120 and 62.5% at 8000.

    Disjoint is not independent. Consecutive segments are separated by one longest window on a
    single trajectory, and MAOOAM's ocean modes evolve on timescales far longer than that, so
    the ocean state stays correlated across them. Read the segment-to-segment spread as a lower
    bound on sampling variability, not as an i.i.d. sample.
    """
    max_w = max(cfg.window_steps)
    spacing = 2 * max_w
    total = (cfg.n_ic - 1) * spacing + max_w
    truth = np.asarray(rollout(lambda y: tw.f_true(y), x_attr, cfg.dt, total))
    ics, trajs = [], []
    for k in range(cfg.n_ic):
        s = k * spacing
        ics.append(x_attr if s == 0 else jnp.asarray(truth[s - 1]))
        trajs.append(truth[s:s + max_w])
    return ics, trajs


def segments_from_states(tw, states, n_steps, dt, discard=0):
    """Return (segment_ics, truth_trajs), one segment per given starting state.

    The sampling standard's counterpart of ``build_segments`` (``plans/sampling.md``): each
    state comes from the base run, so ``f_true`` runs ``discard`` steps from it before the
    segment begins (default 0: the window starts at the switch), then ``n_steps`` more. The
    convention is ``build_segments``'s: ``truth_traj[0]`` is one step after the segment IC.
    """
    ics, trajs = [], []
    for x in states:
        x = jnp.asarray(x)
        if discard:
            x = rollout(lambda y: tw.f_true(y), x, dt, discard)[-1]
        ics.append(x)
        trajs.append(np.asarray(rollout(lambda y: tw.f_true(y), x, dt, n_steps)))
    return ics, trajs


def _recipe(config, n_steps):
    """What fixes a long spun-up state: stored beside it by ``spin_up``, checked on load."""
    from qgs_jax.spec import PARAM_SETS
    return dict(config=config, spinup_steps=int(n_steps), ic_seed=IC_SEED, ic_scale=IC_SCALE,
                dt=DT, params=json.dumps(PARAM_SETS[config], sort_keys=True))


def _mismatched(d, recipe):
    """Recipe keys that a loaded ``.npz`` lacks or disagrees on."""
    return [k for k, v in recipe.items() if k not in d.files or d[k].item() != v]


def spin_up(config="default", n_steps=1_000_000_000, trail_every=1_000_000,
            checkpoint_every=10_000_000, directory=IC_DIR, verbose=True):
    """Spin a named parameter set's base model up from the ``IC_SEED`` x ``IC_SCALE`` recipe.

    Writes ``ic_path(config, n_steps, directory)`` holding the final state ``x``, the
    ``trail`` of states after every ``trail_every`` steps, and the recipe (``_recipe``).
    Nothing is computed from the trail here; it is kept so the approach to the attractor can
    be examined later without repeating the run.

    The tendency tensor is closed over, so it is a compile-time constant: 61,700 steps/s on
    the laptop against 6,100 through ``integrate_final``, which takes it as an argument
    (measured 2026-09-24). Same model and same RK4 step; the two agree to 1.7e-15 over 1,000
    steps.

    Every ``checkpoint_every`` steps the run so far is saved to ``<name>.partial.npz``. A call
    that finds a partial with the same recipe resumes from it and ends bit-identical to an
    uninterrupted run. A finished state is never overwritten.
    """
    from qgs_jax import model as jm
    from qgs_jax.spec import extract_spec, build_reference_params

    if checkpoint_every % trail_every or n_steps % checkpoint_every:
        raise ValueError("trail_every must divide checkpoint_every, which must divide n_steps")
    path = ic_path(config, n_steps, directory)
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; spin_up never overwrites a finished state")
    partial = path[:-len(".npz")] + ".partial.npz"
    tmp = path[:-len(".npz")] + ".tmp.npz"

    T = jnp.asarray(extract_spec(build_reference_params(config)).dense_tensor)
    ndim = T.shape[0] - 1
    recipe = _recipe(config, n_steps)

    def f(x):
        return jm.tendencies(T, x)

    @jax.jit
    def advance(x):
        return jax.lax.scan(lambda y, _: (jm.rk4_step(f, y, DT), None), x, None,
                            length=trail_every)[0]

    if os.path.exists(partial):
        d = np.load(partial)
        bad = _mismatched(d, recipe) + ([] if int(d["trail_every"]) == trail_every
                                        else ["trail_every"])
        if bad:
            raise ValueError(f"{partial} was built with a different recipe: {bad}")
        x, done, trail = jnp.asarray(d["x"]), int(d["steps_done"]), list(d["trail"])
        if verbose:
            print(f"[spin]  {config}: resuming at {done:,} steps", flush=True)
    else:
        x = jnp.asarray(np.random.default_rng(IC_SEED).standard_normal(ndim) * IC_SCALE)
        done, trail = 0, []

    os.makedirs(directory, exist_ok=True)
    t0, done0 = time.time(), done
    while done < n_steps:
        x = advance(x)
        done += trail_every
        trail.append(np.asarray(x))
        if done % checkpoint_every == 0:
            np.savez(tmp, x=np.asarray(x), steps_done=done, trail=np.asarray(trail),
                     trail_every=trail_every, ndim=ndim, **recipe)
            os.replace(tmp, partial)              # atomic: a kill never leaves half a file
            if verbose:
                rate = (done - done0) / (time.time() - t0)
                print(f"[spin]  {config}: {done:,} / {n_steps:,} steps "
                      f"({done / STEPS_PER_DAY / DAYS_PER_YEAR:,.0f} yr)  {rate:,.0f} steps/s",
                      flush=True)
    os.replace(partial, path)
    if verbose:
        print(f"[spin]  wrote {path}", flush=True)
    return x


def load_spun_up_ic(config, n_steps, directory=IC_DIR):
    """The state ``spin_up`` wrote, after checking its recipe. Never builds one.

    Raises ``FileNotFoundError`` if it has not been built, and ``ValueError`` if it was built
    with a different recipe -- which includes every file that predates ``spin_up``, since
    none of them stores the ``params`` it is checked against.
    """
    path = ic_path(config, n_steps, directory)
    d = np.load(path)
    bad = _mismatched(d, _recipe(config, n_steps))
    if bad or int(d["steps_done"]) != int(n_steps):
        raise ValueError(f"{path} does not match the current recipe: {bad or ['steps_done']}")
    return jnp.asarray(d["x"])
