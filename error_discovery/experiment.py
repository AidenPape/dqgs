"""
error_discovery.experiment
==========================

The lead-time error-discovery sweep.

Pipeline (fully reproducible from a seed; the expensive attractor spin-up is
cached to ``results/attractor_ic.npz`` and reused):

1. Load (or build + cache) a spun-up base-model attractor state.
2. Build one long truth trajectory with the perturbed model and slice ``n_ic``
   independent segments out of it.
3. Estimate the base model's Lyapunov time (context; this config is only weakly
   chaotic).
4. Noiseless sweep: for each window and segment, recover the error coefficients
   with Levenberg-Marquardt and record recovery error and cond(J^T J).
5. Noise sweep: for a subset of windows, repeat with observation noise.

Results -> ``results/leadtime_<tag>.npz``.  The headline is the *window of
discoverability*: recovery is accurate at intermediate windows and fails for
windows that are too short (under-determined) or too long (loss of
identifiability), with noise tightening the upper edge.
"""

import os
import time

import numpy as np
import jax.numpy as jnp

from .config import (ExperimentConfig, RESULTS_DIR, IC_CACHE, IC_SEED, IC_SCALE,
                     steps_to_days)
from .models import TwinModels, obs_mask
from .discovery import rollout, recover, lyapunov_time


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


def run(cfg: ExperimentConfig = None, verbose=True, whiten=False):
    """Run the lead-time sweep. With ``whiten=True`` every fit (noiseless and noisy)
    divides each observed channel's residual by that channel's climatological std, so the
    reported conditioning is that of the properly-weighted Jacobian J_w = W J and the
    noisy recovery is the weighted least-squares estimate (reviewer #6/#7). The recovered
    coefficients are unchanged in the noiseless limit (the twin truth is a zero-residual
    minimizer under any positive weighting); only the geometry the SVD sees changes."""
    if cfg is None:
        cfg = ExperimentConfig()
    t0 = time.time()

    tw = TwinModels(cfg.eps, cfg.g_coeffs)
    vr = tw.spec.variables_range
    mask = obs_mask(tw.ndim, vr, cfg.obs)
    if verbose:
        print(f"[setup] ndim={tw.ndim} eps={cfg.eps} g_coeffs={cfg.g_coeffs} "
              f"obs={cfg.obs} c_true={np.asarray(tw.c_true)}")

    x_attr = get_attractor_ic(tw, cfg, verbose)
    seg_ics, seg_trajs = build_segments(tw, cfg, x_attr)

    lam, tau, tau_days = lyapunov_time(tw, x_attr, cfg.dt, n_blocks=cfg.lyap_blocks)
    if verbose:
        print(f"[lyap]  lambda_max={lam:.4g}/nondim  tau_lyap={tau_days:.1f} days "
              f"(weakly chaotic; context only)")

    windows = cfg.window_steps
    nW, nI = len(windows), cfg.n_ic

    # --- noiseless sweep: recovery + conditioning -------------------------- #
    rel = np.full((nW, nI), np.nan)
    cond = np.full((nW, nI), np.nan)         # cond(J^T J) = cond(J)^2 (back-compat)
    cond_J = np.full((nW, nI), np.nan)       # cond(J) from SVD (reviewer #3)
    smin = np.full((nW, nI), np.nan)         # smallest singular value of J
    smax = np.full((nW, nI), np.nan)         # largest singular value of J
    jrank = np.full((nW, nI), np.nan)        # numerical rank of J
    c_hats = np.full((nW, nI, 4), np.nan)
    for iw, w in enumerate(windows):
        for k in range(nI):
            r = recover(tw, seg_ics[k], seg_trajs[k][:w], cfg.dt, mask,
                        eta=0.0, n_iters=cfg.lm_iters, whiten=whiten)
            rel[iw, k] = r["rel_err"]
            cond[iw, k] = r["cond"]
            cond_J[iw, k] = r["cond_J"]
            smin[iw, k] = r["smin"]
            smax[iw, k] = r["smax"]
            jrank[iw, k] = r["rank"]
            c_hats[iw, k] = r["c_hat"]
        if verbose:
            print(f"[clean] w={w:5d} ({steps_to_days(w):6.2f} d)  "
                  f"rel_err med={np.nanmedian(rel[iw]):.3e}  "
                  f"cond(J) med={np.nanmedian(cond_J[iw]):.2e}  "
                  f"smin med={np.nanmedian(smin[iw]):.2e}  "
                  f"rank med={np.nanmedian(jrank[iw]):.0f}")

    # --- noise sweep: recovery vs window for several noise levels ---------- #
    etas = cfg.noise_levels
    nWn = len(cfg.noise_windows)
    rel_noise = np.full((len(etas), nWn, nI, cfg.n_noise_seeds), np.nan)
    for ie, eta in enumerate(etas):
        for jw, w in enumerate(cfg.noise_windows):
            for k in range(nI):
                for s in range(cfg.n_noise_seeds):
                    # unique, reproducible noise stream per (eta, window, segment, seed)
                    ns = cfg.seed + (((ie * nWn + jw) * nI + k) * cfg.n_noise_seeds + s)
                    r = recover(tw, seg_ics[k], seg_trajs[k][:w], cfg.dt, mask,
                                eta=eta, n_iters=cfg.lm_iters, noise_seed=ns,
                                whiten=whiten)
                    rel_noise[ie, jw, k, s] = r["rel_err"]
        if verbose:
            print(f"[noise] eta={eta:.0e}  rel_err med over windows="
                  f"{np.nanmedian(rel_noise[ie], axis=(1, 2))}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"leadtime_{cfg.tag}.npz")
    np.savez(
        out,
        windows=np.asarray(windows),
        window_days=np.asarray([steps_to_days(w) for w in windows]),
        rel_err=rel, cond=cond, cond_J=cond_J, smin=smin, smax=smax, jrank=jrank,
        c_hats=c_hats, c_true=np.asarray(tw.c_true),
        noise_levels=np.asarray(etas),
        noise_windows=np.asarray(cfg.noise_windows),
        noise_window_days=np.asarray([steps_to_days(w) for w in cfg.noise_windows]),
        rel_err_noise=rel_noise,
        lam_max=lam, tau_lyap_days=tau_days,
        eps=cfg.eps, g_coeffs=np.asarray(cfg.g_coeffs), obs=cfg.obs,
        dt=cfg.dt, seed=cfg.seed,
    )
    if verbose:
        print(f"[done]  wrote {out}  ({time.time() - t0:.1f}s total)")
    return out
