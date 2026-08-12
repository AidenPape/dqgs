"""
error_discovery.experiment_sparsity
===================================

Differentiable error discovery under *partial / noisy* observations.

Question: can backpropagation-through-the-dynamics recover the (ocean-coupling)
thermal-exchange error when only part of the state is observed, and how does the
choice of observed variables interact with observation noise?

Two empirical facts motivate the design (verified in the probe):

* **Noiseless**: the error is recovered to ~$10^{-3}$ from *any* observation
  subset (full / atmosphere-only / ocean-only / temperatures-only). The
  fully-coupled dynamics spread the localized error's signature into every
  component, so observation locality is not a barrier in principle.
* **Noisy**: the choice of observed variables dominates. Observing the
  temperatures the error directly perturbs is best; observing *everything*
  (adding noisy streamfunction channels) is worst.

So the sweep is noise-centric. For each observation operator we measure recovery
versus (a) observation-noise level at a fixed window and (b) window length at a
fixed noise level, plus the oracle identifiability floor (regression of the
exact error tendency, which recovers the truth to machine precision).

Results -> ``results/sparsity_<tag>.npz``;  figures via :mod:`error_discovery.plotting`.
"""

import os
import time

import numpy as np
import jax.numpy as jnp

from .config import ExperimentConfig, RESULTS_DIR, steps_to_days
from .models import TwinModels, obs_mask
from .discovery import rollout, recover
from .experiment import get_attractor_ic, build_segments

OBS_KINDS = ("full", "atmos", "ocean", "temp")


def oracle_floor(tw, x_attr, dt, n_samples=3000):
    """Identifiability floor: regress the *exact* error tendency on the library.

    The error tendency m(x)=eps*g(x)*(E x) is exactly linear in the four
    coefficients via the basis phi_k(x)*(E x); least squares over attractor
    samples therefore recovers c_true to machine precision when the full
    instantaneous tendency is available.  Returns the relative recovery error.
    """
    traj = np.asarray(rollout(lambda x: tw.f_base(x), x_attr, dt, n_samples))
    E = np.asarray(tw.E_full)
    rows, rhs = [], []
    for x in traj:
        ex = E @ x
        phi = np.array([1.0, x[tw.idx_thetaa1], x[tw.idx_To1], x[tw.idx_psia1]])
        m = np.asarray(tw.f_true(jnp.asarray(x)) - tw.f_base(jnp.asarray(x)))
        for i in range(tw.ndim):
            rows.append(phi * ex[i])
            rhs.append(m[i])
    chat, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    c_true = np.asarray(tw.c_true)
    return float(np.linalg.norm(chat - c_true) / np.linalg.norm(c_true))


def run(tag="sparsity", n_ic=4, n_seeds=4, lm_iters=12,
        noise_window=1280, noise_levels=(0.0, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3),
        windows=(320, 640, 1280, 2560), eta_fix=3e-4, whiten=True, verbose=True):
    t0 = time.time()
    cfg = ExperimentConfig()
    cfg.n_ic = n_ic
    cfg.window_steps = tuple(sorted(set(windows) | {noise_window}))
    tw = TwinModels(cfg.eps, cfg.g_coeffs)
    dt = cfg.dt

    x_attr = get_attractor_ic(tw, cfg, verbose)
    seg_ics, seg_trajs = build_segments(tw, cfg, x_attr)
    floor = oracle_floor(tw, x_attr, dt)
    if verbose:
        print(f"[setup] obs={OBS_KINDS}  oracle (exact-tendency) floor={floor:.2e}")

    nO, nEta, nW = len(OBS_KINDS), len(noise_levels), len(windows)

    # (a) recovery vs noise at a fixed window
    rel_noise = np.full((nO, nEta, n_ic, n_seeds), np.nan)
    # (b) recovery vs window at a fixed noise level
    rel_window = np.full((nO, nW, n_ic, n_seeds), np.nan)

    for io, obs in enumerate(OBS_KINDS):
        mask = obs_mask(tw.ndim, tw.spec.variables_range, obs)
        for ie, eta in enumerate(noise_levels):
            for k in range(n_ic):
                Y = seg_trajs[k][:noise_window]
                for s in range(n_seeds):
                    # unique, reproducible noise stream per (obs, eta, IC, seed)
                    ns = cfg.seed + (((io * nEta + ie) * n_ic + k) * n_seeds + s)
                    r = recover(tw, seg_ics[k], Y, dt, mask, eta=eta,
                                n_iters=lm_iters, noise_seed=ns, whiten=whiten)
                    rel_noise[io, ie, k, s] = r["rel_err"]
        for jw, w in enumerate(windows):
            for k in range(n_ic):
                Y = seg_trajs[k][:w]
                for s in range(n_seeds):
                    ns = 10 ** 6 + cfg.seed + (((io * nW + jw) * n_ic + k) * n_seeds + s)
                    r = recover(tw, seg_ics[k], Y, dt, mask, eta=eta_fix,
                                n_iters=lm_iters, noise_seed=ns, whiten=whiten)
                    rel_window[io, jw, k, s] = r["rel_err"]
        if verbose:
            mn = np.nanmedian(rel_noise[io], axis=(1, 2))
            print(f"[obs={obs:6s}] median rel_err vs noise={np.array2string(mn, precision=3)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"sparsity_{tag}.npz")
    np.savez(
        out,
        obs_kinds=np.asarray(OBS_KINDS),
        noise_levels=np.asarray(noise_levels),
        noise_window=noise_window,
        noise_window_days=steps_to_days(noise_window),
        rel_noise=rel_noise,
        windows=np.asarray(windows),
        window_days=np.asarray([steps_to_days(w) for w in windows]),
        eta_fix=eta_fix,
        rel_window=rel_window,
        oracle_floor=floor,
        eps=cfg.eps, g_coeffs=np.asarray(cfg.g_coeffs), dt=dt,
    )
    if verbose:
        print(f"[done]  wrote {out}  ({time.time() - t0:.1f}s)")
    return out


def figure(tag="sparsity"):
    """The partial-observation figures, drawn from ``sparsity_<tag>.npz``."""
    from . import plotting
    return plotting.make_sparsity(tag)
