"""
error_discovery.experiments.sparsity
====================================

Differentiable error discovery under *partial / noisy* observations.

Question: can backpropagation-through-the-dynamics recover the (ocean-coupling)
thermal-exchange error when only part of the state is observed, and how does the
choice of observed variables interact with observation noise?

Two empirical facts motivate the design (verified in the probe):

* **Noiseless**: the error is recovered to ~$10^{-3}$ from *any* observation
  subset (full / atmosphere-only / ocean-only / temperatures-only). The
  fully-coupled dynamics spread the localized error's signature into every
  component, so observation locality is not a barrier in principle.
* **Noisy**: the choice of observed variables dominates -- but not in the
  direction the probe suggested. **Corrected 2026-08-27 against this study's own
  output:** ``ocean`` is the worst operator at every noise level, by 3-9x
  (median rel_err 0.058 noiseless to 0.91 at eta=1e-3), and ``full`` is never
  worst. The claim that adding noisy streamfunction channels is worst does not
  hold here; it was already contradicted by the pre-fix run, so this is a
  standing error being fixed, not a consequence of the segment change.
  Observing the temperatures the error directly perturbs (``temp``) is best at
  the three lowest noise levels and loses to ``atmos``/``full`` above eta=3e-4,
  so "temperatures are best" holds only in the low-noise limit.

So the sweep is noise-centric. For each observation operator we measure recovery
versus (a) observation-noise level at a fixed window and (b) window length at a
fixed noise level, plus the oracle identifiability floor (regression of the
exact error tendency, which recovers the truth to machine precision).

Results -> ``results/data/sparsity_<tag>.npz``.
"""

import os
import time

import numpy as np
import jax.numpy as jnp

from ..config import (ExperimentConfig, DATA_DIR, FIGURES_DIR, steps_to_days, N_SAMPLES, SPACING_K,
                      MIN_EPISODE_DAYS, ETA_GRID)
from ..models import TwinModels, obs_mask
from ..discovery import rollout, recover
from .. import figstyle
from ..attractor import get_attractor_ic, build_segments, segments_from_states
from ..sampling import starting_states, sample_record, noise_scale, shared_noise
from ..figstyle import DPI, LBL_BBOX

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
        windows=(320, 640, 1280, 2560), eta_fix=3e-4, whiten=True, verbose=True,
        sampling="legacy", n_samples=N_SAMPLES, spacing_k=SPACING_K,
        min_episode_days=MIN_EPISODE_DAYS, placement="onset", discard=0, eta_grid=ETA_GRID):
    """Recovery against noise and against window, for each observation mask.

    ``sampling="legacy"`` is the original design. ``"passive"``, ``"active"`` or ``"both"``
    follow the sampling standard (``plans/sampling.md``): ``n_samples`` starting states from
    the long base run; the climatological noise scale; one standard-normal draw per
    (state, seed) shared by every mask, level and window; panel (a)'s levels ``(0,) + eta_grid``
    in place of ``noise_levels``; and the oracle floor computed from every starting state
    (``oracle_floor_per_state``, median under ``oracle_floor``). A sampled run left on the
    default tag is written under the mode's name instead.
    """
    t0 = time.time()
    cfg = ExperimentConfig()
    cfg.n_ic = n_ic
    cfg.window_steps = tuple(sorted(set(windows) | {noise_window}))
    tw = TwinModels(cfg.eps, cfg.g_coeffs)
    dt = cfg.dt

    picks, scale = {}, None
    if sampling == "legacy":
        x_attr = get_attractor_ic(tw, cfg, verbose)
        seg_ics, seg_trajs = build_segments(tw, cfg, x_attr)
        floor = oracle_floor(tw, x_attr, dt)
    else:
        tag = sampling if tag == "sparsity" else tag
        max_w = max(cfg.window_steps)
        picks = starting_states(sampling, n_samples, spacing_k, min_episode_days, placement,
                                window_steps=discard + max_w)
        seg_ics, seg_trajs = segments_from_states(tw, picks["states"], max_w, dt, discard)
        scale, n_ic, noise_levels = noise_scale(), n_samples, (0.0,) + tuple(eta_grid)
        noise = [[shared_noise(i, s, max_w, tw.ndim) for s in range(n_seeds)]
                 for i in picks["snapshot_index"]]
        floors = np.asarray([oracle_floor(tw, x, dt) for x in seg_ics])
        floor = float(np.median(floors))
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
                    if sampling == "legacy":
                        # unique, reproducible noise stream per (obs, eta, IC, seed)
                        ns = cfg.seed + (((io * nEta + ie) * n_ic + k) * n_seeds + s)
                        r = recover(tw, seg_ics[k], Y, dt, mask, eta=eta,
                                    n_iters=lm_iters, noise_seed=ns, whiten=whiten)
                    else:
                        r = recover(tw, seg_ics[k], Y, dt, mask, eta=eta, n_iters=lm_iters,
                                    whiten=whiten, scale=scale, noise=noise[k][s])
                    rel_noise[io, ie, k, s] = r["rel_err"]
        for jw, w in enumerate(windows):
            for k in range(n_ic):
                Y = seg_trajs[k][:w]
                for s in range(n_seeds):
                    if sampling == "legacy":
                        ns = 10 ** 6 + cfg.seed + (((io * nW + jw) * n_ic + k) * n_seeds + s)
                        r = recover(tw, seg_ics[k], Y, dt, mask, eta=eta_fix,
                                    n_iters=lm_iters, noise_seed=ns, whiten=whiten)
                    else:
                        r = recover(tw, seg_ics[k], Y, dt, mask, eta=eta_fix, n_iters=lm_iters,
                                    whiten=whiten, scale=scale, noise=noise[k][s])
                    rel_window[io, jw, k, s] = r["rel_err"]
        if verbose:
            mn = np.nanmedian(rel_noise[io], axis=(1, 2))
            print(f"[obs={obs:6s}] median rel_err vs noise={np.array2string(mn, precision=3)}")

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, f"sparsity_{tag}.npz")
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
        sampling=sampling, **sample_record(picks),
        **({} if sampling == "legacy" else {"oracle_floor_per_state": floors}),
    )
    if verbose:
        print(f"[done]  wrote {out}  ({time.time() - t0:.1f}s)")
    return out

# --------------------------------------------------------------------------- #
# Figures. Each reloads the .npz above; none of them computes.
# --------------------------------------------------------------------------- #
def _load_sparsity(tag):
    return np.load(os.path.join(DATA_DIR, f"sparsity_{tag}.npz"), allow_pickle=True)


_OBS_LABEL = {"full": "full (36)", "atmos": "atmosphere (20)",
              "ocean": "ocean (16)", "temp": "temperatures (18)"}


def fig_sparsity_noise(tag="sparsity"):
    """Recovery error vs observation noise, one curve per observation operator."""
    plt = figstyle.pyplot()
    d = _load_sparsity(tag)
    obs = [o.decode() if isinstance(o, bytes) else str(o) for o in d["obs_kinds"]]
    etas = d["noise_levels"]
    rel = d["rel_noise"]                       # (nO, nEta, nIC, nSeed)
    floor = float(d["oracle_floor"])
    wdays = float(d["noise_window_days"])
    # plot eta=0 as a small floor value so it shows on a log axis
    xed = np.array([e if e > 0 else etas[etas > 0].min() / 3 for e in etas])

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for io, o in enumerate(obs):
        med = np.nanmedian(rel[io], axis=(1, 2))
        lo, hi = np.nanpercentile(rel[io], [25, 75], axis=(1, 2))
        line, = ax.plot(xed, med, "o-", lw=2, label=_OBS_LABEL.get(o, o))
        ax.fill_between(xed, lo, hi, color=line.get_color(), alpha=0.15)
    ax.axhline(0.1, ls=":", color="grey", lw=1)
    # left edge is clear (all curves are well below 0.1 at the noiseless point); the
    # right edge is where several curves cross 0.1, so the old right-anchored label sat on them
    ax.text(xed[0], 0.105, "10% (usable)", color="grey", fontsize=8, ha="left",
            va="bottom", bbox=LBL_BBOX)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1.5e-3, 3.0)            # focus on the data; oracle floor is far below (annotated)
    ax.set_xlabel(r"observation noise $\eta$ (relative; leftmost = noiseless)")
    ax.set_ylabel(r"$\|\hat c - c_{\rm true}\|/\|c_{\rm true}\|$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.985, 0.02, f"oracle (exact tendency) ≈ {floor:.0e} — off scale below",
            transform=ax.transAxes, fontsize=7.5, style="italic", color="0.4", ha="right")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f"fig_sparsity_noise_{tag}.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out


def fig_sparsity_window(tag="sparsity"):
    """Recovery error vs window per observation operator at fixed noise."""
    plt = figstyle.pyplot()
    d = _load_sparsity(tag)
    obs = [o.decode() if isinstance(o, bytes) else str(o) for o in d["obs_kinds"]]
    wd = d["window_days"]; rel = d["rel_window"]; eta = float(d["eta_fix"])
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for io, o in enumerate(obs):
        med = np.nanmedian(rel[io], axis=(1, 2))
        ax.plot(wd, med, "o-", label=_OBS_LABEL.get(o, o))
    ax.axhline(0.1, ls=":", color="grey", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("optimisation window  [days]")
    ax.set_ylabel(r"$\|\hat c - c_{\rm true}\|/\|c_{\rm true}\|$")
    ax.set_title(fr"Discovery vs window at fixed noise $\eta={eta:.0e}$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f"fig_sparsity_window_{tag}.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out


def summarize_sparsity(tag="sparsity"):
    d = _load_sparsity(tag)
    obs = [o.decode() if isinstance(o, bytes) else str(o) for o in d["obs_kinds"]]
    etas = d["noise_levels"]
    rel = d["rel_noise"]
    print(f"== sparsity summary [{tag}] (window {float(d['noise_window_days']):.0f} d) ==")
    print(f"oracle (exact-tendency) floor: {float(d['oracle_floor']):.2e}")
    print("median rel_err by obs operator vs noise eta:")
    print("  eta:        " + "  ".join(f"{e:7.0e}" for e in etas))
    for io, o in enumerate(obs):
        med = np.nanmedian(rel[io], axis=(1, 2))
        print(f"  {o:10s} " + "  ".join(f"{m:7.3f}" for m in med))
    return d


def figure(tag="sparsity"):
    figs = [fig_sparsity_noise(tag), fig_sparsity_window(tag)]
    summarize_sparsity(tag)
    return figs
