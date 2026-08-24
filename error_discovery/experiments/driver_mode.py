"""
error_discovery.experiments.driver_mode
=======================================

Does the window of discoverability depend on how much the driver *varies*?

``driver_window.py`` asked whether the window depends on which driver the error rides on and
answered no: all five drivers bottom out at the same 21.5-day window with cond(J_w) agreeing
to 1.034x. That is a weak no, because all four of its drivers are near-constant -- their
coefficients of variation over a 21.5 d window are 0% (const), 5.2% (theta_a1), 3.5% (T_o1)
and 1.6% (psi_a1). Near-identical functions condition alike; nothing else was on offer.

This study supplies the variation. It sweeps the driver across the ten theta_a modes, whose CV
spans 5.2% (mode 1) to 3013% (mode 9) -- a ~580x range against the 0-5% that produced the null.
The library keeps its four slots and its dimension; only which state component fills slot 1
changes, via ``TwinModels(..., feature_idx=...)``:

    phi(x) = (1, theta_{a,k}, T_{o,1}, psi_{a,1}),   g_coeffs = (0, 1, 0, 0)

so the injected error is eps * theta_{a,k}(x) * (E @ x) and mode 1 is exactly
``driver_window``'s ``theta_a1`` run -- which makes it a free plumbing check, not a fifth data
point. Because c stays 4-dimensional, ``discovery.lm_fit``'s ``zeros(4)`` and every
``c_hats[..., 4]`` shape elsewhere are untouched.

Unlike ``driver_window``, the library genuinely differs between runs here: a fast-varying
column is not the same basis as a near-constant one. That is the point. The null in
``driver_window`` followed from cond(J) being built from singular *values* of a *fixed* basis,
so it could not depend on where c_true sat. Here the basis itself moves, so the question is
live.

**The answer, measured 2026-08-20: yes, and the null was an artefact of near-constant
drivers.** The best-conditioned window moves from 21.5 d at mode 1 to 14.4 d at modes 2-4 and
10.8 d at mode 5 -- a 2x leftward shift that ``driver_window`` had no varying driver to see.

The clean comparison is free, because the mode structure supplies four matched-amplitude
groups. Within a group the rms of the driver is identical to three digits, so the amplitude
confound below is held fixed and only the variability differs:

    rms 4.7e-3  modes 4 / 2 / 3   CV   53% / 1287% / 5059%   cond 3.50e3 / 2.67e3 / 2.69e3
    rms 3.3e-3  modes 5 / 6       CV  326% / 1162%           cond 3.06e3 / 3.03e3
    rms 6.6e-4  modes 7 / 8       CV 6200% / 8592%           cond 1.13e4 / 1.06e4
    rms 2.3e-4  modes 10 / 9      CV  713% / 6864%           cond 2.79e4 / 2.66e4

All four groups favour the more variable mode, so the direction is consistent 4/4. But the
size is not: 31% in the first group and 1-7% in the rest. **It reads as a threshold, not a
gradient** -- crossing from CV ~50% to CV ~1300% buys 31%, and going on from 1287% to 5059%
buys 0.6%. Once a driver genuinely varies, more variation buys almost nothing.

Two further results worth keeping.

Mode 1 has 5x the amplitude of modes 2-4 and still conditions worse (4.153e3 against 2.67e3),
so variability beats amplitude where the two are opposed. And noiselessly, at the shortest
0.45 d window, mode 1 fails at rel_err 2.2e-4 while modes 7-10 recover to 1e-9..1e-8 -- four to
five orders better -- even though their cond(J) there is comparable (~1e7). Conditioning alone
does not predict noiseless solvability at the under-determined end.

Under observation noise the picture inverts at the small-amplitude end: modes 2-6 are the best
recovered (below 1% by 28.7 d at eta=1e-4) because they have both variability *and* amplitude,
while modes 9-10 never reach 1%. There is a sweet spot, not a monotone benefit.

**Read this as a numerical experiment on identifiability, not as a second physical scenario.**
A high-CV mode has a near-zero mean, so g(x) changes sign along the trajectory and the injected
heat-exchange error flips sign with it. That is clean as a probe of conditioning geometry and
physically odd as "an error in an exchange coefficient."

**Confound, recorded rather than removed.** Mode amplitude falls with k: theta_a1 has
RMS 2.4e-2, theta_a9 has 2.5e-4. At the fixed eps = 0.1 used here the injected error is
therefore ~96x smaller at mode 9, and J's driver column scales with it. So a finding of the
form "high modes are harder" is confounded with the error simply being smaller and must not be
reported as a variability result. A finding that high modes are *better* conditioned is clean,
since it would hold despite the amplitude penalty. The per-mode ``rms`` is saved so the
normalisation can be applied in analysis without re-running.

That case duly arose: modes 7-10 condition at 1.1e4 to 2.8e4 against 2.7e3 to 4.2e3 for modes
1-6, with rms 5-100x smaller. **That gap is not attributable and is not reported as one.**
Dividing it out does not rescue it either -- ``cond * rms`` does fall monotonically (95.0 at
mode 1 to 6.1 at mode 9), but the correction assumes sigma_min scales with the driver
amplitude and sigma_max does not, and neither holds: across modes at their own optima sigma_max
spans 3.9x and sigma_min/rms spans 7.2x. Use the matched-amplitude groups above, which need no
correction, and treat ``cond * rms`` as suggestive only.

One reading caveat: ``argmin`` over the window axis is only meaningful where the U has
curvature. Modes 7 and 8 vary by under 20% across 3.6-14.4 d, so their reported optima (10.8 d
and 3.6 d) differ by noise, not by physics. The ``within 10% of the minimum`` span is the
honest statistic there; modes 1-6 are the ones with a genuine minimum.

Whitened by default, like every study since 2026-08-20; see ``discovery.recover``.

Results -> ``results/driver_mode_<tag>.npz``.
"""

import os
import time

import numpy as np

from .. import figstyle
from ..attractor import get_attractor_ic, build_segments
from ..config import ExperimentConfig, RESULTS_DIR, steps_to_days
from ..figstyle import DPI, LBL_BBOX
from ..models import TwinModels, obs_mask
from ..discovery import recover

# The driver always sits in slot 1 of phi; only which theta_a mode fills it varies.
G_DRIVER = (0.0, 1.0, 0.0, 0.0)
ALL_MODES = tuple(range(1, 11))


def run(tag="main", modes=ALL_MODES, n_ic=3, n_seeds=2, lm_iters=40,
        noise_levels=(1e-4, 1e-3), whiten=True, verbose=True):
    """Sweep the optimisation window once per theta_a mode, noiseless and under noise.

    ``n_ic``, ``n_seeds``, the window set and the noise windows all match
    ``driver_window`` so the curves overlay ``driver_window_main`` directly, and so mode 1
    reproduces its ``theta_a1`` curve exactly.

    Every mode shares the base model, so all ten read the same cached attractor IC --
    ``get_attractor_ic`` keys on ndim, spin-up length, seed and dt, none of which vary with
    ``feature_idx``. Nothing here regenerates that cache. The truth *trajectory* does vary
    with the mode, so segments are rebuilt per mode.
    """
    t0 = time.time()
    cfg = ExperimentConfig()
    cfg.n_ic = n_ic
    dt = cfg.dt
    windows = cfg.window_steps
    nwins = cfg.noise_windows
    nM, nW, nEta, nWn = len(modes), len(windows), len(noise_levels), len(nwins)

    # One reference twin for the index layout; the per-mode twins are built in the loop.
    ref = TwinModels(cfg.eps, G_DRIVER)
    theta_a_start = int(ref.spec.variables_range[0])
    slots_23 = (ref.idx_To1, ref.idx_psia1)

    rel = np.full((nM, nW, n_ic), np.nan)
    cond_J = np.full((nM, nW, n_ic), np.nan)
    smin = np.full((nM, nW, n_ic), np.nan)
    smax = np.full((nM, nW, n_ic), np.nan)
    jrank = np.full((nM, nW, n_ic), np.nan)
    c_hats = np.full((nM, nW, n_ic, 4), np.nan)
    rel_noise = np.full((nM, nEta, nWn, n_ic, n_seeds), np.nan)

    # The driver's own statistics over the longest window, per mode. cv is the independent
    # variable this whole study is about; rms is what makes the amplitude confound
    # removable in analysis. Both measured on the first segment's truth trajectory.
    drv_idx = np.zeros(nM, dtype=int)
    drv_mean = np.full(nM, np.nan)
    drv_std = np.full(nM, np.nan)
    drv_rms = np.full(nM, np.nan)

    for im, k in enumerate(modes):
        idx = theta_a_start + k - 1
        drv_idx[im] = idx
        tw = TwinModels(cfg.eps, G_DRIVER, feature_idx=(idx, *slots_23))
        mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
        x_attr = get_attractor_ic(tw, cfg, verbose=False)
        seg_ics, seg_trajs = build_segments(tw, cfg, x_attr)
        td = time.time()

        d = np.asarray(seg_trajs[0])[:, idx]
        drv_mean[im], drv_std[im] = d.mean(), d.std()
        drv_rms[im] = np.sqrt(np.mean(d ** 2))

        for iw, w in enumerate(windows):
            for j in range(n_ic):
                r = recover(tw, seg_ics[j], seg_trajs[j][:w], dt, mask,
                            eta=0.0, n_iters=lm_iters, whiten=whiten)
                rel[im, iw, j] = r["rel_err"]
                cond_J[im, iw, j] = r["cond_J"]
                smin[im, iw, j] = r["smin"]
                smax[im, iw, j] = r["smax"]
                jrank[im, iw, j] = r["rank"]
                c_hats[im, iw, j] = r["c_hat"]

        for ie, eta in enumerate(noise_levels):
            for jw, w in enumerate(nwins):
                for j in range(n_ic):
                    for s in range(n_seeds):
                        # unique, reproducible noise stream per (mode, eta, window, IC, seed)
                        ns = cfg.seed + ((((im * nEta + ie) * nWn + jw) * n_ic + j)
                                         * n_seeds + s)
                        r = recover(tw, seg_ics[j], seg_trajs[j][:w], dt, mask,
                                    eta=eta, n_iters=lm_iters, noise_seed=ns,
                                    whiten=whiten)
                        rel_noise[im, ie, jw, j, s] = r["rel_err"]

        if verbose:
            cj = np.nanmedian(cond_J[im], axis=1)
            cv = drv_std[im] / abs(drv_mean[im]) if drv_mean[im] != 0 else np.inf
            print(f"[theta_a{k:<2d}] CV={cv * 100:9.2f}%  rms={drv_rms[im]:.3e}  "
                  f"best cond(J)={cj.min():.2e} at "
                  f"{steps_to_days(windows[int(np.argmin(cj))]):.1f} d   "
                  f"noiseless rel_err med={np.nanmedian(rel[im]):.2e}   "
                  f"[{time.time() - td:.0f}s]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"driver_mode_{tag}.npz")
    np.savez(
        out,
        modes=np.asarray(modes),
        driver_index=drv_idx,
        driver_mean=drv_mean, driver_std=drv_std, driver_rms=drv_rms,
        g_coeffs=np.asarray(G_DRIVER),
        c_true=np.asarray(ref.c_true),
        windows=np.asarray(windows),
        window_days=np.asarray([steps_to_days(w) for w in windows]),
        rel_err=rel, cond_J=cond_J, smin=smin, smax=smax, jrank=jrank, c_hats=c_hats,
        noise_levels=np.asarray(noise_levels),
        noise_windows=np.asarray(nwins),
        noise_window_days=np.asarray([steps_to_days(w) for w in nwins]),
        rel_err_noise=rel_noise,
        eps=cfg.eps, obs=cfg.obs, dt=dt, seed=cfg.seed, n_ic=n_ic, n_seeds=n_seeds,
        whiten=bool(whiten),
    )
    if verbose:
        print(f"[done]  wrote {out}  ({time.time() - t0:.1f}s total)")
    return out


def _load(tag):
    return np.load(os.path.join(RESULTS_DIR, f"driver_mode_{tag}.npz"), allow_pickle=True)


def figure(tag="main", eta_index=0):
    """The conditioning U-curve and the usable window, one curve per theta_a mode.

    Same two panels as ``driver_window``: (a) cond(J) against window, (b) recovery error
    under observation noise. Both share one colour scale, shown beside (b).

    Modes are coloured sequentially rather than categorically because the mode index is an
    ordered quantity -- but note the ordering that matters here is CV, and CV is *not*
    monotonic in mode (mode 4 is 83%, mode 9 is 3013%). Read the curves against the CV
    column that ``run`` prints, not against the colour ramp.
    """
    plt = figstyle.pyplot()
    d = _load(tag)
    modes = [int(m) for m in d["modes"]]
    wd, nwd = d["window_days"], d["noise_window_days"]
    etas = d["noise_levels"]
    wh = bool(d["whiten"]) if "whiten" in d.files else False
    jsym = r"J_w" if wh else "J"
    wnote = "whitened residual (1/std per channel)" if wh else "unwhitened residual"

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    norm = plt.Normalize(vmin=min(modes), vmax=max(modes))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(norm(m)) for m in modes]

    for i, m in enumerate(modes):
        cj = np.nanmedian(d["cond_J"][i], axis=1)
        ax.plot(wd, cj, "o-", ms=4, color=colors[i], label=f"{m}")
        ax.plot(wd[int(np.argmin(cj))], cj.min(), "*", ms=13, color=colors[i])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("optimisation window  [days]")
    ax.set_ylabel(rf"cond$({jsym})=\sigma_{{\max}}/\sigma_{{\min}}$")
    ax.set_title(r"(a)  conditioning vs window, by $\theta_a$ mode of the driver" "\n"
                 rf"{wnote}; $\star$ = each mode's best-conditioned window",
                 fontsize=9.5, loc="left")

    for i, m in enumerate(modes):
        y = np.nanmedian(d["rel_err_noise"][i, eta_index], axis=(1, 2))
        axb.plot(nwd, y, "s-", ms=5, color=colors[i], label=f"{m}")
    axb.axhline(0.1, ls=":", color="grey", lw=1)
    axb.text(nwd[0], 0.105, "10%", color="grey", fontsize=8, va="bottom", ha="left",
             bbox=LBL_BBOX)
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel("optimisation window  [days]")
    axb.set_ylabel(r"$\|\hat c - c_{\rm true}\|/\|c_{\rm true}\|$")
    axb.set_title(rf"(b)  recovery under noise ($\eta={etas[eta_index]:.0e}$)"
                  "\nhigher modes carry less amplitude at fixed $\\epsilon$",
                  fontsize=9.5, loc="left")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axb, pad=0.02)
    cb.set_label(r"driver mode $k$  ($\theta_{a,k}$)", fontsize=9)
    cb.set_ticks(modes)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"fig_driver_mode_{tag}.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return [out]
