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
spans 5.07% (mode 1) to 8592% (mode 8) -- a ~1700x range against the 0-5% that produced the
null. (The three non-theta CVs quoted above are pre-fix; only mode 1 has been re-measured.)
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

**The answer, re-measured 2026-08-27 on disjoint segments: yes, and the effect is larger than
the overlapping-segment run showed.** The best-conditioned window splits the modes cleanly by
variability: the two low-CV modes sit at 57.4 d (mode 1, CV 5%; mode 4, CV 53%) while every mode
above CV 326% sits at 1.8-5.4 d. That is a **10-32x leftward shift across a CV threshold between
53% and 326%**, against the 2x shift the pre-fix run reported. The shift is a threshold, not a
gradient: it is not monotonic in CV above the threshold (mode 8 at CV 8592% optimises at 5.4 d,
mode 10 at CV 713% at 1.8 d).

The clean comparison is free, because the mode structure supplies four matched-amplitude
groups. Within a group the rms of the driver is identical to three digits, so the amplitude
confound below is held fixed and only the variability differs:

    rms 4.7e-3  modes 4 / 2 / 3   CV   53% / 1287% / 5059%   cond 1.50e4 / 6.43e3 / 6.26e3
    rms 3.4e-3  modes 5 / 6       CV  326% / 1162%           cond 6.44e3 / 6.27e3
    rms 6.6e-4  modes 7 / 8       CV 6200% / 8592%           cond 1.60e4 / 1.99e4
    rms 2.3e-4  modes 10 / 9      CV  713% / 6864%           cond 6.99e4 / 7.55e4

**This is where the segment fix cost the study its cleanest argument.** On overlapping segments
all four groups favoured the more variable mode, 4/4. On disjoint segments it is **2/4**: the
two larger-amplitude groups still favour variability, decisively (2.3x in the first group, where
the pre-fix run saw 31%), but both small-amplitude groups reverse -- mode 7 beats mode 8 and
mode 10 beats mode 9, in each case the *less* variable mode winning. Those two groups needed no
amplitude correction, which is exactly why they carried the argument, so the reversal cannot be
explained away as the confound below.

What survives is the threshold reading, now on firmer ground: crossing from CV ~50% to
CV ~1300% buys 2.3x, and going on from 1287% to 5059% buys 2.7%. Once a driver genuinely
varies, more variation buys nothing, and at small amplitude it may cost.

Two further results worth keeping.

Mode 1 has 5x the amplitude of modes 2 and 3 and still conditions worse (1.492e4 against 6.43e3
and 6.26e3), so variability beats amplitude where the two are opposed. Mode 4 is the control
that makes this readable: it shares modes 2-3's amplitude but has CV 53%, and it conditions with
mode 1 at 1.499e4, not with modes 2-3. Amplitude is held fixed across that pair, so the split is
variability.

**The noiseless-solvability claim did not survive the fix and is withdrawn.** It read: at the
shortest 0.45 d window mode 1 fails at rel_err 2.2e-4 while modes 7-10 recover four to five
orders better. On disjoint segments mode 1 reaches 2.9e-7 there and modes 7-10 span 1.4e-8 to
3.9e-6 -- a range that brackets mode 1 rather than beating it, with mode 8 the worst of all ten.
There is no ordering left to report.

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

That case duly arose: modes 7-10 condition at 1.60e4 to 7.55e4 against 6.26e3 to 1.50e4 for
modes 1-6, with rms 5-100x smaller. **That gap is not attributable and is not reported as one.**
Dividing it out does not rescue it either -- and after the segment fix it rescues it less than
before, because ``cond * rms`` no longer even falls monotonically (341 at mode 1, down to 10.6
at mode 7, then back up to 13.2, 17.3 and 16.2 at modes 8, 9 and 10). The correction also assumes
sigma_min scales with the driver amplitude, and it does not: compared at a fixed window,
sigma_min/rms spans 24.9x at 21.5 d and 10.0x at 89.7 d. (Its other assumption, that sigma_max
is amplitude-independent, does hold -- see the next paragraph.) Use the matched-amplitude
groups above, which need no correction, and treat ``cond * rms`` as suggestive only.

**Which half of the ratio moves, measured 2026-08-27 on disjoint segments.** Compared at a
*fixed* window rather than at each mode's own optimum, sigma_max is identical across all ten
modes -- to 1.0008-1.0016x on segment medians from 5.4 d out, 1.042x at the shortest window, and
never worse than 1.0114x within a single segment -- while sigma_min spans 14.5x at 21.5 d and
37.3x at 89.7 d. **The entire 11-26x spread in cond(J_w) across modes is sigma_min, and none of
it is sigma_max.** That is structural: sigma_max tracks the norm of the library's *constant*
column, measuring 1.18-1.23 x ||J_w[:, 0]|| at every window, and that column is unchanged when
slot 1 swaps modes -- the mode column is ~52x smaller at 89.7 d, so the largest singular
direction never notices which mode fills it. Any sigma_max spread quoted across modes is a
window artefact: sigma_max grows from 142 at 0.45 d to 3739 at 89.7 d, so comparing modes at
their own different optima mixes that growth in.

The reading caveat the overlapping-segment run needed has lapsed. It warned that ``argmin``
over the window axis was meaningless for modes 7 and 8, whose U varied by under 20% across
3.6-14.4 d. On disjoint segments every mode has a sharp optimum: the ``within 10% of the
minimum`` span is a single window for nine of the ten, and two windows for mode 8.

Whitened by default, like every study since 2026-08-20; see ``discovery.recover``.

Results -> ``results/driver_mode_<tag>.npz``.
"""

import os
import time

import numpy as np

from .. import figstyle
from ..attractor import get_attractor_ic, build_segments, segments_from_states
from ..config import (ExperimentConfig, RESULTS_DIR, steps_to_days, N_SAMPLES, SPACING_K,
                      MIN_EPISODE_DAYS, ETA_GRID)
from ..figstyle import DPI, LBL_BBOX
from ..models import TwinModels, obs_mask
from ..discovery import recover
from ..sampling import starting_states, sample_record, noise_scale, shared_noise

# The driver always sits in slot 1 of phi; only which theta_a mode fills it varies.
G_DRIVER = (0.0, 1.0, 0.0, 0.0)
ALL_MODES = tuple(range(1, 11))


def run(tag="main", modes=ALL_MODES, n_ic=3, n_seeds=2, lm_iters=40,
        noise_levels=(1e-4, 1e-3), whiten=True, verbose=True,
        sampling="legacy", n_samples=N_SAMPLES, spacing_k=SPACING_K,
        min_episode_days=MIN_EPISODE_DAYS, placement="onset", discard=0, eta_grid=ETA_GRID):
    """Sweep the optimisation window once per theta_a mode, noiseless and under noise.

    ``sampling="legacy"`` is the original design below. ``"passive"``, ``"active"`` or
    ``"both"`` follow the sampling standard (``plans/sampling.md``), as in ``driver_window``:
    ``n_samples`` starting states from the long base run shared by every mode, the
    climatological noise scale, shared noise draws and ``eta_grid``. The driver's statistics
    are then measured on every segment (``driver_*_per_state``) and their medians stored
    under the legacy keys. A sampled run left on the default tag is written under the mode's
    name instead, so it never replaces ``main``.

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
    picks, scale, n_seg = {}, None, n_ic
    if sampling != "legacy":
        tag = sampling if tag == "main" else tag
        picks = starting_states(sampling, n_samples, spacing_k, min_episode_days, placement,
                                window_steps=discard + max(windows))
        scale, n_seg, noise_levels = noise_scale(), n_samples, eta_grid
        noise = [[shared_noise(i, s, max(nwins), len(scale)) for s in range(n_seeds)]
                 for i in picks["snapshot_index"]]
    nM, nW, nEta, nWn = len(modes), len(windows), len(noise_levels), len(nwins)

    # One reference twin for the index layout; the per-mode twins are built in the loop.
    ref = TwinModels(cfg.eps, G_DRIVER)
    theta_a_start = int(ref.spec.variables_range[0])
    slots_23 = (ref.idx_To1, ref.idx_psia1)

    rel = np.full((nM, nW, n_seg), np.nan)
    cond_J = np.full((nM, nW, n_seg), np.nan)
    smin = np.full((nM, nW, n_seg), np.nan)
    smax = np.full((nM, nW, n_seg), np.nan)
    jrank = np.full((nM, nW, n_seg), np.nan)
    c_hats = np.full((nM, nW, n_seg, 4), np.nan)
    rel_noise = np.full((nM, nEta, nWn, n_seg, n_seeds), np.nan)

    # The driver's own statistics over the longest window, per mode. cv is the independent
    # variable this whole study is about; rms is what makes the amplitude confound
    # removable in analysis. Both measured on the first segment's truth trajectory.
    drv_idx = np.zeros(nM, dtype=int)
    drv_mean = np.full(nM, np.nan)
    drv_std = np.full(nM, np.nan)
    drv_rms = np.full(nM, np.nan)
    per_state = {k: np.full((nM, n_seg), np.nan) for k in ("mean", "std", "rms", "cv")}

    for im, k in enumerate(modes):
        idx = theta_a_start + k - 1
        drv_idx[im] = idx
        tw = TwinModels(cfg.eps, G_DRIVER, feature_idx=(idx, *slots_23))
        mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
        if sampling == "legacy":
            x_attr = get_attractor_ic(tw, cfg, verbose=False)
            seg_ics, seg_trajs = build_segments(tw, cfg, x_attr)
        else:
            seg_ics, seg_trajs = segments_from_states(tw, picks["states"], max(windows), dt,
                                                      discard)
        td = time.time()

        if sampling == "legacy":
            d = np.asarray(seg_trajs[0])[:, idx]
            drv_mean[im], drv_std[im] = d.mean(), d.std()
            drv_rms[im] = np.sqrt(np.mean(d ** 2))
        else:
            for j in range(n_seg):
                d = np.asarray(seg_trajs[j])[:, idx]
                per_state["mean"][im, j], per_state["std"][im, j] = d.mean(), d.std()
                per_state["rms"][im, j] = np.sqrt(np.mean(d ** 2))
            per_state["cv"][im] = per_state["std"][im] / np.abs(per_state["mean"][im])
            drv_mean[im] = np.median(per_state["mean"][im])
            drv_std[im] = np.median(per_state["std"][im])
            drv_rms[im] = np.median(per_state["rms"][im])

        for iw, w in enumerate(windows):
            for j in range(n_seg):
                r = recover(tw, seg_ics[j], seg_trajs[j][:w], dt, mask,
                            eta=0.0, n_iters=lm_iters, whiten=whiten, scale=scale)
                rel[im, iw, j] = r["rel_err"]
                cond_J[im, iw, j] = r["cond_J"]
                smin[im, iw, j] = r["smin"]
                smax[im, iw, j] = r["smax"]
                jrank[im, iw, j] = r["rank"]
                c_hats[im, iw, j] = r["c_hat"]

        for ie, eta in enumerate(noise_levels):
            for jw, w in enumerate(nwins):
                for j in range(n_seg):
                    for s in range(n_seeds):
                        if sampling == "legacy":
                            # unique, reproducible noise stream per (mode, eta, window, IC, seed)
                            ns = cfg.seed + ((((im * nEta + ie) * nWn + jw) * n_ic + j)
                                             * n_seeds + s)
                            r = recover(tw, seg_ics[j], seg_trajs[j][:w], dt, mask,
                                        eta=eta, n_iters=lm_iters, noise_seed=ns,
                                        whiten=whiten)
                        else:
                            r = recover(tw, seg_ics[j], seg_trajs[j][:w], dt, mask,
                                        eta=eta, n_iters=lm_iters, whiten=whiten,
                                        scale=scale, noise=noise[j][s])
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
        eps=cfg.eps, obs=cfg.obs, dt=dt, seed=cfg.seed, n_ic=n_seg, n_seeds=n_seeds,
        whiten=bool(whiten), sampling=sampling, **sample_record(picks),
        **({} if sampling == "legacy"
           else {f"driver_{k}_per_state": v for k, v in per_state.items()}),
    )
    if verbose:
        print(f"[done]  wrote {out}  ({time.time() - t0:.1f}s total)")
    return out


def _load(tag):
    return np.load(os.path.join(RESULTS_DIR, f"driver_mode_{tag}.npz"), allow_pickle=True)


def figure(tag="main", eta_index=0):
    """Conditioning, the two singular values behind it, and the usable window, per theta_a mode.

    (a) cond(J_w) against window, as ``driver_window`` plots it. (b) the two singular values
    that ratio is built from, which says which half of it moves: the sigma_max curves collapse
    onto a single line because sigma_max belongs to the library's constant column, so every mode
    difference in (a) is sigma_min. (c) recovery error under observation noise. All three share
    one colour scale, shown beside (c).

    Panel (b) resolves three clusters, not ten curves -- modes 1-6, then 7-8, then 9-10. Within
    the 1-6 cluster the modes overlap each other inside the segment-to-segment scatter (1.16-2.08x
    at 21.5 d, against a 13.5x fan between clusters), so only the cluster separation is readable.

    Modes are coloured sequentially rather than categorically because the mode index is an
    ordered quantity -- but note the ordering that matters here is CV, and CV is *not*
    monotonic in mode (mode 4 is 53%, mode 8 is 8592%). Read the curves against the CV
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

    fig, (ax, axb, axc) = plt.subplots(1, 3, figsize=(16.4, 4.4))
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
                 rf"{wnote}; $\star$ = best window",
                 fontsize=9.5, loc="left")

    for i, m in enumerate(modes):
        # Median over segments, no bands: twenty curves already, and the structure that matters
        # -- the three clusters -- is wider than the per-mode segment scatter anyway.
        axb.plot(wd, np.nanmedian(d["smax"][i], axis=1), "o-", ms=4, color=colors[i])
        axb.plot(wd, np.nanmedian(d["smin"][i], axis=1), ls="--", color=colors[i])
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel("optimisation window  [days]")
    axb.set_ylabel(rf"singular values of ${jsym}$")
    axb.set_title(rf"(b)  cond$({jsym})$ decomposed: the gap between the bundles" "\n"
                  r"$\sigma_{\max}$ is set by the constant column, not by the mode",
                  fontsize=9.5, loc="left")
    # style-only legend; the colorbar beside (c) already carries the mode scale
    axb.legend(handles=[plt.Line2D([], [], color="grey", ls="-", marker="o", ms=4,
                                   label=r"$\sigma_{\max}$"),
                        plt.Line2D([], [], color="grey", ls="--", label=r"$\sigma_{\min}$")],
               frameon=False, fontsize=8)

    for i, m in enumerate(modes):
        y = np.nanmedian(d["rel_err_noise"][i, eta_index], axis=(1, 2))
        axc.plot(nwd, y, "s-", ms=5, color=colors[i], label=f"{m}")
    axc.axhline(0.1, ls=":", color="grey", lw=1)
    axc.text(nwd[0], 0.105, "10%", color="grey", fontsize=8, va="bottom", ha="left",
             bbox=LBL_BBOX)
    axc.set_xscale("log"); axc.set_yscale("log")
    axc.set_xlabel("optimisation window  [days]")
    axc.set_ylabel(r"$\|\hat c - c_{\rm true}\|/\|c_{\rm true}\|$")
    axc.set_title(rf"(c)  recovery under noise ($\eta={etas[eta_index]:.0e}$)"
                  "\nhigher modes carry less amplitude at fixed $\\epsilon$",
                  fontsize=9.5, loc="left")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axc, pad=0.02)
    cb.set_label(r"driver mode $k$  ($\theta_{a,k}$)", fontsize=9)
    cb.set_ticks(modes)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"fig_driver_mode_{tag}.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return [out]
