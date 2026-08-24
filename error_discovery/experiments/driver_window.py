"""
error_discovery.experiments.driver_window
=========================================

Does the window of discoverability depend on *which* driver the error rides on?

The lead-time study sweeps the optimisation window for one injected error and reports
cond(J) against it -- the U-shape that defines the window. Every run of it has used the
project default g = T_{o,1}. This study runs the same sweep for each mode-1 driver in
turn, plus the three-driver mixed error, so the U-curves can be laid on top of each
other.

The comparison is well posed because ``discovery.recover`` always fits the whole
four-vector c whatever ``g_coeffs`` says: every driver is the same 4-D inverse problem,
and what differs is the truth trajectory J is evaluated along.

The answer: **the conditioning U is a property of the base dynamics, not of the driver.**
Measured 2026-08-20 on the whitened residual, all five drivers bottom out at the same 21.5-day
window, cond(J_w) = 4.02e3 to 4.15e3 -- a spread of 1.034x. The U closes on both sides: the
right arm reaches 3.05x the minimum by 89.7 d, the left arm 3.1e3 to 7.2e3 times it by 0.45 d.
Only that left arm separates the drivers, and only by 2.26x, in the order you would expect --
the constant driver, which adds no state dependence at all, is the worst conditioned.
(The earlier unwhitened run put the same collapse at a 10.8-day minimum with cond(J) = 5.8e3.)

That collapse is not a coincidence of the truth. J's columns are the *library*, not the
answer: ``features`` does not depend on ``g_coeffs``, ``E_full`` is bit-identical across
drivers, and ``recover`` starts every fit from zeros. Changing the driver moves where
c_true sits; it does not move the basis J probes. And cond(J) is built from singular
*values*, so it cannot depend on where in that basis the truth lies -- driver dependence
lives in the singular *vectors*, which is why the unknown-form studies separate and this
one does not.

Whitened by default since 2026-08-20 (``whiten=True``). Each observed channel's residual is
divided by that channel's climatological standard deviation. This matters here more than
anywhere: the 36 channels differ in std by 146x, and ``E_full`` has zero rows in psi_a and
psi_o, so unwhitened, psi_a carries 97.3% of the least-squares cost while the 18 temperature
channels that actually carry the injected error carry 1.9% between them. Whitening does not
move the noiseless minimiser -- the residual is exactly zero at c_true under any positive
weighting -- but it does change the geometry the SVD sees, so the numbers quoted above will
not reproduce. Pass ``--set whiten=false`` to recover them.

Whitening also changes the noise result, and this is where it earns its keep. Unwhitened, at
eta=1e-4 only the mixed error and the two temperature drivers ever got below 10% relative
error; the constant and psi_a1 errors plateaued around 12%. Whitened, all five are below 10%
by 7.2 d and below 1% by 57.4 d. That is the 1.9%-of-the-cost problem being fixed: with the
temperature channels no longer buried, the noisy fit sees the signal.

Noiselessly the fit is exact (rel_err <= 3.5e-9, typically ~1e-10) from 1.79 d out to 89.7 d.
It fails only at the two shortest windows, 0.45 and 0.90 d, where cond(J_w) is 1e6 to 1e8 --
genuine under-determination, not an LM budget that ran short, since the same 40 iterations
reach machine precision everywhere to the right of it.

Two things to hold in mind when reading the figure.

The mixed driver's ``rel_err`` is a joint norm over three nonzero coefficients, so it is not
on the same footing as a single-driver one: it divides by ||c_true|| = 0.1*sqrt(3) against
0.1 for the singles, a free 1.73x advantage on the relative scale. An earlier version of this
docstring read "mixed is the easiest of the five to recover" off exactly that artefact.
Retracted 2026-08-20: converted back to absolute error at the 28.7 d window, the six samples
*within* each driver span 18x while the medians differ by 3x, so the ordering is inside the
sampling noise -- as the shared-J argument above says it should be. ``experiments/oracle.py``
separately attributes a ~20% low bias in the noiseless mixed ocean coefficient to
conditioning rather than to anything physical.

And the truth segments overlap at long windows. ``build_segments`` spaces initial conditions
by ``cfg.ic_spacing_steps = 3000`` while the sweep runs out to 8000 steps: overlap is 0%
through 2560 steps, 22% at 3840, 41% at 5120 and 62.5% at 8000. The n_ic spread at the long
end is therefore not an independent sample. Not fixed here -- the fix is a coordinated re-run
across the four studies that call ``build_segments`` (see ``notes/TODO.md``).

Results -> ``results/driver_window_<tag>.npz``.
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

# g_coeffs indexes TwinModels.features(x) = (1, theta_a1, T_o1, psi_a1). "const" carries
# no state dependence at all -- the error is just eps*(E x) -- which makes it the baseline
# here and, in the form-discovery half of the project, the alias every other form collapses
# onto. "mixed" matches oracle.py's G_MIXED.
DRIVERS = {
    "const":    (1.0, 0.0, 0.0, 0.0),
    "theta_a1": (0.0, 1.0, 0.0, 0.0),
    "T_o1":     (0.0, 0.0, 1.0, 0.0),
    "psi_a1":   (0.0, 0.0, 0.0, 1.0),
    "mixed":    (0.0, 1.0, 1.0, 1.0),
}
ALL_DRIVERS = tuple(DRIVERS)


def run(tag="main", drivers=ALL_DRIVERS, n_ic=3, n_seeds=2, lm_iters=40,
        noise_levels=(1e-4, 1e-3), whiten=True, verbose=True):
    """Sweep the optimisation window once per driver, noiseless and under noise.

    ``n_ic`` and ``n_seeds`` are trimmed relative to the lead-time study (4 and 3) because
    this runs the whole sweep five times over; the window set and the noise windows are
    the lead-time defaults, so the curves are directly comparable to ``leadtime_main_whiten``.

    ``whiten`` weights each observed channel's residual by 1/std of that channel (see the
    module docstring for why it is on by default here). It does not change the output
    filename, so a run with ``--set whiten=false`` under the default tag overwrites the
    whitened result -- give it a tag of its own. The flag is stored in the .npz either way.

    Every driver shares the base model, so all five read the same cached attractor IC --
    ``get_attractor_ic`` keys on ndim, spin-up length, seed and dt, none of which vary
    with ``g_coeffs``. Nothing here regenerates that cache.
    """
    t0 = time.time()
    cfg = ExperimentConfig()
    cfg.n_ic = n_ic
    dt = cfg.dt
    windows = cfg.window_steps
    nwins = cfg.noise_windows
    nD, nW, nEta, nWn = len(drivers), len(windows), len(noise_levels), len(nwins)

    rel = np.full((nD, nW, n_ic), np.nan)
    cond_J = np.full((nD, nW, n_ic), np.nan)
    smin = np.full((nD, nW, n_ic), np.nan)
    smax = np.full((nD, nW, n_ic), np.nan)
    jrank = np.full((nD, nW, n_ic), np.nan)
    c_hats = np.full((nD, nW, n_ic, 4), np.nan)
    rel_noise = np.full((nD, nEta, nWn, n_ic, n_seeds), np.nan)
    c_trues = np.full((nD, 4), np.nan)

    for idr, name in enumerate(drivers):
        g = DRIVERS[name]
        tw = TwinModels(cfg.eps, g)
        mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
        x_attr = get_attractor_ic(tw, cfg, verbose=False)
        seg_ics, seg_trajs = build_segments(tw, cfg, x_attr)
        c_trues[idr] = np.asarray(tw.c_true)
        td = time.time()

        for iw, w in enumerate(windows):
            for k in range(n_ic):
                r = recover(tw, seg_ics[k], seg_trajs[k][:w], dt, mask,
                            eta=0.0, n_iters=lm_iters, whiten=whiten)
                rel[idr, iw, k] = r["rel_err"]
                cond_J[idr, iw, k] = r["cond_J"]
                smin[idr, iw, k] = r["smin"]
                smax[idr, iw, k] = r["smax"]
                jrank[idr, iw, k] = r["rank"]
                c_hats[idr, iw, k] = r["c_hat"]

        for ie, eta in enumerate(noise_levels):
            for jw, w in enumerate(nwins):
                for k in range(n_ic):
                    for s in range(n_seeds):
                        # unique, reproducible noise stream per (driver, eta, window, IC, seed)
                        ns = cfg.seed + ((((idr * nEta + ie) * nWn + jw) * n_ic + k)
                                         * n_seeds + s)
                        r = recover(tw, seg_ics[k], seg_trajs[k][:w], dt, mask,
                                    eta=eta, n_iters=lm_iters, noise_seed=ns,
                                    whiten=whiten)
                        rel_noise[idr, ie, jw, k, s] = r["rel_err"]

        if verbose:
            cj = np.nanmedian(cond_J[idr], axis=1)
            print(f"[{name:8s}] g={g}  best cond(J)={cj.min():.2e} at "
                  f"{steps_to_days(windows[int(np.argmin(cj))]):.1f} d   "
                  f"noiseless rel_err med={np.nanmedian(rel[idr]):.2e}   "
                  f"[{time.time() - td:.0f}s]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"driver_window_{tag}.npz")
    np.savez(
        out,
        drivers=np.asarray(list(drivers)),
        g_coeffs=np.asarray([DRIVERS[n] for n in drivers]),
        c_true=c_trues,
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
    return np.load(os.path.join(RESULTS_DIR, f"driver_window_{tag}.npz"), allow_pickle=True)


def figure(tag="main", eta_index=0):
    """The conditioning U-curve and the usable window, one curve per driver.

    (a) is cond(J) against window: the U whose left arm is under-determination (the drivers
    have not varied enough to separate four coefficients) and whose right arm is anisotropy
    under chaotic stretching. The five curves lying on top of each other is the result, not a
    redundancy. (b) is the recovery error under observation noise, which is where a *usable*
    window actually appears and where the drivers genuinely differ; noiseless, the fit is
    exact at every window because the model class contains the truth.

    The axis label follows the ``whiten`` flag stored in the .npz, because the weighting
    changes the curve and has to be visible on the figure rather than assumed.
    """
    plt = figstyle.pyplot()
    d = _load(tag)
    names = [str(n) for n in d["drivers"]]
    wd, nwd = d["window_days"], d["noise_window_days"]
    etas = d["noise_levels"]
    wh = bool(d["whiten"]) if "whiten" in d.files else False
    jsym = r"J_w" if wh else "J"
    wnote = "whitened residual (1/std per channel)" if wh else "unwhitened residual"

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    colors = plt.get_cmap("tab10").colors

    for i, nm in enumerate(names):
        cj = np.nanmedian(d["cond_J"][i], axis=1)
        ax.plot(wd, cj, "o-", ms=4, color=colors[i], label=nm)
        ax.plot(wd[int(np.argmin(cj))], cj.min(), "*", ms=13, color=colors[i])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("optimisation window  [days]")
    ax.set_ylabel(rf"cond$({jsym})=\sigma_{{\max}}/\sigma_{{\min}}$")
    ax.set_title("(a)  conditioning vs window, by error driver\n"
                 rf"{wnote}; $\star$ = each driver's best-conditioned window",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, title="driver $g(x)$", title_fontsize=8)

    for i, nm in enumerate(names):
        m = np.nanmedian(d["rel_err_noise"][i, eta_index], axis=(1, 2))
        axb.plot(nwd, m, "s-", ms=5, color=colors[i], label=nm)
    axb.axhline(0.1, ls=":", color="grey", lw=1)
    axb.text(nwd[0], 0.105, "10%", color="grey", fontsize=8, va="bottom", ha="left",
             bbox=LBL_BBOX)
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel("optimisation window  [days]")
    axb.set_ylabel(r"$\|\hat c - c_{\rm true}\|/\|c_{\rm true}\|$")
    axb.set_title(rf"(b)  recovery under noise ($\eta={etas[eta_index]:.0e}$)"
                  "\nmixed is a joint norm over three coefficients",
                  fontsize=9.5, loc="left")
    axb.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"fig_driver_window_{tag}.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return [out]
