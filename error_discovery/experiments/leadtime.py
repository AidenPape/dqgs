"""
error_discovery.experiments.leadtime
====================================

The lead-time error-discovery sweep — the headline study, and the default flag.

Pipeline (fully reproducible from a seed; the expensive attractor spin-up is
cached to ``results/attractor_ic_default_sp800000.npz`` and reused):

1. Load (or build + cache) a spun-up base-model attractor state.
2. Build one long truth trajectory with the perturbed model and slice ``n_ic``
   disjoint segments out of it, spaced two windows apart so none share a step.
   Disjoint, not independent: the ocean state stays correlated across them.
3. Estimate the base model's Lyapunov time (context; this config is only weakly
   chaotic).
4. Noiseless sweep: for each window and segment, recover the error coefficients
   with Levenberg-Marquardt and record recovery error and cond(J^T J).
5. Noise sweep: for a subset of windows, repeat with observation noise.

Results -> ``results/leadtime_<tag>.npz``.  The headline is the *window of
discoverability*. The 2026-08-27 disjoint-segment re-run left its lower half
intact and substantially weakened its upper half.

**Measured whitened, 16 windows out to 269 d.** The lower edge is real and
sharp: 0.45 d is the only window where the noiseless fit misses, at rel_err
9.5e-4 with cond(J_w) = 2.1e7, and every window from 0.90 d out sits at
1e-10 to 5e-9.

The upper edge is now visible only in the conditioning, and faintly. cond(J_w)
bottoms at 7.3e4 (89.7 d) and climbs to just 1.2e5-1.4e5 by 269 d -- a factor of
1.6-1.9. On the old overlapping segments the same curve bottomed at 4.9e3
(21.5 d) and rose 7.7x to 3.7e4, so **both the depth of that U and the position
of its floor were largely artefacts of segments that shared data.** The whole
curve also sits 5-40x higher now.

**Noiseless recovery never degrades at the long end** in either run: rel_err
stays ~1e-10 out to 269 d. "Loss of identifiability" is a statement about
cond(J_w), not about achieved accuracy.

**And the noise sweep cannot test the upper edge at all.** Its windows stop at
89.7 d, which is now precisely where cond(J_w) bottoms out, so all six sit on
the lower arm. At eta=1e-4 recovery improves monotonically across them (0.186 at
3.6 d to 0.0095 at 89.7 d) with no upper turn to see. Extending
``EXTRA_NOISE_WINDOWS`` past 89.7 d is what would test the claim; until then the
"noise tightens the upper edge" half of the headline is unsupported here.

Removing the overlap also loosened the noisy fits at the long end by ~2x
(89.7 d, eta=1e-4: 0.0056 before, 0.0095 after), which is the expected direction
-- the old segments shared up to 62.5% of their data.
"""

import os
import time

import numpy as np

from matplotlib.transforms import blended_transform_factory

from .. import figstyle
from ..attractor import get_attractor_ic, build_segments, segments_from_states
from ..config import (ExperimentConfig, RESULTS_DIR, steps_to_days, N_SAMPLES, SPACING_K,
                      MIN_EPISODE_DAYS, ETA_GRID)
from ..figstyle import DPI, LBL_BBOX
from ..models import TwinModels, obs_mask
from ..discovery import recover, lyapunov_time
from ..sampling import (starting_states, sample_record, noise_scale, shared_noise,
                        sampling_line)


# This study sweeps further than the shared config does. The right arm of the U only resolves
# past ~90 d, and the manuscript figures are drawn out to 269 d -- these are the windows the
# pre-repo results behind those figures used. The other studies keep config's 13, deliberately:
# ``build_segments`` derives its segment spacing from the longest window, so extending them
# globally would triple every study's truth rollout for a range only this study reads.
EXTRA_WINDOWS = (11_000, 16_000, 24_000)
EXTRA_NOISE_WINDOWS = (8_000,)


def default_config(tag="main"):
    """``ExperimentConfig`` with this study's own longer sweep. ``run_all`` picks this up."""
    cfg = ExperimentConfig()
    cfg.tag = tag
    cfg.window_steps = tuple(cfg.window_steps) + EXTRA_WINDOWS
    cfg.noise_windows = tuple(cfg.noise_windows) + EXTRA_NOISE_WINDOWS
    return cfg


def run(cfg: ExperimentConfig = None, verbose=True, whiten=False,
        sampling="legacy", n_samples=N_SAMPLES, spacing_k=SPACING_K,
        min_episode_days=MIN_EPISODE_DAYS, placement="onset", discard=0, eta_grid=ETA_GRID):
    """Run the lead-time sweep. With ``whiten=True`` every fit (noiseless and noisy)
    divides each observed channel's residual by that channel's climatological std, so the
    reported conditioning is that of the properly-weighted Jacobian J_w = W J and the
    noisy recovery is the weighted least-squares estimate (reviewer #6/#7). The recovered
    coefficients are unchanged in the noiseless limit (the twin truth is a zero-residual
    minimizer under any positive weighting); only the geometry the SVD sees changes.

    ``sampling="legacy"`` is the original design. ``"passive"``, ``"active"`` or ``"both"``
    follow the sampling standard (``plans/sampling.md``): ``n_samples`` starting states from
    the long base run in place of ``cfg.n_ic`` segments; the climatological noise scale; one
    standard-normal draw per (state, seed) reused for every window and level; levels
    ``eta_grid`` in place of ``cfg.noise_levels``; and the Lyapunov time computed from every
    starting state (``lam_max_per_state``, median under ``lam_max``). A sampled run left on
    the default tag is written under the mode's name instead."""
    if cfg is None:
        cfg = default_config()
    t0 = time.time()
    tag = cfg.tag

    tw = TwinModels(cfg.eps, cfg.g_coeffs)
    vr = tw.spec.variables_range
    mask = obs_mask(tw.ndim, vr, cfg.obs)
    if verbose:
        print(f"[setup] ndim={tw.ndim} eps={cfg.eps} g_coeffs={cfg.g_coeffs} "
              f"obs={cfg.obs} c_true={np.asarray(tw.c_true)}")

    windows = cfg.window_steps
    picks, scale, etas = {}, None, cfg.noise_levels
    if sampling == "legacy":
        x_attr = get_attractor_ic(tw, cfg, verbose)
        seg_ics, seg_trajs = build_segments(tw, cfg, x_attr)
        lam, tau, tau_days = lyapunov_time(tw, x_attr, cfg.dt, n_blocks=cfg.lyap_blocks)
    else:
        tag = sampling if tag == "main" else tag
        picks = starting_states(sampling, n_samples, spacing_k, min_episode_days, placement,
                                window_steps=discard + max(windows))
        seg_ics, seg_trajs = segments_from_states(tw, picks["states"], max(windows), cfg.dt,
                                                  discard)
        scale, etas = noise_scale(), eta_grid
        noise = [[shared_noise(i, s, max(cfg.noise_windows), tw.ndim)
                  for s in range(cfg.n_noise_seeds)] for i in picks["snapshot_index"]]
        lyap = np.asarray([lyapunov_time(tw, x, cfg.dt, n_blocks=cfg.lyap_blocks)
                           for x in seg_ics])
        lam, tau_days = float(np.median(lyap[:, 0])), float(np.median(lyap[:, 2]))
    if verbose:
        print(f"[lyap]  lambda_max={lam:.4g}/nondim  tau_lyap={tau_days:.1f} days "
              f"(weakly chaotic; context only)")

    nW, nI = len(windows), (cfg.n_ic if sampling == "legacy" else n_samples)

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
                        eta=0.0, n_iters=cfg.lm_iters, whiten=whiten, scale=scale)
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
    nWn = len(cfg.noise_windows)
    rel_noise = np.full((len(etas), nWn, nI, cfg.n_noise_seeds), np.nan)
    for ie, eta in enumerate(etas):
        for jw, w in enumerate(cfg.noise_windows):
            for k in range(nI):
                for s in range(cfg.n_noise_seeds):
                    if sampling == "legacy":
                        # unique, reproducible noise stream per (eta, window, segment, seed)
                        ns = cfg.seed + (((ie * nWn + jw) * nI + k) * cfg.n_noise_seeds + s)
                        r = recover(tw, seg_ics[k], seg_trajs[k][:w], cfg.dt, mask,
                                    eta=eta, n_iters=cfg.lm_iters, noise_seed=ns,
                                    whiten=whiten)
                    else:
                        r = recover(tw, seg_ics[k], seg_trajs[k][:w], cfg.dt, mask,
                                    eta=eta, n_iters=cfg.lm_iters, whiten=whiten,
                                    scale=scale, noise=noise[k][s])
                    rel_noise[ie, jw, k, s] = r["rel_err"]
        if verbose:
            print(f"[noise] eta={eta:.0e}  rel_err med over windows="
                  f"{np.nanmedian(rel_noise[ie], axis=(1, 2))}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"leadtime_{tag}.npz")
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
        dt=cfg.dt, seed=cfg.seed, sampling=sampling, **sample_record(picks),
        **({} if sampling == "legacy" else {"lam_max_per_state": lyap[:, 0],
                                            "tau_lyap_days_per_state": lyap[:, 2]}),
    )
    if verbose:
        print(f"[done]  wrote {out}  ({time.time() - t0:.1f}s total)")
    return out

# --------------------------------------------------------------------------- #
# Figures. Each reloads the .npz above; none of them computes.
# --------------------------------------------------------------------------- #
def _load(tag):
    return np.load(os.path.join(RESULTS_DIR, f"leadtime_{tag}.npz"), allow_pickle=True)


def fig_recovery_and_conditioning(tag="main"):
    """Coefficient recovery vs window, with the conditioning reported as the SVD
    condition number of the residual Jacobian cond(J)=sigma_max/sigma_min (not the
    squared eigenvalue ratio of J^T J). The noiseless fit is exact at every window
    (the model class contains the truth), so there is no noiseless discoverability
    window; what is U-shaped is cond(J), and the real window appears only under
    observation noise, which the conditioning amplifies. Panel (b) shows the singular
    values sigma_min, sigma_max of J vs window. At short windows sigma_min is tiny
    (drivers barely varied), which is the whole lower arm. J stays full rank 4/4 at
    every window, so nothing here is a rank loss.

    **The long-window half of that reading is withdrawn, 2026-08-27.** It claimed
    sigma_max grows faster than sigma_min under chaotic stretching. Measured on disjoint
    segments, from the cond(J_w) minimum at 89.7 d out to 269 d, sigma_max grows 4.5x and
    sigma_min grows 6.9x -- sigma_min grows *faster*, so the ratio of the two plotted
    medians falls (6.7e4 to 4.4e4) over exactly the stretch where panel (a)'s cond curve
    rises (7.3e4 to 1.4e5).

    Those two statements are not contradictory, they are two different summaries: panel
    (a) plots the median of the per-segment ratio, panel (b) plots the ratio of two
    separately-taken medians, and with segments this dispersed they diverge. The
    per-segment median is the one to trust. Panel (b) is still the right picture of where
    each singular value goes; it is not a decomposition of panel (a)'s curve, and should
    not be read as one."""
    plt = figstyle.pyplot()
    d = _load(tag)
    wd = d["window_days"]
    rel = d["rel_err"]                       # (nW, nI) noiseless
    # cond(J) and singular values (fall back to sqrt(cond(J^T J)) for legacy files)
    cond_J = np.nanmedian(d["cond_J"], axis=1) if "cond_J" in d \
        else np.sqrt(np.nanmedian(d["cond"], axis=1))
    med = np.nanmedian(rel, axis=1)
    w_opt = wd[int(np.nanargmin(cond_J))]    # conditioning sweet spot

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.4, 4.4))

    # ---- (a) recovery error + cond(J) ---------------------------------- #
    ax.plot(wd, np.maximum(med, 1e-13), "o-", color="C0", ms=4,
            label="recovery error (noiseless)")
    nwd = d["noise_window_days"]
    etas = d["noise_levels"]
    reln = d["rel_err_noise"]                 # (nEta, nWn, nI, nSeed)
    ie0 = 0                                   # one representative noise level
    m = np.nanmedian(reln[ie0], axis=(1, 2))
    ax.plot(nwd, m, "s-", color="C1", ms=5,
            label=fr"recovery error ($\eta={etas[ie0]:.0e}$)")
    ax.axhline(0.1, ls=":", color="grey", lw=1)
    ax.text(wd[0], 0.105, "10%", color="grey", fontsize=8, va="bottom",
            ha="left", bbox=LBL_BBOX)
    ax.axvline(w_opt, ls="--", color="C2", lw=1.2)
    # label rides the top of the vline (an empty band), hanging down, masked from the curves
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(w_opt, 0.98, f"best conditioning ~{w_opt:.0f} d ", color="C2",
            fontsize=7.5, rotation=90, va="top", ha="right", transform=trans,
            bbox=LBL_BBOX)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("optimisation window  [days]")
    ax.set_ylabel(r"$\|\hat c - c_{\rm true}\|/\|c_{\rm true}\|$")
    ax.set_title("(a)  recovery error and conditioning", fontsize=9.5, loc="left")

    ax2 = ax.twinx()
    ax2.plot(wd, cond_J, "D--", color="C3", alpha=0.6, label=r"cond$(J)$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"cond$(J)=\sigma_{\max}/\sigma_{\min}$", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    # lower-left is the empty corner (the recovery error bottoms out near the centre)
    ax.legend(l1 + l2, lab1 + lab2, frameon=False, fontsize=8, loc="lower left")

    # ---- (b) the source of the U-shape: sigma_min, sigma_max vs window --- #
    if "smin" in d:
        smin = np.nanmedian(d["smin"], axis=1)
        smax = np.nanmedian(d["smax"], axis=1)
        axb.plot(wd, smax, "^-", color="C4", ms=4, label=r"$\sigma_{\max}(J)$")
        axb.plot(wd, smin, "v-", color="C3", ms=4, label=r"$\sigma_{\min}(J)$")
        axb.axvline(w_opt, ls="--", color="C2", lw=1.2)
        axb.set_xscale("log"); axb.set_yscale("log")
        axb.set_xlabel("optimisation window  [days]")
        axb.set_ylabel("singular values of $J$")
        axb.set_title("(b)  median singular values of $J$",
                      fontsize=9.5, loc="left")
        axb.legend(frameon=False, fontsize=8, loc="center left")
    figstyle.finish(fig, sampling_line(d))
    out = os.path.join(RESULTS_DIR, f"fig_recovery_conditioning_{tag}.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out


def fig_conditioning_only(tag="main_whiten"):
    """Noiseless recovery error and conditioning vs window (no noisy-recovery overlay).
    Panel (a) is the noiseless recovery error together with cond(J)=sigma_max/sigma_min
    on a twin axis; panel (b) is the same sigma_min/sigma_max decomposition as
    fig_recovery_and_conditioning's panel (b)."""
    plt = figstyle.pyplot()
    d = _load(tag)
    wd = d["window_days"]
    rel = d["rel_err"]                       # (nW, nI) noiseless
    cond_J = np.nanmedian(d["cond_J"], axis=1) if "cond_J" in d \
        else np.sqrt(np.nanmedian(d["cond"], axis=1))
    med = np.nanmedian(rel, axis=1)
    w_opt = wd[int(np.nanargmin(cond_J))]

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.4, 4.4))

    # ---- (a) noiseless recovery error + cond(J) -------------------------- #
    ax.plot(wd, np.maximum(med, 1e-13), "o-", color="C0", ms=4,
            label="recovery error (noiseless)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("optimisation window  [days]")
    ax.set_ylabel(r"$\|\hat c - c_{\rm true}\|/\|c_{\rm true}\|$")
    ax.set_title("(a)  median noiseless recovery error and conditioning", fontsize=9.5,
                 loc="left")

    ax2 = ax.twinx()
    ax2.plot(wd, cond_J, "D--", color="C3", alpha=0.6, label=r"cond$(J)$")
    ax2.axvline(w_opt, ls="--", color="C2", lw=1.2)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(w_opt, 0.98, f"best conditioning ~{w_opt:.0f} d ", color="C2",
            fontsize=7.5, rotation=90, va="top", ha="right", transform=trans,
            bbox=LBL_BBOX)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"cond$(J)=\sigma_{\max}/\sigma_{\min}$", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, frameon=False, fontsize=8, loc="lower left")

    # ---- (b) the source of the U-shape: sigma_min, sigma_max vs window --- #
    if "smin" in d:
        smin = np.nanmedian(d["smin"], axis=1)
        smax = np.nanmedian(d["smax"], axis=1)
        axb.plot(wd, smax, "^-", color="C4", ms=4, label=r"$\sigma_{\max}(J)$")
        axb.plot(wd, smin, "v-", color="C3", ms=4, label=r"$\sigma_{\min}(J)$")
        axb.axvline(w_opt, ls="--", color="C2", lw=1.2)
        axb.set_xscale("log"); axb.set_yscale("log")
        axb.set_xlabel("optimisation window  [days]")
        axb.set_ylabel("singular values of $J$")
        axb.set_title("(b)  median singular values of $J$",
                      fontsize=9.5, loc="left")
        axb.legend(frameon=False, fontsize=8, loc="center left")
    figstyle.finish(fig, sampling_line(d))
    out = os.path.join(RESULTS_DIR, f"fig_conditioning_only_{tag}.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out


def fig_coeff_recovery(tag="main"):
    """Per-coefficient recovery vs window (noiseless), against truth."""
    plt = figstyle.pyplot()
    d = _load(tag)
    wd = d["window_days"]; c_hats = d["c_hats"]; c_true = d["c_true"]
    names = [r"$c_0$ (const)", r"$c_1\,\theta_{a,1}$", r"$c_2\,T_{o,1}$", r"$c_3\,\psi_{a,1}$"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        med = np.nanmedian(c_hats[:, :, j], axis=1)
        lo, hi = np.nanpercentile(c_hats[:, :, j], [25, 75], axis=1)
        ax.fill_between(wd, lo, hi, alpha=0.2, color="C0")
        ax.plot(wd, med, "o-", color="C0", label="recovered")
        ax.axhline(c_true[j], ls="--", color="C3", label="true")
        ax.set_xscale("log"); ax.set_title(names[j], fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("optimisation window  [days]")
    figstyle.finish(fig, sampling_line(d))
    out = os.path.join(RESULTS_DIR, f"fig_coeff_recovery_{tag}.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out


def fig_noise(tag="main"):
    """Recovery error vs window for several observation-noise levels."""
    plt = figstyle.pyplot()
    d = _load(tag)
    nwd = d["noise_window_days"]
    etas = d["noise_levels"]
    rel = d["rel_err_noise"]                  # (nEta, nW, nI, nSeed)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    # The noiseless recovery is machine-precision at every window (Fig. window), far
    # below this scale, so it is omitted here; this panel is the noise-limited floor.
    allv = []
    for ie, eta in enumerate(etas):
        med = np.nanmedian(rel[ie], axis=(1, 2))
        ax.plot(nwd, med, "s-", label=fr"$\eta={eta:.0e}$")
        allv.append(med)
    ax.axhline(0.1, ls=":", color="grey", lw=1)
    lo = np.nanmin(np.concatenate(allv))
    ax.set_ylim(min(lo * 0.6, 0.05), np.nanmax(np.concatenate(allv)) * 1.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("optimisation window  [days]")
    ax.set_ylabel(r"$\|\hat c - c_{\rm true}\|/\|c_{\rm true}\|$")
    ax.legend(frameon=False, fontsize=8)
    figstyle.finish(fig, sampling_line(d))
    out = os.path.join(RESULTS_DIR, f"fig_noise_{tag}.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out


def summarize(tag="main"):
    """Print the key numbers behind the figures (for the manuscript / quick read)."""
    d = _load(tag)
    wd = d["window_days"]; rel = np.nanmedian(d["rel_err"], axis=1)
    cond = np.nanmedian(d["cond"], axis=1)
    iopt = int(np.nanargmin(rel))
    print(f"== error-discovery summary [{tag}] ==")
    print(f"eps={float(d['eps'])}  g_coeffs={d['g_coeffs']}  obs={d['obs']}")
    print(f"tau_lyap ~ {float(d['tau_lyap_days']):.0f} days (weakly chaotic; context)")
    print(f"optimal window: {wd[iopt]:.1f} d  (rel_err={rel[iopt]:.2e}, "
          f"cond={cond[iopt]:.1e})")
    print(f"shortest window {wd[0]:.2f} d: rel_err={rel[0]:.2e}, cond={cond[0]:.1e}")
    print(f"longest window  {wd[-1]:.1f} d: rel_err={rel[-1]:.2e}, cond={cond[-1]:.1e}")
    # noiseless upper-edge: first window past the optimum where rel_err exceeds 0.1
    up = [wd[i] for i in range(iopt, len(wd)) if rel[i] > 0.1]
    print(f"noiseless rel_err>0.1 first at: {up[0]:.1f} d" if up else
          "noiseless rel_err stays <0.1 across all windows")
    nwd = d["noise_window_days"]
    for ie, eta in enumerate(d["noise_levels"]):
        m = np.nanmedian(d["rel_err_noise"][ie], axis=(1, 2))
        ok = [nwd[j] for j in range(len(nwd)) if m[j] < 0.1]
        print(f"  eta={float(eta):.0e}: median rel_err over windows={np.array2string(m, precision=2)}"
              f"  (<0.1 at {ok} d)")
    return d


def figure(tag="main"):
    figs = [fig_recovery_and_conditioning(tag), fig_coeff_recovery(tag), fig_noise(tag)]
    summarize(tag)
    return figs
