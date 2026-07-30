"""
error_discovery.plotting
========================

Manuscript figures, generated from ``leadtime_<tag>.npz``.  Written to ``results/``.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from .config import RESULTS_DIR

# A faint white backing so an annotation that must sit near a curve interrupts the
# line cleanly instead of being scrawled over it.
_LBL_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0)


def _load(tag):
    return np.load(os.path.join(RESULTS_DIR, f"leadtime_{tag}.npz"), allow_pickle=True)


def fig_recovery_and_conditioning(tag="main"):
    """Coefficient recovery vs window, with the conditioning reported as the SVD
    condition number of the residual Jacobian cond(J)=sigma_max/sigma_min (not the
    squared eigenvalue ratio of J^T J). The noiseless fit is exact at every window
    (the model class contains the truth), so there is no noiseless discoverability
    window; what is U-shaped is cond(J), and the real window appears only under
    observation noise, which the conditioning amplifies. Panel (b) shows the source of
    the U-shape -- the singular values sigma_min, sigma_max of J vs window: at short
    windows sigma_min is tiny (drivers barely varied), at long windows sigma_max grows
    faster than sigma_min under chaotic stretching. J stays full rank 4/4 at every window,
    so the long-window deterioration is anisotropy, not a rank loss."""
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
            ha="left", bbox=_LBL_BBOX)
    ax.axvline(w_opt, ls="--", color="C2", lw=1.2)
    # label rides the top of the vline (an empty band), hanging down, masked from the curves
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(w_opt, 0.98, f"best conditioning ~{w_opt:.0f} d ", color="C2",
            fontsize=7.5, rotation=90, va="top", ha="right", transform=trans,
            bbox=_LBL_BBOX)
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
        rank = np.nanmedian(d["jrank"], axis=1)
        axb.plot(wd, smax, "^-", color="C4", ms=4, label=r"$\sigma_{\max}(J)$")
        axb.plot(wd, smin, "v-", color="C3", ms=4, label=r"$\sigma_{\min}(J)$")
        axb.axvline(w_opt, ls="--", color="C2", lw=1.2)
        axb.set_xscale("log"); axb.set_yscale("log")
        axb.set_xlabel("optimisation window  [days]")
        axb.set_ylabel("singular values of $J$")
        axb.set_title(rf"(b)  rank {int(np.nanmin(rank))}/4 throughout; "
                      r"long-window loss $=\sigma_{\max}$ growth",
                      fontsize=9.5, loc="left")
        axb.legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"fig_recovery_conditioning_{tag}.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


def fig_conditioning_only(tag="main_whiten"):
    """Noiseless recovery error and conditioning vs window (no noisy-recovery overlay).
    Panel (a) is the noiseless recovery error together with cond(J)=sigma_max/sigma_min
    on a twin axis; panel (b) is the same sigma_min/sigma_max decomposition as
    fig_recovery_and_conditioning's panel (b)."""
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
    ax.set_title("(a)  noiseless recovery error and conditioning", fontsize=9.5, loc="left")

    ax2 = ax.twinx()
    ax2.plot(wd, cond_J, "D--", color="C3", alpha=0.6, label=r"cond$(J)$")
    ax2.axvline(w_opt, ls="--", color="C2", lw=1.2)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(w_opt, 0.98, f"best conditioning ~{w_opt:.0f} d ", color="C2",
            fontsize=7.5, rotation=90, va="top", ha="right", transform=trans,
            bbox=_LBL_BBOX)
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
        rank = np.nanmedian(d["jrank"], axis=1)
        axb.plot(wd, smax, "^-", color="C4", ms=4, label=r"$\sigma_{\max}(J)$")
        axb.plot(wd, smin, "v-", color="C3", ms=4, label=r"$\sigma_{\min}(J)$")
        axb.axvline(w_opt, ls="--", color="C2", lw=1.2)
        axb.set_xscale("log"); axb.set_yscale("log")
        axb.set_xlabel("optimisation window  [days]")
        axb.set_ylabel("singular values of $J$")
        axb.set_title(rf"(b)  rank {int(np.nanmin(rank))}/4 throughout; "
                      r"long-window loss $=\sigma_{\max}$ growth",
                      fontsize=9.5, loc="left")
        axb.legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"fig_conditioning_only_{tag}.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


def fig_coeff_recovery(tag="main"):
    """Per-coefficient recovery vs window (noiseless), against truth."""
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
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"fig_coeff_recovery_{tag}.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


def fig_noise(tag="main"):
    """Recovery error vs window for several observation-noise levels."""
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
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"fig_noise_{tag}.png")
    fig.savefig(out, dpi=160); plt.close(fig)
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


def make_all(tag="main"):
    figs = [fig_recovery_and_conditioning(tag), fig_coeff_recovery(tag), fig_noise(tag)]
    summarize(tag)
    return figs


# --------------------------------------------------------------------------- #
# Partial-observation (sparsity) study
# --------------------------------------------------------------------------- #
def _load_sparsity(tag):
    return np.load(os.path.join(RESULTS_DIR, f"sparsity_{tag}.npz"), allow_pickle=True)


_OBS_LABEL = {"full": "full (36)", "atmos": "atmosphere (20)",
              "ocean": "ocean (16)", "temp": "temperatures (18)"}


def fig_sparsity_noise(tag="sparsity"):
    """Recovery error vs observation noise, one curve per observation operator."""
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
            va="bottom", bbox=_LBL_BBOX)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1.5e-3, 3.0)            # focus on the data; oracle floor is far below (annotated)
    ax.set_xlabel(r"observation noise $\eta$ (relative; leftmost = noiseless)")
    ax.set_ylabel(r"$\|\hat c - c_{\rm true}\|/\|c_{\rm true}\|$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.985, 0.02, f"oracle (exact tendency) ≈ {floor:.0e} — off scale below",
            transform=ax.transAxes, fontsize=7.5, style="italic", color="0.4", ha="right")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"fig_sparsity_noise_{tag}.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


def fig_sparsity_window(tag="sparsity"):
    """Recovery error vs window per observation operator at fixed noise."""
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
    out = os.path.join(RESULTS_DIR, f"fig_sparsity_window_{tag}.png")
    fig.savefig(out, dpi=160); plt.close(fig)
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


def make_sparsity(tag="sparsity"):
    figs = [fig_sparsity_noise(tag), fig_sparsity_window(tag)]
    summarize_sparsity(tag)
    return figs
