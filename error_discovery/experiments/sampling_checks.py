"""
error_discovery.experiments.sampling_checks
===========================================

Checks on the long base run that every sampled study draws its starting states from
(``plans/sampling.md``; machinery in ``sampling.py``).

The registry runs three steps: ``base_run`` (integrate and save the record; existing chunks
are skipped), ``label`` (``sampling.label_regimes``: regime labels, tau_int, high-frequency
std) and ``run`` (this module: the numbers behind the three figures). The figures are first
drafts, to be revised after an exploratory pass over the data.

* **Block check.** Is anything still drifting after the 1e9-step spin-up? The record is cut
  into equal consecutive blocks. For each variable, a straight line is fitted to its block
  means against time; ``drift`` is how far that line moves from the first block to the last,
  in units of the variable's own std, and ``drift_se`` its standard error. No trend means no
  transient left. The error bar treats block means as independent, which holds only if a
  block is much longer than tau_int; the figure states both lengths.
* **Choice of T.** The histogram of log(running std of psi_a1) at each candidate window, the
  two fitted mixture components, and the share of the record between the 0.1 and 0.9
  probability cutoffs. The smallest share picks T.
* **Cross-check.** A stretch of the record in the style of Tondeur et al. (2020) Fig. 2:
  psi_a1, its running std, and the 500 hPa geopotential-height difference between
  (pi/n, pi/4) and (pi/n, 3 pi/4) (qgs's example diagnostic), shaded by regime label with
  episode onsets marked. The shading should line up with the quiet and noisy stretches.

**Measured 2026-10-01** on the 10,000-yr record (``main``; one base run, one start state):

* Block check, 10 blocks of 1,000 yr (21.5x the largest tau_int): |drift| / s.e. > 2 for 3 of
  36 variables -- psi_o4 -0.012 std (-2.3 s.e.), psi_o6 +0.014 std (+3.8 s.e.), T_o7 +0.20 std
  (+2.6 s.e.). Largest |drift| 0.20 std (T_o7). Share of each block active 0.655-0.668,
  passive 0.322-0.336, flagged 0.009-0.010.
* tau_int, largest five: T_o5 46.6 +/- 15.6 yr, T_o7 9.3 +/- 1.4, T_o3 8.9 +/- 1.3, psi_o7 7.1
  +/- 0.9, psi_o8 5.9 +/- 0.7. Sokal's window found for all 36. Spacing at k = 3: 139.7 yr.
* Choice of T: share between the cutoffs 0.9% (30 d), 3.1% (100 d), 11.4% (365 d), 35.1%
  (1000 d), so T = 30 d, the smallest candidate. The passive hump of log(running std) is not
  bell-shaped (skewed, with a sharp upper edge near -10); the active hump matches its
  component. At 30 d the 0.1 and 0.9 cutoffs fall in the near-empty gap between the humps.
  Shorter windows, tried 2026-10-02 outside ``REGIME_T_DAYS`` (one-off relabel): 1.26% (10 d),
  1.12% (20 d), so 30 d stays the minimum and the starting states are unchanged. At 10 d, 295
  snapshots (0.008%) have a running std of exactly zero and drop out of the mixture fit.
* Time active 0.663, passive 0.327. 242 episodes of each (L = 1 yr). Duration, passive:
  median 13.0 yr, IQR 12.4-14.6, range 6.8-15.2; active: median 26.3, IQR 22.3-31.8, range
  1.5-33.9.
* Starting states available at the defaults (k = 3, onset, 269-d span): 61 passive, 61 active,
  72 both.
* High-frequency std / full std: psi_a and theta_a 0.17-0.98; psi_o 2e-6 to 4.4e-3; T_o 1e-4
  to 1.2e-2. Smallest high-frequency std 1.7e-11 (psi_o8), so 1/scale spans a factor 8.6e8
  across the 36 variables.

Results -> ``results/sampling_checks_<tag>.npz``.
"""

import os

import numpy as np

from .. import figstyle
from ..config import RESULTS_DIR, BASE_RUN_YEARS, DAYS_PER_YEAR, MIN_EPISODE_DAYS
from ..figstyle import DPI
from ..sampling import (base_run, label_regimes as label, load_base_run, load_regimes,
                        regime_labels, episodes, onsets, available_picks)

# Brajard et al. (2021)'s three key variables: psi_a1, psi_o2 and theta_o2 (state indices).
KEY_VARS = {"psi_a1": 0, "psi_o2": 21, "theta_o2": 29}
FIELDS = (("psi_a", 0, 10), ("theta_a", 10, 20), ("psi_o", 20, 28), ("T_o", 28, 36))
FIELD_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
ACTIVE, PASSIVE, FLAGGED = "#4a3aa7", "#008300", "#9a9a96"


def geopotential_difference_weights():
    """Weights on the ten psi_a coefficients, and the metres per nondimensional unit, of the
    500 hPa geopotential-height difference between (pi/n, pi/4) and (pi/n, 3 pi/4).

    qgs's ``GeopotentialHeightDifferenceDiagnostic`` substitutes n through a symbol that does
    not match the basis's own (plans/sampling.md, "Reference facts"), so the basis functions
    are evaluated here with the basis's own n symbol. At n = 1.5 the weights are
    (2, 0, 0, 0, -4, 0, 0, 0, 4, 0).
    """
    import sympy
    from qgs_jax.spec import build_reference_params
    pars = build_reference_params("default")
    basis = pars.atmospheric_basis
    n_sym = basis.substitutions[0][0]
    n = float(pars.scale_params.n)
    x, y = sympy.symbols("x y")
    p1, p2 = (np.pi / n, np.pi / 4), (np.pi / n, 3 * np.pi / 4)
    w = []
    for fn in basis.functions:
        g = sympy.lambdify((x, y), fn.subs(n_sym, n))
        w.append(float(g(*p1)) - float(g(*p2)))
    metres = float(pars.geopotential_scaling) * float(pars.streamfunction_scaling)
    return np.asarray(w), metres


def run(n_years=BASE_RUN_YEARS, n_blocks=10, cross_start_year=1000, cross_years=200,
        tag="main", verbose=True):
    """Block check, choice-of-T summary and cross-check stretch, from the saved record."""
    reg = load_regimes(n_years)
    X = load_base_run(n_years)
    N, nv = X.shape
    dd = float(reg["snapshot_days"])
    years = np.arange(N) * dd / DAYS_PER_YEAR
    lab = regime_labels(reg["p_active"], float(reg["cutoff"]))
    std = X.std(axis=0)

    # --- block check ---------------------------------------------------------- #
    B = N // n_blocks
    blocks = X[:B * n_blocks].reshape(n_blocks, B, nv)
    block_mean, block_std = blocks.mean(axis=1), blocks.std(axis=1)
    lb = lab[:B * n_blocks].reshape(n_blocks, B)
    block_share = np.stack([(lb == 1).mean(1), (lb == -1).mean(1), (lb == 0).mean(1)], axis=1)
    t_mid = (np.arange(n_blocks) + 0.5) * B * dd / DAYS_PER_YEAR
    tc = t_mid - t_mid.mean()
    slope = (tc @ (block_mean - block_mean.mean(0))) / (tc @ tc)
    resid = block_mean - block_mean.mean(0) - np.outer(tc, slope)
    slope_se = np.sqrt((resid ** 2).sum(0) / (n_blocks - 2) / (tc @ tc))
    span = t_mid[-1] - t_mid[0]
    drift, drift_se = slope * span / std, slope_se * span / std
    block_years = B * dd / DAYS_PER_YEAR
    tau_years = reg["tau_int"] * dd / DAYS_PER_YEAR

    # --- episodes and picks at the default knobs ------------------------------ #
    min_len = int(round(MIN_EPISODE_DAYS / dd))
    eps_ = episodes(lab, min_len)
    ep_label = np.asarray([e[0] for e in eps_])
    ep_years = np.asarray([(e[2] - e[1]) * dd / DAYS_PER_YEAR for e in eps_])
    n_picks = {}
    for mode in ("passive", "active", "both"):
        picks, spacing, _ = available_picks(mode, window_steps=24_000, n_years=n_years)
        n_picks[mode] = len(picks)
    spacing_years = spacing * dd / DAYS_PER_YEAR

    # --- cross-check stretch -------------------------------------------------- #
    i0 = int(round(cross_start_year * DAYS_PER_YEAR / dd))
    i1 = i0 + int(round(cross_years * DAYS_PER_YEAR / dd))
    w, metres = geopotential_difference_weights()
    z = metres * X[i0:i1, :len(w)] @ w
    on_p = np.asarray([i for i in onsets(lab, -1, min_len) if i0 <= i < i1], dtype=int)
    on_a = np.asarray([i for i in onsets(lab, 1, min_len) if i0 <= i < i1], dtype=int)

    out = os.path.join(RESULTS_DIR, f"sampling_checks_{tag}.npz")
    np.savez(
        out, n_years=n_years, snapshot_days=dd, std=std, hf_std=reg["hf_std"],
        tau_int_years=tau_years, tau_err_years=reg["tau_err"] * dd / DAYS_PER_YEAR,
        tau_found=reg["tau_found"],
        n_blocks=n_blocks, block_years=block_years, t_mid=t_mid, block_mean=block_mean,
        block_std=block_std, block_share=block_share, drift=drift, drift_se=drift_se,
        t_days=reg["t_days"], gm_means=reg["gm_means"], gm_stds=reg["gm_stds"],
        gm_weights=reg["gm_weights"], ambiguous_share=reg["ambiguous_share"],
        hist_counts=reg["hist_counts"], hist_edges=reg["hist_edges"], chosen=reg["chosen"],
        cutoff=reg["cutoff"],
        share_active=np.mean(lab == 1), share_passive=np.mean(lab == -1),
        episode_label=ep_label, episode_years=ep_years, min_episode_days=MIN_EPISODE_DAYS,
        n_picks_passive=n_picks["passive"], n_picks_active=n_picks["active"],
        n_picks_both=n_picks["both"], spacing_years=spacing_years,
        cross_years=years[i0:i1], cross_psi_a1=X[i0:i1, 0],
        cross_running_std=reg["running_std"][i0:i1], cross_label=lab[i0:i1],
        cross_geopotential=z, cross_onsets_passive=on_p - i0, cross_onsets_active=on_a - i0,
    )
    if verbose:
        k = int(np.argmax(np.abs(drift) / drift_se))
        print(f"[checks] blocks of {block_years:,.0f} yr; largest |drift|/se "
              f"{abs(drift[k]) / drift_se[k]:.2f} (variable {k}: drift {drift[k]:+.3f} std)")
        print(f"[checks] largest tau_int {tau_years.max():,.2f} yr (variable "
              f"{int(np.argmax(tau_years))}); T = {reg['t_days'][int(reg['chosen'])]:.0f} d")
        print(f"[checks] time active {np.mean(lab == 1):.3f}, passive {np.mean(lab == -1):.3f}; "
              f"episodes: {np.sum(ep_label == 1)} active, {np.sum(ep_label == -1)} passive")
        print(f"[checks] picks at spacing {spacing_years:,.1f} yr: passive "
              f"{n_picks['passive']}, active {n_picks['active']}, both {n_picks['both']}")
        print(f"[checks] wrote {out}")
    return out


def _load(tag):
    return np.load(os.path.join(RESULTS_DIR, f"sampling_checks_{tag}.npz"), allow_pickle=True)


def figure(tag="main"):
    """The three draft figures: block check, choice of T, cross-check."""
    plt = figstyle.pyplot()
    d = _load(tag)
    return [_fig_blocks(plt, d, tag), _fig_T(plt, d, tag), _fig_cross(plt, d, tag)]


def _fig_blocks(plt, d, tag):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    drift, se = d["drift"], d["drift_se"]
    for (name, a, b), c in zip(FIELDS, FIELD_COLORS):
        i = np.arange(a, b)
        ax[0].errorbar(i, drift[a:b], yerr=2 * se[a:b], fmt="o", ms=4, lw=1.5, color=c,
                       label=name)
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_xlabel("state variable index")
    ax[0].set_ylabel("drift, first to last block (std units)")
    ax[0].set_title("(a) drift of each variable, $\\pm$2 s.e.")
    ax[0].legend(frameon=False, fontsize=8)

    t = d["t_mid"]
    field_of = {i: c for (_, a, b), c in zip(FIELDS, FIELD_COLORS) for i in range(a, b)}
    for name, i in KEY_VARS.items():
        m = (d["block_mean"][:, i] - d["block_mean"][:, i].mean()) / d["std"][i]
        ax[1].plot(t, m, "o-", ms=4, lw=1.5, color=field_of[i], label=name)
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].set_xlabel("model years since the 1e9-step state")
    ax[1].set_ylabel("block mean minus record mean (std units)")
    ax[1].set_title("(b) Brajard et al. (2021) key variables")
    ax[1].legend(frameon=False, fontsize=8)

    for j, (name, c) in enumerate((("active", ACTIVE), ("passive", PASSIVE),
                                   ("flagged", FLAGGED))):
        ax[2].plot(t, d["block_share"][:, j], "o-", ms=4, lw=1.5, color=c, label=name)
    ax[2].set_ylim(0, 1)
    ax[2].set_xlabel("model years since the 1e9-step state")
    ax[2].set_ylabel("share of block")
    ax[2].set_title("(c) time in each regime")
    ax[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Block check: {int(d['n_blocks'])} blocks of "
                 f"{float(d['block_years']):,.0f} yr; error bars assume independent block "
                 f"means (largest tau_int {float(np.max(d['tau_int_years'])):,.1f} yr)",
                 fontsize=10)
    out = os.path.join(RESULTS_DIR, f"fig_sampling_blockcheck_{tag}.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def _fig_T(plt, d, tag):
    nT = len(d["t_days"])
    fig, ax = plt.subplots(1, nT, figsize=(4 * nT, 3.6), constrained_layout=True)
    cut = float(d["cutoff"])
    for j in range(nT):
        e, c = d["hist_edges"][j], d["hist_counts"][j]
        width = np.diff(e)
        dens = c / (c.sum() * width)
        ax[j].bar(e[:-1], dens, width=width, align="edge", color="0.8", edgecolor="none")
        xs = np.linspace(e[0], e[-1], 600)
        m, s, w = d["gm_means"][j], d["gm_stds"][j], d["gm_weights"][j]
        pdf = w[:, None] * np.exp(-0.5 * ((xs - m[:, None]) / s[:, None]) ** 2) / (
            s[:, None] * np.sqrt(2 * np.pi))
        ax[j].plot(xs, pdf[0], color=PASSIVE, lw=2, label="passive component")
        ax[j].plot(xs, pdf[1], color=ACTIVE, lw=2, label="active component")
        p = pdf[1] / pdf.sum(0)
        for level in (1 - cut, cut):
            k = np.flatnonzero(np.diff(np.sign(p - level)))
            for kk in k:
                ax[j].axvline(xs[kk], color="0.4", lw=0.8, ls="--")
        mark = "  (chosen)" if j == int(d["chosen"]) else ""
        ax[j].set_title(f"T = {d['t_days'][j]:.0f} d: between cutoffs "
                        f"{100 * d['ambiguous_share'][j]:.1f}%{mark}", fontsize=9)
        ax[j].set_xlabel("log(running std of psi_a1)")
    ax[0].set_ylabel("density")
    ax[0].legend(frameon=False, fontsize=8)
    out = os.path.join(RESULTS_DIR, f"fig_sampling_T_{tag}.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def _fig_cross(plt, d, tag):
    t = d["cross_years"]
    lab = d["cross_label"]
    fig, ax = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True)
    rows = ((d["cross_psi_a1"], "psi_a1"), (d["cross_running_std"], "running std of psi_a1"),
            (d["cross_geopotential"], "geopotential-height\ndifference (m)"))
    for a, (y, name) in zip(ax, rows):
        for r, c in ((1, ACTIVE), (-1, PASSIVE), (0, FLAGGED)):
            a.fill_between(t, 0, 1, where=lab == r, color=c, alpha=0.12, lw=0,
                           transform=a.get_xaxis_transform())
        a.plot(t, y, color="0.15", lw=0.4)
        a.set_ylabel(name)
    ax[1].set_yscale("log")
    for key, c, name in (("cross_onsets_active", ACTIVE, "active onset"),
                         ("cross_onsets_passive", PASSIVE, "passive onset")):
        i = d[key]
        ax[0].plot(t[i], d["cross_psi_a1"][i], "o", ms=8, color=c, mec="white", mew=1.5,
                   label=name)
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    ax[-1].set_xlabel("model years since the 1e9-step state")
    ax[0].set_title("Regime labels against the signals (shading: active, passive, flagged)",
                    fontsize=10)
    out = os.path.join(RESULTS_DIR, f"fig_sampling_crosscheck_{tag}.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out
