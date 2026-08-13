"""
error_discovery.experiments.driver_scan
=======================================

Does the identifiability boundary depend on *which* mode drives the injected
error? The governing law (:func:`experiments.why.governing_law`) says the
form-identifiability margin is set by the *driver's own* |mean|/std on the
sampling distribution, tracking ~min(1, 1/r). The standard experiment drives the
error with ``T_{o,1}`` -- a slow ocean-temperature mode pinned near-constant on
the attractor (|mean|/std ~ 30-40) -- which is exactly why its form is only
identifiable off-attractor.

This module tests the prediction that a *faster* driver, an atmospheric mode that
oscillates through zero so its |mean|/std is O(1), is identifiable ON the
attractor with no transient at all. The base model (hence the attractor
distribution and the exchange operator E) is identical across drivers; only the
scalar ``g`` multiplying ``E x`` changes, so any difference in identifiability is
attributable purely to the driver's |mean|/std -- not to ocean-vs-atmosphere as a
category.

Drivers available in the g_coeffs family ``(const, theta_a1, T_o1, psi_a1)``:
  const     g = 1            degenerate (|mean|/std = inf): never identifiable
  psi_a1    atmos streamfn   fast, |mean|/std ~ O(1)
  theta_a1  atmos temp       fast, |mean|/std ~ O(1)
  T_o1      ocean temp       slow, |mean|/std ~ 30-40 (the standard hard case)

The scoring comes from :mod:`error_discovery.formfit` and is driver-independent:
it depends only on the constant exchange operator E, not on g. Reads the cached
attractor IC read-only; fixed seeds; shipped with tests/test_driver.py.

The other four questions that used to share this file are :mod:`driver_record`
(does more data help?), :mod:`driver_mixed` (several drivers at once),
:mod:`alias` (the mechanism) and :mod:`roadmap` (the schematic).
"""

import os
import numpy as np

from .. import figstyle
from ..models import TwinModels
from ..formfit import (attractor_traj, transient_traj, err_tendency,
                       fit, resid, margin_named)
from ..config import RESULTS_DIR, THRESH, EPS
from ..figstyle import DPI

# name, g_coeffs, true library term, driver state index (None for the constant)
DRIVERS = [
    ("psi_a1\n(atmos streamfn)",  (0.0, 0.0, 0.0, 1.0), "x0",  0),
    ("theta_a1\n(atmos temp)",    (0.0, 1.0, 0.0, 0.0), "x10", 10),
    ("T_o1\n(ocean temp)",        (0.0, 0.0, 1.0, 0.0), "x28", 28),
    ("const\n(g = 1)",            (1.0, 0.0, 0.0, 0.0), "const", None),
]

# The two drivers the tandem figure contrasts, drawn from the same 20k record the
# scan uses, so its |mean|/std matches the scan's exactly.
TANDEM = [(r"$\psi_{a,1}$ — atmospheric circulation", "x0", 0, "C0"),
          (r"$T_{o,1}$ — ocean temperature", "x28", 28, "C3")]


def run(n=4000, reps=20, noise=0.02, seed=0, save=True):
    """For each driver, measure form-identifiability ON the attractor (bootstrap
    samples of the base-model climatology) vs the driver's own |mean|/std, both
    noiseless and at a representative tendency noise, plus the off-attractor margin
    for contrast. The attractor distribution and E are shared across drivers."""
    rng = np.random.default_rng(seed)
    tw0 = TwinModels(eps=EPS, g_coeffs=(0.0, 0.0, 1.0, 0.0))
    Xatt = attractor_traj(tw0, 20000)               # base-model climatology (driver-independent)
    Xoff = transient_traj(tw0, n, seed=42)          # off-attractor contrast
    rows = []
    tandem = {}                                     # full-record fits for the tandem figure
    for name, gc, tname, didx in DRIVERS:
        tw = TwinModels(eps=EPS, g_coeffs=gc)
        matt = err_tendency(tw, Xatt)
        moff = err_tendency(tw, Xoff)
        if tname in (t[1] for t in TANDEM):
            # one extra fit on the whole record (the scan itself bootstraps subsets)
            cf, nmf = fit(tw, Xatt, matt)
            tandem[tname] = dict(R=resid(tw, cf, Xatt, matt), d=Xatt[:, didx],
                                 names=np.asarray(nmf))
        if didx is None:
            r_att = np.inf
        else:
            r_att = float(abs(Xatt[:, didx].mean()) / Xatt[:, didx].std())
        mg0 = np.zeros(reps); ok0 = np.zeros(reps)
        mgN = np.zeros(reps); okN = np.zeros(reps)
        for rep in range(reps):
            ia = rng.integers(0, len(Xatt), n)
            X, m = Xatt[ia], matt[ia]
            c, nm = fit(tw, X, m)
            mg0[rep], ok0[rep] = margin_named(resid(tw, c, X, m), nm, tname)
            e = noise * m.std() * rng.standard_normal(m.shape)
            cN, nmN = fit(tw, X, m + e)
            mgN[rep], okN[rep] = margin_named(resid(tw, cN, X, m + e), nmN, tname)
        coff, nmoff = fit(tw, Xoff, moff)
        mg_off = margin_named(resid(tw, coff, Xoff, moff), nmoff, tname)[0]
        rows.append(dict(
            name=name, true_term=tname, g_coeffs=gc, r_att=r_att,
            margin_att=float(np.median(mg0)), correct_att=float(ok0.mean()),
            margin_att_noisy=float(np.median(mgN)), correct_att_noisy=float(okN.mean()),
            margin_off=mg_off))
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez(os.path.join(RESULTS_DIR, "driver_scan.npz"),
                 names=np.asarray([r["name"].replace("\n", " ") for r in rows]),
                 r_att=np.asarray([r["r_att"] for r in rows]),
                 margin_att=np.asarray([r["margin_att"] for r in rows]),
                 margin_att_noisy=np.asarray([r["margin_att_noisy"] for r in rows]),
                 correct_att=np.asarray([r["correct_att"] for r in rows]),
                 correct_att_noisy=np.asarray([r["correct_att_noisy"] for r in rows]),
                 margin_off=np.asarray([r["margin_off"] for r in rows]),
                 **{f"tandem_{t}_{k}": v for t, sub in tandem.items()
                    for k, v in sub.items()})
    print(f"{'driver':18s} {'|mean|/std':>10s} {'1/r':>7s} "
          f"{'marg_att':>9s} {'corr':>5s} {'marg_noisy':>11s} {'marg_off':>9s}")
    for r in rows:
        inv = 0.0 if not np.isfinite(r["r_att"]) else 1.0 / r["r_att"]
        print(f"{r['name'].replace(chr(10),' '):18s} {r['r_att']:10.3f} {inv:7.3f} "
              f"{r['margin_att']:9.3f} {r['correct_att']:5.2f} "
              f"{r['margin_att_noisy']:11.3f} {r['margin_off']:9.3f}")
    return rows


def scan_figure(rows=None):
    plt = figstyle.pyplot()
    if rows is None:
        d = np.load(os.path.join(RESULTS_DIR, "driver_scan.npz"), allow_pickle=True)
        rows = [dict(name=n.replace(" ", "\n", 1), r_att=ra, margin_att=ma,
                     margin_att_noisy=mn, margin_off=mo)
                for n, ra, ma, mn, mo in zip(d["names"], d["r_att"], d["margin_att"],
                                             d["margin_att_noisy"], d["margin_off"])]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # (a) on-attractor margin per driver, with the 1/r prediction overlaid
    a = ax[0]
    xpos = np.arange(len(rows)); w = 0.6
    vals = [max(r["margin_att"], 1e-4) for r in rows]
    cols = ["C0" if v >= THRESH else "C3" for v in vals]
    a.bar(xpos, vals, w, color=cols, alpha=0.85, zorder=3)
    pred = [1.0 / r["r_att"] if np.isfinite(r["r_att"]) else 1e-4 for r in rows]
    a.plot(xpos, np.maximum(pred, 1e-4), "k_", ms=22, mew=2, zorder=5,
           label=r"$1/(|\mathrm{mean}|/\mathrm{std})$ prediction")
    a.axhline(THRESH, ls=":", color="grey", lw=1)
    a.text(len(rows) - 0.5, THRESH * 1.15, "identifiable", fontsize=7.5,
           color="0.3", ha="right", va="bottom")
    a.text(len(rows) - 0.5, THRESH * 0.85, "unidentifiable", fontsize=7.5,
           color="C3", ha="right", va="top")
    a.set_yscale("log"); a.set_ylim(8e-4, 2.0)
    a.set_xticks(xpos); a.set_xticklabels([r["name"] for r in rows], fontsize=7.5)
    a.set_ylabel("form-identifiability margin\n(on the attractor)")
    a.set_title("(a)  faster drivers are identifiable on the attractor", fontsize=9.5)
    a.legend(fontsize=7.5, frameon=False, loc="lower left")

    # (b) margin vs the driver's |mean|/std: the collapse onto 1/r
    b = ax[1]
    xx = np.logspace(-0.5, 2, 200)
    b.fill_between(xx, np.minimum(1, 1.0 / (3 * xx)), np.minimum(1, 3.0 / xx),
                   color="grey", alpha=0.18, lw=0, label="factor of 3")
    b.plot(xx, np.minimum(1, 1.0 / xx), "k--", lw=1.3, label=r"$1/(|\mathrm{mean}|/\mathrm{std})$")
    b.axhline(THRESH, ls=":", color="grey", lw=1)
    for r in rows:
        if not np.isfinite(r["r_att"]):
            continue
        b.scatter(r["r_att"], max(r["margin_att"], 1e-4), s=70, zorder=5,
                  edgecolor="white", linewidth=0.6,
                  color="C0" if r["margin_att"] >= THRESH else "C3")
        b.annotate(r["name"].split("\n")[0], (r["r_att"], max(r["margin_att"], 1e-4)),
                   xytext=(6, 4), textcoords="offset points", fontsize=7.5)
    b.set_xscale("log"); b.set_yscale("log"); b.set_ylim(8e-4, 2.0)
    b.set_xlabel(r"the driver's own $|\mathrm{mean}|/\mathrm{std}$ on the attractor")
    b.set_ylabel("form-identifiability margin")
    b.set_title("(b)  one controller, atmosphere and ocean alike", fontsize=9.5)
    b.legend(fontsize=7.5, frameon=False, loc="lower left")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_driver_collapse.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out


def tandem_figure(save=True):
    """The headline contrast on the honest diagnostic. Row 1: the driver's
    distribution over the climatology, with the mean's distance from zero measured
    in standard deviations (= |mean|/std). A broadly-exploring atmospheric-circulation
    mode sits a few std from zero; the ocean-temperature mode is a needle pinned ~40
    std away by thermal damping. Row 2: the consequence -- the pinned mode's true
    error form is buried among near-constant aliases, the broad mode's form stands
    alone. Same 20k-step record for both, so |mean|/std matches ``driver_scan``."""
    plt = figstyle.pyplot()
    z = np.load(os.path.join(RESULTS_DIR, "driver_scan.npz"), allow_pickle=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.9))
    jit = np.random.default_rng(0)
    for col, (title, tname, didx, color) in enumerate(TANDEM):
        R = z[f"tandem_{tname}_R"]
        ti = list(z[f"tandem_{tname}_names"]).index(tname)
        d = z[f"tandem_{tname}_d"]
        mu, sd = float(d.mean()), float(d.std())
        rr = abs(mu) / sd

        # ---- row 1: the driver's distribution, |mean|/std as distance from zero ---- #
        a = ax[0, col]
        lo = min(0.0, mu - 4 * sd) - 0.5 * sd
        hi = max(mu + 4 * sd, 0.0) + 0.5 * sd
        a.hist(d, bins=60, range=(lo, hi), color=color, alpha=0.55, density=True)
        a.axvline(0.0, color="0.4", lw=1.4, ls="-")
        a.text(0.0, a.get_ylim()[1] * 0.92, " zero", color="0.4", fontsize=8, va="top")
        a.axvline(mu, color="k", lw=1.3, ls="--")
        a.text(mu, a.get_ylim()[1] * 0.92, " mean", color="k", fontsize=8, va="top")
        a.annotate("", xy=(mu, a.get_ylim()[1] * 0.45), xytext=(0.0, a.get_ylim()[1] * 0.45),
                   arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.1))
        a.text(mu / 2, a.get_ylim()[1] * 0.5,
               rf"$|\mathrm{{mean}}|/\mathrm{{std}} = {rr:.0f}$",
               ha="center", va="bottom", fontsize=10,
               bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        a.set_title(title, fontsize=11.5)
        a.set_xlabel("driver amplitude", fontsize=8.5)
        if col == 0:
            a.set_ylabel("density over the climatology", fontsize=9.5)

        # ---- row 2: truth vs the alias cloud (residual landscape) ---------- #
        b = ax[1, col]
        others = np.delete(R, ti)
        margin = float(others.min() - R[ti])
        y = jit.uniform(-1, 1, len(others)) * 0.32
        b.axvspan(8e-4, THRESH, color="C3", alpha=0.07)
        b.axvline(THRESH, color="C3", ls=":", lw=1)
        b.scatter(np.maximum(others, 1e-3), y, s=20, color="0.55", alpha=0.7,
                  label="wrong (alias) forms", zorder=3)
        b.scatter([max(R[ti], 1e-3)], [0], s=240, marker="*", color=color,
                  edgecolor="k", linewidth=0.6, zorder=5, label="true form")
        b.annotate("", xy=(others.min(), 0.62), xytext=(max(R[ti], 1e-3), 0.62),
                   arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.1))
        b.text(np.sqrt(others.min() * max(R[ti], 1e-3)), 0.7, "margin",
               ha="center", va="bottom", fontsize=8, color="0.3")
        disc = margin > THRESH
        b.text(0.5, 0.04,
               (f"margin {margin:.2f}  →  DISCOVERABLE" if disc
                else f"margin {margin:.2f}  →  NOT discoverable"),
               transform=b.transAxes, ha="center", va="bottom", fontsize=10,
               color=("C2" if disc else "C3"), fontweight="bold")
        b.set_xscale("log"); b.set_xlim(8e-4, 1.5); b.set_ylim(-0.6, 0.95)
        b.set_yticks([])
        b.set_xlabel("relative residual of each candidate error form\n"
                     r"(on the climatology; the true form sits at $\approx$0)", fontsize=8.5)
        if col == 0:
            b.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.suptitle("Broadly-varying modes expose their error form on the climatology; "
                 "tightly-pinned modes hide it", fontsize=12.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(RESULTS_DIR, "fig_stiff.png")
    if save:
        fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def figure():
    """Both scan figures, from the cached npz."""
    return [scan_figure(), tandem_figure()]
