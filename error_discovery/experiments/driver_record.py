"""
error_discovery.experiments.driver_record
=========================================

Does a longer climatology record cure the blind spot? For the atmosphere yes, for
ocean temperature no — and the two fluids move in opposite directions.

The margin depends on the driver's |mean|/std, and |mean|/std depends on how much
of the driver's variability the record has seen. Atmospheric drivers have slow
low-frequency excursions, so a longer record accumulates variance, lowers their
|mean|/std, and carries them across the identifiability threshold. Ocean
temperature is held near its equilibrium by strong thermal damping, so its
|mean|/std is flat in record length: a **permanent** blind spot, not a sampling
one. The shorter records here are prefixes of the longest, so this is one
climatology observed for longer rather than several different climatologies.
"""

import os
import numpy as np

from .. import figstyle
from ..models import TwinModels
from ..config import DATA_DIR, FIGURES_DIR, THRESH, EPS
from ..figstyle import DPI
from ..formfit import attractor_traj, err_tendency, fit, resid, margin_named


RL_DRIVERS = [
    ("psi_a1",   (0.0, 0.0, 0.0, 1.0), "x0",  0,  "atmospheric circulation", "C0", "o-"),
    ("theta_a1", (0.0, 1.0, 0.0, 0.0), "x10", 10, "atmospheric temperature", "C9", "s--"),
    ("T_o1",     (0.0, 0.0, 1.0, 0.0), "x28", 28, "ocean temperature",       "C3", "D-"),
]

# The two drivers panel (b) contrasts: the broad one and the pinned one. Their
# columns are cached by ``run`` so the figure needs no trajectory of its own.
PANELS = [(0, "C0", r"$\psi_{a,1}$ atmospheric circulation"),
          (28, "C3", r"$T_{o,1}$ ocean temperature")]


def run(Ns=(2000, 4000, 8000, 20000, 40000, 80000), save=True):
    """Form-identifiability margin and the driver's |mean|/std versus the climatology
    record length, for two atmospheric drivers and the ocean-temperature driver. The
    shorter records are prefixes of the longest, so this is one climatology observed
    for longer. Atmospheric drivers cross the identifiability threshold as the record
    lengthens (their slow low-frequency excursions accumulate variance, lowering
    |mean|/std); the ocean-temperature driver stays pinned far above threshold at every
    length (strong thermal damping holds it near its equilibrium), so it is a permanent
    blind spot that more data does not cure."""
    tw0 = TwinModels(eps=EPS, g_coeffs=(0.0, 0.0, 1.0, 0.0))
    Ns = list(Ns)
    Xfull = attractor_traj(tw0, max(Ns))            # one trajectory; prefixes are shorter records
    margin = np.zeros((len(RL_DRIVERS), len(Ns)))
    rr = np.zeros((len(RL_DRIVERS), len(Ns)))
    for di, (name, gc, tname, didx, _lab, _c, _ls) in enumerate(RL_DRIVERS):
        tw = TwinModels(eps=EPS, g_coeffs=gc)
        for ni, N in enumerate(Ns):
            X = Xfull[:N]
            m = err_tendency(tw, X)
            c, nm = fit(tw, X, m)
            margin[di, ni] = margin_named(resid(tw, c, X, m), nm, tname)[0]
            rr[di, ni] = float(abs(X[:, didx].mean()) / X[:, didx].std())
    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        np.savez(os.path.join(DATA_DIR, "driver_record.npz"),
                 Ns=np.asarray(Ns), margin=margin, meanstd=rr,
                 names=np.asarray([d[0] for d in RL_DRIVERS]),
                 # panel (b) shows these two distributions at the longest record;
                 # keeping the columns here saves the figure an 80k-step re-integration
                 **{f"dist_{didx}": Xfull[:, didx] for didx, _, _ in PANELS})
    print(f"{'N':>7s} " + " ".join(f"{d[0]:>20s}" for d in RL_DRIVERS))
    for ni, N in enumerate(Ns):
        print(f"{N:7d} " + " ".join(f"r={rr[di,ni]:6.1f} m={margin[di,ni]:5.3f}"
                                    for di in range(len(RL_DRIVERS))))
    return dict(Ns=np.asarray(Ns), margin=margin, meanstd=rr)


def figure(save=True):
    """The headline: discoverability of an injected error form is set by the driver's
    |mean|/std, and the two fluids move in opposite directions as the record lengthens.
    (a) margin versus record length: atmospheric drivers rise across the threshold,
    ocean temperature stays below it. (b) why: the ocean-temperature driver is a needle
    pinned ~40 std from zero, the atmospheric-circulation driver explores a broad range."""
    plt = figstyle.pyplot()

    d = np.load(os.path.join(DATA_DIR, "driver_record.npz"), allow_pickle=True)
    Ns, margin = d["Ns"], d["margin"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # (a) margin vs record length: the opposite trends
    a = ax[0]
    a.axhspan(8e-4, THRESH, color="C3", alpha=0.07)
    a.axhline(THRESH, ls=":", color="C3", lw=1)
    for di, (name, gc, tname, didx, lab, col, ls) in enumerate(RL_DRIVERS):
        a.plot(Ns, np.maximum(margin[di], 8e-4), ls, color=col, ms=5, lw=1.5, label=lab)
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(8e-4, 0.5)
    a.set_xlabel("climatology record length (steps)")
    a.set_ylabel("form-identifiability margin")
    a.set_title("(a)", fontsize=10, loc="left")
    a.legend(fontsize=8, frameon=False, loc="lower left")
    a.annotate("identifiable", xy=(Ns[-1], THRESH * 1.5), fontsize=7.5, color="0.3",
               ha="right", va="bottom")
    a.annotate("permanent blind spot\n(more data does not help)",
               xy=(Ns[-2], margin[2][-2]), xytext=(-4, 26), textcoords="offset points",
               fontsize=7.5, color="C3", ha="right", va="bottom",
               arrowprops=dict(arrowstyle="-", lw=0.6, color="C3"))

    # (b) distributions at the longest record: the |mean|/std mechanism
    b = ax[1]
    for didx, col, lab in PANELS:
        v = d[f"dist_{didx}"]
        vn = (v - 0.0) / v.std()                    # in std units, so zero stays at 0
        b.hist(vn, bins=80, density=True, color=col, alpha=0.5, label=lab)
    b.axvline(0.0, color="0.4", lw=1.4)
    b.text(0.0, b.get_ylim()[1] * 0.96, " zero", color="0.4", fontsize=8, va="top")
    for didx, col, lab in PANELS:
        v = d[f"dist_{didx}"]
        r = abs(v.mean()) / v.std()
        b.annotate(rf"$|\mathrm{{mean}}|/\mathrm{{std}}={r:.0f}$", xy=(r, 0),
                   xytext=(r, b.get_ylim()[1] * (0.55 if col == "C3" else 0.78)),
                   color=col, fontsize=9, ha="center",
                   arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
    b.set_xlabel("driver amplitude (standard deviations from zero)")
    b.set_ylabel("density over the climatology")
    b.set_title("(b)", fontsize=10, loc="left")
    b.legend(fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_driver.png")
    if save:
        fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return [out]
