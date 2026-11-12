"""
error_discovery.experiments.obstruction
=======================================

Form identifiability is obstructed by near-constant modes, and the obstruction is
library-dependent.

An earlier version of this module claimed that form discoverability is set by the
injected driver's own |mean|/std. An independent review showed that was an
artifact of an impoverished candidate library. This module reports the corrected,
verified result.

Setup. The injected error tendency is m(x) = eps * g(x) * (E x); here the true
driver is a single state mode, g = x[k]. "Form discovery" fits each candidate
predictor g_j in a library to m and asks whether the true driver is the unique
best fit (form-identifiability margin = best-wrong-form relative residual minus
truth residual; truth residual ~ 0 because the true form is in the library, so the
margin is the best wrong form's residual; above config.THRESH means identifiable).

Two findings:

1. Library dependence. Against a library of constants and linear and squared modes
   (no cross-products), many drivers look identifiable on the model attractor. Add
   the pairwise cross-products and that collapses: for any driver x[k], the product
   x[k] * x[j] with a near-constant mode x[j] equals (approximately) a rescaled copy
   of x[k], so it reproduces the error. A single near-constant mode therefore aliases
   *every* linear driver.

2. The controller is the maximum |mean|/std over ALL modes, not the driver's own.
   The best cross-product alias reproduces the error to a relative residual of about
   1 / max_j(|mean|/std of x_j): one near-constant mode (large |mean|/std) is enough
   to make the form unidentifiable. On the attractor and on a spin-up transient that
   settles back toward it, several modes stay near-constant (max |mean|/std of order
   1e2 to 1e3), so against the full quadratic library the form is unidentifiable from
   any such data, whatever mode carries the error. Identifiability returns only on a
   distribution that drives every mode through zero (max |mean|/std of order 1), which
   a single model's own dynamics do not supply.

This is the mechanism behind the companion reanalysis and DA-increment nulls: those
distributions cannot identify the error form because they leave near-constant modes
near-constant. The positive recovery results (forcing recovery, P=1) hold against
the restricted, physically-motivated library; the full library is the adversarial
worst case.
"""

import os
import numpy as np

from .. import figstyle
from ..config import DATA_DIR, FIGURES_DIR, EPS, THRESH
from ..figstyle import DPI
from ..formfit import twin, attractor_traj, transient_traj
from ..screening import max_mean_over_std, margin_driver, identifiable_count


def run(n=4000, n_transient=4000, seed=0):
    """Compare form identifiability across sampling distributions and libraries.

    Distributions: the biased-model attractor (climatology), a spin-up transient
    that settles toward it, and a broad full-state excitation (random states that
    drive every mode through zero — illustrative, not dynamically accessible).
    Libraries: restricted (no cross-products) vs full quadratic.
    """
    tw = twin()
    X = attractor_traj(tw, n)
    Xt = transient_traj(tw, n_transient, seed=seed)
    Xbroad = np.random.default_rng(seed + 1).standard_normal((n, tw.ndim)) * 1.0
    dists = {"attractor (climatology)": X,
             "transient (settles back)": Xt,
             "broad excitation (inaccessible)": Xbroad}

    rows = {}
    for name, Xs in dists.items():
        mm = max_mean_over_std(Xs)
        nr, _ = identifiable_count(tw, Xs, "restricted")
        nf, mf = identifiable_count(tw, Xs, "full")
        rows[name] = dict(max_meanstd=mm, ident_restricted=nr, ident_full=nf,
                          median_margin_full=float(np.median(mf)))

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(os.path.join(DATA_DIR, "meanstd_results.npz"),
             names=np.asarray(list(rows)), eps=EPS, ndim=tw.ndim,
             max_meanstd=np.asarray([rows[k]["max_meanstd"] for k in rows]),
             ident_restricted=np.asarray([rows[k]["ident_restricted"] for k in rows]),
             ident_full=np.asarray([rows[k]["ident_full"] for k in rows]),
             median_margin_full=np.asarray([rows[k]["median_margin_full"] for k in rows]))

    print(f"identifiable drivers out of {tw.ndim} (margin > {THRESH}):")
    print(f"{'distribution':32s} {'max|mean|/std':>13s} {'restricted':>11s} {'full':>6s}")
    for name, r in rows.items():
        print(f"{name:32s} {r['max_meanstd']:13.1f} {r['ident_restricted']:11d} {r['ident_full']:6d}")
    # the analytic relation: best full-library alias residual ~ 1 / max|mean|/std
    for name, r in rows.items():
        print(f"  {name:32s} full median margin {r['median_margin_full']:.4f}  "
              f"~ 1/max|mean|/std = {1.0 / r['max_meanstd']:.4f}")
    return rows


def eps_sweep(eps_list=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0), n=4000, n_spin=40000, seed=0):
    """Does turning the injected error UP or DOWN help discovery? For each error
    strength eps, spin the TRUE model onto its own attractor and measure the
    most-near-constant mode (max|mean|/std) and the form-identifiability margin of
    the true driver g=T_{o,1} under both libraries.

    On a fixed distribution the margin is exactly eps-invariant (eps scales the
    error linearly and the margin is a relative residual); eps can only help by
    deforming the attractor enough to excite the near-constant modes. It does not:
    max|mean|/std and the margins are flat in eps because the error self-limits
    (its driver stays near-constant for any eps), so the attractor barely moves.
    """
    import jax.numpy as jnp
    from ..models import TwinModels
    from ..discovery import rollout
    from ..config import DT
    base = twin()
    xa = attractor_traj(base, 2000)[-1]                   # a biased-attractor point (read-only)
    rows = []
    for eps in eps_list:
        tw = TwinModels(eps=eps, g_coeffs=(0.0, 0.0, 1.0, 0.0))
        xt0 = jnp.asarray(np.asarray(rollout(lambda x: tw.f_true(x), jnp.asarray(xa), DT, n_spin))[-1])
        Xtrue = np.asarray(rollout(lambda x: tw.f_true(x), xt0, DT, n))
        k = tw.idx_To1
        rows.append(dict(eps=float(eps), max_meanstd=max_mean_over_std(Xtrue),
                         margin_restricted=margin_driver(tw, Xtrue, k, "restricted")[0],
                         margin_full=margin_driver(tw, Xtrue, k, "full")[0]))
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(os.path.join(DATA_DIR, "meanstd_eps.npz"),
             eps=np.asarray([r["eps"] for r in rows]),
             max_meanstd=np.asarray([r["max_meanstd"] for r in rows]),
             margin_restricted=np.asarray([r["margin_restricted"] for r in rows]),
             margin_full=np.asarray([r["margin_full"] for r in rows]))
    print(f"{'eps':>6s} {'max|mean|/std':>13s} {'restr margin':>12s} {'full margin':>11s}")
    for r in rows:
        print(f"{r['eps']:6.1f} {r['max_meanstd']:13.1f} "
              f"{r['margin_restricted']:12.4f} {r['margin_full']:11.4f}")
    return rows


def figure():
    plt = figstyle.pyplot()
    d = np.load(os.path.join(DATA_DIR, "meanstd_results.npz"), allow_pickle=True)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in d["names"]]
    ndim = int(d["ndim"])
    short = [n.split(" (")[0] + "\n(" + n.split(" (")[1] if " (" in n else n for n in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.bar(x - 0.2, d["ident_restricted"] / ndim, 0.4, color="C0",
           label="restricted library (no cross-terms)")
    ax.bar(x + 0.2, d["ident_full"] / ndim, 0.4, color="C3",
           label="full quadratic library")
    # Count labels on every bar, and a visible red stub at the baseline where the
    # full library identifies 0 forms, so an empty red bar reads as "red = 0" rather
    # than "no red bar here".
    for xi, nr, nf in zip(x, d["ident_restricted"], d["ident_full"]):
        ax.text(xi - 0.2, nr / ndim + 0.02, f"{nr}/{ndim}",
                ha="center", va="bottom", fontsize=7, color="C0")
        if nf == 0:
            ax.plot([xi, xi + 0.4], [0, 0], color="C3", lw=4,
                    solid_capstyle="butt", zorder=5)
            ax.text(xi + 0.2, 0.035, f"{nf}/{ndim}",
                    ha="center", va="bottom", fontsize=7, color="C3", fontweight="bold")
        else:
            ax.text(xi + 0.2, nf / ndim + 0.02, f"{nf}/{ndim}",
                    ha="center", va="bottom", fontsize=7, color="C3")
    for xi, mm in zip(x, d["max_meanstd"]):
        ax.text(xi, 1.02, f"max|mean|/std\n{mm:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("fraction of error forms identifiable")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_meanstd.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return [out]
