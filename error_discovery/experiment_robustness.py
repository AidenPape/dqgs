"""
error_discovery.experiment_robustness
======================================

Stress tests of the form-identifiability argument, in response to specific
critiques that the max(|mean|/std) diagnostic could be presented as a universal
law rather than a screening diagnostic.

Two experiments:

1. ``law_validation`` -- the screening relation margin ~ 1/max(|mean|/std) is
   approximate, not exact. Across sampling distributions spanning max(|mean|/std)
   from order 1e-2 to order 1e3, the best-alias residual under the full quadratic
   library tracks 1/max(|mean|/std) only to within a factor of a few in the
   obstructed regime, and saturates near 1 once no mode is near-constant. The
   relation is monotone and order-of-magnitude, useful for screening, not a law.

2. ``parsimony_floor`` -- a complexity prior cannot rescue an offset-dominated
   driver, because the simplest possible alias, the constant, caps the achievable
   margin. The full-library best alias of a driver is a higher-order cross-product
   with the most near-constant mode, and a complexity prior would penalize it; but
   the constant is the lowest-order term and cannot be penalized away. The truth's
   margin over the constant is about 1/(the driver's own |mean|/std), so an
   offset-dominated driver stays unidentifiable against the constant no matter the
   penalty. This reconciles two diagnostics: max(|mean|/std) over modes sets the
   raw, prior-free margin (cross-product aliasing); the driver's own |mean|/std sets
   the parsimony-proof margin (constant aliasing).

The error tendency for a candidate driver g = x[k] is m(x) = eps * x[k] * (E x), so
the truth term has residual zero in this clean twin and always has the smallest
residual; identifiability is therefore a question of the residual margin over the
best alias, not of which term is selected. The competition is over the full
quadratic library {const, 36 linear, all pairwise products}.
"""

import os
import numpy as np

# faint white backing for annotations near curves/bands (mask the line behind the text)
_LBL_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0)

from .experiment_form import twin, attractor_traj, transient_traj, EPS
from .experiment_meanstd import _EX, max_mean_over_std, mean_over_std
from .config import RESULTS_DIR


def library_matrix(tw, X):
    """Full quadratic library evaluated on X. Returns the design pieces used by the
    fast residual computation:
      G     (n, P) predictor fields g_j(x)
      names list of P names, order array of P term orders (0, 1, or 2)
      EX    (n, ndim) exchange image E x,  sq (n,) its squared row norm
      den   (P,) the noise-independent denominator sum_n g_j^2 ||E x||^2
    """
    n, nd = X.shape
    cols = [np.ones(n)] + [X[:, j] for j in range(nd)]
    names = ["const"] + [f"x{j}" for j in range(nd)]
    order = [0] + [1] * nd
    for i in range(nd):
        for j in range(i, nd):
            cols.append(X[:, i] * X[:, j]); names.append(f"x{i}*x{j}"); order.append(2)
    G = np.stack(cols, 1)                          # (n, P)
    EX = _EX(tw, X)
    sq = (EX * EX).sum(1)                           # (n,)
    den = (G * G).T @ sq                            # (P,)
    return G, names, np.asarray(order), EX, sq, den


def residuals(G, EX, sq, den, mhat):
    """Relative single-term residual of every library term against the tendency
    estimate mhat (n, ndim). Vectorized over terms via G.T @ w."""
    w = (EX * mhat).sum(1)                          # (n,)
    num = G.T @ w                                   # (P,)
    m2 = float((mhat * mhat).sum())
    c = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return np.sqrt(np.maximum(c * c * den - 2 * c * num + m2, 0.0)) / np.sqrt(m2)


# --------------------------------------------------------------------------- #
# 1. Law validation: margin vs 1/max(|mean|/std) across distributions.
# --------------------------------------------------------------------------- #
def law_validation(n=4000, seed=0):
    tw = twin()
    rng = np.random.default_rng(seed)
    Xa = attractor_traj(tw, n)
    dists = {"attractor n=4000": Xa,
             "attractor n=20000": attractor_traj(tw, 20000),
             "transient": transient_traj(tw, n, seed=0)}
    for f in (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.35, 0.5):
        nb = max(1, int(f * n))
        dists[f"mix {f*100:g}%"] = np.vstack([Xa[:n - nb], rng.standard_normal((nb, tw.ndim))])
    dists["broad"] = rng.standard_normal((n, tw.ndim))

    names_out, mm_out, margin_out = [], [], []
    for nm, X in dists.items():
        G, names, order, EX, sq, den = library_matrix(tw, X)
        margins = []
        for k in range(tw.ndim):
            m = EPS * X[:, k][:, None] * EX
            R = residuals(G, EX, sq, den, m)
            t = names.index(f"x{k}")
            margins.append(float(np.delete(R, t).min() - R[t]))
        names_out.append(nm); mm_out.append(max_mean_over_std(X))
        margin_out.append(float(np.median(margins)))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, "robust_law.npz"),
             names=np.asarray(names_out), max_meanstd=np.asarray(mm_out),
             margin=np.asarray(margin_out))
    print(f"{'distribution':20s} {'max|m|/std':>10s} {'margin':>9s} {'margin*max':>10s}")
    for nm, mm, mg in zip(names_out, mm_out, margin_out):
        print(f"{nm:20s} {mm:10.2f} {mg:9.4f} {mg*mm:10.2f}")
    return dict(names=names_out, max_meanstd=np.asarray(mm_out), margin=np.asarray(margin_out))


# --------------------------------------------------------------------------- #
# 2. Parsimony floor: the constant alias a complexity prior cannot remove.
# --------------------------------------------------------------------------- #
def parsimony_floor(n=4000):
    """For each candidate driver on the attractor, compare two residual margins
    under the full quadratic library: the raw margin over the best alias (a
    cross-product, the prior-free obstruction) and the margin over the constant (the
    simplest alias, which no complexity prior can penalize away). The constant margin
    is about 1/(the driver's own |mean|/std); where it is below the 0.1 threshold the
    driver is unidentifiable even against the simplest alias, so a parsimony prior
    cannot rescue it."""
    tw = twin()
    X = attractor_traj(tw, n)
    G, names, order, EX, sq, den = library_matrix(tw, X)
    ci = names.index("const")
    modes = np.arange(tw.ndim)
    ms, raw, const = np.zeros(tw.ndim), np.zeros(tw.ndim), np.zeros(tw.ndim)
    for k in modes:
        m = EPS * X[:, k][:, None] * EX
        R = residuals(G, EX, sq, den, m)
        t = names.index(f"x{k}")
        ms[k] = mean_over_std(X, k)
        raw[k] = float(np.delete(R, t).min() - R[t])        # full-library best alias
        const[k] = float(R[ci] - R[t])                      # the simplest alias
    named = {"psi_a1": tw.idx_psia1, "theta_a1": tw.idx_thetaa1, "T_o1": tw.idx_To1}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, "robust_parsimony.npz"),
             meanstd=ms, raw_margin=raw, const_margin=const,
             named=np.asarray(list(named.items()), dtype=object))
    proof = const < 0.1                                     # unidentifiable even vs the constant
    print(f"drivers unidentifiable against the constant (parsimony-proof): "
          f"{int(proof.sum())}/{tw.ndim}")
    print(f"{'driver':9s} {'|mean|/std':>10s} {'raw margin':>11s} {'const margin':>12s} {'1/|m|/std':>10s}")
    for nm, k in named.items():
        print(f"{nm:9s} {ms[k]:10.2f} {raw[k]:11.4f} {const[k]:12.4f} {1/ms[k]:10.4f}")
    return dict(meanstd=ms, raw_margin=raw, const_margin=const, named=named)


def figure():
    """Two parallel views of the same quantity, the true form's residual margin
    over its best alias (log y, shared across panels). In both, a margin below the
    0.1 threshold (red zone) means the data cannot separate the true form from the
    alias. Panel (a): the margin across sampling distributions versus the worst
    near-constant mode, max(|mean|/std); the relation margin ~ 1/max(|mean|/std)
    holds within a factor of three. Panel (b): on the attractor, the margin of each
    driver against the best alias of *any* order (grey, the prior-free floor) and
    against the *constant* alone (blue, the floor no complexity prior can lift); the
    constant margin follows 1/(driver's own |mean|/std), so offset-dominated drivers
    (|mean|/std > 10, right of the dotted line) stay unidentifiable under any prior."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    THRESH = 0.1
    YLIM = (8e-4, 2.0)
    guide = r"$1/(|\mathrm{mean}|/\mathrm{std})$"

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    for a in ax:                                            # shared "unidentifiable" zone
        a.axhspan(YLIM[0], THRESH, color="C3", alpha=0.06, zorder=0)
        a.axhline(THRESH, ls=":", color="C3", lw=1, zorder=1)

    # ----- (a) the screening relation across sampling distributions --------- #
    d = np.load(os.path.join(RESULTS_DIR, "robust_law.npz"), allow_pickle=True)
    names = list(d["names"]); mm, mg = d["max_meanstd"], d["margin"]
    a = ax[0]
    xx = np.logspace(-2, 3.2, 200)
    g = np.minimum(1.0, 1.0 / xx)
    a.fill_between(xx, np.minimum(1.0, 1.0 / (3 * xx)), np.minimum(1.0, 3.0 / xx),
                   color="grey", alpha=0.18, lw=0, label="factor of 3", zorder=1)
    a.plot(xx, g, "k--", lw=1.3, label=guide.replace("(|", r"\max(|"), zorder=2)
    a.scatter(mm, mg, s=48, color="C0", zorder=4, edgecolor="white", linewidth=0.6)
    pa = dict(arrowstyle="-", lw=0.6, color="0.4")
    ix = names.index("broad")
    a.annotate("broad excitation\n(every mode crosses 0:\nidentifiable)",
               (mm[ix], mg[ix]), xytext=(34, -86), textcoords="offset points",
               fontsize=7.5, ha="left", va="top", arrowprops=pa, bbox=_LBL_BBOX)
    ix = names.index("attractor n=4000")
    a.annotate("attractor\n(climatology:\nunidentifiable)",
               (mm[ix], mg[ix]), xytext=(-12, 64), textcoords="offset points",
               fontsize=7.5, ha="right", va="bottom", arrowprops=pa, bbox=_LBL_BBOX)
    a.text(0.025, THRESH * 1.25, "identifiable", transform=a.get_yaxis_transform(),
           fontsize=7.5, color="0.3", va="bottom")
    a.text(0.025, THRESH * 0.8, "unidentifiable", transform=a.get_yaxis_transform(),
           fontsize=7.5, color="C3", va="top")
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(*YLIM)
    a.set_xlabel(r"$\max(|\mathrm{mean}|/\mathrm{std})$ over modes,"
                 "\nper sampling distribution")
    a.set_ylabel("residual margin of the true form\nover its best alias")
    a.set_title("(a)", fontsize=10, loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="lower left")

    # ----- (b) the parsimony floor: the constant alias --------------------- #
    p = np.load(os.path.join(RESULTS_DIR, "robust_parsimony.npz"), allow_pickle=True)
    ms, rawm, cm = p["meanstd"], p["raw_margin"], p["const_margin"]
    named = dict(p["named"]); idx = [int(v) for v in named.values()]
    b = ax[1]
    xx = np.logspace(-2.6, 3, 200)
    b.plot(xx, np.minimum(1.0, 1.0 / xx), "k--", lw=1.3, label=guide, zorder=2)
    b.axvline(10, ls=":", color="0.5", lw=0.9, zorder=1)
    b.scatter(ms, rawm, s=22, color="C7", alpha=0.5, zorder=3,
              label="margin over the best alias of any order\n(prior-free floor)")
    b.scatter(ms, cm, s=36, color="C0", zorder=4,
              label="margin over the constant alias\n(survives any complexity prior)")
    # the three mode-1 drivers, split by whether a longer record cures them
    atm = [named["psi_a1"], named["theta_a1"]]; oc = named["T_o1"]
    b.scatter(ms[atm], cm[atm], s=80, color="C0", zorder=5, edgecolor="k",
              linewidth=0.8, marker="^",
              label=r"atmospheric mode-1 $\psi_{a,1},\theta_{a,1}$ (record-curable)")
    b.scatter([ms[oc]], [cm[oc]], s=90, color="C3", zorder=5, edgecolor="k",
              linewidth=0.8, marker="D", label=r"ocean mode-1 $T_{o,1}$ (permanent)")
    tex = {"psi_a1": r"$\psi_{a,1}$", "theta_a1": r"$\theta_{a,1}$", "T_o1": r"$T_{o,1}$"}
    off = {"theta_a1": (-40, 50), "psi_a1": (2, 46), "T_o1": (34, 24)}
    for nm, k in named.items():
        b.annotate(f"{tex[nm]} {ms[k]:.0f}", (ms[k], cm[k]), xytext=off[nm],
                   textcoords="offset points", fontsize=7,
                   color=("C3" if nm == "T_o1" else "C0"),
                   arrowprops=dict(arrowstyle="-", lw=0.5,
                                   color=("C3" if nm == "T_o1" else "C0")))
    b.text(11, YLIM[1] * 0.6, r"$|\mathrm{mean}|/\mathrm{std}=10$", fontsize=7,
           color="0.4", rotation=90, va="top", ha="right", bbox=_LBL_BBOX)
    b.text(0.02, 0.60, "climatology, $n=4000$ steps", transform=b.transAxes,
           fontsize=7, color="0.4", ha="left", va="top", bbox=_LBL_BBOX)
    b.set_xscale("log"); b.set_yscale("log"); b.set_ylim(*YLIM)
    b.set_xlabel(r"the driver's own $|\mathrm{mean}|/\mathrm{std}$"
                 "\n(on the attractor)")
    b.set_title("(b)", fontsize=10, loc="left")
    hb, lb = b.get_legend_handles_labels()

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(hb, lb, loc="lower center", ncol=3, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.4, handletextpad=0.5)
    out = os.path.join(RESULTS_DIR, "fig_robustness.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


def table_figure():
    """The 13 rows behind Fig.~6a as a sorted paired bar chart, so the individual
    dataset-by-dataset numbers (name, max(|mean|/std), median margin) are legible
    rather than only visible as an xy scatter. Reads ``robust_law.npz`` written by
    ``law_validation``; adds nothing new. Rows sorted worst-pinned to best-excited,
    matching the reading order "more excitation -> lower max(|mean|/std) -> higher
    margin" of the panel it accompanies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    THRESH = 0.1
    d = np.load(os.path.join(RESULTS_DIR, "robust_law.npz"), allow_pickle=True)
    names, mm, mg = list(d["names"]), d["max_meanstd"], d["margin"]

    label = {
        "attractor n=4000": "attractor climatology, n=4000",
        "attractor n=20000": "attractor climatology, n=20000",
        "transient": "spin-up transient",
        "broad": "broad excitation (i.i.d., std=1)",
    }
    labels = [label[nm] if nm in label else f"climatology + {nm.split()[1]} broad-noise rows"
              for nm in names]

    order = np.argsort(mm)[::-1]                       # worst-pinned (top) to best-excited (bottom)
    labels = [labels[i] for i in order]
    mm, mg = mm[order], mg[order]
    y = np.arange(len(labels))[::-1]                    # top row = first entry

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 5.2), sharey=True)

    a.barh(y, mm, color="0.6", height=0.62, zorder=3)
    for yi, v in zip(y, mm):
        a.text(v * 1.15, yi, f"{v:,.2f}" if v < 10 else f"{v:,.0f}", va="center",
               fontsize=8, color="0.25")
    a.set_xscale("log"); a.set_xlim(1e-2, 3e3)
    a.invert_xaxis()                                    # grows toward the shared labels
    a.set_xlabel(r"$\max(|\mathrm{mean}|/\mathrm{std})$ over modes")
    a.set_yticks(y); a.set_yticklabels(labels, fontsize=8.5)
    a.set_title("(a) the diagnostic", fontsize=10, loc="left")
    a.grid(axis="x", color="0.9", lw=0.6, zorder=0)

    cols = ["C3" if v < THRESH else "C0" for v in mg]
    b.barh(y, mg, color=cols, height=0.62, zorder=3, alpha=0.9)
    b.set_ylim(y.min() - 0.8, y.max() + 1.3)
    b.axvline(THRESH, ls=":", color="0.3", lw=1, zorder=4)
    b.text(THRESH, y.max() + 0.9, "threshold", fontsize=7.5, color="0.3",
           ha="center", va="bottom", clip_on=False)
    for yi, v in zip(y, mg):
        b.text(max(v, 8e-4) * 1.15, yi, f"{v:.3f}" if v < 0.1 else f"{v:.2f}",
               va="center", fontsize=8, color="0.25")
    b.set_xscale("log"); b.set_xlim(5e-4, 3.0)
    b.set_xlabel("median form-identifiability margin\n(over all 36 candidate drivers)")
    b.set_title("(b) the outcome", fontsize=10, loc="left")
    b.grid(axis="x", color="0.9", lw=0.6, zorder=0)

    fig.suptitle("The 13 datasets behind Fig. 6's screening-relation panel", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.legend(handles=[Patch(color="C3", label="unidentifiable (margin below threshold)"),
                         Patch(color="C0", label="identifiable")],
               fontsize=8.5, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.0))
    out = os.path.join(RESULTS_DIR, "fig_robustness_table.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out
