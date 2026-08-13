"""
error_discovery.experiments.overcomplete
========================================

The over-complete library succeeds off the attractor.

experiment_meanstd shows the failure half of the story: against the full quadratic
library a near-constant mode aliases every driver, so on the attractor the complete
library identifies 0 of 36 error forms. That experiment exhibits the cure only with
a "broad excitation" distribution of random Gaussian states, which is illustrative
but not dynamically accessible.

This experiment supplies the missing, accessible half: it sweeps all 36 modes as
the injected driver and, with the *same* full quadratic library, compares form
identifiability on

* the model attractor (climatology)               -- the failure (0/36),
* a short transient that settles back toward it    -- still near-constant, low,
* a dynamically-accessible off-attractor ensemble  -- the success.

The accessible ensemble is built from early segments of true-model trajectories
launched off the attractor (the same data the differentiable forcing recovery uses
in experiment_form): every mode crosses zero there, so max(|mean|/std) falls to
order one and the complete library recovers the form for nearly every driver. This
is the over-complete-library analogue of SINDy/PDE-FIND recovery, and it closes the
loop: the boundary is a property of the sampling distribution, and leaving the
attractor removes it even with the adversarial complete library.

Results -> results/overcomplete_results.npz; figure -> results/fig_overcomplete.png.
"""

import os
import numpy as np

from .. import figstyle
from ..formfit import twin, attractor_traj, transient_traj, true_transient
from ..screening import identifiable_count, max_mean_over_std
from ..config import RESULTS_DIR, EPS
from ..figstyle import DPI


def accessible_offattractor(tw, n_seg=8, length=600, drop=50, keep=500):
    """A dynamically-accessible off-attractor ensemble: concatenated early segments
    of true-model trajectories launched from diverse off-attractor states."""
    segs = [true_transient(tw, length, seed=700 + s)[drop:drop + keep] for s in range(n_seg)]
    return np.concatenate(segs, 0)


def run(n=4000, seed=0, verbose=True):
    tw = twin()
    dists = {
        "attractor (climatology)": attractor_traj(tw, n),
        "short transient": transient_traj(tw, n, seed=seed),
        "off-attractor (accessible)": accessible_offattractor(tw),
    }
    rows = {}
    for name, X in dists.items():
        mm = max_mean_over_std(X)
        nf, margins = identifiable_count(tw, X, "full")          # full quadratic library
        rows[name] = dict(max_meanstd=mm, ident_full=int(nf),
                          median_margin=float(np.median(margins)), margins=margins)
        if verbose:
            print(f"[{name:28s}] max|mean|/std={mm:8.1f}  full-library identifiable={nf:2d}/{tw.ndim}  "
                  f"median margin={np.median(margins):+.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, "overcomplete_results.npz"),
             names=np.asarray(list(rows)), ndim=tw.ndim, eps=EPS,
             max_meanstd=np.asarray([rows[k]["max_meanstd"] for k in rows]),
             ident_full=np.asarray([rows[k]["ident_full"] for k in rows]),
             median_margin=np.asarray([rows[k]["median_margin"] for k in rows]),
             margins=np.stack([rows[k]["margins"] for k in rows]))
    return rows


def figure():
    plt = figstyle.pyplot()
    d = np.load(os.path.join(RESULTS_DIR, "overcomplete_results.npz"), allow_pickle=True)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in d["names"]]
    ndim = int(d["ndim"])
    short = [n.replace(" (", "\n(") for n in names]
    x = np.arange(len(names))
    colors = ["C3", "C1", "C2"]

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.bar(x, d["ident_full"] / ndim, 0.55, color=colors[:len(x)])
    for xi, nf, mm in zip(x, d["ident_full"], d["max_meanstd"]):
        ax.text(xi, nf / ndim + 0.02, f"{nf}/{ndim}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.annotate(f"max|mean|/std {mm:.1f}", xy=(xi, 0), xytext=(0, -34),
                    textcoords="offset points", ha="center", va="top", fontsize=7,
                    color="0.35", annotation_clip=False)
    ax.axhline(0.1, ls=":", color="grey", lw=1)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("fraction of error forms identifiable\n(full quadratic library)")
    # tight_layout first, then reserve extra bottom room for the per-category sub-labels
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26)
    out = os.path.join(RESULTS_DIR, "fig_overcomplete.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    run()
    print(figure())
