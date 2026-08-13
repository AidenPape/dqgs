"""
error_discovery.experiments.donoharm
====================================

Does sparsity ("do-no-harm" L1 regularization) rescue form identifiability on the
attractor?

A standard remedy for an over-complete library is an L1 penalty: fit every term
jointly and let sparsity drive the spurious coefficients to zero (Wu et al. 2024
name this the "do-no-harm" property; it is the engine of SINDy and PDE-FIND). Since
the boundary of experiment_meanstd is diagnosed with single-term competition, the
fair question is whether a joint sparse fit escapes it. Two cases answer it.

1. Truth exactly in the library (the perfect-model twin). L1 recovers the true
   driver and its coefficient even on the attractor: the true term is the sparsest
   exact fit, so L1 prefers it. Sparsity does no harm here -- but only by a margin of
   1/max(|mean|/std), and the attractor fit must light up spurious near-constant
   modes to reach the truth (it cannot drive its residual below that floor with the
   true term alone).

2. Model-form error (the realistic case): the candidate library does not contain the
   exact closure. We remove the true column and let L1 pick a surrogate, scoring it on
   a held-out trajectory from the OTHER regime (fit on the attractor, test off it, and
   vice versa). No globally-valid surrogate exists: the attractor fit latches onto a
   near-constant-mode cross-product alias that looks converged in-regime (residual
   ~0.002 on held-out attractor data) but is order-100% wrong off the attractor, and
   the off-attractor fit likewise fails to carry over to the attractor. Sparsity cannot
   invent a form the library does not contain, and on the attractor it specifically
   manufactures a confidently-wrong alias.

So sparsity relaxes the single-term boundary only when the closure is exactly
representable, and even then only off the attractor is the recovery clean; it neither
enlarges the margin (still set by max(|mean|/std)) nor supplies a missing form. The
obstruction is a property of the sampling distribution and the library, not of the
estimator or its regularizer.

The regression is exact in the coefficients (m(x) = sum_j c_j g_j(x) (E x)), so its
normal matrix is the weighted Gram G_jk = sum_s |E x_s|^2 g_j(x_s) g_k(x_s); the
L1 path is solved by coordinate descent on the standardized Gram.

Results -> results/donoharm_results.npz; figure -> results/fig_donoharm.png.
"""

import os
import numpy as np

from .. import figstyle
from ..formfit import twin, attractor_traj, true_transient
from ..screening import candidate_library, exchange_image, max_mean_over_std
from ..config import RESULTS_DIR, EPS
from ..figstyle import DPI


def _design(tw, X, k, drop_true=False):
    """Full-quadratic library values Phi, exchange image EX, weight w=|Ex|^2, exact
    tendency m=eps*x[k]*(Ex), and the true-term index. If drop_true, remove the
    column x[k] so the true closure is not exactly representable."""
    cols, names = candidate_library(X, k, "full")
    EX = exchange_image(tw, X)
    w = (EX * EX).sum(1)
    m = EPS * X[:, k][:, None] * EX
    ti = names.index(f"x{k}")
    if drop_true:
        keep = [i for i in range(len(cols)) if i != ti]
        cols = [cols[i] for i in keep]
        names = [names[i] for i in keep]
        ti = -1
    return np.stack(cols, 1), EX, w, m, names, ti


def _gram(Phi, EX, w, m):
    return Phi.T @ (w[:, None] * Phi), Phi.T @ (EX * m).sum(1), float((m * m).sum())


def _lasso_path(G, b, yty, n_alpha=30, n_sweeps=250, tol=1e-10):
    p = len(b)
    d = np.sqrt(np.maximum(np.diag(G), 1e-300))
    Gs = G / np.outer(d, d); bs = b / d
    np.fill_diagonal(Gs, 1.0)
    alphas = float(np.max(np.abs(bs))) * np.logspace(0, -3, n_alpha)
    out, cs = [], np.zeros(p)
    for alpha in alphas:
        Gc = Gs @ cs
        for _ in range(n_sweeps):
            dmax = 0.0
            for j in range(p):
                rho = bs[j] - Gc[j] + cs[j]
                cj = np.sign(rho) * max(abs(rho) - alpha, 0.0)
                if cj != cs[j]:
                    Gc += Gs[:, j] * (cj - cs[j]); dmax = max(dmax, abs(cj - cs[j])); cs[j] = cj
            if dmax < tol:
                break
        c = cs / d
        resid2 = float(c @ (G @ c) - 2 * c @ b + yty)
        active = np.where(np.abs(cs) > 1e-6 * (np.abs(cs).max() + 1e-300))[0]
        out.append(dict(c=c.copy(), active=active[np.argsort(-np.abs(cs[active]))],
                        rel_resid=float(np.sqrt(max(resid2, 0.0)) / np.sqrt(yty + 1e-300))))
    return out


def _generalize(tw, c, names, k, Xtest):
    """Relative residual of a fitted library model (coeffs c over `names`) in
    predicting the true error tendency on a held-out distribution Xtest."""
    colsT, namesT = candidate_library(Xtest, k, "full")
    idx = [namesT.index(nm) for nm in names]
    EXt = exchange_image(tw, Xtest); mt = EPS * Xtest[:, k][:, None] * EXt
    pred = (np.stack([colsT[i] for i in idx], 1) @ c)[:, None] * EXt
    return float(np.linalg.norm(pred - mt) / np.linalg.norm(mt))


def _offattr(tw, seeds):
    return np.concatenate([true_transient(tw, 600, seed=s)[50:550] for s in seeds], 0)


def run(n=4000, seed=0, verbose=True):
    tw = twin(); k = tw.idx_To1
    # independent fit/test splits in each regime (held-out generalization)
    Xatt_fit = attractor_traj(tw, n)
    Xatt_test = attractor_traj(tw, n)[::-1][:3000]            # disjoint attractor sample
    Xoff_fit = _offattr(tw, range(400, 404))
    Xoff_test = _offattr(tw, range(500, 504))

    res = {}
    # (1) exact library: coefficient-recovery path on each distribution
    for label, X in (("attractor", Xatt_fit), ("off_attractor", Xoff_fit)):
        Phi, EX, w, m, names, ti = _design(tw, X, k)
        path = _lasso_path(*_gram(Phi, EX, w, m))
        res[label] = dict(max_meanstd=max_mean_over_std(X),
                          nactive=np.asarray([len(p["active"]) for p in path]),
                          c_true=np.asarray([p["c"][ti] for p in path]),
                          relresid=np.asarray([p["rel_resid"] for p in path]))
        if verbose:
            print(f"[exact     {label:13s}] max|mean|/std={res[label]['max_meanstd']:7.1f}  "
                  f"recovers eps={res[label]['c_true'][-1]:.4f} (true {EPS}) "
                  f"with {res[label]['nactive'][-1]} active terms")

    # (2) model-form error: drop the true term, select a surrogate, score it on a
    # HELD-OUT trajectory from each regime (in-regime and cross-regime).
    gen = {}
    for label, Xfit, Xin, Xcross in (("attractor", Xatt_fit, Xatt_test, Xoff_test),
                                     ("off_attractor", Xoff_fit, Xoff_test, Xatt_test)):
        Phi, EX, w, m, names, _ = _design(tw, Xfit, k, drop_true=True)
        sel = _lasso_path(*_gram(Phi, EX, w, m))[-1]
        gen[label] = dict(selected=names[sel["active"][0]],
                          resid_in=_generalize(tw, sel["c"], names, k, Xin),
                          resid_cross=_generalize(tw, sel["c"], names, k, Xcross))
        if verbose:
            print(f"[mis-spec  {label:13s}] selects '{gen[label]['selected']:8s}'  "
                  f"in-regime residual={gen[label]['resid_in']:.3f}  "
                  f"cross-regime (held-out) residual={gen[label]['resid_cross']:.2f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, "donoharm_results.npz"), eps=EPS, k=k,
             **{f"{lab}_{key}": np.asarray(val) for lab, s in res.items()
                for key, val in s.items()},
             gen_labels=np.asarray(list(gen)),
             gen_selected=np.asarray([gen[l]["selected"] for l in gen]),
             gen_resid_in=np.asarray([gen[l]["resid_in"] for l in gen]),
             gen_resid_cross=np.asarray([gen[l]["resid_cross"] for l in gen]))
    return res, gen


def figure():
    plt = figstyle.pyplot()
    d = np.load(os.path.join(RESULTS_DIR, "donoharm_results.npz"), allow_pickle=True)
    eps = float(d["eps"])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    styles = {"attractor": ("C3", "attractor (climatology)"),
              "off_attractor": ("C2", "off-attractor (excited)")}
    # (a) exact-library coefficient recovery vs sparsity
    for lab, (color, pretty) in styles.items():
        ax[0].plot(d[f"{lab}_nactive"], d[f"{lab}_c_true"], "o-", color=color, ms=3, label=pretty)
    ax[0].axhline(eps, ls=":", color="grey", lw=1)
    ax[0].text(0.97, 0.6, r"true $\varepsilon$", transform=ax[0].transAxes, ha="right",
               fontsize=8, color="grey")
    ax[0].set_xlabel("active terms in the L1 fit")
    ax[0].set_ylabel(r"recovered coefficient on $T_{o,1}$")
    ax[0].set_title("(a) truth in library", loc="left", fontsize=9)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    # (b) model-form error: in-regime vs held-out cross-regime residual of the
    # surrogate L1 selects. Low in-regime, high cross-regime = a confidently-wrong fit.
    labs = [l.decode() if isinstance(l, bytes) else str(l) for l in d["gen_labels"]]
    pretty = ["attractor\nfit" if ("att" in l and "off" not in l) else "off-attractor\nfit"
              for l in labs]
    sel = [s.decode() if isinstance(s, bytes) else str(s) for s in d["gen_selected"]]
    x = np.arange(len(labs))
    ax[1].bar(x - 0.2, d["gen_resid_in"], 0.4, color="0.6", label="in-regime (held-out)")
    ax[1].bar(x + 0.2, d["gen_resid_cross"], 0.4, color="C3", label="other regime (held-out)")
    for xi, ri, rc, sname in zip(x, d["gen_resid_in"], d["gen_resid_cross"], sel):
        ax[1].text(xi - 0.2, ri + 0.08, f"{ri:.2f}", ha="center", va="bottom", fontsize=7)
        ax[1].text(xi + 0.2, rc + 0.08, f"{rc:.2f}", ha="center", va="bottom", fontsize=7)
        ax[1].text(xi, max(d["gen_resid_cross"]) * 1.18, sname, ha="center", va="center",
                   fontsize=7, style="italic", color="0.3")
    ax[1].axhline(0.1, ls=":", color="grey", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(pretty, fontsize=8)
    ax[1].set_ylabel("error of the selected surrogate")
    ax[1].set_title("(b) closure not in library", loc="left", fontsize=9)
    ax[1].set_ylim(0, max(d["gen_resid_cross"]) * 1.32 + 0.1)
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_donoharm.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return [out]


if __name__ == "__main__":
    run()
    print(figure())
