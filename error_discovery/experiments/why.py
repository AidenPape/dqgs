"""
error_discovery.experiments.why
===============================

Why does form discovery succeed only on off-attractor / transient data? This module
turns that empirical asymmetry (E1/E2 in ``experiment_form``) into an explained,
quantitative statement, and rebuts the natural objection that the positive result is
an artifact of fitting in a regime the model never operates in.

Three results, all on the standard MAOOAM twin (truth g = T_{o,1} = x[28], eps=0.1):

1. ``governing_law`` -- the form-identifiability margin is a function of the *driver's
   own* |mean|/std on the sampling distribution, collapsing onto min(1, 1/r). The
   attractor pins the slow ocean driver near-constant (r ~ 30), so the form is
   unidentifiable there; off-attractor data is simply the only accessible way to lower
   r below the identifiability threshold. "Only transient works" = "only transient
   excites the driver" = "only transient lowers the driver's |mean|/std".

2. ``global_form_validation`` -- the recovered error is a *global* property of the
   model's equations; only its *identifiability* is local. A form fit on off-attractor
   data predicts the error tendency back on the attractor (out-of-distribution
   generalization), whereas the best on-attractor *alias* matches where it is fit and
   fails off-attractor. The off-attractor fit is therefore valid everywhere: it is the
   only place that distinguishes the global truth from a local impostor, not a regime
   artifact.

3. ``lyapunov_ics`` -- committed provenance for the "identifiability, not chaos"
   argument: the largest Lyapunov exponent over several attractor initial conditions.

Reuses the form-discovery machinery; reads the cached attractor IC read-only; fixed
seeds; shipped with test_why.py.
"""

import os
import numpy as np
import jax.numpy as jnp

from ..formfit import (twin, attractor_traj, transient_traj, true_transient,
                       err_tendency, forcing_recover, library, fit, resid,
                       margin_named, true_term)
from .. import figstyle
from ..config import RESULTS_DIR, THRESH, DT
from ..figstyle import DPI, LBL_BBOX
from ..screening import exchange_image


# --------------------------------------------------------------------------- #
# Single-named-term helpers (a recovered error model is c * g(x) * (E x)).
# --------------------------------------------------------------------------- #
def _term_col(tw, X, name):
    Phi, names = library(X, tw)
    return Phi[:, names.index(name)]


def _fit_one(tw, X, m, name):
    """Least-squares coefficient of the single term c * g_name(x) * (E x) for m."""
    pred = _term_col(tw, X, name)[:, None] * exchange_image(tw, X)        # (n, ndim) per unit c
    den = float((pred * pred).sum())
    return float((pred * m).sum() / den) if den > 0 else 0.0


def _relerr_one(tw, c, name, X, m):
    """Relative L2 error of the model c * g_name(x) * (E x) against m on X."""
    pred = c * _term_col(tw, X, name)[:, None] * exchange_image(tw, X)
    return float(np.linalg.norm(pred - m) / np.linalg.norm(m))


def _driver_meanstd(X, tw):
    x = X[:, tw.idx_To1]
    return float(abs(x.mean()) / x.std())


# --------------------------------------------------------------------------- #
# 1. The governing law: margin vs the driver's own |mean|/std.
# --------------------------------------------------------------------------- #
def governing_law(fracs=None, n=2000, reps=30, seed=0, save=True):
    """Move the training distribution from deep on the attractor (frac 0) to fully
    off-attractor (frac 1) by mixing in transient states, and for each mixture record
    the driver's own |mean|/std and the noiseless form-identifiability margin
    (restricted library). The margin tracks min(1, 1/r): off-attractor data identifies
    the form only because it lowers the driver's |mean|/std."""
    tw = twin()
    if fracs is None:
        fracs = [0.0, 5e-4, 1e-3, 1.5e-3, 2.5e-3, 4e-3, 7e-3,
                 1e-2, 2e-2, 4e-2, 1e-1, 3e-1, 1.0]
    Xatt = attractor_traj(tw, 20000)
    matt = err_tendency(tw, Xatt)
    rng = np.random.default_rng(seed)
    r = np.zeros((len(fracs), reps))
    margin = np.zeros((len(fracs), reps))
    for rep in range(reps):
        Xtr = transient_traj(tw, 6000, seed=500 + rep)
        mtr = err_tendency(tw, Xtr)
        for i, f in enumerate(fracs):
            ntr = int(round(f * n)); na = n - ntr
            ia = rng.integers(0, len(Xatt), na)
            X = [Xatt[ia]]; m = [matt[ia]]
            if ntr > 0:
                it = rng.integers(0, len(Xtr), ntr)
                X.append(Xtr[it]); m.append(mtr[it])
            X = np.concatenate(X, 0); m = np.concatenate(m, 0)
            r[i, rep] = _driver_meanstd(X, tw)
            c, nm = fit(tw, X, m)
            margin[i, rep] = margin_named(resid(tw, c, X, m), nm, true_term(tw))[0]
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez(os.path.join(RESULTS_DIR, "why_law.npz"),
                 fracs=np.asarray(fracs), r=r, margin=margin)
    print(f"{'frac':>6s} {'driver|m|/std':>13s} {'margin':>9s}")
    for i, f in enumerate(fracs):
        print(f"{f:6.3f} {np.median(r[i]):13.3f} {np.median(margin[i]):9.4f}")
    return dict(fracs=np.asarray(fracs), r=r, margin=margin)


# --------------------------------------------------------------------------- #
# 2. The recovered form is global; only its identifiability is local.
# --------------------------------------------------------------------------- #
def global_form_validation(n=2000, reps=30, seed=0, window=400, save=True):
    """Does a model recovered off-attractor predict the error on the attractor? This is
    the load-bearing test of the regime-artifact objection, so the recovered model must
    come from the paper's actual procedure, not from the algebraic identity that the
    exact error tendency *is* the true single-term model.

      - TRUE form, recovered off-attractor: run the differentiable forcing recovery on an
        excited true trajectory, let the competition select the form, and take its
        recovered coefficient (~0.098, the same ~1.7% coefficient error as E5). Then test
        that recovered model against the exact error tendency on the attractor
        (out-of-regime) and on held-out off-attractor states (in-regime).
      - best ALIAS, given every advantage: selected and fit from *exact* tendencies on
        the attractor, then tested there (in) and off-attractor (out).

    The recovered true-form model carries the same small error in both regimes, because
    the true form's optimal coefficient is global; the alias matches only on the
    attractor and is ~100% wrong off it, because its optimal coefficient is local."""
    tw = twin()
    Xatt = attractor_traj(tw, 20000)
    matt = err_tendency(tw, Xatt)
    rng = np.random.default_rng(seed)
    keys = ("true_recOff_testAtt", "true_recOff_testOff",
            "alias_fitAtt_testAtt", "alias_fitAtt_testOff")
    res = {k: np.zeros(reps) for k in keys}
    alias_names = []
    sel_correct = 0
    tname = true_term(tw)
    for rep in range(reps):
        # realistic off-attractor recovery (forcing recovery + competition, as in E5)
        Xt = true_transient(tw, window + 1, seed=700 + rep)
        dhat = forcing_recover(tw, Xt)
        c_all, nm = fit(tw, Xt[:-1], dhat)
        sel = int(np.argmin(resid(tw, c_all, Xt[:-1], dhat)))
        sel_correct += int(nm[sel] == tname)
        c_rec, rec_name = c_all[nm.index(tname)], tname     # recovered true-form coefficient
        # exact test sets: an attractor sample (out-of-regime) and held-out off-attractor (in)
        ia = rng.integers(0, len(Xatt), n); Xa, ma = Xatt[ia], matt[ia]
        Xoff = transient_traj(tw, n, seed=900 + rep); moff = err_tendency(tw, Xoff)
        res["true_recOff_testAtt"][rep] = _relerr_one(tw, c_rec, rec_name, Xa, ma)
        res["true_recOff_testOff"][rep] = _relerr_one(tw, c_rec, rec_name, Xoff, moff)
        # best ALIAS, exact fit on the attractor (its strongest possible case)
        ca, _ = fit(tw, Xa, ma)
        R = resid(tw, ca, Xa, ma); R[nm.index(tname)] = np.inf
        j = int(np.argmin(R)); aname = nm[j]; alias_names.append(aname)
        res["alias_fitAtt_testAtt"][rep] = _relerr_one(tw, ca[j], aname, Xa, ma)
        res["alias_fitAtt_testOff"][rep] = _relerr_one(tw, ca[j], aname, Xoff, moff)
    summ = {k: (float(np.median(v)), float(v.std())) for k, v in res.items()}
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez(os.path.join(RESULTS_DIR, "why_global.npz"),
                 **{k: res[k] for k in keys},
                 alias=np.asarray(alias_names), sel_correct=sel_correct, reps=reps)
    from collections import Counter
    print(f"off-attractor form selection correct: {sel_correct}/{reps}")
    print("best on-attractor alias:", Counter(alias_names).most_common(3))
    for k in keys:
        print(f"  {k:24s} relerr median {summ[k][0]:.4f}")
    return summ


# --------------------------------------------------------------------------- #
# 3. Lyapunov provenance over several attractor initial conditions.
# --------------------------------------------------------------------------- #
def lyapunov_ics(n_ic=5, block=200, n_blocks=600, drop=120, save=True):
    """Largest Lyapunov exponent of the base model over several widely-separated
    attractor initial conditions (committed provenance for "identifiability, not
    chaos"). Deterministic: ICs are sampled from the cached-IC attractor rollout."""
    from ..discovery import lyapunov_time
    from ..config import steps_to_days
    tw = twin()
    Xlong = attractor_traj(tw, 50000)
    idx = np.linspace(0, len(Xlong) - 1, n_ic).astype(int)
    lams = np.array([lyapunov_time(tw, jnp.asarray(Xlong[i]), DT,
                                   block=block, n_blocks=n_blocks, drop=drop, seed=j)[0]
                     for j, i in enumerate(idx)])
    lam_mean, lam_std = float(lams.mean()), float(lams.std())
    pred_days = float(steps_to_days((1.0 / abs(lam_mean)) / DT)) if lam_mean != 0 else np.inf
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez(os.path.join(RESULTS_DIR, "lyap_ics.npz"),
                 lambdas=lams, lam_mean=lam_mean, lam_std=lam_std,
                 predictability_days=pred_days, block=block, n_blocks=n_blocks)
    print(f"lambda over {n_ic} ICs: {lams}")
    print(f"mean {lam_mean:+.3e} +/- {lam_std:.3e} /nondim, "
          f"predictability ~ {pred_days:.0f} days")
    return dict(lambdas=lams, lam_mean=lam_mean, lam_std=lam_std,
                predictability_days=pred_days)


# --------------------------------------------------------------------------- #
def figure():
    plt = figstyle.pyplot()

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ----- (a) margin governed by the driver's own |mean|/std -------------- #
    d = np.load(os.path.join(RESULTS_DIR, "why_law.npz"))
    r, mg = d["r"], d["margin"]
    a = ax[0]
    xx = np.logspace(-2, 2, 200)
    a.fill_between(xx, np.minimum(1, 1.0 / (3 * xx)), np.minimum(1, 3.0 / xx),
                   color="grey", alpha=0.18, lw=0, label="factor of 3")
    a.plot(xx, np.minimum(1, 1.0 / xx), "k--", lw=1.3,
           label=r"$1/(|\mathrm{mean}|/\mathrm{std})$ of the driver")
    a.axhspan(8e-4, THRESH, color="C3", alpha=0.06)
    a.axhline(THRESH, ls=":", color="C3", lw=1)
    rm, mm = np.median(r, 1), np.median(mg, 1)
    a.scatter(rm, mm, s=46, color="C0", zorder=4, edgecolor="white", linewidth=0.6)
    pa = dict(arrowstyle="-", lw=0.6, color="0.4")
    a.annotate("attractor\n(slow driver pinned:\nunidentifiable)", (rm[0], mm[0]),
               xytext=(-6, 30), textcoords="offset points", fontsize=7.5,
               ha="right", va="bottom", arrowprops=pa, bbox=LBL_BBOX)
    a.annotate("off-attractor\n(driver excited:\nidentifiable)", (rm[-1], mm[-1]),
               xytext=(-2, -52), textcoords="offset points", fontsize=7.5,
               ha="center", va="top", arrowprops=pa, bbox=LBL_BBOX)
    a.text(0.03, THRESH * 1.3, "identifiable", transform=a.get_yaxis_transform(),
           fontsize=7.5, color="0.3", va="bottom")
    a.text(0.03, THRESH * 0.78, "unidentifiable", transform=a.get_yaxis_transform(),
           fontsize=7.5, color="C3", va="top")
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(8e-4, 2.0)
    a.set_xlabel(r"the driver's own $|\mathrm{mean}|/\mathrm{std}$"
                 "\n(attractor $\\to$ off-attractor)")
    a.set_ylabel("form-identifiability margin")
    a.set_title("(a)", fontsize=10, loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="lower left")

    # ----- (b) the form is global; identifiability is local ---------------- #
    g = np.load(os.path.join(RESULTS_DIR, "why_global.npz"))
    # bars ordered [in-regime, out-of-regime] for each model
    true_bars = [np.median(g["true_recOff_testOff"]), np.median(g["true_recOff_testAtt"])]
    alias_bars = [np.median(g["alias_fitAtt_testAtt"]), np.median(g["alias_fitAtt_testOff"])]
    b = ax[1]
    x = np.arange(2); w = 0.36
    b1 = b.bar(x - w / 2, true_bars, w, color="C0",
               label="true form, recovered OFF-attractor\n(noisy forcing recovery)")
    b2 = b.bar(x + w / 2, alias_bars, w, color="C3", alpha=0.85,
               label="best alias, fit ON the attractor\n(exact tendencies, best case)")
    b.set_xticks(x); b.set_xticklabels(["tested where fit\n(in-regime)",
                                        "tested in the OTHER regime\n(out-of-regime)"],
                                       fontsize=8)
    b.set_ylabel("relative error of the recovered\nerror tendency (higher = worse)")
    b.set_title("(b)", fontsize=10, loc="left")
    b.legend(fontsize=7.5, frameon=False, loc="upper left")
    b.annotate("worse", xy=(0.012, 0.96), xytext=(0.012, 0.55),
               xycoords="axes fraction", textcoords="axes fraction",
               fontsize=7.5, color="0.45", ha="center", va="center", rotation=90,
               arrowprops=dict(arrowstyle="->", lw=1.0, color="0.45"))
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            b.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                       xytext=(0, 2), textcoords="offset points", ha="center",
                       fontsize=7.5)
    b.set_ylim(0, max(alias_bars + true_bars) * 1.25)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_why.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return [out]


def run():
    governing_law()
    global_form_validation()
    lyapunov_ics()
