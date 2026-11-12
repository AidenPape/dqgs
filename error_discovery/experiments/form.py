"""
error_discovery.experiments.form
================================

Discovering the model error when its *functional form* is unknown (experiments
E1, E2, E3, E5). The discovery model no longer knows that
g = T_{o,1}; it is given an over-complete predictor library and must find which
term(s) explain the error tendency m(x) = eps * g(x) * (E x).

Truth: g = T_{o,1} = x[28] (eps=0.1).  Diagnostics are model-agnostic residuals;
"discovery" is judged by whether the recovered form *generalises off the
attractor*, not by attractor fit.

E1  identifiability vs noise, on- vs off-attractor (bootstrap CIs)
E2  identifiability vs amount of off-attractor excitation
E3  differentiable sparse discovery from noisy trajectory observations
E5  form-agnostic recovery of a free forcing, then regression onto the library

The twin, the sampling distributions and the library scoring all come from
:mod:`error_discovery.formfit`; the reanalysis variant of the same question is
:mod:`error_discovery.experiments.reanalysis`.
"""

import os
import numpy as np
import jax

from .. import figstyle
from ..config import DATA_DIR, FIGURES_DIR, EPS
from ..figstyle import DPI
from ..formfit import (attractor_traj, transient_traj, true_transient, twin,
                       err_tendency, fd_tendency, forcing_recover,
                       fit, resid, margin_named, true_term)


# --------------------------------------------------------------------------- #
# E1 — identifiability vs noise: attractor ambiguity vs off-attractor validation
# --------------------------------------------------------------------------- #
def e1(noises=(0.0, 0.01, 0.02, 0.05, 0.1), n=2000, reps=40, seed=0):
    """Fit each predictor's coefficient ON the attractor, then measure the margin
    two ways: in-set on the attractor (ambiguity — many aliases fit) and on a
    held-out transient test set where T_{o,1} actually varies (a true form keeps
    R≈0, aliases fail). Also reports the dynamic range of T_{o,1} on each set."""
    tw = twin()
    Xlong = attractor_traj(tw, 20000)
    mlong = err_tendency(tw, Xlong)
    rng = np.random.default_rng(seed)
    insample = np.zeros((len(noises), reps))
    generalise = np.zeros((len(noises), reps))
    for r in range(reps):
        idx = rng.integers(0, len(Xlong), n)            # bootstrap attractor sample
        Xtr, mtr = Xlong[idx], mlong[idx]
        Xte = transient_traj(tw, 4000, seed=200 + r)[:n]   # held-out off-attractor
        mte = err_tendency(tw, Xte)
        for i, nz in enumerate(noises):
            e = nz * mtr.std() * rng.standard_normal(mtr.shape)
            c, nm = fit(tw, Xtr, mtr + e)              # coefficients fit on attractor
            insample[i, r] = margin_named(resid(tw, c, Xtr, mtr + e), nm, true_term(tw))[0]
            generalise[i, r] = margin_named(resid(tw, c, Xte, mte), nm, true_term(tw))[0]
    To = tw.idx_To1
    rng_att = Xlong[:, To].std()
    rng_tr = np.mean([transient_traj(tw, 4000, seed=200 + k)[:, To].std() for k in range(5)])
    return dict(noises=np.asarray(noises), insample=insample, generalise=generalise,
                To1_std_attractor=float(rng_att), To1_std_transient=float(rng_tr))


# --------------------------------------------------------------------------- #
# E2 — identifiability vs amount of off-attractor excitation
# --------------------------------------------------------------------------- #
def e2(fracs=(0.0, 0.1, 0.25, 0.5, 0.75), n=2000, noise=0.02, reps=40, seed=1):
    tw = twin()
    Xatt = attractor_traj(tw, 20000)
    matt = err_tendency(tw, Xatt)
    rng = np.random.default_rng(seed)
    margin = np.zeros((len(fracs), reps))
    correct = np.zeros((len(fracs), reps))
    for r in range(reps):
        Xtr = transient_traj(tw, 4000, seed=100 + r)[:2000]   # off-attractor source
        mtr = err_tendency(tw, Xtr)
        for i, f in enumerate(fracs):
            ntr = int(f * n)
            ia = rng.integers(0, len(Xatt), n - ntr)
            it = rng.integers(0, len(Xtr), ntr)
            X = np.concatenate([Xatt[ia], Xtr[it]], 0)
            m = np.concatenate([matt[ia], mtr[it]], 0)
            e = noise * m.std() * rng.standard_normal(m.shape)
            c, nm = fit(tw, X, m + e)
            margin[i, r], correct[i, r] = margin_named(resid(tw, c, X, m + e), nm, true_term(tw))
    return dict(fracs=np.asarray(fracs), noise=noise, margin=margin, correct=correct)


# --------------------------------------------------------------------------- #
# E3 — discovery from noisy observations: naive FD vs differentiable forcing
# recovery as the tendency estimator, then sparse competition, vs obs noise.
# --------------------------------------------------------------------------- #
def e3(obs_noises=(0.0, 1e-4, 3e-4, 1e-3, 3e-3), window=400, reps=30, seed=2):
    tw = twin()
    rng = np.random.default_rng(seed)
    out = {k: np.zeros((len(obs_noises), reps)) for k in ("fd", "forcing")}
    for r in range(reps):
        Xt = true_transient(tw, window + 1, seed=300 + r)          # excited true trajectory
        sd = Xt.std(0)
        for i, nz in enumerate(obs_noises):
            Y = Xt + nz * sd * rng.standard_normal(Xt.shape)       # noisy observation
            for key, mhat in (("fd", fd_tendency(tw, Y)),
                              ("forcing", forcing_recover(tw, Y))):
                c, nm = fit(tw, Y[:-1], mhat)
                out[key][i, r] = int(np.argmin(resid(tw, c, Y[:-1], mhat))
                                     == nm.index(true_term(tw)))
    return dict(obs_noises=np.asarray(obs_noises), **{f"recover_{k}": v for k, v in out.items()})


# --------------------------------------------------------------------------- #
# E5 — the form-agnostic forcing recovery characterised: delta_hat vs the true
# error tendency, and the form recovered by regressing delta_hat on the library.
# --------------------------------------------------------------------------- #
def e5(window=400, reps=30, seed=4):
    tw = twin()
    rel = np.zeros(reps); cerr = np.zeros(reps); correct = np.zeros(reps)
    for r in range(reps):
        Xt = true_transient(tw, window + 1, seed=seed + r)
        m = err_tendency(tw, Xt[:-1])
        dhat = forcing_recover(tw, Xt)
        rel[r] = np.linalg.norm(dhat - m) / np.linalg.norm(m)
        c, nm = fit(tw, Xt[:-1], dhat)
        correct[r] = int(np.argmin(resid(tw, c, Xt[:-1], dhat)) == nm.index(true_term(tw)))
        cerr[r] = abs(c[nm.index(true_term(tw))] - EPS) / EPS
    return dict(rel_delta=rel, correct=correct, c_err=cerr)

# --------------------------------------------------------------------------- #
# Driver: run all four experiments and save; the figures are drawn separately.
# --------------------------------------------------------------------------- #
def run():
    d1, d2, d3, d5 = e1(), e2(), e3(), e5()
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(os.path.join(DATA_DIR, "form_results.npz"),
             **{f"e1_{k}": v for k, v in d1.items()},
             **{f"e2_{k}": v for k, v in d2.items()},
             **{f"e3_{k}": v for k, v in d3.items()},
             **{f"e5_{k}": v for k, v in d5.items()})
    print(f"E1 margin@noise0: attractor {np.median(d1['insample'][0]):.3f}  transient {np.median(d1['generalise'][0]):.2f}")
    print(f"E2 margin: frac0 {np.median(d2['margin'][0]):.3f}  frac0.25 {np.median(d2['margin'][2]):.2f}")
    print(f"E3 recover@1e-3: FD {d3['recover_fd'][3].mean():.2f}  forcing {d3['recover_forcing'][3].mean():.2f}")
    print(f"E5 rel|delta-m| {np.median(d5['rel_delta']):.3f}  coeff err {np.median(d5['c_err']):.3f}")
    return d1, d2, d3, d5


def _load_form():
    """Regroup ``form_results.npz`` back into the per-experiment dicts ``run`` saved."""
    z = np.load(os.path.join(DATA_DIR, "form_results.npz"))
    d = {"e1": {}, "e2": {}, "e3": {}, "e5": {}}
    for key in z.files:
        exp, _, name = key.partition("_")
        d[exp][name] = z[key]
    return d["e1"], d["e2"], d["e3"]


def figure():
    plt = figstyle.pyplot()

    d1, d2, d3 = _load_form()

    def band(ax, x, A, **kw):
        ax.plot(x, np.median(A, 1), "o-", **kw)
        ax.fill_between(x, np.percentile(A, 25, 1), np.percentile(A, 75, 1),
                        alpha=0.2, color=kw.get("color"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    band(ax[0], d1["noises"], d1["insample"], color="C1", label="attractor (in-sample)")
    band(ax[0], d1["noises"], d1["generalise"], color="C0", label="transient (generalisation)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("tendency noise"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_ylabel("truth margin over best wrong form")
    ax[0].set_title("(a)", fontsize=10, loc="left")
    band(ax[1], d2["fracs"], d2["margin"], color="C2")
    ax[1].set_xlabel("off-attractor (transient) fraction of training data")
    ax[1].set_ylabel("truth margin over best wrong form")
    ax[1].set_title("(b)", fontsize=10, loc="left")
    ident = os.path.join(FIGURES_DIR, "fig_form_identifiability.png")
    fig.tight_layout(); fig.savefig(ident, dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    xn = np.where(d3["obs_noises"] > 0, d3["obs_noises"], d3["obs_noises"][d3["obs_noises"] > 0].min() / 3)
    ax.plot(xn, d3["recover_fd"].mean(1), "s-", label="naive finite difference")
    ax.plot(xn, d3["recover_forcing"].mean(1), "o-", label="differentiable forcing recovery")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("observation noise (leftmost = noiseless)")
    ax.set_ylabel("P(recover T$_{o,1}$ as top term)")
    ax.legend(frameon=False, fontsize=9)
    disc = os.path.join(FIGURES_DIR, "fig_form_discovery.png")
    fig.tight_layout(); fig.savefig(disc, dpi=DPI)
    plt.close(fig)
    return [ident, disc]
