"""
error_discovery.experiment_form
==============================

Discovering the model error when its *functional form* is unknown (experiments
E1, E2, E3, E5; see NOTES.md). The discovery model no longer knows that
g = T_{o,1}; it is given an over-complete predictor library and must find which
term(s) explain the error tendency m(x) = eps * g(x) * (E x).

Truth: g = T_{o,1} = x[28] (eps=0.1).  Diagnostics are model-agnostic residuals;
"discovery" is judged by whether the recovered form *generalises off the
attractor*, not by attractor fit.

E1  identifiability vs noise, on- vs off-attractor (bootstrap CIs)
E2  identifiability vs amount of off-attractor excitation
E3  differentiable sparse discovery from noisy trajectory observations
E5  form-agnostic recovery of a free forcing, then regression onto the library
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

from .models import TwinModels
from .discovery import rollout
from .config import RESULTS_DIR
from .figstyle import DPI

DT = 0.1
EPS = 0.1


def twin():
    return TwinModels(eps=EPS, g_coeffs=(0.0, 0.0, 1.0, 0.0))   # truth g = x[28]


def library(X, tw):
    """Over-complete predictor library: const + 36 linear + 4 quadratics."""
    vr = tw.spec.variables_range
    cols = [np.ones(len(X))] + [X[:, k] for k in range(tw.ndim)]
    names = ["const"] + [f"x{k}" for k in range(tw.ndim)]
    for a, nm in [(0, "psi_a1^2"), (tw.idx_thetaa1, "th_a1^2"),
                  (tw.idx_To1, "T_o1^2"), (vr[1], "psi_o1^2")]:
        cols.append(X[:, a] ** 2)
        names.append(nm)
    return np.stack(cols, 1), names


def true_term(tw):
    return f"x{tw.idx_To1}"


def err_tendency(tw, X):
    g = jax.vmap(lambda x: tw.f_true(x) - tw.f_base(x))
    return np.asarray(g(jnp.asarray(X)))


def _fit(tw, X, m):
    """Best single-term coefficient c_j for every library predictor j (vectorised)."""
    Phi, names = library(X, tw)
    EX = X @ np.asarray(tw.E_full).T
    den = (Phi ** 2).T @ (EX * EX).sum(1)
    num = Phi.T @ (EX * m).sum(1)
    c = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return c, names


def _resid(tw, c, X, m):
    """Relative residual ‖c_j phi_j(X)(E X) − m‖/‖m‖ for each predictor j (fixed c_j)."""
    Phi, _ = library(X, tw)
    EX = X @ np.asarray(tw.E_full).T
    aa = (Phi ** 2).T @ (EX * EX).sum(1)
    am = Phi.T @ (EX * m).sum(1)
    m2 = float((m * m).sum())
    return np.sqrt(np.maximum(c ** 2 * aa - 2 * c * am + m2, 0.0)) / np.sqrt(m2)


def attractor_traj(tw, n):
    from .experiment import get_attractor_ic
    from .config import ExperimentConfig
    xa = get_attractor_ic(tw, ExperimentConfig(), verbose=False)   # canonical, provenance-checked
    return np.asarray(rollout(lambda x: tw.f_base(x), xa, DT, n))


def transient_traj(tw, n, seed):
    x0 = jnp.asarray(np.random.default_rng(seed).standard_normal(tw.ndim) * 0.5)
    return np.asarray(rollout(lambda x: tw.f_base(x), x0, DT, n))


def true_transient(tw, n, seed):
    """An excited trajectory of the TRUE (perturbed) model — used where the error
    must manifest dynamically (E3, E5)."""
    x0 = jnp.asarray(np.random.default_rng(seed).standard_normal(tw.ndim) * 0.5)
    return np.asarray(rollout(lambda x: tw.f_true(x), x0, DT, n))


def _margin(R, names, tw):
    """Best wrong-form residual minus truth residual (>0 means truth wins)."""
    t = names.index(true_term(tw))
    others = np.delete(R, t)
    return float(others.min() - R[t]), int(np.argmin(R) == t)


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
            c, nm = _fit(tw, Xtr, mtr + e)              # coefficients fit on attractor
            insample[i, r] = _margin(_resid(tw, c, Xtr, mtr + e), nm, tw)[0]
            generalise[i, r] = _margin(_resid(tw, c, Xte, mte), nm, tw)[0]
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
            c, nm = _fit(tw, X, m + e)
            margin[i, r], correct[i, r] = _margin(_resid(tw, c, X, m + e), nm, tw)
    return dict(fracs=np.asarray(fracs), noise=noise, margin=margin, correct=correct)


# --------------------------------------------------------------------------- #
# Tendency estimators from an observed trajectory (shared by E3 and E5).
# --------------------------------------------------------------------------- #
def fd_tendency(tw, Y):
    """Naive finite-difference error tendency:  (y' - y)/dt - f_base(y)."""
    fbase = np.asarray(jax.vmap(lambda x: tw.f_base(x))(jnp.asarray(Y[:-1])))
    return (Y[1:] - Y[:-1]) / DT - fbase


def forcing_recover(tw, Y):
    """Form-agnostic per-step forcing: the free delta_n with
    RK4(f_base + delta_n, y_n) = y_{n+1}. Removes the base-dynamics truncation
    that swamps the tiny error in a naive finite difference; delta_n approximates
    m(y_n) to first order in the RK4 stages (the true error varies within a step)."""
    from qgs_jax.model import rk4_step
    Yn = jnp.asarray(Y[:-1])
    step0 = jax.vmap(lambda x: rk4_step(lambda z: tw.f_base(z), x, DT))
    M = jax.vmap(lambda x: jax.jacfwd(lambda d: rk4_step(lambda z: tw.f_base(z) + d, x, DT))
                 (jnp.zeros(tw.ndim)))(Yn)
    r = jnp.asarray(Y[1:]) - step0(Yn)
    return np.asarray(jax.vmap(jnp.linalg.solve)(M, r))


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
                c, nm = _fit(tw, Y[:-1], mhat)
                out[key][i, r] = int(np.argmin(_resid(tw, c, Y[:-1], mhat))
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
        c, nm = _fit(tw, Xt[:-1], dhat)
        correct[r] = int(np.argmin(_resid(tw, c, Xt[:-1], dhat)) == nm.index(true_term(tw)))
        cerr[r] = abs(c[nm.index(true_term(tw))] - EPS) / EPS
    return dict(rel_delta=rel, correct=correct, c_err=cerr)


# --------------------------------------------------------------------------- #
# Reanalysis test: is the TRUE attractor ("reanalysis") far enough from the
# biased climatology to identify the error form? Margin of g=T_{o,1} vs best
# alias on the biased climatology, the true climatology, and nudged reanalysis
# (biased model relaxed toward true obs at strength tau), swept over error
# strength eps. Reuses E1's _fit/_resid/_margin.
# --------------------------------------------------------------------------- #
def _margin_of(tw, X):
    m = err_tendency(tw, X)
    c, nm = _fit(tw, X, m)
    return _margin(_resid(tw, c, X, m), nm, tw)[0], float(X[:, tw.idx_To1].std())


def _nudge(tw, x0, Xref, tau):
    """Biased model relaxed toward a true reference trajectory: reanalysis."""
    from qgs_jax.model import rk4_step
    def step(x, xr):
        xn = rk4_step(lambda z: tw.f_base(z) + (xr - z) / tau, x, DT)
        return xn, xn
    _, traj = jax.lax.scan(step, x0, jnp.asarray(Xref))
    return np.asarray(traj)


def reanalysis(eps_list=(0.1, 0.3, 1.0), taus=(100.0, 20.0, 5.0), n=4000, n_spin=40000):
    from .experiment import get_attractor_ic
    from .config import ExperimentConfig
    base = twin()
    xa = get_attractor_ic(base, ExperimentConfig(), verbose=False)
    Xb = np.asarray(rollout(lambda x: base.f_base(x), xa, DT, n))   # biased climatology
    biased = _margin_of(base, Xb)
    rows = []
    for eps in eps_list:
        tw = TwinModels(eps=eps, g_coeffs=(0., 0., 1., 0.))
        xt0 = jnp.asarray(np.asarray(rollout(lambda x: tw.f_true(x), xa, DT, n_spin))[-1])
        Xtrue = np.asarray(rollout(lambda x: tw.f_true(x), xt0, DT, n))    # true climatology
        nud = [(tau, *_margin_of(tw, _nudge(tw, xt0, Xtrue, tau))) for tau in taus]
        rows.append(dict(eps=eps, true=_margin_of(tw, Xtrue), nudged=nud))
    # off-attractor contrast: a transient distribution (the excitation that works)
    transient = _margin_of(base, transient_traj(base, n, seed=0))
    return dict(biased=biased, rows=rows, transient=transient)


def reanalysis_run():
    r = reanalysis()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, "reanalysis_results.npz"),
             biased=np.asarray(r["biased"]), transient=np.asarray(r["transient"]),
             eps=np.asarray([x["eps"] for x in r["rows"]]),
             true=np.asarray([x["true"] for x in r["rows"]]),
             # (tau, margin, std) per nudging timescale; the figure labels itself with tau
             nudged=np.asarray([x["nudged"] for x in r["rows"]]))
    print(f"biased climatology : margin {r['biased'][0]:+.3f}  T_o1_std {r['biased'][1]:.4f}")
    for x in r["rows"]:
        print(f"eps={x['eps']}: true {x['true'][0]:+.3f} (std {x['true'][1]:.4f})  "
              f"nudged {[round(n[1], 3) for n in x['nudged']]}")
    print(f"transient (off-attr): margin {r['transient'][0]:+.3f}  T_o1_std {r['transient'][1]:.4f}")
    return r


def reanalysis_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(os.path.join(RESULTS_DIR, "reanalysis_results.npz"))
    nud = d["nudged"][-1]                                # strongest error (eps=1.0)
    labels = ["biased\nclimatology", 'true climatology\n("reanalysis")',
              f"nudged reanalysis\n(τ={nud[-1][0]:.0f})", "transient\n(off-attractor)"]
    vals = [d["biased"][0], d["true"][-1][0], nud[-1][1], d["transient"][0]]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(4), vals, color=["C0", "C0", "C0", "C2"])
    ax.axhline(0.1, ls=":", color="grey", lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("form-identifiability margin")
    ax.set_title(r"Is reanalysis far enough to identify $g=T_{o,1}$?")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_reanalysis.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return [out]


# --------------------------------------------------------------------------- #
# Driver: run all four experiments and save; the figures are drawn separately.
# --------------------------------------------------------------------------- #
def run():
    d1, d2, d3, d5 = e1(), e2(), e3(), e5()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, "form_results.npz"),
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
    z = np.load(os.path.join(RESULTS_DIR, "form_results.npz"))
    d = {"e1": {}, "e2": {}, "e3": {}, "e5": {}}
    for key in z.files:
        exp, _, name = key.partition("_")
        d[exp][name] = z[key]
    return d["e1"], d["e2"], d["e3"]


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    ident = os.path.join(RESULTS_DIR, "fig_form_identifiability.png")
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
    disc = os.path.join(RESULTS_DIR, "fig_form_discovery.png")
    fig.tight_layout(); fig.savefig(disc, dpi=DPI)
    plt.close(fig)
    return [ident, disc]
