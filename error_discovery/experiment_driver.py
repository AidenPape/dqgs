"""
error_discovery.experiment_driver
=================================

Does the identifiability boundary depend on *which* mode drives the injected
error? The governing law (``experiment_why.governing_law``) says the
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

Reuses the form-discovery machinery (library/_fit/_resid are driver-independent:
they depend only on the constant exchange operator E, not on g); reads the cached
attractor IC read-only; fixed seeds; shipped with test_driver.py.
"""

import os
import numpy as np

from .models import TwinModels
from .experiment_form import (attractor_traj, transient_traj, err_tendency,
                              library, _fit, _resid, EPS)
from .config import RESULTS_DIR

# faint white backing for annotations that sit near curves (mask the line behind the text)
_LBL_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0)

# name, g_coeffs, true library term, driver state index (None for the constant)
DRIVERS = [
    ("psi_a1\n(atmos streamfn)",  (0.0, 0.0, 0.0, 1.0), "x0",  0),
    ("theta_a1\n(atmos temp)",    (0.0, 1.0, 0.0, 0.0), "x10", 10),
    ("T_o1\n(ocean temp)",        (0.0, 0.0, 1.0, 0.0), "x28", 28),
    ("const\n(g = 1)",            (1.0, 0.0, 0.0, 0.0), "const", None),
]


def _margin_named(R, names, true_name):
    """Best wrong-form residual minus the named truth's residual (>0 = truth wins)."""
    t = names.index(true_name)
    others = np.delete(R, t)
    return float(others.min() - R[t]), int(np.argmin(R) == t)


def driver_scan(n=4000, reps=20, noise=0.02, seed=0, save=True):
    """For each driver, measure form-identifiability ON the attractor (bootstrap
    samples of the base-model climatology) vs the driver's own |mean|/std, both
    noiseless and at a representative tendency noise, plus the off-attractor margin
    for contrast. The attractor distribution and E are shared across drivers."""
    rng = np.random.default_rng(seed)
    tw0 = TwinModels(eps=EPS, g_coeffs=(0.0, 0.0, 1.0, 0.0))
    Xatt = attractor_traj(tw0, 20000)               # base-model climatology (driver-independent)
    Xoff = transient_traj(tw0, n, seed=42)          # off-attractor contrast
    rows = []
    for name, gc, tname, didx in DRIVERS:
        tw = TwinModels(eps=EPS, g_coeffs=gc)
        matt = err_tendency(tw, Xatt)
        moff = err_tendency(tw, Xoff)
        if didx is None:
            r_att = np.inf
        else:
            r_att = float(abs(Xatt[:, didx].mean()) / Xatt[:, didx].std())
        mg0 = np.zeros(reps); ok0 = np.zeros(reps)
        mgN = np.zeros(reps); okN = np.zeros(reps)
        for rep in range(reps):
            ia = rng.integers(0, len(Xatt), n)
            X, m = Xatt[ia], matt[ia]
            c, nm = _fit(tw, X, m)
            mg0[rep], ok0[rep] = _margin_named(_resid(tw, c, X, m), nm, tname)
            e = noise * m.std() * rng.standard_normal(m.shape)
            cN, nmN = _fit(tw, X, m + e)
            mgN[rep], okN[rep] = _margin_named(_resid(tw, cN, X, m + e), nmN, tname)
        coff, nmoff = _fit(tw, Xoff, moff)
        mg_off = _margin_named(_resid(tw, coff, Xoff, moff), nmoff, tname)[0]
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
                 margin_off=np.asarray([r["margin_off"] for r in rows]))
    print(f"{'driver':18s} {'|mean|/std':>10s} {'1/r':>7s} "
          f"{'marg_att':>9s} {'corr':>5s} {'marg_noisy':>11s} {'marg_off':>9s}")
    for r in rows:
        inv = 0.0 if not np.isfinite(r["r_att"]) else 1.0 / r["r_att"]
        print(f"{r['name'].replace(chr(10),' '):18s} {r['r_att']:10.3f} {inv:7.3f} "
              f"{r['margin_att']:9.3f} {r['correct_att']:5.2f} "
              f"{r['margin_att_noisy']:11.3f} {r['margin_off']:9.3f}")
    return rows


def figure(rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if rows is None:
        d = np.load(os.path.join(RESULTS_DIR, "driver_scan.npz"), allow_pickle=True)
        rows = [dict(name=n.replace(" ", "\n", 1), r_att=ra, margin_att=ma,
                     margin_att_noisy=mn, margin_off=mo)
                for n, ra, ma, mn, mo in zip(d["names"], d["r_att"], d["margin_att"],
                                             d["margin_att_noisy"], d["margin_off"])]
    THRESH = 0.1
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
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


RL_DRIVERS = [
    ("psi_a1",   (0.0, 0.0, 0.0, 1.0), "x0",  0,  "atmospheric circulation", "C0", "o-"),
    ("theta_a1", (0.0, 1.0, 0.0, 0.0), "x10", 10, "atmospheric temperature", "C9", "s--"),
    ("T_o1",     (0.0, 0.0, 1.0, 0.0), "x28", 28, "ocean temperature",       "C3", "D-"),
]


def record_length_scan(Ns=(2000, 4000, 8000, 20000, 40000, 80000), save=True):
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
            c, nm = _fit(tw, X, m)
            margin[di, ni] = _margin_named(_resid(tw, c, X, m), nm, tname)[0]
            rr[di, ni] = float(abs(X[:, didx].mean()) / X[:, didx].std())
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez(os.path.join(RESULTS_DIR, "driver_record.npz"),
                 Ns=np.asarray(Ns), margin=margin, meanstd=rr,
                 names=np.asarray([d[0] for d in RL_DRIVERS]))
    print(f"{'N':>7s} " + " ".join(f"{d[0]:>20s}" for d in RL_DRIVERS))
    for ni, N in enumerate(Ns):
        print(f"{N:7d} " + " ".join(f"r={rr[di,ni]:6.1f} m={margin[di,ni]:5.3f}"
                                    for di in range(len(RL_DRIVERS))))
    return dict(Ns=np.asarray(Ns), margin=margin, meanstd=rr)


def headline_figure(save=True):
    """The headline: discoverability of an injected error form is set by the driver's
    |mean|/std, and the two fluids move in opposite directions as the record lengthens.
    (a) margin versus record length: atmospheric drivers rise across the threshold,
    ocean temperature stays below it. (b) why: the ocean-temperature driver is a needle
    pinned ~40 std from zero, the atmospheric-circulation driver explores a broad range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(os.path.join(RESULTS_DIR, "driver_record.npz"), allow_pickle=True)
    Ns, margin = d["Ns"], d["margin"]
    THRESH = 0.1
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
    tw0 = TwinModels(eps=EPS, g_coeffs=(0.0, 0.0, 1.0, 0.0))
    X = attractor_traj(tw0, int(Ns[-1]))
    b = ax[1]
    panels = [(0, "C0", r"$\psi_{a,1}$ atmospheric circulation"),
              (28, "C3", r"$T_{o,1}$ ocean temperature")]
    for didx, col, lab in panels:
        v = X[:, didx]
        vn = (v - 0.0) / v.std()                    # in std units, so zero stays at 0
        b.hist(vn, bins=80, density=True, color=col, alpha=0.5, label=lab)
    b.axvline(0.0, color="0.4", lw=1.4)
    b.text(0.0, b.get_ylim()[1] * 0.96, " zero", color="0.4", fontsize=8, va="top")
    for didx, col, lab in panels:
        r = abs(X[:, didx].mean()) / X[:, didx].std()
        b.annotate(rf"$|\mathrm{{mean}}|/\mathrm{{std}}={r:.0f}$", xy=(r, 0),
                   xytext=(r, b.get_ylim()[1] * (0.55 if col == "C3" else 0.78)),
                   color=col, fontsize=9, ha="center",
                   arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
    b.set_xlabel("driver amplitude (standard deviations from zero)")
    b.set_ylabel("density over the climatology")
    b.set_title("(b)", fontsize=10, loc="left")
    b.legend(fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_driver.png")
    if save:
        fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def tandem_figure(n=20000, save=True):
    """The headline contrast on the honest diagnostic. Row 1: the driver's
    distribution over the climatology, with the mean's distance from zero measured
    in standard deviations (= |mean|/std). A broadly-exploring atmospheric-circulation
    mode sits a few std from zero; the ocean-temperature mode is a needle pinned ~40
    std away by thermal damping. Row 2: the consequence -- the pinned mode's true
    error form is buried among near-constant aliases, the broad mode's form stands
    alone. Same 20k-step record for both, so |mean|/std matches ``driver_scan``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tw0 = TwinModels(eps=EPS, g_coeffs=(0.0, 0.0, 1.0, 0.0))
    Xatt = attractor_traj(tw0, n)
    pairs = [
        (r"$\psi_{a,1}$ — atmospheric circulation", (0.0, 0.0, 0.0, 1.0), "x0", 0, "C0"),
        (r"$T_{o,1}$ — ocean temperature",          (0.0, 0.0, 1.0, 0.0), "x28", 28, "C3"),
    ]
    THRESH = 0.1
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.9))
    jit = np.random.default_rng(0)
    for col, (title, gc, tname, didx, color) in enumerate(pairs):
        tw = TwinModels(eps=EPS, g_coeffs=gc)
        m = err_tendency(tw, Xatt)
        c, nm = _fit(tw, Xatt, m)
        R = _resid(tw, c, Xatt, m)
        ti = nm.index(tname)
        d = Xatt[:, didx]
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
        fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# the known mode-1 driver set assumed for the mixed-driver coefficient recovery:
# (const, psi_a1, theta_a1, T_o1). The form is assumed, exactly as in the canonical
# single-driver experiment; the mixed test asks whether the *coefficients* of several
# simultaneously-active drivers are each recoverable, i.e. whether the clean result is
# an artifact of having only one active driver. Fitting the full 36-mode linear library
# instead would confound this sparsity question with the separate form-discovery question
# (Sec. boundary) and is severely multicollinear through the near-constant ocean modes.
_MIXED_MODES = [("const", None), ("x0", 0), ("x10", 10), ("x28", 28)]


def _joint_fit_restricted(tw, X, mhat):
    """Joint weighted least squares for the mixed-driver coefficient recovery: assume the
    error is gated by the mode-1 driver set {1, psi_a1, theta_a1, T_o1} and recover its
    coefficient vector c with mhat_i approx (c . phi)(E x)_i. Because every term shares the
    same (E x) multiplier, the objective sum_i ((c.phi)(Ex)_i - mhat_i)^2 reduces to a
    |Ex|^2-weighted regression of alpha = (Ex.mhat)/|Ex|^2 onto phi, i.e. min ||A c - t||^2
    with A = sqrt(w) phi, t = sqrt(w) alpha, w = |Ex|^2. Solved by SVD least squares
    (np.linalg.lstsq): the design is well conditioned (cond(A) ~ 2e3) and recovers the exact
    c_true to machine precision in the noiseless limit, with the noise-driven coefficient
    spread ordered by each driver's own |mean|/std (experiment_oracle)."""
    n = X.shape[0]
    cols = [np.ones(n) if k is None else X[:, k] for _, k in _MIXED_MODES]
    phi = np.stack(cols, axis=1)                              # (n, P)
    names = [nm for nm, _ in _MIXED_MODES]
    EX = X @ np.asarray(tw.E_full).T
    w = (EX * EX).sum(1)                                       # (n,)
    s = (EX * mhat).sum(1)                                     # (n,)
    sw = np.sqrt(np.maximum(w, 0.0))
    A = sw[:, None] * phi                                      # (n, P)
    t = np.where(w > 0, s / np.where(w > 0, w, 1.0), 0.0) * sw # (n,)
    c, *_ = np.linalg.lstsq(A, t, rcond=None)
    return c, names


DECOMP_COMPS = [("psi_a1", 0, "C0", r"$\psi_{a,1}$ atmos. circ."),
                ("theta_a1", 10, "C1", r"$\theta_{a,1}$ atmos. temp."),
                ("T_o1", 28, "C3", r"$T_{o,1}$ ocean temp."),
                ("const", None, "0.6", "const (not in truth)")]


def full_error_decomposition(noises=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2), n=8000, reps=40,
                             seed=0, record=40000, save=True):
    """Inject the FULL mode-1 error g = theta_a1 + T_o1 + psi_a1 (all three drivers at
    eps=0.1) and recover the coefficients of the known mode-1 driver set jointly from the
    climatology (SVD least squares), as a function of tendency noise. The blind spot is not
    assumed but emerges as a conditioning effect: in the noiseless limit all coefficients
    are recovered exactly, and under noise the atmospheric-circulation coefficient stays
    pinned near 0.1 at every level while the ocean-temperature coefficient is the least
    determined, its bootstrap spread growing fastest (to ~half the signal by 20% noise).
    The amplification tracks the driver's own |mean|/std (its conditioning). A purely
    atmospheric error is therefore the optimistic case; the ocean-temperature component is
    the one a differentiable model loses first."""
    tw = TwinModels(eps=EPS, g_coeffs=(0.0, 1.0, 1.0, 1.0))   # (const, theta_a1, T_o1, psi_a1)
    Xall = attractor_traj(tw, record)
    mall = err_tendency(tw, Xall)
    rng = np.random.default_rng(seed)
    noises = list(noises)
    chat = {nm: np.zeros((len(noises), reps)) for nm, _, _, _ in DECOMP_COMPS}
    for ni, noise in enumerate(noises):
        for r in range(reps):
            idx = rng.integers(0, len(Xall), n)
            X, m = Xall[idx], mall[idx]
            e = noise * m.std() * rng.standard_normal(m.shape)
            c, names = _joint_fit_restricted(tw, X, m + e)
            for nm, k, _, _ in DECOMP_COMPS:
                chat[nm][ni, r] = c[names.index("const" if k is None else f"x{k}")]
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez(os.path.join(RESULTS_DIR, "driver_full.npz"),
                 noises=np.asarray(noises), eps=EPS, record=record,
                 **{nm: chat[nm] for nm, _, _, _ in DECOMP_COMPS})
    print(f"full error g = theta_a1 + T_o1 + psi_a1 (each eps={EPS}); joint recovery from "
          f"a {record}-step climatology")
    print(f"{'noise':>6s} " + " ".join(f"{nm:>18s}" for nm, _, _, _ in DECOMP_COMPS))
    for ni, noise in enumerate(noises):
        print(f"{noise:6.2f} " + " ".join(
            f"{np.median(chat[nm][ni]):.3f}+-{chat[nm][ni].std():.3f}"
            for nm, _, _, _ in DECOMP_COMPS))
    return dict(noises=np.asarray(noises), chat=chat, eps=EPS)


def decomposition_figure(save=True):
    """Two panels for the full mode-1 error recovered jointly from the climatology vs
    tendency noise. (a) recovered coefficient with bootstrap 5-95 band: all three centre
    near eps, but the bands differ. (b) the bootstrap spread itself, log scale: the clean
    conditioning ordering psi_a1 < theta_a1 < T_o1 at every noise, the ocean-temperature
    coefficient several times the least-determined, lost first as noise grows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(os.path.join(RESULTS_DIR, "driver_full.npz"))
    noises, eps = d["noises"], float(d["eps"])
    x = 100.0 * noises
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # (a) recovered coefficient with band
    a = ax[0]
    a.axhline(eps, ls="--", color="k", lw=1, zorder=1,
              label=rf"true ($\varepsilon={eps}$)")
    a.axhline(0.0, color="0.5", lw=0.8, zorder=1)
    for nm, _, col, lab in DECOMP_COMPS:
        if nm == "const":
            continue
        med = np.median(d[nm], 1)
        lo = np.percentile(d[nm], 5, 1); hi = np.percentile(d[nm], 95, 1)
        a.plot(x, med, "o-", color=col, lw=1.6, ms=4, label=lab, zorder=3)
        a.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0, zorder=2)
    a.set_xlabel("tendency noise (% of error-tendency std)")
    a.set_ylabel("recovered coefficient")
    a.set_ylim(-0.15, 0.32)
    a.set_title("(a)", fontsize=10, loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)

    # (b) the bootstrap spread itself, the conditioning ordering
    b = ax[1]
    for nm, _, col, lab in DECOMP_COMPS:
        if nm == "const":
            continue
        spread = d[nm].std(1)
        b.plot(x, np.maximum(spread, 1e-4), "o-", color=col, lw=1.6, ms=4, label=lab)
    b.set_yscale("log")
    b.set_xlabel("tendency noise (% of error-tendency std)")
    b.set_ylabel("bootstrap spread of the recovered coefficient")
    b.set_title("(b)", fontsize=10, loc="left")
    b.legend(fontsize=7.5, frameon=False, loc="lower right")
    # the curves climb on a diagonal to the upper right, so the upper-left band is the
    # empty region; anchor the note there (was at 0.62, sitting on the rising red curve)
    b.text(0.04, 0.97, "circulation pinned near $\\varepsilon$;\nocean temperature the\nleast determined "
           "(its spread\ngrows fastest with noise)",
           transform=b.transAxes, fontsize=7.5, color="0.25", va="top", bbox=_LBL_BBOX)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_decomposition.png")
    if save:
        fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def roadmap_figure(save=True):
    """Schematic: the three ways the injected error is used, so the clean single-driver
    default reads as intentional, not under-ambitious. (1) canonical single-driver
    g=T_o1, c=(0,0,1,0); (2) mixed-driver stress test, all atmospheric+ocean drivers
    active; (3) the 36-mode identifiability sweep, each mode injected in turn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrow

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
    labels4 = [r"$1$", r"$\theta_{a,1}$", r"$T_{o,1}$", r"$\psi_{a,1}$"]
    TX = 5.3                                   # body-text left edge, clears the widest strip

    def cellrow(x0, y0, active, names=None, w=0.62, h=0.5, gap=0.1):
        for k in range(4):
            on = active[k]
            ax.add_patch(Rectangle((x0 + k * (w + gap), y0), w, h,
                                   facecolor=("C2" if on else "white"),
                                   edgecolor="0.4", lw=1.0, alpha=0.85 if on else 1))
            ax.text(x0 + k * (w + gap) + w / 2, y0 + h / 2, names[k],
                    ha="center", va="center", fontsize=8,
                    color="white" if on else "0.4")

    rows = [
        (4.3, "1.  Canonical single-driver  (the hardest case)",
         "$g=T_{o,1}$, $\\;c=(0,0,1,0)$.  Isolates the identifiability\n"
         "mechanism in the most tightly pinned driver.\nUsed in the mainline experiments and "
         "the appendix.", "single"),
        (2.3, "2.  Mixed-driver stress test",
         "$g=\\theta_{a,1}+T_{o,1}+\\psi_{a,1}$, $\\;c=(0,1,1,1)$.\nDoes the mechanism survive a "
         "less-sparse\nerror, not just one surgical driver?", "mixed"),
        (0.3, "3.  36-driver identifiability sweep",
         "$g=x_k,\\;k=1\\ldots36$, each mode injected\nsingly. Characterizes the boundary over\n"
         "all modes, not a particular error.", "sweep"),
    ]
    for y, title, body, kind in rows:
        ax.add_patch(Rectangle((0.1, y + 0.05), 9.8, 1.75, fill=False,
                               edgecolor="0.8", lw=0.8))
        ax.text(0.25, y + 1.55, title, fontsize=10.5, fontweight="bold", va="top")
        if kind == "single":
            cellrow(0.35, y + 0.45, [0, 0, 1, 0], names=labels4)
        elif kind == "mixed":
            cellrow(0.35, y + 0.45, [0, 1, 1, 1], names=labels4)
        else:
            w, gap = 0.105, 0.02
            for k in range(36):
                on = (k == 17)
                ax.add_patch(Rectangle((0.35 + k * (w + gap), y + 0.5), w, 0.5,
                                       facecolor=("C2" if on else "0.92"),
                                       edgecolor="0.6", lw=0.4))
            ax.annotate("each in turn", (0.35 + 17 * (w + gap) + w / 2, y + 0.5),
                        xytext=(0, -12), textcoords="offset points", fontsize=7.5,
                        ha="center", va="top", color="C2",
                        arrowprops=dict(arrowstyle="->", color="C2", lw=0.8))
        ax.text(TX, y + 1.5, body, fontsize=8.6, va="top")

    ax.text(5, 6.4, "How the injected error is used", fontsize=12.5,
            fontweight="bold", ha="center", va="top")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_roadmap.png")
    if save:
        fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# the near-constant multiplier and the driver it aliases, for the cartoon
_ALIAS_M = 20      # psi_o1, a near-constant ocean-streamfunction mode
_ALIAS_D = 0       # psi_a1, a driver that varies on the attractor


def alias_data(n=3000, seed=0):
    """The fields behind the alias cartoon: a candidate driver d and the cross-product
    m*d with a near-constant mode m, on the attractor and on broad excitation. On the
    attractor m is nearly constant so m*d is proportional to d (they alias); under broad
    excitation m varies so the proportionality breaks."""
    tw = TwinModels(eps=EPS, g_coeffs=(0.0, 0.0, 1.0, 0.0))
    Xa = attractor_traj(tw, n)
    rng = np.random.default_rng(seed)
    Xb = rng.standard_normal((n, tw.ndim))
    out = {}
    for tag, X in (("att", Xa), ("broad", Xb)):
        d = X[:, _ALIAS_D]
        m = X[:, _ALIAS_M]
        out[tag] = dict(d=d, m=m, md=m * d,
                        corr=float(np.corrcoef(d, m * d)[0, 1]),
                        m_meanstd=float(abs(Xa[:, _ALIAS_M].mean()) / Xa[:, _ALIAS_M].std()))
    return out


def alias_cartoon(save=True):
    """Why a near-constant mode times a candidate driver mimics the driver on attractor
    data. (a) time series of d and the rescaled cross-product m*d/mean(m): they overlap
    on the attractor. (b) scatter d vs m*d on the attractor: a tight line (alias). (c) the
    same scatter under broad excitation, where m varies and the alias breaks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dat = alias_data()
    a_att, a_br = dat["att"], dat["broad"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # (a) standardized time series overlap on the attractor (same shape = collinear)
    def z(v):
        return (v - v.mean()) / v.std()
    w = slice(0, 1200)
    t = np.arange(1200)
    ax[0].plot(t, z(a_att["d"])[w], color="C0", lw=1.1, label=r"driver $d=\psi_{a,1}$")
    ax[0].plot(t, z(a_att["md"])[w], color="C3", lw=1.0, ls="--",
               label=r"alias $m\,d$  ($m=\psi_{o,1}$, near-constant)")
    ax[0].set_xlabel("step (on the attractor)")
    ax[0].set_ylabel("standardized amplitude")
    ax[0].set_title("(a)", fontsize=10, loc="left")
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper right")

    # (b) attractor scatter: tight line
    ax[1].scatter(a_att["d"], a_att["md"], s=6, color="C0", alpha=0.4)
    ax[1].set_xlabel(r"driver $d$"); ax[1].set_ylabel(r"cross-product $m\,d$")
    ax[1].set_title("(b)", fontsize=10, loc="left")

    # (c) broad excitation scatter: cloud
    ax[2].scatter(a_br["d"], a_br["md"], s=6, color="C3", alpha=0.4)
    ax[2].set_xlabel(r"driver $d$"); ax[2].set_ylabel(r"cross-product $m\,d$")
    ax[2].set_title("(c)", fontsize=10, loc="left")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_alias.png")
    if save:
        fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def run():
    rows = driver_scan()
    figure(rows)
    record_length_scan()
    tandem_figure()
    full_error_decomposition()
    decomposition_figure()
    roadmap_figure()
    alias_cartoon()
    return headline_figure()
