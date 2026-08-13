"""
error_discovery.experiments.driver_mixed
========================================

Is the clean single-driver result an artifact of having only one driver active?

Inject all three mode-1 drivers at once (g = theta_a1 + T_o1 + psi_a1, each at
eps) and recover their coefficients *jointly* from the climatology. The form is
assumed here, exactly as in the canonical experiment; what is being asked is
whether several simultaneously-active coefficients are each recoverable.

The blind spot is not assumed but emerges as a conditioning effect: noiseless,
every coefficient comes back exactly; under noise the atmospheric-circulation
coefficient stays pinned near eps at every level while the ocean-temperature one
is the least determined, its spread growing fastest. The ordering tracks each
driver's own |mean|/std.
"""

import os
import numpy as np

from .. import figstyle
from ..models import TwinModels
from ..config import RESULTS_DIR, EPS
from ..figstyle import DPI, LBL_BBOX
from ..formfit import attractor_traj, err_tendency


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


def run(noises=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2), n=8000, reps=40,
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


def figure(save=True):
    """Two panels for the full mode-1 error recovered jointly from the climatology vs
    tendency noise. (a) recovered coefficient with bootstrap 5-95 band: all three centre
    near eps, but the bands differ. (b) the bootstrap spread itself, log scale: the clean
    conditioning ordering psi_a1 < theta_a1 < T_o1 at every noise, the ocean-temperature
    coefficient several times the least-determined, lost first as noise grows."""
    plt = figstyle.pyplot()

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
           transform=b.transAxes, fontsize=7.5, color="0.25", va="top", bbox=LBL_BBOX)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_decomposition.png")
    if save:
        fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return [out]
