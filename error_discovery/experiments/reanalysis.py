"""
error_discovery.experiments.reanalysis
======================================

Is reanalysis far enough off the biased attractor to identify the error form?

The form is identifiable on a transient but not on the biased model's own
climatology (see :mod:`error_discovery.experiments.form`), which raises the
practical question: real analyses are not free transients, they are a biased
model pulled toward observations. Does that pull excite T_{o,1} enough?

Three distributions are scored with the same margin: the biased climatology, the
true model's own climatology ("perfect reanalysis"), and the biased model nudged
toward true observations at relaxation timescale tau. All three are swept over the
injected error strength eps, with a free transient as the positive control.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

from .. import figstyle
from ..models import TwinModels
from ..discovery import rollout
from ..config import DATA_DIR, FIGURES_DIR, DT
from ..figstyle import DPI
from ..formfit import twin, transient_traj, err_tendency, fit, resid, margin_named, true_term


def _margin_of(tw, X):
    """The margin of g=T_{o,1} on distribution X, and how much T_{o,1} varies there."""
    m = err_tendency(tw, X)
    c, nm = fit(tw, X, m)
    return margin_named(resid(tw, c, X, m), nm, true_term(tw))[0], float(X[:, tw.idx_To1].std())


def _nudge(tw, x0, Xref, tau):
    """Biased model relaxed toward a true reference trajectory: reanalysis."""
    from qgs_jax.model import rk4_step
    def step(x, xr):
        xn = rk4_step(lambda z: tw.f_base(z) + (xr - z) / tau, x, DT)
        return xn, xn
    _, traj = jax.lax.scan(step, x0, jnp.asarray(Xref))
    return np.asarray(traj)


def reanalysis(eps_list=(0.1, 0.3, 1.0), taus=(100.0, 20.0, 5.0), n=4000, n_spin=40000):
    from ..attractor import get_attractor_ic
    from ..config import ExperimentConfig
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


def run():
    r = reanalysis()
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(os.path.join(DATA_DIR, "reanalysis_results.npz"),
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


def figure():
    plt = figstyle.pyplot()
    d = np.load(os.path.join(DATA_DIR, "reanalysis_results.npz"))
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
    out = os.path.join(FIGURES_DIR, "fig_reanalysis.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return [out]
