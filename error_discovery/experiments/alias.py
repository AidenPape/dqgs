"""
error_discovery.experiments.alias
=================================

The cartoon behind the obstruction: why a near-constant mode times a candidate
driver mimics the driver itself.

Take a driver d that varies on the attractor and a mode m that does not. On the
attractor m is nearly constant, so the cross-product m*d is very close to a
rescaled copy of d — the two are collinear and no fit can separate them. Excite
the system broadly enough that m varies, and the proportionality breaks. That is
the whole mechanism, in three panels, on two variables.
"""

import os
import numpy as np

from .. import figstyle
from ..config import DATA_DIR, FIGURES_DIR
from ..figstyle import DPI
from ..formfit import attractor_traj, twin


# the near-constant multiplier and the driver it aliases, for the cartoon
_ALIAS_M = 20      # psi_o1, a near-constant ocean-streamfunction mode
_ALIAS_D = 0       # psi_a1, a driver that varies on the attractor


def alias_data(n=3000, seed=0):
    """The fields behind the alias cartoon: a candidate driver d and the cross-product
    m*d with a near-constant mode m, on the attractor and on broad excitation. On the
    attractor m is nearly constant so m*d is proportional to d (they alias); under broad
    excitation m varies so the proportionality breaks."""
    tw = twin()
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


def run(n=3000, seed=0):
    """Cache the two distributions so ``figure`` need not re-integrate the attractor."""
    dat = alias_data(n=n, seed=seed)
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(os.path.join(DATA_DIR, "alias.npz"),
             **{f"{tag}_{k}": v for tag, sub in dat.items() for k, v in sub.items()})
    for tag, sub in dat.items():
        print(f"{tag:6s} corr(d, m*d) = {sub['corr']:+.3f}   "
              f"|mean|/std of m on the attractor = {sub['m_meanstd']:.1f}")
    return dat


def _load():
    """Regroup ``alias.npz`` back into the per-distribution dicts ``run`` saved."""
    z = np.load(os.path.join(DATA_DIR, "alias.npz"))
    out = {"att": {}, "broad": {}}
    for key in z.files:
        tag, _, name = key.partition("_")
        out[tag][name] = z[key]
    return out


def figure(save=True):
    """Why a near-constant mode times a candidate driver mimics the driver on attractor
    data. (a) time series of d and the rescaled cross-product m*d/mean(m): they overlap
    on the attractor. (b) scatter d vs m*d on the attractor: a tight line (alias). (c) the
    same scatter under broad excitation, where m varies and the alias breaks."""
    plt = figstyle.pyplot()
    dat = _load()
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
    out = os.path.join(FIGURES_DIR, "fig_alias.png")
    if save:
        fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return [out]
