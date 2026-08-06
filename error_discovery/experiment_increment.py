"""
error_discovery.experiment_increment
====================================

Do **DA increments / initialized forecasts** unlock model-error *form* discovery
where free-running **climatology** cannot?

This re-uses the project's *validated* form-identifiability margin
(`experiment_form`), not a parametric noise-gain — an independent review showed the
gain metric credited zero-sensitivity parameters as "identifiable" and was
distribution-blind for an error that is linear in the parameters. Here the error
is the project's genuinely **state-dependent closure** `m(x) = eps·g(x)·(E x)`,
`g = T_{o,1}` — the case whose form discovery is excitation-limited.

The DA-increment hope, stated precisely: the states a DA cycle actually visits —
**biased short forecasts launched from true (analysis) states** — swing the
error's predictor `T_{o,1}` through its range, breaking the climatological
degeneracy that aliases `g=T_{o,1}` with a constant. We test it directly.

Distributions compared (form-margin of `g=T_{o,1}`; >0.1 = identifiable):
- biased climatology   : free run of the biased model
- true climatology     : free run of the true model ("reanalysis")
- DA forecast (short/long): biased forecasts launched from true states
- transient            : off-attractor excursion (the excitation that *does* work)

We also report the excitation (std) of the slow predictor `T_{o,1}` and a fast one
(`ψ_{a,1}`) on each distribution — the mechanism behind the margins.
"""

import os
import numpy as np
import jax.numpy as jnp

from .discovery import rollout
from .experiment_form import twin, err_tendency, _fit, _resid, _margin, DT, EPS
from .config import RESULTS_DIR
from .figstyle import DPI


def _attractor_ic(tw):
    from .experiment import get_attractor_ic
    from .config import ExperimentConfig
    return get_attractor_ic(tw, ExperimentConfig(), verbose=False)   # canonical, spun-up


def _margin_exc(tw, X):
    """Form-identifiability margin of g=T_o1 on X, plus slow/fast excitation std."""
    m = err_tendency(tw, X)
    c, nm = _fit(tw, X, m)
    margin = _margin(_resid(tw, c, X, m), nm, tw)[0]
    return margin, float(X[:, tw.idx_To1].std()), float(X[:, 0].std())


def true_climatology(tw, xa, n, spin):
    xt0 = jnp.asarray(np.asarray(rollout(lambda x: tw.f_true(x), xa, DT, spin))[-1])
    return np.asarray(rollout(lambda x: tw.f_true(x), xt0, DT, n)), xt0


def da_forecasts(tw, xa, n, window, spin):
    """States a DA cycle visits: biased forecasts launched from true (analysis)
    states. `window` steps per forecast; concatenate the segments."""
    Xtrue, _ = true_climatology(tw, xa, n, spin)
    starts = Xtrue[::window][: max(1, n // window)]            # analysis times
    segs = [np.asarray(rollout(lambda x: tw.f_base(x), jnp.asarray(s0), DT, window))
            for s0 in starts]
    return np.concatenate(segs, 0)


def run(n=4000, spin=40000):
    tw = twin()                                               # true = biased + eps·g, g=T_o1
    xa = _attractor_ic(tw)
    Xtrue, _ = true_climatology(tw, xa, n, spin)
    xb0 = jnp.asarray(np.asarray(rollout(lambda x: tw.f_base(x), xa, DT, spin))[-1])  # spin biased too
    dists = {
        "biased climatology": np.asarray(rollout(lambda x: tw.f_base(x), xb0, DT, n)),
        "true climatology (reanalysis)": Xtrue,
        "DA forecast, short (3 d)": da_forecasts(tw, xa, n, window=27, spin=spin),
        "DA forecast, long (22 d)": da_forecasts(tw, xa, n, window=200, spin=spin),
        "transient (off-attractor)": np.asarray(
            rollout(lambda x: tw.f_base(x),
                    jnp.asarray(np.random.default_rng(0).standard_normal(tw.ndim) * 0.5),
                    DT, n)),
    }
    names = list(dists)
    res = {nm: _margin_exc(tw, X) for nm, X in dists.items()}
    for nm in names:
        mg, slo, fas = res[nm]
        print(f"[{nm:32s}] margin={mg:+.3f}  std(T_o1)={slo:.4f}  std(psi_a1)={fas:.4f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, "increment_results.npz"),
             names=np.asarray(names), eps=EPS,
             margin=np.asarray([res[k][0] for k in names]),
             To1_std=np.asarray([res[k][1] for k in names]),
             psi_a1_std=np.asarray([res[k][2] for k in names]))
    return res


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(os.path.join(RESULTS_DIR, "increment_results.npz"), allow_pickle=True)
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in d["names"]]
    short = ["biased\nclimate", "true climate\n(reanalysis)", "DA fc\nshort (3 d)",
             "DA fc\nlong (22 d)", "transient\n(off-attractor)"]
    is_exc = np.array(["transient" in n for n in names])
    x = np.arange(len(names))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) form-identifiability margin: only the off-attractor transient clears 0.1
    ax[0].bar(x, d["margin"], color=np.where(is_exc, "C2", "C0"))
    ax[0].axhline(0.1, ls=":", color="grey", lw=1)
    ax[0].text(0, 0.12, "0.1 = identifiable", fontsize=8, color="grey")
    ax[0].set_xticks(x); ax[0].set_xticklabels(short, fontsize=8)
    ax[0].set_ylabel(r"form-identifiability margin of $g=T_{o,1}$")
    ax[0].set_title("(a)", fontsize=10, loc="left")

    # (b) why: the slow predictor stays quiescent on every forecast distribution
    ax[1].bar(x - 0.2, d["psi_a1_std"], 0.4, label=r"$\psi_{a,1}$ (fast, atmos.)")
    ax[1].bar(x + 0.2, d["To1_std"], 0.4, label=r"$T_{o,1}$ (slow, ocean)")
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(short, fontsize=8)
    ax[1].set_ylabel("predictor excitation  std")
    ax[1].set_title("(b)", fontsize=10, loc="left")
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_increment.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out
