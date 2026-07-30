"""
test_ideas/chaotic_window_converged.py   (SCRATCH -> test figure)

Does the noiseless window of discoverability fail GENUINELY (multimodal loss) once
the window exceeds the predictability horizon -- the chaos regime conjectured in
Sec. window and grounded in Pires et al. 1996? We now have a truly chaotic converged
attractor (58-dim, horizon ~11 d) and a weakly chaotic one (36-dim, horizon ~38 d).

For each: inject the canonical error (g=T_{o,1}, eps=0.1) from the CONVERGED attractor
IC, and measure noiseless 4-coefficient recovery vs window (well-converged optimizer),
plus a multistart LM (10 random starts) at each window -- 10/10 reaching the truth =
unimodal/recoverable; scatter = multimodal/genuinely lost.

Prediction: 36-dim recovers across all windows we use (<= its 38-d horizon-ish);
58-dim recovers at short windows but FAILS once the window exceeds ~11 d, and there
the multistart scatters -- a genuine loss-surface failure, not an optimizer artifact.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from error_discovery.models import TwinModels
from qgs.params.params import QgParams
from qgs_jax.spec import (DEFAULT_MAOOAM_PARAMS, DEFAULT_ATEMP_PARAMS, DEFAULT_GOTEMP_PARAMS,
                          DEFAULT_A_INSOLATION, DEFAULT_O_INSOLATION)
from error_discovery.discovery import rollout, make_residual, lm_fit, recover, obs_indices
from error_discovery.config import DT, steps_to_days

HERE = os.path.dirname(os.path.abspath(__file__))
# (config, horizon_days from lyap_converged)
CASES = [((2, 2), 38.0), ((3, 3), 11.0)]
WINDOWS = [80, 160, 320, 480, 640, 960, 1280, 1920, 2560, 3840, 5120, 8000]


def build_params(ax, ay, ox=2, oy=4):
    p = QgParams()
    p.set_atmospheric_channel_fourier_modes(ax, ay)
    p.set_oceanic_basin_fourier_modes(ox, oy)
    p.set_params(DEFAULT_MAOOAM_PARAMS)
    p.atemperature_params.set_params(DEFAULT_ATEMP_PARAMS)
    p.gotemperature_params.set_params(DEFAULT_GOTEMP_PARAMS)
    p.atemperature_params.set_insolation(*DEFAULT_A_INSOLATION)
    p.gotemperature_params.set_insolation(*DEFAULT_O_INSOLATION)
    return p


def run():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for (axm, aym), horizon in CASES:
        tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0), params=build_params(axm, aym))
        xa = jnp.asarray(np.load(os.path.join(HERE, f"converged_ndim{tw.ndim}.npz"))["x"])
        ctrue = np.asarray(tw.c_true)
        mask = jnp.ones(tw.ndim); oidx = obs_indices(mask)
        days = [steps_to_days(w) for w in WINDOWS]
        rel = []                      # noiseless recovery error, start from c=0, 60 iters
        ms = []                       # multistart success fraction (10 starts)
        for W in WINDOWS:
            Y = np.asarray(rollout(lambda x: tw.f_true(x), xa, DT, W))
            r = recover(tw, xa, Y, DT, mask, eta=0.0, n_iters=60)
            rel.append(r["rel_err"])
            resid = make_residual(tw, xa, Y, DT, oidx)
            rng = np.random.default_rng(0); hits = 0
            for s in range(10):
                chat, _, _ = lm_fit(resid, jnp.asarray(rng.normal(0, 0.05, 4)), n_iters=60)
                hits += (np.linalg.norm(np.asarray(chat) - ctrue) / np.linalg.norm(ctrue) < 0.05)
            ms.append(hits / 10)
            print(f"ndim {tw.ndim}  W={W:5d} ({steps_to_days(W):5.1f}d)  "
                  f"noiseless rel_err={r['rel_err']:.2e}  multistart {hits}/10", flush=True)
        c = "C0" if tw.ndim == 36 else "C3"
        lab = fr"ndim {tw.ndim} (atm {axm}$\times${aym}), horizon $\approx${horizon:.0f} d"
        ax[0].plot(days, np.maximum(rel, 1e-13), "o-", color=c, ms=4, label=lab)
        ax[1].plot(days, ms, "o-", color=c, ms=4, label=lab)
        ax[0].axvline(horizon, ls="--", color=c, lw=1, alpha=0.6)
        ax[1].axvline(horizon, ls="--", color=c, lw=1, alpha=0.6)

    ax[0].set_yscale("log"); ax[0].set_xscale("log")
    ax[0].axhline(0.1, ls=":", color="grey", lw=1)
    ax[0].set_xlabel("optimization window [days]")
    ax[0].set_ylabel(r"noiseless recovery error $\|\hat c-c_{\rm true}\|/\|c_{\rm true}\|$")
    ax[0].set_title("(a) noiseless recovery (converged optimizer)", loc="left", fontsize=9)
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("optimization window [days]")
    ax[1].set_ylabel("multistart success: P(reach truth)")
    ax[1].set_title("(b) multistart: unimodal vs multimodal", loc="left", fontsize=9)
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "chaotic_window_converged.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    run()
