"""
test_ideas/lyap_converged.py   (SCRATCH -> test figure)

Rigorous leading-Lyapunov-exponent measurement on the CONVERGED (30k-yr spun-up)
attractors for ndim 36 / 56 / 58. Long Benettin with block renormalization, RUNNING
estimate (so we see whether it settles), two perturbation seeds for robustness.

Question: is any truncation TRULY chaotic on its converged attractor -- a clearly
positive lambda whose predictability horizon 1/lambda is shorter than the days-to-weeks
discovery window? If so, the chaos regime (Sec. window caveat) is accessible and adds
to the story; if all stay weakly chaotic (horizon >> window), the 36-dim picture holds
and chaos stays a literature-grounded conjecture.
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
from qgs_jax import model as jm
from error_discovery.config import DT, steps_to_days

HERE = os.path.dirname(os.path.abspath(__file__))
# The ~30k-yr states; were converged_ndim<N>.npz, renamed 2026-09-24 (notes/rename_map.md).
CONVERGED = {36: "attractor_ic_default_sp1030000000.npz",
             56: "attractor_ic_default-ndim56_sp980000000.npz",
             58: "attractor_ic_default-ndim58_sp910000000.npz"}
CONFIGS = [(2, 2), (2, 4), (3, 3)]
BLOCK = 200
N_BLOCKS = 3000          # 600k steps ~ 18 model-yr per run
D0 = 1e-8


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


def benettin(T, x0, ndim, seed):
    step = jax.jit(lambda y: jm.integrate_final(T, y, DT, BLOCK))
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(ndim); p = D0 * p / np.linalg.norm(p)
    x = jnp.asarray(np.asarray(x0)); xp = x + jnp.asarray(p)
    logs = np.empty(N_BLOCKS); run = np.empty(N_BLOCKS)
    for b in range(N_BLOCKS):
        x = step(x); xp = step(xp)
        d = float(jnp.linalg.norm(xp - x))
        logs[b] = np.log(d / D0)
        xp = x + (D0 / d) * (xp - x)
        run[b] = logs[:b+1].mean() / (BLOCK * DT)        # running lambda
    return run


def run():
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = {36: "C0", 56: "C2", 58: "C3"}
    summary = []
    yrs = np.arange(1, N_BLOCKS + 1) * BLOCK / 89.165 / 365
    for axm, aym in CONFIGS:
        tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0), params=build_params(axm, aym))
        x0 = jnp.asarray(np.load(os.path.join(HERE, CONVERGED[tw.ndim]))["x"])
        runs = np.array([benettin(tw.T_base, x0, tw.ndim, s) for s in (1, 7)])
        lam = float(runs[:, -1].mean())
        horizon_d = steps_to_days((1.0 / lam) / DT) if lam > 0 else np.inf
        summary.append((tw.ndim, axm, aym, lam, horizon_d))
        for r in runs:
            ax.plot(yrs, r, color=colors[tw.ndim], alpha=0.8, lw=1.3)
        ax.plot([], [], color=colors[tw.ndim], lw=1.6,
                label=fr"ndim {tw.ndim} (atm {axm}$\times${aym}): $\lambda$={lam:.2e}, "
                      + (f"$\\tau$={horizon_d:.0f} d" if lam > 0 else r"$\tau=\infty$"))
        print(f"ndim {tw.ndim} (atm {axm}x{aym}): lambda={lam:.3e}/nondim  "
              f"horizon={horizon_d:.1f} d  (final running est, 2 seeds)", flush=True)
    ax.axhline(0, ls=":", color="grey", lw=1)
    ax.set_xlabel("Benettin averaging length (model-years)")
    ax.set_ylabel(r"running leading Lyapunov exponent $\lambda$  [1/nondim]")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Lyapunov exponent on converged attractors", fontsize=10)
    fig.tight_layout()
    out = os.path.join(HERE, "lyap_converged.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)
    # discovery-window context
    print("\nDiscovery windows are days-to-weeks; truly chaotic for our purposes means "
          "horizon < ~2 weeks.")
    for ndim, axm, aym, lam, h in summary:
        verdict = "TRULY CHAOTIC (horizon < window)" if (lam > 0 and h < 21) else \
                  ("weakly chaotic (horizon >> window)" if lam > 0 else "non-chaotic (lambda<=0)")
        print(f"  ndim {ndim}: {verdict}")


if __name__ == "__main__":
    run()
