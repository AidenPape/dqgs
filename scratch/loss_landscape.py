"""
test_ideas/loss_landscape.py   (SCRATCH -- not part of the paper)

Is the long-window failure of the 4-coefficient fit (a) genuine multimodality from
chaos, or (b) a single minimum in an ill-conditioned flat valley?

If MAOOAM here is too weakly chaotic to destroy predictability inside our windows
(Lyapunov time ~140 d >> 90 d), the loss surface should stay UNIMODAL but get very
flat along the worst-conditioned coefficient direction as the window grows -- not
rugged/multimodal. We test two ways at a short (11 d) and long (90 d) window:

  1. 1-D loss slices through the truth along the best- and worst-conditioned
     eigenvectors of J^T J (multiple dips = multimodal; one shallow dip = flat valley);
  2. a multistart LM (does the global minimum, cost 0 at c_true, remain findable?).
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from error_discovery.models import TwinModels
from error_discovery.config import ExperimentConfig, steps_to_days
from error_discovery.discovery import rollout, make_residual, lm_fit, obs_indices
from error_discovery.experiment import get_attractor_ic

HERE = os.path.dirname(os.path.abspath(__file__))


def run():
    cfg = ExperimentConfig()
    tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0))      # g = T_{o,1}, c_true=(0,0,0.1,0)
    dt = cfg.dt
    xa = get_attractor_ic(tw, cfg, verbose=False)
    ctrue = np.asarray(tw.c_true)
    scale = float(np.linalg.norm(ctrue))            # 0.1
    oidx = obs_indices(np.ones(tw.ndim))
    windows = [("11 d (sweet spot)", 960), ("90 d (long)", 8000)]
    svals = np.linspace(-2.0, 2.0, 41)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for name, W in windows:
        Y = np.asarray(rollout(lambda x: tw.f_true(x), xa, dt, W))
        resid = make_residual(tw, xa, Y, dt, oidx)
        rj = jax.jit(resid)

        def cost(c):
            r = np.asarray(rj(jnp.asarray(c)))
            return 0.5 * float(r @ r)

        # best/worst-conditioned coefficient directions at the truth
        J = np.asarray(jax.jacfwd(resid)(jnp.asarray(ctrue)))
        JtJ = J.T @ J
        evals, V = np.linalg.eigh(JtJ)              # ascending
        vmin, vmax = V[:, 0], V[:, -1]
        cond = evals[-1] / max(evals[0], 1e-300)

        Lmin = np.array([cost(ctrue + s * scale * vmin) for s in svals])
        Lmax = np.array([cost(ctrue + s * scale * vmax) for s in svals])
        n_local_min = int(((np.diff(np.sign(np.diff(Lmin)))) > 0).sum())

        # multistart LM: is the global minimum (cost ~0 at c_true) still findable?
        rng = np.random.default_rng(0)
        hits, relerrs, costs = 0, [], []
        for s in range(10):
            c0 = rng.normal(0.0, 0.1, 4)
            chat, _, hist = lm_fit(resid, jnp.asarray(c0), n_iters=40)
            re = float(np.linalg.norm(np.asarray(chat) - ctrue) / scale)
            relerrs.append(re); costs.append(float(hist[-1]))
            hits += (re < 0.05)

        print(f"[{name:18s}] cond(JtJ)={cond:.2e}  loss slice (worst dir): "
              f"{n_local_min} interior local min  |  multistart reached truth "
              f"{hits}/10 (median rel_err {np.median(relerrs):.2e})")

        c = "C0" if W == 960 else "C3"
        ax[0].plot(svals, np.maximum(Lmin, 1e-30), "-", color=c, label=f"{name}")
        ax[1].plot(svals, np.maximum(Lmax, 1e-30), "-", color=c, label=f"{name}")

    for a, ttl in zip(ax, ["(a) worst-conditioned direction", "(b) best-conditioned direction"]):
        a.set_yscale("log"); a.axvline(0, ls=":", color="grey", lw=1)
        a.set_xlabel(r"step from truth along eigenvector  ($\times\,||c_{true}||$)")
        a.set_ylabel(r"loss  $0.5\,||r||^2$")
        a.set_title(ttl, loc="left", fontsize=10)
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "loss_landscape.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    run()
