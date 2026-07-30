"""
test_ideas/single_param_jtj.py   (SCRATCH -- not part of the paper)

Question: for a SINGLE free parameter, J^T J is a scalar (no condition number, just a
value). Is that scalar U-shaped in the window, or monotone? And does the single-
parameter RECOVERY error ever U-turn at long windows if the observation noise is large?

Setup: canonical error g = T_{o,1}, eps=0.1, truth coefficient = 0.1. We free ONLY that
coefficient (others fixed at truth). As a function of the optimization window:

  (a) J^T J = sum_n (d r_n / d theta)^2, the scalar total sensitivity at the truth
      (noise-independent);
  (b) the single-parameter recovery error by 1-D Gauss-Newton through the differentiable
      integrator, at several observation-noise levels (median over seeds).

Note: d(resid)/d(theta) does not depend on the additive observation noise, so the
Jacobian is compiled once per window and reused across all noise levels and seeds.
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
from error_discovery.discovery import rollout, add_obs_noise
from error_discovery.experiment import get_attractor_ic

HERE = os.path.dirname(os.path.abspath(__file__))
THETA_TRUE = 0.1
NOISES = (1e-4, 1e-3, 1e-2, 1e-1)
N_SEEDS = 3


def run():
    cfg = ExperimentConfig()
    tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0))     # g = T_{o,1}
    dt = cfg.dt
    xa = get_attractor_ic(tw, cfg, verbose=False)
    oidx = np.arange(tw.ndim)

    windows = [40, 80, 160, 320, 480, 640, 960, 1280, 1920, 2560, 3840, 5120, 8000]
    jtj = []
    relerr = np.full((len(NOISES), len(windows)), np.nan)

    for iw, W in enumerate(windows):
        def model_traj(theta):                      # flattened model trajectory
            c = jnp.array([0.0, 0.0, theta, 0.0])
            return rollout(lambda x: tw.f_model(x, c), xa, dt, W).ravel()
        traj_j = jax.jit(model_traj)
        jac_j = jax.jit(jax.jacfwd(model_traj))     # d(traj)/d(theta), noise-independent

        Y = np.asarray(traj_j(THETA_TRUE)).reshape(W, tw.ndim)   # truth = model at 0.1
        J0 = np.asarray(jac_j(THETA_TRUE))
        jtj.append(float(J0 @ J0))

        for ie, eta in enumerate(NOISES):
            errs = []
            for s in range(N_SEEDS):
                rng = np.random.default_rng(1000 * ie + s)
                Yn = add_obs_noise(Y, oidx, eta, rng).ravel()
                th = 0.0
                for _ in range(10):                 # 1-D Gauss-Newton
                    r = np.asarray(traj_j(th)) - Yn
                    J = np.asarray(jac_j(th))
                    d = float(J @ J)
                    if d <= 0:
                        break
                    th = th - float(J @ r) / d
                errs.append(abs(th - THETA_TRUE) / THETA_TRUE)
            relerr[ie, iw] = np.median(errs)
        print(f"W={W:5d} ({steps_to_days(W):5.1f} d)  JtJ={jtj[-1]:.3e}  "
              f"rel_err@noise={np.array2string(relerr[:, iw], precision=2)}")

    days = np.array([steps_to_days(w) for w in windows])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax[0].loglog(days, jtj, "o-", color="C3")
    ax[0].set_xlabel("optimization window (days)")
    ax[0].set_ylabel(r"$J^\top J$  (scalar, single parameter)")
    ax[0].set_title("(a) single-parameter sensitivity (noise-independent)", loc="left", fontsize=9)
    ax[0].grid(True, which="both", alpha=0.2)

    colors = ["C0", "C2", "C1", "C3"]
    for ie, eta in enumerate(NOISES):
        ax[1].loglog(days, relerr[ie], "o-", color=colors[ie], ms=4,
                     label=rf"$\eta={eta:g}$")
    ax[1].axhline(0.1, ls=":", color="grey", lw=1)
    ax[1].set_xlabel("optimization window (days)")
    ax[1].set_ylabel("single-parameter recovery error")
    ax[1].set_title("(b) recovery error vs observation noise", loc="left", fontsize=9)
    ax[1].legend(frameon=False, fontsize=8, ncol=2)
    ax[1].grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    out = os.path.join(HERE, "single_param_jtj.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)
    return windows, jtj, relerr


if __name__ == "__main__":
    run()
