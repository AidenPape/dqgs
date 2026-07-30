"""
test_ideas/spinup56_resumable.py   (SCRATCH)

Resumable, checkpointing long spin-up of the 56-dim (2x4) base model, to find out
whether it EVER reaches a stationary attractor (the convergence check showed lambda
and max(|mean|/std) still wandering at 74 model-years / 2.4M steps).

Every CHUNK steps it: advances the trajectory, SAVES the state + cumulative step
count to a checkpoint (so progress is never lost and the run is resumable), and
measures the leading Lyapunov exponent and max(|mean|/std). Re-running resumes from
the checkpoint; it stops when both quantities look stationary over the last few
chunks, or at MAX_STEPS.
"""

import os
import numpy as np
import jax.numpy as jnp

from error_discovery.models import TwinModels
from qgs.params.params import QgParams
from qgs_jax.spec import (DEFAULT_MAOOAM_PARAMS, DEFAULT_ATEMP_PARAMS, DEFAULT_GOTEMP_PARAMS,
                          DEFAULT_A_INSOLATION, DEFAULT_O_INSOLATION)
from qgs_jax import model as jm
from error_discovery.discovery import rollout, lyapunov_time
from error_discovery.experiment_meanstd import max_mean_over_std
from error_discovery.config import DT, steps_to_days

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "spinup56_ckpt.npz")
CHUNK = 1_000_000
MAX_STEPS = 30_000_000          # ~900 model-years before giving up


def build_params(ax=2, ay=4, ox=2, oy=4):
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
    tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0), params=build_params())

    class _Shim:
        T_base = tw.T_base; ndim = tw.ndim

    if os.path.exists(CKPT):
        d = np.load(CKPT)
        x = jnp.asarray(d["x"]); nsteps = int(d["nsteps"])
        print(f"[resume] from checkpoint at {nsteps} steps "
              f"({steps_to_days(nsteps)/365:.0f} yr)", flush=True)
    else:
        x = jnp.asarray(np.random.default_rng(1).standard_normal(tw.ndim) * 0.1)
        nsteps = 0
        print(f"[start] fresh from seed-1 IC", flush=True)
        print(f"{'steps':>10s} {'yrs':>5s} {'lambda':>11s} {'Lyap(d)':>8s} "
              f"{'max|mean|/std':>13s}", flush=True)

    while nsteps < MAX_STEPS:
        x = jm.integrate_final(tw.T_base, x, DT, CHUNK)
        nsteps += CHUNK
        np.savez(CKPT, x=np.asarray(x), nsteps=nsteps, ndim=tw.ndim)
        Xs = np.asarray(rollout(lambda y: tw.f_base(y), x, DT, 3000))
        mm = max_mean_over_std(Xs)
        lam, _, dd = lyapunov_time(_Shim(), x, DT, block=200, n_blocks=250, seed=1)
        print(f"{nsteps:10d} {steps_to_days(nsteps)/365:5.0f} {lam:11.3e} {dd:8.1f} "
              f"{mm:13.1f}", flush=True)


if __name__ == "__main__":
    run()
