"""
test_ideas/resumable_spinup.py   (SCRATCH)

Generic resumable, checkpointing spin-up + convergence measurement for any MAOOAM
truncation. Every CHUNK steps it advances the trajectory, SAVES the state and
cumulative step count to spinup_ckpt_<ndim>.npz (resumable, never loses progress),
and measures the leading Lyapunov exponent and max(|mean|/std). If those two stop
changing across chunks, that spin-up is enough; if they keep wandering, it is not.

Usage:  python resumable_spinup.py <ax> <ay>      e.g. 2 2  (36-dim) or 2 4 (56-dim)
"""

import os
import sys
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
CHUNK = 1_000_000
MAX_STEPS = 30_000_000          # ~900 model-years before giving up


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


def run(ax, ay):
    tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0), params=build_params(ax, ay))
    ckpt = os.path.join(HERE, f"spinup_ckpt_{tw.ndim}.npz")

    class _Shim:
        T_base = tw.T_base; ndim = tw.ndim

    if os.path.exists(ckpt):
        d = np.load(ckpt)
        x = jnp.asarray(d["x"]); nsteps = int(d["nsteps"])
        print(f"[resume] {tw.ndim}-dim from {nsteps} steps "
              f"({steps_to_days(nsteps)/365:.0f} yr)", flush=True)
    else:
        x = jnp.asarray(np.random.default_rng(1).standard_normal(tw.ndim) * 0.1)
        nsteps = 0
        print(f"[start] {tw.ndim}-dim (atmosphere {ax}x{ay}) fresh from seed-1 IC", flush=True)
        print(f"{'steps':>10s} {'yrs':>5s} {'lambda':>11s} {'Lyap(d)':>8s} "
              f"{'max|mean|/std':>13s}", flush=True)

    while nsteps < MAX_STEPS:
        x = jm.integrate_final(tw.T_base, x, DT, CHUNK)
        nsteps += CHUNK
        np.savez(ckpt, x=np.asarray(x), nsteps=nsteps, ndim=tw.ndim)
        Xs = np.asarray(rollout(lambda y: tw.f_base(y), x, DT, 3000))
        mm = max_mean_over_std(Xs)
        lam, _, dd = lyapunov_time(_Shim(), x, DT, block=200, n_blocks=250, seed=1)
        print(f"{nsteps:10d} {steps_to_days(nsteps)/365:5.0f} {lam:11.3e} {dd:8.1f} "
              f"{mm:13.1f}", flush=True)


if __name__ == "__main__":
    ax, ay = int(sys.argv[1]), int(sys.argv[2])
    run(ax, ay)
