"""
test_ideas/spinup_marathon.py <ax> <ay>   (SCRATCH -> saved attractors for reuse)

Long, resumable, checkpointed spin-up of a MAOOAM truncation, intended to be run as
a 12-hour PBS job (and re-run to extend). Brajard et al. 2021 spun the 56-dim config
up for 30,000 years because the ocean is that slow; this builds the same kind of
deeply-spun-up attractor state for any truncation and SAVES it for other studies.

Checkpoint / saved state: test_ideas/attractor_ndim<N>_spun.npz  (x, nsteps, ndim,
ax, ay). Re-running resumes from it. Every CHUNK steps it advances and saves; every
MEASURE_EVERY chunks it logs lambda and max(|mean|/std) so convergence is visible.
Runs until TARGET (effectively until PBS walltime kills it); the checkpoint is always
current, so nothing is lost.
"""

import os
import sys
import time
import numpy as np
import jax
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
CHUNK = 10_000_000
MEASURE_EVERY = 2
TARGET = 10_000_000_000          # ~300k yr; PBS walltime is the real cap


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
    print("JAX DEVICES:", jax.devices(), flush=True)
    tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0), params=build_params(ax, ay))
    ckpt = os.path.join(HERE, f"attractor_ndim{tw.ndim}_spun.npz")

    class _Shim:
        T_base = tw.T_base; ndim = tw.ndim

    if os.path.exists(ckpt):
        d = np.load(ckpt)
        x = jnp.asarray(d["x"]); nsteps = int(d["nsteps"])
        print(f"[resume] ndim={tw.ndim} (atm {ax}x{ay}) from {nsteps/1e6:.0f}M steps "
              f"({steps_to_days(nsteps)/365:.0f} yr)", flush=True)
    else:
        x = jnp.asarray(np.random.default_rng(1).standard_normal(tw.ndim) * 0.1)
        nsteps = 0
        print(f"[start] ndim={tw.ndim} (atm {ax}x{ay}) fresh from seed-1 IC", flush=True)
    print(f"{'Msteps':>8s} {'yrs':>7s} {'lambda':>11s} {'Lyap(d)':>9s} "
          f"{'max|mean|/std':>13s} {'chunk_s':>8s}", flush=True)

    chunk_idx = 0
    while nsteps < TARGET:
        t = time.time()
        x = jm.integrate_final(tw.T_base, x, DT, CHUNK)
        x.block_until_ready()
        nsteps += CHUNK
        chunk_idx += 1
        np.savez(ckpt, x=np.asarray(x), nsteps=nsteps, ndim=tw.ndim, ax=ax, ay=ay)
        cs = time.time() - t
        if chunk_idx % MEASURE_EVERY == 0:
            Xs = np.asarray(rollout(lambda y: tw.f_base(y), x, DT, 3000))
            mm = max_mean_over_std(Xs)
            lam, _, dd = lyapunov_time(_Shim(), x, DT, block=200, n_blocks=250, seed=1)
            print(f"{nsteps/1e6:8.0f} {steps_to_days(nsteps)/365:7.0f} {lam:11.3e} "
                  f"{dd:9.1f} {mm:13.1f} {cs:8.1f}", flush=True)
        else:
            print(f"{nsteps/1e6:8.0f} {steps_to_days(nsteps)/365:7.0f} {'':>11s} "
                  f"{'':>9s} {'':>13s} {cs:8.1f}", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]), int(sys.argv[2]))
