"""
test_ideas/spinup56_brajard.py   (SCRATCH)

Spin up the 56-dim (atmosphere 2x4, na=20, no=8) MAOOAM the way Brajard et al. 2021
did for the SAME config: a ~30,000-year transient (~976M steps at dt=0.1), because
the ocean is that slow. Our earlier attempts (0.8-5M steps, 24-154 yr) were short by
~200-1000x, which is why nothing converged.

Resumable and checkpointed: every CHUNK steps it advances (block_until_ready, real
compute), SAVES state + cumulative steps to a checkpoint, and every MEASURE_EVERY
chunks it logs lambda and max(|mean|/std) so we can watch them finally settle. Stops
at TARGET steps or when killed; re-running resumes from the checkpoint.
"""

import os
import time
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
CKPT = os.path.join(HERE, "spinup56_brajard_ckpt.npz")
CHUNK = 10_000_000             # ~307 model-years per chunk
MEASURE_EVERY = 2             # measure lambda + max|mean|/std every 2 chunks (~614 yr)
TARGET = 1_000_000_000        # ~30,700 model-years (a bit past Brajard's 30k)


def build_params():
    p = QgParams()
    p.set_atmospheric_channel_fourier_modes(2, 4)
    p.set_oceanic_basin_fourier_modes(2, 4)
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
        print(f"[resume] {tw.ndim}-dim from {nsteps/1e6:.0f}M steps "
              f"({steps_to_days(nsteps)/365:.0f} yr)", flush=True)
    else:
        x = jnp.asarray(np.random.default_rng(1).standard_normal(tw.ndim) * 0.1)
        nsteps = 0
        print(f"[start] {tw.ndim}-dim (atm 2x4) Brajard 30000-yr spin-up; "
              f"target {TARGET/1e6:.0f}M steps", flush=True)
        print(f"{'Msteps':>8s} {'yrs':>6s} {'lambda':>11s} {'Lyap(d)':>9s} "
              f"{'max|mean|/std':>13s} {'chunk_s':>8s}", flush=True)

    chunk_idx = 0
    while nsteps < TARGET:
        t = time.time()
        x = jm.integrate_final(tw.T_base, x, DT, CHUNK)
        x.block_until_ready()
        nsteps += CHUNK
        chunk_idx += 1
        np.savez(CKPT, x=np.asarray(x), nsteps=nsteps, ndim=tw.ndim)
        cs = time.time() - t
        if chunk_idx % MEASURE_EVERY == 0:
            Xs = np.asarray(rollout(lambda y: tw.f_base(y), x, DT, 3000))
            mm = max_mean_over_std(Xs)
            lam, _, dd = lyapunov_time(_Shim(), x, DT, block=200, n_blocks=250, seed=1)
            print(f"{nsteps/1e6:8.0f} {steps_to_days(nsteps)/365:6.0f} {lam:11.3e} "
                  f"{dd:9.1f} {mm:13.1f} {cs:8.1f}", flush=True)
        else:
            print(f"{nsteps/1e6:8.0f} {steps_to_days(nsteps)/365:6.0f} "
                  f"{'':>11s} {'':>9s} {'':>13s} {cs:8.1f}", flush=True)


if __name__ == "__main__":
    run()
