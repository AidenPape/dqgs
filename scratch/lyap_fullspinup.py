"""
test_ideas/lyap_fullspinup.py   (SCRATCH)

Re-measure the leading Lyapunov exponent ON THE ATTRACTOR (long spin-up), to tell
genuine attractor chaos from the transient chaos the earlier short-spin-up sweep
picked up. (3x3 at 400k spin-up relaxed to a non-chaotic attractor, lambda<0; the
150k estimates were transient.) For each truncation: spin up long, estimate lambda
with a long Benettin run, and check stability by estimating it again on a second
segment. Caches each attractor (separate path) so a window test can reuse it.

A config is a usable "chaotic attractor" if lambda is clearly positive and stable
across the two segments, with a Lyapunov time shorter than the discovery windows.
"""

import os
import numpy as np
import jax.numpy as jnp

from error_discovery.models import TwinModels
from qgs.params.params import QgParams
from qgs_jax.spec import (DEFAULT_MAOOAM_PARAMS, DEFAULT_ATEMP_PARAMS, DEFAULT_GOTEMP_PARAMS,
                          DEFAULT_A_INSOLATION, DEFAULT_O_INSOLATION)
from qgs_jax import model as jm
from error_discovery.discovery import lyapunov_time
from error_discovery.config import DT

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [(2, 4), (2, 6), (3, 3), (3, 5), (4, 4)]
SPINUP = 600_000


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


def get_attractor(T_base, ndim, tag, spinup, seed=1):
    cache = os.path.join(HERE, f"attractor_ic_{tag}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if int(d["ndim"]) == ndim and int(d["spinup"]) == spinup:
            return jnp.asarray(d["x"])
    rng = np.random.default_rng(seed)
    x0 = jnp.asarray(rng.standard_normal(ndim) * 0.1)
    print(f"[ic] spin up {ndim}-dim, {spinup} steps ...", flush=True)
    xa = np.asarray(jm.integrate_final(T_base, x0, DT, spinup))
    np.savez(cache, x=xa, ndim=ndim, spinup=spinup, seed=seed)
    return jnp.asarray(xa)


def run():
    print(f"{'atm':>6s} {'ndim':>5s} {'lambda_1':>11s} {'lambda_2':>11s} "
          f"{'Lyap(d)':>9s} {'chaotic?':>9s}", flush=True)
    for (ax, ay) in CONFIGS:
        tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0), params=build_params(ax, ay))

        class _Shim:
            T_base = tw.T_base; ndim = tw.ndim
        xa = get_attractor(tw.T_base, tw.ndim, f"ndim{tw.ndim}_sp{SPINUP}", SPINUP)
        # two independent Benettin segments (different perturbation seeds) on the attractor
        lam1, _, d1 = lyapunov_time(_Shim(), xa, DT, block=200, n_blocks=400, seed=1)
        lam2, _, d2 = lyapunov_time(_Shim(), xa, DT, block=200, n_blocks=400, seed=7)
        chaotic = (lam1 > 1e-3 and lam2 > 1e-3)
        dd = d1 if lam1 > 0 else float("inf")
        print(f"{str((ax,ay)):>6s} {tw.ndim:5d} {lam1:11.3e} {lam2:11.3e} "
              f"{dd:9.1f} {'YES' if chaotic else 'no':>9s}", flush=True)


if __name__ == "__main__":
    run()
