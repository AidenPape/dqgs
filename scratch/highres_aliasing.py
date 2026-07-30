"""
test_ideas/highres_aliasing.py   (SCRATCH -- not part of the paper)

High-ROI generalization check: does the near-constant-mode aliasing obstruction
survive at higher spectral resolution? Build the 56-dim MAOOAM (atmosphere 2x4),
spin onto its attractor (separate cache -- the 36-dim provenance file is never
touched), and report the two numbers the whole identifiability thesis rides on:

  * max(|mean|/std) over modes  (the controller of the alias residual ~ 1/max);
  * how many of the 56 candidate error forms a restricted vs a full quadratic
    library can identify on the attractor.

At 36-dim these are about 517 and 20/36 vs 0/36. If 56-dim stays in the same place
(max(|mean|/std) in the hundreds, full library ~0), the generalization claim is
evidence, not argument.
"""

import os
import numpy as np
import jax.numpy as jnp

from error_discovery.models import TwinModels       # sets up the qgs path
from qgs.params.params import QgParams
from qgs_jax.spec import (DEFAULT_MAOOAM_PARAMS, DEFAULT_ATEMP_PARAMS, DEFAULT_GOTEMP_PARAMS,
                          DEFAULT_A_INSOLATION, DEFAULT_O_INSOLATION)
from qgs_jax import model as jm
from error_discovery.discovery import rollout
from error_discovery.experiment_meanstd import max_mean_over_std, identifiable_count
from error_discovery.config import DT

HERE = os.path.dirname(os.path.abspath(__file__))


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


def get_attractor(tw, tag, spinup=300_000, seed=1):
    cache = os.path.join(HERE, f"attractor_ic_{tag}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if int(d["ndim"]) == tw.ndim and int(d["spinup"]) == spinup:
            print(f"[ic] loaded cached {tag} attractor")
            return jnp.asarray(d["x"])
    rng = np.random.default_rng(seed)
    x0 = jnp.asarray(rng.standard_normal(tw.ndim) * 0.1)
    print(f"[ic] spinning up {tw.ndim}-dim, {spinup} steps ...", flush=True)
    xa = np.asarray(jm.integrate_final(tw.T_base, x0, DT, spinup))
    np.savez(cache, x=xa, ndim=tw.ndim, spinup=spinup, seed=seed)
    print(f"[ic] |x|={np.linalg.norm(xa):.3f}", flush=True)
    return jnp.asarray(xa)


def run(ax=2, ay=4, n=4000, spinup=800_000):
    tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0), params=build_params(ax, ay))
    tag = f"ndim{tw.ndim}_sp{spinup}"
    xa = get_attractor(tw, tag, spinup=spinup)
    X = np.asarray(rollout(lambda x: tw.f_base(x), xa, DT, n))
    mm = max_mean_over_std(X)
    nr, _ = identifiable_count(tw, X, "restricted")
    nf, margins = identifiable_count(tw, X, "full")
    print(f"\n=== {tw.ndim}-dim MAOOAM (atmosphere {ax}x{ay}) attractor ===")
    print(f"max(|mean|/std) over modes : {mm:.1f}")
    print(f"identifiable forms  restricted library : {nr}/{tw.ndim}")
    print(f"identifiable forms  full quadratic lib : {nf}/{tw.ndim}")
    print(f"full-library median margin : {np.median(margins):.4f}  "
          f"(~ 1/max(|mean|/std) = {1.0/mm:.4f})")
    print(f"[compare 36-dim: max~517, restricted 20/36, full 0/36]")
    return dict(ndim=tw.ndim, max_meanstd=mm, ident_restricted=nr, ident_full=nf)


if __name__ == "__main__":
    run()
