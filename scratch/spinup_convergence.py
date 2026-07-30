"""
test_ideas/spinup_convergence.py   (SCRATCH)

Is the 56-dim (2x4) trajectory actually ON its attractor, or still in a slow
transient? The (3,3) case looked chaotic at 150k steps and was non-chaotic at 400k,
so spin-up length is not a detail. The ocean modes are slow (800k steps ~ 24 yr),
which may be marginal. We advance ONE trajectory and, at increasing cumulative
spin-up lengths, measure the two quantities the results depend on:

  * the leading Lyapunov exponent (is the chaos stationary?);
  * max(|mean|/std) over modes (does the alias controller settle?),
  * plus the slow ocean mode means (have they stopped drifting?).

If lambda and max(|mean|/std) stop changing past some spin-up, that is the
attractor and that spin-up is enough. If they keep drifting, they are not spun up.
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


def run(ax=2, ay=4):
    tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0), params=build_params(ax, ay))
    vr = tw.spec.variables_range
    ocean = list(range(vr[1], vr[3]))                 # psi_o + T_o indices (slow modes)

    class _Shim:
        T_base = tw.T_base; ndim = tw.ndim

    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal(tw.ndim) * 0.1)
    checkpoints = [200_000, 400_000, 800_000, 1_200_000, 1_600_000, 2_400_000]
    prev = 0
    print(f"{tw.ndim}-dim (atmosphere {ax}x{ay}); ocean modes = {ocean[0]}..{ocean[-1]}")
    print(f"{'spinup':>9s} {'yrs':>5s} {'lambda':>11s} {'Lyap(d)':>8s} "
          f"{'max|mean|/std':>13s} {'mean|ocean mean|':>16s}", flush=True)
    for cp in checkpoints:
        x = jm.integrate_final(tw.T_base, x, DT, cp - prev)
        prev = cp
        Xs = np.asarray(rollout(lambda y: tw.f_base(y), x, DT, 3000))
        mm = max_mean_over_std(Xs)
        lam, _, dd = lyapunov_time(_Shim(), x, DT, block=200, n_blocks=300, seed=1)
        ocean_mean = float(np.mean(np.abs(Xs[:, ocean].mean(0))))
        yrs = steps_to_days(cp) / 365.0
        print(f"{cp:9d} {yrs:5.0f} {lam:11.3e} {dd:8.1f} {mm:13.1f} {ocean_mean:16.4f}",
              flush=True)


if __name__ == "__main__":
    run()
