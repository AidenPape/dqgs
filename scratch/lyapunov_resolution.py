"""
test_ideas/lyapunov_resolution.py   (SCRATCH -- not part of the paper)

Does a higher-resolution MAOOAM get more chaotic (shorter predictability horizon)?

Build the base MAOOAM tendency at several spectral truncations, spin each onto its
attractor, and estimate the leading Lyapunov exponent (Benettin, block
renormalization) -> Lyapunov time in days. Compare to the days-to-weeks discovery
windows: if the horizon shrinks toward the window as resolution grows, chaos starts
to bite inside the window (and the noiseless long-window recovery could become a
genuine limit rather than the optimizer artifact we found at 2x2).
"""

import os
import sys

_REPO = "/glade/u/home/wchapman/help_aiden"
for _p in (_REPO, os.path.join(_REPO, "qgs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import jax
import jax.numpy as jnp

from qgs.params.params import QgParams
from qgs_jax.spec import (extract_spec, DEFAULT_MAOOAM_PARAMS, DEFAULT_ATEMP_PARAMS,
                          DEFAULT_GOTEMP_PARAMS, DEFAULT_A_INSOLATION, DEFAULT_O_INSOLATION)
from qgs_jax import model as jm
from error_discovery.discovery import lyapunov_time

DT = 0.1


def build_params(ax, ay, ox, oy):
    p = QgParams()
    p.set_atmospheric_channel_fourier_modes(ax, ay)
    p.set_oceanic_basin_fourier_modes(ox, oy)
    p.set_params(DEFAULT_MAOOAM_PARAMS)
    p.atemperature_params.set_params(DEFAULT_ATEMP_PARAMS)
    p.gotemperature_params.set_params(DEFAULT_GOTEMP_PARAMS)
    p.atemperature_params.set_insolation(*DEFAULT_A_INSOLATION)
    p.gotemperature_params.set_insolation(*DEFAULT_O_INSOLATION)
    return p


class _Shim:
    """Minimal object lyapunov_time needs: a dense base tensor and ndim."""
    def __init__(self, spec):
        self.T_base = jnp.asarray(spec.dense_tensor)
        self.ndim = int(spec.ndim)


def run(spinup=150_000, n_blocks=300, block=200, seed=1):
    configs = [(2, 2, 2, 4), (2, 4, 2, 4), (3, 3, 2, 4), (3, 5, 2, 4), (4, 4, 2, 4)]
    print(f"{'atm':>6s} {'oc':>6s} {'ndim':>5s} {'lambda(nondim)':>15s} "
          f"{'Lyap time (d)':>14s}")
    rows = []
    for (ax, ay, ox, oy) in configs:
        p = build_params(ax, ay, ox, oy)
        spec = extract_spec(p)
        sh = _Shim(spec)
        rng = np.random.default_rng(seed)
        x0 = jnp.asarray(rng.standard_normal(sh.ndim) * 0.1)
        xa = jm.integrate_final(sh.T_base, x0, DT, spinup)          # spin onto attractor
        lam, tau, tau_days = lyapunov_time(sh, xa, DT, block=block, n_blocks=n_blocks, seed=seed)
        rows.append(((ax, ay), sh.ndim, lam, tau_days))
        print(f"{str((ax,ay)):>6s} {str((ox,oy)):>6s} {sh.ndim:5d} {lam:15.3e} {tau_days:14.1f}")
    return rows


if __name__ == "__main__":
    run()
