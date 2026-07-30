"""
test_ideas/converged_diagnostic.py   (SCRATCH)

The real higher-resolution generalization test, on PROPERLY spun-up attractors
(Brajard 30,000-year recipe), for ndim 36 / 56 / 58.

For each config: load the spun-up state, roll out a long base-model trajectory, and
report -- using the paper's 4000-step window protocol but over many windows to handle
the inherent noise in max(|mean|/std):
  * max(|mean|/std): median / IQR / max over 50 windows;
  * full-quadratic-library identifiable count (median over a few windows);
  * restricted-library identifiable count;
  * a stationarity check: per-mode mean/std over the first vs second half of the long
    rollout (if the attractor is spun up, these match).
"""

import os
import numpy as np
import jax.numpy as jnp

from error_discovery.models import TwinModels
from qgs.params.params import QgParams
from qgs_jax.spec import (DEFAULT_MAOOAM_PARAMS, DEFAULT_ATEMP_PARAMS, DEFAULT_GOTEMP_PARAMS,
                          DEFAULT_A_INSOLATION, DEFAULT_O_INSOLATION)
from error_discovery.discovery import rollout
from error_discovery.experiment_meanstd import max_mean_over_std, identifiable_count
from error_discovery.config import DT

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [(2, 2), (2, 4), (3, 3)]


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


def run(ax, ay, n_long=200_000, win=4000):
    tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0), params=build_params(ax, ay))
    d = np.load(os.path.join(HERE, f"converged_ndim{tw.ndim}.npz"))
    x0 = jnp.asarray(d["x"]); yr = float(d["nsteps"]) / 89.165 / 365
    X = np.asarray(rollout(lambda y: tw.f_base(y), x0, DT, n_long))      # long sample

    nwin = n_long // win
    mm = np.array([max_mean_over_std(X[i*win:(i+1)*win]) for i in range(nwin)])
    # identifiable counts on 3 representative windows (median-ish, low, high mm)
    order = np.argsort(mm)
    reps = [order[len(order)//2], order[len(order)//4], order[3*len(order)//4]]
    nf, nr = [], []
    for i in reps:
        W = X[i*win:(i+1)*win]
        nr.append(identifiable_count(tw, W, "restricted")[0])
        nf.append(identifiable_count(tw, W, "full")[0])
    # stationarity: per-mode mean/std, first half vs second half of the long rollout
    h = n_long // 2
    m1, s1 = X[:h].mean(0), X[:h].std(0)
    m2, s2 = X[h:].mean(0), X[h:].std(0)
    sd = X.std(0)
    mean_drift = float(np.median(np.abs(m1 - m2) / np.where(sd > 0, sd, 1)))
    std_ratio = float(np.median(np.maximum(s1, s2) / np.minimum(s1, s2)))

    print(f"\n=== ndim {tw.ndim} (atm {ax}x{ay}), spun up {yr:.0f} yr ===")
    print(f"max(|mean|/std) over {nwin} windows of {win} steps: "
          f"median={np.median(mm):.0f}  IQR=[{np.percentile(mm,25):.0f},{np.percentile(mm,75):.0f}]  max={mm.max():.0f}")
    print(f"identifiable forms (3 windows)  restricted: {nr}/{tw.ndim}   full quadratic: {nf}/{tw.ndim}")
    print(f"stationarity (1st vs 2nd half): median |mean shift|/std = {mean_drift:.3f}  "
          f"median std ratio = {std_ratio:.2f}")
    return dict(ndim=tw.ndim, yr=yr, mm_median=float(np.median(mm)),
                nf=nf, nr=nr, mean_drift=mean_drift, std_ratio=std_ratio)


if __name__ == "__main__":
    for ax, ay in CONFIGS:
        run(ax, ay)
