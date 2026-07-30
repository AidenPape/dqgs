"""
test_ideas/chaotic_window.py   (SCRATCH -> destined for a paper appendix)

The chaotic-window test. At 2x2 (Lyapunov time ~ infinite) the noiseless
coefficient recovery is exact at every window and the loss surface is unimodal, so
the "window of discoverability" is a conditioning x noise effect (Sec. window). This
script runs the SAME recovery at a genuinely chaotic truncation, 3x3 atmosphere
(ndim 58, Lyapunov time ~ 7 days, shorter than the discovery windows), to show the
OTHER regime: once the window exceeds the predictability horizon, the loss surface
goes multimodal and the noiseless recovery fails for real.

Reports, for ndim 58:
  * the Lyapunov time (context);
  * noiseless recovery error and cond(J^T J) vs window (does it fail past ~horizon?);
  * a multistart LM at a short (< horizon) and long (>> horizon) window: 10/10
    reaching the truth means unimodal; scatter means multimodal.
A separate attractor cache is used; the committed 36-dim file is never touched.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

from error_discovery.models import TwinModels
from qgs.params.params import QgParams
from qgs_jax.spec import (DEFAULT_MAOOAM_PARAMS, DEFAULT_ATEMP_PARAMS, DEFAULT_GOTEMP_PARAMS,
                          DEFAULT_A_INSOLATION, DEFAULT_O_INSOLATION)
from qgs_jax import model as jm
from error_discovery.discovery import (rollout, make_residual, lm_fit, recover,
                                       obs_indices, lyapunov_time)
from error_discovery.config import DT, steps_to_days

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


def get_attractor(tw, tag, spinup, seed=1):
    cache = os.path.join(HERE, f"attractor_ic_{tag}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if int(d["ndim"]) == tw.ndim and int(d["spinup"]) == spinup:
            return jnp.asarray(d["x"])
    rng = np.random.default_rng(seed)
    x0 = jnp.asarray(rng.standard_normal(tw.ndim) * 0.1)
    print(f"[ic] spinning up {tw.ndim}-dim, {spinup} steps ...", flush=True)
    xa = np.asarray(jm.integrate_final(tw.T_base, x0, DT, spinup))
    np.savez(cache, x=xa, ndim=tw.ndim, spinup=spinup, seed=seed)
    return jnp.asarray(xa)


def run(ax=2, ay=4, spinup=800_000):
    tw = TwinModels(0.1, (0.0, 0.0, 1.0, 0.0), params=build_params(ax, ay))
    ctrue = np.asarray(tw.c_true)
    mask = jnp.ones(tw.ndim)
    oidx = obs_indices(mask)
    xa = get_attractor(tw, f"ndim{tw.ndim}_sp{spinup}", spinup)

    class _Shim:
        T_base = tw.T_base; ndim = tw.ndim
    lam, tau, tau_days = lyapunov_time(_Shim(), xa, DT, block=200, n_blocks=300)
    print(f"\n=== {tw.ndim}-dim (atmosphere {ax}x{ay}) ===")
    print(f"Lyapunov time ~ {tau_days:.1f} days (lambda={lam:.3e}/nondim)\n")

    windows = [40, 80, 160, 320, 480, 640, 960, 1280, 1920, 2560, 3840, 5120, 8000]
    print(f"{'window':>7s} {'days':>6s} {'noiseless rel_err':>18s} {'cond(JtJ)':>11s}")
    res = []
    for W in windows:
        Y = np.asarray(rollout(lambda x: tw.f_true(x), xa, DT, W))
        r = recover(tw, xa, Y, DT, mask, eta=0.0, n_iters=40)
        res.append((steps_to_days(W), r["rel_err"], r["cond"]))
        print(f"{W:7d} {steps_to_days(W):6.1f} {r['rel_err']:18.2e} {r['cond']:11.2e}", flush=True)

    # multistart: short (< horizon) vs long (>> horizon) window
    print("\nmultistart LM (10 random starts, does it reach the truth?):")
    for W in (320, 2560):                                   # 3.6 d vs 28.7 d
        Y = np.asarray(rollout(lambda x: tw.f_true(x), xa, DT, W))
        resid = make_residual(tw, xa, Y, DT, oidx)
        rng = np.random.default_rng(0)
        hits, errs = 0, []
        for s in range(10):
            c0 = rng.normal(0.0, 0.1, 4)
            chat, _, _ = lm_fit(resid, jnp.asarray(c0), n_iters=40)
            e = float(np.linalg.norm(np.asarray(chat) - ctrue) / np.linalg.norm(ctrue))
            errs.append(e); hits += (e < 0.05)
        print(f"  W={W:5d} ({steps_to_days(W):4.1f} d, horizon {tau_days:.0f} d): "
              f"reached truth {hits}/10  (median rel_err {np.median(errs):.2e})", flush=True)

    np.savez(os.path.join(HERE, "chaotic_window.npz"),
             ndim=tw.ndim, lam=lam, tau_days=tau_days,
             days=np.asarray([r[0] for r in res]),
             rel_err=np.asarray([r[1] for r in res]),
             cond=np.asarray([r[2] for r in res]))
    print("\nwrote chaotic_window.npz")


if __name__ == "__main__":
    run()
