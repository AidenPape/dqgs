"""Stage 0a, run 1 (plans/fitting-experiments.md): does the QR solver make the noiseless
fits reach c_true?

T_o1, noiseless, c0 = 0, 40 LM iterations, whitened by the high-frequency std, each case once
with the normal equations and once with QR. Pick indices refer to the 60-state lists of
``driver_uncertainty_*_n60``. Writes ``results/data/stage0a_run1.npz``.

    python scratch/stage0a_run1.py
"""

import os
import time

import numpy as np

from error_discovery.attractor import segments_from_states
from error_discovery.config import ExperimentConfig, DATA_DIR, steps_to_days
from error_discovery.discovery import recover
from error_discovery.experiments.driver_window import DRIVERS
from error_discovery.models import TwinModels, obs_mask
from error_discovery.sampling import starting_states, noise_scale

N_STATES = 60
LM_ITERS = 40
FAIL = 1e-3                         # a fit failed iff rel_err >= FAIL
SOLVERS = ("normal", "qr")
CASES = [                           # (mode, pick, windows in steps): rows 1-9 of the plan
    ("both", 1, (1280, 5120, 8000, 16000, 24000)),
    ("active", 0, (16000, 24000)),
    ("passive", 40, (320,)),
    ("passive", 20, (320,)),
    ("both", 35, (320,)),
    ("both", 16, (1280,)),
    ("both", 53, (8000,)),
    ("passive", 14, (1280,)),
    ("active", 33, (1280,)),
]


def main():
    t0 = time.time()
    cfg = ExperimentConfig()
    tw = TwinModels(cfg.eps, DRIVERS["T_o1"])
    mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
    scale = noise_scale()
    rows = []
    for row, (mode, pick, windows) in enumerate(CASES, start=1):
        picks = starting_states(mode, N_STATES, window_steps=max(cfg.window_steps))
        ics, trajs = segments_from_states(tw, picks["states"][pick:pick + 1], max(windows),
                                          cfg.dt)
        for w in windows:
            for solver in SOLVERS:
                ts = time.time()
                r = recover(tw, ics[0], trajs[0][:w], cfg.dt, mask, eta=0.0,
                            n_iters=LM_ITERS, whiten=True, scale=scale, solver=solver)
                r.update(row=row, mode=mode, pick=pick, window_steps=w, solver=solver,
                         regime=str(picks["regime"][pick]),
                         snapshot=int(picks["snapshot_index"][pick]), seconds=time.time() - ts)
                rows.append(r)
                print(f"row {row}  {mode:7s} pick {pick:2d}  {steps_to_days(w):6.1f} d  "
                      f"{solver:6s}  rel_err {r['rel_err']:.2e}  "
                      f"cost {r['cost0']:.2e} -> {r['cost_final']:.2e}  "
                      f"iters {len(r['cost_hist']) - 1:2d}  "
                      f"{'FAIL' if r['rel_err'] >= FAIL else 'ok'}  [{r['seconds']:.0f}s]",
                      flush=True)

    cost_hist = np.full((len(rows), LM_ITERS + 1), np.nan)
    for i, r in enumerate(rows):
        cost_hist[i, :len(r["cost_hist"])] = r["cost_hist"]
    col = lambda k: np.asarray([r[k] for r in rows])
    out = os.path.join(DATA_DIR, "stage0a_run1.npz")
    np.savez(out, row=col("row"), mode=col("mode"), pick=col("pick"), regime=col("regime"),
             snapshot=col("snapshot"), window_steps=col("window_steps"),
             window_days=np.asarray([steps_to_days(w) for w in col("window_steps")]),
             solver=col("solver"), c_hat=col("c_hat"), c_true=np.asarray(tw.c_true),
             rel_err=col("rel_err"), cond_J=col("cond_J"), cost0=col("cost0"),
             cost_final=col("cost_final"), cost_hist=cost_hist,
             n_iters_run=np.asarray([len(r["cost_hist"]) - 1 for r in rows]),
             seconds=col("seconds"), lm_iters=LM_ITERS, fail=FAIL, eps=cfg.eps)
    print(f"wrote {out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
