"""Stage 0a, run 2 (notes/fitting_log.md): in the short-window fits that reached cost ~0 with
the wrong coefficients, does the error c_hat - c_true lie along J's weakest direction?

For each case, J is the residual Jacobian at c_true (same whitening as run 1), J = U S V^T. The
error is split over the right singular vectors: share_i = (e . v_i)^2 / |e|^2, so share of the
weakest (last) direction = cos^2 of the angle between e and v_min. Coefficient space is the raw
one (const, theta_a1, T_o1, psi_a1). c_hat is run 1's normal-equation fit.
Writes ``results/data/stage0a_run2.npz``.

    python scratch/stage0a_run2.py
"""

import os

import numpy as np

from error_discovery.attractor import segments_from_states
from error_discovery.config import ExperimentConfig, DATA_DIR
from error_discovery.discovery import residual_jacobian
from error_discovery.experiments.driver_window import DRIVERS
from error_discovery.models import TwinModels, obs_mask
from error_discovery.sampling import starting_states, noise_scale

N_STATES = 60
CASES = [("passive", 40, 320), ("both", 35, 320), ("both", 16, 1280)]   # (list, #, steps)


def main():
    cfg = ExperimentConfig()
    tw = TwinModels(cfg.eps, DRIVERS["T_o1"])
    mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
    scale = noise_scale()
    run1 = np.load(os.path.join(DATA_DIR, "stage0a_run1.npz"))
    c_true = np.asarray(tw.c_true)

    out = {k: [] for k in ("mode", "pick", "regime", "window_steps", "c_hat", "error",
                           "sigma", "V", "share", "angle_deg")}
    for mode, pick, w in CASES:
        i = np.flatnonzero((run1["mode"] == mode) & (run1["pick"] == pick)
                           & (run1["window_steps"] == w) & (run1["solver"] == "normal"))[0]
        picks = starting_states(mode, N_STATES, window_steps=max(cfg.window_steps))
        ics, trajs = segments_from_states(tw, picks["states"][pick:pick + 1], w, cfg.dt)
        J = residual_jacobian(tw, ics[0], trajs[0][:w], cfg.dt, mask, whiten=True, scale=scale)
        _, s, Vt = np.linalg.svd(J, full_matrices=False)
        e = run1["c_hat"][i] - c_true
        share = (Vt @ e) ** 2 / (e @ e)
        angle = np.degrees(np.arccos(min(1.0, np.sqrt(share[-1]))))
        for k, v in (("mode", mode), ("pick", pick), ("regime", str(picks["regime"][pick])),
                     ("window_steps", w), ("c_hat", run1["c_hat"][i]), ("error", e),
                     ("sigma", s), ("V", Vt.T), ("share", share), ("angle_deg", angle)):
            out[k].append(v)
        print(f"{str(picks['regime'][pick]):7s} `{mode}` #{pick:<2d} {w:5d} steps  "
              f"sigma {np.array2string(s, precision=2)}  "
              f"share of |e|^2 per direction (strong -> weak) "
              f"{np.array2string(share, precision=4)}  angle to v_min {angle:.2f} deg",
              flush=True)

    path = os.path.join(DATA_DIR, "stage0a_run2.npz")
    np.savez(path, c_true=c_true, **{k: np.asarray(v) for k, v in out.items()})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
