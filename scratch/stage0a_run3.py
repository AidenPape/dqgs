"""Stage 0a, run 3 (notes/fitting_log.md): in the long-window fits that failed with the cost
barely moved, does the cost fall all the way along the straight line from c = 0 (LM's start)
to c_true, or does it rise somewhere?

T_o1 driver, noiseless, whitened by the high-frequency std, run 1's cost (0.5 |r|^2) but
evaluated without jax.jit, so it is exactly 0 at c_true. Points c = a c_true for 101 evenly
spaced a in [0, 1]; with c_true = (0, 0, 0.1, 0) only the T_o1 coefficient moves, in steps of
0.001. Nothing is fitted. Writes ``results/stage0a_run3.npz`` and ``results/stage0a_run3.png``.
A first pass with the residual jitted, as ``lm_fit`` evaluates it, is kept as
``results/stage0a_run3_jit.*``; its cost at a = 0 matches run 1's starting cost.

    python scratch/stage0a_run3.py
"""

import os

import jax.numpy as jnp
import numpy as np

from error_discovery import figstyle
from error_discovery.attractor import segments_from_states
from error_discovery.config import ExperimentConfig, RESULTS_DIR, steps_to_days
from error_discovery.discovery import make_residual, obs_indices
from error_discovery.experiments.driver_window import DRIVERS
from error_discovery.models import TwinModels, obs_mask
from error_discovery.sampling import starting_states, noise_scale

N_STATES = 60
N_POINTS = 101
CASES = [                           # (list, #, windows in steps): run 1's long-window failures
    ("both", 1, (5120, 8000, 16000, 24000)),
    ("active", 0, (24000,)),
    ("both", 53, (8000,)),
]
OUT = os.path.join(RESULTS_DIR, "stage0a_run3")


def main():
    cfg = ExperimentConfig()
    tw = TwinModels(cfg.eps, DRIVERS["T_o1"])
    oidx = obs_indices(obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs))
    wchan = 1.0 / np.asarray(noise_scale())[oidx]
    run1 = np.load(OUT.replace("run3", "run1") + ".npz")
    c_true = np.asarray(tw.c_true)
    a = np.linspace(0.0, 1.0, N_POINTS)

    out = {k: [] for k in ("mode", "pick", "regime", "window_steps", "cost")}
    for mode, pick, windows in CASES:
        picks = starting_states(mode, N_STATES, window_steps=max(cfg.window_steps))
        ics, trajs = segments_from_states(tw, picks["states"][pick:pick + 1], max(windows),
                                          cfg.dt)
        regime = str(picks["regime"][pick])
        for w in windows:
            # not jax.jit, so the cost at c_true is exactly 0. lm_fit (run 1) jits it; compiled,
            # the c_true trajectory leaves the eager truth by rounding (results/*_jit.npz).
            resid = make_residual(tw, ics[0], trajs[0][:w], cfg.dt, oidx, wchan=wchan)
            cost = np.asarray([0.5 * float(r @ r) for r in
                               (np.asarray(resid(jnp.asarray(ai * c_true))) for ai in a)])
            i1 = np.flatnonzero((run1["mode"] == mode) & (run1["pick"] == pick)
                                & (run1["window_steps"] == w) & (run1["solver"] == "normal"))[0]
            d = np.diff(cost)
            rises = int(np.sum((d > 0) & np.concatenate([[True], d[:-1] <= 0])))
            print(f"{regime:7s} `{mode}` #{pick:<2d} {steps_to_days(w):6.1f} d  "
                  f"cost at 0 {cost[0]:.4e} (run 1 {run1['cost0'][i1]:.4e})  "
                  f"falls every step {bool(np.all(d < 0))}  stretches rising {rises}  "
                  f"max {cost.max():.3e} at T_o1 {a[cost.argmax()] * c_true[2]:.3f}  "
                  f"at truth {cost[-1]:.1e}", flush=True)
            for k, v in (("mode", mode), ("pick", pick), ("regime", regime),
                         ("window_steps", w), ("cost", cost)):
                out[k].append(v)

    np.savez(OUT + ".npz", a=a, c_true=c_true, eps=cfg.eps,
             window_days=np.asarray([steps_to_days(w) for w in out["window_steps"]]),
             **{k: np.asarray(v) for k, v in out.items()})
    print(f"wrote {OUT}.npz")


def figure():
    """One panel per case: cost against the T_o1 coefficient along the line, log y."""
    plt = figstyle.pyplot()
    d = np.load(OUT + ".npz")
    x = d["a"] * d["c_true"][2]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.semilogy(x, d["cost"][i], ".-", ms=3, lw=0.8)
        ax.set_title(f"{d['regime'][i]}, `{d['mode'][i]}` #{d['pick'][i]}, "
                     f"{d['window_days'][i]:.1f} d", fontsize=10)
        ax.axvline(d["c_true"][2], color="0.6", lw=0.8, ls="--")
    for ax in axes[-1]:
        ax.set_xlabel("T_o1 coefficient (others 0)")
    for ax in axes[:, 0]:
        ax.set_ylabel("cost, 0.5 |r|^2 (whitened)")
    figstyle.finish(fig, "Stage 0a run 3: cost along c = 0 -> c_true (dashed), "
                         "T_o1 driver, noiseless")
    fig.savefig(OUT + ".png", dpi=figstyle.DPI)
    return [OUT + ".png"]


if __name__ == "__main__":
    main()
    print(figure())
