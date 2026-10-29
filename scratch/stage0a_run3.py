"""Stage 0a, run 3 (notes/fitting_log.md; plans/stage0a-cost-along-a-line.md): the cost along
straight lines in coefficient space, in run 1's long- and short-window fits, failed and reached.

T_o1 driver, noiseless, whitened by the high-frequency std, run 1's cost (0.5 |r|^2) but
evaluated without jax.jit, so it is exactly 0 at c_true. Nothing is fitted; c_hat is run 1's
normal-equation fit, "reached" is run 1's rel_err < 1e-3.

- Every case: the line c = a c_true, a from 0 (LM's start) to 1.5 in steps of 0.01. With
  c_true = (0, 0, 0.1, 0) only the T_o1 coefficient moves, in steps of 0.001.
- Short-window cases: the lines c = c_true + t v_min, t from -0.1 to 0.1 in steps of 0.001.
  v_min is the weakest right singular vector of the whitened residual Jacobian in raw
  coefficients (as run 2), taken at c_true and, for the failed fits, at c_hat; unit, signed so
  its T_o1 component is positive. c_hat's position on each is t = (c_hat - c_true) . v_min.

Fig A1 / A2: long windows, A1 `both` #1 at its 5 windows, A2 the other two states. Fig B1 / B2:
short windows, one row per window; B1 the line from c = 0, B2 the weakest-direction lines.
Writes ``results/stage0a_run3.npz`` and ``results/stage0a_run3_{A1,A2,B1,B2}.png``. A first pass with the residual jitted, as ``lm_fit``
evaluates it, is recorded in the fitting log (step 9).

    python scratch/stage0a_run3.py          # ~1.1 h (est.)
    python scratch/stage0a_run3.py smoke    # 4 / 3 points per line, writes stage0a_run3_smoke.*
"""

import os
import sys

import jax.numpy as jnp
import numpy as np

from error_discovery import figstyle
from error_discovery.attractor import segments_from_states
from error_discovery.config import ExperimentConfig, RESULTS_DIR, steps_to_days
from error_discovery.discovery import make_residual, obs_indices, residual_jacobian
from error_discovery.experiments.driver_window import DRIVERS
from error_discovery.models import TwinModels, obs_mask
from error_discovery.sampling import starting_states, noise_scale

N_STATES = 60
A_MAX, A_STEP = 1.5, 0.01
T_MAX, T_STEP = 0.1, 0.001
LONG = [(0, "both", 1, (1280, 5120, 8000, 16000, 24000)),     # Fig A: (row, list, #, windows
        (1, "active", 0, (16000, 24000)),                      # in steps)
        (1, "both", 53, (8000,))]
SHORT = [(0, 320, (("passive", 40), ("both", 35), ("passive", 20))),   # Fig B: (row, window,
         (1, 1280, (("both", 16), ("passive", 14), ("active", 33)))]   # cases)
OUT = os.path.join(RESULTS_DIR, "stage0a_run3")


def _cases():
    """(figure, row, col, list, #, window) for every case, in figure order."""
    cases, cols = [], {}
    for row, mode, pick, windows in LONG:
        for w in windows:
            cols[row] = cols.get(row, -1) + 1
            cases.append(("A", row, cols[row], mode, pick, w))
    for row, w, picks in SHORT:
        for col, (mode, pick) in enumerate(picks):
            cases.append(("B", row, col, mode, pick, w))
    return cases


def _costs(resid, points):
    """0.5 |r|^2 at each point, uncompiled."""
    return np.asarray([0.5 * float(r @ r) for r in
                       (np.asarray(resid(jnp.asarray(c))) for c in points)])


def _weakest(J):
    """Singular values and the weakest right singular vector, signed so T_o1's part is > 0."""
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    return s, Vt[-1] * np.sign(Vt[-1, 2])


def main(a_step=A_STEP, t_step=T_STEP, out=OUT):
    cfg = ExperimentConfig()
    tw = TwinModels(cfg.eps, DRIVERS["T_o1"])
    mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
    oidx = obs_indices(mask)
    scale = noise_scale()
    wchan = 1.0 / np.asarray(scale)[oidx]
    run1 = np.load(os.path.join(RESULTS_DIR, "stage0a_run1.npz"))
    c_true = np.asarray(tw.c_true)
    a = np.linspace(0.0, A_MAX, int(round(A_MAX / a_step)) + 1)
    t = np.linspace(-T_MAX, T_MAX, int(round(2 * T_MAX / t_step)) + 1)
    cases = _cases()

    # one truth trajectory per state, as long as its longest window
    picks, segs = {}, {}
    for _, _, _, mode, pick, _ in cases:
        if mode not in picks:
            picks[mode] = starting_states(mode, N_STATES, window_steps=max(cfg.window_steps))
        if (mode, pick) not in segs:
            w = max(c[5] for c in cases if c[3:5] == (mode, pick))
            segs[mode, pick] = segments_from_states(tw, picks[mode]["states"][pick:pick + 1], w,
                                                    cfg.dt)

    out_case = {k: [] for k in ("figure", "row", "col", "mode", "pick", "regime", "window_steps",
                                "reached", "c_hat", "cost0_run1", "cost_final_run1", "cost")}
    out_b = {k: [] for k in ("case", "sigma_true", "v_true", "sigma_hat", "v_hat", "t_hat_true",
                             "t_hat_hat", "cost_t_true", "cost_t_hat")}
    for n, (fig, row, col, mode, pick, w) in enumerate(cases):
        i1 = np.flatnonzero((run1["mode"] == mode) & (run1["pick"] == pick)
                            & (run1["window_steps"] == w) & (run1["solver"] == "normal"))[0]
        reached = bool(run1["rel_err"][i1] < run1["fail"])
        c_hat = run1["c_hat"][i1]
        ics, trajs = segs[mode, pick]
        resid = make_residual(tw, ics[0], trajs[0][:w], cfg.dt, oidx, wchan=wchan)
        cost = _costs(resid, a[:, None] * c_true)
        regime = str(picks[mode]["regime"][pick])
        for k, v in (("figure", fig), ("row", row), ("col", col), ("mode", mode), ("pick", pick),
                     ("regime", regime), ("window_steps", w), ("reached", reached),
                     ("c_hat", c_hat), ("cost0_run1", run1["cost0"][i1]),
                     ("cost_final_run1", run1["cost_final"][i1]), ("cost", cost)):
            out_case[k].append(v)
        line = (f"{regime:7s} `{mode}` #{pick:<2d} {steps_to_days(w):6.1f} d  "
                f"{'reached' if reached else 'failed '}  line: cost at 0 {cost[0]:.4e} "
                f"(run 1 {run1['cost0'][i1]:.4e}), at c_true {cost[np.argmin(abs(a - 1))]:.1e}, "
                f"max {cost.max():.3e} at T_o1 {a[cost.argmax()] * c_true[2]:.3f}")
        if fig == "B":
            def J(c):
                return residual_jacobian(tw, ics[0], trajs[0][:w], cfg.dt, mask, c=c,
                                         whiten=True, scale=scale)
            s_true, v_true = _weakest(J(c_true))
            s_hat, v_hat = (np.full(4, np.nan),) * 2 if reached else _weakest(J(c_hat))
            costs = [np.full(len(t), np.nan) if reached and v is v_hat
                     else _costs(resid, c_true + t[:, None] * v) for v in (v_true, v_hat)]
            t_hat = [(c_hat - c_true) @ v for v in (v_true, v_hat)]
            for k, v in (("case", n), ("sigma_true", s_true), ("v_true", v_true),
                         ("sigma_hat", s_hat), ("v_hat", v_hat), ("t_hat_true", t_hat[0]),
                         ("t_hat_hat", t_hat[1]), ("cost_t_true", costs[0]),
                         ("cost_t_hat", costs[1])):
                out_b[k].append(v)
            line += (f"\n{'':40s}weakest: c_hat at t {t_hat[0]:+.4f} (v_min at c_true), "
                     f"{t_hat[1]:+.4f} (at c_hat); sigma_min {s_true[-1]:.2e} / {s_hat[-1]:.2e}")
        print(line, flush=True)

    np.savez(out + ".npz", a=a, t=t, c_true=c_true, eps=cfg.eps,
             window_days=np.asarray([steps_to_days(c[5]) for c in cases]),
             **{k: np.asarray(v) for k, v in out_case.items()},
             **{f"b_{k}": np.asarray(v) for k, v in out_b.items()})
    print(f"wrote {out}.npz")


def _title(d, n):
    return (f"{d['regime'][n]}, `{d['mode'][n]}` #{d['pick'][n]}, {d['window_days'][n]:.1f} d, "
            f"{'reached' if d['reached'][n] else 'failed'}")


def _line_figure(d, cases, shape, path):
    """Fig A1 / A2 / B1: cost along c = a c_true against the T_o1 coefficient, one panel per
    case, filled in case order."""
    plt = figstyle.pyplot()
    x = d["a"] * d["c_true"][2]
    fig, axes = plt.subplots(*shape, figsize=(3.3 * shape[1], 3.0 * shape[0]), squeeze=False)
    for ax, n in zip(axes.flat, cases):
        ax.semilogy(x, d["cost"][n], ".-", ms=2, lw=0.8)
        ax.axvline(d["c_true"][2], color="0.6", lw=0.8, ls="--")
        ax.axhline(d["cost_final_run1"][n], color="k", lw=0.8, ls=":")
        ax.set_title(_title(d, n), fontsize=8.5)
        ax.set_xlabel("T_o1 coefficient (others 0)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("cost, 0.5 |r|^2 (whitened)")
    figstyle.finish(fig, "cost along c = a c_true, a from 0 (LM's start) to 1.5; dashed = c_true "
                         "(cost 0), dotted = run 1's final cost. T_o1 driver, noiseless")
    fig.savefig(path, dpi=figstyle.DPI); plt.close(fig)
    return path


def _weakest_figure(d, path):
    """Fig B2: cost along c = c_true + t v_min, v_min at c_true and at run 1's c_hat."""
    plt = figstyle.pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), squeeze=False)
    for j, n in enumerate(d["b_case"]):
        ax = axes[d["row"][n], d["col"][n]]
        for key, label, color in (("true", "v_min at c_true", "C0"),
                                  ("hat", "v_min at c_hat", "C1")):
            if d["reached"][n] and key == "hat":
                continue
            ax.semilogy(d["t"], d[f"b_cost_t_{key}"][j], ".-", ms=2, lw=0.8, color=color,
                        label=label)
            if not d["reached"][n]:
                ax.axvline(d[f"b_t_hat_{key}"][j], color=color, lw=0.8, ls="-.")
        ax.axvline(0.0, color="0.6", lw=0.8, ls="--")
        ax.axhline(d["cost_final_run1"][n], color="k", lw=0.8, ls=":")
        ax.set_title(_title(d, n), fontsize=8.5)
        ax.set_xlabel("t along v_min (unit, T_o1 part > 0)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("cost, 0.5 |r|^2 (whitened)")
    axes[0, 0].legend(fontsize=7.5)
    figstyle.finish(fig, "cost along c = c_true + t v_min (weakest right singular vector of J); "
                         "dashed = c_true (cost 0), dash-dot = run 1's c_hat,\ndotted = run 1's "
                         "final cost. T_o1 driver, noiseless")
    fig.savefig(path, dpi=figstyle.DPI); plt.close(fig)
    return path


def figure(out=OUT):
    d = np.load(out + ".npz")
    a, b = (np.flatnonzero(d["figure"] == f) for f in "AB")
    a1, a2 = a[d["row"][a] == 0], a[d["row"][a] == 1]
    return [_line_figure(d, a1, (1, len(a1)), out + "_A1.png"),
            _line_figure(d, a2, (1, len(a2)), out + "_A2.png"),
            _line_figure(d, b, (2, 3), out + "_B1.png"),
            _weakest_figure(d, out + "_B2.png")]


if __name__ == "__main__":
    if sys.argv[1:] == ["smoke"]:
        main(a_step=0.5, t_step=0.1, out=OUT + "_smoke")
        print(figure(OUT + "_smoke"))
    else:
        main()
        print(figure())
