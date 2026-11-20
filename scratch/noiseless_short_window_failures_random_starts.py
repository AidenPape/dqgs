"""Short-window failures from random starts (``plans/short-window-failures.md``): does a
short-window fit that failed from c = 0 reach c_true (rel_err < 1e-3) when LM starts from random
values instead?

Cases: Stage 0a run 1's three converged-but-wrong fits, T_o1 driver (``CASES``): ``passive`` #40
at 3.6 d, ``both`` #35 at 3.6 d, ``both`` #16 at 14.4 d; and one active-regime state, Stage 0b's
``active`` #1 at 1.8 d (no active state fails from c = 0 at 3.6 d or longer in 0b). Their c = 0
fits are the baseline, read from ``stage0a_run1.npz`` (normal equations) or
``driver_window_stage0b_active.npz``, not re-run. Each case is refit from the same 50
starts, drawn uniform in [-0.2, 0.2] on each of the four coefficients (one fixed seed); no
bounds. Otherwise as run 1: noiseless, x0 exact, all 36 observed, whitened by the high-frequency
std, 40 LM iterations, normal equations.

Per fit it saves 0b's fields under 0b's names, the start, and c and cost after every LM
iteration (NaN-padded to 41 rows). ``table`` prints and logs, per case: the c = 0 fit's rel_err,
starts reached of 50, and starts whose c left the start box on any coefficient at any iteration.

    python scratch/noiseless_short_window_failures_random_starts.py smoke   # c = 0 vs run 1; 2 starts
    python scratch/noiseless_short_window_failures_random_starts.py full    # 3 cases x 50 starts
    python scratch/noiseless_short_window_failures_random_starts.py table   # table from the saved run

Part 2 (a) of the plan: the 4 cases refit from c = 0 with c and cost recorded after every LM
iteration (each refit must equal its baseline), and a 2 x 4 figure, cost (top) and rel_err
(bottom) against LM iteration, one column per case.

    python scratch/noiseless_short_window_failures_random_starts.py c0_loss_path
    python scratch/noiseless_short_window_failures_random_starts.py c0_loss_path_figure

Part 2 (b): the compiled cost (as LM evaluates it) along each coefficient axis through c_true,
one coefficient varied, the others at their true values, and along v_min, the weakest right
singular vector of the residual Jacobian at c_true (oriented with a positive T_o1 component);
offsets -0.2 to 0.2, 201 points. Nothing is fitted. A 4 x 4 figure, one column per case, one row
per coefficient, the v_min slice dashed on the T_o1 row.

    python scratch/noiseless_short_window_failures_random_starts.py cost_along_coefficient_axes
    python scratch/noiseless_short_window_failures_random_starts.py cost_along_coefficient_axes_figure
    python scratch/noiseless_short_window_failures_random_starts.py cost_along_T_o1_axis_and_v_min_log_figure

The valley: the compiled cost on a 101 x 101 grid in the plane through c_true spanned by v_min
(x, along the valley floor) and the part of the T_o1 axis orthogonal to it (y, across). The T_o1
axis, c = 0 and every wrong answer c_true + t v_min lie in this plane; (a)'s LM path is projected
onto it. Figure: log10 cost, one panel per case.

    python scratch/noiseless_short_window_failures_random_starts.py cost_in_plane_of_T_o1_axis_and_v_min
    python scratch/noiseless_short_window_failures_random_starts.py cost_in_plane_of_T_o1_axis_and_v_min_figure
"""

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from error_discovery import figstyle
from error_discovery.attractor import segments_from_states
from error_discovery.config import (ExperimentConfig, DATA_DIR, FIGURES_DIR, RESULTS_DIR,
                                    steps_to_days)
from error_discovery.discovery import recover, make_residual, obs_indices, residual_jacobian
from error_discovery.experiments.driver_window import DRIVERS
from error_discovery.models import TwinModels, obs_mask
from error_discovery.sampling import starting_states, noise_scale

STEM = "noiseless_short_window_failures_random_starts"
C0_PATH_STEM = "noiseless_short_window_failures_c0_loss_path"
SLICE_STEM = "noiseless_short_window_failures_cost_along_coefficient_axes"
SLICE_HALF_WIDTH = 0.2            # offsets from c_true, linear (plan P2-3)
SLICE_POINTS = 201
COEFS = ("const", "theta_a1", "T_o1", "psi_a1")
PLANE_STEM = "noiseless_short_window_failures_cost_in_plane_of_T_o1_axis_and_v_min"
PLANE_POINTS = 101                # per side of the grid
PLANE_ACROSS = 0.25               # y half-range = PLANE_ACROSS * sin(angle T_o1 axis to v_min)
LOG_COST_FLOOR = -20.0            # colour scale clipped below 1e-20
# (list, pick, steps, baseline): run 1's three, then one active state from Stage 0b
CASES = (("passive", 40, 320, "run1"), ("both", 35, 320, "run1"), ("both", 16, 1280, "run1"),
         ("active", 1, 160, "0b"))
N_STATES = 60                     # the lists run 1 drew its picks from
N_STARTS = 50
BOX = 0.2                         # starts uniform in [-BOX, BOX] per coefficient (2 x eps)
START_SEED = 0
LM_ITERS = 40
REACHED = 1e-3                    # rel_err below this = reached c_true (0a / 0b threshold)
FIELDS = ("rel_err", "cond_J", "smin", "smax", "rank", "cost0", "cost_final")
PER_COEF = ("c_hat", "col_norm", "se_unit", "vif", "align")
SAVED_NAME = {"c_hat": "c_hats", "rank": "jrank"}             # as 0b names them
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")


def draw_starts():
    """The 50 shared starts, (N_STARTS, 4): (const, theta_a1, T_o1, psi_a1)."""
    return np.random.default_rng(START_SEED).uniform(-BOX, BOX, size=(N_STARTS, 4))


def _baseline(mode, pick, steps, source):
    """The c = 0 fit of this case: run 1's (normal equations) or Stage 0b's T_o1 fit."""
    if source == "run1":
        b = np.load(os.path.join(DATA_DIR, "stage0a_run1.npz"))
        i = np.flatnonzero((b["mode"] == mode) & (b["pick"] == pick)
                           & (b["window_steps"] == steps) & (b["solver"] == "normal"))
        assert len(i) == 1, (mode, pick, steps)
        return {k: b[k][i[0]] for k in ("snapshot", "c_hat", "rel_err", "cost0", "cost_final",
                                        "cost_hist", "n_iters_run")}
    b = np.load(os.path.join(DATA_DIR, f"driver_window_stage0b_{mode}.npz"))
    assert int(b["lm_iters"]) == LM_ITERS and bool(b["whiten"])
    i, w = list(b["drivers"]).index("T_o1"), list(b["windows"]).index(steps)
    return dict(snapshot=b["sample_snapshot_index"][pick], c_hat=b["c_hats"][i, w, pick],
                rel_err=b["rel_err"][i, w, pick], cost0=b["cost0"][i, w, pick],
                cost_final=b["cost_final"][i, w, pick], n_iters_run=b["n_iters"][i, w, pick],
                rollout_steps=int(b["windows"][-1]))


def _setup():
    cfg = ExperimentConfig()
    tw = TwinModels(cfg.eps, DRIVERS["T_o1"])
    mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
    cases = []
    for mode, pick, steps, source in CASES:
        # truth built as the baseline built it: run 1 rolls out the window itself; 0b rolls
        # out its longest window and slices
        base = _baseline(mode, pick, steps, source)
        picks = starting_states(mode, N_STATES, window_steps=max(cfg.window_steps))
        ics, trajs = segments_from_states(tw, picks["states"][pick:pick + 1],
                                          base.get("rollout_steps", steps), cfg.dt)
        snap = int(picks["snapshot_index"][pick])
        assert snap == base["snapshot"], (mode, pick)
        cases.append(dict(x0=ics[0], Y=trajs[0][:steps], regime=str(picks["regime"][pick]),
                          snapshot=snap))
    return cfg, tw, mask, noise_scale(), cases


def _fit(cfg, tw, mask, scale, case, c0):
    return recover(tw, case["x0"], case["Y"], cfg.dt, mask, eta=0.0, n_iters=LM_ITERS,
                   whiten=True, scale=scale, c0=c0)


def run(n_starts=N_STARTS, tag=STEM):
    """Every case x start; writes ``results/data/<tag>.npz``."""
    cfg, tw, mask, scale, cases = _setup()
    starts = draw_starts()[:n_starts]
    shape = (len(CASES), n_starts)
    out = {k: np.full(shape, np.nan) for k in FIELDS}
    out.update({k: np.full(shape + (4,), np.nan) for k in PER_COEF})
    n_iters = np.full(shape, -1)
    c_hist = np.full(shape + (LM_ITERS + 1, 4), np.nan)
    cost_hist = np.full(shape + (LM_ITERS + 1,), np.nan)
    for ic, case in enumerate(cases):
        for s in range(n_starts):
            t0 = time.time()
            r = _fit(cfg, tw, mask, scale, case, starts[s])
            for key, arr in out.items():
                arr[ic, s] = r[key]
            n = len(r["cost_hist"])
            n_iters[ic, s] = n - 1
            c_hist[ic, s, :n] = r["c_hist"]
            cost_hist[ic, s, :n] = r["cost_hist"]
            mode, pick, steps, _ = CASES[ic]
            print(f"{mode:7s} #{pick:2d} {steps_to_days(steps):5.1f} d  start {s:2d}  "
                  f"rel_err {r['rel_err']:.2e}  iters {n - 1:2d}  {time.time() - t0:4.1f} s",
                  flush=True)

    path = os.path.join(DATA_DIR, f"{tag}.npz")
    np.savez(path, **{SAVED_NAME.get(k, k): v for k, v in out.items()}, n_iters=n_iters,
             c_hist=c_hist, cost_hist=cost_hist, starts=starts,
             mode=np.asarray([c[0] for c in CASES]), pick=np.asarray([c[1] for c in CASES]),
             window_steps=np.asarray([c[2] for c in CASES]),
             window_days=np.asarray([steps_to_days(c[2]) for c in CASES]),
             regime=np.asarray([c["regime"] for c in cases]),
             snapshot=np.asarray([c["snapshot"] for c in cases]),
             baseline_rel_err=np.asarray([_baseline(*c)["rel_err"] for c in CASES]),
             baseline_source=np.asarray([c[3] for c in CASES]),
             driver="T_o1", c_true=np.asarray(tw.c_true), eps=cfg.eps, start_seed=START_SEED,
             box=BOX, n_states=N_STATES, lm_iters=LM_ITERS, solver="normal", whiten=True,
             dt=cfg.dt, obs=cfg.obs)
    print(f"[done] wrote {path}", flush=True)
    return path


def smoke():
    """Decision 9 check (2): every case from c = 0 against its baseline; 2 starts per case, timed."""
    cfg, tw, mask, scale, cases = _setup()
    print("c = 0 refit vs baseline (stage0a_run1.npz normal equations, or Stage 0b):")
    for (mode, pick, steps, source), case in zip(CASES, cases):
        r, b = _fit(cfg, tw, mask, scale, case, None), _baseline(mode, pick, steps, source)
        new = dict(c_hat=r["c_hat"], rel_err=r["rel_err"], cost0=r["cost0"],
                   cost_final=r["cost_final"], cost_hist=r["cost_hist"],
                   n_iters_run=len(r["cost_hist"]) - 1)
        for key, v in new.items():
            if key not in b:                 # 0b did not save cost_hist
                continue
            old = b[key][~np.isnan(b[key])] if key == "cost_hist" else b[key]
            same = np.shape(v) == np.shape(old) and np.array_equal(v, old)
            diff = np.max(np.abs(np.asarray(v, float) - old)) if np.shape(v) == np.shape(old) \
                else np.nan
            print(f"  {mode:7s} #{pick:2d} {key:11s} bit-identical {same}  max |diff| {diff:.1e}")
    t0 = time.time()
    run(n_starts=2, tag=f"{STEM}_smoke")
    per_fit = (time.time() - t0) / (2 * len(CASES))
    print(f"\nwall time per random-start fit {per_fit:.1f} s (setup included); "
          f"full run est. {per_fit * N_STARTS * len(CASES) / 60:.1f} min")


def table(tag=STEM):
    """Per case: c = 0 rel_err, starts reached of N, starts that left the box of N."""
    d = np.load(os.path.join(DATA_DIR, f"{tag}.npz"))
    n = d["rel_err"].shape[1]
    reached = (d["rel_err"] < REACHED).sum(axis=1)
    left = (np.nanmax(np.abs(d["c_hist"]), axis=(2, 3)) > float(d["box"])).sum(axis=1)
    lines = [f"Short-window failures from {n} random starts, uniform in "
             f"[-{float(d['box'])}, {float(d['box'])}] per coefficient, seed {int(d['start_seed'])}; "
             f"T_o1, noiseless, {int(d['lm_iters'])} LM iterations. Reached = rel_err < {REACHED}.",
             "",
             f"| Regime | Drawn from | Window | c = 0 fit rel_err | Reached of {n} | "
             f"Left the box of {n} |",
             "|---|---|---|---|---|---|"]
    for i in range(len(d["mode"])):
        lines.append(f"| {d['regime'][i]} | `{d['mode'][i]}` #{d['pick'][i]} | "
                     f"{d['window_days'][i]:.1f} d | {d['baseline_rel_err'][i]:.3g} | "
                     f"{reached[i]} | {left[i]} |")
    print("\n".join(lines))
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(os.path.join(LOGS_DIR, f"{tag}_table.log"), "w") as f:
        f.write("\n".join(lines) + "\n")


def c0_loss_path():
    """Part 2 (a): every case refit from c = 0 with c and cost after each LM iteration. Each
    refit must equal its baseline (c_hat, iterations, and cost history where it was saved).
    Writes ``results/data/<C0_PATH_STEM>.npz``."""
    cfg, tw, mask, scale, cases = _setup()
    c_hist = np.full((len(CASES), LM_ITERS + 1, 4), np.nan)
    cost_hist = np.full((len(CASES), LM_ITERS + 1), np.nan)
    for ic, (c, case) in enumerate(zip(CASES, cases)):
        r, b = _fit(cfg, tw, mask, scale, case, None), _baseline(*c)
        n = len(r["cost_hist"])
        assert np.array_equal(r["c_hat"], b["c_hat"]) and n - 1 == b["n_iters_run"], c
        if "cost_hist" in b:
            assert np.array_equal(r["cost_hist"], b["cost_hist"][~np.isnan(b["cost_hist"])]), c
        c_hist[ic, :n], cost_hist[ic, :n] = r["c_hist"], r["cost_hist"]
        print(f"{c[0]:7s} #{c[1]:2d} {steps_to_days(c[2]):5.1f} d  iters {n - 1:2d}  "
              f"cost {r['cost0']:.2e} -> {r['cost_final']:.2e}  rel_err {r['rel_err']:.3g}  "
              "= baseline", flush=True)
    path = os.path.join(DATA_DIR, f"{C0_PATH_STEM}.npz")
    np.savez(path, c_hist=c_hist, cost_hist=cost_hist,
             mode=np.asarray([c[0] for c in CASES]), pick=np.asarray([c[1] for c in CASES]),
             window_steps=np.asarray([c[2] for c in CASES]),
             window_days=np.asarray([steps_to_days(c[2]) for c in CASES]),
             regime=np.asarray([c["regime"] for c in cases]),
             snapshot=np.asarray([c["snapshot"] for c in cases]),
             baseline_source=np.asarray([c[3] for c in CASES]),
             driver="T_o1", c_true=np.asarray(tw.c_true), c0=np.zeros(4), eps=cfg.eps,
             lm_iters=LM_ITERS, solver="normal", whiten=True, dt=cfg.dt, obs=cfg.obs)
    print(f"[done] wrote {path}", flush=True)
    return path


def c0_loss_path_figure():
    """2 x 4: cost (top) and rel_err (bottom) after each LM iteration, one column per case."""
    plt = figstyle.pyplot()
    d = np.load(os.path.join(DATA_DIR, f"{C0_PATH_STEM}.npz"))
    ct = d["c_true"]
    rel = np.linalg.norm(d["c_hist"] - ct, axis=2) / np.linalg.norm(ct)   # NaN rows stay NaN
    it = np.arange(d["cost_hist"].shape[1])
    n = len(d["mode"])
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.8), sharex=True, squeeze=False)
    for i in range(n):
        ok = ~np.isnan(d["cost_hist"][i])
        axes[0, i].semilogy(it[ok], d["cost_hist"][i][ok], "o-", ms=3, lw=1, color="k")
        axes[1, i].semilogy(it[ok], rel[i][ok], "o-", ms=3, lw=1, color="k")
        axes[1, i].axhline(REACHED, ls="--", lw=1, color="0.5")
        axes[0, i].set_title(f"{d['regime'][i]}, '{d['mode'][i]}' #{d['pick'][i]}, "
                             f"{d['window_days'][i]:.1f} d", fontsize=9)
        axes[1, i].set_xlabel("LM iteration")
    axes[0, 0].set_ylabel("cost 0.5 |r|^2 (whitened)")
    axes[1, 0].set_ylabel("rel_err = |c - c_true| / |c_true|")
    figstyle.finish(fig, "Cost (top) and rel_err (bottom) after each LM iteration, fits from "
                         f"c = 0; T_o1, noiseless; dashed: rel_err = {REACHED:g}")
    path = os.path.join(FIGURES_DIR, f"{C0_PATH_STEM}.png")
    fig.savefig(path, dpi=figstyle.DPI)
    plt.close(fig)
    return [path]


def cost_along_coefficient_axes():
    """Part 2 (b): compiled cost along the 4 coefficient axes and v_min through c_true, per
    case. Checks: centre point is c_true; cost at c = 0 equals the baseline's cost0; v_min
    against run 2's V (first three cases). Writes ``results/data/<SLICE_STEM>.npz``."""
    cfg, tw, mask, scale, cases = _setup()
    oidx = obs_indices(mask)
    ct = np.asarray(tw.c_true)
    offsets = np.linspace(-SLICE_HALF_WIDTH, SLICE_HALF_WIDTH, SLICE_POINTS)
    assert offsets[SLICE_POINTS // 2] == 0.0
    run2 = np.load(os.path.join(DATA_DIR, "stage0a_run2.npz"))
    cost = np.full((len(CASES), len(COEFS) + 1, SLICE_POINTS), np.nan)  # 4 axes, then v_min
    v_min = np.full((len(CASES), 4), np.nan)
    sigma = np.full((len(CASES), 4), np.nan)
    col_norm = np.full((len(CASES), 4), np.nan)
    cost_ctrue_compiled = np.full(len(CASES), np.nan)
    cost_ctrue_uncompiled = np.full(len(CASES), np.nan)
    for ic, (c, case) in enumerate(zip(CASES, cases)):
        t0 = time.time()
        resid = make_residual(tw, case["x0"], case["Y"], cfg.dt, oidx,
                              wchan=1.0 / np.asarray(scale)[oidx])
        resid_j = jax.jit(resid)

        def cost_at(cc):                     # as lm_fit evaluates it
            r = resid_j(jnp.asarray(cc))
            return float(0.5 * r @ r)

        assert cost_at(np.zeros(4)) == _baseline(*c)["cost0"], c
        J = residual_jacobian(tw, case["x0"], case["Y"], cfg.dt, mask, whiten=True, scale=scale)
        _, s, Vt = np.linalg.svd(J, full_matrices=False)
        v = Vt[-1] * np.sign(Vt[-1][COEFS.index("T_o1")])
        v_min[ic], sigma[ic], col_norm[ic] = v, s, np.linalg.norm(J, axis=0)
        for k, direction in enumerate(list(np.eye(4)) + [v]):
            for p, o in enumerate(offsets):
                cost[ic, k, p] = cost_at(ct + o * direction)
        cost_ctrue_compiled[ic] = cost_at(ct)
        r = np.asarray(resid(jnp.asarray(ct)))
        cost_ctrue_uncompiled[ic] = float(0.5 * r @ r)
        msg = ""
        if c[3] == "run1":
            i2 = np.flatnonzero((run2["mode"] == c[0]) & (run2["pick"] == c[1]))[0]
            v2 = run2["V"][i2][:, -1]
            msg = f"  v_min vs run 2: {np.degrees(np.arccos(min(1.0, abs(v2 @ v)))):.2e} deg"
        print(f"{c[0]:7s} #{c[1]:2d} {steps_to_days(c[2]):5.1f} d  cost at c_true compiled "
              f"{cost_ctrue_compiled[ic]:.2e}, uncompiled {cost_ctrue_uncompiled[ic]:.2e}  "
              f"cost(c = 0) = baseline cost0{msg}  {time.time() - t0:.0f} s", flush=True)
    assert not np.isnan(cost).any()
    path = os.path.join(DATA_DIR, f"{SLICE_STEM}.npz")
    np.savez(path, cost=cost, offsets=offsets, directions=np.asarray(COEFS + ("v_min",)),
             v_min=v_min, sigma=sigma, col_norm=col_norm,
             cost_ctrue_compiled=cost_ctrue_compiled,
             cost_ctrue_uncompiled=cost_ctrue_uncompiled,
             mode=np.asarray([c[0] for c in CASES]), pick=np.asarray([c[1] for c in CASES]),
             window_steps=np.asarray([c[2] for c in CASES]),
             window_days=np.asarray([steps_to_days(c[2]) for c in CASES]),
             regime=np.asarray([c["regime"] for c in cases]),
             snapshot=np.asarray([c["snapshot"] for c in cases]),
             driver="T_o1", c_true=ct, eps=cfg.eps, whiten=True, dt=cfg.dt, obs=cfg.obs)
    print(f"[done] wrote {path}", flush=True)
    return path


def cost_along_coefficient_axes_figure():
    """4 x 4: cost against offset from c_true, one column per case, one row per coefficient;
    the v_min slice dashed on the T_o1 row (x = distance along v_min)."""
    plt = figstyle.pyplot()
    d = np.load(os.path.join(DATA_DIR, f"{SLICE_STEM}.npz"))
    n, iv = len(d["mode"]), len(COEFS)
    fig, axes = plt.subplots(len(COEFS), n, figsize=(3.2 * n, 10.0), sharex=True,
                             squeeze=False)
    for i in range(n):
        for k, name in enumerate(COEFS):
            ax = axes[k, i]
            ax.plot(d["offsets"], d["cost"][i, k], "-", lw=1.2, color="k",
                    label="one coefficient varied, others at c_true")
            if name == "T_o1":
                ax.plot(d["offsets"], d["cost"][i, iv], "--", lw=1.2, color="C3",
                        label="along v_min (weakest direction of J at c_true); x = distance")
            if i == 0:
                ax.set_ylabel(f"cost, {name} varied")
        axes[0, i].set_title(f"{d['regime'][i]}, '{d['mode'][i]}' #{d['pick'][i]}, "
                             f"{d['window_days'][i]:.1f} d", fontsize=9)
        axes[-1, i].set_xlabel("offset from c_true")
    figstyle.finish(fig, "Cost 0.5 |r|^2 (whitened, compiled) along each coefficient axis "
                         "through c_true, and along v_min (dashed); T_o1, noiseless")
    fig.subplots_adjust(bottom=0.09)
    fig.legend(*axes[COEFS.index("T_o1"), 0].get_legend_handles_labels(), loc="lower center",
               ncol=2, fontsize=8, frameon=False)
    path = os.path.join(FIGURES_DIR, f"{SLICE_STEM}.png")
    fig.savefig(path, dpi=figstyle.DPI)
    plt.close(fig)
    return [path]


def cost_along_T_o1_axis_and_v_min_log_figure():
    """1 x 4: (b)'s T_o1-axis slice (solid) and v_min slice (dashed) on a log cost axis."""
    plt = figstyle.pyplot()
    d = np.load(os.path.join(DATA_DIR, f"{SLICE_STEM}.npz"))
    n, k = len(d["mode"]), COEFS.index("T_o1")
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True, squeeze=False)
    for i, ax in enumerate(axes[0]):
        ax.semilogy(d["offsets"], d["cost"][i, k], "-", lw=1.2, color="k",
                    label="T_o1 varied, others at c_true")
        ax.semilogy(d["offsets"], d["cost"][i, len(COEFS)], "--", lw=1.2, color="C3",
                    label="along v_min (weakest direction of J at c_true); x = distance")
        ax.set_title(f"{d['regime'][i]}, '{d['mode'][i]}' #{d['pick'][i]}, "
                     f"{d['window_days'][i]:.1f} d", fontsize=9)
        ax.set_xlabel("offset from c_true")
    axes[0, 0].set_ylabel("cost 0.5 |r|^2 (whitened, compiled)")
    figstyle.finish(fig, "Cost along the T_o1 axis and along v_min through c_true, log scale; "
                         "T_o1, noiseless")
    fig.subplots_adjust(bottom=0.27)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8,
               frameon=False)
    path = os.path.join(FIGURES_DIR, "noiseless_short_window_failures_cost_along_T_o1_axis_and_"
                                     "v_min_log.png")
    fig.savefig(path, dpi=figstyle.DPI)
    plt.close(fig)
    return [path]


def cost_in_plane_of_T_o1_axis_and_v_min():
    """Compiled cost on a grid in the plane through c_true spanned by e1 = v_min and e2 = the
    T_o1 axis with its v_min part removed (Gram-Schmidt). Checks: the y = 0 row equals (b)'s
    v_min slice at the shared offsets; the grid centre equals (b)'s cost at c_true. Projects
    (a)'s LM path onto the plane and keeps its distance off the plane."""
    cfg, tw, mask, scale, cases = _setup()
    oidx = obs_indices(mask)
    ct = np.asarray(tw.c_true)
    b = np.load(os.path.join(DATA_DIR, f"{SLICE_STEM}.npz"))
    a = np.load(os.path.join(DATA_DIR, f"{C0_PATH_STEM}.npz"))
    k = COEFS.index("T_o1")
    xs = np.linspace(-SLICE_HALF_WIDTH, SLICE_HALF_WIDTH, PLANE_POINTS)
    shared = np.flatnonzero(np.isin(b["offsets"], xs))
    assert np.array_equal(b["offsets"][shared], xs)
    nc = len(CASES)
    cost = np.full((nc, PLANE_POINTS, PLANE_POINTS), np.nan)          # [case, y, x]
    ys = np.full((nc, PLANE_POINTS), np.nan)
    e1s, e2s = np.full((nc, 4), np.nan), np.full((nc, 4), np.nan)
    angle = np.full(nc, np.nan)
    path_xy = np.full((nc, LM_ITERS + 1, 2), np.nan)
    path_off = np.full((nc, LM_ITERS + 1), np.nan)
    for ic, (c, case) in enumerate(zip(CASES, cases)):
        t0 = time.time()
        resid_j = jax.jit(make_residual(tw, case["x0"], case["Y"], cfg.dt, oidx,
                                        wchan=1.0 / np.asarray(scale)[oidx]))

        def cost_at(cc):                     # as lm_fit evaluates it
            r = resid_j(jnp.asarray(cc))
            return float(0.5 * r @ r)

        e1 = b["v_min"][ic]
        axis = np.eye(4)[k]
        e2 = axis - (axis @ e1) * e1
        e2 /= np.linalg.norm(e2)
        e1s[ic], e2s[ic] = e1, e2
        angle[ic] = np.degrees(np.arccos(abs(axis @ e1)))
        ys[ic] = np.linspace(-1, 1, PLANE_POINTS) * PLANE_ACROSS * np.sin(np.radians(angle[ic]))
        for iy, y in enumerate(ys[ic]):
            for ix, x in enumerate(xs):
                cost[ic, iy, ix] = cost_at(ct + x * e1 + y * e2)
        mid = PLANE_POINTS // 2
        assert ys[ic, mid] == 0.0
        assert np.array_equal(cost[ic, mid], b["cost"][ic, len(COEFS), shared]), c
        assert cost[ic, mid, mid] == b["cost_ctrue_compiled"][ic], c
        ok = ~np.isnan(a["cost_hist"][ic])
        dc = a["c_hist"][ic][ok] - ct
        xy = np.stack([dc @ e1, dc @ e2], axis=1)
        path_xy[ic, :ok.sum()] = xy
        path_off[ic, :ok.sum()] = np.linalg.norm(dc - xy[:, :1] * e1 - xy[:, 1:] * e2, axis=1)
        print(f"{c[0]:7s} #{c[1]:2d} {steps_to_days(c[2]):5.1f} d  angle T_o1 axis to v_min "
              f"{angle[ic]:.2f} deg  y half-range {ys[ic, -1]:.2e}  cost {np.nanmin(cost[ic]):.1e}"
              f" to {np.nanmax(cost[ic]):.1e}  LM path off the plane: max "
              f"{np.nanmax(path_off[ic]):.1e}, end {path_off[ic][ok.sum() - 1]:.1e}  "
              f"y = 0 row = (b) v_min slice  {time.time() - t0:.0f} s", flush=True)
    assert not np.isnan(cost).any()
    path = os.path.join(DATA_DIR, f"{PLANE_STEM}.npz")
    np.savez(path, cost=cost, xs=xs, ys=ys, e1=e1s, e2=e2s, angle_deg=angle, path_xy=path_xy,
             path_off_plane=path_off, mode=b["mode"], pick=b["pick"],
             window_days=b["window_days"], regime=b["regime"], c_true=ct)
    print(f"[done] wrote {path}", flush=True)
    return path


def cost_in_plane_of_T_o1_axis_and_v_min_figure():
    """1 x 4: log10 cost over the plane, with the T_o1 axis, c_true, c = 0 and (a)'s LM path
    (projected) drawn on it."""
    plt = figstyle.pyplot()
    d = np.load(os.path.join(DATA_DIR, f"{PLANE_STEM}.npz"))
    ct, k = d["c_true"], COEFS.index("T_o1")
    logc = np.log10(np.maximum(d["cost"], 10 ** LOG_COST_FLOOR))
    levels = np.arange(LOG_COST_FLOOR, np.ceil(logc.max()) + 1, 1.0)
    n = len(d["mode"])
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.6), squeeze=False)
    for i, ax in enumerate(axes[0]):
        xs, ys = d["xs"], d["ys"][i]
        cs = ax.contourf(xs, ys, logc[i], levels=levels, cmap="viridis")
        # T_o1 axis: c_true + a e_T_o1, a in [-0.2, 0.2], in plane coordinates
        a = np.linspace(-SLICE_HALF_WIDTH, SLICE_HALF_WIDTH, 2)
        ax.plot(a * d["e1"][i][k], a * d["e2"][i][k], "-", color="k", lw=1, label="T_o1 axis")
        start = -ct                                      # c = 0, relative to c_true
        ax.plot(start @ d["e1"][i], start @ d["e2"][i], "s", color="w", mec="k", ms=6,
                label="c = 0 (LM start)")
        p = d["path_xy"][i][~np.isnan(d["path_xy"][i, :, 0])]
        ax.plot(p[:, 0], p[:, 1], "o-", color="C1", ms=2.5, lw=0.8,
                label="LM path from c = 0, projected")
        ax.plot(p[-1, 0], p[-1, 1], "o", color="C3", mec="k", ms=6, label="LM end (c_hat)")
        ax.plot(0, 0, "*", color="w", mec="k", ms=10, label="c_true")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_title(f"{d['regime'][i]}, '{d['mode'][i]}' #{d['pick'][i]}, "
                     f"{d['window_days'][i]:.1f} d\nT_o1 axis {d['angle_deg'][i]:.2f} deg "
                     "from v_min", fontsize=9)
        ax.set_xlabel("distance along v_min (from c_true)")
    axes[0, 0].set_ylabel("distance across, toward the T_o1 axis")
    figstyle.finish(fig, "log10 cost in the plane through c_true spanned by v_min and the T_o1 "
                         f"axis (clipped at {LOG_COST_FLOOR:g}); T_o1, noiseless")
    fig.subplots_adjust(bottom=0.3, right=0.9)
    cax = fig.add_axes([0.92, 0.3, 0.012, 0.55])
    fig.colorbar(cs, cax=cax, label="log10 cost")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8,
               frameon=False)
    path = os.path.join(FIGURES_DIR, f"{PLANE_STEM}.png")
    fig.savefig(path, dpi=figstyle.DPI)
    plt.close(fig)
    return [path]


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "full"
    if arg == "smoke":
        smoke()
    elif arg == "full":
        run()
    elif arg == "table":
        table()
    elif arg == "c0_loss_path":
        c0_loss_path()
    elif arg == "c0_loss_path_figure":
        c0_loss_path_figure()
    elif arg == "cost_along_coefficient_axes":
        cost_along_coefficient_axes()
    elif arg == "cost_along_coefficient_axes_figure":
        cost_along_coefficient_axes_figure()
    elif arg == "cost_along_T_o1_axis_and_v_min_log_figure":
        cost_along_T_o1_axis_and_v_min_log_figure()
    elif arg == "cost_in_plane_of_T_o1_axis_and_v_min":
        cost_in_plane_of_T_o1_axis_and_v_min()
    elif arg == "cost_in_plane_of_T_o1_axis_and_v_min_figure":
        cost_in_plane_of_T_o1_axis_and_v_min_figure()
    else:
        sys.exit(f"unknown argument {arg!r}")
