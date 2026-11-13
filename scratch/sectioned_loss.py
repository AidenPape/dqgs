"""Sectioned loss (``plans/sectioned-loss.md``; Gjini 2026, ch. 4): does restarting the 89.7-d
window every tau steps change whether the noiseless fit reaches c_true, compared with the
unsectioned fit on the same state?

tau is the section length in model steps (``TAUS``: 40 to 4000, 0.45 to 44.9 d). For each of the
16 Stage 0b ``both`` starting states and each of the five drivers, one noiseless fit per tau:
``recover(..., tau=tau)`` with Stage 0b's settings (LM, normal equations, c0 = 0, 0b's iteration
count, whitened by the high-frequency std). Section 0 starts at the state, section k at the true
state one step before its first observed row; every section's residual goes into one sum.
Writes ``results/data/sectioned_loss_both_<driver>.npz``, one per driver.

``figure`` reads those with ``driver_window_stage0b_both.npz`` for the two baselines: the
unsectioned 8000-step fit, and at tau = 40 / 80 / 160 / 320 the single window of length tau from
the same state. Reached = rel_err < 1e-3. It draws (a), states reached of 16 against tau, and
prints and logs (b), per-state changes against each baseline, and (c), failed fits by kind:
cost_final / cost0 (cost left when LM stops over cost at c = 0) below 1e-10 "fell to ~0", above
0.5 "barely moved", else "partial".

    python scratch/sectioned_loss.py              # all five drivers, one process
    python scratch/sectioned_loss.py T_o1         # one driver
    python scratch/sectioned_loss.py smoke        # both #0, #1, T_o1, every tau + 8000; checks
    python scratch/sectioned_loss.py figure       # figure and tables from the saved runs
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
from error_discovery.discovery import recover, make_sectioned_residual, obs_indices
from error_discovery.experiments.driver_window import DRIVERS
from error_discovery.models import TwinModels, obs_mask
from error_discovery.sampling import load_snapshots, noise_scale

MODE = "both"
TAUS = (40, 80, 125, 160, 250, 320, 500, 1000, 2000, 4000)    # decision 2
REACHED = 1e-3                    # rel_err below this = reached c_true (0a / 0b threshold)
FELL, MOVED = 1e-10, 0.5          # cost_final / cost0 cut-offs for failed fits (decision 6c)
FIELDS = ("rel_err", "cond_J", "smin", "smax", "rank", "cost0", "cost_final")
PER_COEF = ("c_hat", "col_norm", "se_unit", "vif", "align")
SAVED_NAME = {"c_hat": "c_hats", "rank": "jrank"}             # as 0b names them
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")


def _baseline():
    return np.load(os.path.join(DATA_DIR, f"driver_window_stage0b_{MODE}.npz"))


def _path(driver, tag=MODE):
    return os.path.join(DATA_DIR, f"sectioned_loss_{tag}_{driver}.npz")


def run(driver, n_states=None, taus=TAUS, tag=MODE):
    """Every tau x state for one driver, on 0b's starting states and settings."""
    cfg = ExperimentConfig()
    b = _baseline()
    assert str(b["sampling"]) == MODE
    n_steps, lm_iters, whiten = int(b["windows"][-1]), int(b["lm_iters"]), bool(b["whiten"])
    idx = b["sample_snapshot_index"][:n_states]
    tw = TwinModels(cfg.eps, DRIVERS[driver])
    mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
    scale = noise_scale()
    ics, trajs = segments_from_states(tw, load_snapshots(idx), n_steps, cfg.dt)

    shape = (len(taus), len(idx))
    out = {k: np.full(shape, np.nan) for k in FIELDS}
    out.update({k: np.full(shape + (4,), np.nan) for k in PER_COEF})
    n_iters = np.full(shape, -1)
    for it, tau in enumerate(taus):
        for k in range(len(idx)):
            t0 = time.time()
            r = recover(tw, ics[k], trajs[k], cfg.dt, mask, eta=0.0, n_iters=lm_iters,
                        whiten=whiten, scale=scale, tau=tau)
            for key, arr in out.items():
                arr[it, k] = r[key]
            n_iters[it, k] = len(r["cost_hist"]) - 1
            print(f"[{driver}] tau {tau:5d}  state {k:2d}  rel_err {r['rel_err']:.2e}  "
                  f"{time.time() - t0:5.1f} s", flush=True)

    path = _path(driver, tag)
    np.savez(path, **{SAVED_NAME.get(k, k): v for k, v in out.items()}, n_iters=n_iters,
             drivers=np.asarray([driver]), g_coeffs=np.asarray([DRIVERS[driver]]),
             c_true=np.asarray(tw.c_true)[None], tau_steps=np.asarray(taus),
             tau_days=np.asarray([steps_to_days(t) for t in taus]),
             n_sections=np.asarray([n_steps // t for t in taus]), n_steps=n_steps,
             lm_iters=lm_iters, c0=np.zeros(4), whiten=whiten, eps=cfg.eps, obs=cfg.obs,
             dt=cfg.dt, sampling=MODE,
             **{k: b[k][:len(idx)] if b[k].ndim else b[k]
                for k in b.files if k.startswith("sample_")})
    print(f"[done] wrote {path}", flush=True)
    return path


def _cost_at_c_true(tw, x0, Y, tau, oidx, wchan, dt):
    """Compiled cost at c_true, the way ``lm_fit`` evaluates it: rounding only."""
    starts = np.vstack([np.asarray(x0)[None], np.asarray(Y)[tau - 1:-1:tau]])
    r = jax.jit(make_sectioned_residual(tw, starts, Y, dt, oidx, wchan))(jnp.asarray(tw.c_true))
    return 0.5 * float(r @ r)


def smoke(driver="T_o1", n_states=2):
    """Decision 9 check (2): tau = 8000 against 0b's fit, cost at c_true per tau, time per fit."""
    taus = TAUS + (8000,)
    t0 = time.time()
    path = run(driver, n_states=n_states, taus=taus, tag="smoke")
    per_fit = (time.time() - t0) / (len(taus) * n_states)
    d, b = np.load(path), _baseline()
    i = list(b["drivers"]).index(driver)
    print(f"\nwall time per fit {per_fit:.1f} s (setup included)")
    print("tau = 8000 vs 0b's unsectioned fit, per state:")
    for key in ("c_hats", "rel_err", "cost0", "cost_final", "n_iters"):
        for k in range(n_states):
            new, old = d[key][-1, k], b[key][i, -1, k]
            print(f"  {key:10s} state {k}: bit-identical {np.array_equal(new, old)}  "
                  f"max |diff| {np.max(np.abs(np.asarray(new, float) - old)):.1e}")
    cfg = ExperimentConfig()
    tw = TwinModels(cfg.eps, DRIVERS[driver])
    oidx = obs_indices(obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs))
    wchan = 1.0 / noise_scale()[oidx]
    ics, trajs = segments_from_states(tw, load_snapshots(b["sample_snapshot_index"][:n_states]),
                                      int(b["windows"][-1]), cfg.dt)
    print("compiled cost at c_true (cost0 for scale):")
    for it, tau in enumerate(taus):
        costs = [_cost_at_c_true(tw, ics[k], trajs[k], tau, oidx, wchan, cfg.dt)
                 for k in range(n_states)]
        print(f"  tau {tau:5d}: " + "  ".join(f"state {k} {c:.2e} (cost0 {d['cost0'][it, k]:.2e})"
                                              for k, c in enumerate(costs)))


def _tables(runs, b):
    """Lines of tables (b) and (c)."""
    out = []
    taus = runs[next(iter(runs))]["tau_steps"]
    wins = list(b["windows"])
    head = (f"{'driver':9s}{'tau':>6s}{'days':>7s} | vs unsectioned: both  sec  base  none | "
            f"vs single window: both  sec  base  none")
    out += ["(b) states by outcome against each baseline; reached = rel_err < 1e-3; "
            "sec = sectioned only, base = baseline only", head]
    for driver, d in runs.items():
        i = list(b["drivers"]).index(driver)
        for it, tau in enumerate(taus):
            sec = d["rel_err"][it] < REACHED
            cells = []
            n = len(sec)
            for base in (b["rel_err"][i, -1, :n], b["rel_err"][i, wins.index(tau), :n]
                         if tau in wins else None):
                if base is None:
                    cells.append(f"{'-':>5s}{'-':>5s}{'-':>6s}{'-':>6s}")
                    continue
                ok = base < REACHED
                cells.append(f"{np.sum(sec & ok):5d}{np.sum(sec & ~ok):5d}{np.sum(~sec & ok):6d}"
                             f"{np.sum(~sec & ~ok):6d}")
            out.append(f"{driver:9s}{tau:6d}{steps_to_days(tau):7.2f} |{'':16s}{cells[0]} |"
                       f"{'':18s}{cells[1]}")
    out += ["", "(c) failed fits (rel_err >= 1e-3) by cost_final / cost0: < 1e-10 fell to ~0, "
            "> 0.5 barely moved, else partial",
            f"{'driver':9s}{'tau':>6s}{'days':>7s}{'failed':>8s}{'~0':>6s}{'partial':>9s}"
            f"{'barely':>8s}"]
    for driver, d in runs.items():
        for it, tau in enumerate(taus):
            fail = d["rel_err"][it] >= REACHED
            left = d["cost_final"][it][fail] / d["cost0"][it][fail]
            out.append(f"{driver:9s}{tau:6d}{steps_to_days(tau):7.2f}{fail.sum():8d}"
                       f"{np.sum(left < FELL):6d}{np.sum((left >= FELL) & (left <= MOVED)):9d}"
                       f"{np.sum(left > MOVED):8d}")
    return out


def figure(tag=MODE):
    """(a) as a figure; (b) and (c) printed and written to ``results/logs/``."""
    plt = figstyle.pyplot()
    b = _baseline()
    runs = {n: np.load(_path(n, tag)) for n in DRIVERS if os.path.exists(_path(n, tag))}
    for n, d in runs.items():
        k = d["rel_err"].shape[1]
        assert np.array_equal(d["sample_snapshot_index"], b["sample_snapshot_index"][:k]), n
        assert bool(d["whiten"]) == bool(b["whiten"]) and n in list(b["drivers"]), n
        assert int(b["windows"][-1]) == int(d["n_steps"]), n
    wins = list(b["windows"])
    colors = plt.get_cmap("tab10").colors           # driver colours as driver_window.figure
    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 3.6), sharey=True,
                             squeeze=False)
    for ax, (n, d) in zip(axes[0], runs.items()):
        i, col = list(b["drivers"]).index(n), colors[list(DRIVERS).index(n)]
        k = d["rel_err"].shape[1]
        ax.plot(d["tau_days"], (d["rel_err"] < REACHED).sum(axis=1), "o-", ms=4, color=col,
                label="sectioned")
        ax.axhline((b["rel_err"][i, -1, :k] < REACHED).sum(), ls="--", color=col, lw=1,
                   label="unsectioned 89.7 d (0b)")
        single = [(t, (b["rel_err"][i, wins.index(t), :k] < REACHED).sum())
                  for t in d["tau_steps"] if t in wins]
        ax.plot([steps_to_days(t) for t, _ in single], [c for _, c in single], "o", ms=6,
                mfc="none", color=col, label="single window of length tau (0b)")
        ax.set_xscale("log")
        ax.set_ylim(-0.5, k + 0.5)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_xlabel("section length tau  [days]")
        ax.set_title(n, fontsize=10)
    axes[0, 0].set_ylabel(f"states reached (rel_err < {REACHED:g}) of {k}")
    figstyle.finish(fig, f"States reached of {k}, '{MODE}' mode, noiseless: fit with the 89.7-d "
                         "window restarted every tau, unsectioned fit, single window of length tau")
    fig.subplots_adjust(bottom=0.27)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8,
               frameon=False)
    path = os.path.join(FIGURES_DIR, f"sectioned_loss_{tag}_reached.png")
    fig.savefig(path, dpi=figstyle.DPI)
    plt.close(fig)

    lines = _tables(runs, b)
    print("\n".join(lines))
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(os.path.join(LOGS_DIR, f"sectioned_loss_{tag}_tables.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return [path]


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "smoke":
        smoke()
    elif arg == "figure":
        figure()
    elif arg in DRIVERS:
        run(arg)
    elif arg == "all":
        for name in DRIVERS:
            run(name)
    else:
        sys.exit(f"unknown argument {arg!r}")
