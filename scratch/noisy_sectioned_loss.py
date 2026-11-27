"""Sectioned loss with observation noise (``plans/sectioned-loss-noisy.md``): with noise, does
restarting the 89.7-d window every tau steps change the fit, compared with the unsectioned noisy
fit on the same state and noise draw?

Everything else as the noiseless run (``scratch/sectioned_loss.py``): the 16 Stage 0b ``both``
starting states, all five drivers, LM from c = 0, 0b's iteration count, whitened by the
high-frequency std. One noise draw per state (seed index 0), drawn per eta
(``shared_noise(..., eta=)``, decision 2). Two start conditions (decision 3):

- ``true_state``: ``x0`` exact; section k starts from the true state one step before its rows.
- ``noisy_obs``: ``x0 + eta * w0 * hf_std`` (``start_noise``); section k starts from the noisy
  observation one step before its rows, so its start error is that row's observation noise.

Per (driver, state): the sectioned fit at every tau in ``TAUS``, the unsectioned 8000-step fit
(one rollout, ``tau=None``, the path Stage 1 used), and single windows of ``SINGLES`` steps from
the same ``x0`` (decision 4). Writes one file per (driver, eta, condition) under ``_path``.

Longer window (``plans/sectioned-loss-longer-window.md``): ``long`` runs one (eta, condition) on
a 16000-step (179.4 d) window, theta_a1 and T_o1 only: every tau in ``TAUS``, tau = 8000 (two
89.7-d sections) and the unsectioned 16000-step fit. Same keys, so the draw and section starts
over the first 8000 rows are the 8000-step run's (``check_prefix``); no single windows, which
would repeat that run's. File names gain ``_window_16000_steps``.

    python scratch/noisy_sectioned_loss.py 1e-3 noisy_obs     # one (eta, condition), 5 drivers
    python scratch/noisy_sectioned_loss.py smoke 1e-3 noisy_obs   # both #0, #1, T_o1
    python scratch/noisy_sectioned_loss.py check_stage1        # old key vs Stage 1's 8000 fit
    python scratch/noisy_sectioned_loss.py figure              # figure A per start condition
    python scratch/noisy_sectioned_loss.py long 1e-3 noisy_obs        # 16000 steps, 2 drivers
    python scratch/noisy_sectioned_loss.py smoke_long 1e-3 noisy_obs  # both #0, #1, T_o1
    python scratch/noisy_sectioned_loss.py check_prefix        # 16000-step truth, noise prefix
    python scratch/noisy_sectioned_loss.py figure_long         # 89.7 d vs 179 d figures, table

``figure`` (figures decisions 1-4, ``plans/sectioned-loss-noisy.md``): one figure per start
condition, one panel per driver, median rel_err over the 16 states against tau, eta 1e-4 and 1e-3
sectioned (solid, last point the unsectioned fit; band = 4th to 13th of the 16 sorted values) and
single window (dashed). The states-reached figure adds the noiseless sectioned run in black (its
last point 0b's unsectioned fit); the median figure leaves it out (Aiden, 2026-11-25), and the
table keeps its median. The values go to ``results/logs/noisy_sectioned_loss_both_figure_tables.log``.
"""

import os
import sys
import time

import numpy as np

from error_discovery import figstyle
from error_discovery.attractor import segments_from_states
from error_discovery.config import (ExperimentConfig, DATA_DIR, FIGURES_DIR, RESULTS_DIR,
                                    steps_to_days)
from error_discovery.discovery import recover
from error_discovery.experiments.driver_window import DRIVERS
from error_discovery.models import TwinModels, obs_mask
from error_discovery.sampling import (load_snapshots, noise_scale, shared_noise, start_noise,
                                      eta_key)

MODE = "both"
TAUS = (40, 80, 125, 160, 250, 320, 500, 1000, 2000, 4000)    # as the noiseless run
SINGLES = (40, 80, 160, 320)
CONDITIONS = ("true_state", "noisy_obs")
ETAS = (1e-4, 1e-3)                  # decision 2
SEED_INDEX = 0                       # one noise draw per state (decision 4)
LONG_STEPS = 16000                   # longer-window plan, decision 2 (179.4 d)
LONG_DRIVERS = ("theta_a1", "T_o1")  # its decision 4
REACHED = 1e-3                       # rel_err below this = reached c_true (the noiseless criterion)
FIELDS = ("rel_err", "cond_J", "smin", "smax", "rank", "cost0", "cost_final", "cost_true")
PER_COEF = ("c_hat", "col_norm", "se_unit", "vif", "align")
SAVED_NAME = {"c_hat": "c_hats", "rank": "jrank"}             # as 0b names them
CONDITION_LABEL = {"true_state": "true state",
                   "noisy_obs": "noisy observation (x0 and every section start)"}
START_ERROR = {
    "true_state": "none: x0 exact, section k starts from Y_true[k*tau - 1]",
    "noisy_obs": "x0 + eta * w0 * hf_std (w0 = start_noise); section k starts from the noisy "
                 "observation Y_obs[k*tau - 1]",
}


def _baseline():
    return np.load(os.path.join(DATA_DIR, f"driver_window_stage0b_{MODE}.npz"))


def _eta_name(eta):
    return f"eta{eta:.0e}".replace("e-0", "e-")


def _path(driver, eta, condition, prefix="noisy_sectioned_loss", window=None):
    suffix = f"_window_{window}_steps" if window else ""
    return os.path.join(DATA_DIR, f"{prefix}_{MODE}_{driver}_{_eta_name(eta)}_section_starts_"
                                  f"{condition}{suffix}.npz")


def _empty(n_rows, n_states):
    out = {k: np.full((n_rows, n_states), np.nan) for k in FIELDS}
    out.update({k: np.full((n_rows, n_states, 4), np.nan) for k in PER_COEF})
    out["n_iters"] = np.full((n_rows, n_states), -1)
    return out


def _store(out, row, k, r):
    for key in FIELDS + PER_COEF:
        out[key][row, k] = r[key]
    out["n_iters"][row, k] = len(r["cost_hist"]) - 1


def run(eta, condition, drivers=tuple(DRIVERS), n_states=None, prefix="noisy_sectioned_loss",
        window=None):
    """Every tau, the unsectioned fit and the single windows, per state, for each driver.

    ``window``: a window length in steps other than 0b's 8000. Adds tau = 8000, drops the single
    windows, and the file name gains the window. ``None``: 0b's 8000 steps, as before.
    """
    assert condition in CONDITIONS
    cfg = ExperimentConfig()
    b = _baseline()
    assert str(b["sampling"]) == MODE
    base_steps = int(b["windows"][-1])
    n_steps = window or base_steps
    lm_iters, whiten = int(b["lm_iters"]), bool(b["whiten"])
    idx = b["sample_snapshot_index"][:n_states]
    scale = noise_scale()
    states = load_snapshots(idx)
    noisy = condition == "noisy_obs"
    taus = TAUS + ((base_steps,) if window else ()) + (n_steps,)
    singles = () if window else SINGLES
    paths = []
    for driver in drivers:
        tw = TwinModels(cfg.eps, DRIVERS[driver])
        mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
        ics, trajs = segments_from_states(tw, states, n_steps, cfg.dt)
        sec, single = _empty(len(taus), len(idx)), _empty(len(singles), len(idx))
        for k, snap in enumerate(idx):
            z = shared_noise(snap, SEED_INDEX, n_steps, tw.ndim, eta=eta)
            x0 = np.asarray(ics[k])
            if noisy:
                x0 = x0 + eta * start_noise(snap, SEED_INDEX, tw.ndim, eta) * scale
            kw = dict(eta=eta, n_iters=lm_iters, whiten=whiten, scale=scale, noise=z,
                      cost_at_true=True)
            for it, tau in enumerate(taus):
                t0 = time.time()
                r = recover(tw, x0, trajs[k], cfg.dt, mask,
                            tau=None if tau == n_steps else tau, noisy_starts=noisy, **kw)
                _store(sec, it, k, r)
                print(f"[{driver}] tau {tau:5d}  state {k:2d}  rel_err {r['rel_err']:.2e}  "
                      f"{time.time() - t0:5.1f} s", flush=True)
            for iw, w in enumerate(singles):
                t0 = time.time()
                r = recover(tw, x0, trajs[k][:w], cfg.dt, mask, **kw)
                _store(single, iw, k, r)
                print(f"[{driver}] single {w:4d}  state {k:2d}  rel_err {r['rel_err']:.2e}  "
                      f"{time.time() - t0:5.1f} s", flush=True)

        path = _path(driver, eta, condition, prefix, window)
        np.savez(path, **{SAVED_NAME.get(k, k): v for k, v in sec.items()},
                 **{"single_" + SAVED_NAME.get(k, k): v for k, v in single.items()},
                 drivers=np.asarray([driver]), g_coeffs=np.asarray([DRIVERS[driver]]),
                 c_true=np.asarray(tw.c_true)[None], tau_steps=np.asarray(taus),
                 tau_days=np.asarray([steps_to_days(t) for t in taus]),
                 n_sections=np.asarray([n_steps // t for t in taus]),
                 single_steps=np.asarray(singles, dtype=int),
                 single_days=np.asarray([steps_to_days(w) for w in singles]),
                 n_steps=n_steps, lm_iters=lm_iters, c0=np.zeros(4), whiten=whiten,
                 eps=cfg.eps, obs=cfg.obs, dt=cfg.dt, sampling=MODE, eta=eta,
                 condition=condition, start_error=START_ERROR[condition],
                 seed_index=SEED_INDEX, eta_key=eta_key(eta),
                 noise_key="z: default_rng([21217, snapshot, seed_index, eta_key]); "
                           "w0: default_rng([21217, snapshot, seed_index, eta_key, 1])",
                 **{k: b[k][:len(idx)] if b[k].ndim else b[k]
                    for k in b.files if k.startswith("sample_")})
        print(f"[done] wrote {path}", flush=True)
        paths.append(path)
    return paths


def check_stage1(eta=1e-3, driver="T_o1", n_states=2):
    """Plan decision 6 check (2): the old key (``eta=None``), true starts, 8000 steps, against
    Stage 1's unsectioned fit on the same states, seed 0."""
    cfg = ExperimentConfig()
    s1 = np.load(os.path.join(DATA_DIR, f"driver_window_stage1_{driver}_{MODE}.npz"))
    b = _baseline()
    idx = b["sample_snapshot_index"][:n_states]
    assert np.array_equal(s1["sample_snapshot_index"][:n_states], idx)
    ie = list(s1["noise_levels"]).index(eta)
    n_steps = int(s1["noise_windows"][-1])
    tw = TwinModels(cfg.eps, DRIVERS[driver])
    mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
    scale = noise_scale()
    ics, trajs = segments_from_states(tw, load_snapshots(idx), n_steps, cfg.dt)
    for k, snap in enumerate(idx):
        z = shared_noise(snap, 0, n_steps, tw.ndim)
        r = recover(tw, ics[k], trajs[k], cfg.dt, mask, eta=eta, n_iters=int(s1["lm_iters"]),
                    whiten=bool(s1["whiten"]), scale=scale, noise=z)
        old = {"c_hats": s1["c_hats_noise"][0, ie, -1, k, 0],
               "rel_err": s1["rel_err_noise"][0, ie, -1, k, 0],
               "cost0": s1["cost0_noise"][0, ie, -1, k, 0],
               "cost_final": s1["cost_final_noise"][0, ie, -1, k, 0],
               "n_iters": s1["n_iters_noise"][0, ie, -1, k, 0]}
        new = {"c_hats": r["c_hat"], "rel_err": r["rel_err"], "cost0": r["cost0"],
               "cost_final": r["cost_final"], "n_iters": len(r["cost_hist"]) - 1}
        for key in old:
            diff = np.max(np.abs(np.asarray(new[key], float) - old[key]))
            print(f"state {k} {key:10s} bit-identical {np.array_equal(new[key], old[key])}  "
                  f"max |diff| {diff:.1e}", flush=True)


def check_prefix(n_states=1):
    """Longer-window plan, decision 6 check (1): rows 0-7999 of the 16000-step truth (per
    driver) and noise draw (per eta) are bit-identical to the 8000-step ones, so the longer fits
    see the 8000-step run's data plus 8000 rows more. ``x0``'s draw takes no length."""
    cfg = ExperimentConfig()
    b = _baseline()
    base_steps = int(b["windows"][-1])
    idx = b["sample_snapshot_index"][:n_states]
    states = load_snapshots(idx)
    for driver in LONG_DRIVERS:
        tw = TwinModels(cfg.eps, DRIVERS[driver])
        _, short = segments_from_states(tw, states, base_steps, cfg.dt)
        _, long = segments_from_states(tw, states, LONG_STEPS, cfg.dt)
        for k in range(len(idx)):
            print(f"[{driver}] state {k}: truth rows 0-{base_steps - 1} bit-identical "
                  f"{np.array_equal(long[k][:base_steps], short[k])}", flush=True)
    for k, snap in enumerate(idx):
        for eta in ETAS:
            z_short = shared_noise(snap, SEED_INDEX, base_steps, tw.ndim, eta=eta)
            z_long = shared_noise(snap, SEED_INDEX, LONG_STEPS, tw.ndim, eta=eta)
            print(f"state {k} eta {eta:g}: noise rows 0-{base_steps - 1} bit-identical "
                  f"{np.array_equal(z_long[:base_steps], z_short)}", flush=True)


def _median_band(x):
    """Median over states (last axis) and the band drawn: the 4th smallest and 4th largest
    (4th and 13th of 16)."""
    s = np.sort(x, axis=-1)
    return np.median(x, axis=-1), s[..., 3], s[..., -4]


def _shared_draw_singles(driver, eta, b):
    """Single windows longer than ``SINGLES`` and shorter than the full window from the earlier
    fits with the old shared noise key (seed 0, ``x0`` exact, true-state starts only):
    ``driver_window_both.npz`` (eta 1e-4, all five drivers), else Stage 1 (T_o1, theta_a1, eta
    1e-3). Not this run's draw. Returns (steps, rel_err of shape windows x states) or None."""
    for name in (f"driver_window_{MODE}.npz", f"driver_window_stage1_{driver}_{MODE}.npz"):
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            continue
        d = np.load(path)
        if driver not in list(d["drivers"]) or eta not in list(d["noise_levels"]):
            continue
        assert np.array_equal(d["sample_snapshot_index"], b["sample_snapshot_index"]), name
        assert bool(d["whiten"]) == bool(b["whiten"]) and int(d["lm_iters"]) == int(b["lm_iters"])
        w = np.asarray(d["noise_windows"])
        keep = (w > max(SINGLES)) & (w < int(b["windows"][-1]))
        rel = d["rel_err_noise"][list(d["drivers"]).index(driver),
                                 list(d["noise_levels"]).index(eta)][keep][..., 0]
        return w[keep], rel
    return None


def _legend_rank(label):
    """Line kind first, then eta: the legend fills column by column, so each eta is a row."""
    if label.startswith("noiseless"):
        kind = 3
    elif "earlier" in label:
        kind = 2
    elif "single window" in label:
        kind = 1
    else:
        kind = 0
    eta = [f"eta {e:.0e}" in label for e in ETAS] + [True]
    return kind, eta.index(True)


def figure():
    """Per start condition, figure A (median rel_err) and the states reached of 16
    (rel_err < ``REACHED``, as the noiseless figure); the values drawn as a table."""
    plt = figstyle.pyplot()
    b = _baseline()
    noiseless = {n: np.load(os.path.join(DATA_DIR, f"sectioned_loss_{MODE}_{n}.npz"))
                 for n in DRIVERS}
    colors = plt.get_cmap("tab10").colors
    lines, paths = [], []
    for condition in CONDITIONS:
        runs = {(n, e): np.load(_path(n, e, condition)) for n in DRIVERS for e in ETAS}
        for (n, e), d in runs.items():
            assert np.array_equal(d["sample_snapshot_index"], b["sample_snapshot_index"]), n
            assert np.array_equal(d["sample_snapshot_index"],
                                  noiseless[n]["sample_snapshot_index"]), n
            assert bool(d["whiten"]) == bool(b["whiten"]) and int(d["n_steps"]) == \
                int(b["windows"][-1]) and float(d["eta"]) == e and str(d["condition"]) == condition
            assert np.array_equal(d["tau_steps"][:-1], noiseless[n]["tau_steps"]), n
        n_states = b["sample_snapshot_index"].shape[0]
        figs = {kind: plt.subplots(1, len(DRIVERS), figsize=(3.4 * len(DRIVERS), 3.9),
                                   sharey=True, squeeze=False) for kind in ("median", "reached")}
        lines.append(f"section starts: {CONDITION_LABEL[condition]}; median [4th, 13th] of "
                     f"{n_states} states' rel_err, then states with rel_err < {REACHED:g}")
        lines.append(f"{'driver':9s}{'tau':>6s}{'days':>7s}" + "".join(
            f"{'eta ' + format(e, '.0e'):>38s}" for e in ETAS) + f"{'noiseless':>18s}")
        old_rows = {}
        for k_ax, n in enumerate(DRIVERS):
            ax, ax_r = figs["median"][1][0, k_ax], figs["reached"][1][0, k_ax]
            i = list(b["drivers"]).index(n)
            ref = np.concatenate([noiseless[n]["rel_err"], b["rel_err"][i, -1][None]])
            ref_med = np.median(ref, axis=-1)
            for j, e in enumerate(ETAS):
                d = runs[(n, e)]
                med, lo, hi = _median_band(d["rel_err"])
                ax.fill_between(d["tau_days"], lo, hi, color=colors[j], alpha=0.18, lw=0)
                ax.plot(d["tau_days"], med, "o-", ms=4, color=colors[j],
                        label=f"eta {e:.0e}, sectioned (band: 4th-13th of {n_states})")
                x, y = list(d["single_days"]), list(np.median(d["single_rel_err"], axis=-1))
                old = _shared_draw_singles(n, e, b) if condition == "true_state" else None
                if old is not None:
                    x_old, y_old = [steps_to_days(w) for w in old[0]], np.median(old[1], axis=-1)
                    ax.plot(x + x_old, y + list(y_old), "--", lw=1, color=colors[j])
                    ax.plot(x_old, y_old, "o", ms=4, mfc="white", color=colors[j],
                            label=f"eta {e:.0e}, single window, earlier fits (shared draw)")
                    old_rows[(n, e)] = (old[0], y_old, (old[1] < REACHED).sum(axis=1))
                ax.plot(x, y, "o--", ms=4, lw=1, color=colors[j],
                        label=f"eta {e:.0e}, single window")
                ax_r.plot(d["tau_days"], (d["rel_err"] < REACHED).sum(axis=1), "o-", ms=4,
                          color=colors[j], label=f"eta {e:.0e}, sectioned")
                ax_r.plot(d["single_days"], (d["single_rel_err"] < REACHED).sum(axis=1), "o--",
                          ms=4, lw=1, color=colors[j], label=f"eta {e:.0e}, single window")
            d = runs[(n, ETAS[0])]
            ax_r.plot(d["tau_days"], (ref < REACHED).sum(axis=1), "o-", ms=3, lw=1,
                      color="black", label="noiseless, sectioned")
            for it, tau in enumerate(d["tau_steps"]):
                cells = "".join("{:>10.2e} [{:.2e}, {:.2e}] {:>2d}/{}".format(
                    *(v[it] for v in _median_band(runs[(n, e)]["rel_err"])),
                    int((runs[(n, e)]["rel_err"][it] < REACHED).sum()), n_states) for e in ETAS)
                lines.append(f"{n:9s}{int(tau):6d}{d['tau_days'][it]:7.2f}{cells}"
                             f"{ref_med[it]:12.2e} {int((ref[it] < REACHED).sum()):>2d}/{n_states}")
            for iw, w in enumerate(d["single_steps"]):
                cells = "".join("{:>10.2e}{:>23s}{:>2d}/{}".format(
                    np.median(runs[(n, e)]["single_rel_err"][iw]), "",
                    int((runs[(n, e)]["single_rel_err"][iw] < REACHED).sum()), n_states)
                    for e in ETAS)
                lines.append(f"{n:9s}{'s' + str(int(w)):>6s}{d['single_days'][iw]:7.2f}{cells}")
            for (n_old, e), (ws, meds, hits) in old_rows.items():
                if n_old != n:
                    continue
                for w, med, hit in zip(ws, meds, hits):
                    lines.append(f"{n:9s}{'s' + str(int(w)) + '*':>6s}{steps_to_days(w):7.2f}  "
                                 f"eta {e:.0e}: {med:.2e}  {hit:>2d}/{n_states}")
            for a in (ax, ax_r):
                a.set_xscale("log")
                a.set_xlabel("section length tau / single-window length  [days]", fontsize=9)
                a.set_title(n, fontsize=10)
            ax.set_yscale("log")
            ax_r.set_ylim(-0.5, n_states + 0.5)
            ax_r.yaxis.get_major_locator().set_params(integer=True)
        lines.append("")
        what = {"median": f"Median rel_err over {n_states} '{MODE}' states",
                "reached": f"States reached (rel_err < {REACHED:g}) of {n_states} '{MODE}' states"}
        ylabel = {"median": f"median rel_err over {n_states} states",
                  "reached": f"states reached (rel_err < {REACHED:g}) of {n_states}"}
        stem = {"median": "median_rel_err", "reached": "states_reached"}
        for kind, (fig, axes) in figs.items():
            axes[0, 0].set_ylabel(ylabel[kind])
            extra = ("; open markers: earlier single-window fits, shared noise draw, x0 exact"
                     if kind == "median" and old_rows else "")
            figstyle.finish(fig, f"{what[kind]}, section starts: {CONDITION_LABEL[condition]}. "
                                 "89.7-d window restarted every tau (solid; last point "
                                 f"unsectioned), single window of length tau (dashed){extra}")
            fig.subplots_adjust(bottom=0.3)
            handles = {}
            for a in axes[0]:
                for h, lab in zip(*a.get_legend_handles_labels()):
                    handles.setdefault(lab, h)
            labs = sorted(handles, key=_legend_rank)
            fig.legend([handles[lab] for lab in labs], labs, loc="lower center",
                       ncol=-(-len(labs) // len(ETAS)), fontsize=8, frameon=False)
            path = os.path.join(FIGURES_DIR, f"noisy_sectioned_loss_{MODE}_{stem[kind]}_section_"
                                             f"starts_{condition}.png")
            fig.savefig(path, dpi=figstyle.DPI)
            plt.close(fig)
            paths.append(path)

    lines.insert(0, "Single windows: rows s<steps>, median and states reached only; s<steps>* = "
                    "earlier fits with the shared noise draw (seed 0, x0 exact). tau 8000 = "
                    "unsectioned fit (noiseless: 0b's).")
    print("\n".join(lines))
    logs = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(logs, exist_ok=True)
    with open(os.path.join(logs, f"noisy_sectioned_loss_{MODE}_figure_tables.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return paths


def figure_long():
    """Longer-window plan, figure decisions 1-4: per start condition, median rel_err (band 4th to
    13th of 16) and states reached of 16 against tau, the 89.7-d (8000-step) and 179-d
    (16000-step) sectioned runs on the same axes, theta_a1 and T_o1. Each run's last point is its
    unsectioned fit (open marker; noiseless 89.7 d: 0b's). No single windows. The values drawn
    go to a table."""
    plt = figstyle.pyplot()
    b = _baseline()
    base_steps, n_states = int(b["windows"][-1]), b["sample_snapshot_index"].shape[0]
    windows = {"89.7 d": None, "179 d": LONG_STEPS}          # label -> ``window`` argument
    style = {"89.7 d": "o-", "179 d": "s--"}
    colors = plt.get_cmap("tab10").colors
    noiseless = {}
    for n in LONG_DRIVERS:
        short = np.load(os.path.join(DATA_DIR, f"sectioned_loss_{MODE}_{n}.npz"))
        long = np.load(os.path.join(DATA_DIR, f"sectioned_loss_{MODE}_{n}_window_{LONG_STEPS}"
                                              f"_steps.npz"))
        for d in (short, long):
            assert np.array_equal(d["sample_snapshot_index"], b["sample_snapshot_index"]), n
            assert bool(d["whiten"]) == bool(b["whiten"]) and int(d["lm_iters"]) == \
                int(b["lm_iters"]), n
        i = list(b["drivers"]).index(n)
        noiseless[(n, "89.7 d")] = np.concatenate([short["rel_err"], b["rel_err"][i, -1][None]])
        noiseless[(n, "179 d")] = long["rel_err"]
    lines, paths = [], []
    for condition in CONDITIONS:
        runs = {(n, e, w): np.load(_path(n, e, condition, window=windows[w]))
                for n in LONG_DRIVERS for e in ETAS for w in windows}
        for (n, e, w), d in runs.items():
            assert np.array_equal(d["sample_snapshot_index"], b["sample_snapshot_index"]), (n, w)
            assert bool(d["whiten"]) == bool(b["whiten"]) and int(d["lm_iters"]) == \
                int(b["lm_iters"]) and float(d["eta"]) == e and str(d["condition"]) == condition
            assert int(d["n_steps"]) == (windows[w] or base_steps), (n, w)
            assert np.array_equal(d["tau_steps"][:len(TAUS)], TAUS), (n, w)
            assert len(d["tau_steps"]) == len(noiseless[(n, w)]), (n, w)
        figs = {kind: plt.subplots(1, len(LONG_DRIVERS), figsize=(4.2 * len(LONG_DRIVERS), 4.1),
                                   sharey=True, squeeze=False) for kind in ("median", "reached")}
        lines.append(f"section starts: {CONDITION_LABEL[condition]}; median [4th, 13th] of "
                     f"{n_states} states' rel_err, then states with rel_err < {REACHED:g}")
        lines.append(f"{'driver':9s}{'window':>7s}{'tau':>6s}{'days':>7s}" + "".join(
            f"{'eta ' + format(e, '.0e'):>38s}" for e in ETAS) + f"{'noiseless':>18s}")
        for k_ax, n in enumerate(LONG_DRIVERS):
            ax, ax_r = figs["median"][1][0, k_ax], figs["reached"][1][0, k_ax]
            for w in windows:
                for j, e in enumerate(ETAS):
                    d = runs[(n, e, w)]
                    x = d["tau_days"]
                    med, lo, hi = _median_band(d["rel_err"])
                    ax.fill_between(x, lo, hi, color=colors[j], alpha=0.12, lw=0)
                    for a, y in ((ax, med), (ax_r, (d["rel_err"] < REACHED).sum(axis=1))):
                        a.plot(x, y, style[w], ms=4, lw=1.2, color=colors[j],
                               label=f"eta {e:.0e}, {w} window")
                        a.plot(x[-1], y[-1], style[w][0], ms=4, mfc="white", color=colors[j])
                x = runs[(n, ETAS[0], w)]["tau_days"]
                y = (noiseless[(n, w)] < REACHED).sum(axis=1)
                ax_r.plot(x, y, style[w], ms=3, lw=1, color="black",
                          label=f"noiseless, {w} window")
                ax_r.plot(x[-1], y[-1], style[w][0], ms=3, mfc="white", color="black")
            taus = sorted({int(t) for w in windows for t in runs[(n, ETAS[0], w)]["tau_steps"]})
            for tau in taus:
                for w in windows:
                    steps = list(runs[(n, ETAS[0], w)]["tau_steps"])
                    if tau not in steps:
                        continue
                    it = steps.index(tau)
                    cells = "".join("{:>10.2e} [{:.2e}, {:.2e}] {:>2d}/{}".format(
                        *(v[it] for v in _median_band(runs[(n, e, w)]["rel_err"])),
                        int((runs[(n, e, w)]["rel_err"][it] < REACHED).sum()), n_states)
                        for e in ETAS)
                    ref = noiseless[(n, w)][it]
                    lines.append(f"{n:9s}{w:>7s}{tau:6d}{steps_to_days(tau):7.2f}{cells}"
                                 f"{np.median(ref):12.2e} {int((ref < REACHED).sum()):>2d}/"
                                 f"{n_states}")
            for a in (ax, ax_r):
                a.set_xscale("log")
                a.set_xlabel("section length tau  [days]", fontsize=9)
                a.set_title(n, fontsize=10)
            ax.set_yscale("log")
            ax_r.set_ylim(-0.5, n_states + 0.5)
            ax_r.yaxis.get_major_locator().set_params(integer=True)
        lines.append("")
        what = {"median": f"Median rel_err over {n_states} '{MODE}' states (band: 4th-13th)",
                "reached": f"States reached (rel_err < {REACHED:g}) of {n_states} '{MODE}' states"}
        ylabel = {"median": f"median rel_err over {n_states} states",
                  "reached": f"states reached (rel_err < {REACHED:g}) of {n_states}"}
        stem = {"median": "median_rel_err", "reached": "states_reached"}
        for kind, (fig, axes) in figs.items():
            axes[0, 0].set_ylabel(ylabel[kind])
            figstyle.finish(fig, f"{what[kind]},\nsection starts: {CONDITION_LABEL[condition]}.\n"
                                 "Window restarted every tau: 89.7 d (circles, solid), 179 d "
                                 "(squares, dashed); open marker: unsectioned fit")
            fig.subplots_adjust(bottom=0.3)
            handles, labs = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labs, loc="lower center", ncol=2, fontsize=8, frameon=False)
            path = os.path.join(FIGURES_DIR, f"noisy_sectioned_loss_{MODE}_{stem[kind]}_section_"
                                             f"starts_{condition}_window_{base_steps}_vs_"
                                             f"{LONG_STEPS}_steps.png")
            fig.savefig(path, dpi=figstyle.DPI)
            plt.close(fig)
            paths.append(path)

    lines.insert(0, f"89.7-d ({base_steps}-step) and 179-d ({LONG_STEPS}-step) sectioned runs. "
                    f"Last tau of each window = its unsectioned fit (89.7 d noiseless: 0b's); "
                    f"179 d at tau {base_steps} = two 89.7-d sections.")
    print("\n".join(lines))
    logs = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(logs, exist_ok=True)
    with open(os.path.join(logs, f"noisy_sectioned_loss_{MODE}_window_{base_steps}_vs_"
                                 f"{LONG_STEPS}_steps_figure_tables.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return paths


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "check_stage1":
        check_stage1()
    elif args[0] == "figure":
        print(figure())
    elif args[0] == "smoke":
        t0 = time.time()
        run(float(args[1]), args[2], drivers=("T_o1",), n_states=2,
            prefix="smoke_noisy_sectioned_loss")
        n_fits = 2 * (len(TAUS) + 1 + len(SINGLES))
        print(f"wall time {time.time() - t0:.0f} s, {(time.time() - t0) / n_fits:.1f} s per fit "
              f"(setup included)", flush=True)
    elif args[0] == "check_prefix":
        check_prefix()
    elif args[0] == "figure_long":
        print(figure_long())
    elif args[0] == "long":
        run(float(args[1]), args[2], drivers=LONG_DRIVERS, window=LONG_STEPS)
    elif args[0] == "smoke_long":
        t0 = time.time()
        run(float(args[1]), args[2], drivers=("T_o1",), n_states=2,
            prefix="smoke_noisy_sectioned_loss", window=LONG_STEPS)
        n_fits = 2 * (len(TAUS) + 2)
        print(f"wall time {time.time() - t0:.0f} s, {(time.time() - t0) / n_fits:.1f} s per fit "
              f"(setup included)", flush=True)
    else:
        run(float(args[0]), args[1])
