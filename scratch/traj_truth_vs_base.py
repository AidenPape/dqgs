"""The true model and the model with the error left out (c = 0), run from each Stage 0b ``both``
starting state over the sweep's longest window (8,000 steps, 89.7 d). Nothing is fitted.
``plans/noisy-initial-conditions.md``, "First: trajectory figure".

One figure per state, 6 x 6 panels: all 36 variables in state-vector order. Each panel has the true
model once per driver (``driver_window.DRIVERS``) and ``TwinModels.f_base`` (c = 0), which is the
same for every driver. The noiseless obs equal the truth, so none are drawn. Only the figures are
written; re-rendering re-runs the model (~3 min on the laptop).

    python scratch/traj_truth_vs_base.py
"""

import os

import numpy as np
import jax.numpy as jnp

from error_discovery import figstyle
from error_discovery.config import ExperimentConfig, DATA_DIR, FIGURES_DIR, steps_to_days
from error_discovery.discovery import rollout
from error_discovery.experiments.driver_window import DRIVERS
from error_discovery.models import TwinModels
from error_discovery.sampling import load_snapshots

MODE = "both"
OUT_DIR = os.path.join(FIGURES_DIR, "trajectories")


def variable_names(vr):
    """'psi_a,1' ... 'T_o,8' from ``spec.variables_range`` (block ends, e.g. (10, 20, 28, 36))."""
    blocks = (("psi_a", 0, vr[0]), ("theta_a", vr[0], vr[1]), ("psi_o", vr[1], vr[2]),
              ("T_o", vr[2], vr[3]))
    return [f"{name},{i - lo + 1}" for name, lo, hi in blocks for i in range(lo, hi)]


def trajectories(models, x0, dt, n_steps):
    """{"c = 0": ..., driver: ...}, each (n_steps + 1, ndim) with x0 as row 0."""
    x0 = jnp.asarray(x0)
    runs = {"c = 0": next(iter(models.values())).f_base,
            **{name: tw.f_true for name, tw in models.items()}}
    return {name: np.vstack([np.asarray(x0)[None], np.asarray(rollout(f, x0, dt, n_steps))])
            for name, f in runs.items()}


def draw(trajs, names, t_days, title, path):
    plt = figstyle.pyplot()
    colors = plt.get_cmap("tab10").colors           # driver colours as driver_window.figure
    fig, axes = plt.subplots(6, 6, figsize=(18, 15), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t_days, trajs["c = 0"][:, j], color="k", lw=0.8, label="c = 0 (error left out)")
        for i, name in enumerate(DRIVERS):
            ax.plot(t_days, trajs[name][:, j], color=colors[i], lw=0.7, label=f"truth, g = {name}")
        ax.set_title(names[j], fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("days", fontsize=7)
    figstyle.finish(fig, title)
    fig.subplots_adjust(bottom=0.06)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=9,
               frameon=False)
    fig.savefig(path, dpi=figstyle.DPI)
    plt.close(fig)


def main():
    cfg = ExperimentConfig()
    d = np.load(os.path.join(DATA_DIR, f"driver_window_stage0b_{MODE}.npz"))
    assert str(d["sampling"]) == MODE
    n_steps = int(d["windows"][-1])                   # 8000, the longest window
    models = {name: TwinModels(cfg.eps, g) for name, g in DRIVERS.items()}
    names = variable_names(next(iter(models.values())).spec.variables_range)
    t_days = steps_to_days(np.arange(n_steps + 1))
    states = load_snapshots(d["sample_snapshot_index"])
    os.makedirs(OUT_DIR, exist_ok=True)
    for k, (snap, year, regime) in enumerate(zip(d["sample_snapshot_index"],
                                                 d["sample_start_year"], d["sample_regime"])):
        trajs = trajectories(models, states[k], cfg.dt, n_steps)
        assert all(np.isfinite(v).all() for v in trajs.values()), k
        title = (f"True model per driver g and c = 0 (error left out), {t_days[-1]:.1f} d from "
                 f"'{MODE}' state #{k} (snapshot {snap}, year {year:.1f}, {regime} at start); "
                 f"36 variables, nondimensional")
        path = os.path.join(OUT_DIR, f"traj_truth_c0_{MODE}_s{k:02d}.png")
        draw(trajs, names, t_days, title, path)
        print(path, flush=True)


if __name__ == "__main__":
    main()
