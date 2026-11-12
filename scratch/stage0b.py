"""Stage 0b (plans/fitting-experiments.md, "Staging"; notes/fitting_log.md): in how many of
16 starting states does the noiseless fit reach c_true, per regime x driver x window?

``driver_window.run`` once per regime (passive, active, both), 16 states, no noisy fits
(``eta_grid=()``), every other knob at its default: 5 drivers, the 13 windows 0.45-89.7 d,
40 LM iterations, c0 = 0, normal equations, whitened by the high-frequency std. A fit reached
c_true iff rel_err < 1e-3 (0a pass threshold). Writes
``results/data/driver_window_stage0b_<regime>.npz``, prints the counts and draws
``results/figures/stage0b.png`` and the table as ``results/figures/stage0b_table.png``.

    python scratch/stage0b.py            # full run, ~4-5 h (est.)
    python scratch/stage0b.py smoke      # T_o1, passive, 2 states: checks the code only
    python scratch/stage0b.py figure     # table and figure from the saved runs
"""

import os
import sys

import numpy as np

from error_discovery import figstyle
from error_discovery.config import DATA_DIR, FIGURES_DIR
from error_discovery.experiments import driver_window

REGIMES = ("passive", "active", "both")
N_STATES = 16
FAIL = 1e-3                          # a fit failed iff rel_err >= FAIL


def _path(tag, regime):
    return os.path.join(DATA_DIR, f"driver_window_{tag}_{regime}.npz")


def run(tag="stage0b", regimes=REGIMES, n_states=N_STATES, drivers=driver_window.ALL_DRIVERS):
    for regime in regimes:
        driver_window.run(tag=f"{tag}_{regime}", sampling=regime, n_samples=n_states,
                          eta_grid=(), drivers=drivers)


def table(tag="stage0b", regimes=REGIMES):
    """Print the number of states reached, rows regime x driver, columns windows."""
    runs = {r: np.load(_path(tag, r)) for r in regimes if os.path.exists(_path(tag, r))}
    d0 = next(iter(runs.values()))
    head = "".join(f"{w:>6.1f}" for w in d0["window_days"])
    print(f"states reached (rel_err < {FAIL:g}) of {d0['rel_err'].shape[2]}; columns = window (d)")
    print(f"{'regime':8s}{'driver':9s}{head}")
    for regime, d in runs.items():
        reached = (d["rel_err"] < FAIL).sum(axis=2)          # (driver, window)
        for name, row in zip(d["drivers"], reached):
            print(f"{regime:8s}{name:9s}" + "".join(f"{v:6d}" for v in row))
    return runs


def figure(tag="stage0b", regimes=REGIMES):
    """One panel per regime: states reached against window, one line per driver."""
    plt = figstyle.pyplot()
    runs = {r: np.load(_path(tag, r)) for r in regimes if os.path.exists(_path(tag, r))}
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.8), sharey=True,
                             squeeze=False)
    for ax, (regime, d) in zip(axes[0], runs.items()):
        n = d["rel_err"].shape[2]
        for name, row in zip(d["drivers"], (d["rel_err"] < FAIL).sum(axis=2)):
            ax.semilogx(d["window_days"], row, "o-", ms=3, lw=1, label=name)
        ax.set_ylim(-0.5, n + 0.5)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_title(regime, fontsize=10)
        ax.set_xlabel("window (days)")
    axes[0, 0].set_ylabel(f"states reached c_true (of {n})")
    axes[0, -1].legend(fontsize=8, loc="lower right")
    figstyle.finish(fig, f"Stage 0b: noiseless fits reaching c_true (rel_err < {FAIL:g})")
    out = os.path.join(FIGURES_DIR, f"{tag}.png")
    fig.savefig(out, dpi=figstyle.DPI)
    return [out]


def table_figure(tag="stage0b", regimes=REGIMES):
    """The terminal table as an image: rows regime x driver, columns windows, cells = states
    reached. Alternate regimes shaded grey."""
    plt = figstyle.pyplot()
    runs = {r: np.load(_path(tag, r)) for r in regimes if os.path.exists(_path(tag, r))}
    d0 = next(iter(runs.values()))
    rows, shade = [], []
    for i, (regime, d) in enumerate(runs.items()):
        for name, row in zip(d["drivers"], (d["rel_err"] < FAIL).sum(axis=2)):
            rows.append([regime, str(name)] + [str(v) for v in row])
            shade.append("0.92" if i % 2 else "white")
    head = ["regime", "driver"] + [f"{w:.2g}" if w < 1 else f"{w:.1f}"
                                   for w in d0["window_days"]]
    fig, ax = plt.subplots(figsize=(6.4, 0.2 * len(rows) + 0.6))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=head, loc="center", cellLoc="center",
                 cellColours=[[c] * len(head) for c in shade])
    t.auto_set_font_size(False); t.set_fontsize(9)
    t.auto_set_column_width(range(len(head)))
    for j in range(len(head)):
        t[0, j].set_text_props(weight="bold")
    figstyle.finish(fig, f"Stage 0b: states reaching c_true (rel_err < {FAIL:g}) of "
                         f"{d0['rel_err'].shape[2]}; columns = window (days)")
    out = os.path.join(FIGURES_DIR, f"{tag}_table.png")
    fig.savefig(out, dpi=figstyle.DPI, bbox_inches="tight")
    return [out]


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "smoke":
        kw = dict(tag="stage0b_smoke", regimes=("passive",))
        run(n_states=2, drivers=("T_o1",), **kw)
    else:
        kw = {}
        if mode == "full":
            run()
    table(**kw)
    print(figure(**kw), table_figure(**kw))
