"""Stage 1 (plans/fitting-experiments.md, "Staging"; notes/fitting_log.md): on one state, does
the spread of the noisy fits across seeds match linear theory, eta x s.e. per unit noise, and
how does it change with eta?

``driver_window.run`` once per regime (passive, active, both): T_o1 only, 16 states, noisy fits
on all 13 windows (0.45-89.7 d) at eta = 1e-4, 1e-3, 1e-2, 1e-1, 3 seeds (common random
numbers), no noiseless sweep (0b makes those fits). Every other knob at its default. Writes
``results/driver_window_stage1_<regime>.npz``, prints tables A and B, draws Figs. 1 and 2.

Measured std = std of the fitted T_o1 coefficient across seeds (ddof 1). Predicted std = eta x
s.e. per unit noise from the residual Jacobian at c_true, read from
``driver_uncertainty_<regime>_n60.npz`` (same states, windows and whitening; checked on load).

    python scratch/stage1.py            # full run, ~11 h (est.)
    python scratch/stage1.py smoke      # passive, 2 states, 2 seeds, eta 1e-4 and 1e-1
    python scratch/stage1.py figure     # tables and figures from the saved runs
"""

import os
import sys

import numpy as np
from scipy import stats

from error_discovery import figstyle
from error_discovery.config import RESULTS_DIR, ETA_GRID
from error_discovery.experiments import driver_window

REGIMES = ("passive", "active", "both")
N_STATES = 16
N_SEEDS = 3
DRIVER = "T_o1"
K = 2                                # T_o1's index in c = (const, theta_a1, T_o1, psi_a1)


def _path(tag, regime):
    return os.path.join(RESULTS_DIR, f"driver_window_{tag}_{regime}.npz")


def _load(tag, regimes):
    return {r: np.load(_path(tag, r)) for r in regimes if os.path.exists(_path(tag, r))}


def run(tag="stage1", regimes=REGIMES, n_states=N_STATES, etas=ETA_GRID, n_seeds=N_SEEDS):
    for regime in regimes:
        driver_window.run(tag=f"{tag}_{regime}", sampling=regime, n_samples=n_states,
                          drivers=(DRIVER,), eta_grid=etas, n_seeds=n_seeds,
                          noiseless=False, noise_on_all_windows=True)


def spreads(d, regime):
    """Measured std, predicted std and |bias| of the T_o1 coefficient, each (eta, window, state).
    Bias = mean error across seeds."""
    u = np.load(os.path.join(RESULTS_DIR, f"driver_uncertainty_{regime}_n60.npz"))
    n = d["c_hats_noise"].shape[3]
    assert np.array_equal(u["sample_snapshot_index"][:n], d["sample_snapshot_index"]), regime
    assert np.array_equal(u["windows"], d["noise_windows"]), regime
    assert bool(u["whiten"]) == bool(d["whiten"]), regime
    se = u["se_unit"][list(u["drivers"]).index(DRIVER), :, :n, K]     # (window, state)
    err = d["c_hats_noise"][0, ..., K] - d["c_true"][0, K]            # (eta, window, state, seed)
    predicted = d["noise_levels"][:, None, None] * se[None]
    return err.std(axis=-1, ddof=1), predicted, np.abs(err.mean(axis=-1))


def median_band(x, axis=-1):
    """The r-th smallest and r-th largest of n along ``axis``: holds the median with probability
    1 - 2 P(Bin(n, 1/2) <= r - 1) whatever the distribution (Conover 1999, ch. 3). r is the
    largest reaching 95%: r = 4 of 16 (97.9%). Min to max when none does."""
    n = x.shape[axis]
    ok = [r for r in range(1, n // 2 + 1) if 1 - 2 * stats.binom.cdf(r - 1, n, 0.5) >= 0.95]
    r = max(ok, default=1)
    s = np.sort(x, axis=axis)
    return np.take(s, r - 1, axis=axis), np.take(s, n - r, axis=axis)


def table(tag="stage1", regimes=REGIMES):
    """A: median over states of measured / predicted std. B: of |bias| / predicted std.
    Rows regime x eta, columns windows."""
    runs = _load(tag, regimes)
    d0 = next(iter(runs.values()))
    n, n_seeds = d0["c_hats_noise"].shape[3:5]
    # |mean of n_seeds errors| is half-normal with scale predicted / sqrt(n_seeds) when the fit is
    # unbiased and the theory exact; its median is 0.674 of that scale
    ref = stats.norm.ppf(0.75) / np.sqrt(n_seeds)
    head = "".join(f"{w:>9.1f}" for w in d0["noise_window_days"])
    rows = {r: spreads(d, r) for r, d in runs.items()}
    for title, j in (("A: measured / predicted std (1 if linear theory holds)", 0),
                     (f"B: |bias| / predicted std ({ref:.2f} if unbiased and theory exact)", 2)):
        print(f"{title}; T_o1 coefficient, median over {n} states, {n_seeds} seeds; "
              f"columns = window (d)")
        print(f"{'regime':8s}{'eta':>7s}{head}")
        for regime, s in rows.items():
            q = np.median(s[j] / s[1], axis=-1)                       # (eta, window)
            for eta, row in zip(runs[regime]["noise_levels"], q):
                print(f"{regime:8s}{eta:7.0e}" + "".join(f"{v:9.3g}" for v in row))
        print()
    return runs


def figure(tag="stage1", regimes=REGIMES):
    """Fig. 1 (one file per window): measured vs predicted std, one dot per state and eta.
    Fig. 2: median over states of measured / predicted against window, one line per eta."""
    plt = figstyle.pyplot()
    runs = _load(tag, regimes)
    d0 = next(iter(runs.values()))
    etas, wd = d0["noise_levels"], d0["noise_window_days"]
    n, n_seeds = d0["c_hats_noise"].shape[3:5]
    # 90% range of a std from n_seeds draws around the true one (chi, n_seeds - 1 dof)
    lo, hi = np.sqrt(stats.chi2.ppf([0.05, 0.95], n_seeds - 1) / (n_seeds - 1))
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.85, len(etas)))
    S = {r: spreads(d, r) for r, d in runs.items()}
    out = []

    for iw, w in enumerate(wd):
        fig, axes = plt.subplots(1, len(S), figsize=(4.2 * len(S), 4.0), squeeze=False)
        for ax, (regime, (m, p, _)) in zip(axes[0], S.items()):
            for ie, eta in enumerate(etas):
                ax.loglog(p[ie, iw], m[ie, iw], "o", ms=4, color=colors[ie],
                          label=f"eta = {eta:.0e}")
            both = np.concatenate([p[:, iw].ravel(), m[:, iw].ravel()])
            lims = np.array([both.min() / 3, both.max() * 3])
            ax.fill_between(lims, lo * lims, hi * lims, color="0.88", lw=0, zorder=0)
            ax.plot(lims, lims, color="0.5", lw=0.8, zorder=1)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_title(regime, fontsize=10)
            ax.set_xlabel("predicted std (eta x s.e. at c_true)")
        axes[0, 0].set_ylabel(f"measured std across {n_seeds} seeds")
        axes[0, -1].legend(fontsize=8, loc="upper left")
        figstyle.finish(fig, f"Stage 1: T_o1 coefficient, {w:.1f} d window; one dot per state "
                             f"and eta; grey = 1:1 and the 90% range of a {n_seeds}-seed std")
        path = os.path.join(RESULTS_DIR, f"{tag}_states_{w:06.2f}d.png")
        fig.savefig(path, dpi=figstyle.DPI); plt.close(fig)
        out.append(path)

    fig, axes = plt.subplots(1, len(S), figsize=(4.2 * len(S), 3.8), sharey=True,
                             squeeze=False)
    for ax, (regime, (m, p, _)) in zip(axes[0], S.items()):
        q = m / p                                                     # (eta, window, state)
        band_lo, band_hi = median_band(q)
        for ie, eta in enumerate(etas):
            ax.fill_between(wd, band_lo[ie], band_hi[ie], color=colors[ie], alpha=0.15, lw=0)
            ax.loglog(wd, np.median(q[ie], axis=-1), "o-", ms=3, lw=1, color=colors[ie],
                      label=f"eta = {eta:.0e}")
        ax.axhline(1, color="0.5", lw=0.8)
        ax.set_title(regime, fontsize=10)
        ax.set_xlabel("window (days)")
    axes[0, 0].set_ylabel(f"measured / predicted std (median of {n} states)")
    axes[0, -1].legend(fontsize=8, loc="best")
    figstyle.finish(fig, "Stage 1: T_o1 coefficient; band = order-statistic interval of the "
                         "median (97.9% at 16 states)")
    path = os.path.join(RESULTS_DIR, f"{tag}_ratio.png")
    fig.savefig(path, dpi=figstyle.DPI); plt.close(fig)
    out.append(path)
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "smoke":
        kw = dict(tag="stage1_smoke", regimes=("passive",))
        run(n_states=2, etas=(1e-4, 1e-1), n_seeds=2, **kw)
    else:
        kw = {}
        if mode == "full":
            run()
    table(**kw)
    print(figure(**kw))
