"""
error_discovery.experiment_climatology
======================================

Climatology fidelity of the recovered correction.

The hybrid-model literature (e.g. He et al. 2025, Brajard et al. 2021) judges a
learned model-error correction not only by short-range forecast error but by
whether the *corrected model's long-run climatology* matches the truth: the
quartiles and means of every state variable over a multi-decade integration. This
appendix reports the same diagnostic for the differentiable correction.

We compare three long climatologies from the same attractor initial condition:

* truth    -- the perturbed model (g = T_{o,1}, eps = 0.1) that generated the data;
* base     -- the uncorrected model (the structural error left in place);
* corrected-- the discovery model integrated at the recovered coefficients c_hat.

For each of the 36 modes we take the mean and the first/median/third quartiles of
the marginal distribution, and report the per-mode bias relative to the truth's own
spread, std-normalised: b_base = |stat_base - stat_truth| / std_truth and likewise
for the corrected model. The corrected climatology should match the truth (the
recovered form is global, so it reproduces the attractor, not just the fitting
window), while the uncorrected base climatology carries the injected error's bias.

Results -> results/climatology_results.npz; figure -> results/fig_climatology.png.
"""

import os
import time

import numpy as np
import jax.numpy as jnp

from .config import ExperimentConfig, RESULTS_DIR, steps_to_days
from .models import TwinModels
from .discovery import rollout, recover
from .experiment import get_attractor_ic, build_segments
from .figstyle import DPI


def _quantile_stats(traj):
    """Per-mode (mean, Q1, median, Q3, std) of a (n_steps, ndim) climatology."""
    q1, q2, q3 = np.percentile(traj, [25, 50, 75], axis=0)
    return dict(mean=traj.mean(0), q1=q1, median=q2, q3=q3, std=traj.std(0))


def run(n_clim=150_000, recover_window=1280, verbose=True):
    t0 = time.time()
    cfg = ExperimentConfig()
    tw = TwinModels(cfg.eps, cfg.g_coeffs)
    dt = cfg.dt
    xa = get_attractor_ic(tw, cfg, verbose)

    # Recover the correction from a finite observation window (the canonical
    # window-of-discoverability recovery); c_hat, not c_true, drives the corrected
    # climatology so the test is honest.
    seg_ics, seg_trajs = build_segments(tw, cfg, xa)
    r = recover(tw, seg_ics[0], seg_trajs[0][:recover_window], dt,
                jnp.ones(tw.ndim), eta=0.0, n_iters=cfg.lm_iters)
    c_hat = jnp.asarray(r["c_hat"])
    if verbose:
        print(f"[recover] c_hat={np.asarray(c_hat)}  c_true={np.asarray(tw.c_true)}  "
              f"rel_err={r['rel_err']:.2e}")

    if verbose:
        print(f"[climo]  integrating 3 x {n_clim} steps ({steps_to_days(n_clim):.0f} days each)")
    truth = np.asarray(rollout(lambda x: tw.f_true(x), xa, dt, n_clim))
    base = np.asarray(rollout(lambda x: tw.f_base(x), xa, dt, n_clim))
    corr = np.asarray(rollout(lambda x: tw.f_model(x, c_hat), xa, dt, n_clim))

    st_t = _quantile_stats(truth)
    st_b = _quantile_stats(base)
    st_c = _quantile_stats(corr)
    sd = st_t["std"]                                  # truth's own per-mode spread

    def norm_bias(st, ref=st_t):                      # std-normalised |stat - ref|
        return {k: np.abs(st[k] - ref[k]) / np.where(sd > 0, sd, 1.0)
                for k in ("mean", "q1", "median", "q3")}

    nb_base = norm_bias(st_b)
    nb_corr = norm_bias(st_c)
    # irreducible sampling-noise floor at this record length: two independent halves
    # of the truth climatology differ by this much for purely statistical reasons.
    h = n_clim // 2
    nb_floor = norm_bias(_quantile_stats(truth[:h]), _quantile_stats(truth[h:]))

    def rms(nb):
        return float(np.sqrt((nb["mean"] ** 2).mean()))
    rms_base, rms_corr, rms_floor = rms(nb_base), rms(nb_corr), rms(nb_floor)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, "climatology_results.npz"),
             n_clim=n_clim, n_clim_days=steps_to_days(n_clim),
             c_hat=np.asarray(c_hat), c_true=np.asarray(tw.c_true), rel_err=r["rel_err"],
             truth_mean=st_t["mean"], truth_std=st_t["std"],
             truth_q1=st_t["q1"], truth_q3=st_t["q3"],
             base_mean=st_b["mean"], base_q1=st_b["q1"], base_q3=st_b["q3"],
             corr_mean=st_c["mean"], corr_q1=st_c["q1"], corr_q3=st_c["q3"],
             nb_base_mean=nb_base["mean"], nb_corr_mean=nb_corr["mean"],
             nb_floor_mean=nb_floor["mean"],
             nb_base_q1=nb_base["q1"], nb_corr_q1=nb_corr["q1"],
             nb_base_q3=nb_base["q3"], nb_corr_q3=nb_corr["q3"],
             rms_base=rms_base, rms_corr=rms_corr, rms_floor=rms_floor)
    if verbose:
        print(f"[done]  RMS std-normalised mean-bias over 36 modes:  base={rms_base:.4f}  "
              f"corrected={rms_corr:.4f}  sampling-noise floor={rms_floor:.4f}  "
              f"({time.time() - t0:.0f}s)")
    return dict(rms_base=rms_base, rms_corr=rms_corr, rms_floor=rms_floor)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(os.path.join(RESULTS_DIR, "climatology_results.npz"))
    ndim = len(d["truth_mean"])
    x = np.arange(ndim)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # (a) per-mode std-normalised mean bias (log scale): base and corrected both sit
    # far below the run-to-run sampling spread of the climatology at this record.
    floor = float(np.median(d["nb_floor_mean"])) if "nb_floor_mean" in d else None
    ax[0].bar(x - 0.2, np.maximum(d["nb_base_mean"], 1e-4), 0.4, color="C0", label="uncorrected base")
    ax[0].bar(x + 0.2, np.maximum(d["nb_corr_mean"], 1e-4), 0.4, color="C2", label="corrected (recovered)")
    if floor is not None:
        ax[0].axhline(floor, ls="--", color="grey", lw=1.2,
                      label=f"sampling spread (median {floor:.2f})")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("state mode")
    ax[0].set_ylabel(r"$|\mathrm{mean}-\mathrm{truth}|/\mathrm{std}_{\mathrm{truth}}$")
    ax[0].set_title("(a)", loc="left", fontsize=10)
    # white-framed so the legend (which sits on the sampling-spread dashed line) masks it
    ax[0].legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.9,
                 edgecolor="none", facecolor="white")

    # (b) climatology quartile box (truth vs corrected) for a few representative modes,
    # with the base mean marked, showing the corrected model tracks truth's spread.
    modes = [0, int(d["truth_mean"].shape[0]) - 8, 10, 28]   # psi_a1, psi_o1, theta_a1, T_o1
    labels = [r"$\psi_{a,1}$", r"$\psi_{o,1}$", r"$\theta_{a,1}$", r"$T_{o,1}$"]
    xb = np.arange(len(modes))
    for off, key, color, lab in ((-0.22, "truth", "k", "truth"),
                                 (0.22, "corr", "C2", "corrected")):
        lo = d[f"{key if key!='corr' else 'corr'}_q1"][modes]
        hi = d[f"{key if key!='corr' else 'corr'}_q3"][modes]
        md = d[f"{key if key!='corr' else 'corr'}_mean"][modes]
        ax[1].vlines(xb + off, lo, hi, color=color, lw=6, alpha=0.5,
                     label=f"{lab} (Q1–Q3)")
        ax[1].plot(xb + off, md, "_", color=color, ms=12, mew=2)
    ax[1].plot(xb, d["base_mean"][modes], "x", color="C0", ms=8, label="base mean")
    ax[1].set_xticks(xb); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("climatological value")
    ax[1].set_title("(b)", loc="left", fontsize=10)
    ax[1].legend(frameon=False, fontsize=7, ncol=1)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_climatology.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    run()
    print(figure())
