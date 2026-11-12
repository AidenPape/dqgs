"""
error_discovery.experiments.lyapunov
====================================

Lyapunov exponents we can quote: the full spectrum, with a convergence curve and a
blocking error bar for each exponent, checked against an exact identity, against qgs's own
estimator, and against published values.

The earlier estimate, ``discovery.lyapunov_time`` as called by ``why.lyapunov_ics``, is the
leading exponent alone, by finite differences, with the spread over five initial conditions
standing in for the error bar. It is left untouched, and nothing that uses it is re-pointed
here.

Three compute steps, each writing its own ``.npz``:

* ``run``: per named configuration (``qgs_jax.spec.PARAM_SETS``), ``n_ic`` disjoint records
  of ``years`` each, back to back along one trajectory from the spun-up state. Record k
  starts where record k-1 ended, with a fresh random frame. Record 0 is the headline. Its
  per-block log-stretch array is saved whole, and every headline number is a post-hoc
  reduction of it: running estimates on a geometric ladder, the blocking error bar, the
  transient sweep, Kolmogorov entropy, Kaplan-Yorke dimension, and counts in Tondeur et
  al.'s five bins. The other records give the spread over initial conditions. The
  divergence-sum identity is checked on record 0, to ``DIVERGENCE_TOL`` because the
  exponents are those of the RK4 map. The leading exponent is also reported at
  the old estimator's settings (blocks of 200 steps, 600 of them, first 120 dropped, 5
  initial conditions), so a change in the headline number can be attributed to the
  estimator rather than to the configuration.
* ``block_sensitivity``: the spectrum of "default" at renormalisation intervals from 10 to
  500 steps, from the same state and the same starting frame.
* ``qgs_check``: qgs's ``LyapunovsEstimator`` (numba, its own analytic Jacobian, RK4
  tangent linear model, QR every step) against ours on the same trajectory. Ours starts
  from qgs's own frame at the end of its pre-time, so in exact arithmetic the two agree
  block by block, and any difference is a method difference.

Nothing is fitted anywhere.

Tondeur et al. 2020 Table 3 is the only validation target with tabulated numbers. De Cruz
et al. 2018 gives figures only, and at 56 variables. ``qgs_jax.spec.PARAM_SETS`` records
how each published configuration was read.

Each step can be run on its own, and a configuration per process::

    python -c "from error_discovery.experiments import lyapunov as L; L.run(configs=('default',))"
    python -m error_discovery.run_all --lyapunov --list-params
"""

import json
import os
import time

import numpy as np
import jax.numpy as jnp

from .. import figstyle
from ..config import (RESULTS_DIR, IC_DIR, DT, NDIM_PER_DAY, STEPS_PER_DAY, DAYS_PER_YEAR,
                      IC_SEED, IC_SCALE, ExperimentConfig, ic_path)
from ..discovery import (lyapunov_spectrum, running_exponents, blocking_error,
                         kolmogorov_entropy, kaplan_yorke_dim)

CONFIGS = ("default", "tondeur2020_36wk", "tondeur2020_36st", "decruz2018_nodissip")
LABELS = {"default": "default (36 var.)",
          "tondeur2020_36wk": "Tondeur 36wk",
          "tondeur2020_36st": "Tondeur 36st",
          "decruz2018_nodissip": "De Cruz nodissip (56 var.)"}

# Discard prefixes for the transient sweep, in model-years.
TRANSIENT_YEARS = (0, 1, 3, 10, 30, 100)
N_LADDER = 40                       # points on the running-estimate ladder, 1 yr to the end

# Tondeur et al. 2020, Table 3: five exponent ranges in per day, and what they report.
TONDEUR_BINS = ((1e-2, 1.0), (1e-5, 1e-2), (-1e-5, 1e-5), (-1e-2, -1e-5), (-1.0, -1e-2))
# Published values, per day where they have units. Tondeur et al. 2020 Table 3 is tabulated.
# De Cruz et al. 2018 is not: its atm 2x4 / oc 2x4 "nodissip" values were read off the
# figures from marker positions on a 2400-px render, lambda_1 from Fig. 10 (one pixel =
# 0.0005 /day), h_KS from Fig. 17 (0.008 /day), KY-dim from Fig. 16's D_KY/N times 56
# (Fig. 15 reads 26.9).
PUBLISHED = {
    "tondeur2020_36wk": dict(ke=0.498, ky=25.06, bins=(3, 3, 1, 13, 16)),
    "tondeur2020_36st": dict(ke=0.139, ky=20.29, bins=(2, 2, 1, 11, 20)),
    "decruz2018_nodissip": dict(lam1=0.215, ke=0.71, ky=26.5),
}

# Pass bound for the divergence-sum identity, relative. The exponents are those of the RK4
# map, whose volume contraction departs from trace(Df) at high order in dt: up to ~1e-4
# relative at dt = 0.1 on the stretches measured, and collapsing as dt shrinks
# (tests/test_lyapunov.py). A broken QR or a sign error is an O(1) miss.
DIVERGENCE_TOL = 1e-3

# The old estimator's settings (why.lyapunov_ics): blocks of 200 steps, 600 blocks, the
# first 120 dropped.
OLD_BLOCK, OLD_N_BLOCKS, OLD_DROP = 200, 600, 120


def _path(name, tag):
    return os.path.join(RESULTS_DIR, f"lyapunov_{name}_{tag}.npz")


def _years_to_steps(years):
    return years * DAYS_PER_YEAR * STEPS_PER_DAY


def _model(name):
    """Tendency tensor, tendency and closed-form Jacobian of a named configuration."""
    from qgs_jax.spec import extract_spec, build_reference_params
    from qgs_jax import model as jm
    T = jnp.asarray(extract_spec(build_reference_params(name)).dense_tensor)
    return T, (lambda x: jm.tendencies(T, x)), (lambda x: jm.jacobian(T, x))


def _spun_up_ic(name, T):
    """The spun-up attractor state of a named configuration.

    "default" is the project's cached, provenance-checked IC. The others get the same recipe,
    800k steps from ``IC_SEED`` and ``IC_SCALE``, cached beside it under their own name, so
    ``IC_CACHE`` is never touched. File names follow ``config.ic_path``.
    """
    cfg = ExperimentConfig()
    if name == "default":
        from ..attractor import get_attractor_ic
        from ..formfit import twin
        return np.asarray(get_attractor_ic(twin(), cfg, verbose=False))
    ndim = T.shape[0] - 1
    path = ic_path(name, cfg.spinup_steps)
    if os.path.exists(path):
        d = np.load(path)
        if (int(d["ndim"]) == ndim and int(d["spinup_steps"]) == cfg.spinup_steps
                and int(d["ic_seed"]) == IC_SEED and float(d["dt"]) == DT):
            return d["x"]
    from qgs_jax import model as jm
    t = time.time()
    x0 = jnp.asarray(np.random.default_rng(IC_SEED).standard_normal(ndim) * IC_SCALE)
    x = np.asarray(jm.integrate_final(T, x0, DT, cfg.spinup_steps))
    os.makedirs(IC_DIR, exist_ok=True)
    np.savez(path, x=x, ndim=ndim, spinup_steps=cfg.spinup_steps, ic_seed=IC_SEED, dt=DT)
    print(f"[ic]    {name}: spun up {cfg.spinup_steps} steps ({time.time() - t:.0f}s)")
    return x


def _old_settings_lam1(logs1, block):
    """Leading exponent, per nondim, over the old estimator's window of one record."""
    per = OLD_BLOCK // block
    if OLD_BLOCK % block or logs1.size < OLD_N_BLOCKS * per:
        return np.nan
    return logs1[OLD_DROP * per:OLD_N_BLOCKS * per].sum() / (
        (OLD_N_BLOCKS - OLD_DROP) * OLD_BLOCK * DT)


def _bin_counts(spec_day):
    """Counts in Tondeur's five ranges; the top range is closed, the rest half-open."""
    counts = [int(np.sum((spec_day >= lo) & (spec_day < hi))) for lo, hi in TONDEUR_BINS]
    counts[0] += int(np.sum(spec_day == TONDEUR_BINS[0][1]))
    return np.array(counts)


# --------------------------------------------------------------------------- #
def run(configs=CONFIGS, years=614.0, n_ic=5, block=100, tag="main"):
    """Headline spectrum and its reductions, per configuration. Writes one ``.npz`` each."""
    return {name: _run_config(name, years, n_ic, block, tag) for name in configs}


def _run_config(name, years, n_ic, block, tag):
    from qgs_jax.spec import PARAM_SETS
    T, f, jac = _model(name)
    t_block = block * DT                              # nondim time per block
    yr_per_block = block / STEPS_PER_DAY / DAYS_PER_YEAR
    n_blocks = int(round(_years_to_steps(years) / block))
    ladder = np.unique(np.round(np.geomspace(1.0, years, N_LADDER) / yr_per_block).astype(int))
    ladder = ladder[(ladder >= 1) & (ladder <= n_blocks)]

    x = _spun_up_ic(name, T)
    x_starts, spectra, running, old_lam1 = [], [], [], []
    for k in range(n_ic):
        x_starts.append(x)
        t = time.time()
        logs, x = lyapunov_spectrum(f, x, DT, block=block, n_blocks=n_blocks, jac=jac, seed=k)
        if k == 0:
            logs0 = logs
        spectra.append(logs.mean(axis=0) / t_block)
        running.append(running_exponents(logs, t_block)[ladder - 1])
        old_lam1.append(_old_settings_lam1(logs[:, 0], block))
        print(f"[lyap]  {name} record {k + 1}/{n_ic}: {years:g} yr in {time.time() - t:.0f}s, "
              f"lambda_1 = {spectra[-1][0] * NDIM_PER_DAY:+.4e} /day", flush=True)

    spec_nd = spectra[0]
    spec_day = spec_nd * NDIM_PER_DAY
    L, se = blocking_error(logs0, t_block)
    se_day = se * NDIM_PER_DAY

    transient = []
    for y in TRANSIENT_YEARS:
        d = int(round(y / yr_per_block))
        transient.append(logs0[d:].mean(axis=0) / t_block * NDIM_PER_DAY
                         if d < n_blocks else np.full(spec_nd.size, np.nan))

    traces = np.array([float(jnp.trace(jac(jnp.asarray(xs)))) for xs in x_starts])
    div_rel = abs(spec_nd.sum() - traces[0]) / abs(traces[0])

    spectra_day = np.array(spectra) * NDIM_PER_DAY
    out = dict(
        name=name, params=json.dumps(PARAM_SETS[name]), ndim=spec_nd.size, dt=DT, block=block,
        years=years, n_ic=n_ic, n_blocks=n_blocks,
        # record 0 whole, float32 to halve the file; every number below is from float64
        logs0=logs0.astype(np.float32),
        spectrum_nd=spec_nd, spectrum_day=spec_day,
        blocking_L=L, blocking_se_day=se_day, se_day=se_day[-1],
        ke_day=kolmogorov_entropy(spec_day), ky_dim=kaplan_yorke_dim(spec_day),
        bins=_bin_counts(spec_day), n_outside_bins=int(np.sum(np.abs(spec_day) > 1.0)),
        lam1_day=spec_day[0], tau_days=1.0 / spec_day[0] if spec_day[0] > 0 else np.inf,
        transient_years=np.array(TRANSIENT_YEARS, dtype=float),
        transient_spectra_day=np.array(transient),
        trace_jacobian=traces, sum_spectrum_nd=spec_nd.sum(), divergence_rel_err=div_rel,
        divergence_pass=div_rel < DIVERGENCE_TOL,
        spectra_ic_day=spectra_day,
        ke_ic_day=np.array([kolmogorov_entropy(s) for s in spectra_day]),
        ky_ic=np.array([kaplan_yorke_dim(s) for s in spectra_day]),
        ladder_years=ladder * yr_per_block, running_day=np.array(running) * NDIM_PER_DAY,
        old_settings_lam1_nd=np.array(old_lam1), x_starts=np.array(x_starts),
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(_path(name, tag), **out)

    lam_ic = spectra_day[:, 0]
    print(f"[lyap]  {name}: lambda_1 = {spec_day[0]:+.4e} +/- {se_day[-1, 0]:.1e} /day "
          f"(tau {out['tau_days']:.1f} d), KE {out['ke_day']:.4f} /day, "
          f"KY-dim {out['ky_dim']:.2f}, bins {out['bins'].tolist()}")
    print(f"[lyap]  {name}: sum of exponents {spec_nd.sum():+.10e} vs trace(Df) "
          f"{traces[0]:+.10e} /nondim, rel. diff {div_rel:.1e} "
          f"({'PASS' if div_rel < DIVERGENCE_TOL else 'FAIL'} at {DIVERGENCE_TOL:g}); "
          f"trace spread over record starts {np.ptp(traces):.1e}")
    print(f"[lyap]  {name}: lambda_1 over {n_ic} records {lam_ic.mean():+.4e} "
          f"(std {lam_ic.std():.1e}) /day; at old settings {np.nanmean(old_lam1):+.3e} "
          f"(std {np.nanstd(old_lam1):.1e}) /nondim")
    return out


# --------------------------------------------------------------------------- #
def block_sensitivity(blocks=(10, 50, 100, 200, 500), block_years=60.0, tag="main"):
    """Spectrum of "default" at several renormalisation intervals, same state and frame.

    In exact arithmetic the interval does not change the exponents, so any spread here is
    what renormalising less often costs in floating point. Every interval covers the same
    steps: the record is rounded to a multiple of all of them.
    """
    T, f, jac = _model("default")
    x0 = _spun_up_ic("default", T)
    unit = int(np.lcm.reduce(np.asarray(blocks)))
    n_steps = unit * max(1, int(round(_years_to_steps(block_years) / unit)))
    spectra = []
    for b in blocks:
        t = time.time()
        logs, _ = lyapunov_spectrum(f, x0, DT, block=b, n_blocks=n_steps // b, jac=jac, seed=0)
        spectra.append(logs.mean(axis=0) / (b * DT) * NDIM_PER_DAY)
        print(f"[block] b = {b:4d}: lambda_1 {spectra[-1][0]:+.6e}, lambda_36 "
              f"{spectra[-1][-1]:+.6e} /day ({time.time() - t:.0f}s)", flush=True)
    spectra = np.array(spectra)
    out = dict(blocks=np.array(blocks), block_years=block_years, spectra_day=spectra,
               max_abs_diff_from_100=np.abs(spectra - spectra[list(blocks).index(100)]).max(axis=1)
               if 100 in blocks else np.full(len(blocks), np.nan))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, f"lyapunov_block_{tag}.npz"), **out)
    return out


# --------------------------------------------------------------------------- #
def qgs_check(qgs_pre_years=1.0, qgs_years=2.0, tag="main"):
    """qgs's ``LyapunovsEstimator`` against ours, on the same trajectory and frame.

    qgs converges its frame over ``qgs_pre_years``, then records the one-step local
    exponents for ``qgs_years`` (every step, so memory is about 11 kB per step). Ours then
    starts from qgs's state and frame at the end of the pre-time and runs in blocks of 100
    steps. qgs records the step ending at each stored time and never the final step, so the
    comparison covers every full block except the last.
    """
    from qgs.functions.tendencies import create_tendencies
    from qgs.toolbox.lyapunov import LyapunovsEstimator
    from qgs_jax.spec import build_reference_params

    block = 100
    T, f, jac = _model("default")
    x0 = _spun_up_ic("default", T)
    n_pre = int(round(_years_to_steps(qgs_pre_years)))
    nb = int(round(_years_to_steps(qgs_years) / block))
    n_rec = nb * block

    t = time.time()
    f_np, Df_np = create_tendencies(build_reference_params("default"))
    est = LyapunovsEstimator(num_threads=1)
    est.set_func(f_np, Df_np)
    est.compute_lyapunovs(0.0, n_pre * DT, (n_pre + n_rec) * DT, DT, DT, np.asarray(x0),
                          write_steps=1)
    tt, traj, exps, vecs = est.get_lyapunovs()
    est.terminate()
    # qgs builds its time grid with np.arange; a stray sub-step would corrupt every mean.
    assert tt.size == n_rec + 1 and np.allclose(np.diff(tt), DT), "qgs time grid misaligned"
    t_qgs = time.time() - t

    t = time.time()
    logs, _ = lyapunov_spectrum(f, traj[:, 0], DT, block=block, n_blocks=nb, jac=jac,
                                Q0=vecs[:, :, 0])
    t_ours = time.time() - t

    # qgs record i (i >= 1) is the local exponent of the step ending at stored time i.
    qgs_logs = exps[:, 1:1 + block * (nb - 1)].reshape(-1, nb - 1, block).sum(axis=2).T * DT
    ours_logs = logs[:nb - 1]
    t_win = (nb - 1) * block * DT
    ours_day = ours_logs.sum(axis=0) / t_win * NDIM_PER_DAY
    qgs_day = qgs_logs.sum(axis=0) / t_win * NDIM_PER_DAY
    out = dict(pre_years=qgs_pre_years, years=qgs_years, n_blocks_compared=nb - 1,
               ours_day=ours_day, qgs_day=qgs_day, diff_day=ours_day - qgs_day,
               block_max_abs_diff=np.abs(ours_logs - qgs_logs).max(axis=0),
               seconds_qgs=t_qgs, seconds_ours=t_ours)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, f"lyapunov_qgs_{tag}.npz"), **out)
    print(f"[qgs]   {nb - 1} blocks compared: max |ours - qgs| {np.abs(out['diff_day']).max():.2e} "
          f"/day over the window, {out['block_max_abs_diff'].max():.2e} per block log-stretch "
          f"(qgs {t_qgs:.0f}s, ours {t_ours:.0f}s)")
    return out


# --------------------------------------------------------------------------- #
def _load_runs(tag):
    return {n: np.load(_path(n, tag)) for n in CONFIGS if os.path.exists(_path(n, tag))}


def figure(tag="main"):
    """Convergence, spectrum, and the qgs cross-check; then the summary table."""
    plt = figstyle.pyplot()
    runs = _load_runs(tag)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # (a) running estimate of lambda_1; band = range over the disjoint records
    for i, (n, d) in enumerate(runs.items()):
        yrs, r1 = d["ladder_years"], d["running_day"][:, :, 0]
        a.fill_between(yrs, r1.min(axis=0), r1.max(axis=0), color=f"C{i}", alpha=0.18, lw=0)
        a.plot(yrs, r1[0], color=f"C{i}", lw=1.4, label=LABELS[n])
    a.set_xscale("log")
    a.set_xlabel("accumulation length (model-years)")
    a.set_ylabel(r"running estimate of $\lambda_1$ (day$^{-1}$)")
    n_ic = int(next(iter(runs.values()))["n_ic"]) if runs else 0
    a.set_title(f"(a) convergence; band = range over {n_ic} disjoint records",
                fontsize=10, loc="left")
    a.legend(fontsize=8, frameon=False)

    # (b) spectrum with +/- 1 blocking standard error
    for i, (n, d) in enumerate(runs.items()):
        idx = np.arange(1, int(d["ndim"]) + 1)
        b.errorbar(idx, d["spectrum_day"], yerr=d["se_day"], fmt="o", ms=3, color=f"C{i}",
                   elinewidth=0.8, capsize=1.5, label=LABELS[n])
    b.axhline(0.0, color="0.3", lw=0.8)
    b.set_yscale("symlog", linthresh=1e-4)
    b.set_xlabel("exponent index")
    b.set_ylabel(r"$\lambda_i$ (day$^{-1}$)")
    b.set_title(r"(b) spectrum, $\pm$1 blocking standard error", fontsize=10, loc="left")

    # (c) ours minus qgs on the same trajectory and frame, against our error bar
    qpath = os.path.join(RESULTS_DIR, f"lyapunov_qgs_{tag}.npz")
    if os.path.exists(qpath) and "default" in runs:
        q, se = np.load(qpath), runs["default"]["se_day"]
        idx = np.arange(1, se.size + 1)
        c.fill_between(idx, -se, se, step="mid", color="0.85", lw=0,
                       label=r"$\pm$1 blocking s.e. of our headline estimate")
        c.plot(idx, q["diff_day"], "o", ms=3, color="C0",
               label=f"ours $-$ qgs, {float(q['years']):g}-yr window")
        c.axhline(0.0, color="0.3", lw=0.8)
        c.text(0.02, 0.03, f"max |ours $-$ qgs| = {np.abs(q['diff_day']).max():.1e} day$^{{-1}}$",
               transform=c.transAxes, fontsize=8.5, bbox=figstyle.LBL_BBOX)
        c.legend(fontsize=8, frameon=False, loc="upper right")
    c.set_xlabel("exponent index")
    c.set_ylabel(r"$\lambda_i$, ours $-$ qgs (day$^{-1}$)")
    c.set_title("(c) cross-check against qgs, our configuration", fontsize=10, loc="left")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"fig_lyapunov_{tag}.png")
    fig.savefig(out, dpi=figstyle.DPI, bbox_inches="tight")
    plt.close(fig)
    return [out, table_figure(tag)]


def table_figure(tag="main"):
    """Headline numbers per configuration, with the published value beside each where one
    exists. Cells without a published counterpart are left blank."""
    plt = figstyle.pyplot()
    runs = _load_runs(tag)
    cols = [r"$\lambda_1$ (day$^{-1}$)", r"published $\lambda_1$", "Lyapunov time (d)",
            "KE (day$^{-1}$)", "published KE", "KY-dim", "published KY-dim",
            r"# $\lambda \geq 10^{-2}$/day", "published #"]
    rows, cells = [], []
    for n, d in runs.items():
        pub = PUBLISHED.get(n, {})
        rows.append(LABELS[n])
        cells.append([f"{float(d['lam1_day']):.3e} $\\pm$ {float(d['se_day'][0]):.0e}",
                      f"{pub['lam1']:.3f}" if "lam1" in pub else "",
                      f"{float(d['tau_days']):.1f}",
                      f"{float(d['ke_day']):.3f}", f"{pub['ke']:.3f}" if "ke" in pub else "",
                      f"{float(d['ky_dim']):.2f}", f"{pub['ky']:.2f}" if "ky" in pub else "",
                      f"{int(d['bins'][0])}", f"{pub['bins'][0]}" if "bins" in pub else ""])
    fig, ax = plt.subplots(figsize=(15.0, 0.5 + 0.42 * max(len(rows), 1)))
    ax.axis("off")
    if rows:
        tbl = ax.table(cellText=cells, rowLabels=rows, colLabels=cols, loc="center",
                       cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.35)
    ax.set_title("Lyapunov diagnostics. Published: Tondeur et al. (2020) Table 3; De Cruz et "
                 "al. (2018) Figs. 10, 16, 17, read off the figures", fontsize=10)
    out = os.path.join(RESULTS_DIR, f"fig_lyapunov_table_{tag}.png")
    fig.savefig(out, dpi=figstyle.DPI, bbox_inches="tight")
    plt.close(fig)
    return out
