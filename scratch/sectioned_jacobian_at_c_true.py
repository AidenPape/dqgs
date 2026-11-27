"""Conditioning of the sectioned residual Jacobian at c_true (``plans/sectioned-conditioning.md``).

``driver_uncertainty``'s quantities -- column norms, standard errors, variance inflation factors,
alignment with the worst-determined direction, singular values -- for the 89.7-d window restarted
every tau steps instead of a single window of length tau. No fit: J is evaluated at c_true.

tau is the section length in model steps (``TAUS``: 40 to 4000, 0.45 to 44.9 d), plus 8000, the
unsectioned window, computed with ``residual_jacobian`` exactly as ``driver_uncertainty`` does.
Section 0 starts at the state, section k at the true state ``Y_true[k*tau - 1]``, as ``recover``
builds them. States: the 60 ``both`` states of ``driver_uncertainty_both_n60.npz`` (the first 16
are the sectioned fits' states); all five drivers. Whitened by the high-frequency std.
Jacobians are eager ``jax.jacfwd``, as ``residual_jacobian``. Writes one file,
``results/data/sectioned_jacobian_at_c_true_both_n60_section_starts_true_state.npz``.

``check`` prints three comparisons: tau = 8000 against ``driver_uncertainty_both_n60.npz``'s
8000-step window; the sectioned values against the J at ``c_hat`` saved by the noiseless
sectioned fits (``sectioned_loss_both_<driver>.npz``), reached and failed fits apart; and the
cost at c_true per tau.

``figure`` redraws three single-window figures against tau instead of window length, each with its
original's panels, quantities and spread (plan, figure decisions 1-4): (A) median cond(J_w), 5
drivers, as ``fig_driver_conditioning_recovery_both.png`` panel (a); (B) the driver's own
coefficient's median se_unit / |c_k|, VIF and align, as ``fig_driver_uncertainty_both_n60.png``;
(C) the nested bounds, as ``fig_driver_uncertainty_bounds_both_n60.png``. B and C: the 4 single
drivers. The last point, 89.7 d, is the unsectioned window. Writes the values drawn and two
checks (bound nesting; 89.7-d point = the n60 file's) to
``results/logs/<stem>_figure_tables.log``.

    python -u scratch/sectioned_jacobian_at_c_true.py all     # 60 states x 5 drivers, then check
    python -u scratch/sectioned_jacobian_at_c_true.py smoke   # both #0, #1; check; time per J
    python -u scratch/sectioned_jacobian_at_c_true.py figure  # figures A, B, C and tables
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
from error_discovery.discovery import (residual_jacobian, make_residual, make_sectioned_residual,
                                       obs_indices, coefficient_uncertainty)
from error_discovery.experiments.driver_uncertainty import own
from error_discovery.experiments.driver_window import DRIVERS
from error_discovery.figstyle import DPI, LBL_BBOX
from error_discovery.models import TwinModels, obs_mask
from error_discovery.sampling import load_snapshots, noise_scale, sampling_line

MODE = "both"
N_STEPS = 8000
TAUS = (40, 80, 125, 160, 250, 320, 500, 1000, 2000, 4000)    # decision 3
ALL_TAUS = TAUS + (N_STEPS,)                                   # 8000 = unsectioned
PER_COEF = ("col_norm", "se_unit", "vif", "align", "sigma")
SCALAR = ("cond_J", "smin", "smax", "jrank")
REACHED = 1e-3                    # rel_err below this = reached c_true (0a / 0b threshold)
STEM = f"sectioned_jacobian_at_c_true_{MODE}_n60_section_starts_true_state"


def _reference():
    return np.load(os.path.join(DATA_DIR, f"driver_uncertainty_{MODE}_n60.npz"))


def jacobian_at_c_true(tw, x0, Y, dt, mask, scale, tau):
    """J at c_true and the cost there. tau = len(Y): ``residual_jacobian``, unsectioned."""
    oidx = obs_indices(mask)
    wchan = 1.0 / np.asarray(scale)[oidx]
    c = jnp.asarray(tw.c_true)
    if tau == len(Y):
        J = residual_jacobian(tw, x0, Y, dt, mask, whiten=True, scale=scale)
        resid = make_residual(tw, x0, Y, dt, oidx, wchan=wchan)
    else:
        starts = np.vstack([np.asarray(x0)[None], np.asarray(Y)[tau - 1:-1:tau]])  # as recover
        resid = make_sectioned_residual(tw, starts, Y, dt, oidx, wchan=wchan)
        J = np.asarray(jax.jacfwd(resid)(c))
    r = np.asarray(resid(c))
    return J, 0.5 * float(r @ r)


def run(n_states=None, tag=STEM):
    """Every driver x tau x state; one file."""
    cfg = ExperimentConfig()
    ref = _reference()
    assert str(ref["sampling"]) == MODE and bool(ref["whiten"])
    assert str(ref["obs"]) == cfg.obs and float(ref["eps"]) == cfg.eps
    idx = ref["sample_snapshot_index"][:n_states]
    states = load_snapshots(idx)
    scale = noise_scale()
    names = tuple(DRIVERS)
    shape = (len(names), len(ALL_TAUS), len(idx))
    out = {k: np.full(shape + (4,), np.nan) for k in PER_COEF}
    out.update({k: np.full(shape, np.nan) for k in SCALAR + ("cost_true",)})
    c_true = np.zeros((len(names), 4))

    for i, name in enumerate(names):
        tw = TwinModels(cfg.eps, DRIVERS[name])
        mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
        c_true[i] = np.asarray(tw.c_true)
        ics, trajs = segments_from_states(tw, states, N_STEPS, cfg.dt)
        for k in range(len(idx)):
            t0 = time.time()
            for it, tau in enumerate(ALL_TAUS):
                J, cost = jacobian_at_c_true(tw, ics[k], trajs[k], cfg.dt, mask, scale, tau)
                u = coefficient_uncertainty(J)
                u["jrank"] = u.pop("rank")
                for key in PER_COEF + SCALAR:
                    out[key][i, it, k] = u[key]
                out["cost_true"][i, it, k] = cost
            print(f"[{name}] state {k:2d}  {len(ALL_TAUS)} Jacobians  {time.time() - t0:5.1f} s",
                  flush=True)

    path = os.path.join(DATA_DIR, f"{tag}.npz")
    np.savez(path, **out, drivers=np.asarray(names),
             g_coeffs=np.asarray([DRIVERS[n] for n in names]), c_true=c_true,
             tau_steps=np.asarray(ALL_TAUS),
             tau_days=np.asarray([steps_to_days(t) for t in ALL_TAUS]),
             n_sections=np.asarray([N_STEPS // t for t in ALL_TAUS]), n_steps=N_STEPS,
             eps=cfg.eps, obs=cfg.obs, dt=cfg.dt, whiten=True, sampling=MODE,
             evaluated_at="c_true, no fit",
             section_starts="true state: section 0 from x0, section k from Y_true[k*tau - 1]; "
                            "tau = 8000 unsectioned (residual_jacobian)",
             **{k: ref[k][:len(idx)] if ref[k].ndim else ref[k]
                for k in ref.files if k.startswith("sample_")})
    print(f"[done] wrote {path}", flush=True)
    return path


def _rel(new, old):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(np.asarray(new, float) / np.asarray(old, float) - 1)


def check(path):
    """Checks (1)-(3) of decision 6, printed."""
    d, ref = np.load(path), _reference()
    n = d["cond_J"].shape[2]
    assert np.array_equal(d["sample_snapshot_index"], ref["sample_snapshot_index"][:n])
    print("\nnon-finite entries:",
          {k: int((~np.isfinite(d[k])).sum()) for k in PER_COEF + SCALAR + ("cost_true",)})

    iw = list(ref["windows"]).index(N_STEPS)
    print(f"\n(1) tau = {N_STEPS} vs driver_uncertainty_{MODE}_n60.npz, window {N_STEPS}, "
          f"{n} states:")
    for name in ref["drivers"]:
        i, j = list(d["drivers"]).index(name), list(ref["drivers"]).index(name)
        for key in PER_COEF + SCALAR:
            new, old = d[key][i, -1], ref[key][j, iw, :n]
            print(f"  {name:8s} {key:8s} bit-identical {np.array_equal(new, old)!s:5s}  "
                  f"max rel diff {np.nanmax(_rel(new, old)):.1e}")

    print("\n(2) J at c_true vs J at c_hat of the noiseless sectioned fits, "
          "max |ratio - 1| (reached) | median, max (failed):")
    for i, name in enumerate(d["drivers"]):
        s = np.load(os.path.join(DATA_DIR, f"sectioned_loss_{MODE}_{name}.npz"))
        assert tuple(s["tau_steps"]) == TAUS
        m = min(n, s["cond_J"].shape[1])
        assert np.array_equal(s["sample_snapshot_index"][:m], d["sample_snapshot_index"][:m])
        ok = s["rel_err"][:, :m] < REACHED
        for key in ("cond_J", "col_norm", "se_unit", "vif", "align"):
            r = _rel(s[key][:, :m], d[key][i, :len(TAUS), :m])
            okk = np.broadcast_to(ok if r.ndim == 2 else ok[..., None], r.shape)
            fail = r[~okk]
            tail = (f"{np.median(fail):.1e}, {fail.max():.1e}  (n={int((~ok).sum())})"
                    if fail.size else "none failed")
            print(f"  {name:8s} {key:8s} {np.nanmax(r[okk]):.1e}  (n={int(ok.sum())}) | {tail}")

    print("\n(3) cost at c_true, max over drivers and states, per tau:")
    for it, tau in enumerate(d["tau_steps"]):
        print(f"  tau {int(tau):5d}: {np.max(d['cost_true'][:, it]):.2e}")


def _curves(d, i):
    """Every quantity the figures draw for driver ``i``, one value per x point; medians over
    states unless named min / max. The own-coefficient ones only for a single driver."""
    ct = d["c_true"][i]
    med = lambda a: np.nanmedian(a, axis=1)
    out = {"cond_J": med(d["cond_J"][i])}
    if np.count_nonzero(ct) == 1:
        ck = abs(float(own(ct, ct)))
        rse = own(d["se_unit"][i], ct, relative=True)
        al = own(d["align"][i], ct)
        out.update(
            se_rel=med(rse), se_rel_min=np.nanmin(rse, axis=1), se_rel_max=np.nanmax(rse, axis=1),
            vif=med(own(d["vif"][i], ct)),
            align=med(al), align_min=np.nanmin(al, axis=1), align_max=np.nanmax(al, axis=1),
            best=med(1.0 / d["smax"][i]) / ck, worst=med(1.0 / d["smin"][i]) / ck,
            known=med(1.0 / own(d["col_norm"][i], ct)) / ck,
            fitted=med(own(d["se_unit"][i], ct)) / ck,
            # per state, then median, as the original
            sqrt_vif=med(np.sqrt(own(d["vif"][i], ct))))
    return out


def _figure_checks(d, curves):
    """Bound nesting everywhere; every curve's 89.7-d point against the n60 file's median."""
    lo, hi = 1.0 / d["smax"][..., None], 1.0 / d["smin"][..., None]
    known, fitted = 1.0 / d["col_norm"], d["se_unit"]
    rtol = 1e-9                                         # as driver_uncertainty._nesting_holds
    bad = sum(int((~(a <= b * (1.0 + rtol))).sum())
              for a, b in ((lo, known), (known, fitted), (fitted, hi)))
    lines = [f"nesting 1/smax <= 1/||J e_k|| <= se_unit <= 1/smin, all {d['cond_J'].shape[0]} "
             f"drivers x {d['cond_J'].shape[1]} tau x {d['cond_J'].shape[2]} states x 4 "
             f"coefficients: {bad} violations"]
    ref = _reference()
    iw = list(ref["windows"]).index(N_STEPS)
    for j, name in enumerate(ref["drivers"]):
        old = _curves(ref, j)
        new = curves[list(d["drivers"]).index(name)]
        diff = [k for k in new if new[k][-1] != old[k][iw]]
        lines.append(f"89.7-d point vs n60 window {N_STEPS}, {name}: {len(new)} curves, "
                     f"{'all equal' if not diff else 'differ: ' + ', '.join(diff)}")
    return lines


def figure(path=None):
    """Figures A, B, C (module docstring) and the tables log. Returns the three paths."""
    plt = figstyle.pyplot()
    d = np.load(path or os.path.join(DATA_DIR, f"{STEM}.npz"))
    names = [str(n) for n in d["drivers"]]
    single = [i for i in range(len(names)) if np.count_nonzero(d["c_true"][i]) == 1]
    x = d["tau_days"]
    xlabel = "section length tau  [days]"
    line = (f"{sampling_line(d)}; {x[-1]:.1f}-d window restarted every tau "
            f"(last point unsectioned)")
    colors = plt.get_cmap("tab10").colors
    curves = [_curves(d, i) for i in range(len(names))]
    paths = []

    # (A) as fig_driver_conditioning_recovery panel (a)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for i, nm in enumerate(names):
        ax.plot(x, curves[i]["cond_J"], "o-", ms=4, color=colors[i], label=nm)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"cond$(J_w)=\sigma_{\max}/\sigma_{\min}$")
    ax.set_title("median conditioning vs section length, by error driver", fontsize=9.5,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, title="driver $g(x)$", title_fontsize=8)
    figstyle.finish(fig, line.replace("; ", "\n"))           # one line is wider than the panel
    paths.append(os.path.join(FIGURES_DIR, f"{STEM.replace('_section_starts', '_median_cond_J_section_starts')}.png"))
    fig.savefig(paths[-1], dpi=DPI); plt.close(fig)

    # (B) as fig_driver_uncertainty
    fig, (ax, axb, axc) = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for i in single:
        c = curves[i]
        ax.fill_between(x, c["se_rel_min"], c["se_rel_max"], color=colors[i], alpha=0.15, lw=0)
        ax.plot(x, c["se_rel"], "o-", ms=4, color=colors[i], label=names[i])
        axb.plot(x, c["vif"], "s-", ms=5, color=colors[i], label=names[i])
        if names[i] in ("theta_a1", "T_o1"):
            axc.fill_between(x, c["align_min"], c["align_max"], color=colors[i], alpha=0.18, lw=0)
        axc.plot(x, c["align"], "^-", ms=5, color=colors[i], label=names[i])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma_{\hat c_k}/(|c_k|\,\sigma_{\rm obs})$")
    ax.set_title("(a)  median uncertainty of the driver's own coefficient", fontsize=9.5,
                 loc="left")
    axb.axhline(1.0, ls=":", color="grey", lw=1)
    axb.text(x[0], 1.05, "VIF = 1", color="grey", fontsize=8, va="bottom", ha="left",
             bbox=LBL_BBOX)
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_ylabel(r"VIF$_k=\|J_{:,k}\|^2\,[(J^TJ)^{-1}]_{kk}$")
    axb.set_title("(b)  median variance inflation factor", fontsize=9.5, loc="left")
    axc.set_xscale("log")
    axc.set_ylim(0, 1.02)
    axc.set_ylabel(r"$|\langle v_{\min},e_k\rangle|$")
    axc.set_title("(c)  median overlap with the worst-determined\nsingular direction",
                  fontsize=9.5, loc="left")
    for a in (ax, axb, axc):
        a.set_xlabel(xlabel)
    figstyle.finish(fig, line)
    # below the panels: inside (a) it covers a band wherever it goes
    fig.subplots_adjust(bottom=0.2)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=len(single),
               frameon=False, fontsize=8, title="driver $g(x)$", title_fontsize=8)
    paths.append(os.path.join(FIGURES_DIR, f"{STEM.replace('_section_starts', '_se_unit_vif_align_section_starts')}.png"))
    fig.savefig(paths[-1], dpi=DPI); plt.close(fig)

    # (C) as fig_driver_uncertainty_bounds
    fig, axes = plt.subplots(1, len(single), figsize=(17.0, 4.5), sharey=True)
    for p, (i, ax) in enumerate(zip(single, axes)):
        c = curves[i]
        ax.fill_between(x, c["best"], c["worst"], color="0.85", alpha=0.55, lw=0)
        ax.plot(x, c["best"], color="0.5", lw=1)
        ax.plot(x, c["worst"], color="0.5", lw=1)
        ax.fill_between(x, c["known"], c["fitted"], color=colors[i], alpha=0.2, lw=0)
        ax.plot(x, c["known"], "s--", ms=4, lw=1.2, color=colors[i])
        ax.plot(x, c["fitted"], "o-", ms=4, color=colors[i])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(f"({chr(97 + p)})  {names[i]}", fontsize=10, loc="left")
        ax.text(0.96, 0.94, rf"$\sqrt{{\rm VIF}}={c['sqrt_vif'][-1]:.0f}$ at {x[-1]:.0f} d",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox=LBL_BBOX)
    axes[0].set_ylabel(r"$\sigma_{\hat c_k}/(|c_k|\,\sigma_{\rm obs})$")
    fig.legend(handles=[
        plt.Line2D([], [], color="0.5", lw=1,
                   label=r"$1/\sigma_{\max}$, $1/\sigma_{\min}$  (best, worst)"),
        plt.Line2D([], [], color="grey", ls="--", marker="s", ms=4,
                   label=r"$1/\|Je_k\|$  (form known)"),
        plt.Line2D([], [], color="grey", ls="-", marker="o", ms=4,
                   label=r"$\sigma_{\hat c_k}$  (all four fitted)")],
        frameon=False, fontsize=8, ncol=3, loc="lower center")
    fig.suptitle(f"median nested bounds on the driver's own coefficient\n{line}", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    paths.append(os.path.join(FIGURES_DIR, f"{STEM.replace('_section_starts', '_nested_bounds_section_starts')}.png"))
    fig.savefig(paths[-1], dpi=DPI); plt.close(fig)

    lines = [f"{line}; values drawn, per tau [steps]: " + " ".join(str(int(t)) for t in d["tau_steps"])]
    for i, nm in enumerate(names):
        for key, v in curves[i].items():
            lines.append(f"{nm:8s} {key:10s} " + " ".join(f"{val:.3e}" for val in v))
    lines += _figure_checks(d, curves)
    print("\n".join(lines))
    with open(os.path.join(RESULTS_DIR, "logs", f"{STEM}_figure_tables.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(paths))
    return paths


def smoke(n_states=2):
    """``both`` #0, #1, every driver and tau; checks; wall time per Jacobian."""
    t0 = time.time()
    path = run(n_states=n_states, tag=STEM + "_smoke")
    per_j = (time.time() - t0) / (len(DRIVERS) * len(ALL_TAUS) * n_states)
    check(path)
    print(f"\nwall time per Jacobian {per_j:.2f} s (truth runs and setup included)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "smoke":
        smoke()
    elif arg == "all":
        check(run())
    elif arg == "figure":
        figure()
    else:
        sys.exit(f"unknown argument {arg!r}")
