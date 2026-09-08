"""
error_discovery.experiments.driver_uncertainty
==============================================

How well is each driver's *own* coefficient determined, and what does not knowing the form
cost it?

``driver_window`` asked whether the window of discoverability depends on which driver the
error rides on and answered no: all five drivers bottom out at the same 21.5-day window with
cond(J_w) agreeing to 1.034x. That null is structural, not empirical. ``recover`` fits the
whole four-vector c whatever the truth is, so J's columns are the *library*; and
cond(J) = sigma_max/sigma_min is built from singular *values*, which cannot know which column
carries the truth. Any question of the form "is this driver harder than that one?" is
invisible to it by construction.

This study restricts attention to the coefficient that is actually nonzero. From one Jacobian
it reads three things per coefficient i, which answer three different questions:

    col_norm[i] = ||J[:, i]||                 information about c_i if the rest were known
    se_unit[i]  = sqrt([(J^T J)^-1]_ii)       its standard error when all four are fitted
    vif[i]      = (col_norm[i]*se_unit[i])^2  the variance inflation between the two

The first is the driver's own signal: under linear theory the restricted standard error is
sigma_obs/||J_k||, which is what "compute the Jacobian in terms of the true driver's
coefficient alone" means. The second is what a study here actually gets, because the form is
not known and all four coefficients are fitted together. Their ratio, the variance inflation
factor, is the price of that ignorance, per driver -- 1 when the driver's column is orthogonal
to the rest of the library, large when the library can imitate it.

``align[i] = |<v_min, e_i>|`` completes the picture: it says whether a badly determined
coefficient is badly determined because it fell into the worst-conditioned direction.

No fitting happens. The discovery model class contains the truth, so the noiseless residual is
exactly zero at c_true and J^T J there is the exact Gauss--Newton Hessian; the whole study is
one ``jax.jacfwd`` per (driver, window, segment). That is why it costs minutes where
``driver_window`` costs ~36 min, and why nothing here carries sampling noise.

Reading the scale: whitening divides each channel by its climatological std, and the assumed
observation-error std is eta * std(channel), so the whitened noise std is eta in every
channel. ``eta * se_unit[i]`` is therefore the predicted uncertainty of c_i in absolute units,
and ``eta * se_unit[i] / |c_true[i]|`` is on the same footing as the ``rel_err`` curves the
other studies plot. The figure reports the eta-free part; multiply by the eta you care about.

**Whether the predicted spread matches the measured spread of noisy fits is not tested here**
-- that check is still open. What this study measures is the linear-theory prediction itself.

**The no-fit shortcut is exact, re-measured 2026-08-27 on disjoint segments.** cond(J) from J
at c_true reproduces ``driver_window``'s cond(J_w) from J at the fitted c_hat to
max |ratio - 1| = 2.98e-6 across all four drivers and all thirteen windows (6.3e-10 for
theta_a1), with the same 57.4-day minimum and the same 1.027x driver spread. Forty LM iterations
per case bought nothing the Jacobian at the truth does not already have. The agreement is the
same to within a factor of 1.006 before and after the segment fix, which is the point: the
shortcut is a statement about the method, and it did not depend on which segments were used.

**The answer, measured 2026-08-27 on three disjoint segments: where cond(J_w) puts the drivers
within 1.027x, their own coefficients span 162x.** Relative standard error per unit sigma_obs at
the 89.7 d window, as a range over the three segments rather than a median, because the segments
disagree by 1.9-3.9x there:

    psi_a1  0.28-0.72      const  1.15-2.68
    theta_a1  16.2-62.2    T_o1   41.1-79.2

**Most of that is amplitude, not identifiability -- for two of the three.** ||J_k|| / ||J_const||
equals the driver's own rms along the trajectory to within 1-2% for theta_a1 (0.0233 against rms
0.0229) and T_o1 (0.0305 against 0.0308), so at the fixed eps = 0.1 used here a small-amplitude
driver is injecting a smaller error and its coefficient is correspondingly worse determined.
**psi_a1 breaks the rule and the segment fix is what exposed it**: its ratio is 0.709 against an
rms of 0.368, 93% off, and it swings 0.349-0.760 across the three disjoint segments where
theta_a1 and T_o1 hold to a few percent. One rms does not characterise psi_a1 -- the
streamfunction amplitude simply differs between segments that no longer share data.

Read the table as "how well is a 10% coefficient on *this* function determined", not as a
statement about which errors are discoverable. It is the same confound ``driver_mode`` records.
The same fix -- match eps to the driver's rms -- would settle it for theta_a1 and T_o1; for
psi_a1 there is no single rms to match to, so that fix is blocked rather than merely pending.

The one amplitude-matched pair is theta_a1 and T_o1 (rms within 1.35x), and there **the
standard errors overlap across segments -- 16.2-62.2 against 41.1-79.2 -- so no difference is
claimed.** The mechanism used to separate them without overlap; **on disjoint segments it no
longer does.** At 89.7 d T_o1's coefficient direction has 0.601-0.999 overlap with the
worst-determined singular direction and theta_a1's has 0.012-0.799 -- ranges that overlap over
0.601-0.799, so the geometric separation is now inside the segment scatter too.

**And the alignment swaps with window, which the pre-fix run did not show.** Median
|<v_min, e_k>| for theta_a1 against T_o1 runs 0.98-0.999 against 0.005-0.17 across 1.8-28.7 d,
then reverses to 0.09-0.14 against 0.99 from 57.4 d out; the crossover is near 43 d. The two
halves are not equally well supported. At the long end all three segments agree that T_o1 sits
nearest the null direction. At the short end segment 0 dissents -- it puts T_o1 there at *every*
window, against segments 1 and 2 which put theta_a1 there -- so that half is 2 against 1. Read
the long-window end as the result, the short-window end as unresolved, and panel (c) as two
regimes rather than one ordering.

**There is no U in the uncertainty.** The relative standard error falls essentially monotonically
out to 89.7 d for every driver -- theta_a1 has no rise at all, const and psi_a1 rise only at
7.2 d, T_o1 at 7.2 and 10.8 d -- while cond(J_w) turns up after 57.4 d. The U is in the *aliasing
cost*, not in the information: VIF bottoms out at 3.56e3 to 2.14e5, but at no shared window --
T_o1 at 5.4 d, const at 57.4 d, theta_a1 and psi_a1 not until 89.7 d -- and reaches 3.9e11 at the
0.45 d window. As a multiplier on the standard error rather than on its square, sqrt(VIF) is
60-463 at each driver's own VIF minimum and 60-718 at the 89.7 d window -- so not knowing the
form costs two to three orders of magnitude wherever the window is put. That the uncertainty
keeps falling past the conditioning optimum is consistent with ``driver_window``, whose noisy
recovery was still improving at the last window it tested.

**How much of this is readable.** Three segments, so only gaps wider than the segment-to-segment
spread mean anything: that spread is 1.9-3.9x at 89.7 d and 25-835x at the shortest window. The
alignment crossings in panel (c) are inside it at every window, the long-window end now included
(see the theta_a1/T_o1 overlap above), so panel (c) should be read as a median ordering and not
as a separation.

Two of those crossings may also be ill-posed rather than merely noisy. v_min is well separated
from the next singular direction at most windows (median sigma_3/sigma_min of 1.7-17x) but not
all: the minimum over drivers and segments falls to 1.017 at 57.4 d and 1.06 at 0.90 d, so in
those cases the "worst-determined direction" is nearly a two-dimensional subspace and which
vector the SVD returns from it is arbitrary. The pre-fix run's claim of at least 1.36x
separation everywhere does not survive disjoint segments.

The segments are at least genuinely disjoint since 2026-08-27, though still not independent --
consecutive segments sit 89.7 d apart on one trajectory and the ocean state stays correlated
across them.

**The four bounds are nested, measured 2026-08-27 on the disjoint-segment run.** At every
window and for every driver,

    1/sigma_max  <=  1/||J e_k||  <=  se_k  <=  1/sigma_min
    best possible   form known      fitted     worst possible

so cond(J_w) is not an independent fifth number: on a log axis it is the distance between the
outer two, and sqrt(VIF) is the distance between the middle two. The outer envelope is nearly
driver-independent above 1.8 d -- 1/sigma_max agrees across the four drivers to 1.07x and
1/sigma_min to 1.13x -- so the drivers differ only in where the middle two sit inside a common
frame.

What moves with window is the aliasing cost, and not the same way for everyone. sqrt(VIF) at
3.6 / 21.5 / 89.7 d runs 2166 / 1848 / 569 for const, 1309 / 1433 / 112 for theta_a1 and
1648 / 298 / 60 for psi_a1 -- but 459 / 259 / 718 for T_o1. **T_o1 is the only driver whose cost
of not knowing the form gets worse with a longer window**: it bottoms out at 202 at the 5.4 d
window and climbs monotonically to 718 by 89.7 d, a 3.6x rise, while the other three fall 4-27x
from their own short-window levels. So T_o1 is the cheapest of the four to fit at 21.5 d and the
most expensive at 89.7 d. A single-window comparison hides that entirely. ``bounds_table``
prints these; ``fig_driver_uncertainty_bounds_<tag>.png`` draws them.

Results -> ``results/driver_uncertainty_<tag>.npz``.
"""

import os
import time

import numpy as np

from .. import figstyle
from ..attractor import get_attractor_ic, build_segments
from ..config import ExperimentConfig, RESULTS_DIR, steps_to_days
from ..figstyle import DPI, LBL_BBOX
from ..models import TwinModels, obs_mask
from ..discovery import residual_jacobian, coefficient_uncertainty

# ``driver_window``'s single drivers, duplicated rather than imported -- studies do not import
# each other. Slots index features(x) = (1, theta_a1, T_o1, psi_a1). Its ``mixed`` driver is
# deliberately absent: every quantity here is the uncertainty of *one* coefficient, and a driver
# with three nonzero coefficients has a 3x3 covariance block instead, which no single number
# summarises. ``driver_window`` covers mixed for the cond and rel_err questions.
DRIVERS = {
    "const":    (1.0, 0.0, 0.0, 0.0),
    "theta_a1": (0.0, 1.0, 0.0, 0.0),
    "T_o1":     (0.0, 0.0, 1.0, 0.0),
    "psi_a1":   (0.0, 0.0, 0.0, 1.0),
}
ALL_DRIVERS = tuple(DRIVERS)


def own(quantity, c_true_row, relative=False):
    """Pick out the driver's own coefficient -- the single nonzero entry of ``c_true``.

    ``quantity`` is any per-coefficient array whose last axis is the four coefficients. With
    ``relative=True`` the value is divided by |c_true| for that coefficient, turning an absolute
    uncertainty into a fractional one.
    """
    k = int(np.flatnonzero(np.abs(np.asarray(c_true_row)) > 0)[0])
    q = np.asarray(quantity)[..., k]
    return q / abs(np.asarray(c_true_row)[k]) if relative else q


def run(tag="main", drivers=ALL_DRIVERS, n_ic=3, whiten=True, verbose=True):
    """Per-coefficient uncertainty of the Jacobian at c_true, once per driver and window.

    ``n_ic`` and the window set match ``driver_window`` so the curves overlay
    ``driver_window_main`` directly. There is no noise sweep and no LM budget: the Jacobian is
    evaluated at the truth, where the noiseless residual vanishes.

    Every driver shares the base model, so all four read the same cached attractor IC --
    ``get_attractor_ic`` keys on ndim, spin-up length, seed and dt, none of which vary with
    ``g_coeffs``. Nothing here regenerates that cache. The truth *trajectory* does vary with
    the driver, so segments are rebuilt per driver.
    """
    t0 = time.time()
    cfg = ExperimentConfig()
    cfg.n_ic = n_ic
    dt = cfg.dt
    windows = cfg.window_steps
    nD, nW = len(drivers), len(windows)

    per_coeff = (nD, nW, n_ic, 4)
    col_norm = np.full(per_coeff, np.nan)
    se_unit = np.full(per_coeff, np.nan)
    vif = np.full(per_coeff, np.nan)
    align = np.full(per_coeff, np.nan)
    sigma = np.full(per_coeff, np.nan)
    cond_J = np.full((nD, nW, n_ic), np.nan)
    smin = np.full((nD, nW, n_ic), np.nan)
    smax = np.full((nD, nW, n_ic), np.nan)
    jrank = np.full((nD, nW, n_ic), np.nan)
    c_trues = np.zeros((nD, 4))

    for idr, name in enumerate(drivers):
        tw = TwinModels(cfg.eps, DRIVERS[name])
        mask = obs_mask(tw.ndim, tw.spec.variables_range, cfg.obs)
        x_attr = get_attractor_ic(tw, cfg, verbose=False)
        seg_ics, seg_trajs = build_segments(tw, cfg, x_attr)
        c_trues[idr] = np.asarray(tw.c_true)
        td = time.time()

        for iw, w in enumerate(windows):
            for k in range(n_ic):
                J = residual_jacobian(tw, seg_ics[k], seg_trajs[k][:w], dt, mask,
                                      whiten=whiten)
                u = coefficient_uncertainty(J)
                col_norm[idr, iw, k] = u["col_norm"]
                se_unit[idr, iw, k] = u["se_unit"]
                vif[idr, iw, k] = u["vif"]
                align[idr, iw, k] = u["align"]
                sigma[idr, iw, k] = u["sigma"]
                cond_J[idr, iw, k] = u["cond_J"]
                smin[idr, iw, k] = u["smin"]
                smax[idr, iw, k] = u["smax"]
                jrank[idr, iw, k] = u["rank"]

        if verbose:
            rse = np.nanmedian(own(se_unit[idr], c_trues[idr], relative=True), axis=1)
            vf = np.nanmedian(own(vif[idr], c_trues[idr]), axis=1)
            best = int(np.argmin(rse))
            print(f"[{name:<8s}] best rel. s.e. {rse[best]:.3e}/sigma_obs at "
                  f"{steps_to_days(windows[best]):.1f} d   "
                  f"VIF there {vf[best]:.2e}   (VIF range {vf.min():.2e}-{vf.max():.2e})   "
                  f"[{time.time() - td:.0f}s]")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"driver_uncertainty_{tag}.npz")
    np.savez(
        out,
        drivers=np.asarray(list(drivers)),
        g_coeffs=np.asarray([DRIVERS[n] for n in drivers]),
        c_true=c_trues,
        windows=np.asarray(windows),
        window_days=np.asarray([steps_to_days(w) for w in windows]),
        col_norm=col_norm, se_unit=se_unit, vif=vif, align=align, sigma=sigma,
        cond_J=cond_J, smin=smin, smax=smax, jrank=jrank,
        eps=cfg.eps, obs=cfg.obs, dt=dt, seed=cfg.seed, n_ic=n_ic,
        whiten=bool(whiten),
    )
    if verbose:
        print(f"[done]  wrote {out}  ({time.time() - t0:.1f}s total)")
    return out


def _load(tag):
    return np.load(os.path.join(RESULTS_DIR, f"driver_uncertainty_{tag}.npz"),
                   allow_pickle=True)


def figure(tag="main"):
    """Three panels, one curve per driver, all of them per-coefficient quantities.

    (a) is the U that cond(J) flattens: the relative standard error of the driver's own
    coefficient per unit observation noise. (b) is the variance inflation from fitting the
    other three coefficients alongside it -- the cost of not knowing the form, which is a
    property of the driver and not of the base dynamics. (c) is the mechanism: how much of the
    driver's own coefficient direction lies in the worst-determined singular direction.

    A second figure, ``fig_driver_uncertainty_bounds_<tag>.png``, is written alongside it: one
    panel per driver, all windows, with the four nested bounds and the two gaps that matter --
    the grey envelope whose height is cond(J_w), and the coloured band whose height is
    sqrt(VIF). Those panels show medians only; the segment range for se_k is in panel (a) here,
    which shares their y-axis meaning.

    Panel (c) is only readable at the long-window end: between 3.6 and 28.7 d the crossings are
    inside the segment-to-segment spread (see the module docstring). v_min itself is never
    degenerate, so the quantity is well posed everywhere -- it is the three-segment median that
    is not resolved there.
    """
    plt = figstyle.pyplot()
    d = _load(tag)
    names = [str(n) for n in d["drivers"]]
    wd = d["window_days"]
    c_true = d["c_true"]
    wh = bool(d["whiten"]) if "whiten" in d.files else False
    wnote = "whitened residual (1/std per channel)" if wh else "unwhitened residual"

    fig, (ax, axb, axc) = plt.subplots(1, 3, figsize=(15.0, 4.4))
    colors = plt.get_cmap("tab10").colors

    for i, nm in enumerate(names):
        q = own(d["se_unit"][i], c_true[i], relative=True)          # (nW, n_ic)
        rse = np.nanmedian(q, axis=1)
        # the band is the segment-to-segment range: two drivers differ only where it separates
        ax.fill_between(wd, np.nanmin(q, axis=1), np.nanmax(q, axis=1),
                        color=colors[i], alpha=0.15, lw=0)
        ax.plot(wd, rse, "o-", ms=4, color=colors[i], label=nm)
        ax.plot(wd[int(np.argmin(rse))], rse.min(), "*", ms=13, color=colors[i])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("optimisation window  [days]")
    ax.set_ylabel(r"$\sigma_{\hat c_k}/(|c_k|\,\sigma_{\rm obs})$")
    ax.set_title("(a)  uncertainty of the driver's own coefficient\n"
                 rf"{wnote}" "\n" r"band = segment range; $\star$ = best window",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, title="driver $g(x)$", title_fontsize=8)

    for i, nm in enumerate(names):
        vf = np.nanmedian(own(d["vif"][i], c_true[i]), axis=1)
        axb.plot(wd, vf, "s-", ms=5, color=colors[i], label=nm)
    axb.axhline(1.0, ls=":", color="grey", lw=1)
    axb.text(wd[0], 1.05, "no aliasing cost", color="grey", fontsize=8,
             va="bottom", ha="left", bbox=LBL_BBOX)
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel("optimisation window  [days]")
    axb.set_ylabel(r"VIF$_k=\|J_{:,k}\|^2\,[(J^TJ)^{-1}]_{kk}$")
    axb.set_title("(b)  price of not knowing the form\n"
                  "variance inflation from fitting the other three coefficients",
                  fontsize=9.5, loc="left")

    for i, nm in enumerate(names):
        q = own(d["align"][i], c_true[i])
        # Five overlapping bands are unreadable, so only the amplitude-matched pair carries
        # one -- theta_a1 vs T_o1 is the single comparison this panel is asked to support.
        if nm in ("theta_a1", "T_o1"):
            axc.fill_between(wd, np.nanmin(q, axis=1), np.nanmax(q, axis=1),
                             color=colors[i], alpha=0.18, lw=0)
        axc.plot(wd, np.nanmedian(q, axis=1), "^-", ms=5, color=colors[i], label=nm)
    axc.set_xscale("log")
    axc.set_ylim(0, 1.02)
    axc.set_xlabel("optimisation window  [days]")
    axc.set_ylabel(r"$|\langle v_{\min},e_k\rangle|$")
    axc.set_title("(c)  mechanism: overlap with the worst-determined singular\n"
                  "direction; bands = the amplitude-matched pair, 3 segments",
                  fontsize=9.5, loc="left")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"fig_driver_uncertainty_{tag}.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return [out, _draw_bounds(plt, d, tag)]


def _nesting_holds(d, rtol=1e-9):
    """True iff 1/smax <= 1/||J e_k|| <= se_k <= 1/smin everywhere in the stored arrays."""
    c_true = d["c_true"]
    for i in range(len(d["drivers"])):
        lo = 1.0 / d["smax"][i]                                   # (nW, n_ic)
        known = 1.0 / own(d["col_norm"][i], c_true[i])
        fitted = own(d["se_unit"][i], c_true[i])
        hi = 1.0 / d["smin"][i]
        for a, b in ((lo, known), (known, fitted), (fitted, hi)):
            if not np.all(a <= b * (1.0 + rtol)):
                return False
    return True


def _draw_bounds(plt, d, tag):
    """Draw the nested-bounds figure -- one panel per driver, all windows. Returns its path."""
    names = [str(n) for n in d["drivers"]]
    wd = d["window_days"]
    c_true = d["c_true"]
    wh = bool(d["whiten"]) if "whiten" in d.files else False
    jsym = "J_w" if wh else "J"
    wnote = "whitened residual (1/std per channel)" if wh else "unwhitened residual"
    colors = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.3), sharey=True)
    for i, (nm, ax) in enumerate(zip(names, axes)):
        # |c_k| -- 0.1 for all four drivers, so this scaling is uniform and changes no comparison
        ck = abs(float(own(c_true[i], c_true[i])))
        med = lambda a: np.nanmedian(a, axis=1)
        best = med(1.0 / d["smax"][i]) / ck
        worst = med(1.0 / d["smin"][i]) / ck
        known = med(1.0 / own(d["col_norm"][i], c_true[i])) / ck
        fitted = med(own(d["se_unit"][i], c_true[i])) / ck

        # each panel draws its *own* driver's envelope -- that they coincide is a result
        ax.fill_between(wd, best, worst, color="0.85", alpha=0.55, lw=0)
        ax.plot(wd, best, color="0.5", lw=1)
        ax.plot(wd, worst, color="0.5", lw=1)
        ax.fill_between(wd, known, fitted, color=colors[i], alpha=0.2, lw=0)
        ax.plot(wd, known, "s--", ms=4, lw=1.2, color=colors[i])
        ax.plot(wd, fitted, "o-", ms=4, color=colors[i])

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("optimisation window  [days]")
        ax.set_title(f"({chr(97 + i)})  {nm}", fontsize=10, loc="left")
        # per segment, then median -- the ratio of the two median curves is not the median
        # ratio, and at short windows the two differ by 3x
        svif = np.nanmedian(np.sqrt(own(d["vif"][i], c_true[i])[-1]))
        ax.text(0.96, 0.94, rf"$\sqrt{{\rm VIF}}={svif:.0f}$ at {wd[-1]:.0f} d",
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

    fig.suptitle(rf"nested bounds on the driver's own coefficient; {wnote}. "
                 rf"grey band $=$ cond$({jsym})$, coloured band $=\sqrt{{\rm VIF}}$, "
                 rf"the price of not knowing the form", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    out = os.path.join(RESULTS_DIR, f"fig_driver_uncertainty_bounds_{tag}.png")
    fig.savefig(out, dpi=DPI); plt.close(fig)
    return out


def bounds_table(tag="main", window_days=89.7):
    """Print the four nested bounds per driver at one window; return them as an array.

    Columns, with every standard error relative to |c_k| and per unit sigma_obs:

        cond(J)   1/sigma_max   1/||J e_k||   se_k   1/sigma_min   sqrt(VIF)

    Every column but the last is a median over segments of that segment's own value. sqrt(VIF)
    is too -- taken per segment and then summarised, not as the ratio of the se_k and
    1/||J e_k|| columns, which are separately-taken medians and would disagree by up to 3x at
    the short windows.

    They answer five different questions about the same coefficient: the best any *combination*
    of coefficients could be determined; this one alone with the other three known exactly; this
    one with all four fitted, which is what the study actually gets; the worst any combination
    could be; and the ratio of the outer two. sqrt(VIF) is the gap between the middle two, as a
    multiplier on the error rather than on its square.

    |c_k| = eps = 0.1 for every driver here, so the relative scaling is a uniform 10x and no
    comparison -- between drivers or between columns -- depends on it.

    The nesting is what the bounds figure draws, so it is *checked* here over every window,
    driver and segment rather than assumed.
    """
    d = _load(tag)
    names = [str(n) for n in d["drivers"]]
    wd, c_true = d["window_days"], d["c_true"]
    n_ic = d["se_unit"].shape[2]
    b = int(np.argmin(np.abs(wd - window_days)))

    rows = np.zeros((len(names), 6))
    for i in range(len(names)):
        ck = abs(float(own(c_true[i], c_true[i])))
        cn = float(np.nanmedian(own(d["col_norm"][i], c_true[i])[b]))
        se = float(np.nanmedian(own(d["se_unit"][i], c_true[i])[b]))
        rows[i] = (np.nanmedian(d["cond_J"][i, b]),
                   np.nanmedian(1.0 / d["smax"][i, b]) / ck,
                   1.0 / cn / ck,
                   se / ck,
                   np.nanmedian(1.0 / d["smin"][i, b]) / ck,
                   # per segment, then median: the ratio of medians is not the median ratio
                   np.nanmedian(np.sqrt(own(d["vif"][i], c_true[i])[b])))

    cols = ("cond(J)", "1/sigma_max", "1/||J e_k||", "se_k", "1/sigma_min", "sqrt(VIF)")
    print(f"window {wd[b]:.1f} d, per unit sigma_obs, relative to |c_k|, "
          f"median of {n_ic} segments")
    print(f"{'driver':<9s}" + "".join(f"{c:>13s}" for c in cols))
    for i, nm in enumerate(names):
        print(f"{nm:<9s}" + "".join(f"{v:13.3e}" for v in rows[i, :5])
              + f"{rows[i, 5]:13.0f}")
    print(f"nesting 1/smax <= 1/||Je_k|| <= se_k <= 1/smin: "
          f"{'holds' if _nesting_holds(d) else 'VIOLATED'} across all "
          f"{len(wd)} windows x {len(names)} drivers x {n_ic} segments")
    return rows
