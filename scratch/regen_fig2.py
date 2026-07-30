"""
Targeted regeneration of the window-of-discoverability data for Figure 2.

Re-runs ONLY the noiseless coefficient-recovery sweep at the converged iteration
budget (cfg.lm_iters=40), so the noiseless curve reflects the true recoverability
(machine precision at every window) rather than an under-converged optimizer. The
observation-noise sweep arrays are preserved from the existing npz (noisy fits are
noise-limited). Rewrites leadtime_main.npz and the three figures.
"""
import os
import numpy as np

from error_discovery.config import ExperimentConfig, RESULTS_DIR, steps_to_days
from error_discovery.models import TwinModels, obs_mask
from error_discovery.discovery import recover
from error_discovery.experiment import get_attractor_ic, build_segments
from error_discovery import plotting

cfg = ExperimentConfig()
tw = TwinModels(cfg.eps, cfg.g_coeffs)
vr = tw.spec.variables_range
mask = obs_mask(tw.ndim, vr, cfg.obs)
xa = get_attractor_ic(tw, cfg, verbose=False)
seg_ics, seg_trajs = build_segments(tw, cfg, xa)

windows = cfg.window_steps
nW, nI = len(windows), cfg.n_ic
rel = np.full((nW, nI), np.nan)
cond = np.full((nW, nI), np.nan)         # cond(J^T J) = cond(J)^2 (back-compat)
cond_J = np.full((nW, nI), np.nan)       # cond(J) from SVD (reviewer #3)
smin = np.full((nW, nI), np.nan)
smax = np.full((nW, nI), np.nan)
jrank = np.full((nW, nI), np.nan)
chats = np.full((nW, nI, 4), np.nan)
for iw, w in enumerate(windows):
    for k in range(nI):
        r = recover(tw, seg_ics[k], seg_trajs[k][:w], cfg.dt, mask,
                    eta=0.0, n_iters=cfg.lm_iters)
        rel[iw, k] = r["rel_err"]; cond[iw, k] = r["cond"]; chats[iw, k] = r["c_hat"]
        cond_J[iw, k] = r["cond_J"]; smin[iw, k] = r["smin"]
        smax[iw, k] = r["smax"]; jrank[iw, k] = r["rank"]
    print(f"w={w:5d} ({steps_to_days(w):5.1f} d)  rel_err med={np.nanmedian(rel[iw]):.3e}  "
          f"cond(J) med={np.nanmedian(cond_J[iw]):.2e}  smin={np.nanmedian(smin[iw]):.2e}  "
          f"smax={np.nanmedian(smax[iw]):.2e}  rank={np.nanmedian(jrank[iw]):.0f}", flush=True)

old = np.load(os.path.join(RESULTS_DIR, "leadtime_main.npz"))
np.savez(
    os.path.join(RESULTS_DIR, "leadtime_main.npz"),
    windows=np.asarray(windows),
    window_days=np.asarray([steps_to_days(w) for w in windows]),
    rel_err=rel, cond=cond, cond_J=cond_J, smin=smin, smax=smax, jrank=jrank,
    c_hats=chats, c_true=np.asarray(tw.c_true),
    noise_levels=old["noise_levels"], noise_windows=old["noise_windows"],
    noise_window_days=old["noise_window_days"], rel_err_noise=old["rel_err_noise"],
    lam_max=old["lam_max"], tau_lyap_days=old["tau_lyap_days"],
    eps=cfg.eps, g_coeffs=np.asarray(cfg.g_coeffs), obs=cfg.obs, dt=cfg.dt, seed=cfg.seed,
)
print("rewrote leadtime_main.npz")
print(plotting.fig_recovery_and_conditioning("main"))
print(plotting.fig_noise("main"))
print(plotting.fig_coeff_recovery("main"))
