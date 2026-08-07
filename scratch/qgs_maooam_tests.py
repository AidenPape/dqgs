#!/usr/bin/env python
"""
qgs_maooam_tests.py
===================
Tests 2-6 for the perturbed MAOOAM implementation.

Test 2: Constant perturbation equivalence
    g_coeffs=(1,0,0,0) -> eps*g(x)=eps (constant), so the perturbed model
    should be identical to running with Lpa and Lpgo scaled by (1+eps).
    Verified by comparing tendency evaluations point-by-point.

Test 3: g(x) = T_{o,1} distribution over the trajectory.

Test 4: Exchange tendency fraction — how big is eps*g(x)*(E@x) relative
    to the unperturbed exchange tendency (E@x)?

Test 5: Phase space portraits theta_{a,1} vs T_{o,1}.

Test 6: Power spectra of theta_{a,1} and T_{o,1}.
"""

import numpy as np
import matplotlib.pyplot as plt

from qgs.params.params import QgParams
from qgs.functions.tendencies import create_tendencies
from qgs.functions.perturbed_tendencies import create_exchange_perturbed_tendencies

OUTDIR   = "/glade/derecho/scratch/wchapman/qgs_runs/"
PERTURB_EPS = 0.1

# ── shared model parameters ───────────────────────────────────────────────────
def make_params(hlambda=15.06):
    p = QgParams()
    p.set_atmospheric_channel_fourier_modes(2, 2)
    p.set_oceanic_basin_fourier_modes(2, 4)
    p.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5,
                  'r': 1.e-7, 'h': 136.5, 'd': 1.1e-7})
    p.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3, 'hlambda': hlambda})
    p.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})
    p.atemperature_params.set_insolation(103.3333, 0)
    p.gotemperature_params.set_insolation(310., 0)
    return p

# ── load saved trajectories ───────────────────────────────────────────────────
print("Loading trajectories...")
traj_ref = np.loadtxt(OUTDIR + "evol_fields.dat")
traj_bad = np.loadtxt(OUTDIR + "evol_fields_perturbed.dat")
time     = traj_ref[:, 0]
x_ref    = traj_ref[:, 1:]   # (nT, 36)
x_bad    = traj_bad[:, 1:]
nT       = x_ref.shape[0]
print(f"  nT={nT},  time={time[0]:.0f}–{time[-1]:.0f}")

# state index layout
idx_thetaa1 = 10   # theta_a_1
idx_To1     = 28   # T_o_1

# ── TEST 2: constant perturbation == scaled lambda ────────────────────────────
print("\n--- Test 2: constant perturbation equivalence ---")

params_orig   = make_params(hlambda=15.06)
params_scaled = make_params(hlambda=15.06 * (1 + PERTURB_EPS))

f_scaled, _          = create_tendencies(params_scaled)
f_const, _, E_atm, E_oc = create_exchange_perturbed_tendencies(
    params_orig, eps=PERTURB_EPS, g_coeffs=(1., 0., 0., 0.)
)

# warm up numba
_ = f_scaled(0., x_ref[0])
_ = f_const(0.,  x_ref[0])

# evaluate at 200 random trajectory points
rng      = np.random.default_rng(42)
idx_test = rng.choice(nT, size=200, replace=False)
max_err  = 0.
for i in idx_test:
    x  = x_ref[i]
    fa = f_scaled(0., x)
    fb = f_const(0.,  x)
    max_err = max(max_err, np.max(np.abs(fa - fb)))

print(f"  max |f_scaled - f_const| over 200 points: {max_err:.2e}")
if max_err < 1e-12:
    print("  PASS — constant perturbation == scaled lambda")
else:
    print("  WARN — non-trivial difference, check implementation")

# ── TEST 3: g(x) = T_{o,1} distribution ──────────────────────────────────────
print("\n--- Test 3: g(x) distribution ---")

g_traj    = x_ref[:, idx_To1]        # g(x) = T_{o,1}  along reference traj
eg_traj   = PERTURB_EPS * g_traj     # eps * g(x)  — actual multiplicative error

print(f"  g(x) = T_o1:  mean={g_traj.mean():.4f},  std={g_traj.std():.4f}"
      f",  min={g_traj.min():.4f},  max={g_traj.max():.4f}")
print(f"  eps*g(x):     mean={eg_traj.mean():.4f},  std={eg_traj.std():.4f}"
      f",  min={eg_traj.min():.4f},  max={eg_traj.max():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(g_traj,  bins=80, color="steelblue", density=True, edgecolor="none")
axes[0].axvline(0, color="k", lw=1, ls="--")
axes[0].set_xlabel(r"$g(x) = T_{o,1}$")
axes[0].set_ylabel("density")
axes[0].set_title("Distribution of g(x) along reference trajectory")

axes[1].plot(time, eg_traj, lw=0.6, color="steelblue")
axes[1].axhline(0, color="k", lw=0.8, ls="--")
axes[1].set_xlabel("Nondimensional time")
axes[1].set_ylabel(r"$\varepsilon \cdot g(x)$")
axes[1].set_title(r"$\varepsilon \cdot g(x)$ time series  ($\varepsilon=0.1$)")
fig.tight_layout()
fig.savefig(OUTDIR + "test3_g_distribution.png", dpi=150)
print("  Saved test3_g_distribution.png")
plt.close(fig)

# ── TEST 4: exchange tendency fraction ────────────────────────────────────────
print("\n--- Test 4: exchange tendency fraction ---")

# exchange tendency at each timestep
exch_a = x_ref @ E_atm.T    # (nT, n_thetaa)  unperturbed exchange for theta_a
exch_o = x_ref @ E_oc.T     # (nT, n_To)      unperturbed exchange for T_o
g_vec  = x_ref[:, idx_To1]  # g(x) along reference trajectory

# perturbation exchange = eps * g(x) * E @ x
pert_a = (PERTURB_EPS * g_vec)[:, None] * exch_a   # (nT, n_thetaa)
pert_o = (PERTURB_EPS * g_vec)[:, None] * exch_o   # (nT, n_To)

# ratio of perturbation to unperturbed exchange (avoid div by zero)
ratio_a = np.where(np.abs(exch_a) > 1e-15, np.abs(pert_a) / np.abs(exch_a), np.nan)
ratio_o = np.where(np.abs(exch_o) > 1e-15, np.abs(pert_o) / np.abs(exch_o), np.nan)

print(f"  |pert_a| / |exch_a|:  median={np.nanmedian(ratio_a):.4f},"
      f"  90th pct={np.nanpercentile(ratio_a, 90):.4f}")
print(f"  |pert_o| / |exch_o|:  median={np.nanmedian(ratio_o):.4f},"
      f"  90th pct={np.nanpercentile(ratio_o, 90):.4f}")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(time, np.abs(pert_a[:, 0]), lw=0.7, label=r"$|\delta\dot\theta_{a,1}|$  (pert)")
axes[0].plot(time, np.abs(exch_a[:, 0]), lw=0.7, alpha=0.6, label=r"$|\dot\theta_{a,1}^{exch}|$  (base)")
axes[0].set_ylabel("Tendency magnitude")
axes[0].set_title(r"$\theta_{a,1}$ exchange: perturbation vs unperturbed exchange tendency")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, np.abs(pert_o[:, 0]), lw=0.7, label=r"$|\delta\dot{T}_{o,1}|$  (pert)")
axes[1].plot(time, np.abs(exch_o[:, 0]), lw=0.7, alpha=0.6, label=r"$|\dot{T}_{o,1}^{exch}|$  (base)")
axes[1].set_xlabel("Nondimensional time")
axes[1].set_ylabel("Tendency magnitude")
axes[1].set_title(r"$T_{o,1}$ exchange: perturbation vs unperturbed exchange tendency")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTDIR + "test4_exchange_fraction.png", dpi=150)
print("  Saved test4_exchange_fraction.png")
plt.close(fig)

# ── TEST 5: phase space portraits ─────────────────────────────────────────────
print("\n--- Test 5: phase space portraits ---")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (x, label, color) in zip(axes, [
        (x_ref, "unperturbed", "steelblue"),
        (x_bad, "perturbed",   "tomato")]):
    ax.scatter(x[:, idx_thetaa1], x[:, idx_To1],
               s=0.3, alpha=0.4, color=color, rasterized=True)
    ax.set_xlabel(r"$\theta_{a,1}$", fontsize=12)
    ax.set_ylabel(r"$T_{o,1}$", fontsize=12)
    ax.set_title(label)
    ax.grid(True, alpha=0.2)
fig.suptitle(r"Phase space: $\theta_{a,1}$ vs $T_{o,1}$", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR + "test5_phase_space.png", dpi=150)
print("  Saved test5_phase_space.png")
plt.close(fig)

# ── TEST 6: power spectra ─────────────────────────────────────────────────────
print("\n--- Test 6: power spectra ---")

dt_save = time[1] - time[0]   # nondim time between saved steps

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, (label, idx) in zip(axes, [(r"$\theta_{a,1}$", idx_thetaa1),
                                    (r"$T_{o,1}$",      idx_To1)]):
    for x, color, name in [(x_ref, "steelblue", "unperturbed"),
                            (x_bad, "tomato",    "perturbed")]:
        sig  = x[:, idx] - x[:, idx].mean()
        ps   = np.abs(np.fft.rfft(sig))**2
        freq = np.fft.rfftfreq(nT, d=dt_save)
        # smooth with a running mean for readability
        w    = 10
        ps_s = np.convolve(ps, np.ones(w)/w, mode='same')
        ax.loglog(freq[1:], ps_s[1:], lw=1.0, color=color, alpha=0.85, label=name)
    ax.set_ylabel("Power")
    ax.set_title(f"Power spectrum: {label}")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
axes[-1].set_xlabel("Frequency  (nondim time)$^{-1}$")
fig.suptitle("Power spectra: unperturbed vs perturbed", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR + "test6_power_spectra.png", dpi=150)
print("  Saved test6_power_spectra.png")
plt.close(fig)

print("\nAll tests complete.")
