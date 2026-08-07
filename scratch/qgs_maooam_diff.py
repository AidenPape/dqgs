#!/usr/bin/env python
"""
qgs_maooam_diff.py
==================
Compare the unperturbed and perturbed MAOOAM trajectories.

Plots:
  1. Time series of selected variables (true vs bad model)
  2. Difference time series (perturbed - unperturbed)
  3. RMS divergence across all state variables over time
  4. Histogram of exchange-relevant variables: theta_a_1, T_o_1
"""

import numpy as np
import matplotlib.pyplot as plt

OUTDIR  = "/glade/derecho/scratch/wchapman/qgs_runs/"
FIGDIR  = OUTDIR

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading trajectories...")
traj_ref  = np.loadtxt(OUTDIR + "evol_fields.dat")
traj_bad  = np.loadtxt(OUTDIR + "evol_fields_perturbed.dat")

time      = traj_ref[:, 0]          # nondimensional time
x_ref     = traj_ref[:, 1:]         # shape (nT, 36)
x_bad     = traj_bad[:, 1:]

diff      = x_bad - x_ref
nT, ndim  = x_ref.shape

print(f"  nT={nT}, ndim={ndim}")
print(f"  time range: {time[0]:.1f} – {time[-1]:.1f}")

# ── index layout ──────────────────────────────────────────────────────────────
# x[0:10]  = psi_a_1 ... psi_a_10
# x[10:20] = theta_a_1 ... theta_a_10
# x[20:28] = psi_o_1  ... psi_o_8
# x[28:36] = T_o_1    ... T_o_8

VARS = {
    r"$\psi_{a,1}$":   0,
    r"$\theta_{a,1}$": 10,
    r"$\psi_{o,1}$":   20,
    r"$T_{o,1}$":      28,
}

# ── Figure 1: time series comparison ─────────────────────────────────────────
fig, axes = plt.subplots(len(VARS), 1, figsize=(12, 10), sharex=True)
for ax, (label, idx) in zip(axes, VARS.items()):
    ax.plot(time, x_ref[:, idx], color="steelblue",  lw=0.8, label="unperturbed")
    ax.plot(time, x_bad[:, idx], color="tomato",     lw=0.8, alpha=0.8, label="perturbed")
    ax.set_ylabel(label, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Nondimensional time")
fig.suptitle("MAOOAM: unperturbed vs perturbed  (ε=0.1, g(x)=T$_{o,1}$)", fontsize=12)
fig.tight_layout()
fig.savefig(FIGDIR + "timeseries_comparison.png", dpi=150)
print("Saved timeseries_comparison.png")
plt.close(fig)

# ── Figure 2: difference time series ─────────────────────────────────────────
fig, axes = plt.subplots(len(VARS), 1, figsize=(12, 10), sharex=True)
for ax, (label, idx) in zip(axes, VARS.items()):
    ax.plot(time, diff[:, idx], color="purple", lw=0.8)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel(f"Δ{label}", fontsize=11)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Nondimensional time")
fig.suptitle("MAOOAM: difference  (perturbed − unperturbed)", fontsize=12)
fig.tight_layout()
fig.savefig(FIGDIR + "timeseries_diff.png", dpi=150)
print("Saved timeseries_diff.png")
plt.close(fig)

# ── Figure 3: RMS divergence over time ───────────────────────────────────────
rms_all   = np.sqrt(np.mean(diff**2, axis=1))
rms_atm   = np.sqrt(np.mean(diff[:, :20]**2, axis=1))
rms_oc    = np.sqrt(np.mean(diff[:, 20:]**2, axis=1))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time, rms_all, color="k",         lw=1.2, label="all variables")
ax.plot(time, rms_atm, color="steelblue", lw=1.0, ls="--", label="atmosphere only")
ax.plot(time, rms_oc,  color="tomato",    lw=1.0, ls="--", label="ocean only")
ax.set_xlabel("Nondimensional time")
ax.set_ylabel("RMS  |perturbed − unperturbed|")
ax.set_title("State divergence over time")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR + "rms_divergence.png", dpi=150)
print("Saved rms_divergence.png")
plt.close(fig)

# ── Figure 4: histograms of exchange-relevant variables ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (label, idx) in zip(axes, [(r"$\theta_{a,1}$", 10), (r"$T_{o,1}$", 28)]):
    ax.hist(x_ref[:, idx], bins=60, color="steelblue", alpha=0.6, label="unperturbed", density=True)
    ax.hist(x_bad[:, idx], bins=60, color="tomato",    alpha=0.6, label="perturbed",   density=True)
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.3)
fig.suptitle("Marginal distributions: exchange-relevant variables", fontsize=12)
fig.tight_layout()
fig.savefig(FIGDIR + "histograms.png", dpi=150)
print("Saved histograms.png")
plt.close(fig)

# ── print summary stats ───────────────────────────────────────────────────────
print("\nSummary statistics")
print(f"{'Variable':<20}  {'ref mean':>10}  {'bad mean':>10}  {'ref std':>10}  {'bad std':>10}  {'diff RMS':>10}")
print("-" * 75)
for label, idx in VARS.items():
    print(f"{label:<20}  {x_ref[:,idx].mean():>10.4f}  {x_bad[:,idx].mean():>10.4f}"
          f"  {x_ref[:,idx].std():>10.4f}  {x_bad[:,idx].std():>10.4f}"
          f"  {diff[:,idx].std():>10.4f}")
