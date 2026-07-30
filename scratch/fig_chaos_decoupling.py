"""Chaos-decoupling figure (test_ideas -> manuscript fig_chaos_decoupling.png).

Recovery / identifiability vs optimization window measured in UNITS OF THE PREDICTABILITY
HORIZON, on a weakly chaotic (36-dim, tau~37.5 d) and a truly chaotic (58-dim, tau~11 d)
converged attractor. The injected error is anchored at the converged IC; we estimate
coefficients only (the state is pinned), so the optimizer never pushes the trajectory
into instability.

(a) noiseless recovery error for IN-CLASS errors -- the canonical single-term error
    g=T_{o,1} (simple, 0,0,1,0) and a three-term mixed error (0,1,1,1). The truth lies in
    the library, so recovery error is meaningful.
(b) multistart P(reach truth) for the same in-class errors -- loss-surface unimodality.
(c) multistart consistency for genuinely OUT-OF-CLASS nonlinear gates (theta_a1*T_o1 and
    psi_a1^2). These cannot be represented by the linear library, so they can only alias;
    consistency = fraction of random starts that agree on the lowest-cost fit.

Vertical dotted line marks one predictability horizon. Decoupling holds several horizons
(~4x), then the conditioning barrier reasserts -- earliest for the hardest errors.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "error_discovery", "results")
HORIZON = {36: 37.5, 58: 11.0}      # 2-seed 18-yr Benettin estimate (lyap_converged.py)


def L(f):
    p = os.path.join(HERE, f)
    return np.load(p) if os.path.exists(p) else None


fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))

# (a),(b) in-class errors: simple (0,0,1,0) and mixed (0,1,1,1)
inclass = [(36, "simple", "chaotic_win_36.npz", "C0", "-", "o"),
           (58, "simple", "chaotic_win_58.npz", "C3", "-", "o"),
           (36, "mixed",  "chaos_0111_36.npz",  "C0", "--", "s"),
           (58, "mixed",  "chaos_0111_58.npz",  "C3", "--", "s")]
for ndim, err, fn, c, ls, mk in inclass:
    d = L(fn)
    if d is None:
        continue
    xh = np.asarray(d["days"]) / HORIZON[ndim]
    ax[0].plot(xh, np.maximum(d["rel"], 1e-13), mk + ls, color=c, ms=4, label=f"{ndim}-dim {err}")
    ax[1].plot(xh, d["ms"], mk + ls, color=c, ms=4, label=f"{ndim}-dim {err}")

# (c) genuinely out-of-class nonlinear gates: theta_a1*T_o1 and psi_a1^2
ooc = [(36, r"$\theta_{a,1}T_{o,1}$", "chaos_nl_36.npz",   "C0", "-",  "^"),
       (58, r"$\theta_{a,1}T_{o,1}$", "chaos_nl_58.npz",   "C3", "-",  "^"),
       (36, r"$\psi_{a,1}^2$",        "chaos_psi2_36.npz", "C0", "--", "v"),
       (58, r"$\psi_{a,1}^2$",        "chaos_psi2_58.npz", "C3", "--", "v")]
for ndim, err, fn, c, ls, mk in ooc:
    d = L(fn)
    if d is None:
        continue
    xh = np.asarray(d["days"]) / HORIZON[ndim]
    ax[2].plot(xh, d["cons"], mk + ls, color=c, ms=4, label=f"{ndim}-dim {err}")

for a in ax:
    a.axvline(1.0, ls=":", color="grey", lw=1.2)
    a.set_xscale("log")
    a.set_xlabel("window / predictability horizon")

# white-framed legends so they mask the (busy) curves behind them instead of overprinting
_LEG = dict(fontsize=7, frameon=True, framealpha=0.9, edgecolor="none", facecolor="white")

ax[0].set_yscale("log")
ax[0].axhline(0.1, ls=":", color="grey", lw=1)
ax[0].set_ylabel(r"noiseless recovery error $\|\hat c-c_{\rm true}\|/\|c_{\rm true}\|$")
ax[0].set_title("(a) recovery error (in-class)", loc="left", fontsize=9)
ax[0].legend(loc="lower right", **_LEG)

ax[1].set_ylim(-0.05, 1.05)
ax[1].set_ylabel("multistart P(reach truth)")
ax[1].set_title("(b) loss surface: unimodal vs multimodal", loc="left", fontsize=9)
ax[1].legend(loc="lower left", **_LEG)

ax[2].set_ylim(-0.05, 1.05)
ax[2].set_ylabel("multistart consistency")
ax[2].set_title("(c) out-of-class nonlinear gates", loc="left", fontsize=9)
ax[2].legend(loc="lower left", **_LEG)

fig.tight_layout()
os.makedirs(RES, exist_ok=True)
out = os.path.join(RES, "fig_chaos_decoupling.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("wrote", out)
