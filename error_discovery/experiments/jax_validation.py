"""Validation figure: the JAX re-implementation reproduces the original qgs model.

Integrates the same physical attractor state with (a) the original numpy/numba
``qgs`` RungeKuttaIntegrator and (b) the ``qgs_jax`` RK4 integrator, from an
identical initial condition, and overlays the evolution of one atmospheric mode
together with the state-wide divergence between the two codes.

This is the only study that touches the original ``qgs`` integrator, which sizes a
multiprocessing pool to ``cpu_count()`` unless told otherwise. ``run`` passes
``num_threads=1`` and imports it inside the function body, so importing this module —
which ``--help`` does not do anyway — spawns nothing.
"""

import os

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from qgs_jax.spec import extract_spec                          # noqa: E402
from qgs_jax import model                                      # noqa: E402
from .. import figstyle                                        # noqa: E402
from ..config import NDIM_PER_DAY, DT, DATA_DIR, FIGURES_DIR, IC_CACHE   # noqa: E402
from ..figstyle import DPI                                     # noqa: E402

IDX_PSIA1 = 0          # leading atmospheric streamfunction mode psi_{a,1}


def run(nsteps=5000, dt=DT):
    """Integrate both codes from the cached attractor state; save what the figure draws.

    Stores the leading atmospheric mode from each code and the state-wide divergence,
    not the two full ``(nsteps, 36)`` trajectories — the figure needs three curves.
    """
    spec = extract_spec()
    T_ref = jnp.asarray(spec.dense_tensor)

    ic = np.load(IC_CACHE)["x"].astype(float)

    from qgs.integrators.integrator import RungeKuttaIntegrator
    integ = RungeKuttaIntegrator(num_threads=1)
    integ.set_func(spec.f_numpy)
    integ.integrate(0.0, nsteps * dt, dt, ic=ic.copy(), write_steps=1)
    _, traj_np = integ.get_trajectories()
    traj_np = np.asarray(traj_np).T[1:]        # drop the stored IC at index 0

    traj_jx = np.asarray(
        model.integrate(T_ref, jnp.asarray(ic), dt, nsteps, write_every=1, method="rk4")
    )
    out = os.path.join(DATA_DIR, "jax_validation.npz")
    np.savez(out,
             t_days=(np.arange(nsteps) + 1) * dt / NDIM_PER_DAY,
             psi_np=traj_np[:, IDX_PSIA1],
             psi_jx=traj_jx[:, IDX_PSIA1],
             state_diff=np.max(np.abs(traj_np - traj_jx), axis=1))   # over all 36 modes
    print(f"wrote {out}")
    return out


def figure():
    plt = figstyle.pyplot()

    d = np.load(os.path.join(DATA_DIR, "jax_validation.npz"))
    t, state_diff = d["t_days"], d["state_diff"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1]))

    ax1.plot(t, d["psi_np"], color="0.2", lw=1.4, label="original qgs (numpy/numba)")
    ax1.plot(t, d["psi_jx"], color="C1", lw=1.4, ls="--", dashes=(4, 4),
             label="qgs_jax (JAX RK4)")
    ax1.set_ylabel(r"$\psi_{a,1}$ (nondim)")
    ax1.legend(loc="upper right", frameon=False, fontsize=8)

    ax2.semilogy(t, state_diff, color="C3", lw=1.2)
    ax2.set_ylabel("max$_i\\,|x_i^{\\rm JAX}-x_i^{\\rm qgs}|$")
    ax2.set_xlabel("time (days)")
    ax2.set_xlim(t[0], t[-1])

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_jax_validation.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"max state divergence over {len(t)} steps: {state_diff.max():.2e}")
    print(f"wrote {out}")
    return [out]


if __name__ == "__main__":
    run()
    figure()
