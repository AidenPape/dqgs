#!/usr/bin/env python
# coding: utf-8

"""
MAOOAM with state-dependent heat-exchange perturbation
=======================================================

Runs the coupled ocean-atmosphere MAOOAM model with an optional
state-dependent error injected into the thermal exchange coefficient.

The "bad" model replaces  lambda'_a  and  lambda'_o  by

    lambda'_eff(x) = lambda'  *  [1 + eps * g(x)]

where
    g(x) = a0 + a1*theta_{a,1} + a2*T_{o,1} + a3*psi_{a,1}

is controlled by the settings below.

Toggle PERTURB to run the bad model (True) or the reference model (False).
"""

import numpy as np
import sys
import time
from multiprocessing import freeze_support, get_start_method

from qgs.params.params import QgParams
from qgs.integrators.integrator import RungeKuttaIntegrator
from qgs.functions.tendencies import create_tendencies
from qgs.functions.perturbed_tendencies import create_exchange_perturbed_tendencies

np.random.seed(21217)

# ── perturbation settings ──────────────────────────────────────────────────────
PERTURB     = False          # False -> standard unperturbed MAOOAM
PERTURB_EPS = 0.1            # epsilon: perturbation amplitude
G_COEFFS    = (0., 0., 1., 0.)  # (a0, a1, a2, a3) for g(x)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if get_start_method() == "spawn":
        freeze_support()

    print_parameters = True

    def print_progress(p):
        sys.stdout.write('Progress {:.2%} \r'.format(p))
        sys.stdout.flush()

    label = "PERTURBED" if PERTURB else "UNPERTURBED"
    print(f"\nMAOOAM ({label})")
    print("=" * 60)
    print("Initialization ...")

    # ── time parameters ────────────────────────────────────────────────────────
    dt               = 0.1
    write_steps      = 100
    transient_time   = 3.e6
    integration_time = 5.e5
    outdir           = "/glade/derecho/scratch/wchapman/qgs_runs/"
    filename         = outdir + ("evol_fields_perturbed.dat" if PERTURB else "evol_fields.dat")

    T_wall = time.process_time()

    # ── model parameters (standard MAOOAM configuration) ──────────────────────
    model_parameters = QgParams()
    model_parameters.set_atmospheric_channel_fourier_modes(2, 2)
    model_parameters.set_oceanic_basin_fourier_modes(2, 4)
    model_parameters.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5,
                                  'r': 1.e-7, 'h': 136.5, 'd': 1.1e-7})
    model_parameters.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3,
                                                      'hlambda': 15.06})
    model_parameters.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})
    model_parameters.atemperature_params.set_insolation(103.3333, 0)
    model_parameters.gotemperature_params.set_insolation(310., 0)

    if print_parameters:
        model_parameters.print_params()

    # ── build tendency function ────────────────────────────────────────────────
    if PERTURB:
        f, Df, E_atm, E_oc = create_exchange_perturbed_tendencies(
            model_parameters,
            eps=PERTURB_EPS,
            g_coeffs=G_COEFFS,
        )
        print(f"\nPerturbation: eps={PERTURB_EPS},  g_coeffs={G_COEFFS}")
        print(f"  E_atm shape: {E_atm.shape},  E_oc shape: {E_oc.shape}")
        print(f"  Lpa  = {float(model_parameters.Lpa):.6f}")
        print(f"  Lpgo = {float(model_parameters.Lpgo):.6f}")
    else:
        f, Df = create_tendencies(model_parameters)

    # ── integrator ────────────────────────────────────────────────────────────
    integrator = RungeKuttaIntegrator()
    integrator.set_func(f)

    # Random initial condition (small amplitude)
    ic = np.random.rand(model_parameters.ndim) * 0.01

    # ── transient integration (spin-up to attractor) ───────────────────────────
    print("Starting transient time integration ...")
    ws = 10000
    y = ic
    total_time = 0.
    t_up = ws * dt / integration_time * 100
    while total_time < transient_time:
        integrator.integrate(0., ws * dt, dt, ic=y, write_steps=0)
        t, y = integrator.get_trajectories()
        total_time += t
        if total_time / transient_time * 100 % 0.1 < t_up:
            print_progress(total_time / transient_time)

    # ── production integration ────────────────────────────────────────────────
    total_time = 0.
    traj = np.insert(y, 0, total_time)[np.newaxis, ...]
    t_up = write_steps * dt / integration_time * 100

    print("Starting the time evolution ...")
    while total_time < integration_time:
        integrator.integrate(0., write_steps * dt, dt, ic=y, write_steps=0)
        t, y = integrator.get_trajectories()
        total_time += t
        traj = np.concatenate((traj, np.insert(y, 0, total_time)[np.newaxis, ...]))
        if total_time / integration_time * 100 % 0.1 < t_up:
            print_progress(total_time / integration_time)

    print(f"\nEvolution finished, writing to {filename}")
    np.savetxt(filename, traj)
    print(f"Time clock: {time.process_time() - T_wall:.1f} s")
