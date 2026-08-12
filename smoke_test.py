"""End-to-end check: build the perturbed MAOOAM tendencies and integrate them.

``RungeKuttaIntegrator.set_func`` starts a multiprocessing pool, so the body must sit
behind the entry-point guard: macOS spawns rather than forks, and without the guard each
child re-imports this module, reaches ``set_func`` and starts a pool of its own.
"""

import numpy as np
from qgs.params.params import QgParams
from qgs.integrators.integrator import RungeKuttaIntegrator
from error_discovery.perturbed_tendencies import create_exchange_perturbed_tendencies


def main():
    np.random.seed(21217)

    model_parameters = QgParams()
    model_parameters.set_atmospheric_channel_fourier_modes(2, 2)
    model_parameters.set_oceanic_basin_fourier_modes(2, 4)
    model_parameters.set_params({'kd': 0.0290, 'kdp': 0.0290, 'n': 1.5,
                                  'r': 1.e-7, 'h': 136.5, 'd': 1.1e-7})
    model_parameters.atemperature_params.set_params({'eps': 0.7, 'T0': 289.3, 'hlambda': 15.06})
    model_parameters.gotemperature_params.set_params({'gamma': 5.6e8, 'T0': 301.46})
    model_parameters.atemperature_params.set_insolation(103.3333, 0)
    model_parameters.gotemperature_params.set_insolation(310., 0)

    print(f"ndim = {model_parameters.ndim}")
    print(f"variables_range = {model_parameters.variables_range}")
    print(f"number_of_variables = {model_parameters.number_of_variables}")

    f, Df, E_atm, E_oc = create_exchange_perturbed_tendencies(
        model_parameters, eps=0.1, g_coeffs=(0., 0., 1., 0.)
    )
    print(f"E_atm shape: {E_atm.shape},  E_oc shape: {E_oc.shape}")
    print(f"Lpa  = {float(model_parameters.Lpa):.6f}")
    print(f"Lpgo = {float(model_parameters.Lpgo):.6f}")
    print()

    ic = np.random.rand(model_parameters.ndim) * 0.01
    integrator = RungeKuttaIntegrator()
    integrator.set_func(f)

    print("Running short transient (1000 steps)...")
    integrator.integrate(0., 1000 * 0.1, 0.1, ic=ic, write_steps=0)
    t, y = integrator.get_trajectories()
    print(f"  state norm = {np.linalg.norm(y):.4f}")

    print("Running short production (100 steps, write every 10)...")
    integrator.integrate(0., 100 * 0.1, 0.1, ic=y, write_steps=10)
    t, traj = integrator.get_trajectories()
    print(f"  trajectory shape: {traj.shape}")
    print(f"  any NaN: {np.any(np.isnan(traj))}")
    print(f"  any Inf: {np.any(np.isinf(traj))}")

    print()
    print("E_atm[0,:] nonzero cols:", np.where(E_atm[0,:] != 0)[0])
    print("E_oc[0,:]  nonzero cols:", np.where(E_oc[0,:]  != 0)[0])
    print()
    print("SMOKE TEST PASSED")

    integrator.terminate()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
