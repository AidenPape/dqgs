#!/usr/bin/env python

import numpy as np
from qgs.params.params import QgParams
from qgs.integrators.integrator import RungeKuttaIntegrator
from qgs.functions.perturbed_tendencies_demo import create_perturbed_tendencies
import time

# Initializing the random number generator (for reproducibility). -- Disable if needed.
np.random.seed(21217)

# Time parameters
dt = 0.1
# Saving the model state n steps
write_steps = 100
# transient time to attractor
transient_time = 1.e4
# integration time on the attractor
integration_time = 1.e3
# file where to write the output
filename = "evol_fields_demo.dat"
T = time.process_time()

# Setting MAOOAM parameters according to the publication linked above
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
print(f"State vector layout: {model_parameters.variables_range}")

# build the perturbed tendency 
f, Df = create_perturbed_tendencies(model_parameters, eps=0.1)

# integrator setup
integrator = RungeKuttaIntegrator()
integrator.set_func(f)
ic = np.random.rand(model_parameters.ndim) * 0.01

# transient
print("Transient integration...")
y = ic
total_time = 0.
while total_time < transient_time:
    integrator.integrate(0., 1000 * dt, dt, ic=y, write_steps=0)
    t, y = integrator.get_trajectories()
    total_time += t

# Now integrate to obtain a trajectory on the attractor
print("Production integration...")
total_time = 0.
traj = np.insert(y, 0, total_time)[np.newaxis, ...]

while total_time < integration_time:
    integrator.integrate(0., write_steps * dt, dt, ic=y, write_steps=0)
    t, y = integrator.get_trajectories()
    total_time += t
    traj = np.concatenate((traj, np.insert(y, 0, total_time)[np.newaxis, ...]))

print(f"Done. Trajectory shape: {traj.shape}")
print(f"Writing to {filename}")
np.savetxt(filename, traj)