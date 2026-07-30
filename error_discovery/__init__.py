"""
error_discovery -- differentiable model-based discovery of MAOOAM model error.

A twin experiment: a "truth" model carries a known state-dependent error injected
into the thermal-exchange coupling; a differentiable JAX "discovery" model
(built on :mod:`qgs_jax`) tries to recover that error's coefficients by
backpropagating a forecast loss through the RK4 integrator.

The headline science (see :mod:`error_discovery.experiment`) is the
*optimisation-window / lead-time at which recovery breaks down* -- set by the
chaotic growth of the loss gradient beyond ~1 Lyapunov time.

Quick start
-----------
>>> from error_discovery import experiment, config
>>> experiment.run(config.QUICK)        # fast smoke run
>>> experiment.run()                     # full study
"""

from . import config        # noqa: F401
from . import models         # noqa: F401
from . import discovery      # noqa: F401
from . import experiment     # noqa: F401

__all__ = ["config", "models", "discovery", "experiment"]
