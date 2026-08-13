"""
error_discovery -- differentiable model-based discovery of MAOOAM model error.

A twin experiment: a "truth" model carries a known state-dependent error injected
into the thermal-exchange coupling; a differentiable JAX "discovery" model
(built on :mod:`qgs_jax`) tries to recover that error's coefficients by
backpropagating a forecast loss through the RK4 integrator.

The package is in two halves. **Machinery** lives here:

    config      every knob and constant, plus the nondimensional-time conversions
    models      the twin: base model, true model, and the error family between them
    attractor   the cached spin-up state every study starts from
    discovery   recovery when the error's *form* is known (Levenberg--Marquardt)
    formfit     recovery when it is not (candidate library, margin)
    screening   the |mean|/std diagnostic that says when the form is knowable
    figstyle    DPI, label boxes, and the headless-backend helper

**Studies** live in :mod:`error_discovery.experiments`, one module each, all
answering to ``run()`` + ``figure()``; :mod:`error_discovery.run_all` is the
launcher and its ``STUDIES`` table is the index of what exists.

Quick start
-----------
>>> from error_discovery import config
>>> from error_discovery.experiments import leadtime
>>> leadtime.run(config.QUICK)        # fast smoke run
>>> leadtime.run()                    # full study
"""

from . import config        # noqa: F401
from . import models        # noqa: F401
from . import discovery     # noqa: F401

__all__ = ["config", "models", "discovery"]
