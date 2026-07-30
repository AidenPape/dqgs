"""
qgs_jax -- a differentiable JAX re-implementation of the MAOOAM model.

The original ``qgs`` code base builds, from a ``QgParams`` configuration, a
sparse rank-3 tendency tensor ``T`` such that the model ODE is

    dx_i/dt = sum_{j,k} T[i, j, k] * xbar_j * xbar_k,   xbar = (1, x).

``qgs_jax`` re-expresses this in JAX so that whole trajectories are
differentiable -- both with respect to the state (free tangent-linear / adjoint)
and with respect to the physical parameters (for automatic parameter tuning).

Layers
------
* :mod:`qgs_jax.spec`      -- extract a reference tensor + geometry + parameters
                             from the original numpy/numba qgs (run once).
* :mod:`qgs_jax.model`     -- JAX tendencies, Jacobian and RK2/RK4 integrators.
* :mod:`qgs_jax.assemble`  -- pure-JAX, parameter-differentiable re-assembly of
                             the tendency tensor (bit-exact with qgs).

Quick start
-----------
>>> from qgs_jax.spec import extract_spec
>>> from qgs_jax import model, assemble
>>> spec = extract_spec()
>>> geom, theta0, tensor_of_theta = assemble.make_tensor_fn(spec)
>>> import jax.numpy as jnp
>>> T = tensor_of_theta(theta0)              # differentiable in theta0
>>> x0 = jnp.asarray(spec.params.ndim*[0.0]) # ... or a real IC
"""

from . import spec        # noqa: F401
from . import model       # noqa: F401
from . import assemble    # noqa: F401

__all__ = ["spec", "model", "assemble"]
