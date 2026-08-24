"""
error_discovery.models
======================

The twin-experiment models, in JAX.

* **Truth model** (data-generating): the base MAOOAM tendency plus the injected
  state-dependent exchange error

      f_true(x) = f_base(x) + eps * g_true(x) * (E @ x),

  with g_true(x) = a0 + a1*theta_{a,1} + a2*T_{o,1} + a3*psi_{a,1} and E the
  constant linear exchange operator (E_atm on the theta_a rows, E_oc on the
  T_o rows; the *same* scalar g multiplies both blocks).

* **Discovery model**: the base MAOOAM tendency plus a *learnable* correction of
  the same functional family,

      f_model(x; c) = f_base(x) + (c . phi(x)) * (E @ x),
      phi(x) = (1, theta_{a,1}, T_{o,1}, psi_{a,1}),

  so the truth is recovered exactly at c_true = eps * (a0, a1, a2, a3).  Which three
  state components fill the non-constant slots of phi is a constructor argument
  (``feature_idx``); the default is the three mode-1 predictors above.  The
  whole point of the study is whether gradient descent through the integrator
  can *find* c_true, and how that ability decays with the optimisation window.

The base tendency tensor and the exchange operator are extracted once from the
original numpy/numba ``qgs`` (via :mod:`qgs_jax.spec` and the validated
``create_exchange_perturbed_tendencies``); the JAX truth model is checked
against the numpy ``f_perturbed`` to machine precision in the test-suite.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from qgs_jax.spec import extract_spec                                  # noqa: E402
from qgs_jax import model as jmodel                                    # noqa: E402
from .perturbed_tendencies import (                                    # noqa: E402
    create_exchange_perturbed_tendencies,
)


class TwinModels:
    """Container for the base / truth / discovery tendency functions."""

    def __init__(self, eps, g_coeffs, params=None, feature_idx=None):
        self.eps = float(eps)
        self.g_coeffs = tuple(float(a) for a in g_coeffs)

        spec = extract_spec(params)
        self.spec = spec
        self.ndim = int(spec.ndim)
        vr = spec.variables_range

        # 0-based indices of the three predictor modes (derived, never hardcoded).
        self.idx_psia1 = 0
        self.idx_thetaa1 = int(vr[0])
        self.idx_To1 = int(vr[2])

        # Which state components fill slots 1-3 of phi(x); slot 0 is the constant.
        # Defaults to the three mode-1 predictors, so leaving this alone reproduces the
        # library every other study uses. Overriding it lets a twin be built over a
        # different (e.g. high-variability) mode *without* changing the dimension of c,
        # which is what keeps ``discovery.lm_fit``'s zeros(4) and every c_hats[..., 4]
        # shape untouched. Note that ``f_pert_np`` below is the numpy truth built for the
        # mode-1 library, so the JAX-vs-numpy cross-check in tests covers the default only.
        self.feature_idx = ((self.idx_thetaa1, self.idx_To1, self.idx_psia1)
                            if feature_idx is None else tuple(int(i) for i in feature_idx))

        # Exchange operator from the validated numpy implementation (run once).
        _f_pert_np, _Df, E_atm, E_oc = create_exchange_perturbed_tendencies(
            spec.params, eps=self.eps, g_coeffs=self.g_coeffs
        )
        self.f_pert_np = _f_pert_np            # numpy truth, for validation
        n_ta, n_to = E_atm.shape[0], E_oc.shape[0]
        E_full = np.zeros((self.ndim, self.ndim))
        E_full[vr[0]:vr[0] + n_ta, :] = E_atm   # theta_a rows
        E_full[vr[2]:vr[2] + n_to, :] = E_oc    # T_o rows
        self.E_full = jnp.asarray(E_full)

        self.T_base = jnp.asarray(spec.dense_tensor)
        self.c_true = jnp.asarray(self.eps * np.asarray(self.g_coeffs, dtype=float))

    # -- predictors / features --------------------------------------------- #
    def features(self, x):
        """phi(x) = (1, x[i1], x[i2], x[i3])  (order matches g_coeffs).

        By default the three predictors are theta_{a,1}, T_{o,1}, psi_{a,1}; see
        ``feature_idx``.
        """
        i1, i2, i3 = self.feature_idx
        return jnp.array([1.0, x[i1], x[i2], x[i3]])

    # -- tendencies -------------------------------------------------------- #
    def f_base(self, x):
        return jmodel.tendencies(self.T_base, x)

    def f_model(self, x, c):
        """Base tendency plus learnable correction ``(c . phi(x)) * (E @ x)``."""
        g = jnp.dot(c, self.features(x))
        return self.f_base(x) + g * (self.E_full @ x)

    def f_true(self, x):
        """Data-generating tendency = ``f_model`` at the true coefficients."""
        return self.f_model(x, self.c_true)


def obs_mask(ndim, vr, kind):
    """Boolean (0/1) mask over the state for the observation operator.

    Blocks (default MAOOAM config, vr=(10,20,28,36)):
        psi_a [0:vr0]  theta_a [vr0:vr1]  psi_o [vr1:vr2]  T_o [vr2:vr3].

    kind:
        "full"  -- all 36 variables
        "atmos" -- atmosphere only (psi_a + theta_a)
        "ocean" -- ocean only (psi_o + T_o)
        "temp"  -- temperatures only (theta_a + T_o)
    """
    m = np.zeros(ndim)
    if kind == "full":
        m[:] = 1.0
    elif kind == "atmos":
        m[: vr[1]] = 1.0                  # psi_a + theta_a
    elif kind == "ocean":
        m[vr[1]:] = 1.0                   # psi_o + T_o
    elif kind == "temp":
        m[vr[0]:vr[1]] = 1.0              # theta_a
        m[vr[2]:vr[3]] = 1.0             # T_o
    else:
        raise ValueError(f"unknown obs kind: {kind!r}")
    return jnp.asarray(m)
