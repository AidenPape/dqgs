"""
qgs_jax.assemble
================

Pure-JAX re-implementation of the MAOOAM tendency-tensor assembly.

This is the piece that makes the model *differentiable with respect to its
physical parameters*.  It reproduces, term by term, the ``stored``-branch of
``qgs.tensors.qgtensor.QgsTensor._compute_tensor_dicts`` for the standard
atmosphere + ocean MAOOAM configuration (``dynamic_T = False``, no ground, no
orography), but every scalar parameter flows through as a traced JAX value.

The only parameter-dependent linear-algebra are four small matrix inverses,

* ``a_inv   = inv(a)``
* ``a_theta = inv(sig0 * a - u)``           (depends on ``sig0``)
* ``M_psio  = inv(M + G * U)``              (depends on ``G``)
* ``U_inv   = inv(U)``

all handled with ``jnp.linalg.inv`` so gradients flow through ``sig0`` and ``G``
as well.

The assembled tensor contracts to *exactly* the same tendencies as the original
qgs tensor (verified to machine precision in the test-suite).  Note the original
tensor is stored "simplified" (upper-triangularised over the last two axes);
the tensor built here is the un-simplified equivalent -- identical under
contraction, and identical after symmetrising over axes (1, 2).
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# Parameters that the assembly understands.  ``theta`` dicts passed to
# :func:`assemble_tensor` must provide all of these keys (see
# ``qgs_jax.spec.ModelSpec.theta``).
THETA_KEYS = (
    "sig0", "kd", "kdp", "beta", "od", "orr", "G",
    "Lpa", "Lpgo", "LSBpa", "LSBpgo", "sbpa", "sbpgo", "sc",
    "Cpa0", "Cpgo0",
)


def _index_maps(nvar, vr):
    """Return python helpers mapping a variable index to its tensor row/col."""
    na, _, no_psi, no_T = nvar
    psi_a = lambda i: i + 1
    theta_a = lambda i: i + vr[0] + 1
    psi_o = lambda i: i + vr[1] + 1
    deltaT_o = lambda i: i + vr[2] + 1
    return psi_a, theta_a, psi_o, deltaT_o


def assemble_tensor(geom, theta, nvar, vr):
    """Build the dense MAOOAM tendency tensor as a function of parameters.

    Parameters
    ----------
    geom : dict of jnp arrays
        Fixed inner-product arrays: ``a, u, c, b, g, d, s`` (atmosphere),
        ``K, W, M, U, N, O, C`` (ocean), and the insolation vectors
        ``Cpa, Cpgo``.  (See :func:`qgs_jax.spec.extract_spec`.)
    theta : dict
        Scalar physical parameters (see :data:`THETA_KEYS`).
    nvar : tuple(int)
        ``(n_psi_a, n_theta_a, n_psi_o, n_deltaT_o)``.
    vr : tuple(int)
        ``variables_range`` from ``QgParams``.

    Returns
    -------
    (ndim+1, ndim+1, ndim+1) jnp array
        The tendency tensor ``T`` with ``f(x)_i = sum_jk T[i,j,k] xbar_j xbar_k``.
    """
    na = nvar[0]
    no = nvar[2]
    ndim = vr[-1]
    n1 = ndim + 1

    psi_a, theta_a, psi_o, deltaT_o = _index_maps(nvar, vr)

    # --- geometry arrays ------------------------------------------------------
    a = geom["a"]; u = geom["u"]; c = geom["c"]; b = geom["b"]; g = geom["g"]
    d = geom["d"]; s = geom["s"]
    K = geom["K"]; W = geom["W"]; M = geom["M"]; U = geom["U"]; N = geom["N"]
    O = geom["O"]; C = geom["C"]
    # Insolation vectors: scale the (already-shaped) reference vector by the
    # tunable component-0 amplitude so Cpa0 / Cpgo0 are differentiable while
    # the spatial structure stays fixed.
    Cpa = geom["Cpa_unit"] * theta["Cpa0"]
    Cpgo = geom["Cpgo_unit"] * theta["Cpgo0"]

    # --- scalar parameters ----------------------------------------------------
    sig0 = theta["sig0"]; kd = theta["kd"]; kdp = theta["kdp"]; beta = theta["beta"]
    od = theta["od"]; orr = theta["orr"]; G = theta["G"]
    Lpa = theta["Lpa"]; Lpgo = theta["Lpgo"]
    LSBpa = theta["LSBpa"]; LSBpgo = theta["LSBpgo"]
    sbpa = theta["sbpa"]; sbpgo = theta["sbpgo"]; sc = theta["sc"]

    # --- parameter-dependent inverses (differentiable) ------------------------
    a_inv = jnp.linalg.inv(a)
    a_theta = jnp.linalg.inv(sig0 * a - u)
    M_psio = jnp.linalg.inv(M + G * U)
    U_inv = jnp.linalg.inv(U)

    rows = []          # static int indices
    cols1 = []
    cols2 = []
    vals = []          # traced jnp scalars

    def add(i, j, k, v):
        rows.append(i); cols1.append(j); cols2.append(k); vals.append(v)

    delta = lambda i, j: 1.0 if i == j else 0.0

    # ======================= psi_a part ======================================
    for i in range(na):
        ri = psi_a(i)
        for j in range(na):
            val = a_inv[i, :] @ c[:, j]
            add(ri, psi_a(j), 0, -val * beta)
            add(ri, psi_a(j), 0, -(kd * delta(i, j)) / 2)
            add(ri, theta_a(j), 0, (kd * delta(i, j)) / 2)
            for k in range(na):
                val = a_inv[i, :] @ b[:, j, k]
                add(ri, psi_a(j), psi_a(k), -val)
                add(ri, theta_a(j), theta_a(k), -val)
        for j in range(no):
            val = a_inv[i, :] @ d[:, j]
            add(ri, psi_o(j), 0, val * kd / 2)

    # ======================= theta_a part ====================================
    for i in range(na):
        ri = theta_a(i)
        add(ri, 0, 0, -(a_theta[i, :] @ (u @ Cpa)))
        for j in range(na):
            val = a_theta[i, :] @ a[:, j]
            add(ri, psi_a(j), 0, val * kd * sig0 / 2)
            add(ri, theta_a(j), 0, -val * (kd / 2 + 2 * kdp) * sig0)

            val = -(a_theta[i, :] @ c[:, j])
            add(ri, theta_a(j), 0, val * beta * sig0)

            for k in range(na):
                valb = a_theta[i, :] @ b[:, j, k]
                add(ri, psi_a(j), theta_a(k), -valb * sig0)
                add(ri, theta_a(j), psi_a(k), -valb * sig0)
                valg = a_theta[i, :] @ g[:, j, k]
                add(ri, psi_a(j), theta_a(k), valg)
        for j in range(na):
            val = a_theta[i, :] @ u[:, j]
            add(ri, theta_a(j), 0, val * sc * Lpa)
            add(ri, theta_a(j), 0, val * LSBpa)
        for j in range(no):
            val = -(a_theta[i, :] @ d[:, j])
            add(ri, psi_o(j), 0, val * sig0 * kd / 2)
        for j in range(no):
            val = -(a_theta[i, :] @ s[:, j])
            add(ri, deltaT_o(j), 0, val * Lpa / 2)
            add(ri, deltaT_o(j), 0, val * LSBpgo)

    # ======================= psi_o part ======================================
    for i in range(no):
        ri = psi_o(i)
        for j in range(na):
            val = (M_psio[i, :] @ K[:, j]) * od
            add(ri, psi_a(j), 0, val)
            add(ri, theta_a(j), 0, -val)
        for j in range(no):
            val = -(M_psio[i, :] @ N[:, j])
            add(ri, psi_o(j), 0, val * beta)
            val = -(M_psio[i, :] @ M[:, j])
            add(ri, psi_o(j), 0, val * (orr + od))
            for k in range(no):
                add(ri, psi_o(j), psi_o(k), -(M_psio[i, :] @ C[:, j, k]))

    # ======================= deltaT_o part ===================================
    for i in range(no):
        ri = deltaT_o(i)
        add(ri, 0, 0, U_inv[i, :] @ (W @ Cpgo))
        for j in range(na):
            val = U_inv[i, :] @ W[:, j]
            add(ri, theta_a(j), 0, val * 2 * sc * Lpgo)
            add(ri, theta_a(j), 0, val * sbpa)
        for j in range(no):
            add(ri, deltaT_o(j), 0, -Lpgo * delta(i, j))
            add(ri, deltaT_o(j), 0, -sbpgo * delta(i, j))
        for j in range(no):
            for k in range(no):
                add(ri, psi_o(j), deltaT_o(k), -(U_inv[i, :] @ O[:, j, k]))

    I = np.asarray(rows, dtype=np.int32)
    J = np.asarray(cols1, dtype=np.int32)
    Kk = np.asarray(cols2, dtype=np.int32)
    V = jnp.stack(vals)

    T = jnp.zeros((n1, n1, n1), dtype=V.dtype)
    T = T.at[I, J, Kk].add(V)
    return T


def prepare_geom(geom_np):
    """Convert a numpy ``geom`` dict (from spec) into jnp arrays for assembly.

    Adds unit insolation vectors so that ``Cpa0`` / ``Cpgo0`` can be tuned as
    scalar amplitudes of the fixed spatial pattern.
    """
    out = {}
    for key, arr in geom_np.items():
        out[key] = jnp.asarray(arr)
    # Reference component-0 amplitudes -> unit patterns.
    cpa = np.asarray(geom_np["Cpa"], dtype=float)
    cpgo = np.asarray(geom_np["Cpgo"], dtype=float)
    a0 = cpa[0] if cpa[0] != 0 else 1.0
    g0 = cpgo[0] if cpgo[0] != 0 else 1.0
    out["Cpa_unit"] = jnp.asarray(cpa / a0)
    out["Cpgo_unit"] = jnp.asarray(cpgo / g0)
    return out


def make_tensor_fn(spec):
    """Return ``(geom_jnp, theta0, tensor_of_theta)`` for a given ModelSpec.

    ``tensor_of_theta(theta_dict)`` -> dense JAX tensor, differentiable in the
    physical parameters.
    """
    geom_jnp = prepare_geom(spec.geom)
    theta0 = {k: jnp.asarray(float(spec.theta[k])) for k in THETA_KEYS}
    nvar = spec.nvar
    vr = spec.variables_range

    def tensor_of_theta(theta):
        return assemble_tensor(geom_jnp, theta, nvar, vr)

    return geom_jnp, theta0, tensor_of_theta
