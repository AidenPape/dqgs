"""
perturbed_tendencies.py
=======================

State-dependent heat-exchange error injection for MAOOAM.

Replaces the constant thermal exchange coefficients lambda'_a and lambda'_o by

    lambda'_a,eff(x) = lambda'_a * [1 + eps * g(x)]
    lambda'_o,eff(x) = lambda'_o * [1 + eps * g(x)]

where  g(x) = a0 + a1*theta_{a,1} + a2*T_{o,1} + a3*psi_{a,1}.

The injected error tendency (bad minus true) is:

    d(theta_{a,i})/dt |_err = eps * g(x) * E_atm[i, :] @ x
    d(T_{o,i})/dt     |_err = eps * g(x) * E_oc[i, :]  @ x

where E_atm and E_oc are the (constant) linear operators encoding the
original exchange tendencies.  Both matrices are returned alongside the
perturbed tendency function so they can be reused for diagnostics.

Only the exchange contributions (Lpa and Lpgo terms) are perturbed;
radiation and Newtonian-cooling terms are left unchanged.

Requires: standard MAOOAM atmosphere-ocean configuration, dynamic_T=False.
"""

import numpy as np
from numba import njit

from qgs.functions.tendencies import create_tendencies


def create_exchange_perturbed_tendencies(params, eps=0.1, g_coeffs=(0., 0., 1., 0.)):
    """Build MAOOAM tendencies with a state-dependent exchange-coefficient perturbation.

    Parameters
    ----------
    params : QgParams
        Atmosphere-ocean MAOOAM configuration. dynamic_T must be False.
    eps : float
        Perturbation amplitude epsilon.  Pass 0 for no perturbation (returns
        the base tendency with zero overhead from the exchange matrices).
    g_coeffs : tuple of 4 floats
        (a0, a1, a2, a3) for g(x) = a0 + a1*theta_{a,1} + a2*T_{o,1} + a3*psi_{a,1}.
        Modes are 1-indexed in the LaTeX but 0-indexed internally; "mode 1" maps
        to the first coefficient of each block in the state vector.

    Returns
    -------
    f_perturbed : callable  (numba njit)
        Perturbed tendency  f(t, x) -> dx/dt.
    Df_base : callable  (numba njit)
        Unperturbed Jacobian.  The perturbation Jacobian is not implemented;
        use this for the tangent-linear / TLAD model only if eps is small.
    E_atm : ndarray, shape (n_theta_a, ndim)
        Linear exchange operator for atmospheric temperature modes:
            theta_a exchange tendency = E_atm @ x
    E_oc : ndarray, shape (n_To, ndim)
        Linear exchange operator for oceanic temperature modes:
            T_o exchange tendency = E_oc @ x
    """

    # ── build base tendencies and retrieve inner products ─────────────────────
    f_base, Df_base, (aip, oip, _gip) = create_tendencies(
        params, return_inner_products=True
    )

    if oip is None:
        raise ValueError(
            "Exchange perturbation requires an ocean component (oip is None). "
            "Use set_oceanic_basin_fourier_modes() before calling this function."
        )
    if params.dynamic_T:
        raise NotImplementedError(
            "dynamic_T=True is not supported by this perturbation function."
        )

    # ── index layout ──────────────────────────────────────────────────────────
    # State vector x (0-indexed, length ndim):
    #   psi_a   block: indices [0,                  vr[0])
    #   theta_a block: indices [vr[0],              vr[1])
    #   psi_o   block: indices [vr[1],              vr[2])
    #   T_o     block: indices [vr[2],              vr[3])
    vr   = params.variables_range       # [end_psia, end_thetaa, end_psio, end_To]
    nvar = params.number_of_variables   # [n_psia, n_thetaa, n_psio, n_To]
    ndim = int(params.ndim)

    n_thetaa = int(nvar[1])
    n_To     = int(nvar[3])

    thetaa_start = int(vr[0])   # first theta_a index in x
    To_start     = int(vr[2])   # first T_o index in x

    # ── physical scalars ──────────────────────────────────────────────────────
    Lpa  = float(params.Lpa)   # lambda'_a  (nondim atmosphere exchange coeff)
    Lpgo = float(params.Lpgo)  # lambda'_o  (nondim ocean exchange coeff)
    sc   = float(params.atemperature_params.sc)    # temperature scaling (=1 in standard MAOOAM)
    sig0 = float(params.atmospheric_params.sig0)   # static stability sigma_0

    # ── a_theta = inv(sig0 * a - u),  shape (n_thetaa, n_thetaa) ─────────────
    # This is the matrix that projects the atmospheric temperature equations
    # onto the spectral basis (same computation as in QgsTensor._compute_tensor_dicts).
    a_mat = np.zeros((n_thetaa, n_thetaa))
    for i in range(n_thetaa):
        for j in range(n_thetaa):
            a_mat[i, j] = sig0 * float(aip.a(i, j)) - float(aip.u(i, j))
    a_theta = np.linalg.inv(a_mat)

    # ── U_inv,  shape (n_To, n_To) ────────────────────────────────────────────
    # Inverse of the oceanic temperature mass matrix U.
    U_mat = np.zeros((n_To, n_To))
    for i in range(n_To):
        for j in range(n_To):
            U_mat[i, j] = float(oip.U(i, j))
    U_inv = np.linalg.inv(U_mat)

    # ── exchange matrices ─────────────────────────────────────────────────────
    # E_atm[i, j] = coefficient of x[j] in  theta_dot_{a,i}^exch
    #             = [a_theta @ u]_{i,j} * sc * Lpa        for j in theta_a block
    #             - [a_theta @ s]_{i,j} * Lpa / 2         for j in T_o block
    E_atm = np.zeros((n_thetaa, ndim))
    for i in range(n_thetaa):
        # theta_a contribution  (the u inner product term)
        for j in range(n_thetaa):
            coeff = sum(a_theta[i, k] * float(aip.u(k, j)) for k in range(n_thetaa))
            E_atm[i, thetaa_start + j] += coeff * sc * Lpa
        # T_o contribution  (the s coupling term)
        for j in range(n_To):
            coeff = -sum(a_theta[i, k] * float(aip.s(k, j)) for k in range(n_thetaa))
            E_atm[i, To_start + j] += coeff * Lpa / 2.0

    # E_oc[i, j] = coefficient of x[j] in  T_dot_{o,i}^exch
    #            = [U_inv @ W]_{i,j} * 2 * sc * Lpgo     for j in theta_a block
    #            - Lpgo                                    for j == To_start + i
    E_oc = np.zeros((n_To, ndim))
    for i in range(n_To):
        # theta_a contribution  (the W coupling term, W = s^T)
        for j in range(n_thetaa):
            coeff = sum(U_inv[i, k] * float(oip.W(k, j)) for k in range(n_To))
            E_oc[i, thetaa_start + j] += coeff * 2.0 * sc * Lpgo
        # T_o diagonal contribution
        E_oc[i, To_start + i] = -Lpgo

    # Ensure contiguous C-order layout for Numba
    E_atm = np.ascontiguousarray(E_atm)
    E_oc  = np.ascontiguousarray(E_oc)

    # ── g(x) index lookups (0-indexed into state vector x) ───────────────────
    # "Mode 1" in the LaTeX == index 0 within each block.
    idx_thetaa1 = thetaa_start   # theta_{a,1}
    idx_To1     = To_start       # T_{o,1}
    idx_psia1   = 0              # psi_{a,1}

    a0_f  = float(g_coeffs[0])
    a1_f  = float(g_coeffs[1])
    a2_f  = float(g_coeffs[2])
    a3_f  = float(g_coeffs[3])
    eps_f = float(eps)

    # ── perturbed tendency (numba jit) ────────────────────────────────────────
    @njit
    def f_perturbed(t, x):
        dx = f_base(t, x)

        # scalar state-dependent multiplier  g(x)
        gx = (a0_f
              + a1_f * x[idx_thetaa1]
              + a2_f * x[idx_To1]
              + a3_f * x[idx_psia1])
        eg = eps_f * gx

        # atmospheric temperature error:  delta(theta_dot_{a,i}) = eg * E_atm[i,:] @ x
        for i in range(n_thetaa):
            exch = 0.0
            for j in range(ndim):
                exch += E_atm[i, j] * x[j]
            dx[thetaa_start + i] += eg * exch

        # oceanic temperature error:  delta(T_dot_{o,i}) = eg * E_oc[i,:] @ x
        for i in range(n_To):
            exch = 0.0
            for j in range(ndim):
                exch += E_oc[i, j] * x[j]
            dx[To_start + i] += eg * exch

        return dx

    return f_perturbed, Df_base, E_atm, E_oc
