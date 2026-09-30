"""
qgs_jax.spec
============

Extraction layer that bridges the original numpy/numba ``qgs`` code base and the
JAX re-implementation.

The whole MAOOAM model (with ``dynamic_T = False``) reduces to a single sparse
rank-3 "tendency tensor" :math:`\\mathcal{T}` such that

.. math:: \\dot x_i = \\sum_{j,k} \\mathcal{T}_{i,j,k}\\, \\bar x_j\\, \\bar x_k,
          \\qquad \\bar x = (1, x_1, \\dots, x_n).

``spec.py`` builds a reference ``QgParams`` object, runs the original
``create_tendencies`` and exposes two things:

* a **dense** copy of the tendency tensor (for an exact, assembly-independent
  1:1 check of the JAX model), and
* the **geometry** (inner-product) arrays plus the **scalar physical
  parameters**, packaged so that the pure-JAX assembly in :mod:`qgs_jax.assemble`
  can rebuild that very tensor as a differentiable function of the parameters.

Everything here is plain numpy -- it runs once at set-up time.  The JAX code
never calls back into this module.
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from qgs.params.params import QgParams
from qgs.functions.tendencies import create_tendencies


# Reference MAOOAM parameter set (Vannitsem et al. 2015 / De Cruz et al. 2016),
# identical to upstream qgs's qgs_maooam.py example. The "default" set below; its values and
# source are tabulated in QGS_JAX_README.md, "Parameter sets".
DEFAULT_MAOOAM_PARAMS = dict(
    kd=0.0290, kdp=0.0290, n=1.5, r=1.0e-7, h=136.5, d=1.1e-7,
)
DEFAULT_ATEMP_PARAMS = dict(eps=0.7, T0=289.3, hlambda=15.06)
DEFAULT_GOTEMP_PARAMS = dict(gamma=5.6e8, T0=301.46)
DEFAULT_A_INSOLATION = (103.3333, 0)
DEFAULT_O_INSOLATION = (310.0, 0)


def _dense(arr):
    """Return a dense float64 numpy array from a sparse.COO / ndarray / scalar."""
    if hasattr(arr, "todense"):
        arr = arr.todense()
    return np.asarray(arr, dtype=np.float64)


# Named parameter sets, so a study can ask for a published configuration by name. "default"
# is the set above and stays the default everywhere. Parameters a set does not list take
# qgs's defaults; in particular gamma_a = 1e7 J m^-2 K^-1 in all of them.
_KD_DECRUZ = 3.0e-6 / 1.032e-4          # k_d = k'_d = 3.0e-6 s^-1 at f0 = 1.032e-4 s^-1

PARAM_SETS = dict(
    default=dict(
        atm_modes=(2, 2), oc_modes=(2, 4),
        maooam=DEFAULT_MAOOAM_PARAMS, atemp=DEFAULT_ATEMP_PARAMS, gotemp=DEFAULT_GOTEMP_PARAMS,
        a_insolation=DEFAULT_A_INSOLATION, o_insolation=DEFAULT_O_INSOLATION),
    # De Cruz et al. 2018 (NPG 25, 387), experiment "nodissip": their Table 3 plus d = 1.1e-7.
    # Their smallest resolution is atmosphere 2x4 + ocean 2x4, ndim 56, not our 2x2
    # atmosphere. Table 3 is De Cruz et al. 2016's Table 1, which differs from "default" in
    # gamma_o (5.46e8, not 5.6e8) and in k_d converted to 0.02907 rather than rounded to
    # 0.0290. They integrate at dt = 0.2; a caller using dt = 0.1 is not matching that.
    decruz2018_nodissip=dict(
        atm_modes=(2, 4), oc_modes=(2, 4),
        maooam=dict(kd=_KD_DECRUZ, kdp=_KD_DECRUZ, n=1.5, r=1.0e-7, h=136.5, d=1.1e-7),
        atemp=dict(eps=0.7, T0=289.30, hlambda=15.06),
        gotemp=dict(gamma=5.46e8, T0=301.46),
        a_insolation=(310.0 / 3.0, 0), o_insolation=(310.0, 0)),
    # Tondeur et al. 2020 (J. Stat. Phys. 179, 1161), Tables 1-2, read literally. Their k and
    # kp, "the friction coefficients at the bottom of the atmosphere [and] between internal
    # atmospheric layers", are taken as qgs's kd and kdp, which are De Cruz et al. 2016's
    # k_d and k'_d. No factor-2 convention is stated anywhere in their text. C_atm = C_ocn =
    # 310 W m^-2 as their Table 2 prints, not "default"'s C_o / 3. n, r and eps_a are not in
    # their tables and take De Cruz et al. 2016's values. Their parameter reference,
    # Vannitsem 2015 GRL, was not accessible, so the literal reading of k and C_atm is not
    # verified against the code they ran.
    tondeur2020_36wk=dict(
        atm_modes=(2, 2), oc_modes=(2, 4),
        maooam=dict(kd=0.010, kdp=0.020, n=1.5, r=1.0e-7, h=165.0, d=6e-8),
        atemp=dict(eps=0.7, T0=289.0, hlambda=10.0),
        gotemp=dict(gamma=6.6e8, T0=301.0),
        a_insolation=(310.0, 0), o_insolation=(310.0, 0)),
    tondeur2020_36st=dict(
        atm_modes=(2, 2), oc_modes=(2, 4),
        maooam=dict(kd=0.0145, kdp=0.029, n=1.5, r=1.0e-7, h=165.0, d=9e-8),
        atemp=dict(eps=0.7, T0=290.20, hlambda=15.06),
        gotemp=dict(gamma=6.6e8, T0=299.35),
        a_insolation=(310.0, 0), o_insolation=(310.0, 0)),
)


def build_reference_params(name="default"):
    """Build a named MAOOAM :class:`QgParams` from :data:`PARAM_SETS`.

    "default" is the canonical configuration: atmosphere 2x2 + ocean 2x4.
    """
    s = PARAM_SETS[name]
    p = QgParams()
    p.set_atmospheric_channel_fourier_modes(*s["atm_modes"])
    p.set_oceanic_basin_fourier_modes(*s["oc_modes"])
    p.set_params(s["maooam"])
    p.atemperature_params.set_params(s["atemp"])
    p.gotemperature_params.set_params(s["gotemp"])
    p.atemperature_params.set_insolation(*s["a_insolation"])
    p.gotemperature_params.set_insolation(*s["o_insolation"])
    return p


@dataclass
class ModelSpec:
    """Everything the JAX side needs, in pure numpy / python containers."""

    ndim: int
    nvar: tuple                       # (n_psi_a, n_theta_a, n_psi_o, n_deltaT_o)
    variables_range: tuple
    dense_tensor: np.ndarray          # (ndim+1, ndim+1, ndim+1), reference tendency tensor
    geom: Dict[str, np.ndarray]       # fixed inner-product arrays
    theta: Dict[str, float]           # tunable scalar physical parameters
    # Keep the numpy callables around for tests / convenience.
    f_numpy: object = field(default=None, repr=False)
    Df_numpy: object = field(default=None, repr=False)
    params: object = field(default=None, repr=False)


def _dense_tensor_from_qgtensor(agotensor, ndim):
    """Reconstruct a dense (ndim+1)^3 array from the qgs sparse COO tensor."""
    coo = agotensor.tensor.coords          # shape (3, nnz)
    val = agotensor.tensor.data
    T = np.zeros((ndim + 1, ndim + 1, ndim + 1), dtype=np.float64)
    # The qgs tensor is "simplified" (upper-triangularised in axes 1,2); that is
    # fine -- it contracts to the same tendencies.  We store it verbatim.
    np.add.at(T, (coo[0], coo[1], coo[2]), val)
    return T


def extract_spec(params=None):
    """Build a :class:`ModelSpec` from a MAOOAM ``QgParams`` configuration.

    Parameters
    ----------
    params : QgParams, optional
        A fully-specified MAOOAM configuration.  Defaults to
        :func:`build_reference_params`.

    Returns
    -------
    ModelSpec
    """
    if params is None:
        params = build_reference_params()

    if params.dynamic_T or params.T4:
        raise NotImplementedError("qgs_jax targets the dynamic_T=False MAOOAM config.")
    if params.ground_params is not None:
        raise NotImplementedError("qgs_jax targets the atmosphere+ocean MAOOAM config (no ground).")

    f_np, Df_np, (aip, oip, gip), agotensor = create_tendencies(
        params, return_inner_products=True, return_qgtensor=True
    )

    ndim = params.ndim
    nvar = tuple(params.number_of_variables)
    vr = tuple(params.variables_range)

    # --- geometry: inner-product arrays (fixed; depend only on truncation) ----
    geom = dict(
        # atmosphere
        a=_dense(aip._a),       # (na, na)
        u=_dense(aip._u),       # (na, na)
        c=_dense(aip._c),       # (na, na)
        b=_dense(aip._b),       # (na, na, na)
        g=_dense(aip._g),       # (na, na, na)
        d=_dense(aip._d),       # (na, no)
        s=_dense(aip._s),       # (na, no)
        # ocean
        K=_dense(oip._K),       # (no, na)
        W=_dense(oip._W),       # (no, na)
        M=_dense(oip._M),       # (no, no)
        U=_dense(oip._U),       # (no, no)
        N=_dense(oip._N),       # (no, no)
        O=_dense(oip._O),       # (no, no, no)
        C=_dense(oip._C),       # (no, no, no)
    )

    # --- scalar physical parameters (the things one would tune) ---------------
    ap = params.atmospheric_params
    op = params.oceanic_params
    scp = params.scale_params
    atp = params.atemperature_params

    theta = dict(
        sig0=float(ap.sig0),
        kd=float(ap.kd),
        kdp=float(ap.kdp),
        beta=float(scp.beta),
        od=float(op.d),
        orr=float(op.r),
        G=float(params.G),
        Lpa=float(params.Lpa),
        Lpgo=float(params.Lpgo),
        LSBpa=float(params.LSBpa),
        LSBpgo=float(params.LSBpgo),
        sbpa=float(params.sbpa),
        sbpgo=float(params.sbpgo),
        sc=float(atp.sc),
        Cpa0=float(np.asarray(params.Cpa, dtype=float)[0]),
        Cpgo0=float(np.asarray(params.Cpgo, dtype=float)[0]),
    )
    # Full Cpa / Cpgo vectors (only component 0 is non-zero in the reference
    # config, but keep the vectors so the assembly is general).
    geom["Cpa"] = np.asarray(params.Cpa, dtype=float)
    geom["Cpgo"] = np.asarray(params.Cpgo, dtype=float)

    dense_tensor = _dense_tensor_from_qgtensor(agotensor, ndim)

    return ModelSpec(
        ndim=ndim,
        nvar=nvar,
        variables_range=vr,
        dense_tensor=dense_tensor,
        geom=geom,
        theta=theta,
        f_numpy=f_np,
        Df_numpy=Df_np,
        params=params,
    )


if __name__ == "__main__":
    spec = extract_spec()
    print("ndim          :", spec.ndim)
    print("nvar          :", spec.nvar)
    print("tensor shape  :", spec.dense_tensor.shape)
    print("tensor nnz    :", int(np.count_nonzero(spec.dense_tensor)))
    print("theta keys    :", sorted(spec.theta))
    print("geom keys     :", sorted(spec.geom))
