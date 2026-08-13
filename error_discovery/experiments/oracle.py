"""
error_discovery.experiments.oracle
==================================

C1 (reviewer point #8): diagnose the ~20%% low bias of the ocean-temperature
coefficient in the *noiseless* mixed-driver fit (Fig. fig_decomposition).

The injected error tendency is exactly linear in the restricted library:
    m(x) = (c_true . phi(x)) (E x),  phi = [1, x0, ..., x35],
    c_true = eps at the three active modes (psi_a1=x0, theta_a1=x10, T_o1=x28).
So an unregularized, full-rank weighted least squares MUST return c_true to
machine precision in the noiseless limit.  Any residual bias is numerical.

This script runs the reviewer's oracle battery:
  * confirm x64 is on;
  * confirm the target lies in the column space (||A c_true - t|| ~ eps_machine);
  * report singular values / numerical rank / cond(A) of the weighted design;
  * compare solvers: current absolute ridge (1e-12 I), plain normal equations,
    SVD least squares (lstsq), and pseudo-inverse with a relative cutoff;
  * report the recovered T_o1 coefficient and ||c-c_true|| for each.

Read-only on the cached attractor IC; no result files are written.
"""

import numpy as np
import jax

from ..config import EPS
from ..models import TwinModels
from ..formfit import attractor_traj, err_tendency

# active modes in the mixed error g = theta_a1 + T_o1 + psi_a1
PSI_A1, THETA_A1, T_O1 = 0, 10, 28


def _weighted_design(tw, X, m):
    """Return (A, t, w, phi) for the |Ex|^2-weighted scalar regression
    alpha(x) = c . phi(x), alpha_s = (Ex_s . m_s)/|Ex_s|^2, weight w_s = |Ex_s|^2.
    The full objective sum_s sum_i ((c.phi)(Ex)_i - m_i)^2 reduces to
    min_c sum_s w_s (c.phi_s - alpha_s)^2 + const, i.e. min ||A c - t||^2 with
    A = sqrt(w) phi, t = sqrt(w) alpha."""
    n, nd = X.shape
    phi = np.concatenate([np.ones((n, 1)), X], axis=1)          # (n, 1+nd)
    EX = X @ np.asarray(tw.E_full).T
    w = (EX * EX).sum(1)                                        # (n,)
    s = (EX * m).sum(1)                                         # (n,)
    alpha = s / w
    sw = np.sqrt(w)
    A = sw[:, None] * phi
    t = sw * alpha
    return A, t, w, phi


def _c_true_vec(P):
    c = np.zeros(P)
    c[1 + PSI_A1] = EPS
    c[1 + THETA_A1] = EPS
    c[1 + T_O1] = EPS
    return c


def run(record=40000):
    print(f"x64 enabled: {jax.config.read('jax_enable_x64')}")
    tw = TwinModels(eps=EPS, g_coeffs=(0.0, 1.0, 1.0, 1.0))     # const, theta_a1, T_o1, psi_a1
    X = attractor_traj(tw, record)
    m = err_tendency(tw, X)
    A, t, w, phi = _weighted_design(tw, X, m)
    P = A.shape[1]
    c_true = _c_true_vec(P)
    iTo = 1 + T_O1

    # --- exactness of the linear model: is m in the column space at c_true? ---
    res_true = np.linalg.norm(A @ c_true - t) / (np.linalg.norm(t) + 1e-300)
    # also the *full* (vector) residual, including any component of m off (Ex):
    EX = X @ np.asarray(tw.E_full).T
    alpha_true = phi @ c_true
    m_perp = m - alpha_true[:, None] * EX
    perp_frac = np.linalg.norm(m_perp) / (np.linalg.norm(m) + 1e-300)
    print(f"\n||A c_true - t|| / ||t||          = {res_true:.3e}   (in-span check)")
    print(f"||m - alpha_true (Ex)|| / ||m||   = {perp_frac:.3e}   (parallel check)")

    # --- conditioning of the weighted design ---
    sv = np.linalg.svd(A, compute_uv=False)
    eps_mach = np.finfo(A.dtype).eps
    tol = max(A.shape) * eps_mach * sv[0]
    rank = int((sv > tol).sum())
    print(f"\nweighted design A: shape {A.shape}, dtype {A.dtype}")
    print(f"  sigma_max = {sv[0]:.3e}, sigma_min = {sv[-1]:.3e}, "
          f"cond(A) = {sv[0]/sv[-1]:.3e}")
    print(f"  numerical rank (tol={tol:.1e}) = {rank} / {P}")
    print(f"  smallest 5 singular values: {sv[-5:]}")

    # cond of the normal matrix G = A^T A (what the manuscript currently reports)
    G = A.T @ A
    print(f"  cond(G=A^T A) = cond(A)^2 = {(sv[0]/sv[-1])**2:.3e}")

    # --- solver comparison ---
    b = A.T @ t

    def report(name, c):
        rel = np.linalg.norm(c - c_true) / np.linalg.norm(c_true)
        print(f"  {name:32s}  T_o1={c[iTo]:.5f} (truth {EPS})  "
              f"frac low={1 - c[iTo]/EPS:+.1%}  ||c-c_true||/||c_true||={rel:.2e}")

    print("\nsolver comparison (noiseless):")
    c_ridge = np.linalg.solve(G + 1e-12 * np.eye(P), b)         # CURRENT method
    report("current: ridge 1e-12 (absolute)", c_ridge)
    try:
        c_norm = np.linalg.solve(G, b)
        report("plain normal equations", c_norm)
    except np.linalg.LinAlgError as e:
        print(f"  plain normal equations          singular: {e}")
    c_lstsq, *_ = np.linalg.lstsq(A, t, rcond=None)
    report("SVD lstsq (rcond=None)", c_lstsq)
    c_pinv = np.linalg.pinv(A, rcond=1e-10) @ t
    report("pinv (rcond=1e-10)", c_pinv)
    # relative ridge for scale reference
    c_rel = np.linalg.solve(G + 1e-10 * sv[0]**2 * np.eye(P), b)
    report("ridge 1e-10*sigma_max^2 (relative)", c_rel)

    return dict(sv=sv, rank=rank, cond=sv[0]/sv[-1], c_true=c_true,
                c_ridge=c_ridge, c_lstsq=c_lstsq, res_true=res_true,
                perp_frac=perp_frac)


if __name__ == "__main__":
    run()
