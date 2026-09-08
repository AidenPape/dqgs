"""
error_discovery.discovery
=========================

The differentiable error-discovery machinery: roll the discovery model forward
through the RK4 integrator, score it against (noisy) observations of the truth,
and recover the error coefficients by backpropagating the forecast loss to the
coefficients.

Because the discovery model class contains the truth exactly
(``f_model(., c_true) == f_true``), the recovery is a nonlinear least-squares
inverse problem with a known global minimum at ``c_true`` and zero residual in
the noiseless limit.  We solve it with Levenberg--Marquardt using the autodiff
residual Jacobian ``J = d(observed trajectory)/dc``, and diagnose breakdown
through ``cond(J^T J)`` -- the conditioning of the parameter -> observation map.

Two limits bound discovery:

* short windows are *under-determined* -- the predictors have not varied enough
  to separate the four coefficients (``cond(J^T J)`` is huge);
* long windows lose identifiability as finite-time sensitivities grow, so with
  finite observation noise the chaotically-amplified directions become
  unrecoverable.  The upper edge of the usable window is the "lead time beyond
  which the error can no longer be discovered".
"""

import numpy as np
import jax
import jax.numpy as jnp

from qgs_jax.model import rk4_step


def rollout(f, x0, dt, n_steps):
    """Integrate autonomous field ``f`` with RK4; return the (n_steps, ndim) trajectory."""
    def step(x, _):
        xn = rk4_step(f, x, dt)
        return xn, xn
    _, traj = jax.lax.scan(step, x0, None, length=n_steps)
    return traj


def obs_indices(mask):
    return np.where(np.asarray(mask) > 0)[0]


def channel_weights(Y_true, oidx):
    """Per-observed-channel whitening weights, 1/std, from the *clean* trajectory.

    The assumed observation-error scale is ``eta * std(channel)`` (see ``add_obs_noise``),
    so dividing each channel's residual by its climatological std makes the implied
    observation-error covariance ``R = eta^2 I``: channels of different amplitude
    contribute on equal terms, and the whitened noise std is ``eta`` in every channel.
    """
    s = np.asarray(Y_true)[:, oidx].std(axis=0)
    s = np.where(s > 0, s, 1.0)
    return 1.0 / s


def make_residual(tw, x0, Y_obs, dt, oidx, wchan=None):
    """Return ``resid(c)`` = flattened observed (model - obs) trajectory. With
    ``wchan`` (per-observed-channel weights) the residual is whitened, r_w = W r, so
    the least-squares problem uses J_w = W J and the implied observation-error
    covariance R = diag(1/wchan^2) (reviewer #6)."""
    n_steps = Y_obs.shape[0]
    oidx = jnp.asarray(oidx)
    Yo = jnp.asarray(Y_obs)[:, oidx]
    wv = None if wchan is None else jnp.asarray(wchan)

    def resid(c):
        traj = rollout(lambda x: tw.f_model(x, c), x0, dt, n_steps)
        diff = traj[:, oidx] - Yo
        if wv is not None:
            diff = diff * wv                      # per-channel whitening (broadcast over time)
        return diff.ravel()

    return resid


def lm_fit(resid, c0, n_iters=15, lam0=1e-2, lam_up=3.0, lam_dn=0.5):
    """Levenberg--Marquardt least squares; returns (c_hat, J_final, cost_hist).
    ``J_final`` is the residual Jacobian evaluated at the converged coefficients, so
    its SVD gives the local conditioning of the parameter -> observation map at the
    solution (at the noiseless twin truth the Gauss--Newton Hessian J^T J is exact)."""
    resid_j = jax.jit(resid)
    jac_j = jax.jit(jax.jacfwd(resid))

    c = jnp.asarray(c0)
    rc = resid_j(c)
    cost = float(0.5 * rc @ rc)
    lam = lam0
    hist = [cost]
    for _ in range(n_iters):
        J = np.asarray(jac_j(c))
        rcn = np.asarray(rc)
        JtJ = J.T @ J
        Jtr = J.T @ rcn
        D = np.diag(np.maximum(np.diag(JtJ), 1e-30))
        # try a step; back off (increase damping) until it decreases the cost
        accepted = False
        for _try in range(8):
            try:
                step = np.linalg.solve(JtJ + lam * D, -Jtr)
            except np.linalg.LinAlgError:
                lam *= lam_up
                continue
            c_new = c + jnp.asarray(step)
            r_new = resid_j(c_new)
            cost_new = float(0.5 * r_new @ r_new)
            if cost_new < cost:
                c, rc, cost = c_new, r_new, cost_new
                lam = max(lam * lam_dn, 1e-12)
                accepted = True
                break
            lam *= lam_up
        hist.append(cost)
        if not accepted:
            break
    J_final = np.asarray(jac_j(c))             # Jacobian at the converged coefficients
    return c, J_final, np.asarray(hist)


def jacobian_spectrum(J):
    """SVD conditioning of the residual Jacobian J (reviewer #2/#3): report the
    singular values directly rather than eigenvalues of J^T J, whose conditioning is
    the square. Returns sigma_max, sigma_min, cond(J)=sigma_max/sigma_min, numerical
    rank at tol = max(shape)*eps*sigma_max, and cond(J^T J)=cond(J)^2."""
    sv = np.linalg.svd(np.asarray(J), compute_uv=False)
    smax = float(sv[0])
    smin = float(sv[-1])
    tol = max(J.shape) * np.finfo(np.asarray(J).dtype).eps * smax
    rank = int((sv > tol).sum())
    cond_J = smax / smin if smin > 0 else np.inf
    return dict(sigma=sv, smax=smax, smin=smin, cond_J=cond_J,
                rank=rank, cond_JtJ=cond_J ** 2)


def residual_jacobian(tw, x0, Y_true, dt, mask, c=None, whiten=False):
    """The residual Jacobian ``d resid / dc`` evaluated at ``c`` (default ``tw.c_true``).

    No optimisation is involved. Because the discovery model class contains the truth, the
    noiseless residual is *exactly* zero at ``c_true``, so ``J^T J`` there is the exact
    Gauss--Newton Hessian -- the conditioning and uncertainty diagnostics can be read off a
    single Jacobian rather than off a converged fit. ``Y_true`` is the clean trajectory:
    observation noise moves the minimiser, but not this Jacobian, which is a property of the
    model and the truth trajectory alone.
    """
    oidx = obs_indices(mask)
    wchan = channel_weights(Y_true, oidx) if whiten else None
    resid = make_residual(tw, x0, Y_true, dt, oidx, wchan=wchan)
    c = tw.c_true if c is None else c
    return np.asarray(jax.jacfwd(resid)(jnp.asarray(c)))


def coefficient_uncertainty(J):
    """Per-coefficient uncertainty from the residual Jacobian, under linear theory.

    ``cov(c_hat) = sigma_obs^2 (J^T J)^-1``, so coefficient i has standard error
    ``sigma_obs * se_unit[i]``. Three quantities, because they answer different questions:

    * ``col_norm[i] = ||J[:, i]||`` -- the information about ``c_i`` *if the other
      coefficients were known*. The restricted standard error is ``sigma_obs / col_norm[i]``.
    * ``se_unit[i] = sqrt([(J^T J)^-1]_ii)`` -- the standard error when all four coefficients
      are fitted together, which is what every study here actually does.
    * ``vif[i] = (col_norm[i] * se_unit[i])^2 >= 1`` -- the variance inflation between the
      two: how much of coefficient i's own signal is spent separating it from the rest of the
      library. 1 means its column is orthogonal to the others.

    ``align[i] = |V[i, -1]|`` is the overlap of coefficient i with the worst-determined right
    singular direction. Unlike ``cond(J)``, which is built from singular *values* and so
    cannot know which coefficient is the true one, all four of these are per-coefficient.
    """
    J = np.asarray(J)
    _, sv, Vt = np.linalg.svd(J, full_matrices=False)
    smax, smin = float(sv[0]), float(sv[-1])
    tol = max(J.shape) * np.finfo(J.dtype).eps * smax
    col_norm = np.linalg.norm(J, axis=0)
    # (J^T J)^-1 = V diag(1/sigma^2) V^T; only its diagonal is needed, and taking it this
    # way never forms J^T J, whose conditioning is the square of J's.
    with np.errstate(divide="ignore", over="ignore"):
        inv_diag = ((Vt ** 2) / (sv[:, None] ** 2)).sum(axis=0)
        se_unit = np.sqrt(inv_diag)
    return dict(col_norm=col_norm, se_unit=se_unit,
                vif=(col_norm * se_unit) ** 2, align=np.abs(Vt[-1]),
                sigma=sv, smax=smax, smin=smin,
                cond_J=smax / smin if smin > 0 else np.inf,
                rank=int((sv > tol).sum()))


def add_obs_noise(Y, oidx, eta, rng):
    """Add Gaussian noise of std ``eta * std(component)`` to the observed columns."""
    Y = np.array(Y, dtype=float)
    if eta <= 0:
        return Y
    s = Y[:, oidx].std(axis=0)
    s[s == 0] = np.mean(s[s > 0]) if np.any(s > 0) else 1.0
    Y[:, oidx] = Y[:, oidx] + rng.standard_normal(Y[:, oidx].shape) * (eta * s)
    return Y


def recover(tw, x0, Y_true, dt, mask, eta=0.0, n_iters=15, noise_seed=0, whiten=False):
    """Run one discovery (optionally with observation noise). Returns a dict. With
    ``whiten=True`` each observed channel's residual is divided by that channel's
    climatological standard deviation, the assumed per-channel observation-error scale
    (noise std = eta * std), so channels of different amplitude contribute on equal
    terms and adding a channel cannot reduce the Fisher information (reviewer #6/#7)."""
    oidx = obs_indices(mask)
    rng = np.random.default_rng(noise_seed)
    Y = add_obs_noise(Y_true, oidx, eta, rng)
    wchan = channel_weights(Y_true, oidx) if whiten else None
    resid = make_residual(tw, x0, Y, dt, oidx, wchan=wchan)
    c_hat, J_final, hist = lm_fit(resid, jnp.zeros(4), n_iters=n_iters)
    c_hat = np.asarray(c_hat)
    c_true = np.asarray(tw.c_true)
    rel_err = float(np.linalg.norm(c_hat - c_true) / (np.linalg.norm(c_true) + 1e-300))
    spec = jacobian_spectrum(J_final)
    unc = coefficient_uncertainty(J_final)
    return dict(c_hat=c_hat, c_true=c_true, rel_err=rel_err,
                cond=spec["cond_JtJ"],          # back-compat: cond(J^T J) = cond(J)^2
                cond_J=spec["cond_J"], smin=spec["smin"], smax=spec["smax"],
                rank=spec["rank"],
                # per-coefficient, where cond() is a single number for the whole fit:
                # se_unit * sigma_obs is coefficient i's standard error (sigma_obs = eta
                # when whitened), and vif is what fitting the other three costs it.
                col_norm=unc["col_norm"], se_unit=unc["se_unit"],
                vif=unc["vif"], align=unc["align"],
                cost0=float(hist[0]), cost_final=float(hist[-1]), eta=float(eta))


# --------------------------------------------------------------------------- #
# Lyapunov time (Benettin, block renormalisation) on the base model.
# This MAOOAM config is only weakly chaotic, so the estimate needs a long run
# and should be read as an order-of-magnitude predictability scale.
# --------------------------------------------------------------------------- #
def lyapunov_time(tw, x0, dt, block=200, n_blocks=300, drop=40, d0=1e-7, seed=0):
    """Largest Lyapunov exponent of the base model; returns (lambda, tau, tau_days)."""
    from .config import steps_to_days
    from qgs_jax import model as jm

    f = jax.jit(lambda y: jm.integrate_final(tw.T_base, y, dt, block))
    rng = np.random.default_rng(seed)
    p = rng.standard_normal(tw.ndim)
    p = d0 * p / np.linalg.norm(p)
    x = jnp.asarray(np.asarray(x0))
    xp = x + jnp.asarray(p)
    logs = []
    for _ in range(n_blocks):
        x = f(x)
        xp = f(xp)
        d = float(jnp.linalg.norm(xp - x))
        logs.append(np.log(d / d0))
        xp = x + (d0 / d) * (xp - x)
    drop = min(drop, n_blocks // 5)          # keep a non-empty post-transient set
    logs = np.asarray(logs[drop:])
    lam = logs.mean() / (block * dt)
    tau = 1.0 / lam if lam > 0 else np.inf
    return lam, tau, steps_to_days(tau / dt)
