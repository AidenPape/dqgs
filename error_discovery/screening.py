"""
error_discovery.screening
=========================

The |mean|/std diagnostic and the adversarial quadratic library it explains.

A mode that never crosses zero on the sampling distribution is *near-constant*:
its |mean|/std is large. Multiply any candidate driver x[k] by such a mode and you
get back a rescaled copy of x[k] — an alias the fit cannot tell from the truth. So
a single near-constant mode is enough to hide the form of *every* linear driver,
and the best alias reproduces the error to a relative residual of roughly

    1 / max_j(|mean|/std of x_j)

taken over all modes, not over the driver alone. That inequality is the mechanism
behind the reanalysis and DA-increment nulls, and :mod:`experiments.robustness`
tests how far it holds.

:func:`candidate_library` is where the "restricted vs full" distinction lives:
whether the cross-products are in the library at all decides whether a form looks
identifiable, which is why the earlier driver-only claim was an artifact.
"""

import numpy as np

from .config import EPS, THRESH


def exchange_image(tw, X):
    """The exchange-operator image E x for each row of X."""
    return X @ np.asarray(tw.E_full).T


def mean_over_std(X, k):
    """|mean|/std of mode k on sample X: how offset-dominated (near-constant) it is."""
    s = X[:, k].std()
    return float(abs(X[:, k].mean()) / s) if s > 0 else np.inf


def max_mean_over_std(X):
    """The most near-constant mode sets the strongest cross-product alias."""
    return max(mean_over_std(X, j) for j in range(X.shape[1]))


def candidate_library(X, k, kind):
    """Candidate predictor columns g_j(x) and names.

    kind="restricted": const + 36 linear + 36 squares (no cross-products) — the
        physically-motivated set a modeler would propose.
    kind="full": const + 36 linear + all pairwise products (squares and cross
        terms) — the adversarial complete quadratic library.
    """
    n, nd = X.shape
    cols = [np.ones(n)] + [X[:, j] for j in range(nd)]
    names = ["const"] + [f"x{j}" for j in range(nd)]
    if kind == "restricted":
        cols += [X[:, j] ** 2 for j in range(nd)]
        names += [f"x{j}^2" for j in range(nd)]
    elif kind == "full":
        for i in range(nd):
            for j in range(i, nd):
                cols.append(X[:, i] * X[:, j])
                names.append(f"x{i}*x{j}")
    else:
        raise ValueError(f"unknown library kind: {kind!r}")
    return cols, names


def margin_driver(tw, X, k, kind="full"):
    """Form-identifiability margin for the true driver g = x[k] on sample X, against
    the chosen library. Each predictor g_j gets its best single coefficient c_j
    minimizing ||c_j g_j (E x) - m||; R_j is that relative residual. Returns
    (margin, truth_residual, best_wrong_residual)."""
    EX = exchange_image(tw, X)
    sq = (EX * EX).sum(1)                                  # ||E x||^2 per row
    m = EPS * X[:, k][:, None] * EX                        # injected error tendency
    m2 = float((m * m).sum())
    cols, names = candidate_library(X, k, kind)
    R = np.empty(len(cols))
    for i, g in enumerate(cols):
        den = float((g * g * sq).sum())                   # <g E x, g E x>
        num = float((g[:, None] * EX * m).sum())          # <g E x, m>
        c = num / den if den > 0 else 0.0
        R[i] = np.sqrt(max(c * c * den - 2 * c * num + m2, 0.0)) / np.sqrt(m2)
    t = names.index(f"x{k}")
    others = np.delete(R, t)
    return float(others.min() - R[t]), float(R[t]), float(others.min())


def identifiable_count(tw, X, kind, modes=None):
    """How many of the candidate drivers are identifiable on X.

    This applies :data:`config.THRESH` as a hard cut to a continuous quantity, and the
    count it returns is a headline number. See the note on ``THRESH`` in ``config.py``:
    the cutoff is a round 10%, not a derived quantity.
    """
    modes = range(tw.ndim) if modes is None else modes
    margins = np.array([margin_driver(tw, X, k, kind)[0] for k in modes])
    return int((margins > THRESH).sum()), margins
