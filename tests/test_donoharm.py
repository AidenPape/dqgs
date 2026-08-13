"""
test_donoharm.py
================

An L1/sparsity penalty over the full quadratic library does not escape the
identifiability boundary. Two pinned facts:

* with the true driver in the library, the joint L1 fit recovers it off the
  attractor with a single term, but needs many spurious near-constant terms to do so
  on the attractor (the margin is only 1/max|mean|/std);
* with the true column removed (model-form error), the on-attractor fit selects a
  cross-product alias that is global nonsense off the attractor, while the
  off-attractor fit selects a surrogate that generalizes.

Run:  pytest error_discovery/test_donoharm.py -v
"""

import numpy as np

from error_discovery.experiments import donoharm as D
from error_discovery.formfit import twin, attractor_traj, true_transient, EPS


def _offattractor(tw, n_seg=3):
    return np.concatenate([true_transient(tw, 600, seed=400 + s)[50:550]
                           for s in range(n_seg)], 0)


def test_exact_library_recovers_offattractor_cheaply():
    """With the true term present, the off-attractor L1 fit recovers eps with a
    single active term; the attractor needs many more to reach the same coefficient."""
    tw = twin(); k = tw.idx_To1
    Xoff = _offattractor(tw)
    Phi, EX, w, m, names, ti = D._design(tw, Xoff, k)
    path = D._lasso_path(*D._gram(Phi, EX, w, m))
    assert abs(path[-1]["c"][ti] - EPS) < 0.1 * EPS          # recovers eps
    assert len(path[-1]["active"]) <= 3                       # cheaply (few terms)


def test_misspecification_no_global_surrogate():
    """Drop the true column. The on-attractor fit selects a near-constant-mode alias
    that looks converged in-regime (held-out attractor) but is order-100% wrong on a
    held-out off-attractor trajectory: a confidently-wrong sparse fit. Sparsity cannot
    invent the missing form, so the cross-regime residual is large."""
    tw = twin(); k = tw.idx_To1
    Xfit = attractor_traj(tw, 3000)
    Xin = attractor_traj(tw, 3000)[::-1][:2000]              # held-out attractor
    Xcross = _offattractor(tw, n_seg=4)                      # held-out off-attractor
    Phi, EX, w, m, names, _ = D._design(tw, Xfit, k, drop_true=True)
    sel = D._lasso_path(*D._gram(Phi, EX, w, m))[-1]
    r_in = D._generalize(tw, sel["c"], names, k, Xin)
    r_cross = D._generalize(tw, sel["c"], names, k, Xcross)
    assert "*" in names[sel["active"][0]], "expected a cross-product alias"
    assert r_in < 0.1, f"alias should look converged in-regime ({r_in:.3f})"
    assert r_cross > 0.5, f"alias should fail to generalize cross-regime ({r_cross:.2f})"
