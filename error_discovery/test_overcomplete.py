"""
test_overcomplete.py
====================

The over-complete-library complement to the boundary result: against the full
quadratic library, form identifiability is a property of the sampling distribution,
not of the library. On the attractor the complete library identifies (almost) no
form; on a dynamically-accessible off-attractor ensemble that drives every mode
through zero it identifies (almost) all of them.

Run:  pytest error_discovery/test_overcomplete.py -v
"""

import numpy as np

from error_discovery import experiment_overcomplete as O
from error_discovery.experiment_form import twin, attractor_traj
from error_discovery.experiment_meanstd import identifiable_count, max_mean_over_std


def test_offattractor_lowers_max_meanstd():
    """The accessible off-attractor ensemble drives the modes through zero, so its
    max|mean|/std is order one, far below the attractor's."""
    tw = twin()
    Xatt = attractor_traj(tw, 2000)
    Xoff = O.accessible_offattractor(tw, n_seg=4)
    assert max_mean_over_std(Xatt) > 50.0
    assert max_mean_over_std(Xoff) < 10.0


def test_full_library_succeeds_offattractor_fails_on():
    """Full quadratic library: (almost) no form identifiable on the attractor,
    (almost) every form identifiable off it."""
    tw = twin()
    nf_att, _ = identifiable_count(tw, attractor_traj(tw, 2000), "full")
    nf_off, _ = identifiable_count(tw, O.accessible_offattractor(tw, n_seg=4), "full")
    assert nf_att <= 2, f"attractor unexpectedly identified {nf_att}"
    assert nf_off >= tw.ndim - 3, f"off-attractor identified only {nf_off}/{tw.ndim}"
