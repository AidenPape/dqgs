"""
error_discovery.experiments
===========================

One module per study. Every one of them answers to the same pair::

    run(...)     computes, writes .npz to RESULTS_DIR, returns its results
    figure(...)  reloads that .npz, draws, returns the list of paths written

``run`` never draws and ``figure`` never computes, so a figure can be re-rendered
from cached results in about a second. :mod:`error_discovery.run_all` maps a
command-line flag onto each of these modules.

Nothing is imported here: every study pulls in jax, so importing the package would
cost seconds for anyone who only wanted one study. Import the module you want.

The studies build on the machinery one level up — :mod:`~error_discovery.formfit`
(twin, sampling distributions, library scoring), :mod:`~error_discovery.screening`
(the |mean|/std diagnostic), :mod:`~error_discovery.discovery` (known-form
recovery) and :mod:`~error_discovery.attractor` (the cached spin-up). They do not
import each other.
"""
