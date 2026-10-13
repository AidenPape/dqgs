"""
error_discovery.figstyle
========================

Shared figure styling for the manuscript figures.

Deliberately holds no module-level matplotlib import: ``config`` and ``figstyle``
are pulled in by every study, and matplotlib is only needed by the ``figure()``
half. :func:`pyplot` does the import — and the backend choice — at the call site.
"""

DPI = 160

# A faint white backing so an annotation that must sit near a curve interrupts the
# line cleanly instead of being scrawled over it.
LBL_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0)


def pyplot():
    """Return ``matplotlib.pyplot`` with the Agg backend selected.

    Every figure runs headless (batch nodes, CI, and the launcher), so the backend
    must be chosen before pyplot is first imported. Call this at the top of a
    ``figure()`` body::

        plt = figstyle.pyplot()
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def finish(fig, line=None):
    """``tight_layout``, with ``line`` (a sampled run's ``sampling.sampling_line``) as a
    suptitle when given -- ``tight_layout`` makes room for it. Without one the layout is
    unchanged."""
    if line:
        fig.suptitle(line, fontsize=10)
    fig.tight_layout()
