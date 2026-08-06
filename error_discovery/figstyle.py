"""
error_discovery.figstyle
========================

Shared figure styling for the manuscript figures.  Deliberately holds no matplotlib
import: the backend is still selected at each ``figure()`` call site.
"""

DPI = 160

# A faint white backing so an annotation that must sit near a curve interrupts the
# line cleanly instead of being scrawled over it.
LBL_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0)
