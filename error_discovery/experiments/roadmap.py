"""
error_discovery.experiments.roadmap
===================================

A schematic, not a study: the three ways the injected error is used, drawn so the
clean single-driver default reads as intentional rather than under-ambitious.

  1. canonical single-driver, g = T_{o,1}, c = (0,0,1,0) — the hardest case;
  2. the mixed-driver stress test (:mod:`driver_mixed`), all drivers active;
  3. the 36-mode identifiability sweep (:mod:`obstruction`), each mode in turn.

Pure matplotlib patch drawing: no numerics, no data, and therefore no ``run()`` —
the registry lists it with no compute step.
"""

import os

from .. import figstyle
from ..config import FIGURES_DIR
from ..figstyle import DPI


def figure(save=True):
    """Draw the schematic; see the module docstring for what the three rows are."""
    plt = figstyle.pyplot()
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
    labels4 = [r"$1$", r"$\theta_{a,1}$", r"$T_{o,1}$", r"$\psi_{a,1}$"]
    TX = 5.3                                   # body-text left edge, clears the widest strip

    def cellrow(x0, y0, active, names=None, w=0.62, h=0.5, gap=0.1):
        for k in range(4):
            on = active[k]
            ax.add_patch(Rectangle((x0 + k * (w + gap), y0), w, h,
                                   facecolor=("C2" if on else "white"),
                                   edgecolor="0.4", lw=1.0, alpha=0.85 if on else 1))
            ax.text(x0 + k * (w + gap) + w / 2, y0 + h / 2, names[k],
                    ha="center", va="center", fontsize=8,
                    color="white" if on else "0.4")

    rows = [
        (4.3, "1.  Canonical single-driver  (the hardest case)",
         "$g=T_{o,1}$, $\\;c=(0,0,1,0)$.  Isolates the identifiability\n"
         "mechanism in the most tightly pinned driver.\nUsed in the mainline experiments and "
         "the appendix.", "single"),
        (2.3, "2.  Mixed-driver stress test",
         "$g=\\theta_{a,1}+T_{o,1}+\\psi_{a,1}$, $\\;c=(0,1,1,1)$.\nDoes the mechanism survive a "
         "less-sparse\nerror, not just one surgical driver?", "mixed"),
        (0.3, "3.  36-driver identifiability sweep",
         "$g=x_k,\\;k=1\\ldots36$, each mode injected\nsingly. Characterizes the boundary over\n"
         "all modes, not a particular error.", "sweep"),
    ]
    for y, title, body, kind in rows:
        ax.add_patch(Rectangle((0.1, y + 0.05), 9.8, 1.75, fill=False,
                               edgecolor="0.8", lw=0.8))
        ax.text(0.25, y + 1.55, title, fontsize=10.5, fontweight="bold", va="top")
        if kind == "single":
            cellrow(0.35, y + 0.45, [0, 0, 1, 0], names=labels4)
        elif kind == "mixed":
            cellrow(0.35, y + 0.45, [0, 1, 1, 1], names=labels4)
        else:
            w, gap = 0.105, 0.02
            for k in range(36):
                on = (k == 17)
                ax.add_patch(Rectangle((0.35 + k * (w + gap), y + 0.5), w, 0.5,
                                       facecolor=("C2" if on else "0.92"),
                                       edgecolor="0.6", lw=0.4))
            ax.annotate("each in turn", (0.35 + 17 * (w + gap) + w / 2, y + 0.5),
                        xytext=(0, -12), textcoords="offset points", fontsize=7.5,
                        ha="center", va="top", color="C2",
                        arrowprops=dict(arrowstyle="->", color="C2", lw=0.8))
        ax.text(TX, y + 1.5, body, fontsize=8.6, va="top")

    ax.text(5, 6.4, "How the injected error is used", fontsize=12.5,
            fontweight="bold", ha="center", va="top")
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_roadmap.png")
    if save:
        fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return [out]
