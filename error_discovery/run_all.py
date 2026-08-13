#!/usr/bin/env python
"""
error_discovery.run_all
=======================

One-command, reproducible driver for every study in the differentiable
error-discovery project. Each study writes its results and figures to
``error_discovery/results/``; everything is seeded.

Every study module answers to the same two entry points::

    run(...)     computes, writes .npz to RESULTS_DIR, returns its results
    figure(...)  reloads that .npz, draws, returns the list of paths written

``run`` never draws and ``figure`` never computes, so a figure can be re-rendered
from cached results without repeating the expensive part. ``STUDIES`` below maps a
command-line flag onto the module implementing it; adding a study is one entry.

The studies map onto the paper as follows: the default lead-time study and
``--sparsity`` cover known-form recovery (window of discoverability, observation
noise, partial observations); ``--form``, ``--obstruction``, ``--reanalysis`` and
``--increment`` cover unknown-form discovery and its identifiability boundary.

``experiments.jax_validation`` and ``experiments.oracle`` are deliberately not
here. The first sizes a multiprocessing pool to ``cpu_count()`` at import; the
second writes no files and hardcodes a non-default ``g_coeffs``. Run those
directly.
"""

import argparse
import importlib
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Study:
    """One study: which module implements it and how to drive it.

    ``steps`` names the compute functions to call in order — most studies just
    have ``run``; ``--roadmap`` is a schematic with nothing to compute and so
    lists none. ``figure`` names the drawing entry point.
    """

    module: str
    help: str
    steps: tuple = ("run",)
    figure: str = "figure"
    default_tag: str = None      # study is tagged; None means it takes no arguments
    wants_cfg: bool = False      # run() takes the whole ExperimentConfig


DEFAULT = "leadtime"

STUDIES = {
    # --- known form: how far ahead, and from which observations? -------------- #
    "leadtime": Study(
        "leadtime", wants_cfg=True,
        help="lead-time window of discoverability (the default study)"),
    "sparsity": Study(
        "sparsity", default_tag="sparsity",
        help="partial-observation study (which variables are observed)"),
    # --- unknown form: can it be discovered at all, and from what data? ------- #
    "form": Study(
        "form",
        help="form discovery from trajectories (E1-E5)"),
    "obstruction": Study(
        "obstruction", steps=("run", "eps_sweep"),
        help="near-constant-mode identifiability obstruction + eps-scaling"),
    "reanalysis": Study(
        "reanalysis",
        help="is reanalysis far enough to identify the error form?"),
    "increment": Study(
        "increment",
        help="do DA increments unlock discovery?"),
    "why": Study(
        "why",
        help="why only off-attractor data identifies the form (governing law, "
             "global-form control, Lyapunov provenance)"),
    "robustness": Study(
        "robustness",
        help="robustness of the mean/std diagnostic (approximate law + parsimony floor)"),
    "overcomplete": Study(
        "overcomplete",
        help="the over-complete library succeeds off the attractor (boundary complement)"),
    "donoharm": Study(
        "donoharm",
        help="does an L1/sparsity penalty rescue the attractor? (no, under model-form error)"),
    "climatology": Study(
        "climatology",
        help="climatology fidelity of the recovered correction (the error is "
             "climatologically silent)"),
    # --- which driver? (one file per question, split out of experiment_driver) - #
    "driver-scan": Study(
        "driver_scan",
        help="which driver is discoverable on the attractor: margin vs the driver's "
             "own mean/std"),
    "driver-record": Study(
        "driver_record",
        help="does a longer record help? (atmosphere yes, ocean temperature never)"),
    "driver-mixed": Study(
        "driver_mixed",
        help="all three drivers active at once: joint coefficient recovery vs noise"),
    "alias": Study(
        "alias",
        help="the alias mechanism in three panels (near-constant mode times a driver)"),
    "roadmap": Study(
        "roadmap", steps=(),
        help="schematic of how the injected error is used (no computation)"),
}


def _print_figs(figs):
    print("[figures]")
    for f in figs:
        print("  ", f)


def _config(args):
    """The lead-time study's config, with the command-line overrides applied.

    ``QUICK`` is a module-level singleton, so it is replaced rather than mutated —
    editing it in place would leak the overrides into anything importing it later.
    """
    from .config import ExperimentConfig, QUICK
    base = QUICK if args.quick else ExperimentConfig()
    over = {k: v for k, v in (("obs", args.obs), ("tag", args.tag)) if v is not None}
    return replace(base, **over)


def dispatch(name, args):
    """Run one study and return the figure paths it wrote."""
    s = STUDIES[name]
    mod = importlib.import_module(f".experiments.{s.module}", __package__)

    # Imports stay inside dispatch: every study module pulls in jax, so importing
    # the whole table up front would cost seconds on `--help`.
    if s.wants_cfg:
        cfg = _config(args)
        for step in s.steps:
            getattr(mod, step)(cfg)
        return getattr(mod, s.figure)(cfg.tag)

    if s.default_tag is not None:
        tag = args.tag or s.default_tag
        for step in s.steps:
            getattr(mod, step)(tag=tag)
        return getattr(mod, s.figure)(tag)

    for step in s.steps:
        getattr(mod, step)()
    return getattr(mod, s.figure)()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fast smoke configuration (lead-time study)")
    ap.add_argument("--obs", choices=["full", "atmos", "ocean", "temp"], default=None,
                    help="observation operator for the lead-time study (overrides config)")
    ap.add_argument("--tag", default=None, help="output tag (overrides config)")
    study = ap.add_mutually_exclusive_group()
    for name, s in STUDIES.items():
        if name != DEFAULT:
            study.add_argument(f"--{name}", action="store_true", help=s.help)
    args = ap.parse_args(argv)

    # argparse turns --driver-scan into args.driver_scan
    chosen = next((n for n in STUDIES
                   if n != DEFAULT and getattr(args, n.replace("-", "_"))), DEFAULT)
    _print_figs(dispatch(chosen, args))


if __name__ == "__main__":
    main()
