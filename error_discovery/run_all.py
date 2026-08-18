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

A study's knobs are its own keyword arguments, reached with ``--set``::

    python -m error_discovery.run_all --driver-scan --list-params
    python -m error_discovery.run_all --driver-scan --set n=8000 --set reps=40

There is no second list of parameters to maintain: ``--list-params`` reads the
signatures of the functions the study actually calls, so it cannot drift from them.
For the default lead-time study the knobs are the fields of
:class:`config.ExperimentConfig`, because that is what its ``run()`` takes; ``--quick``
swaps in the smoke-test preset and ``--set`` overrides fields on top of it.

Values are parsed against the type of the parameter's default, so ``--set whiten=false``
means what it says and ``--set windows=(320,640)`` gives a tuple.

The whitened lead-time sweep (manuscript Figure 2) is now reachable::

    python -m error_discovery.run_all --set whiten=true --set tag=main_whiten

**Give it a tag of its own.** ``whiten`` does not change the output filename, so running
it under the default tag overwrites ``leadtime_main.npz``. ``finish_fig2_whiten.py`` stays
the scripted version of that run, because it also extends the window set out to ~270 days.

The studies map onto the paper as follows: the default lead-time study and
``--sparsity`` cover known-form recovery (window of discoverability, observation
noise, partial observations); ``--form``, ``--obstruction``, ``--reanalysis`` and
``--increment`` cover unknown-form discovery and its identifiability boundary.
"""

import argparse
import ast
import importlib
import inspect
from dataclasses import dataclass, fields, replace

from .config import ExperimentConfig, QUICK

# The parameter name by which a step receives the whole ExperimentConfig. Only the
# lead-time study has one; every other study's knobs are plain keyword arguments.
# Whether a study takes a config is therefore read off its signature, not declared.
CFG_ARG = "cfg"
CFG_FIELDS = tuple(f.name for f in fields(ExperimentConfig))


@dataclass(frozen=True)
class Study:
    """One study: which module implements it and how to drive it.

    ``steps`` names the compute functions to call in order — most studies just
    have ``run``; ``--roadmap`` is a schematic with nothing to compute and so
    lists none. ``figure`` names the drawing entry point, or is ``None`` for a
    study that prints a diagnostic instead of writing one.
    """

    module: str
    help: str
    steps: tuple = ("run",)
    figure: str = "figure"


DEFAULT = "leadtime"

STUDIES = {
    # --- known form: how far ahead, and from which observations? -------------- #
    "leadtime": Study(
        "leadtime",
        help="lead-time window of discoverability (the default study)"),
    "sparsity": Study(
        "sparsity",
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
    # --- validation and numerical diagnostics -------------------------------- #
    "param-tuning": Study(
        "param_tuning",
        help="warm-up: Adam through the integrator recovers mis-set physical "
             "parameters (kd, kdp, Lpa) -- the easy problem the rest of this is not"),
    "jax-validation": Study(
        "jax_validation",
        help="the JAX rewrite reproduces the original qgs model (imports the original "
             "qgs integrator; see its docstring)"),
    "oracle": Study(
        "oracle", figure=None,
        help="why the noiseless mixed-driver ocean coefficient comes out low: rank, "
             "conditioning and solver comparison (prints a table, writes nothing)"),
}


def _entry_points(mod, s):
    """The study's callables, in the order :func:`dispatch` invokes them."""
    names = (*s.steps, *((s.figure,) if s.figure else ()))
    return [(n, getattr(mod, n)) for n in names]


def _takes_cfg(mod, s):
    """Is this study driven by an ExperimentConfig rather than by plain kwargs?

    Read off the signature rather than declared in ``STUDIES``: a study takes a config
    if and only if its ``run`` says so, and one place to state that is enough.
    """
    return any(CFG_ARG in inspect.signature(getattr(mod, n)).parameters for n in s.steps)


def _knobs(mod, s):
    """Every settable parameter of a study, as ``(defaults, where)``.

    Both dicts are keyed by parameter name; ``where`` records which entry points
    accept it, which is how the listing shows that ``--set n=...`` reaches both
    ``run`` and ``eps_sweep`` in the obstruction study. A config-driven study
    exposes the config's fields instead of the ``cfg`` argument itself.
    """
    defaults, where = {}, {}
    if _takes_cfg(mod, s):
        base = ExperimentConfig()
        for name in CFG_FIELDS:
            defaults[name] = getattr(base, name)
            where[name] = ["config"]
    for fname, fn in _entry_points(mod, s):
        for name, p in inspect.signature(fn).parameters.items():
            if name == CFG_ARG or p.default is inspect.Parameter.empty:
                continue
            defaults.setdefault(name, p.default)
            where.setdefault(name, []).append(fname)
    return defaults, where


def _coerce(key, raw, default):
    """Parse one ``--set`` value against the type of the parameter's default.

    Typed against the default rather than guessed, because the guess that matters is
    the boolean one: ``ast.literal_eval("false")`` raises, and quietly falling back to
    the *string* ``"false"`` would be truthy — silently the opposite of what was asked.
    """
    if isinstance(default, bool):                    # before int: bool is an int
        if raw.lower() in ("true", "1", "yes"):
            return True
        if raw.lower() in ("false", "0", "no"):
            return False
        raise SystemExit(f"--set {key}={raw}: expected true or false")
    if isinstance(default, str):
        return raw
    try:
        val = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        raise SystemExit(f"--set {key}={raw}: not a Python literal")
    if isinstance(default, tuple):
        if not isinstance(val, (tuple, list)):
            raise SystemExit(f"--set {key}={raw}: expected a sequence, e.g. '(320,640)'")
        return tuple(val)
    if isinstance(default, (int, float)) and not isinstance(val, (int, float)):
        raise SystemExit(f"--set {key}={raw}: expected a number")
    return val


def _call(fn, values, cfg):
    """Call ``fn`` with whichever of ``values`` its signature declares."""
    sig = inspect.signature(fn)
    kw = {k: v for k, v in values.items() if k in sig.parameters}
    if cfg is not None and CFG_ARG in sig.parameters:
        return fn(cfg, **kw)
    return fn(**kw)


def _print_figs(figs):
    if not figs:
        print("[no figures] this study reports to stdout")
        return
    print("[figures]")
    for f in figs:
        print("  ", f)


def _load(name):
    """Import the module implementing a study.

    Imports stay lazy: every study module pulls in jax, so importing the whole table
    up front would cost seconds on ``--help``.
    """
    return importlib.import_module(f".experiments.{STUDIES[name].module}", __package__)


def list_params(name):
    """Print a study's parameters, defaults, and which entry points take each."""
    defaults, where = _knobs(_load(name), STUDIES[name])
    print(f"--{name}: parameters, set with --set key=value\n")
    if not defaults:
        print("  (none — this study takes no arguments)")
        return
    w = max(len(k) for k in defaults)
    for k, v in defaults.items():
        print(f"  {k:<{w}} = {v!r:<30} [{', '.join(where[k])}]")


def dispatch(name, overrides, quick=False):
    """Run one study and return the figure paths it wrote."""
    s = STUDIES[name]
    mod = _load(name)
    defaults, _ = _knobs(mod, s)

    unknown = [k for k in overrides if k not in defaults]
    if unknown:
        raise SystemExit(f"--{name}: no such parameter {', '.join(unknown)}; "
                         f"see --{name} --list-params")
    over = {k: _coerce(k, v, defaults[k]) for k, v in overrides.items()}

    # Only the *overrides* are ever passed on; the defaults stay where they are
    # declared. Re-passing them would silently undo --quick, whose whole content is a
    # different set of field values.
    cfg, values = None, over
    if _takes_cfg(mod, s):
        cfg = replace(QUICK if quick else ExperimentConfig(),
                      **{k: v for k, v in over.items() if k in CFG_FIELDS})
        values = {**{f: getattr(cfg, f) for f in CFG_FIELDS},
                  **{k: v for k, v in over.items() if k not in CFG_FIELDS}}
    elif quick:
        raise SystemExit(f"--quick is the lead-time study's config preset; "
                         f"--{name} has no config. Use --set to shrink it.")

    for step in s.steps:
        _call(getattr(mod, step), values, cfg)
    if s.figure is None:
        return []
    return _call(getattr(mod, s.figure), values, cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="smoke-test config preset (lead-time study only)")
    ap.add_argument("--set", metavar="KEY=VALUE", action="append", default=[],
                    dest="overrides",
                    help="override one of the study's parameters; repeatable")
    ap.add_argument("--list-params", action="store_true",
                    help="print the chosen study's parameters and defaults, then exit")
    study = ap.add_mutually_exclusive_group()
    for name, s in STUDIES.items():
        if name != DEFAULT:
            study.add_argument(f"--{name}", action="store_true", help=s.help)
    args = ap.parse_args(argv)

    # argparse turns --driver-scan into args.driver_scan
    chosen = next((n for n in STUDIES
                   if n != DEFAULT and getattr(args, n.replace("-", "_"))), DEFAULT)
    if args.list_params:
        list_params(chosen)
        return

    over = {}
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set {item}: expected key=value")
        k, v = item.split("=", 1)
        over[k.strip()] = v
    _print_figs(dispatch(chosen, over, quick=args.quick))


if __name__ == "__main__":
    main()
