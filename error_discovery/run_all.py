#!/usr/bin/env python
"""
error_discovery.run_all
=======================

One-command, reproducible driver for every study in the differentiable
error-discovery project. Each study writes its results and figures to
``error_discovery/results/``; everything is seeded.

Usage
-----
    python -m error_discovery.run_all                 # lead-time window of discoverability (default)
    python -m error_discovery.run_all --quick         # fast smoke configuration
    python -m error_discovery.run_all --obs atmos     # lead-time study, atmosphere-only observations
    python -m error_discovery.run_all --sparsity      # partial-observation (which variables) study
    python -m error_discovery.run_all --form          # form discovery from trajectories (E1-E5)
    python -m error_discovery.run_all --obstruction   # near-constant-mode identifiability obstruction + eps-scaling
    python -m error_discovery.run_all --reanalysis    # is reanalysis far enough to identify the form?
    python -m error_discovery.run_all --increment     # do DA increments unlock discovery?
    python -m error_discovery.run_all --why           # why only off-attractor data identifies the form
    python -m error_discovery.run_all --driver         # which driver is discoverable (per-driver mean/std, record length)

The studies map onto the paper as follows: --quick/default and --sparsity cover
the known-form recovery (window of discoverability, observation noise, partial
observations); --form, --obstruction, --reanalysis, --increment cover the
unknown-form discovery and its identifiability boundary.
"""

import argparse


def _print_figs(figs):
    print("[figures]")
    for f in figs:
        print("  ", f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fast smoke configuration (lead-time study)")
    ap.add_argument("--obs", choices=["full", "atmos", "ocean", "temp"], default=None,
                    help="observation operator for the lead-time study (overrides config)")
    ap.add_argument("--tag", default=None, help="output tag (overrides config)")
    study = ap.add_mutually_exclusive_group()
    study.add_argument("--sparsity", action="store_true", help="partial-observation study")
    study.add_argument("--form", action="store_true", help="form discovery from trajectories (E1-E5)")
    study.add_argument("--obstruction", action="store_true",
                       help="near-constant-mode identifiability obstruction + eps-scaling")
    study.add_argument("--reanalysis", action="store_true",
                       help="is reanalysis far enough to identify the error form?")
    study.add_argument("--increment", action="store_true", help="do DA increments unlock discovery?")
    study.add_argument("--robustness", action="store_true",
                       help="robustness of the mean/std diagnostic (approximate law + parsimony floor)")
    study.add_argument("--why", action="store_true",
                       help="why only off-attractor data identifies the form (governing law, "
                            "global-form control, Lyapunov provenance)")
    study.add_argument("--driver", action="store_true",
                       help="which driver is discoverable: per-driver mean/std and the "
                            "record-length dichotomy (atmosphere curable, ocean temperature not)")
    study.add_argument("--overcomplete", action="store_true",
                       help="the over-complete library succeeds off the attractor (boundary complement)")
    study.add_argument("--donoharm", action="store_true",
                       help="does an L1/sparsity penalty rescue the attractor? (no, under model-form error)")
    study.add_argument("--climatology", action="store_true",
                       help="climatology fidelity of the recovered correction (the error is climatologically silent)")
    args = ap.parse_args(argv)

    if args.sparsity:
        from . import experiment_sparsity, plotting
        experiment_sparsity.run(tag=args.tag or "sparsity")
        _print_figs(plotting.make_sparsity(args.tag or "sparsity"))
    elif args.form:
        from . import experiment_form
        experiment_form.run()                       # writes fig_form_identifiability, fig_form_discovery
    elif args.obstruction:
        from . import experiment_meanstd
        experiment_meanstd.run()
        experiment_meanstd.eps_sweep()
        _print_figs([experiment_meanstd.figure()])
    elif args.reanalysis:
        from . import experiment_form
        experiment_form.reanalysis_run()            # writes fig_reanalysis
    elif args.increment:
        from . import experiment_increment
        experiment_increment.run()
        _print_figs([experiment_increment.figure()])
    elif args.robustness:
        from . import experiment_robustness
        experiment_robustness.law_validation()
        experiment_robustness.parsimony_floor()
        _print_figs([experiment_robustness.figure()])
    elif args.why:
        from . import experiment_why
        _print_figs([experiment_why.run()])         # writes fig_why; commits why_*/lyap_ics npz
    elif args.driver:
        from . import experiment_driver
        _print_figs([experiment_driver.run()])      # writes fig_driver(+_collapse), fig_stiff; commits driver_*.npz
    elif args.overcomplete:
        from . import experiment_overcomplete
        experiment_overcomplete.run()
        _print_figs([experiment_overcomplete.figure()])
    elif args.donoharm:
        from . import experiment_donoharm
        experiment_donoharm.run()
        _print_figs([experiment_donoharm.figure()])
    elif args.climatology:
        from . import experiment_climatology
        experiment_climatology.run()
        _print_figs([experiment_climatology.figure()])
    else:
        from . import experiment, plotting
        from .config import ExperimentConfig, QUICK
        cfg = QUICK if args.quick else ExperimentConfig()
        if args.obs is not None:
            cfg.obs = args.obs
        if args.tag is not None:
            cfg.tag = args.tag
        experiment.run(cfg)
        _print_figs(plotting.make_all(cfg.tag))


if __name__ == "__main__":
    main()
