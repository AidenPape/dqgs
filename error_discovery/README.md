# error_discovery — differentiable discovery of MAOOAM model error

A twin experiment that asks: once a climate model is **differentiable** (its whole
time integration is built from automatically-differentiable operations), can one
recover an injected model error by gradient descent on a forecast-mismatch loss,
and what limits that?

The model is MAOOAM, a 36-variable reduced-order coupled ocean-atmosphere model. A
"truth" model carries a known, state-dependent error in the atmosphere-ocean
thermal-exchange coupling. A differentiable JAX twin (built on `qgs_jax`) recovers
the error by differentiating a trajectory-matching loss through the integrator.

**Headline.** The differentiable model recovers the error, including its functional
form, from short trajectories when the candidate error terms are restricted to a
physically-motivated set. The limit is identifiability, not the optimizer: against a
complete quadratic library, a single near-constant mode on the data aliases every
candidate error form, so the form is unidentifiable from any near-attractor data
(the model's climatology, reanalysis, or short forecasts). The controlling quantity
is the maximum `|mean|/std` over all state modes.

## Where to read next

- **`notebooks/getting_started.ipynb`** — a guided tour: what the model is, what error
  is injected into it, and how to drive the package.
- **The Layout table below** — the index of what exists.
- **Each study's module docstring** — the question it asks, the result, and the caveats.

## Install

```bash
conda activate dqgs                  # any env with python >= 3.9
pip install --no-deps qgs==1.0.0     # upstream declares ipython+jupyter; unused here
pip install -e ..                    # installs this repo and its remaining dependencies
```

## Layout

The package is in two halves: **machinery** at the top level, **studies** in
`experiments/`. Studies import machinery; they do not import each other.

### Machinery

| file | role |
|------|------|
| `config.py`   | all knobs (`ExperimentConfig`, `QUICK`), constants (`DT`, `EPS`, `THRESH`), nondim-to-days conversion |
| `models.py`   | `TwinModels`: base / truth / discovery tendencies in JAX; observation operators |
| `attractor.py`| the cached 800k-step spin-up state, and the truth segments cut from it |
| `sampling.py` | the sampling standard: the long base run, its passive / active regime labels, and the starting states studies draw from it |
| `discovery.py`| known-form recovery: rollout, Levenberg-Marquardt, `cond(JᵀJ)`, Lyapunov time |
| `formfit.py`  | unknown-form recovery: the twin, the sampling distributions, the candidate library, the closed-form fit, and the identifiability **margin** |
| `screening.py`| the `\|mean\|/std` diagnostic and the restricted-vs-full quadratic library |
| `figstyle.py` | `DPI`, `LBL_BBOX`, and `pyplot()` (the headless-backend helper) |
| `run_all.py`  | one-command launcher; the `STUDIES` table is the index of what exists |

### Studies — `experiments/`, one question each

| file | question |
|------|----------|
| `leadtime.py`     | over what forecast window is a known-form error recoverable? |
| `sparsity.py`     | which observed variables are enough? |
| `sampling_checks.py` | is the base run free of drift, which regime window T, and do the labels match psi_a1? |
| `form.py`         | can the *form* be discovered from trajectories at all? (E1-E5) |
| `obstruction.py`  | what blocks it — near-constant modes, and how the library choice decides |
| `reanalysis.py`   | is reanalysis far enough off the biased attractor to identify the form? |
| `increment.py`    | do DA increments unlock discovery? |
| `why.py`          | the governing law: margin vs the driver's own `\|mean\|/std` |
| `robustness.py`   | how far does that law hold? (screening law + parsimony floor) |
| `overcomplete.py` | the over-complete library succeeds off the attractor |
| `donoharm.py`     | does an L1 penalty rescue the attractor? (no, under model-form error) |
| `climatology.py`  | the error is climatologically silent |
| `driver_scan.py`  | which driver is discoverable on the attractor? |
| `driver_record.py`| does a longer record help? (atmosphere yes, ocean temperature never) |
| `driver_mixed.py` | all three drivers at once: joint coefficient recovery vs noise |
| `driver_window.py`| does the window of discoverability depend on which driver? |
| `driver_mode.py`  | does it depend on how much the driver *varies*? (ten `theta_a` modes) |
| `alias.py`        | the alias mechanism, in three panels |
| `roadmap.py`      | schematic of how the injected error is used (no computation) |
| `param_tuning.py`   | the warm-up: can Adam recover mis-set *physical* parameters? (yes, to ~1e-8) |
| `jax_validation.py` | does the JAX rewrite reproduce the original qgs model? (to 1e-14) |
| `oracle.py`       | why the noiseless mixed-driver ocean coefficient comes out low — prints a table, writes no figure |

## Reproduce

```bash
python -m error_discovery.run_all --help       # every study, with a one-line description
python -m error_discovery.run_all              # lead-time window of discoverability (default)
python -m error_discovery.run_all --quick      # fast smoke run (minutes)
python -m error_discovery.run_all --form       # form discovery from trajectories
```

One study runs per invocation. The flags are generated from `run_all.STUDIES`, so `--help` is
the authoritative list rather than anything written down here.

A study's parameters are its own keyword arguments, and `--set` reaches them:

```bash
python -m error_discovery.run_all --driver-scan --list-params
python -m error_discovery.run_all --driver-scan --set n=8000 --set reps=40
python -m error_discovery.run_all --set whiten=true --set tag=main_whiten   # Fig. 2
```

`--list-params` reads the signatures of the functions the study actually calls, so it cannot
drift from them. For the default lead-time study the parameters are the fields of
`ExperimentConfig`; `--quick` swaps in the smoke-test preset and `--set` overrides fields on
top of it. Give a whitened run its own `tag` — `whiten` does not change the output filename.

Each study module exposes `run()`, which computes and writes a `.npz`, and `figure()`, which
reloads that `.npz` and draws. Because they are separate, a figure can be redrawn without
repeating the computation:

```python
from error_discovery.experiments import form
form.figure()                   # redraws from results/form_results.npz
```

Everything is seeded. The base model is the `default` parameter set of
`qgs_jax.spec.PARAM_SETS` (values and source in `QGS_JAX_README.md`, "Parameter sets").

### Spun-up states

A spun-up state is saved in `initial_conditions/` (`config.IC_DIR`, tracked in git, unlike
`results/*.npz`) as `attractor_ic_<config>_sp<steps>.npz`:

- `<config>` is a `PARAM_SETS` name, which fixes both the parameters and the truncation. A
  `-ndim<N>` suffix marks a state at a dimension other than that set's own
  (`default-ndim56`: the `default` parameters with a 2 x 4 atmosphere).
- `<steps>` is the spin-up length in RK4 steps at `dt = 0.1`, written in full
  (`sp800000`, `sp1000000000`).
- A run still in progress is `attractor_ic_<config>_sp<steps>.partial.npz`.

Build names with `config.ic_path(config, steps)`, never by hand. Files written by
`get_attractor_ic` and `spin_up` store the recipe that produced them, and neither loader uses
a file whose recipe does not match: `load_spun_up_ic` raises, `get_attractor_ic` rebuilds.

Every study starts from `initial_conditions/attractor_ic_default_sp800000.npz` (800,000 steps, 24.6 yr,
seed 1), read through `get_attractor_ic`. Do not pass a non-default spin-up to
`get_attractor_ic`: it overwrites that shared cache. Delete the file to rebuild.

A 1e9-step (30,706 yr) state of the same set is built by `attractor.spin_up`, about 4.5 h on
a laptop, checkpointed and resumable, and read back with `attractor.load_spun_up_ic`:

```bash
python -c "from error_discovery.attractor import spin_up; spin_up('default', 1_000_000_000)"
```

### The sampling standard

New runs of the five segment studies (`leadtime`, `driver_window`, `driver_mode`,
`driver_uncertainty`, `sparsity`) can take their starting states from a long base run instead
of the 800k state: `--set sampling=passive`, `active` or `both` (default `legacy`, the original
design, which every `*_main.npz` uses). The design and every choice in it are in
`plans/sampling.md`, two levels up. Files, all in `results/`, named by functions in `config.py`:

- `baserun_<config>_sp<steps>_yr<start>-<end>.npz` (`config.baserun_path`): one 1,000-yr chunk of
  the base-model run forward from `attractor_ic_<config>_sp<steps>.npz`, the full state once per
  simulated day (every 89 steps), years counted from the start of the run.
- `regimes_<config>_sp<steps>_yr<start>-<end>.npz` (`config.regimes_path`): passive / active
  probabilities for every snapshot, the per-variable tau_int and high-frequency std.

```bash
python -m error_discovery.run_all --sampling-checks       # base run, labels, checks, figures
python -m error_discovery.run_all --driver-uncertainty --set sampling=both --set tag=both
```

Existing chunks are skipped, so a larger `--set n_years=` extends the record forward. A sampled
run left on its study's default tag is written under the mode's name, never over `main`.

## Tests

All tests live in `../tests/`.

```bash
pytest tests/test_meanstd.py tests/test_increment.py   # fast, pure-JAX
pytest tests/test_error_discovery.py
pytest tests/test_qgs_jax.py                           # run SEPARATELY
```

Run `test_qgs_jax.py` on its own: it imports the original `qgs` integrator, which
sizes its multiprocessing pool to `cpu_count()` and will overwhelm a core-limited
node. Cap `os.cpu_count` via a throwaway `sitecustomize.py` on `PYTHONPATH` if it
thrashes.
