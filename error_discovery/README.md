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

- **`STORY.md`** — the one-page spine: the claim, the per-finding disposition, and
  the conviction assessment. Start here.
- **`NOTES.md`** — the full technical reference: every method, number, parameter
  choice, gotcha, and dead-end. For running and extending the toolbox.
- **`../manuscript/main.tex`** — the paper.

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
| `alias.py`        | the alias mechanism, in three panels |
| `roadmap.py`      | schematic of how the injected error is used (no computation) |
| `oracle.py`, `jax_validation.py` | produce paper figures but are **not** in `run_all` — see its docstring for why |

## Reproduce

```bash
python -m error_discovery.run_all --help       # every study, with a one-line description
python -m error_discovery.run_all              # lead-time window of discoverability (default)
python -m error_discovery.run_all --quick      # fast smoke run (minutes)
python -m error_discovery.run_all --form       # form discovery from trajectories
```

One study runs per invocation. The flags are generated from `run_all.STUDIES`, so `--help` is
the authoritative list rather than anything written down here.

Each study module exposes `run()`, which computes and writes a `.npz`, and `figure()`, which
reloads that `.npz` and draws. Because they are separate, a figure can be redrawn without
repeating the computation:

```python
from error_discovery.experiments import form
form.figure()                   # redraws from results/form_results.npz
```

Everything is seeded. The one expensive step, spinning the base model onto the
attractor, is cached with a provenance check in `results/attractor_ic.npz` (an
800k-step spin-up, seed 1) and reused. Do not pass a non-default spin-up to
`get_attractor_ic`: it overwrites the shared cache. Delete the file to rebuild.

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
