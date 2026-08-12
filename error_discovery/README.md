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

| file | role |
|------|------|
| `config.py`            | all knobs (`ExperimentConfig`, `QUICK`); nondim-to-days conversion |
| `models.py`            | `TwinModels`: base / truth / discovery tendencies in JAX; observation operators |
| `discovery.py`         | rollout, Levenberg-Marquardt recovery, `cond(JᵀJ)`, forcing recovery, Lyapunov time |
| `experiment.py`        | lead-time + observation-noise sweep (known-form recovery) |
| `experiment_sparsity.py` | partial-observation (which variables) sweep |
| `experiment_form.py`   | form discovery from trajectories (E1-E5), reanalysis test |
| `experiment_increment.py` | do DA increments unlock discovery? |
| `experiment_meanstd.py`  | the near-constant-mode identifiability obstruction + eps-scaling |
| `experiment_robustness.py` | robustness of the diagnostic: approximate screening law + parsimony floor |
| `experiment_why.py`    | why only off-attractor data identifies the form: governing law, global-form control, Lyapunov provenance |
| `experiment_driver.py` | which driver is discoverable: per-driver mean/std, the record-length dichotomy (atmosphere curable, ocean temperature not), full-error decomposition (circulation recovered, temperature components not), and presentation figures (experiment roadmap, alias-mechanism cartoon) |
| `plotting.py`          | manuscript figures from the `.npz` results |
| `run_all.py`           | one-command driver for every study; the `STUDIES` table is the index |

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
from error_discovery import experiment_form
experiment_form.figure()        # redraws from results/form_results.npz
```

Everything is seeded. The one expensive step, spinning the base model onto the
attractor, is cached with a provenance check in `results/attractor_ic.npz` (an
800k-step spin-up, seed 1) and reused. Do not pass a non-default spin-up to
`get_attractor_ic`: it overwrites the shared cache. Delete the file to rebuild.

## Tests

```bash
pytest error_discovery/test_meanstd.py error_discovery/test_increment.py   # fast, pure-JAX
pytest test_error_discovery.py                                             # at repo root
pytest test_qgs_jax.py                                                     # run SEPARATELY
```

Run `test_qgs_jax.py` on its own: it imports the original `qgs` integrator, which
sizes its multiprocessing pool to `cpu_count()` and will overwhelm a core-limited
node. Cap `os.cpu_count` via a throwaway `sitecustomize.py` on `PYTHONPATH` if it
thrashes.
