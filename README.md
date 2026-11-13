# dqgs — a differentiable MAOOAM model in JAX

`qgs_jax` is a JAX re-implementation of the MAOOAM coupled ocean-atmosphere configuration of
[`qgs`](https://github.com/Climdyn/qgs) (pinned to v1.0.0). It reproduces qgs to machine
precision and is differentiable end to end, so gradients of anything computed from a model
trajectory can be taken with respect to the model state or its physical parameters.

## The model

With `dynamic_T = False`, the whole of MAOOAM is one sparse rank-3 tendency tensor `T`:

```
dx_i/dt = sum_{j,k} T[i,j,k] * xb_j * xb_k,    xb = (1, x_1, ..., x_n)
```

`qgs_jax` extracts `T` from the original qgs code as an exact reference, then re-assembles it in
pure JAX as a function of the physical parameters (`kd`, `sig0`, `Lpa`, `G`, ...). The
tendencies and their Jacobian are JAX functions and the RK2 / RK4 integrators run under
`jax.lax.scan`, so a loss on an integrated trajectory can be differentiated back through the
integrator to those parameters. `tests/test_qgs_jax.py` checks tendencies, Jacobians, RK4 trajectories and the
re-assembled tensor against qgs, and parameter gradients against finite differences.

The default configuration is a 2 x 2 atmosphere and a 2 x 4 ocean, 36 variables. The parameter
sets, their values and their sources are in [`QGS_JAX_README.md`](QGS_JAX_README.md), which also
has a quick-start example.

## The experiments

[`error_discovery/`](error_discovery/README.md) uses the differentiable model for twin
experiments: a known, state-dependent error is injected into the ocean-atmosphere
heat-exchange coupling, and gradient-based fitting tries to recover its magnitude and
functional form from trajectory data. Its README covers the studies, their layout and how to
reproduce them.

## Repository

| path | what it holds |
|---|---|
| `qgs_jax/` | the differentiable model: `spec.py` (qgs reference and extraction), `model.py` (tendencies, Jacobian, integrators), `assemble.py` (differentiable tensor assembly) |
| `error_discovery/` | the error-discovery experiments |
| `tests/` | pytest suite |
| `notebooks/getting_started.ipynb` | a guided tour |
| `smoke_test.py` | builds the perturbed tendencies and integrates them with qgs's own integrator |
| `tune_demo.py` | recovers corrupted model parameters by gradient descent through the integrator |
| `pyproject.toml` | dependencies; the distribution is named `dqgs-error-discovery` |

## Install

From the repository root. `pyproject.toml` declares the dependencies without versions; the pins
below are the laptop env's versions as of 2026-11-12. conda-forge supplies everything except jax
and qgs, which come from PyPI.

```bash
conda create -y -n dqgs -c conda-forge --override-channels \
    python=3.11.15 numpy=2.4.6 scipy=1.17.1 matplotlib=3.10.9 scikit-learn=1.9.1 \
    numba=0.65.1 sympy=1.14.0 sparse=0.18.0 pebble=5.2.0 pytest=9.0.3 pip \
    "libblas=*=*openblas" "libopenblas=0.3.33=openmp*"
conda activate dqgs
# CPU jax; numpy and scipy are repeated so pip does not upgrade them
pip install jax==0.10.1 jaxlib==0.10.1 numpy==2.4.6 scipy==1.17.1
pip install --no-deps qgs==1.0.0     # upstream declares ipython+jupyter; unused here
pip install --no-deps -e .           # this repo, editable (distribution `dqgs-error-discovery`)
```

The two `libblas` / `libopenblas` terms fix the linear-algebra library (OpenBLAS 0.3.33, OpenMP
build), which numpy and scipy use for the Levenberg-Marquardt steps. Use a dedicated env:
another project declares the distribution name `dqgs`, and installing both in one env replaces
whichever came first.

### Check

Basic, about 80 s on the laptop; needs only files in this repo:

```bash
python -c "import qgs_jax, error_discovery.models"   # the install resolves, qgs included (~2 s)
python smoke_test.py                                  # qgs's own integrator runs the perturbed model (~10 s)
pytest tests/test_qgs_jax.py                          # JAX model vs qgs; gradients vs finite differences (~70 s)
```

Full, about 5 min on the laptop: `pytest tests/ --ignore=tests/test_lectures.py`
(`test_lectures.py` reads lecture notes that are not in this repo).

### On NCAR Casper

```bash
module load conda          # NCAR's conda; job scripts need it too, before `conda activate dqgs`
```

then the commands above. With NCAR's module the env lands in `/glade/work/$USER/conda-envs/dqgs`
(NCAR HPC Docs, "Using conda for Python Package Management"). Build it on a compute node, inside
a `qcmd` job, not on a login node: login nodes are for short, light processes and stop heavy
ones automatically (NCAR HPC Docs, "Casper use policies").
