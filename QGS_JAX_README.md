# qgs_jax — a differentiable JAX MAOOAM model

`qgs_jax` is a JAX re-implementation of the MAOOAM atmosphere–ocean configuration
of [`qgs`](https://github.com/Climdyn/qgs). It is **1:1 with the original code to
machine precision** and **differentiable end-to-end**, so the model can be used
for gradient-based (automatic) parameter tuning, tangent-linear/adjoint
sensitivity, and ML-in-the-loop experiments.

## The key idea

The whole MAOOAM model (with `dynamic_T = False`) is a single sparse rank-3
"tendency tensor" `T`:

```
dx_i/dt = Σ_{j,k} T[i,j,k] · x̄_j · x̄_k ,    x̄ = (1, x₁, …, x_n)
```

`qgs_jax` (a) extracts that tensor from the original numpy/numba code for an
exact reference, and (b) **re-assembles it in pure JAX as a differentiable
function of the physical parameters**, so gradients flow all the way from a
trajectory loss back to parameters like `kd`, `sig0`, `Lpa`, `G`, ….

## Layout

| File | Purpose |
|------|---------|
| `qgs_jax/spec.py`     | Build a reference `QgParams`, run original `qgs`, extract the dense tensor + inner-product geometry + scalar parameters into a portable `ModelSpec`. |
| `qgs_jax/model.py`    | JAX tendencies `f(x)`, analytic + autodiff Jacobian, RK2/RK4 integrators (`jax.lax.scan`, jit-able, differentiable). |
| `qgs_jax/assemble.py` | Pure-JAX, parameter-differentiable re-assembly of the tendency tensor (bit-exact with `qgs`). |
| `test_qgs_jax.py`     | pytest suite verifying 1:1 agreement and gradient correctness. |
| `tune_demo.py`        | End-to-end automatic parameter tuning demo (recovers corrupted parameters by gradient descent). |

## Parameter sets

`qgs_jax.spec.PARAM_SETS` names every configuration a study can ask for, and
`build_reference_params(name)` turns a name into a qgs `QgParams`. Anything built from a
set carries the same name: the spun-up state of the `default` set after 800,000 steps is
`attractor_ic_default_sp800000.npz` (naming convention in `error_discovery/README.md`).

### `default`

The configuration every study uses unless it names another. Its values are the
`DEFAULT_*` dictionaries in `qgs_jax/spec.py`:

| parameter | qgs name | value | units as given to qgs |
|---|---|---|---|
| atmosphere truncation | `set_atmospheric_channel_fourier_modes` | 2 x 2 (10 psi_a, 10 theta_a) | |
| ocean truncation | `set_oceanic_basin_fourier_modes` | 2 x 4 (8 psi_o, 8 T_o) | |
| state dimension | `ndim` | 36 | |
| atmosphere surface friction k_d | `kd` | 0.0290 | nondimensional (x f0 = 2.99e-6 s^-1) |
| atmosphere internal friction k'_d | `kdp` | 0.0290 | nondimensional |
| channel aspect ratio n = 2 L_y / L_x | `n` | 1.5 | nondimensional |
| ocean bottom friction r | `r` | 1.0e-7 | s^-1 |
| ocean layer depth h | `h` | 136.5 | m |
| ocean-atmosphere mechanical coupling d | `d` | 1.1e-7 | s^-1 |
| atmosphere emissivity eps_a | `eps` | 0.7 | nondimensional |
| atmosphere reference temperature T0_a | `T0` (atemperature) | 289.3 | K |
| ocean-atmosphere heat exchange lambda | `hlambda` | 15.06 | W m^-2 K^-1 |
| ocean heat capacity gamma_o | `gamma` (gotemperature) | 5.6e8 | J m^-2 K^-1 |
| ocean reference temperature T0_o | `T0` (gotemperature) | 301.46 | K |
| atmosphere short-wave input C_a (first component) | `set_insolation` | 103.3333 (= C_o / 3) | W m^-2 |
| ocean short-wave input C_o (first component) | `set_insolation` | 310.0 | W m^-2 |
| atmosphere heat capacity gamma_a | not set; qgs default | 1e7 | J m^-2 K^-1 |
| Coriolis parameter f0 | not set; qgs default | 1.032e-4 | s^-1 |

Every study integrates at `dt = 0.1` (nondimensional, 16.2 min); that is a choice of the
studies (`error_discovery/config.py`), not part of the set.

**Source.** The values are exactly those of upstream qgs v1.0.0's `qgs_maooam.py` example
(checked against the v1.0.0 tag), which cites:

- Vannitsem, S., Demaeyer, J., De Cruz, L., & Ghil, M. (2015). Low-frequency variability
  and heat transport in a low-order nonlinear coupled ocean-atmosphere model. *Physica D*,
  309, 71-85. doi:10.1016/j.physd.2015.07.006
- De Cruz, L., Demaeyer, J., & Vannitsem, S. (2016). The Modular Arbitrary-Order
  Ocean-Atmosphere Model: MAOOAM v1.0. *Geosci. Model Dev.*, 9, 2793-2808.
  doi:10.5194/gmd-9-2793-2016

The set is not De Cruz et al. 2016's Table 1: that table has gamma_o = 5.46e8 and
k_d = 3.0e-6 s^-1, which is 0.02907 nondimensional rather than 0.0290. The values have not
been checked against Vannitsem et al. 2015's own parameter table.

### The other sets

Each is documented in full in the comment above it in `qgs_jax/spec.py`.

- `decruz2018_nodissip`: De Cruz et al. 2018 (*Nonlin. Processes Geophys.* 25, 387-412),
  experiment "nodissip": their Table 3 plus d = 1.1e-7. Atmosphere 2 x 4, ocean 2 x 4,
  ndim 56. They integrate at dt = 0.2.
- `tondeur2020_36wk`, `tondeur2020_36st`: Tondeur et al. 2020 (*J. Stat. Phys.* 179,
  1161), Tables 1-2, read literally. Atmosphere 2 x 2, ocean 2 x 4, ndim 36. How their
  friction coefficients map onto qgs's `kd` and `kdp` is not verified against the code they
  ran.

## Verification status

`pytest test_qgs_jax.py` — **15/15 pass**. It checks, to ~1e-12:

1. Tendencies `f(x)`: JAX vs original qgs.
2. Jacobian `Df(x)`: JAX-analytic vs JAX-autodiff vs original qgs.
3. Full RK4 trajectories (1500 steps): JAX vs original `RungeKuttaIntegrator`
   (max abs diff ≈ 1e-16).
4. Differentiable assembly == original qgs tensor (symmetrised diff = 0).
5. Parameter gradients vs finite differences (incl. `sig0`, `G` which flow
   through matrix inverses).
6. Gradient of a trajectory loss w.r.t. a physical parameter.

`tune_demo.py` recovers `kd`, `kdp`, `Lpa` (corrupted ±25–30%) to ~2e-8 relative
error via Adam through the integrator (loss 4.7e-2 → 1e-16).

## Quick start

```python
from qgs_jax.spec import extract_spec
from qgs_jax import model, assemble
import jax, jax.numpy as jnp

spec = extract_spec()                                   # numpy reference (runs once)
geom, theta0, tensor_of_theta = assemble.make_tensor_fn(spec)

# Differentiable in physical parameters:
def trajectory_loss(theta):
    T = tensor_of_theta(theta)
    traj = model.integrate(T, x0, dt=0.1, n_steps=100, method="rk4")
    return jnp.sum((traj - target)**2)

grad = jax.grad(trajectory_loss)(theta0)                # dL/dθ for every parameter
```

## Environment

Dependencies are declared in `pyproject.toml`; `pip install -e .` from the repo root
provides them, including the pinned `qgs==1.0.0`. `numpy` x64 is used throughout
(`jax_enable_x64`).

## Notes / scope

- Targets the standard atmosphere(2×2) + ocean(2×4) MAOOAM config, `dynamic_T=False`,
  no ground, no orography — the configuration in upstream qgs's `qgs_maooam.py` example.
  The assembly raises `NotImplementedError` for unsupported configs.
- The `qgs` package is an unmodified third-party dependency, pinned to 1.0.0.
- `Cpa`/`Cpgo` insolation are tuned as scalar amplitudes (`Cpa0`,`Cpgo0`) of their
  fixed spatial pattern.
