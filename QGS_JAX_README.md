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

Requires `jax`, `numba`, `sparse`, `pebble` (the original qgs deps) — all
installed in the active `base` env. `numpy` x64 is used throughout
(`jax_enable_x64`).

## Notes / scope

- Targets the standard atmosphere(2×2) + ocean(2×4) MAOOAM config, `dynamic_T=False`,
  no ground, no orography — the configuration in `qgs/qgs_maooam.py`. The assembly
  raises `NotImplementedError` for unsupported configs.
- The original `qgs` package is left untouched; everything lives in `help_aiden/`.
- `Cpa`/`Cpgo` insolation are tuned as scalar amplitudes (`Cpa0`,`Cpgo0`) of their
  fixed spatial pattern.
