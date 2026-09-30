"""Tests for the spun-up state files (attractor.py, config.ic_path).

The long spin-up is I/O that would silently corrupt every result built on it: a resume that
drifts, a loader that accepts a state from another recipe, a name built by hand. These pin
those down on a 1,000-step run."""

import os

import numpy as np
import jax.numpy as jnp
import pytest

from error_discovery.config import IC_CACHE, IC_SEED, IC_SCALE, DT, ic_path
from error_discovery.attractor import spin_up, load_spun_up_ic, _recipe

N, TRAIL, CKPT = 1000, 100, 500


def test_ic_path_convention():
    assert os.path.basename(ic_path("default", 800_000)) == "attractor_ic_default_sp800000.npz"
    assert ic_path("default", 800_000) == IC_CACHE


def test_default_set_is_the_twin_tensor():
    """spin_up builds its tensor from PARAM_SETS["default"]; the studies use the twin's."""
    from qgs_jax.spec import extract_spec, build_reference_params
    from error_discovery.formfit import twin
    T = extract_spec(build_reference_params("default")).dense_tensor
    assert np.array_equal(T, np.asarray(twin().T_base))


@pytest.fixture(scope="module")
def straight(tmp_path_factory):
    """One uninterrupted 1,000-step spin-up, and the file it wrote."""
    d = tmp_path_factory.mktemp("straight")
    x = spin_up("default", N, trail_every=TRAIL, checkpoint_every=CKPT, directory=str(d),
                verbose=False)
    return np.asarray(x), np.load(ic_path("default", N, str(d))), str(d)


def test_matches_integrate_final(straight):
    """Tensor as a compile-time constant vs as an argument: same model, rounding only."""
    from qgs_jax import model as jm
    from error_discovery.formfit import twin
    x, f, _ = straight
    x0 = jnp.asarray(np.random.default_rng(IC_SEED).standard_normal(36) * IC_SCALE)
    ref = np.asarray(jm.integrate_final(twin().T_base, x0, DT, N))
    assert np.max(np.abs(x - ref)) < 1e-13
    assert np.array_equal(f["trail"][-1], x)
    assert f["trail"].shape == (N // TRAIL, 36)


def test_resume_is_bit_identical(straight, tmp_path):
    """A partial holding the straight run's state at 500 steps resumes to the same end."""
    x, f, _ = straight
    k = CKPT // TRAIL
    partial = ic_path("default", N, str(tmp_path))[:-len(".npz")] + ".partial.npz"
    np.savez(partial, x=f["trail"][k - 1], steps_done=CKPT, trail=f["trail"][:k],
             trail_every=TRAIL, ndim=36, **_recipe("default", N))
    xr = spin_up("default", N, trail_every=TRAIL, checkpoint_every=CKPT,
                 directory=str(tmp_path), verbose=False)
    fr = np.load(ic_path("default", N, str(tmp_path)))
    assert np.array_equal(np.asarray(xr), x)
    assert np.array_equal(fr["trail"], f["trail"])
    assert not os.path.exists(partial)


def test_loader_checks_recipe_and_never_builds(straight, tmp_path):
    x, f, d = straight
    assert np.array_equal(np.asarray(load_spun_up_ic("default", N, d)), x)

    tampered = {k: f[k] for k in f.files}
    tampered["ic_seed"] = 2
    np.savez(ic_path("default", N, str(tmp_path)), **tampered)
    with pytest.raises(ValueError, match="ic_seed"):
        load_spun_up_ic("default", N, str(tmp_path))

    with pytest.raises(FileNotFoundError):
        load_spun_up_ic("default", 2 * N, d)
    with pytest.raises(FileExistsError):
        spin_up("default", N, trail_every=TRAIL, checkpoint_every=CKPT, directory=d,
                verbose=False)


def test_loader_refuses_the_800k_cache():
    """Files that predate spin_up store no params, so the strict loader rejects them."""
    with pytest.raises(ValueError, match="params"):
        load_spun_up_ic("default", 800_000)
