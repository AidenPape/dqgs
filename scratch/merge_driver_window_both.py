"""Noisy fits on all 13 windows for ``fig_driver_conditioning_recovery_both``, panel (b).

``driver_window_both.npz`` has the noiseless sweep for all five drivers and noisy fits on the 5
default noise windows only. The other two sources cover all 13 windows at eta = 1e-4, on the
same 16 states and noise draws:

- T_o1 and theta_a1: Stage 1 (``driver_window_stage1_<driver>_both.npz``), seeds 0-1 of its 3;
- const, psi_a1 and mixed: ``driver_window_both_noise13.npz`` (noisy only).

This rewrites ``driver_window_both.npz`` with every ``*_noise`` array on 13 windows, after
checking that the 5 windows it already had come through bit-identical, and redraws the figure.
``noise_source`` records which file each driver's noisy fits came from.

    python scratch/merge_driver_window_both.py
"""

import os

import numpy as np

from error_discovery.config import DATA_DIR
from error_discovery.experiments import driver_window

ETA_INDEX = 0                        # eta = 1e-4 in every source
N_SEEDS = 2


def _path(tag):
    return os.path.join(DATA_DIR, f"driver_window_{tag}.npz")


def _load(tag):
    with np.load(_path(tag), allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def merge():
    base = _load("both")
    rest = _load("both_noise13")
    stage1 = {drv: _load(f"stage1_{drv}_both") for drv in ("T_o1", "theta_a1")}
    for d in (rest, *stage1.values()):
        assert np.array_equal(d["noise_windows"], base["windows"])
        assert np.array_equal(d["sample_snapshot_index"], base["sample_snapshot_index"])
        assert d["noise_levels"][ETA_INDEX] == base["noise_levels"][ETA_INDEX]
        assert bool(d["whiten"]) == bool(base["whiten"])

    rest_drivers = [str(x) for x in rest["drivers"]]
    drivers = [str(x) for x in base["drivers"]]
    source = ["driver_window_both_noise13" if drv in rest_drivers
              else f"driver_window_stage1_{drv}_both" for drv in drivers]

    out = dict(base)
    shared = [list(base["windows"]).index(w) for w in base["noise_windows"]]
    for k in (k for k in base if k.endswith("_noise")):
        rows = [rest[k][rest_drivers.index(drv), ETA_INDEX:ETA_INDEX + 1]
                if drv in rest_drivers
                else stage1[drv][k][0, ETA_INDEX:ETA_INDEX + 1, :, :, :N_SEEDS]
                for drv in drivers]
        out[k] = np.stack(rows)
        assert np.array_equal(out[k][:, :, shared], base[k], equal_nan=True), k
    assert not np.isnan(out["rel_err_noise"]).any()

    out["noise_windows"] = base["windows"]
    out["noise_window_days"] = base["window_days"]
    out["noise_source"] = np.array(source)
    np.savez(_path("both"), **out)


if __name__ == "__main__":
    merge()
    print(driver_window.figure(tag="both"))
