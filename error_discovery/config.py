"""
error_discovery.config
======================

Single source of truth for the differentiable error-discovery experiment.

Times are quoted in both the model's nondimensional units and in days.  MAOOAM
uses f0 = 1.032e-4 s^-1, so one day = f0 * 86400 = 8.9165 nondim units, i.e.
89.16 integration steps at dt = 0.1.
"""

from dataclasses import dataclass
import os

# --- physical time conversion (MAOOAM) ------------------------------------- #
F0 = 1.032e-4                       # Coriolis parameter [s^-1]
NDIM_PER_DAY = F0 * 86400.0         # = 8.91648 nondim time units per day
DT = 0.1                            # integration step (nondim)
STEPS_PER_DAY = NDIM_PER_DAY / DT   # = 89.16 steps per day
DAYS_PER_YEAR = 365.25              # for records quoted in model-years


def steps_to_days(n_steps):
    return n_steps / STEPS_PER_DAY


def days_to_steps(days):
    return int(round(days * STEPS_PER_DAY))


# --- where results live (kept beside the code for easy re-runs) ------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
DATA_DIR = os.path.join(RESULTS_DIR, "data")          # .npz outputs, not tracked
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")    # .png, tracked: the manuscript includes them
IC_DIR = os.path.join(_HERE, "initial_conditions")    # spun-up states, tracked in git


def ic_path(config, steps, directory=IC_DIR):
    """Path of a spun-up state: ``attractor_ic_<config>_sp<steps>.npz``.

    ``config`` is a ``qgs_jax.spec.PARAM_SETS`` name, which fixes both the parameters and the
    truncation; a ``-ndim<N>`` suffix marks a state at a dimension other than that set's own.
    ``steps`` is the spin-up length in RK4 steps at ``DT``, written in full. The convention is
    documented in ``error_discovery/README.md``; build every name here, never by hand.
    """
    return os.path.join(directory, f"attractor_ic_{config}_sp{int(steps)}.npz")


IC_CACHE = ic_path("default", 800_000)     # the state every study starts from


# --- the sampling standard (plans/sampling.md; machinery in sampling.py) ----- #
# A long base-model run forward from the 1e9-step state, saved as daily snapshots, labelled
# passive / active on psi_a1's variability (Tondeur et al. 2020), from which each study draws
# its starting states.
SAMPLING_SOURCE = ("default", 1_000_000_000)   # (config, spin-up steps) the base run starts from
BASE_RUN_YEARS = 10_000             # length of the base run the studies currently sample from
CHUNK_YEARS = 1_000                 # one base-run file per this many model-years
SNAPSHOT_EVERY = 89                 # steps between saved snapshots (0.998 d)
REGIME_T_DAYS = (30, 100, 365, 1000)   # candidate running-std windows; the labels pick one
REGIME_CUTOFF = 0.9                 # passive / active only at >= this mixture probability
HIGHPASS_DAYS = 30                  # running mean removed before the noise-scale std (Brajard 2021)
TAU_C = 6                           # Sokal's window constant: sum the autocorrelation to M >= c tau
N_SAMPLES = 16                      # starting states per study and sampling mode
SPACING_K = 3                       # starting states at least k x the largest tau_int apart
MIN_EPISODE_DAYS = 365.25           # a regime run counts as an episode only if at least this long
ETA_GRID = (1e-4, 1e-3, 1e-2, 1e-1)    # noise levels x high-frequency std, sampled modes


def baserun_path(config, start_steps, yr0, yr1, directory=DATA_DIR):
    """Path of one base-run chunk: ``baserun_<config>_sp<steps>_yr<yr0>-<yr1>.npz``.

    ``config`` and ``start_steps`` name the spun-up state the run starts from (``ic_path``);
    ``yr0``-``yr1`` are model-years since the run began. Build every name here, never by hand.
    """
    return os.path.join(directory, f"baserun_{config}_sp{int(start_steps)}"
                                   f"_yr{int(yr0):05d}-{int(yr1):05d}.npz")


def regimes_path(config, start_steps, yr0, yr1, directory=DATA_DIR):
    """Path of the regime labels and record statistics for base-run years ``yr0``-``yr1``."""
    return os.path.join(directory, f"regimes_{config}_sp{int(start_steps)}"
                                   f"_yr{int(yr0):05d}-{int(yr1):05d}.npz")

# Form-identifiability margin below which a driver counts as unidentifiable.
#
# OPEN QUESTION (flagged 2026-08-13). This number is not derived. It is a round 10%:
# the manuscript defines it as "the truth beats every wrong form by more than 10% of
# the error magnitude", and says the continuous margin is what is reported and that
# the 0.1 label is "a secondary summary". No note anywhere derives it from the noise
# level, the bootstrap spread, or any statistical criterion.
#
# Where it is load-bearing rather than cosmetic: `screening.identifiable_count` (the
# "N of 36 drivers identifiable" headline) and `experiments/robustness.parsimony_floor`
# both apply it as a hard cut to a continuous quantity. Everywhere else it is a line
# on a plot.
#
# Worth investigating: (1) the studies already bootstrap the margin (reps=20-40), so a
# threshold could be set relative to its own sampling spread instead of fixed; (2) the
# governing law says margin ~ 1/(|mean|/std), so THRESH=0.1 is equivalent to cutting at
# |mean|/std ~ 10 — which may be the more natural place to put it, since that is a
# statement about the data distribution rather than about residuals; (3) the manuscript
# already concedes borderline cases flip between 0.03 and 0.3, but no sweep exists.
#
# Changing this one number now re-labels every verdict and every figure consistently,
# which it did not before 2026-08-13 (two verdict sites had it hardcoded).
THRESH = 0.1

# Amplitude of the injected error: lambda'_eff = lambda' * [1 + eps * g(x)].
# The whole form-discovery family uses this one value; the lead-time sweep reaches
# it through ``ExperimentConfig.eps`` so it can be overridden per run.
EPS = 0.1

# Initial-condition spin-up recipe (kept fixed so the cached attractor state is
# reproducible from scratch, independent of the noise seed).
IC_SEED = 1
IC_SCALE = 0.1


@dataclass
class ExperimentConfig:
    """All knobs for the lead-time error-discovery sweep."""

    # injected ("true") model error:  lambda'_eff = lambda' * [1 + eps * g(x)],
    # g(x) = a0 + a1*theta_{a,1} + a2*T_{o,1} + a3*psi_{a,1}.
    # Default matches the project's real experiment: g(x) = T_{o,1}.
    eps: float = EPS
    g_coeffs: tuple = (0.0, 0.0, 1.0, 0.0)

    dt: float = DT

    # spin-up onto the attractor (used only when the IC cache is absent).
    spinup_steps: int = 800_000

    # truth segments ("initial conditions"): start points spaced along one long
    # truth trajectory.  The spacing is derived from max(window_steps) inside
    # ``attractor.build_segments`` rather than set here, so it cannot drift out
    # of sync with the sweep; see that docstring.
    n_ic: int = 4

    # optimisation-window lengths (steps) to sweep -- spans under-determined
    # (<~2 d), optimal (~2 wk), and degraded (>~1 mo) regimes.
    window_steps: tuple = (40, 80, 160, 320, 480, 640, 960, 1280,
                           1920, 2560, 3840, 5120, 8000)

    # observation operator: "full" (all 36) or "atmos" (psi_a + theta_a).
    obs: str = "full"

    # observation-noise sweep (relative, per-component std).  eta=0 is the
    # noiseless main result; the rest probe how noise tightens the window.
    noise_levels: tuple = (1e-4, 3e-4, 1e-3)
    noise_windows: tuple = (320, 640, 1280, 2560, 5120)
    n_noise_seeds: int = 3

    # Levenberg-Marquardt iterations. Set high enough that the noiseless fit is
    # converged at every window: because the model class contains the truth, the
    # noiseless global minimum is at c_true with zero residual, and a converged
    # optimizer reaches it at all windows we test. A small budget (e.g. 12) instead
    # under-converges on the ill-conditioned long windows and produces a spurious
    # upper edge to the window; the real upper edge is a conditioning-times-noise
    # effect, not a noiseless limit. 40 iters reaches machine precision at every
    # window we test (12 under-converges on the ill-conditioned long windows).
    lm_iters: int = 40

    # Lyapunov-time estimator blocks (context diagnostic).
    lyap_blocks: int = 200

    seed: int = 21217
    tag: str = "main"

    def __post_init__(self):
        self.window_steps = tuple(int(w) for w in self.window_steps)
        self.noise_windows = tuple(int(w) for w in self.noise_windows)


# A small, fast configuration for smoke tests / CI (uses the cached IC).
QUICK = ExperimentConfig(
    spinup_steps=50_000,
    n_ic=2,
    window_steps=(160, 1280, 5120),
    noise_levels=(1e-3,),
    noise_windows=(1280,),
    n_noise_seeds=1,
    lm_iters=8,
    lyap_blocks=20,
    tag="quick",
)
