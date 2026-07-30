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


def steps_to_days(n_steps):
    return n_steps / STEPS_PER_DAY


def days_to_steps(days):
    return int(round(days * STEPS_PER_DAY))


# --- where results live (kept inside help_aiden for easy re-runs) ----------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
IC_CACHE = os.path.join(RESULTS_DIR, "attractor_ic.npz")

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
    eps: float = 0.1
    g_coeffs: tuple = (0.0, 0.0, 1.0, 0.0)

    dt: float = DT

    # spin-up onto the attractor (used only when the IC cache is absent).
    spinup_steps: int = 800_000

    # truth segments ("initial conditions"): independent start points spaced
    # along one long truth trajectory.
    n_ic: int = 4
    ic_spacing_steps: int = 3_000

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
