"""
Fig-2 finisher: regenerate the FULL lead-time sweep (noiseless recovery + conditioning
+ noise recovery) with WHITENED residuals, so every curve on Figure 2 is the
properly-weighted estimate. The conditioning panels report cond(J_w), sigma_min,
sigma_max of the weighted Jacobian J_w = W J (W = diag(1/std_channel)); the overlaid
noisy-recovery curves are the weighted least-squares fits at the same weighting --
consistent with the whitened partial-obs result (C3) and the whitened information
geometry (C2).

Why whitening changes the geometry but not the noiseless recovery:
  * The noiseless minimum is c_true with zero residual under ANY positive channel
    weighting, so the recovered coefficients stay at machine precision at every window.
  * The Jacobian the SVD sees changes: J_w = W J reweights each observed channel by
    1/std. Unwhitened (R = I) the small-amplitude ocean-temperature channel that carries
    the injected error is numerically buried near sigma_min; whitening lifts it to O(1),
    so cond(J_w) reflects identifiability rather than the dynamic range of channel
    amplitudes. This is the conditioning a properly-weighted (R = diag(std^2))
    Gauss-Newton solver actually faces.

Writes a SEPARATE tag 'main_whiten' (does NOT clobber leadtime_main.npz). Reads the
cached attractor IC read-only. Prints 'FIG2_WHITEN_DONE' on success.

Run on the GPU env:
    JAX_ENABLE_X64=1 python -m error_discovery.finish_fig2_whiten
"""
import time
from dataclasses import replace

from .config import ExperimentConfig
from .experiments import leadtime as E

TAG = "main_whiten"


# Extended window set: the standard windows PLUS long windows out to ~270 days, so the
# WHITENED conditioning U closes on the right. Under whitening the right arm (sigma_max
# growth from chaotic stretching, sigma_min saturated) is gentle, so cond(J_w) only rises
# a decade above its minimum once windows reach ~180-270 days; the standard sweep (<=90 d)
# stops before that. One extra long noise window (8000, ~90 d) tracks the noisy recovery
# past the conditioning minimum.
_WINDOWS_EXT = (40, 80, 160, 320, 480, 640, 960, 1280, 1920, 2560, 3840, 5120, 8000,
                11000, 16000, 24000)
_NOISE_WINDOWS_EXT = (320, 640, 1280, 2560, 5120, 8000)


def main():
    t0 = time.time()
    cfg = replace(ExperimentConfig(), tag=TAG,
                  window_steps=_WINDOWS_EXT, noise_windows=_NOISE_WINDOWS_EXT)
    out = E.run(cfg, verbose=True, whiten=True)
    print("WROTE", out, flush=True)
    print(E.fig_recovery_and_conditioning(TAG), flush=True)
    print(E.fig_noise(TAG), flush=True)
    print(E.fig_coeff_recovery(TAG), flush=True)
    print(f"FIG2_WHITEN_DONE in {(time.time() - t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
