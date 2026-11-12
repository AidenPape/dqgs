"""
C3 finisher: regenerate the WHITENED partial-observation sweep and its figures.

Detached, restartable, single source of truth for the partial-obs result.
Run on the GPU env:
    JAX_ENABLE_X64=1 /glade/work/wchapman/conda-envs/run_graphcast/bin/python \
        -m error_discovery.finish_c3
or detached so it survives a dead shell:
    cd /glade/u/home/wchapman/help_aiden
    setsid bash -c 'JAX_ENABLE_X64=1 \
        /glade/work/wchapman/conda-envs/run_graphcast/bin/python -m error_discovery.finish_c3 \
        > error_discovery/results/logs/c3_sweep.log 2>&1' &

Writes results/data/sparsity_sparsity.npz, results/figures/fig_sparsity_noise_sparsity.png,
fig_sparsity_window_sparsity.png. Reads the cached attractor IC read-only (it does
NOT rebuild or clobber initial_conditions/attractor_ic_default_sp800000.npz, since the default
recipe matches).
Prints 'C3_DONE' on success so a watcher can grep for completion.
"""
import time

from .experiments import sparsity as S


def main():
    t0 = time.time()
    out = S.run(tag="sparsity", whiten=True, verbose=True)
    print("WROTE", out, flush=True)
    print(S.fig_sparsity_noise("sparsity"), flush=True)
    print(S.fig_sparsity_window("sparsity"), flush=True)
    print(f"C3_DONE in {(time.time() - t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
