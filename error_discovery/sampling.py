"""
error_discovery.sampling
========================

The sampling standard (``plans/sampling.md``): where every study's starting states come from.

1. ``base_run`` integrates the base model forward from the 1e9-step ``default`` state
   (``config.SAMPLING_SOURCE``) and saves the full state once per simulated day, one file per
   ``CHUNK_YEARS``. A starting state is a snapshot index into this record.
2. ``label_regimes`` labels every snapshot passive or active from the variability of psi_a1,
   the regimes of Tondeur et al. (2020): a centred running standard deviation over T, a
   two-component Gaussian mixture on its logarithm, and the probability of the high-variability
   (active) component. T is the candidate in ``REGIME_T_DAYS`` whose two components overlap
   least. The same file holds the two record statistics sampling needs: each variable's
   integrated autocorrelation time (Sokal windowing, Madras-Sokal error) and its
   high-frequency standard deviation (after removing a 1-month running mean, as Brajard et al.
   2021 and He et al. 2025 scale their observation noise).
3. ``starting_states`` picks ``n`` snapshots for a mode: ``"passive"`` or ``"active"`` (episode
   onsets by default) or ``"both"`` (every snapshot, whatever its label), at least
   ``spacing_k`` times the largest tau_int apart.

Every choice here, and why, is in ``plans/sampling.md``; the published ones are in
``notes/sampling_literature.md``.
"""

import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from .config import (RESULTS_DIR, DT, STEPS_PER_DAY, DAYS_PER_YEAR, SAMPLING_SOURCE,
                     BASE_RUN_YEARS, CHUNK_YEARS, SNAPSHOT_EVERY, REGIME_T_DAYS, REGIME_CUTOFF,
                     HIGHPASS_DAYS, TAU_C, N_SAMPLES, SPACING_K, MIN_EPISODE_DAYS,
                     baserun_path, regimes_path)
from .attractor import load_spun_up_ic, _recipe, _mismatched

PSI_A1 = 0                          # psi_a1 is the first state variable


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #
def snapshot_days(snapshot_every=SNAPSHOT_EVERY):
    return snapshot_every / STEPS_PER_DAY


def snapshots_per_chunk(chunk_years=CHUNK_YEARS, snapshot_every=SNAPSHOT_EVERY):
    """Whole snapshots in one chunk, the nearest to ``chunk_years`` model-years."""
    return int(round(chunk_years * DAYS_PER_YEAR / snapshot_days(snapshot_every)))


def odd_window(days, snapshot_every=SNAPSHOT_EVERY):
    """A centred window of ``days``, in snapshots, rounded to the nearest odd count."""
    n = int(round(days / snapshot_days(snapshot_every)))
    return n + 1 - n % 2


# --------------------------------------------------------------------------- #
# 1. The base run
# --------------------------------------------------------------------------- #
def _chunk_recipe(config, start_steps, chunk_years, snapshot_every):
    r = _recipe(config, start_steps)
    r.update(chunk_years=int(chunk_years), snapshot_every=int(snapshot_every))
    return r


def base_run(n_years=BASE_RUN_YEARS, chunk_years=CHUNK_YEARS, snapshot_every=SNAPSHOT_EVERY,
             directory=RESULTS_DIR, verbose=True):
    """Integrate the base model forward from ``SAMPLING_SOURCE``, saving daily snapshots.

    Writes one file per chunk (``config.baserun_path``) holding ``snapshots`` (state at every
    ``snapshot_every`` steps, the first being the chunk's starting state), ``x_next`` (the
    state that starts the next chunk) and the recipe. A chunk that already exists is checked
    and skipped, never rewritten, so a longer ``n_years`` extends the record forward and an
    interrupted run resumes at the first missing chunk; nothing already written changes.

    The tensor is closed over, as in ``attractor.spin_up``, so it is a compile-time constant.
    """
    from qgs_jax import model as jm
    from qgs_jax.spec import extract_spec, build_reference_params

    config, start_steps = SAMPLING_SOURCE
    if n_years % chunk_years:
        raise ValueError("n_years must be a whole number of chunks")
    per_chunk = snapshots_per_chunk(chunk_years, snapshot_every)
    recipe = _chunk_recipe(config, start_steps, chunk_years, snapshot_every)
    T = jnp.asarray(extract_spec(build_reference_params(config)).dense_tensor)

    def f(x):
        return jm.tendencies(T, x)

    def hop(x, _):              # emit the state, then advance it to the next snapshot
        nxt = jax.lax.fori_loop(0, snapshot_every, lambda i, y: jm.rk4_step(f, y, DT), x)
        return nxt, x

    @jax.jit
    def advance(x):
        x_next, snaps = jax.lax.scan(hop, x, None, length=per_chunk)
        return snaps, x_next

    os.makedirs(directory, exist_ok=True)
    x = load_spun_up_ic(config, start_steps)
    for c in range(n_years // chunk_years):
        yr0, yr1 = c * chunk_years, (c + 1) * chunk_years
        path = baserun_path(config, start_steps, yr0, yr1, directory)
        if os.path.exists(path):
            d = np.load(path)
            bad = _mismatched(d, recipe)
            if bad:
                raise ValueError(f"{path} was built with a different recipe: {bad}")
            if not np.array_equal(d["snapshots"][0], np.asarray(x)):
                raise ValueError(f"{path} does not continue the chunk before it")
            x = jnp.asarray(d["x_next"])
            continue
        t0 = time.time()
        snaps, x = advance(x)
        tmp = path[:-len(".npz")] + ".tmp.npz"
        np.savez(tmp, snapshots=np.asarray(snaps), x_next=np.asarray(x), chunk=c,
                 first_snapshot=c * per_chunk, **recipe)
        os.replace(tmp, path)                     # atomic: a kill never leaves half a file
        if verbose:
            rate = per_chunk * snapshot_every / (time.time() - t0)
            print(f"[base]  years {yr0:,}-{yr1:,} written  {rate:,.0f} steps/s  ({path})",
                  flush=True)
    return x


def load_base_run(n_years=BASE_RUN_YEARS, columns=None, chunk_years=CHUNK_YEARS,
                  snapshot_every=SNAPSHOT_EVERY, directory=RESULTS_DIR):
    """The base-run snapshots for the first ``n_years``, as one (N, ndim) array.

    ``columns`` selects state variables (e.g. ``[PSI_A1]``) to save memory. Every chunk's
    recipe is checked, and each chunk must begin exactly where the one before it ended.
    """
    config, start_steps = SAMPLING_SOURCE
    recipe = _chunk_recipe(config, start_steps, chunk_years, snapshot_every)
    parts, x_prev = [], None
    for c in range(n_years // chunk_years):
        path = baserun_path(config, start_steps, c * chunk_years, (c + 1) * chunk_years,
                            directory)
        d = np.load(path)
        bad = _mismatched(d, recipe)
        if bad:
            raise ValueError(f"{path} was built with a different recipe: {bad}")
        snaps = d["snapshots"]
        if x_prev is not None and not np.array_equal(snaps[0], x_prev):
            raise ValueError(f"{path} does not continue the chunk before it")
        x_prev = d["x_next"]
        parts.append(snaps if columns is None else snaps[:, columns])
    return np.concatenate(parts)


def load_snapshots(indices, chunk_years=CHUNK_YEARS, snapshot_every=SNAPSHOT_EVERY,
                   directory=RESULTS_DIR):
    """The full states at the given global snapshot indices, reading only the chunks needed."""
    config, start_steps = SAMPLING_SOURCE
    per_chunk = snapshots_per_chunk(chunk_years, snapshot_every)
    out, cache = [], {}
    for i in indices:
        c = int(i) // per_chunk
        if c not in cache:
            cache[c] = np.load(baserun_path(config, start_steps, c * chunk_years,
                                            (c + 1) * chunk_years, directory))["snapshots"]
        out.append(cache[c][int(i) - c * per_chunk])
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# Pure numeric pieces (each pinned in tests/test_sampling.py)
# --------------------------------------------------------------------------- #
def running_mean(x, w):
    """Centred running mean over an odd window of ``w`` samples, along axis 0; NaN within
    ``w // 2`` of either end."""
    x = np.asarray(x, dtype=float)
    h = w // 2
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    out = np.full(x.shape, np.nan)
    out[h:len(x) - h] = (c[w:] - c[:-w]) / w
    return out


def running_std(x, w):
    """Centred running standard deviation (ddof 0) over an odd window of ``w`` samples;
    NaN within ``w // 2`` of either end. The series is centred first, so the running sums
    of x and x^2 do not cancel catastrophically."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=0)
    var = running_mean(x * x, w) - running_mean(x, w) ** 2
    return np.sqrt(np.maximum(var, 0.0))


def highpass_std(X, w):
    """Each column's standard deviation after removing its centred running mean over ``w``
    samples: the climatological std of high frequencies (Brajard et al. 2021, sigma_hf)."""
    X = np.asarray(X, dtype=float)
    r = X - running_mean(X, w)
    return np.nanstd(r, axis=0)


def tau_int(x, c=TAU_C):
    """Integrated autocorrelation time with Sokal's automatic window, in samples.

    tau_int = 1/2 + sum_{t=1..M} rho(t), with M the smallest window satisfying M >= c tau_int(M);
    under this convention an N-sample record holds N / (2 tau_int) effectively independent
    samples. rho uses the unbiased autocovariance (divide lag t by N - t). The Madras-Sokal
    error is tau sqrt(2 (2M + 1) / N). Returns (tau, err, M, found); ``found`` is False when no
    window up to N - 1 satisfies the rule, i.e. the record is too short to measure tau.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(F * np.conj(F), nfft)[:n] / (n - np.arange(n))
    rho = acov / acov[0]
    taus = 0.5 + np.cumsum(rho[1:])               # taus[M - 1] = tau_int(M)
    M = np.arange(1, n)
    ok = M >= c * taus
    found = bool(ok.any())
    m = int(M[np.argmax(ok)]) if found else n - 1
    tau = float(taus[m - 1])
    return tau, tau * np.sqrt(2.0 * (2 * m + 1) / n), m, found


def fit_mixture(values, seed=0):
    """Two-component Gaussian mixture on 1-D ``values`` (finite entries only).

    Returns (means, stds, weights, model), with component 1 the one of larger mean: on
    log(running std of psi_a1), component 1 is the high-variability (active) regime.
    """
    from sklearn.mixture import GaussianMixture
    v = np.asarray(values, dtype=float)
    gm = GaussianMixture(n_components=2, random_state=seed).fit(v[np.isfinite(v)][:, None])
    order = np.argsort(gm.means_[:, 0])
    means = gm.means_[order, 0]
    stds = np.sqrt(gm.covariances_[order, 0, 0])
    return means, stds, gm.weights_[order], (gm, order)


def mixture_prob_high(model, values):
    """Probability of the larger-mean component at each value; NaN where the value is not."""
    gm, order = model
    v = np.asarray(values, dtype=float)
    p = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    p[ok] = gm.predict_proba(v[ok][:, None])[:, order[1]]
    return p


def regime_labels(p_active, cutoff=REGIME_CUTOFF):
    """+1 active (p >= cutoff), -1 passive (p <= 1 - cutoff), 0 neither or unlabelled."""
    p = np.asarray(p_active)
    lab = np.zeros(p.shape, dtype=np.int8)
    lab[p >= cutoff] = 1
    lab[p <= 1 - cutoff] = -1
    return lab


def episodes(labels, min_len):
    """Regime episodes as ``[(label, start, end), ...]``, ``end`` exclusive.

    Runs of a constant nonzero label shorter than ``min_len`` are flickers and are ignored;
    consecutive surviving runs of the same label are one episode, the stretch between them
    included. Consecutive episodes therefore alternate between the two regimes.
    """
    lab = np.asarray(labels)
    cut = np.flatnonzero(np.diff(lab) != 0) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(lab)]
    merged = []
    for s, e in zip(starts, ends):
        r = int(lab[s])
        if r == 0 or e - s < min_len:
            continue
        if merged and merged[-1][0] == r:
            merged[-1] = (r, merged[-1][1], int(e))
        else:
            merged.append((r, int(s), int(e)))
    return merged


def onsets(labels, regime, min_len):
    """Start of every ``regime`` episode that follows an episode of the other regime.

    The first episode in the record has no onset: nothing shows it began there.
    """
    eps = episodes(labels, min_len)
    return [s for i, (r, s, _) in enumerate(eps) if r == regime and i > 0]


def pick_spaced(candidates, spacing, group=None):
    """Greedy forward pick: the first candidate, then each next one at least ``spacing``
    after the last pick and, if ``group`` is given, in a group not picked yet."""
    picks, last, used = [], -np.inf, set()
    for i in candidates:
        g = None if group is None else group[i]
        if i - last >= spacing and g not in used:
            picks.append(int(i))
            last = i
            if group is not None:
                used.add(g)
    return picks


# --------------------------------------------------------------------------- #
# 2. Labels and record statistics
# --------------------------------------------------------------------------- #
def label_regimes(n_years=BASE_RUN_YEARS, t_days=REGIME_T_DAYS, cutoff=REGIME_CUTOFF,
                  n_bins=200, directory=RESULTS_DIR, verbose=True):
    """Label the first ``n_years`` of the base run and write ``config.regimes_path``.

    For every candidate T: the running std of psi_a1, the mixture on its logarithm, and the
    share of labelled snapshots with p_active strictly between 1 - cutoff and cutoff (the
    ambiguous share). The T with the smallest share is chosen; its running std and p_active
    are saved per snapshot, the other candidates as mixture parameters and histograms only.
    Also saved per variable: full std, high-frequency std, and tau_int with its error.
    """
    config, start_steps = SAMPLING_SOURCE
    t0 = time.time()
    X = load_base_run(n_years, directory=directory)
    psi = X[:, PSI_A1]
    n_T = len(t_days)
    means, stds, weights = np.zeros((n_T, 2)), np.zeros((n_T, 2)), np.zeros((n_T, 2))
    share = np.zeros(n_T)
    edges = np.zeros((n_T, n_bins + 1))
    counts = np.zeros((n_T, n_bins))
    best = None
    for j, T in enumerate(t_days):
        logs = np.log(running_std(psi, odd_window(T)))
        means[j], stds[j], weights[j], model = fit_mixture(logs)
        p = mixture_prob_high(model, logs)
        ok = np.isfinite(p)
        share[j] = np.mean((p[ok] > 1 - cutoff) & (p[ok] < cutoff))
        counts[j], edges[j] = np.histogram(logs[np.isfinite(logs)], bins=n_bins)
        if best is None or share[j] < share[best[0]]:
            best = (j, np.exp(logs), p)
        if verbose:
            print(f"[label] T={T:5.0f} d  ambiguous share {share[j]:.4f}  "
                  f"means {np.exp(means[j])}  weights {weights[j]}", flush=True)
    j_best, s_best, p_best = best

    tau = np.zeros(X.shape[1])
    tau_err = np.zeros(X.shape[1])
    tau_M = np.zeros(X.shape[1], dtype=int)
    tau_found = np.zeros(X.shape[1], dtype=bool)
    for i in range(X.shape[1]):
        tau[i], tau_err[i], tau_M[i], tau_found[i] = tau_int(X[:, i])
    hf = highpass_std(X, odd_window(HIGHPASS_DAYS))

    path = regimes_path(config, start_steps, 0, n_years, directory)
    np.savez(path, t_days=np.asarray(t_days, dtype=float),
             t_snapshots=np.asarray([odd_window(T) for T in t_days]),
             gm_means=means, gm_stds=stds, gm_weights=weights, ambiguous_share=share,
             hist_counts=counts, hist_edges=edges, chosen=j_best,
             running_std=s_best, p_active=p_best, cutoff=cutoff,
             std=X.std(axis=0), hf_std=hf, highpass_days=HIGHPASS_DAYS,
             tau_int=tau, tau_err=tau_err, tau_window=tau_M, tau_found=tau_found,
             tau_c=TAU_C, snapshot_days=snapshot_days(), n_years=n_years,
             config=config, start_steps=start_steps)
    if verbose:
        dd = snapshot_days()
        k = int(np.argmax(tau))
        print(f"[label] chose T={t_days[j_best]} d; largest tau_int {tau[k] * dd:,.1f} d "
              f"(variable {k}, +/-{tau_err[k] * dd:,.1f}); wrote {path} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return path


def load_regimes(n_years=BASE_RUN_YEARS, directory=RESULTS_DIR):
    config, start_steps = SAMPLING_SOURCE
    return np.load(regimes_path(config, start_steps, 0, n_years, directory))


def noise_scale(n_years=BASE_RUN_YEARS, directory=RESULTS_DIR):
    """Per-variable high-frequency climatological std: the sampling standard's noise and
    whitening scale."""
    return np.asarray(load_regimes(n_years, directory)["hf_std"])


def shared_noise(snapshot_index, seed_index, n_steps, ndim, seed=21217):
    """Standard-normal noise for one starting state and seed index: common random numbers,
    reused for every window (as a prefix), noise level and driver on that state."""
    return np.random.default_rng([seed, int(snapshot_index), int(seed_index)]).standard_normal(
        (n_steps, ndim))


# --------------------------------------------------------------------------- #
# 3. Starting states
# --------------------------------------------------------------------------- #
def available_picks(mode, spacing_k=SPACING_K, min_episode_days=MIN_EPISODE_DAYS,
                    placement="onset", window_steps=0, n_years=BASE_RUN_YEARS,
                    directory=RESULTS_DIR):
    """Every starting state a mode could use, in order, and the regimes file they came from.

    ``mode`` is ``"passive"``, ``"active"`` or ``"both"``. ``window_steps`` is everything a
    segment integrates from its starting state (any ``f_true`` discard plus the longest
    window); the whole of it must fit in the record and, in a regime mode, keep the regime's
    label on the base run. ``placement="onset"`` takes episode onsets; ``"anywhere"`` takes any
    such snapshot, one per episode. Picks are greedy from the start of the record, at least
    ``spacing_k`` x the largest tau_int over the state variables apart. Returns
    ``(snapshot indices, spacing in snapshots, regimes file)``.
    """
    reg = load_regimes(n_years, directory)
    lab = regime_labels(reg["p_active"], float(reg["cutoff"]))
    N = len(lab)
    span = int(np.ceil(window_steps / SNAPSHOT_EVERY))
    spacing = int(np.ceil(spacing_k * float(np.max(reg["tau_int"]))))
    min_len = int(round(min_episode_days / float(reg["snapshot_days"])))

    if mode == "both":
        cands, group = range(N - span - 1), None
    elif mode in ("passive", "active"):
        r = 1 if mode == "active" else -1
        # stays[i]: labels[i .. i + span] all equal r
        same = np.concatenate([[0], np.cumsum(lab == r)])
        stays = np.zeros(N, dtype=bool)
        stays[:N - span] = (same[span + 1:] - same[:N - span]) == span + 1
        if placement == "onset":
            cands, group = [i for i in onsets(lab, r, min_len) if stays[i]], None
        elif placement == "anywhere":
            group = np.full(N, -1)
            for e, (_, s, t) in enumerate(episodes(lab, min_len)):
                group[s:t] = e
            cands = np.flatnonzero(stays & (group >= 0))
        else:
            raise ValueError(f"placement must be 'onset' or 'anywhere', not {placement!r}")
    else:
        raise ValueError(f"mode must be 'passive', 'active' or 'both', not {mode!r}")
    return pick_spaced(cands, spacing, group), spacing, reg


def starting_states(mode, n=N_SAMPLES, spacing_k=SPACING_K, min_episode_days=MIN_EPISODE_DAYS,
                    placement="onset", window_steps=0, n_years=BASE_RUN_YEARS,
                    directory=RESULTS_DIR):
    """The first ``n`` of ``available_picks``, with their states and the recipe, as a dict.

    Raises ``ValueError`` naming how many were found if the record holds fewer than ``n``.
    """
    picks, spacing, reg = available_picks(mode, spacing_k, min_episode_days, placement,
                                          window_steps, n_years, directory)
    dd = float(reg["snapshot_days"])
    if len(picks) < n:
        raise ValueError(f"{mode}: {len(picks)} starting states in {n_years:,} yr at spacing "
                         f"{spacing * dd / DAYS_PER_YEAR:,.1f} yr; {n} needed. Extend the base "
                         f"run or lower spacing_k.")
    idx = np.asarray(picks[:n])
    return dict(states=load_snapshots(idx, directory=directory),
                sampling=mode, snapshot_index=idx, start_year=idx * dd / DAYS_PER_YEAR,
                n_samples=n, spacing_k=spacing_k, spacing_years=spacing * dd / DAYS_PER_YEAR,
                tau_max_days=float(np.max(reg["tau_int"])) * dd,
                min_episode_days=min_episode_days, placement=placement,
                regime_T_days=float(reg["t_days"][int(reg["chosen"])]),
                regime_cutoff=float(reg["cutoff"]), base_run_years=n_years)


def sample_record(picks):
    """``starting_states``' output minus the states and the mode (which a study saves itself
    as ``sampling``), keys prefixed ``sample_`` for its .npz. Empty for legacy sampling."""
    return {f"sample_{k}": v for k, v in picks.items() if k not in ("states", "sampling")}


def is_sampled(d):
    """True iff a study's loaded .npz came from the sampling standard rather than legacy."""
    return "sampling" in d.files and str(d["sampling"]) != "legacy"


def sampling_line(d):
    """The figure suptitle naming a sampled run's starting states; None for legacy.

    ``both`` takes every state whatever its label, so placement does not apply to it
    (``available_picks``) and is not named."""
    if not is_sampled(d):
        return None
    mode, n = str(d["sampling"]), int(d["sample_n_samples"])
    if mode == "both":
        return f"both regimes: {n} starting states"
    where = {"onset": "at episode onsets",
             "anywhere": "anywhere in an episode"}[str(d["sample_placement"])]
    return f"{mode}: {n} starting states {where}"


def whiten_note(d):
    """Which std a run's residual was whitened by: the high-frequency climatological std on
    sampled runs, the std over the window being fit on legacy ones (``channel_weights``)."""
    if not ("whiten" in d.files and bool(d["whiten"])):
        return "unwhitened residual"
    std = "high-frequency climatological" if is_sampled(d) else "within-window"
    return f"whitened residual (1/{std} std per channel)"
