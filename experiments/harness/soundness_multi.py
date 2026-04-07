"""Experiment C: Soundness against dishonest provers with multi-element targets."""

import time

from numpy.random import default_rng

from experiments.harness.phi import make_k_sparse
from experiments.harness.results import ExperimentResult
from experiments.harness.worker import TrialSpec, run_trials_parallel


def run_soundness_multi_experiment(
    n_range: range = range(4, 7),
    k_range: list[int] | None = None,
    num_trials: int = 50,
    epsilon: float = 0.3,
    classical_samples_verifier: int = 3000,
    base_seed: int = 42,
    max_workers: int = 1,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> ExperimentResult:
    r"""Empirical soundness against dishonest provers with k-sparse targets.

    Extends the single-parity soundness experiment by testing four
    adversarial strategies against target functions with **multiple**
    nonzero Fourier coefficients (k-sparse, Dirichlet-drawn).

    This closes Gap 3 in the experiment gap analysis: the accumulated
    weight check is validated against partially-correct adversarial
    lists where the adversary may include some genuine heavy
    coefficients alongside fabricated ones.

    Strategies
    ----------
    ``"partial_real"``
        Prover includes the weaker half of real heavy coefficients
        plus fake indices.  Tests whether partial real weight passes
        the threshold.

    ``"diluted_list"``
        Prover includes all real heavy coefficients but pads with
        20 random indices.  Tests list-size rejection.

    ``"shifted_coefficients"``
        Prover sends entirely wrong indices with fabricated large
        coefficient claims.  Tests that independent estimation
        exposes zero true weight.

    ``"subset_plus_noise"``
        Prover includes only the single heaviest real coefficient
        plus several near-threshold fake indices.  Tests the
        marginal case where one real coefficient's weight alone
        is insufficient.

    Parameters
    ----------
    n_range : range
        Range of :math:`n` values to sweep.
    k_range : list[int] or None
        Fourier sparsity values to test.  Default: ``[2, 4]``.
    num_trials : int
        Number of trials per :math:`(n, k, \text{strategy})` cell.
    epsilon : float
        Accuracy parameter :math:`\varepsilon`.
    classical_samples_verifier : int
        Classical samples for the verifier.
    base_seed : int
        Base random seed.
    max_workers : int
        Number of parallel worker processes.

    Returns
    -------
    ExperimentResult
    """
    if k_range is None:
        k_range = [2, 4]

    strategies = ["partial_real", "diluted_list", "shifted_coefficients", "subset_plus_noise"]

    print(
        f"=== Soundness Multi-Element Experiment: n in {list(n_range)}, "
        f"k in {k_range}, {num_trials} trials/strategy, "
        f"{max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    specs: list[TrialSpec] = []
    for n in n_range:
        for k in k_range:
            theta = min(epsilon, max(0.01, (1.0 / k) * 0.9))

            for strategy in strategies:
                for _ in range(num_trials):
                    seed = int(rng.integers(0, 2**31))
                    trial_rng = default_rng(seed)
                    phi, target_s, pw = make_k_sparse(n, k, trial_rng)
                    specs.append(
                        TrialSpec(
                            n=n,
                            phi=phi,
                            noise_rate=0.0,
                            target_s=target_s,
                            epsilon=epsilon,
                            delta=0.1,
                            theta=theta,
                            a_sq=pw,
                            b_sq=pw,
                            qfs_shots=0,
                            classical_samples_prover=0,
                            classical_samples_verifier=classical_samples_verifier,
                            seed=seed,
                            phi_description=f"soundness_multi_{strategy}_k={k}",
                            dishonest_strategy=strategy,
                            k=k,
                        )
                    )

    t0 = time.time()
    trials = run_trials_parallel(
        specs, max_workers=max_workers, label="sound_multi",
        shard_index=shard_index, num_shards=num_shards,
    )
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="soundness_multi",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n_range": list(n_range),
            "k_range": k_range,
            "num_trials": num_trials,
            "epsilon": epsilon,
            "strategies": strategies,
        },
    )

    # Print per-strategy rejection rates
    print()
    for strategy in strategies:
        st = [t for t in trials if strategy in t.phi_description]
        rej = sum(1 for t in st if not t.accepted)
        total = len(st)
        rate = rej / total if total > 0 else 0.0
        print(f"  {strategy:25s}: rejected {rej}/{total} ({rate:.0%})")

    print(f"  Wall-clock time: {wall:.1f}s")
    return result
