"""Experiment 5: Soundness against dishonest provers."""

import time

from numpy.random import default_rng

from experiments.harness.phi import make_single_parity
from experiments.harness.results import ExperimentResult
from experiments.harness.worker import TrialSpec, run_trials_parallel


def run_soundness_experiment(
    n_range: range = range(4, 7),
    num_trials: int = 50,
    epsilon: float = 0.3,
    classical_samples_verifier: int = 3000,
    base_seed: int = 42,
    max_workers: int = 1,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> ExperimentResult:
    r"""Empirical soundness against dishonest provers.

    For each :math:`n` in *n_range*, tests four adversarial prover
    strategies against the verifier's information-theoretic soundness
    guarantee (Definition 7):

    ``"random_list"``
        Prover sends 5 uniformly random frequency indices.  Expected
        acceptance rate :math:`\approx 5/2^n` (the probability of
        accidentally including :math:`s^*`).

    ``"wrong_parity"``
        Prover sends a single incorrect index :math:`s \neq s^*`.
        The verifier's independent estimate will be
        :math:`\hat\xi(s) \approx 0`, causing rejection.

    ``"partial_list"``
        Prover sends an empty list.  Accumulated weight is 0, well
        below the threshold :math:`1 - \varepsilon^2/8`.

    ``"inflated_list"``
        Prover sends 10 wrong indices with fabricated estimates.
        The verifier's independent estimates expose all of them as
        having negligible true Fourier weight.

    The empirical rejection rate should be
    :math:`\geq 1 - \delta` for all strategies (excluding the
    combinatorial chance of ``"random_list"`` hitting :math:`s^*`).

    Parameters
    ----------
    n_range : range
        Range of :math:`n` values to sweep.
    num_trials : int
        Number of trials per :math:`(n, \text{strategy})` cell.
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
    strategies = ["random_list", "wrong_parity", "partial_list", "inflated_list"]

    print(
        f"=== Soundness Experiment: n in {list(n_range)}, "
        f"{num_trials} trials/strategy, {max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    specs: list[TrialSpec] = []
    for n in n_range:
        target_s = 1
        phi = make_single_parity(n, target_s)

        for strategy in strategies:
            for _ in range(num_trials):
                seed = int(rng.integers(0, 2**31))
                specs.append(
                    TrialSpec(
                        n=n,
                        phi=phi,
                        noise_rate=0.0,
                        target_s=target_s,
                        epsilon=epsilon,
                        delta=0.1,
                        theta=epsilon,
                        a_sq=1.0,
                        b_sq=1.0,
                        qfs_shots=0,
                        classical_samples_prover=0,
                        classical_samples_verifier=classical_samples_verifier,
                        seed=seed,
                        phi_description=f"soundness_{strategy}",
                        dishonest_strategy=strategy,
                    )
                )

    t0 = time.time()
    trials = run_trials_parallel(
        specs, max_workers=max_workers, label="sound",
        shard_index=shard_index, num_shards=num_shards,
    )
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="soundness",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n_range": list(n_range),
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
        print(f"  {strategy:20s}: rejected {rej}/{len(st)} ({rej / len(st):.0%})")

    print(f"  Wall-clock time: {wall:.1f}s")
    return result
