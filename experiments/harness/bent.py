"""Experiment 2: Bent function worst-case."""

import time

from numpy.random import default_rng

from experiments.harness.phi import make_bent_function
from experiments.harness.results import ExperimentResult
from experiments.harness.worker import TrialSpec, run_trials_parallel


def run_bent_experiment(
    n_range: range = range(4, 13, 2),
    num_trials: int = 10,
    epsilon: float = 0.3,
    theta: float = 0.3,
    qfs_shots: int = 3000,
    classical_samples_prover: int = 2000,
    classical_samples_verifier: int = 5000,
    base_seed: int = 42,
    max_workers: int = 1,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> ExperimentResult:
    r"""Bent function worst-case experiment.

    Bent functions (even :math:`n` only) have all Fourier coefficients
    at :math:`|\hat{g}(s)| = 2^{-n/2}`.  As :math:`n` grows, this
    magnitude shrinks below the resolution threshold :math:`\vartheta`,
    and the prover's heavy coefficient list collapses.

    The **crossover** occurs at :math:`2^{-n/2} \approx \vartheta`:

    - Below the crossover (:math:`n` small), all :math:`2^n` entries
      appear in :math:`L` and the verifier accepts.
    - Above the crossover (:math:`n` large), the prover finds few or
      no entries above threshold, and the verifier rejects.

    This is the theoretically predicted worst case for the QFS-based
    spectrum approximation of Corollary 5: the DKW-based extraction
    cannot distinguish signal from the uniform floor
    :math:`(1 - \mathbb{E}[\tilde\phi^2]) / 2^n` when all coefficients
    are equally small.

    Parameters
    ----------
    n_range : range
        Range of even :math:`n` values to sweep.
    num_trials : int
        Number of independent trials per :math:`n`.
    epsilon : float
        Accuracy parameter :math:`\varepsilon`.
    theta : float
        Fourier resolution threshold :math:`\vartheta`.
    qfs_shots : int
        QFS copy budget per trial.
    classical_samples_prover : int
        Classical samples for the prover.
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
    print(
        f"=== Bent Function Experiment: n in {list(n_range)}, {max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    specs: list[TrialSpec] = []
    for n in n_range:
        phi_bent = make_bent_function(n)
        for _ in range(num_trials):
            seed = int(rng.integers(0, 2**31))
            specs.append(
                TrialSpec(
                    n=n,
                    phi=phi_bent,
                    noise_rate=0.0,
                    target_s=0,  # placeholder — all equally heavy
                    epsilon=epsilon,
                    delta=0.1,
                    theta=theta,
                    a_sq=1.0,
                    b_sq=1.0,
                    qfs_shots=qfs_shots,
                    classical_samples_prover=classical_samples_prover,
                    classical_samples_verifier=classical_samples_verifier,
                    seed=seed,
                    phi_description=f"bent_n={n}",
                )
            )

    t0 = time.time()
    trials = run_trials_parallel(
        specs, max_workers=max_workers, label="bent",
        shard_index=shard_index, num_shards=num_shards,
    )
    wall = time.time() - t0

    # For bent functions, "correct" means "accepted" (all parities equally heavy)
    for t in trials:
        t.hypothesis_correct = t.accepted

    result = ExperimentResult(
        experiment_name="bent_function",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n_range": list(n_range),
            "num_trials": num_trials,
            "epsilon": epsilon,
            "theta": theta,
            "qfs_shots": qfs_shots,
            "note": "all |hat(s)| = 2^{-n/2}; threshold crossover is key",
        },
    )
    print(f"\n{result.summary_table()}")
    print(f"  Wall-clock time: {wall:.1f}s")
    return result
