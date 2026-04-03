"""Experiment 7: Average-case performance across function families."""

import time
from typing import Optional

import numpy as np
from numpy.random import default_rng

from experiments.harness.phi import (
    make_k_sparse,
    make_random_boolean,
    make_sparse_plus_noise,
)
from experiments.harness.results import ExperimentResult
from experiments.harness.worker import TrialSpec, run_trials_parallel

_DEFAULT_FAMILIES = ["k_sparse_2", "k_sparse_4", "random_boolean", "sparse_plus_noise"]


def _generate_trial(
    n: int,
    family: str,
    epsilon: float,
    delta: float,
    qfs_shots: int,
    classical_samples_prover: int,
    classical_samples_verifier: int,
    rng: np.random.Generator,
) -> TrialSpec:
    r"""Build a :class:`TrialSpec` for one (n, family) trial.

    Dispatches to the appropriate phi generator and sets
    protocol parameters (:math:`\vartheta`, :math:`a^2`, :math:`b^2`)
    to match the Fourier structure of the chosen function family.
    """
    seed = int(rng.integers(0, 2**31))
    trial_rng = default_rng(seed)

    if family.startswith("k_sparse_"):
        k = int(family.split("_")[-1])
        phi, target_s, pw = make_k_sparse(n, k, trial_rng)
        max_coeff = 1.0 / k  # expected order; actual varies per draw
        # Use the Dirichlet draw's actual max via pw is imprecise;
        # the safe bound is that the heaviest coeff >= 1/k in
        # expectation. Adapt theta to stay below the expected
        # heaviest coefficient so GL can detect it.
        theta = min(epsilon, max(0.01, max_coeff * 0.9))
        a_sq = b_sq = pw
    elif family == "random_boolean":
        phi, target_s = make_random_boolean(n, trial_rng)
        theta = epsilon
        a_sq = b_sq = 1.0
        pw = 1.0
    elif family == "sparse_plus_noise":
        phi, target_s, pw = make_sparse_plus_noise(n, trial_rng)
        theta = epsilon  # dominant coeff 0.7 >> epsilon
        a_sq = b_sq = pw
    else:
        raise ValueError(f"Unknown function family: {family}")

    return TrialSpec(
        n=n,
        phi=phi,
        noise_rate=0.0,
        target_s=target_s,
        epsilon=epsilon,
        delta=delta,
        theta=theta,
        a_sq=a_sq,
        b_sq=b_sq,
        qfs_shots=qfs_shots,
        classical_samples_prover=classical_samples_prover,
        classical_samples_verifier=classical_samples_verifier,
        seed=seed,
        phi_description=f"{family}_n={n}",
    )


def run_average_case_experiment(
    n_range: range = range(4, 11),
    families: Optional[list[str]] = None,
    num_trials: int = 20,
    epsilon: float = 0.3,
    delta: float = 0.1,
    qfs_shots: int = 2000,
    classical_samples_prover: int = 1000,
    classical_samples_verifier: int = 3000,
    base_seed: int = 42,
    max_workers: int = 1,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> ExperimentResult:
    r"""Average-case experiment: protocol performance on diverse function families.

    For each :math:`n` in *n_range* and each function family, samples
    *num_trials* random functions from the family, runs the full honest
    prover :math:`\to` verifier protocol, and records acceptance rate
    and hypothesis quality.

    The experiment goes beyond the single-parity regime of Exp 1 by
    testing three additional function families:

    ``k_sparse_2``, ``k_sparse_4``
        :math:`k`-Fourier-sparse functions with random Dirichlet
        coefficients on the :math:`k`-simplex (Corollary 7).
        :math:`\vartheta` is adapted per *k* so the Goldreich--Levin
        threshold remains below the expected heaviest coefficient.

    ``random_boolean``
        Uniform random truth tables -- maximally Fourier-dense, the
        hardest case.  :math:`a^2 = b^2 = 1` (functional).

    ``sparse_plus_noise``
        One dominant parity (:math:`c_{\mathrm{dom}} = 0.7`) plus
        three secondary coefficients (:math:`c_{\mathrm{sec}} = 0.1`
        each).  Tests whether the protocol correctly identifies the
        dominant coefficient in the presence of structured Fourier noise.

    Parameters
    ----------
    n_range : range
        Range of :math:`n` values to sweep.
    families : list[str] or None
        Function family names.  Default:
        ``["k_sparse_2", "k_sparse_4", "random_boolean",
        "sparse_plus_noise"]``.
    num_trials : int
        Trials per :math:`(n, \text{family})` cell.
    epsilon : float
        Accuracy parameter :math:`\varepsilon`.
    delta : float
        Confidence parameter :math:`\delta`.
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
    if families is None:
        families = list(_DEFAULT_FAMILIES)

    print(
        f"=== Average-Case Experiment: n in {list(n_range)}, "
        f"families={families}, {num_trials} trials each, "
        f"{max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    specs: list[TrialSpec] = []
    for n in n_range:
        for family in families:
            for _ in range(num_trials):
                specs.append(
                    _generate_trial(
                        n, family, epsilon, delta,
                        qfs_shots, classical_samples_prover,
                        classical_samples_verifier, rng,
                    )
                )

    t0 = time.time()
    trials = run_trials_parallel(
        specs, max_workers=max_workers, label="avg_case",
        shard_index=shard_index, num_shards=num_shards,
    )
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="average_case",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n_range": list(n_range),
            "families": families,
            "num_trials": num_trials,
            "epsilon": epsilon,
            "delta": delta,
            "qfs_shots": qfs_shots,
            "classical_samples_prover": classical_samples_prover,
            "classical_samples_verifier": classical_samples_verifier,
        },
    )

    # Per-family summary
    print(f"\n  {'family':>20s} {'accept%':>8s} {'correct%':>9s}")
    for fam in families:
        ft = [t for t in trials if t.phi_description.startswith(fam)]
        ar = np.mean([t.accepted for t in ft]) if ft else 0.0
        cr = np.mean([t.hypothesis_correct for t in ft]) if ft else 0.0
        print(f"  {fam:>20s} {ar:8.0%} {cr:9.0%}")

    print(f"\n{result.summary_table()}")
    print(f"  Wall-clock time: {wall:.1f}s")
    return result
