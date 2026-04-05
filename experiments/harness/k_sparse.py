"""Experiment A: k-Fourier-sparse verification path (Theorems 9/10/14/15)."""

import time
from typing import Optional

import numpy as np
from numpy.random import default_rng

from experiments.harness.phi import make_k_sparse
from experiments.harness.results import ExperimentResult
from experiments.harness.worker import TrialSpec, run_trials_parallel

_DEFAULT_K_VALUES = [1, 2, 4, 8]


def run_k_sparse_experiment(
    n_range: range = range(4, 11, 2),
    k_values: Optional[list[int]] = None,
    num_trials: int = 20,
    epsilon: float = 0.3,
    delta: float = 0.1,
    qfs_shots: int = 2000,
    classical_samples_prover: int = 1000,
    classical_samples_verifier: int = 3000,
    misclassification_samples: int = 1000,
    base_seed: int = 42,
    max_workers: int = 1,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> ExperimentResult:
    r"""k-Sparse verification experiment.

    Exercises :meth:`~ql.verifier.MoSVerifier.verify_fourier_sparse`
    (Theorems 9/10/14/15) by sweeping Fourier sparsity *k* and
    dimension *n*.  For ``k = 1``, uses ``verify_parity`` as a
    comparison baseline.

    Measures both index-correctness (heaviest coefficient match) and
    empirical misclassification rate for k-sparse hypotheses.
    """
    if k_values is None:
        k_values = list(_DEFAULT_K_VALUES)

    print(
        f"=== k-Sparse Experiment: n in {list(n_range)}, "
        f"k in {k_values}, {num_trials} trials each, "
        f"{max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    specs: list[TrialSpec] = []
    for n in n_range:
        for k in k_values:
            for _ in range(num_trials):
                seed = int(rng.integers(0, 2**31))
                trial_rng = default_rng(seed)
                phi, target_s, pw = make_k_sparse(n, k, trial_rng)
                max_coeff = 1.0 / k
                theta = min(epsilon, max(0.01, max_coeff * 0.9))
                a_sq = b_sq = pw

                specs.append(
                    TrialSpec(
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
                        phi_description=f"k_sparse_k={k}_n={n}",
                        k=k if k > 1 else None,
                        misclassification_samples=misclassification_samples,
                    )
                )

    t0 = time.time()
    trials = run_trials_parallel(
        specs, max_workers=max_workers, label="k_sparse",
        shard_index=shard_index, num_shards=num_shards,
    )
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="k_sparse",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n_range": list(n_range),
            "k_values": k_values,
            "num_trials": num_trials,
            "epsilon": epsilon,
            "delta": delta,
            "qfs_shots": qfs_shots,
            "classical_samples_prover": classical_samples_prover,
            "classical_samples_verifier": classical_samples_verifier,
            "misclassification_samples": misclassification_samples,
        },
    )

    # Per-k summary
    print(f"\n  {'k':>3s} {'accept%':>8s} {'correct%':>9s} {'misclass':>9s}")
    for k in k_values:
        kt = [t for t in trials if f"k={k}_" in t.phi_description]
        ar = np.mean([t.accepted for t in kt]) if kt else 0.0
        cr = np.mean([t.hypothesis_correct for t in kt]) if kt else 0.0
        mr_vals = [
            t.misclassification_rate
            for t in kt
            if t.misclassification_rate is not None
        ]
        mr = np.mean(mr_vals) if mr_vals else float("nan")
        print(f"  {k:3d} {ar:8.0%} {cr:9.0%} {mr:9.3f}")

    print(f"\n{result.summary_table()}")
    print(f"  Wall-clock time: {wall:.1f}s")
    return result
