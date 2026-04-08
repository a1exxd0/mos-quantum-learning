"""Experiment 1: Scaling with n."""

import time

from numpy.random import default_rng

from experiments.harness.phi import make_random_parity
from experiments.harness.results import ExperimentResult
from experiments.harness.worker import TrialSpec, run_trials_parallel


def run_scaling_experiment(
    n_range: range = range(4, 13),
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
    r"""Scaling experiment: completeness vs :math:`n` for random parities.

    For each :math:`n` in *n_range*, generates *num_trials* random
    nonzero parities :math:`\varphi(x) = s^* \cdot x`, runs the full
    honest prover :math:`\to` verifier protocol, and records whether
    the correct parity was recovered.

    This is the key experiment distinguishing the project from a
    reproduction study: at :math:`n = 4` it matches the paper's baseline,
    at :math:`n = 16{+}` it demonstrates GL extraction at scale.

    The prover uses :math:`O(\log(1/\delta) / \varepsilon^4)` QFS copies
    (Corollary 5) and the verifier uses
    :math:`O(b^4 \log(1/\delta\vartheta^2) / \varepsilon^4\vartheta^4)`
    classical examples (Theorem 12).  The distribution class promise is
    :math:`a^2 = b^2 = 1` (functional case, Theorem 8).

    .. note::

       **Hard-coded sample budgets bypass the analytic formulas.** The
       defaults ``qfs_shots=2000``, ``classical_samples_prover=1000`` and
       ``classical_samples_verifier=3000`` are *constants*, not values
       derived from the Hoeffding budgets in
       :func:`ql.verifier.MoSVerifier._verify_core` (lines 487-497) or
       the Corollary 5 formula in :meth:`ql.prover.MoSProver.run_protocol`
       (lines 316-321). Consequently ``total_copies = qfs_shots +
       classical_samples_prover + classical_samples_verifier = 6000`` is
       constant across every :math:`n` in the sweep by construction, and
       the experiment tests "this fixed 6000-copy budget suffices on
       single-parity targets up to :math:`n=16`" rather than the
       n-independence of Theorem 12's analytic verifier sample formula.

       To actually validate the n-independence claim of Theorem 12, the
       sweep would need to be re-run with ``qfs_shots=None``,
       ``classical_samples_prover=None`` and
       ``classical_samples_verifier=None`` so the worker falls through to
       the per-trial analytic budgets. See ``audit/scaling.md`` (M1, M2)
       and ``audit/FOLLOW_UPS.md``.

    Parameters
    ----------
    n_range : range
        Range of :math:`n` values to sweep.
    num_trials : int
        Number of independent trials per :math:`n`.
    epsilon : float
        Accuracy parameter :math:`\varepsilon`.
    delta : float
        Confidence parameter :math:`\delta`.
    qfs_shots : int
        QFS copy budget per trial.
    classical_samples_prover : int
        Classical samples for the prover's coefficient estimation.
    classical_samples_verifier : int
        Classical samples for the verifier's independent estimation.
    base_seed : int
        Base random seed for reproducibility.
    max_workers : int
        Number of parallel worker processes.

    Returns
    -------
    ExperimentResult
    """
    print(
        f"=== Scaling Experiment: n in {list(n_range)}, {num_trials} trials each, "
        f"{max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    specs: list[TrialSpec] = []
    for n in n_range:
        for _ in range(num_trials):
            seed = int(rng.integers(0, 2**31))
            trial_rng = default_rng(seed)
            phi, target_s = make_random_parity(n, trial_rng)
            specs.append(
                TrialSpec(
                    n=n,
                    phi=phi,
                    noise_rate=0.0,
                    target_s=target_s,
                    epsilon=epsilon,
                    delta=delta,
                    theta=epsilon,
                    a_sq=1.0,
                    b_sq=1.0,
                    qfs_shots=qfs_shots,
                    classical_samples_prover=classical_samples_prover,
                    classical_samples_verifier=classical_samples_verifier,
                    seed=seed,
                    phi_description=f"random_parity_s={target_s}",
                )
            )

    t0 = time.time()
    trials = run_trials_parallel(
        specs, max_workers=max_workers, label="scaling",
        shard_index=shard_index, num_shards=num_shards,
    )
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="scaling",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n_range": list(n_range),
            "num_trials": num_trials,
            "epsilon": epsilon,
            "delta": delta,
            "qfs_shots": qfs_shots,
            "classical_samples_prover": classical_samples_prover,
            "classical_samples_verifier": classical_samples_verifier,
        },
    )
    print(f"\n{result.summary_table()}")
    print(f"  Wall-clock time: {wall:.1f}s")
    return result
