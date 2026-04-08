"""Experiment D: theta sensitivity — extraction boundary mapping (Gap 5)."""

import time

import numpy as np
from numpy.random import default_rng

from experiments.harness.phi import make_sparse_plus_noise
from experiments.harness.results import ExperimentResult
from experiments.harness.worker import TrialSpec, run_trials_parallel

_DEFAULT_THETA_VALUES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]


def run_theta_sensitivity_experiment(
    n_range: range = range(4, 9, 2),
    theta_values: list[float] | None = None,
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
    r"""Theta sensitivity experiment.

    Sweeps the Fourier resolution threshold :math:`\vartheta` against
    :func:`~experiments.harness.phi.make_sparse_plus_noise` functions
    (dominant coefficient 0.7, three secondary coefficients 0.1 each).

    At :math:`\vartheta \approx 0.20`, the secondary coefficients sit
    at the extraction boundary :math:`|\hat{\tilde\phi}(s)| = 0.1`
    vs threshold :math:`\vartheta/2 = 0.10`, probing where finite-sample
    noise determines inclusion in :math:`L`.

    Records :math:`|L|`, acceptance outcome, and accumulated weight
    per trial to map the extraction frontier (Corollary 5).

    .. warning::

       **Maps the acceptance boundary; does NOT validate the
       :math:`1/\vartheta^4` sample-complexity scaling.** Two MAJOR
       caveats from ``audit/theta_sensitivity.md``:

       - **M1.** The hard-coded ``qfs_shots=2000``,
         ``classical_samples_prover=1000`` and
         ``classical_samples_verifier=3000`` override the
         :math:`\vartheta`-dependent formulas in
         :meth:`ql.prover.MoSProver.run_protocol` (lines 316-321) and
         :func:`ql.verifier.MoSVerifier._verify_core` (lines 487-497).
         For :math:`\vartheta = 0.05` the analytic prover budget is
         :math:`\sim 1.9 \times 10^8` shots vs 2000 used (about 5 orders
         short).  The analytic verifier budget at the same
         :math:`\vartheta` ranges from :math:`\sim 2.3 \times 10^5`
         (:math:`|L|=1`) to :math:`\sim 1 \times 10^8` (the empirical
         median :math:`|L| \approx 484` at :math:`n=16`) to
         :math:`\sim 1.5 \times 10^{14}` (the worst-case enforced
         Theorem 12 list bound :math:`64 b^2/\vartheta^2 = 13312`),
         versus 3000 used --- between 2 and 11 orders of magnitude
         short depending on which :math:`|L|` you plug in.  The
         experiment maps where the verifier accepts/rejects; it does
         NOT empirically test the theoretical :math:`1/\vartheta^4`
         scaling.

       - **M2.** :func:`make_sparse_plus_noise` has nonzero Fourier
         coefficients of magnitude exactly 0.1, so for any
         :math:`\vartheta > 0.1` the function lies *outside*
         :math:`\mathfrak{D}^{\mathrm{func}}_{U_n;\ge\vartheta}`
         (Definition 11).  This means Theorems 8/12 do not formally
         apply at :math:`\vartheta \in \{0.12, 0.15, 0.20, 0.30,
         0.50\}` --- the experiment is intentionally probing the
         out-of-promise regime.

       To actually validate the :math:`1/\vartheta^4` scaling, the
       sweep would need to be re-run with ``qfs_shots=None``,
       ``classical_samples_prover=None`` and
       ``classical_samples_verifier=None`` so the analytic formulas
       drive the per-trial budgets.  See ``audit/FOLLOW_UPS.md``.
    """
    if theta_values is None:
        theta_values = list(_DEFAULT_THETA_VALUES)

    print(
        f"=== Theta Sensitivity Experiment: n in {list(n_range)}, "
        f"theta in {theta_values}, {num_trials} trials each, "
        f"{max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    specs: list[TrialSpec] = []
    for n in n_range:
        for theta in theta_values:
            for _ in range(num_trials):
                seed = int(rng.integers(0, 2**31))
                trial_rng = default_rng(seed)
                phi, target_s, pw = make_sparse_plus_noise(n, trial_rng)
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
                        phi_description=f"sparse_plus_noise_theta={theta}_n={n}",
                    )
                )

    t0 = time.time()
    trials = run_trials_parallel(
        specs, max_workers=max_workers, label="theta_sensitivity",
        shard_index=shard_index, num_shards=num_shards,
    )
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="theta_sensitivity",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n_range": list(n_range),
            "theta_values": theta_values,
            "num_trials": num_trials,
            "epsilon": epsilon,
            "delta": delta,
            "qfs_shots": qfs_shots,
            "classical_samples_prover": classical_samples_prover,
            "classical_samples_verifier": classical_samples_verifier,
        },
    )

    # Per-theta summary
    print(f"\n  {'theta':>6s} {'accept%':>8s} {'correct%':>9s} {'|L| med':>7s}")
    for theta in theta_values:
        tt = [t for t in trials if f"theta={theta}_" in t.phi_description]
        ar = np.mean([t.accepted for t in tt]) if tt else 0.0
        cr = np.mean([t.hypothesis_correct for t in tt]) if tt else 0.0
        med_L = np.median([t.list_size for t in tt]) if tt else 0
        print(f"  {theta:6.2f} {ar:8.0%} {cr:9.0%} {med_L:7.0f}")

    print(f"\n{result.summary_table()}")
    print(f"  Wall-clock time: {wall:.1f}s")
    return result
