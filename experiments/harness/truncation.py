"""Experiment 3: Verifier truncation tradeoffs."""

import time
from typing import Optional

import numpy as np
from numpy.random import default_rng

from experiments.harness.phi import make_single_parity
from experiments.harness.results import ExperimentResult
from experiments.harness.worker import TrialSpec, run_trials_parallel


def run_truncation_experiment(
    n: int = 6,
    epsilon_range: Optional[list[float]] = None,
    verifier_sample_range: Optional[list[int]] = None,
    num_trials: int = 20,
    qfs_shots: int = 2000,
    classical_samples_prover: int = 1000,
    base_seed: int = 42,
    max_workers: int = 1,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> ExperimentResult:
    r"""Verifier truncation tradeoff experiment.

    Fixes :math:`n` and the prover's resources, then sweeps the
    verifier's classical sample budget :math:`m_V` and accuracy
    parameter :math:`\varepsilon` to map the completeness frontier.

    Uses noisy parity with :math:`\eta = 0.15` (effective coefficient
    :math:`(1 - 2\eta) = 0.7`, so :math:`a^2 = b^2 = 0.49`), which
    gives a tight Fourier weight margin:

    .. math::

        \text{weight} \approx 0.49, \quad
        \tau = a^2 - \varepsilon^2 / 8

    At small :math:`m_V`, the Hoeffding estimation noise in the
    verifier's coefficient estimates pushes the accumulated weight
    below :math:`\tau`, causing rejection even for honest provers.
    At large :math:`\varepsilon`, :math:`\tau` drops and the check
    becomes lenient.

    .. warning::

       **Sub-Hoeffding regime — figures are NOT Theorem 12 boundary
       measurements.** The default grid
       :math:`m_V \in \{50, \dots, 3000\}` lies 3-4 orders of magnitude
       *below* the Theorem 12 verifier sample-complexity prescription
       :math:`m \ge (2/\mathrm{tol}^2)\,\log(4|L|/\delta)` with
       :math:`\mathrm{tol} = \varepsilon^2/(16|L|)` (see
       ``ql/verifier.py:493``).  For :math:`|L|=1` and :math:`\delta=0.1`
       the analytic minimum is roughly :math:`1.9 \times 10^7` at
       :math:`\varepsilon=0.1` and :math:`3.0 \times 10^4` at
       :math:`\varepsilon=0.5`, so the grid never reaches the asymptotic
       regime that the threshold :math:`\tau = a^2 - \varepsilon^2/8`
       was designed for.

       Consequently the visible "knees" for :math:`\varepsilon \le 0.3`
       are **squaring-bias artefacts** of the unbiased estimator
       :math:`\widehat\xi(s)^2 = \widehat{\varphi}(s)^2 +
       \mathrm{Var}(\widehat\xi)`, whose bias :math:`\approx 0.51/m_V`
       is comparable to or exceeds the acceptance margin
       :math:`\varepsilon^2/8`.  Some rows in
       ``truncation_summary.csv`` therefore show
       ``accept@50 > accept@3000`` (acceptance *decreasing* with
       budget), which would be impossible in the true Hoeffding regime.

       Additionally, the prover budgets ``qfs_shots=2000`` and
       ``classical_samples_prover=1000`` sit far below the
       Corollary 5 prescription, so the figures partly reflect
       prover-side QFS failure modes (the prover may fail to place
       the target string into :math:`L` at small :math:`\varepsilon`
       and large :math:`n`) rather than pure verifier-side truncation.

       The 2D grid :math:`(\varepsilon \times m_V)` is therefore best
       interpreted as a **non-asymptotic / sub-Hoeffding feasibility
       sweep** at fixed prover and verifier budgets, *not* as a
       measurement of the tradeoff surface predicted by Theorem 12.
       See ``audit/truncation.md`` (M1, M2) and ``audit/FOLLOW_UPS.md``
       for the rerun specifications that would actually cross the
       Theorem 12 boundary.

    Parameters
    ----------
    n : int
        Number of input bits (fixed across the sweep).
    epsilon_range : list[float] or None
        Values of :math:`\varepsilon` to test.
        Default: ``[0.1, 0.2, 0.3, 0.4, 0.5]``.
    verifier_sample_range : list[int] or None
        Verifier classical sample budgets :math:`m_V` to test.
        Default: ``[50, 100, 200, 500, 1000, 3000]``.
    num_trials : int
        Number of independent trials per :math:`(\varepsilon, m_V)` cell.
    qfs_shots : int
        QFS copy budget per trial (prover).
    classical_samples_prover : int
        Classical samples for the prover.
    base_seed : int
        Base random seed.
    max_workers : int
        Number of parallel worker processes.

    Returns
    -------
    ExperimentResult
    """
    if epsilon_range is None:
        epsilon_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    if verifier_sample_range is None:
        verifier_sample_range = [50, 100, 200, 500, 1000, 3000]

    noise_rate = 0.15
    a_sq = (1.0 - 2.0 * noise_rate) ** 2

    print(
        f"=== Truncation Experiment: n={n}, eta={noise_rate}, {max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    target_s = 1
    phi = make_single_parity(n, target_s)

    specs: list[TrialSpec] = []
    for eps in epsilon_range:
        for v_samples in verifier_sample_range:
            for _ in range(num_trials):
                seed = int(rng.integers(0, 2**31))
                specs.append(
                    TrialSpec(
                        n=n,
                        phi=phi,
                        noise_rate=noise_rate,
                        target_s=target_s,
                        epsilon=eps,
                        delta=0.1,
                        theta=min(eps, 0.5),
                        a_sq=a_sq,
                        b_sq=a_sq,
                        qfs_shots=qfs_shots,
                        classical_samples_prover=classical_samples_prover,
                        classical_samples_verifier=v_samples,
                        seed=seed,
                        phi_description=f"trunc_eps={eps}_vsamp={v_samples}",
                    )
                )

    t0 = time.time()
    trials = run_trials_parallel(
        specs, max_workers=max_workers, label="trunc",
        shard_index=shard_index, num_shards=num_shards,
    )
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="verifier_truncation",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n": n,
            "noise_rate": noise_rate,
            "a_sq": a_sq,
            "epsilon_range": epsilon_range,
            "verifier_sample_range": verifier_sample_range,
            "num_trials": num_trials,
            "qfs_shots": qfs_shots,
        },
    )

    # Print summary grid
    print("\n  Accept rates (eps x verifier_samples):")
    header = f"  {'eps':>5s}" + "".join(f"  {v:>5d}" for v in verifier_sample_range)
    print(header)
    for eps in epsilon_range:
        row_trials = [t for t in trials if abs(t.epsilon - eps) < 1e-9]
        cells = []
        for v in verifier_sample_range:
            vt = [t for t in row_trials if t.verifier_samples == v]
            rate = np.mean([t.accepted for t in vt]) if vt else 0.0
            cells.append(f"  {rate:5.0%}")
        print(f"  {eps:5.2f}" + "".join(cells))

    print(f"  Wall-clock time: {wall:.1f}s")
    return result
