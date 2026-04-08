"""Experiment: a^2 != b^2 distributional regime."""

import time
from typing import Optional

import numpy as np
from numpy.random import default_rng

from experiments.harness.phi import make_sparse_plus_noise
from experiments.harness.results import ExperimentResult
from experiments.harness.worker import TrialSpec, run_trials_parallel


def run_ab_regime_experiment(
    n_range: range = range(4, 7),
    gaps: Optional[list[float]] = None,
    num_trials: int = 20,
    epsilon: float = 0.3,
    qfs_shots: int = 2000,
    classical_samples_prover: int = 1000,
    classical_samples_verifier: int = 3000,
    base_seed: int = 42,
    max_workers: int = 1,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> ExperimentResult:
    r"""Distributional regime sweep: verification with :math:`a^2 \neq b^2`.

    For each :math:`n` in *n_range* and each *gap* value, the experiment
    constructs a sparse-plus-noise :math:`\varphi` whose true Parseval
    weight is :math:`pw \approx 0.52`, then sets

    .. math::

        a^2 = pw - \tfrac{\text{gap}}{2}, \qquad
        b^2 = pw + \tfrac{\text{gap}}{2}

    with :math:`a^2` clamped to a minimum of 0.01.  When gap = 0 this
    recovers the standard :math:`a = b` regime used by all other
    experiments; positive gaps exercise Definition 14 of [ITCS2024]_
    where the verifier is told that :math:`a < b`.

    .. note::

       **Theorem 12 completeness precondition is satisfied only at
       gap = 0.** Theorem 12 requires
       :math:`\varepsilon \ge 2\sqrt{b^2 - a^2}`; with
       :math:`\varepsilon = 0.3` this means
       :math:`\mathrm{gap} \le (\varepsilon/2)^2 = 0.0225`, i.e. only
       the ``gap = 0`` cell of the default sweep formally satisfies
       the precondition.  For ``gap`` :math:`> 0.0225` the experiment
       is intentionally running outside Theorem 12's completeness
       guarantee on a benign ``sparse_plus_noise`` :math:`\varphi`
       whose true Parseval mass :math:`\|\tilde\varphi\|_2^2 = 0.52`
       sits at the centre of :math:`[a^2, b^2]`, so honest
       interactions still produce
       :math:`\sum \widehat\xi^2 \ge a^2 - \varepsilon^2/8`.

       This is **not** evidence that Theorem 13 (a worst-case
       sample-complexity lower bound for distinguishing random
       parities from :math:`U_{n+1}`) is loose; Theorem 13 does not
       upper-bound the acceptance probability of any specific
       honest run on benign inputs.  The previous figure-script
       interpretation was corrected per
       ``audit/ab_regime.md`` (M1).

       The "ab regime" sweep is structurally a **1-D**
       :math:`a^2`-sweep: :math:`b^2 \le 0.72` and the list-size cap
       :math:`64 b^2/\vartheta^2 \le 512` never binds the maximum
       honest list of 4 (M2 in the audit).  To probe both
       completeness and soundness of Definition 14, the centre
       :math:`pw` would need to vary independently --- see
       ``audit/FOLLOW_UPS.md``.

    Parameters
    ----------
    n_range : range
        Range of :math:`n` values to sweep.
    gaps : list[float] or None
        Values of :math:`b^2 - a^2` to sweep.
        Default: ``[0.0, 0.05, 0.1, 0.2, 0.3, 0.4]``.
    num_trials : int
        Trials per :math:`(n, \text{gap})` cell.
    epsilon : float
        Accuracy parameter :math:`\varepsilon`.
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
    shard_index : int or None
        0-based shard index for distributed execution.
    num_shards : int or None
        Total number of shards for distributed execution.

    Returns
    -------
    ExperimentResult
    """
    if gaps is None:
        gaps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]

    print(
        f"=== Ab Regime: n in {list(n_range)}, gaps in {gaps}, "
        f"{max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    specs: list[TrialSpec] = []
    for n in n_range:
        for gap in gaps:
            for _ in range(num_trials):
                seed = int(rng.integers(0, 2**31))
                trial_rng = default_rng(seed)
                phi, target_s, parseval_weight = make_sparse_plus_noise(n, trial_rng)

                a_sq = max(parseval_weight - gap / 2.0, 0.01)
                b_sq = min(parseval_weight + gap / 2.0, 1.0)
                # In the verifier path theta only enters via the list
                # bound 64 b^2/theta^2 (which doesn't bind at |L| <= 4),
                # so any theta <= epsilon works here.  Audit fix m4
                # (audit/ab_regime.md): the previous comment about
                # "keep theta below the dominant coefficient" was a
                # prover-side heuristic, irrelevant in the verifier
                # path tested by this experiment.
                theta = min(epsilon, 0.6)

                specs.append(
                    TrialSpec(
                        n=n,
                        phi=phi,
                        noise_rate=0.0,
                        target_s=target_s,
                        epsilon=epsilon,
                        delta=0.1,
                        theta=theta,
                        a_sq=a_sq,
                        b_sq=b_sq,
                        qfs_shots=qfs_shots,
                        classical_samples_prover=classical_samples_prover,
                        classical_samples_verifier=classical_samples_verifier,
                        seed=seed,
                        phi_description=f"ab_regime_gap={gap}_n={n}",
                    )
                )

    t0 = time.time()
    trials = run_trials_parallel(
        specs, max_workers=max_workers, label="ab_regime",
        shard_index=shard_index, num_shards=num_shards,
    )
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="ab_regime",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n_range": list(n_range),
            "gaps": gaps,
            "num_trials": num_trials,
            "epsilon": epsilon,
        },
    )

    # Print per-gap summary
    print(f"\n  {'gap':>5s} {'a_sq':>6s} {'b_sq':>6s} {'accept%':>8s} {'correct%':>9s}")
    for gap in gaps:
        gt = [t for t in trials if f"gap={gap}" in t.phi_description]
        if not gt:
            continue
        ar = np.mean([t.accepted for t in gt])
        cr = np.mean([t.hypothesis_correct for t in gt])
        a_sq_avg = np.mean([t.a_sq for t in gt])
        b_sq_avg = np.mean([t.b_sq for t in gt])
        print(f"  {gap:5.2f} {a_sq_avg:6.3f} {b_sq_avg:6.3f} {ar:8.0%} {cr:9.0%}")

    print(f"  Wall-clock time: {wall:.1f}s")
    return result
