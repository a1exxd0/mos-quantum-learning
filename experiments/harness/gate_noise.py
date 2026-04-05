"""Experiment 3: Gate-level depolarising noise sweep."""

import time
from typing import Optional

import numpy as np
from numpy.random import default_rng

from experiments.harness.phi import make_random_parity
from experiments.harness.results import ExperimentResult
from experiments.harness.worker import TrialSpec, run_trials_parallel


def run_gate_noise_experiment(
    n_range: range = range(4, 7),
    gate_noise_rates: Optional[list[float]] = None,
    num_trials: int = 24,
    epsilon: float = 0.3,
    qfs_shots: int = 2000,
    classical_samples_prover: int = 1000,
    classical_samples_verifier: int = 3000,
    base_seed: int = 42,
    max_workers: int = 1,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> ExperimentResult:
    r"""Gate-level noise sweep: verification under depolarising gate errors.

    Unlike the label-flip noise experiment (Exp 2 / ``noise.py``), which
    applies noise at the distribution level via
    :math:`\varphi_{\mathrm{eff}}(x) = (1 - 2\eta)\,\varphi(x) + \eta`
    (Definition 5(iii)), this experiment applies depolarising noise
    channels directly to the quantum gates in the QFS circuit:

    - 1-qubit depolarising error (rate *p*) on H and X gates
    - 2-qubit depolarising error (rate *p*) on CX gates

    Multi-controlled X gates (used in the oracle :math:`U_f`) decompose
    into CX and single-qubit gates during transpilation, so gate noise
    propagates through the full circuit.

    This goes beyond the noise models analysed in Caro et al. (Lemmas
    4--6, §4.2) and produces inherently empirical results with no
    theoretical prediction to compare against.

    The experiment uses ``qfs_mode="circuit"`` (required for gate-level
    noise) and ``noise_rate=0`` (no label-flip noise), with
    :math:`a^2 = b^2 = 1` (noiseless distribution class promise, since
    the noise is purely at the gate level rather than the label level).

    Parameters
    ----------
    n_range : range
        Range of :math:`n` values to sweep.
    gate_noise_rates : list[float] or None
        Depolarising error rates *p* to sweep.
        Default: ``[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]``.
    num_trials : int
        Trials per :math:`(n, p)` cell.
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

    Returns
    -------
    ExperimentResult
    """
    if gate_noise_rates is None:
        gate_noise_rates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    print(
        f"=== Gate Noise: n in {list(n_range)}, p in {gate_noise_rates}, "
        f"{max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    specs: list[TrialSpec] = []
    for n in n_range:
        for p in gate_noise_rates:
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
                        delta=0.1,
                        theta=epsilon,
                        a_sq=1.0,
                        b_sq=1.0,
                        qfs_shots=qfs_shots,
                        classical_samples_prover=classical_samples_prover,
                        classical_samples_verifier=classical_samples_verifier,
                        seed=seed,
                        phi_description=f"gate_noise_p={p}_s={target_s}",
                        gate_noise_rate=p if p > 0 else None,
                        qfs_mode="circuit",
                    )
                )

    t0 = time.time()
    trials = run_trials_parallel(
        specs, max_workers=max_workers, label="gate_noise",
        shard_index=shard_index, num_shards=num_shards,
    )
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="gate_noise",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n_range": list(n_range),
            "gate_noise_rates": gate_noise_rates,
            "num_trials": num_trials,
            "epsilon": epsilon,
        },
    )

    # Print per-p summary
    print(f"\n  {'p':>7s} {'accept%':>8s} {'correct%':>9s}")
    for p in gate_noise_rates:
        et = [t for t in trials if f"p={p}" in t.phi_description]
        ar = np.mean([t.accepted for t in et]) if et else 0.0
        cr = np.mean([t.hypothesis_correct for t in et]) if et else 0.0
        print(f"  {p:7.4f} {ar:8.0%} {cr:9.0%}")

    print(f"  Wall-clock time: {wall:.1f}s")
    return result
