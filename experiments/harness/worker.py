"""Trial specification, worker functions, and parallel dispatch."""

import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import default_rng

from experiments.harness.results import TrialResult


@dataclass
class TrialSpec:
    r"""Fully serialisable specification for one trial.

    :class:`~concurrent.futures.ProcessPoolExecutor` requires picklable
    arguments.  Since :class:`~mos.MoSState` contains Qiskit objects that
    do not always pickle cleanly, we instead pass this plain dataclass
    to worker processes.  Each worker reconstructs the
    :class:`~mos.MoSState` from these fields, runs the protocol, and
    returns a :class:`TrialResult`.
    """

    #: Number of input bits.
    n: int
    #: Label-bias function :math:`\varphi(x) = \Pr[y{=}1 \mid x]`,
    #: stored as a plain list (not ``ndarray``) for pickle safety.
    #: Length :math:`2^n`.
    phi: list[float]
    #: Label-flip noise rate :math:`\eta \in [0, 0.5)`.
    noise_rate: float
    #: The ground-truth heavy parity index
    #: :math:`s^* \in \{0,\ldots,2^n{-}1\}`.
    target_s: int
    #: Accuracy parameter :math:`\varepsilon`.
    epsilon: float
    #: Confidence parameter :math:`\delta`.
    delta: float
    #: Fourier resolution threshold :math:`\vartheta`.
    theta: float
    #: Lower bound :math:`a^2` on
    #: :math:`\mathbb{E}[\tilde\phi(x)^2]` (Definition 14).
    a_sq: float
    #: Upper bound :math:`b^2` on
    #: :math:`\mathbb{E}[\tilde\phi(x)^2]` (Definition 14).
    b_sq: float
    #: Number of MoS copies for QFS (prover).
    qfs_shots: int
    #: Number of classical samples for the prover's coefficient
    #: estimation.
    classical_samples_prover: int
    #: Number of classical samples for the verifier's independent
    #: estimation.
    classical_samples_verifier: int
    #: Per-trial random seed.
    seed: int
    #: Human-readable label for the distribution.
    phi_description: str
    #: If not ``None``, the worker runs a dishonest prover with this
    #: strategy instead of the honest protocol.  One of
    #: ``"random_list"``, ``"wrong_parity"``, ``"partial_list"``, or
    #: ``"inflated_list"``.
    dishonest_strategy: Optional[str] = None
    #: Gate-level depolarising noise rate :math:`p`.  When set, the
    #: worker constructs a Qiskit ``NoiseModel`` with depolarising
    #: channels on H, X (1-qubit) and CX (2-qubit) gates.
    gate_noise_rate: Optional[float] = None
    #: QFS simulation mode passed to ``MoSProver.run_protocol``.
    qfs_mode: str = "statevector"
    #: Fourier sparsity parameter :math:`k`.  When set and ``> 1``,
    #: the worker calls :meth:`~ql.verifier.MoSVerifier.verify_fourier_sparse`
    #: instead of :meth:`~ql.verifier.MoSVerifier.verify_parity`.
    k: Optional[int] = None
    #: Number of fresh samples for misclassification rate estimation.
    #: When ``None``, defaults to 1000.
    misclassification_samples: Optional[int] = None


def _compute_misclassification_rate(state, hypothesis, seed, num_samples=1000):
    """Compute empirical P[h(x) != y] on fresh classical samples."""
    rng = default_rng(seed)
    xs, ys = state.sample_classical_batch(num_samples=num_samples, rng=rng)
    predictions = hypothesis.evaluate_batch(xs, rng=rng)
    return float(np.mean(predictions != ys))


def _run_trial_worker(spec: TrialSpec) -> TrialResult:
    r"""Execute a single trial in a worker process.

    Reconstructs a :class:`~mos.MoSState` from the fields in *spec*,
    runs the honest prover (:class:`~ql.prover.MoSProver`) followed by
    the classical verifier (:class:`~ql.verifier.MoSVerifier`), and
    returns a :class:`TrialResult` capturing every observable quantity.

    All imports are performed inside the function body so that each
    forked process obtains a clean module state with no shared mutable
    objects.

    If ``spec.dishonest_strategy`` is set, the worker delegates to
    :func:`_run_dishonest_trial` instead.

    Parameters
    ----------
    spec : TrialSpec
        Serialisable trial specification.

    Returns
    -------
    TrialResult
    """
    # Late imports — avoid contaminating the parent process and
    # ensure each worker has independent module state.
    from mos import MoSState
    from ql.prover import MoSProver
    from ql.verifier import MoSVerifier

    phi = np.array(spec.phi, dtype=np.float64)
    state = MoSState(n=spec.n, phi=phi, noise_rate=spec.noise_rate, seed=spec.seed)

    if spec.dishonest_strategy is not None:
        return _run_dishonest_trial(spec, state)

    # --- Build gate-level noise model (if requested) ---
    noise_model = None
    if spec.gate_noise_rate is not None and spec.gate_noise_rate > 0:
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(spec.gate_noise_rate, 1), ["h", "x"]
        )
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(spec.gate_noise_rate, 2), ["cx"]
        )

    # --- Prover ---
    t0 = time.time()
    prover = MoSProver(state, seed=spec.seed, noise_model=noise_model)
    msg = prover.run_protocol(
        epsilon=spec.epsilon,
        delta=spec.delta,
        theta=spec.theta,
        qfs_mode=spec.qfs_mode,
        qfs_shots=spec.qfs_shots,
        classical_samples=spec.classical_samples_prover,
    )
    prover_time = time.time() - t0

    # --- Verifier ---
    t1 = time.time()
    verifier = MoSVerifier(state, seed=spec.seed + 1_000_000)

    if spec.k is not None and spec.k > 1:
        result = verifier.verify_fourier_sparse(
            msg,
            epsilon=spec.epsilon,
            k=spec.k,
            delta=spec.delta,
            theta=spec.theta,
            a_sq=spec.a_sq,
            b_sq=spec.b_sq,
            num_samples=spec.classical_samples_verifier,
        )
    else:
        result = verifier.verify_parity(
            msg,
            epsilon=spec.epsilon,
            delta=spec.delta,
            theta=spec.theta,
            a_sq=spec.a_sq,
            b_sq=spec.b_sq,
            num_samples=spec.classical_samples_verifier,
        )
    verifier_time = time.time() - t1

    # --- Extract hypothesis ---
    from ql.verifier import FourierSparseHypothesis

    hyp_s = None
    hyp_coefficients = None
    misclass_rate = None
    correct = False

    if result.accepted and result.hypothesis is not None:
        if isinstance(result.hypothesis, FourierSparseHypothesis):
            hyp_coefficients = dict(result.hypothesis.coefficients)
            hyp_s = max(hyp_coefficients, key=lambda s: abs(hyp_coefficients[s]))
            correct = hyp_s == spec.target_s
            misclass_rate = _compute_misclassification_rate(
                state, result.hypothesis, spec.seed + 2_000_000,
                num_samples=spec.misclassification_samples or 1000,
            )
        else:
            hyp_s = result.hypothesis.s
            correct = hyp_s == spec.target_s if hyp_s is not None else False

    return TrialResult(
        n=spec.n,
        seed=spec.seed,
        prover_time_s=prover_time,
        qfs_shots=spec.qfs_shots,
        qfs_postselected=msg.qfs_result.postselected_shots,
        postselection_rate=msg.qfs_result.postselection_rate,
        list_size=msg.list_size,
        prover_found_target=(spec.target_s in msg.L),
        verifier_time_s=verifier_time,
        verifier_samples=spec.classical_samples_verifier,
        outcome=result.outcome.value,
        accepted=result.accepted,
        accumulated_weight=result.accumulated_weight,
        acceptance_threshold=result.acceptance_threshold,
        hypothesis_s=hyp_s,
        hypothesis_correct=correct,
        total_copies=msg.total_copies_used + spec.classical_samples_verifier,
        total_time_s=prover_time + verifier_time,
        epsilon=spec.epsilon,
        theta=spec.theta,
        delta=spec.delta,
        a_sq=spec.a_sq,
        b_sq=spec.b_sq,
        phi_description=spec.phi_description,
        k=spec.k,
        hypothesis_coefficients=hyp_coefficients,
        misclassification_rate=misclass_rate,
    )


def _extract_spectrum(phi: list[float], threshold: float = 0.01) -> list[tuple[int, float]]:
    """Compute Fourier spectrum of phi and return (index, coefficient) pairs above threshold."""
    from experiments.harness.phi import walsh_hadamard
    phi_arr = np.array(phi)
    tilde_phi = 1.0 - 2.0 * phi_arr
    spectrum = walsh_hadamard(tilde_phi)
    return [(s, float(spectrum[s])) for s in range(len(spectrum)) if abs(spectrum[s]) > threshold]


def _strategy_random_list(n, rng, target_s, epsilon, theta, phi, dummy_sa, dummy_qfs):
    from ql.prover import ProverMessage
    L = sorted(rng.choice(2**n, size=min(5, 2**n), replace=False).tolist())
    return ProverMessage(L, {s: 0.0 for s in L}, n, epsilon, epsilon, dummy_sa, dummy_qfs, 0)


def _strategy_wrong_parity(n, rng, target_s, epsilon, theta, phi, dummy_sa, dummy_qfs):
    from ql.prover import ProverMessage
    wrong_s = (target_s + 1) % (2**n)
    if wrong_s == 0:
        wrong_s = (target_s + 2) % (2**n)
    return ProverMessage([wrong_s], {wrong_s: 1.0}, n, epsilon, epsilon, dummy_sa, dummy_qfs, 0)


def _strategy_partial_list(n, rng, target_s, epsilon, theta, phi, dummy_sa, dummy_qfs):
    from ql.prover import ProverMessage
    return ProverMessage([], {}, n, epsilon, epsilon, dummy_sa, dummy_qfs, 0)


def _strategy_inflated_list(n, rng, target_s, epsilon, theta, phi, dummy_sa, dummy_qfs):
    from ql.prover import ProverMessage
    candidates = [s for s in range(2**n) if s != target_s]
    chosen = sorted(rng.choice(candidates, size=min(10, len(candidates)), replace=False).tolist())
    return ProverMessage(chosen, {s: 0.5 for s in chosen}, n, epsilon, epsilon, dummy_sa, dummy_qfs, 0)


def _strategy_partial_real(n, rng, target_s, epsilon, theta, phi, dummy_sa, dummy_qfs):
    from ql.prover import ProverMessage
    heavy = _extract_spectrum(phi)
    heavy_sorted = sorted(heavy, key=lambda x: abs(x[1]), reverse=True)
    n_real = max(1, len(heavy_sorted) // 2)
    real_part = [s for s, _ in heavy_sorted[n_real:]]
    used = {s for s, _ in heavy}
    fake_candidates = [s for s in range(2**n) if s not in used]
    n_fake = min(3, len(fake_candidates))
    fakes = sorted(rng.choice(fake_candidates, size=n_fake, replace=False).tolist()) if n_fake > 0 else []
    L = sorted(real_part + fakes)
    return ProverMessage(L, {s: 0.5 for s in L}, n, epsilon, theta, dummy_sa, dummy_qfs, 0)


def _strategy_diluted_list(n, rng, target_s, epsilon, theta, phi, dummy_sa, dummy_qfs):
    from ql.prover import ProverMessage
    heavy = _extract_spectrum(phi)
    heavy_sorted = sorted(heavy, key=lambda x: abs(x[1]), reverse=True)
    n_keep = max(1, len(heavy_sorted) // 4)
    kept_indices = [s for s, _ in heavy_sorted[-n_keep:]]
    used = {s for s, _ in heavy}
    padding_candidates = [s for s in range(2**n) if s not in used]
    n_padding = min(20, len(padding_candidates))
    padding = sorted(rng.choice(padding_candidates, size=n_padding, replace=False).tolist()) if n_padding > 0 else []
    L = sorted(kept_indices + padding)
    return ProverMessage(L, {s: 0.5 for s in L}, n, epsilon, theta, dummy_sa, dummy_qfs, 0)


def _strategy_shifted_coefficients(n, rng, target_s, epsilon, theta, phi, dummy_sa, dummy_qfs):
    from ql.prover import ProverMessage
    heavy = _extract_spectrum(phi)
    used = {s for s, _ in heavy}
    wrong_candidates = [s for s in range(2**n) if s not in used]
    n_wrong = min(len(heavy), len(wrong_candidates))
    chosen = sorted(rng.choice(wrong_candidates, size=max(1, n_wrong), replace=False).tolist())
    return ProverMessage(chosen, {s: 0.8 for s in chosen}, n, epsilon, theta, dummy_sa, dummy_qfs, 0)


def _strategy_subset_plus_noise(n, rng, target_s, epsilon, theta, phi, dummy_sa, dummy_qfs):
    from ql.prover import ProverMessage
    heavy = _extract_spectrum(phi)
    heavy_sorted = sorted(heavy, key=lambda x: abs(x[1]), reverse=True)
    heaviest_s = heavy_sorted[0][0] if heavy_sorted else 0
    used = {s for s, _ in heavy}
    fake_candidates = [s for s in range(2**n) if s not in used]
    n_fake = min(5, len(fake_candidates))
    fakes = sorted(rng.choice(fake_candidates, size=n_fake, replace=False).tolist()) if n_fake > 0 else []
    L = sorted([heaviest_s] + fakes)
    return ProverMessage(L, {s: 0.3 for s in L}, n, epsilon, theta, dummy_sa, dummy_qfs, 0)


_DISHONEST_STRATEGIES = {
    "random_list": _strategy_random_list,
    "wrong_parity": _strategy_wrong_parity,
    "partial_list": _strategy_partial_list,
    "inflated_list": _strategy_inflated_list,
    "partial_real": _strategy_partial_real,
    "diluted_list": _strategy_diluted_list,
    "shifted_coefficients": _strategy_shifted_coefficients,
    "subset_plus_noise": _strategy_subset_plus_noise,
}


def _run_dishonest_trial(spec: TrialSpec, state) -> TrialResult:
    r"""Execute a dishonest-prover trial inside a worker process.

    Constructs a fake :class:`~ql.prover.ProverMessage` according to
    the adversarial strategy in ``spec.dishonest_strategy``, then runs
    the verifier against it.  Strategies are registered in
    :data:`_DISHONEST_STRATEGIES`.

    Parameters
    ----------
    spec : TrialSpec
        Trial specification with ``dishonest_strategy`` set.
    state : MoSState
        Reconstructed MoS state (used only by the verifier for
        classical sampling via Lemma 1).

    Returns
    -------
    TrialResult
    """
    from ql.prover import SpectrumApproximation
    from ql.verifier import MoSVerifier
    from mos.sampler import QFSResult

    n = spec.n
    rng = default_rng(spec.seed)
    target_s = spec.target_s
    epsilon = spec.epsilon

    dummy_qfs = QFSResult({}, {}, 0, 0, n, "statevector")
    dummy_sa = SpectrumApproximation({}, 0.0, n, 0, 0)

    strategy_fn = _DISHONEST_STRATEGIES.get(spec.dishonest_strategy)
    if strategy_fn is None:
        raise ValueError(f"Unknown dishonest strategy: {spec.dishonest_strategy}")

    fake_msg = strategy_fn(n, rng, target_s, epsilon, spec.theta, spec.phi, dummy_sa, dummy_qfs)

    verifier = MoSVerifier(state, seed=spec.seed + 1_000_000)
    vresult = verifier.verify_parity(
        fake_msg,
        epsilon=epsilon,
        delta=spec.delta,
        theta=spec.theta,
        a_sq=spec.a_sq,
        b_sq=spec.b_sq,
        num_samples=spec.classical_samples_verifier,
    )

    return TrialResult(
        n=n,
        seed=spec.seed,
        prover_time_s=0.0,
        qfs_shots=0,
        qfs_postselected=0,
        postselection_rate=0.0,
        list_size=len(fake_msg.L),
        prover_found_target=(target_s in fake_msg.L),
        verifier_time_s=0.0,
        verifier_samples=spec.classical_samples_verifier,
        outcome=vresult.outcome.value,
        accepted=vresult.accepted,
        accumulated_weight=vresult.accumulated_weight,
        acceptance_threshold=vresult.acceptance_threshold,
        hypothesis_s=vresult.hypothesis.s if vresult.accepted else None,
        hypothesis_correct=False,
        total_copies=spec.classical_samples_verifier,
        total_time_s=0.0,
        epsilon=epsilon,
        theta=spec.theta,
        delta=spec.delta,
        a_sq=spec.a_sq,
        b_sq=spec.b_sq,
        phi_description=f"soundness_{spec.dishonest_strategy}",
    )


def run_trials_parallel(
    specs: list[TrialSpec],
    max_workers: Optional[int] = None,
    label: str = "",
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> list[TrialResult]:
    r"""Dispatch a batch of trials across worker processes.

    When ``max_workers > 1``, uses
    :class:`~concurrent.futures.ProcessPoolExecutor` with
    :func:`~concurrent.futures.as_completed` for progress reporting.
    Results are stored in an index-mapped list so the output preserves
    the original spec ordering regardless of completion order.

    When ``max_workers == 1``, falls back to sequential execution in
    the main process (useful for debugging — parallel tracebacks are
    less readable).

    When *shard_index* and *num_shards* are both set, only the
    contiguous slice of *specs* assigned to this shard is executed.
    This enables SLURM Job Array distribution where each array task
    regenerates the full spec list but runs only its portion.

    Parameters
    ----------
    specs : list[TrialSpec]
        Trial specifications to execute.
    max_workers : int or None
        Number of worker processes.  ``None`` defaults to
        :func:`os.cpu_count`.
    label : str
        Short label printed in progress output (e.g. ``"scaling"``).
    shard_index : int or None
        0-based index of this shard (requires *num_shards*).
    num_shards : int or None
        Total number of shards (requires *shard_index*).

    Returns
    -------
    list[TrialResult]
        Results in the same order as the (possibly sharded) *specs*.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    # --- Apply sharding if requested ---
    if shard_index is not None and num_shards is not None:
        from experiments.harness.sharding import shard_specs

        full_count = len(specs)
        specs = shard_specs(specs, shard_index, num_shards)
        label = f"{label} shard {shard_index + 1}/{num_shards}" if label else f"shard {shard_index + 1}/{num_shards}"
        print(
            f"  Shard {shard_index + 1}/{num_shards}: {len(specs)} of {full_count} specs",
            flush=True,
        )

    total = len(specs)
    if total == 0:
        return []

    if max_workers <= 1:
        # Sequential — easier to debug
        results = []
        for i, spec in enumerate(specs, 1):
            t = _run_trial_worker(spec)
            _print_trial_progress(t, i, total, label)
            results.append(t)
        return results

    # Parallel
    results: list[Optional[TrialResult]] = [None] * total
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:

        def _terminate_pool(signum, frame):
            """Kill all worker processes on SIGTERM/SIGINT."""
            pool.shutdown(wait=False, cancel_futures=True)
            if hasattr(pool, "_processes"):
                for proc in pool._processes.values():
                    if proc.is_alive():
                        proc.kill()
            sys.exit(128 + signum)

        prev_term = signal.signal(signal.SIGTERM, _terminate_pool)
        prev_int = signal.signal(signal.SIGINT, _terminate_pool)

        try:
            future_to_idx = {
                pool.submit(_run_trial_worker, spec): idx
                for idx, spec in enumerate(specs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                t = future.result()
                results[idx] = t
                completed += 1
                _print_trial_progress(t, completed, total, label)
        finally:
            signal.signal(signal.SIGTERM, prev_term)
            signal.signal(signal.SIGINT, prev_int)

    return results  # type: ignore[return-value]


def _print_trial_progress(t: TrialResult, completed: int, total: int, label: str):
    """Print a single-line progress update for a completed trial."""
    status = (
        "✓" if t.hypothesis_correct else ("accept_wrong" if t.accepted else "✗reject")
    )
    prefix = f"  [{label}] " if label else "  "
    print(
        f"{prefix}{completed:4d}/{total}: n={t.n:2d} "
        f"|L|={t.list_size:3d} {status:12s} {t.total_time_s:.2f}s "
        f"({t.phi_description})",
        flush=True,
    )
