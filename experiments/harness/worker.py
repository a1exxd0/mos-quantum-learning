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

    hyp_s = result.hypothesis.s if result.accepted and result.hypothesis else None
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
    )


def _run_dishonest_trial(spec: TrialSpec, state) -> TrialResult:
    r"""Execute a dishonest-prover trial inside a worker process.

    Constructs a fake :class:`~ql.prover.ProverMessage` according to
    the adversarial strategy in ``spec.dishonest_strategy``, then runs
    the verifier against it.  The four strategies test distinct failure
    modes of information-theoretic soundness (Definition 7):

    ``"random_list"``
        Prover sends 5 uniformly random indices (no QFS).  Expected
        acceptance rate :math:`\approx 5 / 2^n` (chance inclusion of
        :math:`s^*`).

    ``"wrong_parity"``
        Prover claims a single wrong index :math:`s \neq s^*` is heavy.

    ``"partial_list"``
        Prover sends an empty list, omitting all heavy coefficients.

    ``"inflated_list"``
        Prover sends 10 wrong indices (excluding :math:`s^*`) with
        fabricated coefficient estimates.

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
    from ql.prover import ProverMessage, SpectrumApproximation
    from ql.verifier import MoSVerifier
    from mos.sampler import QFSResult

    n = spec.n
    rng = default_rng(spec.seed)
    target_s = spec.target_s
    epsilon = spec.epsilon

    dummy_qfs = QFSResult({}, {}, 0, 0, n, "statevector")
    dummy_sa = SpectrumApproximation({}, 0.0, n, 0, 0)

    if spec.dishonest_strategy == "random_list":
        L = sorted(rng.choice(2**n, size=min(5, 2**n), replace=False).tolist())
        fake_msg = ProverMessage(
            L, {s: 0.0 for s in L}, n, epsilon, epsilon, dummy_sa, dummy_qfs, 0
        )
    elif spec.dishonest_strategy == "wrong_parity":
        wrong_s = (target_s + 1) % (2**n)
        if wrong_s == 0:
            wrong_s = (target_s + 2) % (2**n)
        fake_msg = ProverMessage(
            [wrong_s], {wrong_s: 1.0}, n, epsilon, epsilon, dummy_sa, dummy_qfs, 0
        )
    elif spec.dishonest_strategy == "partial_list":
        fake_msg = ProverMessage([], {}, n, epsilon, epsilon, dummy_sa, dummy_qfs, 0)
    elif spec.dishonest_strategy == "inflated_list":
        candidates = [s for s in range(2**n) if s != target_s]
        chosen = sorted(
            rng.choice(
                candidates, size=min(10, len(candidates)), replace=False
            ).tolist()
        )
        fake_msg = ProverMessage(
            chosen,
            {s: 0.5 for s in chosen},
            n,
            epsilon,
            epsilon,
            dummy_sa,
            dummy_qfs,
            0,
        )
    else:
        raise ValueError(f"Unknown dishonest strategy: {spec.dishonest_strategy}")

    verifier = MoSVerifier(state, seed=spec.seed + 1_000_000)
    vresult = verifier.verify_parity(
        fake_msg,
        epsilon=epsilon,
        delta=spec.delta,
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
        theta=epsilon,
        delta=spec.delta,
        a_sq=1.0,
        b_sq=1.0,
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
        label = f"{label} shard {shard_index}/{num_shards}" if label else f"shard {shard_index}/{num_shards}"
        print(
            f"  Shard {shard_index}/{num_shards}: {len(specs)} of {full_count} specs",
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
