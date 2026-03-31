r"""
Experiment harness for the MoS verification protocol.

Runs instrumented experiments that produce the data needed for
dissertation figures and analysis, with optional process-level
parallelism via :class:`~concurrent.futures.ProcessPoolExecutor`.

Covers five experimental directions from Caro et al. [ITCS2024]_:

1. **Scaling** (:math:`n = 4 \to 16{+}`):
   Sweep :math:`n` with Goldreich--Levin extraction, recording copy
   complexity, post-selection rate, completeness probability, and
   wall-clock time.

2. **Bent function worst-case**:
   Bent functions have maximally flat Fourier spectra
   (:math:`|\hat{\tilde\phi}(s)| = 2^{-n/2}` for all :math:`s`),
   the hardest case for heavy coefficient extraction (Corollary 5).

3. **Verifier truncation tradeoffs**:
   Vary the verifier's classical sample budget :math:`m_V` and accuracy
   parameter :math:`\varepsilon` to map the completeness/soundness
   tradeoff surface (Theorems 8, 12).

4. **Noise sweep**:
   Random :math:`\varphi` functions drawn from the noisy parity ensemble
   at varying label-flip rate :math:`\eta`, testing the effective
   coefficient regime :math:`\hat{\tilde\phi}_{\mathrm{eff}}(s) =
   (1-2\eta)\,\hat{\tilde\phi}(s)` (Definition 5(iii), §6.2).

5. **Soundness verification**:
   Inject dishonest provers with adversarial strategies and measure
   empirical rejection rates against the information-theoretic soundness
   guarantee (Definition 7).

All experiments write results to structured JSON files suitable for
direct plotting with matplotlib or pgfplots.

Usage
-----
Run individual experiments::

    python -m experiments.harness --experiment scaling --n-max 12 --workers 8

Run all experiments::

    python -m experiments.harness --all --workers 4

Programmatic use::

    from experiments.harness import run_scaling_experiment
    results = run_scaling_experiment(
        n_range=range(4, 13), num_trials=20, max_workers=8
    )
    results.save("scaling_results.json")

.. [ITCS2024] M.\,C. Caro, M. Hinsche, M. Ioannou, A. Nietner, and
   R. Sweke, "Classical Verification of Quantum Learning," *ITCS 2024*,
   :doi:`10.4230/LIPIcs.ITCS.2024.24`.
"""

import json
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.random import default_rng


# ===================================================================
# Result containers
# ===================================================================


@dataclass
class TrialResult:
    r"""Result of a single prover--verifier trial.

    Captures every observable quantity from one execution of the
    interactive verification protocol (§6 of [ITCS2024]_):
    prover-side QFS, verifier-side classical estimation, the
    accept/reject outcome, and the output hypothesis.
    """

    #: Number of input bits (dimension of
    #: :math:`\mathcal{X}_n = \{0,1\}^n`).
    n: int
    #: Random seed governing this trial's RNG chain.
    seed: int
    #: Wall-clock time for the prover's computation (seconds).
    prover_time_s: float
    #: Number of MoS copies consumed by Quantum Fourier Sampling.
    qfs_shots: int
    #: Number of QFS shots that survived post-selection on the
    #: label qubit :math:`b = 1` (Theorem 5(i)).
    qfs_postselected: int
    #: Fraction of QFS shots surviving post-selection; should
    #: concentrate around :math:`1/2`.
    postselection_rate: float
    #: :math:`|L|`, the number of candidate heavy Fourier coefficient
    #: indices sent by the prover.
    list_size: int
    #: Whether the target parity :math:`s^*` appears in :math:`L`.
    prover_found_target: bool
    #: Wall-clock time for the verifier's computation (seconds).
    verifier_time_s: float
    #: Number of classical random examples consumed by the verifier
    #: for independent coefficient estimation (Lemma 1).
    verifier_samples: int
    #: Verification outcome: ``"accept"``,
    #: ``"reject_list_too_large"``, or
    #: ``"reject_insufficient_weight"``.
    outcome: str
    #: Whether the verifier accepted the interaction.
    accepted: bool
    #: :math:`\sum_{s \in L} \hat{\xi}(s)^2`, the Fourier weight
    #: accumulated by the verifier's independent estimates.
    accumulated_weight: float
    #: The threshold :math:`\tau` against which
    #: :attr:`accumulated_weight` was compared.  For parity
    #: (Theorem 12): :math:`\tau = a^2 - \varepsilon^2/8`.
    acceptance_threshold: float
    #: The parity index :math:`s_{\mathrm{out}}` of the output
    #: hypothesis, or ``None`` if rejected.
    hypothesis_s: Optional[int]
    #: Whether :math:`s_{\mathrm{out}} = s^*`.
    hypothesis_correct: bool
    #: Total MoS copies consumed (QFS + prover classical + verifier
    #: classical).
    total_copies: int
    #: Total wall-clock time (prover + verifier).
    total_time_s: float
    #: Accuracy parameter :math:`\varepsilon`.
    epsilon: float
    #: Fourier resolution threshold :math:`\vartheta`.
    theta: float
    #: Confidence parameter :math:`\delta`.
    delta: float
    #: Lower bound :math:`a^2` on
    #: :math:`\mathbb{E}_{x \sim U_n}[\tilde\phi(x)^2]`
    #: (Definition 14).
    a_sq: float
    #: Upper bound :math:`b^2` on
    #: :math:`\mathbb{E}_{x \sim U_n}[\tilde\phi(x)^2]`
    #: (Definition 14).
    b_sq: float
    #: Human-readable label for the distribution under test.
    phi_description: str


@dataclass
class ExperimentResult:
    r"""Aggregated results from an experiment sweep.

    Collects :class:`TrialResult` instances from all trials in an
    experiment, together with the sweep parameters and timing metadata.
    Supports serialisation to JSON and tabular summary output.
    """

    #: Identifier for this experiment (e.g. ``"scaling"``).
    experiment_name: str
    #: ISO 8601 timestamp of when the experiment was run.
    timestamp: str
    #: Total wall-clock time for the experiment (seconds).
    wall_clock_s: float = 0.0
    #: Number of parallel worker processes used.
    max_workers: int = 1
    #: Individual trial results.
    trials: list[TrialResult] = field(default_factory=list)
    #: Experiment-level configuration (sweep ranges, shot counts, etc.).
    parameters: dict = field(default_factory=dict)

    def save(self, path: str):
        """Serialise the experiment results to a JSON file.

        Parameters
        ----------
        path : str
            Output file path.  Parent directories are created
            automatically.
        """
        data = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "wall_clock_s": self.wall_clock_s,
            "max_workers": self.max_workers,
            "parameters": self.parameters,
            "num_trials": len(self.trials),
            "trials": [asdict(t) for t in self.trials],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        print(f"  Saved {len(self.trials)} trials to {path}")

    def summary_table(self) -> str:
        r"""Produce a human-readable summary table grouped by :math:`n`.

        Columns: number of trials, acceptance rate, correctness rate,
        median :math:`|L|`, median total copies, median prover time,
        and median verifier time.

        Returns
        -------
        str
            Formatted ASCII table.
        """
        rows_by_n: dict[int, list[TrialResult]] = {}
        for t in self.trials:
            rows_by_n.setdefault(t.n, []).append(t)

        lines = [
            f"{'n':>3} {'trials':>6} {'accept%':>8} {'correct%':>9} "
            f"{'|L| med':>7} {'copies med':>10} {'prover_s':>9} {'verif_s':>8}"
        ]
        lines.append("-" * 75)

        for n in sorted(rows_by_n):
            trials = rows_by_n[n]
            k = len(trials)
            accept_rate = np.mean([t.accepted for t in trials])
            correct_rate = np.mean([t.hypothesis_correct for t in trials])
            med_L = np.median([t.list_size for t in trials])
            med_copies = np.median([t.total_copies for t in trials])
            med_prover = np.median([t.prover_time_s for t in trials])
            med_verif = np.median([t.verifier_time_s for t in trials])
            lines.append(
                f"{n:3d} {k:6d} {accept_rate:8.1%} {correct_rate:9.1%} "
                f"{med_L:7.0f} {med_copies:10.0f} {med_prover:9.3f} {med_verif:8.3f}"
            )

        return "\n".join(lines)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ===================================================================
# Serialisable trial specification
# ===================================================================
#
# ProcessPoolExecutor needs picklable arguments.  MoSState contains
# NumPy arrays and Qiskit objects that don't always pickle cleanly.
# Instead, we pass a plain dict describing how to reconstruct the
# state in the worker process.  This also means each worker imports
# mos/ql independently — no shared mutable state.
#


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


# ===================================================================
# Worker function (runs in child process)
# ===================================================================


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

    # --- Prover ---
    t0 = time.time()
    prover = MoSProver(state, seed=spec.seed)
    msg = prover.run_protocol(
        epsilon=spec.epsilon,
        delta=spec.delta,
        theta=spec.theta,
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


# ===================================================================
# Parallel dispatch
# ===================================================================


def run_trials_parallel(
    specs: list[TrialSpec],
    max_workers: Optional[int] = None,
    label: str = "",
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

    Parameters
    ----------
    specs : list[TrialSpec]
        Trial specifications to execute.
    max_workers : int or None
        Number of worker processes.  ``None`` defaults to
        :func:`os.cpu_count`.
    label : str
        Short label printed in progress output (e.g. ``"scaling"``).

    Returns
    -------
    list[TrialResult]
        Results in the same order as *specs*.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

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
        future_to_idx = {
            pool.submit(_run_trial_worker, spec): idx for idx, spec in enumerate(specs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            t = future.result()
            results[idx] = t
            completed += 1
            _print_trial_progress(t, completed, total, label)

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


# ===================================================================
# Phi generators
# ===================================================================


def make_single_parity(n: int, target_s: int) -> list[float]:
    r"""Construct :math:`\varphi` for a pure parity function.

    .. math::

        \varphi(x) = s^* \cdot x \bmod 2

    so that :math:`\tilde\phi = \chi_{s^*}` and the Fourier spectrum has
    a single nonzero coefficient :math:`\hat{\tilde\phi}(s^*) = 1`.

    Parameters
    ----------
    n : int
        Number of input bits.
    target_s : int
        Parity index :math:`s^* \in \{0, \ldots, 2^n - 1\}`.

    Returns
    -------
    list[float]
        :math:`\varphi(x)` for :math:`x = 0, \ldots, 2^n - 1`.
    """
    return [float(bin(target_s & x).count("1") % 2) for x in range(2**n)]


def make_random_parity(n: int, rng: np.random.Generator) -> tuple[list[float], int]:
    r"""Construct :math:`\varphi` for a uniformly random nonzero parity.

    Draws :math:`s^* \sim \mathrm{Uniform}(\{1, \ldots, 2^n - 1\})` and
    returns the corresponding :math:`\varphi` via :func:`make_single_parity`.

    Parameters
    ----------
    n : int
        Number of input bits.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    phi : list[float]
        Label-bias function.
    target_s : int
        The sampled parity index.
    """
    s = int(rng.integers(1, 2**n))
    return make_single_parity(n, s), s


def make_bent_function(n: int) -> list[float]:
    r"""Construct :math:`\varphi` for a Maiorana--McFarland bent function.

    For even :math:`n`, defines :math:`f(x, y) = \langle x, y \rangle \bmod 2`
    where :math:`x, y \in \{0,1\}^{n/2}`.  The resulting
    :math:`g = (-1)^f` has all Fourier coefficients equal in magnitude:

    .. math::

        |\hat{g}(s)| = 2^{-n/2} \quad \forall\, s \in \{0,1\}^n

    This is the worst case for heavy coefficient extraction because no
    coefficient dominates: Parseval gives :math:`\sum_s \hat{g}(s)^2 = 1`
    spread uniformly over all :math:`2^n` frequencies.

    Parameters
    ----------
    n : int
        Number of input bits (must be even).

    Returns
    -------
    list[float]
        :math:`\varphi(x)` for :math:`x = 0, \ldots, 2^n - 1`.

    Raises
    ------
    ValueError
        If *n* is odd.
    """
    if n % 2 != 0:
        raise ValueError(f"Bent functions require even n, got {n}")
    half = n // 2
    phi = []
    for z in range(2**n):
        x_bits = z & ((1 << half) - 1)
        y_bits = (z >> half) & ((1 << half) - 1)
        phi.append(float(bin(x_bits & y_bits).count("1") % 2))
    return phi


# ===================================================================
# Experiment 1: Scaling with n
# ===================================================================


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
    trials = run_trials_parallel(specs, max_workers=max_workers, label="scaling")
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


# ===================================================================
# Experiment 2: Bent function worst-case
# ===================================================================


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
    trials = run_trials_parallel(specs, max_workers=max_workers, label="bent")
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


# ===================================================================
# Experiment 3: Verifier truncation tradeoffs
# ===================================================================


def run_truncation_experiment(
    n: int = 6,
    epsilon_range: Optional[list[float]] = None,
    verifier_sample_range: Optional[list[int]] = None,
    num_trials: int = 20,
    qfs_shots: int = 2000,
    classical_samples_prover: int = 1000,
    base_seed: int = 42,
    max_workers: int = 1,
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
    becomes lenient.  The 2D grid
    :math:`(\varepsilon \times m_V)` maps the tradeoff surface
    predicted by Theorem 12.

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
    trials = run_trials_parallel(specs, max_workers=max_workers, label="trunc")
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


# ===================================================================
# Experiment 4: Noise sweep
# ===================================================================


def run_noise_sweep_experiment(
    n: int = 6,
    noise_rates: Optional[list[float]] = None,
    num_trials: int = 20,
    epsilon: float = 0.3,
    qfs_shots: int = 2000,
    classical_samples_prover: int = 1000,
    classical_samples_verifier: int = 3000,
    base_seed: int = 42,
    max_workers: int = 1,
) -> ExperimentResult:
    r"""Noise sweep: verification under increasing label-flip noise.

    For each noise rate :math:`\eta`, the MoS state is constructed from
    the effective label probabilities (Definition 5(iii)):

    .. math::

        \varphi_{\mathrm{eff}}(x) = (1 - 2\eta)\,\varphi(x) + \eta

    The effective Fourier coefficient becomes
    :math:`\hat{\tilde\phi}_{\mathrm{eff}}(s) = (1 - 2\eta)\,
    \hat{\tilde\phi}(s)`, and the distribution class promise is
    :math:`a^2 = b^2 = (1 - 2\eta)^2`.

    As :math:`\eta \to 0.5`, the signal :math:`(1 - 2\eta) \to 0` and
    the protocol should eventually fail.  The experiment measures the
    empirical acceptance and correctness rates as functions of
    :math:`\eta`, testing the noise-robust verification results of §6.2.

    The Fourier resolution threshold :math:`\vartheta` is adapted per
    noise level: :math:`\vartheta = \min(\varepsilon,\,
    0.9 \cdot (1 - 2\eta))` to avoid setting the extraction threshold
    above the signal magnitude.

    Parameters
    ----------
    n : int
        Number of input bits.
    noise_rates : list[float] or None
        Values of :math:`\eta` to sweep.
        Default: ``[0.0, 0.05, 0.1, ..., 0.4]``.
    num_trials : int
        Trials per noise level.
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
    if noise_rates is None:
        noise_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    print(f"=== Noise Sweep: n={n}, eta in {noise_rates}, {max_workers} workers ===")
    rng = default_rng(base_seed)

    specs: list[TrialSpec] = []
    for eta in noise_rates:
        effective_coeff = 1.0 - 2.0 * eta
        a_sq = effective_coeff**2
        theta = min(epsilon, effective_coeff * 0.9) if effective_coeff > 0.01 else 0.01

        for _ in range(num_trials):
            seed = int(rng.integers(0, 2**31))
            trial_rng = default_rng(seed)
            phi, target_s = make_random_parity(n, trial_rng)
            specs.append(
                TrialSpec(
                    n=n,
                    phi=phi,
                    noise_rate=eta,
                    target_s=target_s,
                    epsilon=epsilon,
                    delta=0.1,
                    theta=theta,
                    a_sq=a_sq,
                    b_sq=a_sq,
                    qfs_shots=qfs_shots,
                    classical_samples_prover=classical_samples_prover,
                    classical_samples_verifier=classical_samples_verifier,
                    seed=seed,
                    phi_description=f"noisy_parity_eta={eta}_s={target_s}",
                )
            )

    t0 = time.time()
    trials = run_trials_parallel(specs, max_workers=max_workers, label="noise")
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="noise_sweep",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n": n,
            "noise_rates": noise_rates,
            "num_trials": num_trials,
            "epsilon": epsilon,
        },
    )

    # Print per-eta summary
    print(f"\n  {'eta':>5s} {'eff_coeff':>9s} {'accept%':>8s} {'correct%':>9s}")
    for eta in noise_rates:
        et = [t for t in trials if f"eta={eta}" in t.phi_description]
        ar = np.mean([t.accepted for t in et]) if et else 0.0
        cr = np.mean([t.hypothesis_correct for t in et]) if et else 0.0
        print(f"  {eta:5.2f} {1 - 2 * eta:9.2f} {ar:8.0%} {cr:9.0%}")

    print(f"  Wall-clock time: {wall:.1f}s")
    return result


# ===================================================================
# Experiment 5: Soundness against dishonest provers
# ===================================================================


def run_soundness_experiment(
    n: int = 6,
    num_trials: int = 50,
    epsilon: float = 0.3,
    classical_samples_verifier: int = 3000,
    base_seed: int = 42,
    max_workers: int = 1,
) -> ExperimentResult:
    r"""Empirical soundness against dishonest provers.

    Tests four adversarial prover strategies against the verifier's
    information-theoretic soundness guarantee (Definition 7):

    ``"random_list"``
        Prover sends 5 uniformly random frequency indices.  Expected
        acceptance rate :math:`\approx 5/2^n` (the probability of
        accidentally including :math:`s^*`).

    ``"wrong_parity"``
        Prover sends a single incorrect index :math:`s \neq s^*`.
        The verifier's independent estimate will be
        :math:`\hat\xi(s) \approx 0`, causing rejection.

    ``"partial_list"``
        Prover sends an empty list.  Accumulated weight is 0, well
        below the threshold :math:`1 - \varepsilon^2/8`.

    ``"inflated_list"``
        Prover sends 10 wrong indices with fabricated estimates.
        The verifier's independent estimates expose all of them as
        having negligible true Fourier weight.

    The empirical rejection rate should be
    :math:`\geq 1 - \delta` for all strategies (excluding the
    combinatorial chance of ``"random_list"`` hitting :math:`s^*`).

    Parameters
    ----------
    n : int
        Number of input bits.
    num_trials : int
        Number of trials per adversarial strategy.
    epsilon : float
        Accuracy parameter :math:`\varepsilon`.
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
    strategies = ["random_list", "wrong_parity", "partial_list", "inflated_list"]

    print(
        f"=== Soundness Experiment: n={n}, {num_trials} trials/strategy, "
        f"{max_workers} workers ==="
    )
    rng = default_rng(base_seed)

    target_s = 1
    phi = make_single_parity(n, target_s)

    specs: list[TrialSpec] = []
    for strategy in strategies:
        for _ in range(num_trials):
            seed = int(rng.integers(0, 2**31))
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
                    qfs_shots=0,
                    classical_samples_prover=0,
                    classical_samples_verifier=classical_samples_verifier,
                    seed=seed,
                    phi_description=f"soundness_{strategy}",
                    dishonest_strategy=strategy,
                )
            )

    t0 = time.time()
    trials = run_trials_parallel(specs, max_workers=max_workers, label="sound")
    wall = time.time() - t0

    result = ExperimentResult(
        experiment_name="soundness",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=wall,
        max_workers=max_workers,
        trials=trials,
        parameters={
            "n": n,
            "num_trials": num_trials,
            "epsilon": epsilon,
            "strategies": strategies,
        },
    )

    # Print per-strategy rejection rates
    print()
    for strategy in strategies:
        st = [t for t in trials if strategy in t.phi_description]
        rej = sum(1 for t in st if not t.accepted)
        print(f"  {strategy:20s}: rejected {rej}/{len(st)} ({rej / len(st):.0%})")

    print(f"  Wall-clock time: {wall:.1f}s")
    return result


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="MoS verification protocol experiments",
    )
    parser.add_argument(
        "--experiment",
        choices=["scaling", "bent", "truncation", "noise", "soundness", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument("--n-max", type=int, default=10, help="Maximum n for scaling")
    parser.add_argument("--n-min", type=int, default=4, help="Minimum n for scaling")
    parser.add_argument(
        "--trials", type=int, default=20, help="Trials per configuration"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (1 = sequential)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    experiments = []
    t_total = time.time()

    if args.experiment in ("scaling", "all"):
        r = run_scaling_experiment(
            n_range=range(args.n_min, args.n_max + 1),
            num_trials=args.trials,
            base_seed=args.seed,
            max_workers=args.workers,
        )
        r.save(str(output_dir / "scaling.json"))
        experiments.append(r)

    if args.experiment in ("bent", "all"):
        bent_max = args.n_max if args.n_max % 2 == 0 else args.n_max - 1
        r = run_bent_experiment(
            n_range=range(4, bent_max + 1, 2),
            num_trials=args.trials,
            base_seed=args.seed,
            max_workers=args.workers,
        )
        r.save(str(output_dir / "bent.json"))
        experiments.append(r)

    if args.experiment in ("truncation", "all"):
        r = run_truncation_experiment(
            n=6,
            num_trials=args.trials,
            base_seed=args.seed,
            max_workers=args.workers,
        )
        r.save(str(output_dir / "truncation.json"))
        experiments.append(r)

    if args.experiment in ("noise", "all"):
        r = run_noise_sweep_experiment(
            n=6,
            num_trials=args.trials,
            base_seed=args.seed,
            max_workers=args.workers,
        )
        r.save(str(output_dir / "noise_sweep.json"))
        experiments.append(r)

    if args.experiment in ("soundness", "all"):
        r = run_soundness_experiment(
            n=6,
            num_trials=max(args.trials, 50),
            base_seed=args.seed,
            max_workers=args.workers,
        )
        r.save(str(output_dir / "soundness.json"))
        experiments.append(r)

    wall_total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Done. {len(experiments)} experiment(s) saved to {output_dir}/")
    print(f"Total wall-clock time: {wall_total:.1f}s")
    if args.workers > 1:
        seq_est = sum(sum(t.total_time_s for t in e.trials) for e in experiments)
        print(f"Estimated sequential time: {seq_est:.1f}s")
        if wall_total > 0:
            print(
                f"Parallel efficiency: {seq_est / wall_total:.1f}x on {args.workers} workers"
            )


if __name__ == "__main__":
    main()
