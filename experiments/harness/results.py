"""Result containers for experiment trials and sweeps."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from experiments.proto import (
    common_pb2,
    scaling_pb2,
    bent_pb2,
    noise_sweep_pb2,
    soundness_pb2,
    average_case_pb2,
    gate_noise_pb2,
    k_sparse_pb2,
    soundness_multi_pb2,
    theta_sensitivity_pb2,
    ab_regime_pb2,
)


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
    #: Fourier sparsity parameter (``None`` for parity experiments).
    k: Optional[int] = None
    #: For k-sparse hypotheses, maps each selected parity index to its
    #: estimated Fourier coefficient.  ``None`` for parity experiments.
    hypothesis_coefficients: Optional[dict[int, float]] = None
    #: Empirical misclassification rate :math:`\hat{P}[h(x) \neq y]`
    #: on fresh samples.  ``None`` for parity experiments.
    misclassification_rate: Optional[float] = None


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
        """Serialise the experiment results to a Protocol Buffer file.

        Parameters
        ----------
        path : str
            Output file path (conventionally ``*.pb``).  Parent
            directories are created automatically.
        """
        pb = self._to_proto()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(pb.SerializeToString())
        print(f"  Saved {len(self.trials)} trials to {path}")

    def _to_proto(self):
        """Convert to the experiment-specific protobuf message."""
        metadata = common_pb2.ExperimentMetadata(
            experiment_name=self.experiment_name,
            timestamp=self.timestamp,
            wall_clock_s=self.wall_clock_s,
            max_workers=self.max_workers,
            num_trials=len(self.trials),
        )
        trial_pbs = [_trial_to_proto(t) for t in self.trials]
        params = self.parameters

        if self.experiment_name == "scaling":
            return scaling_pb2.ScalingExperimentResult(
                metadata=metadata,
                parameters=scaling_pb2.ScalingParameters(
                    n_range=params["n_range"],
                    num_trials=params["num_trials"],
                    epsilon=params["epsilon"],
                    delta=params["delta"],
                    qfs_shots=params["qfs_shots"],
                    classical_samples_prover=params["classical_samples_prover"],
                    classical_samples_verifier=params["classical_samples_verifier"],
                ),
                trials=trial_pbs,
            )
        elif self.experiment_name == "bent_function":
            return bent_pb2.BentExperimentResult(
                metadata=metadata,
                parameters=bent_pb2.BentParameters(
                    n_range=params["n_range"],
                    num_trials=params["num_trials"],
                    epsilon=params["epsilon"],
                    theta=params["theta"],
                    qfs_shots=params["qfs_shots"],
                    note=params.get("note", ""),
                ),
                trials=trial_pbs,
            )
        elif self.experiment_name == "noise_sweep":
            return noise_sweep_pb2.NoiseSweepExperimentResult(
                metadata=metadata,
                parameters=noise_sweep_pb2.NoiseSweepParameters(
                    n_range=params["n_range"],
                    noise_rates=params["noise_rates"],
                    num_trials=params["num_trials"],
                    epsilon=params["epsilon"],
                ),
                trials=trial_pbs,
            )
        elif self.experiment_name == "soundness":
            return soundness_pb2.SoundnessExperimentResult(
                metadata=metadata,
                parameters=soundness_pb2.SoundnessParameters(
                    n_range=params["n_range"],
                    num_trials=params["num_trials"],
                    epsilon=params["epsilon"],
                    strategies=params["strategies"],
                ),
                trials=trial_pbs,
            )
        elif self.experiment_name == "soundness_multi":
            return soundness_multi_pb2.SoundnessMultiExperimentResult(
                metadata=metadata,
                parameters=soundness_multi_pb2.SoundnessMultiParameters(
                    n_range=params["n_range"],
                    k_range=params["k_range"],
                    num_trials=params["num_trials"],
                    epsilon=params["epsilon"],
                    strategies=params["strategies"],
                ),
                trials=trial_pbs,
            )
        elif self.experiment_name == "gate_noise":
            return gate_noise_pb2.GateNoiseExperimentResult(
                metadata=metadata,
                parameters=gate_noise_pb2.GateNoiseParameters(
                    n_range=params["n_range"],
                    gate_noise_rates=params["gate_noise_rates"],
                    num_trials=params["num_trials"],
                    epsilon=params["epsilon"],
                ),
                trials=trial_pbs,
            )
        elif self.experiment_name == "average_case":
            return average_case_pb2.AverageCaseExperimentResult(
                metadata=metadata,
                parameters=average_case_pb2.AverageCaseParameters(
                    n_range=params["n_range"],
                    families=params["families"],
                    num_trials=params["num_trials"],
                    epsilon=params["epsilon"],
                    delta=params["delta"],
                    qfs_shots=params["qfs_shots"],
                    classical_samples_prover=params["classical_samples_prover"],
                    classical_samples_verifier=params["classical_samples_verifier"],
                ),
                trials=trial_pbs,
            )
        elif self.experiment_name == "k_sparse":
            return k_sparse_pb2.KSparseExperimentResult(
                metadata=metadata,
                parameters=k_sparse_pb2.KSparseParameters(
                    n_range=params["n_range"],
                    k_values=params["k_values"],
                    num_trials=params["num_trials"],
                    epsilon=params["epsilon"],
                    delta=params["delta"],
                    qfs_shots=params["qfs_shots"],
                    classical_samples_prover=params["classical_samples_prover"],
                    classical_samples_verifier=params["classical_samples_verifier"],
                    misclassification_samples=params["misclassification_samples"],
                ),
                trials=trial_pbs,
            )
        elif self.experiment_name == "theta_sensitivity":
            return theta_sensitivity_pb2.ThetaSensitivityExperimentResult(
                metadata=metadata,
                parameters=theta_sensitivity_pb2.ThetaSensitivityParameters(
                    n_range=params["n_range"],
                    theta_values=params["theta_values"],
                    num_trials=params["num_trials"],
                    epsilon=params["epsilon"],
                    delta=params["delta"],
                    qfs_shots=params["qfs_shots"],
                    classical_samples_prover=params["classical_samples_prover"],
                    classical_samples_verifier=params["classical_samples_verifier"],
                ),
                trials=trial_pbs,
            )
        elif self.experiment_name == "ab_regime":
            return ab_regime_pb2.AbRegimeExperimentResult(
                metadata=metadata,
                parameters=ab_regime_pb2.AbRegimeParameters(
                    n_range=params["n_range"],
                    gaps=params["gaps"],
                    num_trials=params["num_trials"],
                    epsilon=params["epsilon"],
                ),
                trials=trial_pbs,
            )
        else:
            raise ValueError(f"No proto schema for experiment: {self.experiment_name}")

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


def _trial_to_proto(t: TrialResult) -> common_pb2.TrialResult:
    """Convert a TrialResult dataclass to its protobuf representation."""
    pb = common_pb2.TrialResult(
        n=int(t.n),
        seed=int(t.seed),
        prover_time_s=float(t.prover_time_s),
        qfs_shots=int(t.qfs_shots),
        qfs_postselected=int(t.qfs_postselected),
        postselection_rate=float(t.postselection_rate),
        list_size=int(t.list_size),
        prover_found_target=bool(t.prover_found_target),
        verifier_time_s=float(t.verifier_time_s),
        verifier_samples=int(t.verifier_samples),
        outcome=str(t.outcome),
        accepted=bool(t.accepted),
        accumulated_weight=float(t.accumulated_weight),
        acceptance_threshold=float(t.acceptance_threshold),
        hypothesis_correct=bool(t.hypothesis_correct),
        total_copies=int(t.total_copies),
        total_time_s=float(t.total_time_s),
        epsilon=float(t.epsilon),
        theta=float(t.theta),
        delta=float(t.delta),
        a_sq=float(t.a_sq),
        b_sq=float(t.b_sq),
        phi_description=str(t.phi_description),
    )
    if t.hypothesis_s is not None:
        pb.hypothesis_s = int(t.hypothesis_s)
    if t.k is not None:
        pb.k = int(t.k)
    if t.hypothesis_coefficients is not None:
        for s_idx, coeff in t.hypothesis_coefficients.items():
            pb.hypothesis_coefficients[int(s_idx)] = float(coeff)
    if t.misclassification_rate is not None:
        pb.misclassification_rate = float(t.misclassification_rate)
    return pb
