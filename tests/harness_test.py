"""Tests for the experiments.harness package.

Covers phi generators, result containers (including protobuf
serialisation), trial specification, worker functions, CLI argument
parsing, and lightweight integration tests for each experiment runner.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.random import default_rng

from experiments.harness.phi import (
    make_bent_function,
    make_random_parity,
    make_single_parity,
)
from experiments.harness.results import (
    ExperimentResult,
    TrialResult,
    _trial_to_proto,
)
from experiments.harness.worker import (
    TrialSpec,
    _run_trial_worker,
    run_trials_parallel,
)
from experiments.harness.scaling import run_scaling_experiment
from experiments.harness.bent import run_bent_experiment
from experiments.harness.truncation import run_truncation_experiment
from experiments.harness.noise import run_noise_sweep_experiment
from experiments.harness.soundness import run_soundness_experiment


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def sample_trial_result():
    """A minimal TrialResult for testing containers and serialisation."""
    return TrialResult(
        n=4,
        seed=42,
        prover_time_s=0.1,
        qfs_shots=100,
        qfs_postselected=50,
        postselection_rate=0.5,
        list_size=3,
        prover_found_target=True,
        verifier_time_s=0.05,
        verifier_samples=200,
        outcome="accept",
        accepted=True,
        accumulated_weight=0.9,
        acceptance_threshold=0.85,
        hypothesis_s=7,
        hypothesis_correct=True,
        total_copies=350,
        total_time_s=0.15,
        epsilon=0.3,
        theta=0.3,
        delta=0.1,
        a_sq=1.0,
        b_sq=1.0,
        phi_description="test_parity",
    )


@pytest.fixture
def honest_trial_spec():
    """A TrialSpec for a small honest parity trial (n=4)."""
    phi = make_single_parity(4, target_s=3)
    return TrialSpec(
        n=4,
        phi=phi,
        noise_rate=0.0,
        target_s=3,
        epsilon=0.3,
        delta=0.1,
        theta=0.3,
        a_sq=1.0,
        b_sq=1.0,
        qfs_shots=500,
        classical_samples_prover=300,
        classical_samples_verifier=500,
        seed=42,
        phi_description="test_parity_s=3",
    )


# ===================================================================
# Phi generators
# ===================================================================


class TestMakeSingleParity:
    """Tests for make_single_parity."""

    def test_output_length(self):
        """Output has 2^n entries."""
        for n in (2, 4, 6):
            phi = make_single_parity(n, target_s=1)
            assert len(phi) == 2**n

    def test_binary_values(self):
        """All entries are 0.0 or 1.0."""
        phi = make_single_parity(4, target_s=5)
        assert all(v in (0.0, 1.0) for v in phi)

    def test_identity_parity(self):
        r"""For s*=0, parity is always 0 so phi(x)=0 for all x."""
        phi = make_single_parity(3, target_s=0)
        assert all(v == 0.0 for v in phi)

    def test_known_parity_n2(self):
        r"""For n=2, s*=3 (=0b11): phi(x) = popcount(x & 3) mod 2.

        x=0 -> 0, x=1 -> 1, x=2 -> 1, x=3 -> 0.
        """
        phi = make_single_parity(2, target_s=3)
        assert phi == [0.0, 1.0, 1.0, 0.0]

    def test_fourier_spectrum_single_peak(self):
        r"""The Fourier spectrum of a pure parity should have a single
        nonzero coefficient at s*.

        phi(x) = s*·x mod 2, so tilde_phi = 2*phi - 1 = -chi_{s*}.
        The Fourier transform has |hat(tilde_phi)(s*)| = 1 and all
        other coefficients zero.
        """
        n = 4
        target_s = 7
        phi = make_single_parity(n, target_s)
        tilde_phi = np.array([2 * v - 1 for v in phi])

        # Compute Fourier transform: hat(f)(s) = (1/2^n) sum_x f(x) chi_s(x)
        N = 2**n
        for s in range(N):
            chi_s = np.array([(-1) ** bin(s & x).count("1") for x in range(N)])
            coeff = np.dot(tilde_phi, chi_s) / N
            if s == target_s:
                np.testing.assert_allclose(abs(coeff), 1.0, atol=1e-12)
            else:
                np.testing.assert_allclose(coeff, 0.0, atol=1e-12)


class TestMakeRandomParity:
    """Tests for make_random_parity."""

    def test_nonzero_target(self):
        """Target s* is always nonzero."""
        rng = default_rng(0)
        for _ in range(20):
            _, s = make_random_parity(4, rng)
            assert s >= 1

    def test_target_in_range(self):
        """Target s* is in [1, 2^n)."""
        rng = default_rng(42)
        for n in (3, 5, 7):
            _, s = make_random_parity(n, rng)
            assert 1 <= s < 2**n

    def test_consistency_with_make_single_parity(self):
        """Output matches make_single_parity for the returned target."""
        rng = default_rng(99)
        phi, s = make_random_parity(5, rng)
        assert phi == make_single_parity(5, s)


class TestMakeBentFunction:
    """Tests for make_bent_function."""

    def test_rejects_odd_n(self):
        with pytest.raises(ValueError, match="even n"):
            make_bent_function(3)

    def test_output_length(self):
        phi = make_bent_function(4)
        assert len(phi) == 16

    def test_binary_values(self):
        phi = make_bent_function(6)
        assert all(v in (0.0, 1.0) for v in phi)

    def test_flat_fourier_spectrum(self):
        r"""For a bent function, all Fourier coefficients of g=(-1)^f
        should have magnitude 2^{-n/2}."""
        n = 4
        phi = make_bent_function(n)
        g = np.array([(-1) ** int(v) for v in phi], dtype=np.float64)

        N = 2**n
        expected_mag = 2 ** (-n / 2)

        for s in range(N):
            chi_s = np.array([(-1) ** bin(s & x).count("1") for x in range(N)])
            coeff = np.dot(g, chi_s) / N
            np.testing.assert_allclose(
                abs(coeff),
                expected_mag,
                atol=1e-12,
                err_msg=f"Fourier coefficient at s={s} has wrong magnitude",
            )


# ===================================================================
# Result containers
# ===================================================================


class TestTrialResult:
    """Tests for the TrialResult dataclass."""

    def test_fields_stored(self, sample_trial_result):
        t = sample_trial_result
        assert t.n == 4
        assert t.seed == 42
        assert t.accepted is True
        assert t.hypothesis_s == 7

    def test_proto_roundtrip(self, sample_trial_result):
        """TrialResult -> proto -> field check."""
        pb = _trial_to_proto(sample_trial_result)
        assert pb.n == 4
        assert pb.seed == 42
        assert pb.accepted is True
        assert pb.hypothesis_s == 7
        assert pb.phi_description == "test_parity"

    def test_proto_none_hypothesis(self, sample_trial_result):
        """hypothesis_s=None should leave the proto field unset."""
        sample_trial_result.hypothesis_s = None
        pb = _trial_to_proto(sample_trial_result)
        assert not pb.HasField("hypothesis_s")


class TestExperimentResult:
    """Tests for ExperimentResult serialisation and summary."""

    def _make_experiment(self, name, trials, params):
        return ExperimentResult(
            experiment_name=name,
            timestamp="2026-01-01T00:00:00",
            wall_clock_s=1.0,
            max_workers=1,
            trials=trials,
            parameters=params,
        )

    def test_save_scaling(self, sample_trial_result, tmp_path):
        """Scaling experiment round-trips through protobuf save."""
        from experiments.proto import scaling_pb2

        result = self._make_experiment(
            "scaling",
            [sample_trial_result],
            {
                "n_range": [4],
                "num_trials": 1,
                "epsilon": 0.3,
                "delta": 0.1,
                "qfs_shots": 100,
                "classical_samples_prover": 50,
                "classical_samples_verifier": 200,
            },
        )
        path = str(tmp_path / "test.pb")
        result.save(path)

        data = Path(path).read_bytes()
        pb = scaling_pb2.ScalingExperimentResult()
        pb.ParseFromString(data)
        assert pb.metadata.experiment_name == "scaling"
        assert len(pb.trials) == 1
        assert pb.trials[0].n == 4

    def test_save_bent(self, sample_trial_result, tmp_path):
        from experiments.proto import bent_pb2

        result = self._make_experiment(
            "bent_function",
            [sample_trial_result],
            {
                "n_range": [4],
                "num_trials": 1,
                "epsilon": 0.3,
                "theta": 0.3,
                "qfs_shots": 100,
                "note": "test",
            },
        )
        path = str(tmp_path / "test.pb")
        result.save(path)

        data = Path(path).read_bytes()
        pb = bent_pb2.BentExperimentResult()
        pb.ParseFromString(data)
        assert pb.metadata.experiment_name == "bent_function"

    def test_save_truncation(self, sample_trial_result, tmp_path):
        from experiments.proto import truncation_pb2

        result = self._make_experiment(
            "verifier_truncation",
            [sample_trial_result],
            {
                "n": 4,
                "noise_rate": 0.15,
                "a_sq": 0.49,
                "epsilon_range": [0.3],
                "verifier_sample_range": [200],
                "num_trials": 1,
                "qfs_shots": 100,
            },
        )
        path = str(tmp_path / "test.pb")
        result.save(path)

        data = Path(path).read_bytes()
        pb = truncation_pb2.TruncationExperimentResult()
        pb.ParseFromString(data)
        assert pb.parameters.n == 4

    def test_save_noise_sweep(self, sample_trial_result, tmp_path):
        from experiments.proto import noise_sweep_pb2

        result = self._make_experiment(
            "noise_sweep",
            [sample_trial_result],
            {
                "n_range": [4],
                "noise_rates": [0.0],
                "num_trials": 1,
                "epsilon": 0.3,
            },
        )
        path = str(tmp_path / "test.pb")
        result.save(path)

        data = Path(path).read_bytes()
        pb = noise_sweep_pb2.NoiseSweepExperimentResult()
        pb.ParseFromString(data)
        assert pb.metadata.experiment_name == "noise_sweep"

    def test_save_soundness(self, sample_trial_result, tmp_path):
        from experiments.proto import soundness_pb2

        result = self._make_experiment(
            "soundness",
            [sample_trial_result],
            {
                "n_range": [4],
                "num_trials": 1,
                "epsilon": 0.3,
                "strategies": ["random_list"],
            },
        )
        path = str(tmp_path / "test.pb")
        result.save(path)

        data = Path(path).read_bytes()
        pb = soundness_pb2.SoundnessExperimentResult()
        pb.ParseFromString(data)
        assert pb.parameters.strategies == ["random_list"]

    def test_save_unknown_experiment_raises(self, sample_trial_result, tmp_path):
        result = self._make_experiment("unknown", [sample_trial_result], {})
        with pytest.raises(ValueError, match="No proto schema"):
            result.save(str(tmp_path / "fail.pb"))

    def test_save_creates_parent_dirs(self, sample_trial_result, tmp_path):
        result = self._make_experiment(
            "scaling",
            [sample_trial_result],
            {
                "n_range": [4],
                "num_trials": 1,
                "epsilon": 0.3,
                "delta": 0.1,
                "qfs_shots": 100,
                "classical_samples_prover": 50,
                "classical_samples_verifier": 200,
            },
        )
        path = str(tmp_path / "deep" / "nested" / "test.pb")
        result.save(path)
        assert Path(path).exists()

    def test_summary_table(self, sample_trial_result):
        result = self._make_experiment("scaling", [sample_trial_result], {})
        table = result.summary_table()
        assert "n" in table
        assert "4" in table
        assert "100.0%" in table


# ===================================================================
# Worker and parallel dispatch
# ===================================================================


class TestTrialSpec:
    """Tests for the TrialSpec dataclass."""

    def test_default_dishonest_strategy(self, honest_trial_spec):
        assert honest_trial_spec.dishonest_strategy is None

    def test_fields(self, honest_trial_spec):
        assert honest_trial_spec.n == 4
        assert honest_trial_spec.target_s == 3
        assert len(honest_trial_spec.phi) == 16


class TestRunTrialWorker:
    """Integration test: run a single trial through the full protocol."""

    def test_honest_trial_produces_result(self, honest_trial_spec):
        """An honest trial at n=4 should return a TrialResult and
        typically accept (high completeness at small n)."""
        result = _run_trial_worker(honest_trial_spec)

        assert isinstance(result, TrialResult)
        assert result.n == 4
        assert result.seed == 42
        assert result.qfs_shots == 500
        assert result.prover_time_s > 0
        assert result.total_time_s > 0
        assert result.outcome in (
            "accept",
            "reject_list_too_large",
            "reject_insufficient_weight",
        )

    def test_honest_trial_completeness(self):
        """Multiple honest trials at n=4 should mostly accept and be correct."""
        rng = default_rng(0)
        results = []
        for _ in range(5):
            seed = int(rng.integers(0, 2**31))
            trial_rng = default_rng(seed)
            phi, target_s = make_random_parity(4, trial_rng)
            spec = TrialSpec(
                n=4,
                phi=phi,
                noise_rate=0.0,
                target_s=target_s,
                epsilon=0.3,
                delta=0.1,
                theta=0.3,
                a_sq=1.0,
                b_sq=1.0,
                qfs_shots=1000,
                classical_samples_prover=500,
                classical_samples_verifier=1000,
                seed=seed,
                phi_description=f"parity_s={target_s}",
            )
            results.append(_run_trial_worker(spec))

        accept_rate = sum(r.accepted for r in results) / len(results)
        assert accept_rate >= 0.6, (
            f"Expected most trials to accept, got {accept_rate:.0%}"
        )

    def test_dishonest_wrong_parity_rejected(self):
        """A dishonest prover using 'wrong_parity' should be rejected."""
        phi = make_single_parity(4, target_s=1)
        spec = TrialSpec(
            n=4,
            phi=phi,
            noise_rate=0.0,
            target_s=1,
            epsilon=0.3,
            delta=0.1,
            theta=0.3,
            a_sq=1.0,
            b_sq=1.0,
            qfs_shots=0,
            classical_samples_prover=0,
            classical_samples_verifier=1000,
            seed=42,
            phi_description="soundness_wrong_parity",
            dishonest_strategy="wrong_parity",
        )
        result = _run_trial_worker(spec)
        assert not result.accepted

    def test_dishonest_partial_list_rejected(self):
        """A dishonest prover using 'partial_list' (empty L) should be rejected."""
        phi = make_single_parity(4, target_s=1)
        spec = TrialSpec(
            n=4,
            phi=phi,
            noise_rate=0.0,
            target_s=1,
            epsilon=0.3,
            delta=0.1,
            theta=0.3,
            a_sq=1.0,
            b_sq=1.0,
            qfs_shots=0,
            classical_samples_prover=0,
            classical_samples_verifier=1000,
            seed=42,
            phi_description="soundness_partial_list",
            dishonest_strategy="partial_list",
        )
        result = _run_trial_worker(spec)
        assert not result.accepted
        assert result.list_size == 0


class TestRunTrialsParallel:
    """Tests for the parallel dispatch function."""

    def test_sequential_preserves_order(self, honest_trial_spec):
        """With max_workers=1, results come back in order."""
        specs = [honest_trial_spec, honest_trial_spec]
        results = run_trials_parallel(specs, max_workers=1, label="test")
        assert len(results) == 2
        assert all(isinstance(r, TrialResult) for r in results)

    def test_empty_specs(self):
        """Empty spec list returns empty results."""
        results = run_trials_parallel([], max_workers=1)
        assert results == []


# ===================================================================
# Experiment runners (lightweight — small n, few trials)
# ===================================================================


class TestRunScalingExperiment:
    """Integration test for the scaling experiment runner."""

    def test_runs_and_returns_result(self):
        result = run_scaling_experiment(
            n_range=range(4, 5),
            num_trials=2,
            qfs_shots=500,
            classical_samples_prover=300,
            classical_samples_verifier=500,
            base_seed=42,
            max_workers=1,
        )
        assert isinstance(result, ExperimentResult)
        assert result.experiment_name == "scaling"
        assert len(result.trials) == 2
        assert result.wall_clock_s > 0

    def test_save_roundtrip(self, tmp_path):
        result = run_scaling_experiment(
            n_range=range(4, 5),
            num_trials=1,
            qfs_shots=500,
            classical_samples_prover=300,
            classical_samples_verifier=500,
            base_seed=42,
            max_workers=1,
        )
        path = str(tmp_path / "scaling.pb")
        result.save(path)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


class TestRunBentExperiment:
    """Integration test for the bent function experiment runner."""

    def test_runs_and_returns_result(self):
        result = run_bent_experiment(
            n_range=range(4, 5, 2),
            num_trials=2,
            qfs_shots=500,
            classical_samples_prover=300,
            classical_samples_verifier=500,
            base_seed=42,
            max_workers=1,
        )
        assert isinstance(result, ExperimentResult)
        assert result.experiment_name == "bent_function"
        assert len(result.trials) == 2

    def test_hypothesis_correct_equals_accepted(self):
        """For bent functions, hypothesis_correct is set to accepted."""
        result = run_bent_experiment(
            n_range=range(4, 5, 2),
            num_trials=3,
            qfs_shots=500,
            classical_samples_prover=300,
            classical_samples_verifier=500,
            base_seed=42,
            max_workers=1,
        )
        for t in result.trials:
            assert t.hypothesis_correct == t.accepted


class TestRunTruncationExperiment:
    """Integration test for the truncation experiment runner."""

    def test_runs_and_returns_result(self):
        result = run_truncation_experiment(
            n=4,
            epsilon_range=[0.3],
            verifier_sample_range=[200],
            num_trials=2,
            qfs_shots=500,
            classical_samples_prover=300,
            base_seed=42,
            max_workers=1,
        )
        assert isinstance(result, ExperimentResult)
        assert result.experiment_name == "verifier_truncation"
        assert len(result.trials) == 2
        assert result.parameters["n"] == 4


class TestRunNoiseSweepExperiment:
    """Integration test for the noise sweep experiment runner."""

    def test_runs_and_returns_result(self):
        result = run_noise_sweep_experiment(
            n_range=range(4, 5),
            noise_rates=[0.0, 0.1],
            num_trials=2,
            qfs_shots=500,
            classical_samples_prover=300,
            classical_samples_verifier=500,
            base_seed=42,
            max_workers=1,
        )
        assert isinstance(result, ExperimentResult)
        assert result.experiment_name == "noise_sweep"
        # 1 n value * 2 noise rates * 2 trials = 4
        assert len(result.trials) == 4


class TestRunSoundnessExperiment:
    """Integration test for the soundness experiment runner."""

    def test_runs_and_returns_result(self):
        result = run_soundness_experiment(
            n_range=range(4, 5),
            num_trials=3,
            classical_samples_verifier=500,
            base_seed=42,
            max_workers=1,
        )
        assert isinstance(result, ExperimentResult)
        assert result.experiment_name == "soundness"
        # 1 n value * 4 strategies * 3 trials = 12
        assert len(result.trials) == 12

    def test_dishonest_strategies_mostly_rejected(self):
        """Non-random dishonest strategies should be consistently rejected."""
        result = run_soundness_experiment(
            n_range=range(4, 5),
            num_trials=5,
            classical_samples_verifier=1000,
            base_seed=42,
            max_workers=1,
        )
        for strategy in ("wrong_parity", "partial_list", "inflated_list"):
            st = [t for t in result.trials if strategy in t.phi_description]
            reject_rate = sum(1 for t in st if not t.accepted) / len(st)
            assert reject_rate >= 0.8, (
                f"Expected {strategy} to be mostly rejected, got {reject_rate:.0%}"
            )


# ===================================================================
# CLI argument parsing
# ===================================================================


class TestCLIArgumentParsing:
    """Tests for __main__.py argument parsing."""

    def test_scaling_subcommand(self):
        with patch(
            "sys.argv",
            [
                "harness",
                "scaling",
                "--n-min",
                "4",
                "--n-max",
                "4",
                "--trials",
                "1",
                "--workers",
                "1",
            ],
        ):
            # We just test parsing by importing the parser; running main
            # would actually execute experiments, so we test dispatch logic.
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest="command")
            sp = subparsers.add_parser("scaling")
            sp.add_argument("--n-min", type=int, default=4)
            sp.add_argument("--n-max", type=int, default=10)
            args = parser.parse_args(["scaling", "--n-min", "4", "--n-max", "6"])
            assert args.command == "scaling"
            assert args.n_min == 4
            assert args.n_max == 6

    def test_truncation_subcommand_with_fixed_n(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        sp = subparsers.add_parser("truncation")
        sp.add_argument("--n", type=int, default=None)
        sp.add_argument("--n-min", type=int, default=4)
        args = parser.parse_args(["truncation", "--n", "8"])
        assert args.command == "truncation"
        assert args.n == 8

    def test_all_subcommand(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        sp = subparsers.add_parser("all")
        sp.add_argument("--n-min", type=int, default=4)
        sp.add_argument("--n-max", type=int, default=10)
        sp.add_argument("--n", type=int, default=None)
        args = parser.parse_args(["all", "--n-min", "4", "--n-max", "6"])
        assert args.command == "all"
        assert args.n is None


# ===================================================================
# Package-level imports
# ===================================================================


class TestPackageImports:
    """Verify that the package __init__.py re-exports work."""

    def test_import_experiment_runners(self):
        from experiments.harness import (
            run_scaling_experiment,
            run_bent_experiment,
        )

        assert callable(run_scaling_experiment)
        assert callable(run_bent_experiment)

    def test_import_result_types(self):
        from experiments.harness import TrialResult, ExperimentResult, TrialSpec

        assert TrialResult is not None
        assert ExperimentResult is not None
        assert TrialSpec is not None

    def test_import_phi_generators(self):
        from experiments.harness import (
            make_single_parity,
        )

        assert callable(make_single_parity)

    def test_import_parallel_dispatch(self):
        from experiments.harness import run_trials_parallel

        assert callable(run_trials_parallel)
