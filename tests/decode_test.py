"""Tests for experiments.decode — protobuf-to-JSON decoder."""

from pathlib import Path

import pytest

from experiments.decode import _guess_experiment, decode, main
from experiments.harness.results import ExperimentResult, TrialResult


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def sample_trial():
    """A minimal TrialResult for building test .pb files."""
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


def _save_experiment(name, trial, params, tmp_path, filename):
    """Helper: save an ExperimentResult to a .pb file and return the path."""
    result = ExperimentResult(
        experiment_name=name,
        timestamp="2026-01-01T00:00:00",
        wall_clock_s=1.0,
        max_workers=1,
        trials=[trial],
        parameters=params,
    )
    path = tmp_path / filename
    result.save(str(path))
    return path


@pytest.fixture
def scaling_pb(sample_trial, tmp_path):
    return _save_experiment(
        "scaling",
        sample_trial,
        {
            "n_range": [4],
            "num_trials": 1,
            "epsilon": 0.3,
            "delta": 0.1,
            "qfs_shots": 100,
            "classical_samples_prover": 50,
            "classical_samples_verifier": 200,
        },
        tmp_path,
        "scaling_test.pb",
    )


@pytest.fixture
def bent_pb(sample_trial, tmp_path):
    return _save_experiment(
        "bent_function",
        sample_trial,
        {
            "n_range": [4],
            "num_trials": 1,
            "epsilon": 0.3,
            "theta": 0.3,
            "qfs_shots": 100,
            "note": "test",
        },
        tmp_path,
        "bent_test.pb",
    )


@pytest.fixture
def truncation_pb(sample_trial, tmp_path):
    return _save_experiment(
        "verifier_truncation",
        sample_trial,
        {
            "n": 4,
            "noise_rate": 0.15,
            "a_sq": 0.49,
            "epsilon_range": [0.3],
            "verifier_sample_range": [200],
            "num_trials": 1,
            "qfs_shots": 100,
        },
        tmp_path,
        "truncation_test.pb",
    )


@pytest.fixture
def noise_sweep_pb(sample_trial, tmp_path):
    return _save_experiment(
        "noise_sweep",
        sample_trial,
        {
            "n_range": [4],
            "noise_rates": [0.0],
            "num_trials": 1,
            "epsilon": 0.3,
        },
        tmp_path,
        "noise_sweep_test.pb",
    )


@pytest.fixture
def soundness_pb(sample_trial, tmp_path):
    return _save_experiment(
        "soundness",
        sample_trial,
        {
            "n_range": [4],
            "num_trials": 1,
            "epsilon": 0.3,
            "strategies": ["random_list"],
        },
        tmp_path,
        "soundness_test.pb",
    )


# ===================================================================
# _guess_experiment
# ===================================================================


class TestGuessExperiment:
    """Tests for filename-prefix detection."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("scaling_4_10_20.pb", "scaling"),
            ("bent_4_16_24.pb", "bent_function"),
            ("truncation_6_6_24.pb", "verifier_truncation"),
            ("noise_sweep_4_13_24.pb", "noise_sweep"),
            ("soundness_4_16_50.pb", "soundness"),
        ],
    )
    def test_valid_prefixes(self, filename, expected):
        assert _guess_experiment(Path(filename)) == expected

    def test_case_insensitive(self):
        assert _guess_experiment(Path("SCALING_test.pb")) == "scaling"

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Cannot determine experiment type"):
            _guess_experiment(Path("unknown_test.pb"))


# ===================================================================
# decode
# ===================================================================


class TestDecode:
    """Tests for the decode function (pb -> JSON string)."""

    def test_scaling_roundtrip(self, scaling_pb):
        json_str = decode(scaling_pb)
        assert '"experimentName": "scaling"' in json_str

    def test_bent_roundtrip(self, bent_pb):
        json_str = decode(bent_pb)
        assert '"experimentName": "bent_function"' in json_str

    def test_truncation_roundtrip(self, truncation_pb):
        json_str = decode(truncation_pb)
        assert '"n": 4' in json_str

    def test_noise_sweep_roundtrip(self, noise_sweep_pb):
        json_str = decode(noise_sweep_pb)
        assert '"experimentName": "noise_sweep"' in json_str

    def test_soundness_roundtrip(self, soundness_pb):
        json_str = decode(soundness_pb)
        assert '"strategies"' in json_str

    def test_output_is_valid_json(self, scaling_pb):
        import json

        json_str = decode(scaling_pb)
        parsed = json.loads(json_str)
        assert "metadata" in parsed
        assert "trials" in parsed

    def test_trial_fields_present(self, scaling_pb):
        import json

        parsed = json.loads(decode(scaling_pb))
        trial = parsed["trials"][0]
        assert trial["n"] == 4
        assert int(trial["seed"]) == 42
        assert trial["accepted"] is True


# ===================================================================
# CLI (main)
# ===================================================================


class TestCLI:
    """Tests for the main() CLI entry point."""

    def test_single_file_to_stdout(self, scaling_pb, capsys, monkeypatch):
        monkeypatch.setattr("sys.argv", ["decode", str(scaling_pb)])
        main()
        out = capsys.readouterr().out
        assert '"experimentName": "scaling"' in out

    def test_multiple_files_to_stdout(self, scaling_pb, bent_pb, capsys, monkeypatch):
        monkeypatch.setattr(
            "sys.argv", ["decode", str(scaling_pb), str(bent_pb)]
        )
        main()
        out = capsys.readouterr().out
        assert f"--- {scaling_pb} ---" in out
        assert f"--- {bent_pb} ---" in out

    def test_single_file_with_output(self, scaling_pb, tmp_path, monkeypatch):
        out_path = tmp_path / "out.json"
        monkeypatch.setattr(
            "sys.argv", ["decode", str(scaling_pb), "-o", str(out_path)]
        )
        main()
        assert out_path.exists()
        import json

        parsed = json.loads(out_path.read_text())
        assert "metadata" in parsed

    def test_output_creates_parent_dirs(self, scaling_pb, tmp_path, monkeypatch):
        out_path = tmp_path / "nested" / "dir" / "out.json"
        monkeypatch.setattr(
            "sys.argv", ["decode", str(scaling_pb), "-o", str(out_path)]
        )
        main()
        assert out_path.exists()

    def test_output_with_multiple_files_exits(
        self, scaling_pb, bent_pb, tmp_path, monkeypatch
    ):
        out_path = tmp_path / "out.json"
        monkeypatch.setattr(
            "sys.argv",
            ["decode", str(scaling_pb), str(bent_pb), "-o", str(out_path)],
        )
        with pytest.raises(SystemExit, match="1"):
            main()
