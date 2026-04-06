"""Tests for SLURM Job Array sharding and merge utilities.

Covers shard_specs partitioning, shard_output_path naming,
merge_shard_files round-trip, and end-to-end determinism
(sharded runs produce identical results to non-sharded runs).

Tests authored by Claude Opus 4.6 in full.
"""

import json
from pathlib import Path

import pytest

from experiments.decode import decode
from experiments.harness.results import ExperimentResult, TrialResult
from experiments.harness.sharding import (
    merge_shard_files,
    shard_output_path,
    shard_specs,
)
from experiments.harness.worker import TrialSpec, run_trials_parallel
from experiments.harness.phi import make_single_parity
from experiments.harness.scaling import run_scaling_experiment


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


def _save_scaling(trials, tmp_path, filename):
    """Helper: save a scaling ExperimentResult and return the path."""
    result = ExperimentResult(
        experiment_name="scaling",
        timestamp="2026-01-01T00:00:00",
        wall_clock_s=1.0,
        max_workers=1,
        trials=trials,
        parameters={
            "n_range": [4],
            "num_trials": len(trials),
            "epsilon": 0.3,
            "delta": 0.1,
            "qfs_shots": 100,
            "classical_samples_prover": 50,
            "classical_samples_verifier": 200,
        },
    )
    path = tmp_path / filename
    result.save(str(path))
    return path


# ===================================================================
# shard_specs
# ===================================================================


class TestShardSpecs:
    """Tests for the shard_specs partitioning function."""

    def test_even_split(self):
        """10 items into 5 shards: 2 each."""
        specs = list(range(10))
        for i in range(5):
            assert len(shard_specs(specs, i, 5)) == 2

    def test_uneven_split(self):
        """10 items into 3 shards: 4, 3, 3."""
        specs = list(range(10))
        s0 = shard_specs(specs, 0, 3)
        s1 = shard_specs(specs, 1, 3)
        s2 = shard_specs(specs, 2, 3)
        assert len(s0) == 4
        assert len(s1) == 3
        assert len(s2) == 3

    def test_reconstruction(self):
        """Concatenating all shards reproduces the original set of items."""
        specs = list(range(17))
        num_shards = 4
        reconstructed = []
        for i in range(num_shards):
            reconstructed.extend(shard_specs(specs, i, num_shards))
        assert sorted(reconstructed) == specs

    def test_no_overlap(self):
        """Shards should have disjoint elements."""
        specs = list(range(20))
        num_shards = 7
        seen = set()
        for i in range(num_shards):
            shard = shard_specs(specs, i, num_shards)
            for item in shard:
                assert item not in seen, f"Duplicate item {item} in shard {i}"
                seen.add(item)
        assert seen == set(specs)

    def test_single_shard(self):
        """One shard returns the full list."""
        specs = list(range(5))
        assert shard_specs(specs, 0, 1) == specs

    def test_empty_input(self):
        """Empty spec list returns empty for any shard."""
        assert shard_specs([], 0, 3) == []
        assert shard_specs([], 2, 3) == []

    def test_more_shards_than_specs(self):
        """Extra shards get empty lists."""
        specs = list(range(2))
        assert shard_specs(specs, 0, 5) == [0]
        assert shard_specs(specs, 1, 5) == [1]
        assert shard_specs(specs, 2, 5) == []
        assert shard_specs(specs, 3, 5) == []
        assert shard_specs(specs, 4, 5) == []

    def test_round_robin_slicing(self):
        """Shards are interleaved (round-robin) for balanced workloads."""
        specs = list(range(12))
        s0 = shard_specs(specs, 0, 3)
        s1 = shard_specs(specs, 1, 3)
        s2 = shard_specs(specs, 2, 3)
        assert s0 == [0, 3, 6, 9]
        assert s1 == [1, 4, 7, 10]
        assert s2 == [2, 5, 8, 11]

    def test_invalid_shard_index_raises(self):
        with pytest.raises(ValueError, match="shard_index"):
            shard_specs(list(range(5)), 3, 3)

    def test_negative_shard_index_raises(self):
        with pytest.raises(ValueError, match="shard_index"):
            shard_specs(list(range(5)), -1, 3)

    def test_invalid_num_shards_raises(self):
        with pytest.raises(ValueError, match="num_shards"):
            shard_specs(list(range(5)), 0, 0)


# ===================================================================
# shard_output_path
# ===================================================================


class TestShardOutputPath:
    """Tests for shard filename generation."""

    def test_basic(self):
        result = shard_output_path("results/scaling_4_10_20.pb", 3, 8)
        assert result == "results/scaling_4_10_20_shard4of8.pb"

    def test_preserves_directory(self):
        result = shard_output_path("/tmp/deep/nested/foo.pb", 0, 2)
        assert result == "/tmp/deep/nested/foo_shard1of2.pb"

    def test_preserves_extension(self):
        result = shard_output_path("test.pb", 1, 4)
        assert result.endswith(".pb")

    def test_single_shard(self):
        result = shard_output_path("test.pb", 0, 1)
        assert result == "test_shard1of1.pb"


# ===================================================================
# merge_shard_files
# ===================================================================


class TestMergeShardFiles:
    """Tests for merging sharded .pb results."""

    def test_merge_two_shards(self, sample_trial, tmp_path):
        """Merging two shard files concatenates trials."""
        t1 = sample_trial
        t2 = TrialResult(**{**t1.__dict__, "seed": 99, "n": 5})

        p1 = _save_scaling([t1], tmp_path, "scaling_shard0of2.pb")
        p2 = _save_scaling([t2], tmp_path, "scaling_shard1of2.pb")

        out = tmp_path / "scaling_merged.pb"
        merge_shard_files([p1, p2], out)

        assert out.exists()
        parsed = json.loads(decode(out))
        assert parsed["metadata"]["numTrials"] == 2
        seeds = [int(t["seed"]) for t in parsed["trials"]]
        assert 42 in seeds
        assert 99 in seeds

    def test_merge_preserves_parameters(self, sample_trial, tmp_path):
        """Merged result keeps the parameters from the first shard."""
        p1 = _save_scaling([sample_trial], tmp_path, "scaling_shard0of2.pb")
        p2 = _save_scaling([sample_trial], tmp_path, "scaling_shard1of2.pb")

        out = tmp_path / "scaling_merged.pb"
        merge_shard_files([p1, p2], out)

        parsed = json.loads(decode(out))
        assert parsed["parameters"]["epsilon"] == 0.3
        assert parsed["parameters"]["nRange"] == [4]

    def test_merge_sums_wall_clock(self, sample_trial, tmp_path):
        """Wall-clock time is summed across shards."""
        p1 = _save_scaling([sample_trial], tmp_path, "scaling_shard0of2.pb")
        p2 = _save_scaling([sample_trial], tmp_path, "scaling_shard1of2.pb")

        out = tmp_path / "scaling_merged.pb"
        merge_shard_files([p1, p2], out)

        parsed = json.loads(decode(out))
        # Each shard has wall_clock_s=1.0, so merged should be 2.0
        assert parsed["metadata"]["wallClockS"] == pytest.approx(2.0)

    def test_merge_creates_parent_dirs(self, sample_trial, tmp_path):
        p1 = _save_scaling([sample_trial], tmp_path, "scaling_shard0of1.pb")
        out = tmp_path / "deep" / "nested" / "scaling_merged.pb"
        merge_shard_files([p1], out)
        assert out.exists()

    def test_merge_empty_list_exits(self, tmp_path):
        """Merging zero files should exit with an error."""
        with pytest.raises(SystemExit):
            merge_shard_files([], tmp_path / "out.pb")

    def test_merge_mixed_types_exits(self, sample_trial, tmp_path):
        """Merging different experiment types should exit with an error."""
        scaling_path = _save_scaling([sample_trial], tmp_path, "scaling_shard0of2.pb")
        # Save a soundness result under a soundness filename
        soundness_result = ExperimentResult(
            experiment_name="soundness",
            timestamp="2026-01-01T00:00:00",
            wall_clock_s=1.0,
            max_workers=1,
            trials=[sample_trial],
            parameters={
                "n_range": [4],
                "num_trials": 1,
                "epsilon": 0.3,
                "strategies": ["random_list"],
            },
        )
        soundness_path = tmp_path / "soundness_shard1of2.pb"
        soundness_result.save(str(soundness_path))

        with pytest.raises(SystemExit):
            merge_shard_files([scaling_path, soundness_path], tmp_path / "out.pb")

    def test_merge_skips_missing_with_warning(self, sample_trial, tmp_path, capsys):
        """Missing shard files are skipped with a warning."""
        p1 = _save_scaling([sample_trial], tmp_path, "scaling_shard0of2.pb")
        missing = tmp_path / "scaling_shard1of2.pb"  # does not exist

        out = tmp_path / "scaling_merged.pb"
        merge_shard_files([p1, missing], out)

        assert out.exists()
        parsed = json.loads(decode(out))
        assert parsed["metadata"]["numTrials"] == 1
        assert "not found" in capsys.readouterr().err


# ===================================================================
# run_trials_parallel with sharding
# ===================================================================


class TestRunTrialsParallelSharding:
    """Tests for shard_index/num_shards params in run_trials_parallel."""

    def _make_specs(self, count):
        """Build count trivial TrialSpecs."""
        phi = make_single_parity(4, target_s=1)
        return [
            TrialSpec(
                n=4,
                phi=phi,
                noise_rate=0.0,
                target_s=1,
                epsilon=0.3,
                delta=0.1,
                theta=0.3,
                a_sq=1.0,
                b_sq=1.0,
                qfs_shots=200,
                classical_samples_prover=100,
                classical_samples_verifier=200,
                seed=i,
                phi_description=f"test_{i}",
            )
            for i in range(count)
        ]

    def test_shard_reduces_work(self):
        """With 6 specs and 3 shards, each shard runs only 2."""
        specs = self._make_specs(6)
        results = run_trials_parallel(
            specs,
            max_workers=1,
            label="test",
            shard_index=0,
            num_shards=3,
        )
        assert len(results) == 2

    def test_none_shards_runs_all(self):
        """When shard params are None, all specs are run."""
        specs = self._make_specs(3)
        results = run_trials_parallel(
            specs,
            max_workers=1,
            label="test",
            shard_index=None,
            num_shards=None,
        )
        assert len(results) == 3


# ===================================================================
# End-to-end determinism
# ===================================================================


class TestShardedDeterminism:
    """Sharded execution produces identical results to non-sharded."""

    def test_scaling_shards_match_single_run(self, tmp_path):
        """Run scaling with 3 shards, merge, compare seeds to a non-sharded run."""
        kwargs = dict(
            n_range=range(4, 6),
            num_trials=2,
            qfs_shots=200,
            classical_samples_prover=100,
            classical_samples_verifier=200,
            base_seed=42,
            max_workers=1,
        )

        # Non-sharded baseline
        baseline = run_scaling_experiment(**kwargs)
        baseline.save(str(tmp_path / "scaling_baseline.pb"))

        # Sharded
        shard_paths = []
        for i in range(3):
            r = run_scaling_experiment(**kwargs, shard_index=i, num_shards=3)
            p = tmp_path / f"scaling_shard{i + 1}of3.pb"
            r.save(str(p))
            shard_paths.append(p)

        merged_path = tmp_path / "scaling_merged.pb"
        merge_shard_files(shard_paths, merged_path)

        # Compare
        baseline_json = json.loads(decode(tmp_path / "scaling_baseline.pb"))
        merged_json = json.loads(decode(merged_path))

        baseline_seeds = sorted(t["seed"] for t in baseline_json["trials"])
        merged_seeds = sorted(t["seed"] for t in merged_json["trials"])
        assert baseline_seeds == merged_seeds

        baseline_by_seed = {
            t["seed"]: t["hypothesisCorrect"] for t in baseline_json["trials"]
        }
        merged_by_seed = {
            t["seed"]: t["hypothesisCorrect"] for t in merged_json["trials"]
        }
        assert baseline_by_seed == merged_by_seed


# ===================================================================
# Package-level imports
# ===================================================================


class TestShardingImports:
    """Verify that sharding symbols are exported from the package."""

    def test_import_shard_specs(self):
        from experiments.harness import shard_specs

        assert callable(shard_specs)

    def test_import_merge_shard_files(self):
        from experiments.harness import merge_shard_files

        assert callable(merge_shard_files)
