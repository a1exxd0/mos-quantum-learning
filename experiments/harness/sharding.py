"""SLURM Job Array sharding and shard-merge utilities.

Enables distributing experiment trials across multiple SLURM array
tasks.  Each task deterministically regenerates the full spec list,
takes its contiguous slice via :func:`shard_specs`, and saves a
partial ``.pb`` file.  After all tasks complete, :func:`merge_shard_files`
combines the shards into a single result file.
"""

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.decode import _guess_experiment, _RESULT_TYPES

if TYPE_CHECKING:
    from experiments.harness.worker import TrialSpec


def shard_specs(
    specs: "list[TrialSpec]",
    shard_index: int,
    num_shards: int,
) -> "list[TrialSpec]":
    """Return the contiguous slice of *specs* assigned to this shard.

    Uses ``divmod`` chunking so the first ``remainder`` shards each get
    one extra spec, ensuring no spec is missed or duplicated.

    Parameters
    ----------
    specs : list[TrialSpec]
        The full, deterministically generated spec list.
    shard_index : int
        0-based index of this shard.
    num_shards : int
        Total number of shards.

    Returns
    -------
    list[TrialSpec]
        The slice of specs for this shard (may be empty if
        ``num_shards > len(specs)``).
    """
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )
    # Round-robin assignment: shard k gets specs k, k+K, k+2K, ...
    # This interleaves specs across shards, balancing work when specs
    # are ordered by cost (e.g. ascending n).
    return specs[shard_index::num_shards]


def shard_output_path(base_path: str, shard_index: int, num_shards: int) -> str:
    """Append a shard suffix to a ``.pb`` output path.

    ``"results/scaling_4_10_20.pb"`` becomes
    ``"results/scaling_4_10_20_shard3of8.pb"`` for shard 3 of 8.
    """
    p = Path(base_path)
    return str(p.with_stem(f"{p.stem}_shard{shard_index}of{num_shards}"))


def merge_shard_files(shard_paths: list[Path], output_path: Path) -> None:
    """Merge sharded ``.pb`` experiment results into a single file.

    Reads each shard, concatenates all trial results, aggregates
    metadata (summed wall-clock time, total trial count), and writes
    the merged protobuf to *output_path*.

    Parameters
    ----------
    shard_paths : list[Path]
        Paths to shard ``.pb`` files.  All must be from the same
        experiment type.
    output_path : Path
        Destination for the merged ``.pb`` file.
    """
    if not shard_paths:
        print("Error: no shard files provided", file=sys.stderr)
        sys.exit(1)

    # Parse first shard to determine experiment type
    experiment_name = _guess_experiment(shard_paths[0])
    cls = _RESULT_TYPES[experiment_name]

    merged_trials = []
    total_wall_clock = 0.0
    max_workers = 0
    parameters_msg = None

    for path in shard_paths:
        if not path.exists():
            print(f"Warning: shard file not found, skipping: {path}", file=sys.stderr)
            continue

        shard_name = _guess_experiment(path)
        if shard_name != experiment_name:
            print(
                f"Error: mixed experiment types: {experiment_name} vs {shard_name} "
                f"(file: {path})",
                file=sys.stderr,
            )
            sys.exit(1)

        msg = cls()
        msg.ParseFromString(path.read_bytes())
        merged_trials.extend(msg.trials)
        total_wall_clock += msg.metadata.wall_clock_s
        max_workers = max(max_workers, msg.metadata.max_workers)
        if parameters_msg is None:
            parameters_msg = msg.parameters

    # Build merged message
    from experiments.proto import common_pb2

    merged_metadata = common_pb2.ExperimentMetadata(
        experiment_name=experiment_name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        wall_clock_s=total_wall_clock,
        max_workers=max_workers,
        num_trials=len(merged_trials),
    )

    merged_msg = cls(
        metadata=merged_metadata,
        parameters=parameters_msg,
        trials=merged_trials,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(merged_msg.SerializeToString())

    print(
        f"Merged {len(shard_paths)} shards -> {output_path} "
        f"({len(merged_trials)} trials, {total_wall_clock:.1f}s total wall-clock)"
    )
