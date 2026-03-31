"""Decode experiment protobuf files to JSON.

Usage::

    python -m experiments.decode results/scaling_4_10_20.pb
    python -m experiments.decode results/scaling_4_10_20.pb -o results/scaling_4_10_20.json
    python -m experiments.decode results/*.pb
"""

import argparse
import sys
from pathlib import Path

from google.protobuf.json_format import MessageToJson

from experiments.proto import (
    scaling_pb2,
    bent_pb2,
    truncation_pb2,
    noise_sweep_pb2,
    soundness_pb2,
)

# Maps experiment_name (from ExperimentMetadata) to its top-level proto class.
_RESULT_TYPES = {
    "scaling": scaling_pb2.ScalingExperimentResult,
    "bent_function": bent_pb2.BentExperimentResult,
    "verifier_truncation": truncation_pb2.TruncationExperimentResult,
    "noise_sweep": noise_sweep_pb2.NoiseSweepExperimentResult,
    "soundness": soundness_pb2.SoundnessExperimentResult,
}

# Filename prefixes to experiment names, for files whose experiment_name
# can't be read until we know which message type to try.
_PREFIX_MAP = {
    "scaling": "scaling",
    "bent": "bent_function",
    "truncation": "verifier_truncation",
    "noise_sweep": "noise_sweep",
    "soundness": "soundness",
}


def _guess_experiment(path: Path) -> str:
    """Guess experiment name from the filename prefix."""
    stem = path.stem.lower()
    for prefix, name in _PREFIX_MAP.items():
        if stem.startswith(prefix):
            return name
    raise ValueError(
        f"Cannot determine experiment type from filename: {path.name}. "
        f"Expected prefix in: {', '.join(_PREFIX_MAP)}"
    )


def decode(path: Path) -> str:
    """Read a protobuf file and return its JSON representation."""
    experiment_name = _guess_experiment(path)
    cls = _RESULT_TYPES[experiment_name]
    msg = cls()
    msg.ParseFromString(path.read_bytes())
    return MessageToJson(msg, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Decode experiment .pb files to JSON",
    )
    parser.add_argument("files", nargs="+", type=Path, help="Protobuf file(s) to decode")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON path (only valid with a single input file; "
             "omit to print to stdout)",
    )
    args = parser.parse_args()

    if args.output and len(args.files) > 1:
        print("Error: -o/--output can only be used with a single input file",
              file=sys.stderr)
        sys.exit(1)

    for pb_path in args.files:
        json_str = decode(pb_path)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json_str)
            print(f"  Wrote {args.output}")
        else:
            if len(args.files) > 1:
                print(f"--- {pb_path} ---")
            print(json_str)


if __name__ == "__main__":
    main()
