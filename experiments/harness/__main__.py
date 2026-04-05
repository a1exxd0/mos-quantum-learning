"""CLI entry point for the experiment harness.

Usage::

    python -m experiments.harness scaling --n-min 4 --n-max 12 --workers 8
    python -m experiments.harness all --n-min 4 --n-max 12 --workers 4

SLURM distributed execution::

    python -m experiments.harness scaling --n-min 4 --n-max 16 --workers 32 \\
        --shard-index $SLURM_ARRAY_TASK_ID --num-shards 8

    python -m experiments.harness merge results/scaling_4_16_20_shard*.pb \\
        -o results/scaling_4_16_20.pb
"""

import time
import argparse
from pathlib import Path

from experiments.harness.average_case import run_average_case_experiment
from experiments.harness.gate_noise import run_gate_noise_experiment
from experiments.harness.ab_regime import run_ab_regime_experiment
from experiments.harness.k_sparse import run_k_sparse_experiment
from experiments.harness.theta_sensitivity import run_theta_sensitivity_experiment
from experiments.harness.scaling import run_scaling_experiment
from experiments.harness.bent import run_bent_experiment
from experiments.harness.truncation import run_truncation_experiment
from experiments.harness.noise import run_noise_sweep_experiment
from experiments.harness.soundness import run_soundness_experiment
from experiments.harness.soundness_multi import run_soundness_multi_experiment


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by all subcommands."""
    parser.add_argument("--n-min", type=int, default=4, help="Minimum n for sweep experiments")
    parser.add_argument("--n-max", type=int, default=10, help="Maximum n for sweep experiments")
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
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="0-based shard index for SLURM Job Array distribution (requires --num-shards)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards for SLURM Job Array distribution (requires --shard-index)",
    )


def _output_path(output_dir: Path, base_name: str, args) -> str:
    """Build an output path, appending a shard suffix when sharding is active."""
    if args.num_shards is not None:
        from experiments.harness.sharding import shard_output_path

        return shard_output_path(
            str(output_dir / f"{base_name}.pb"), args.shard_index, args.num_shards
        )
    return str(output_dir / f"{base_name}.pb")


def _shard_kwargs(args) -> dict:
    """Extract shard keyword arguments from parsed CLI args."""
    return {"shard_index": args.shard_index, "num_shards": args.num_shards}


def _run_scaling(args):
    output_dir = Path(args.output_dir)
    r = run_scaling_experiment(
        n_range=range(args.n_min, args.n_max + 1),
        num_trials=args.trials,
        base_seed=args.seed,
        max_workers=args.workers,
        **_shard_kwargs(args),
    )
    r.save(_output_path(output_dir, f"scaling_{args.n_min}_{args.n_max}_{args.trials}", args))
    return [r]


def _run_bent(args):
    output_dir = Path(args.output_dir)
    bent_min = args.n_min if args.n_min % 2 == 0 else args.n_min + 1
    bent_max = args.n_max if args.n_max % 2 == 0 else args.n_max - 1
    r = run_bent_experiment(
        n_range=range(bent_min, bent_max + 1, 2),
        num_trials=args.trials,
        base_seed=args.seed,
        max_workers=args.workers,
        **_shard_kwargs(args),
    )
    r.save(_output_path(output_dir, f"bent_{bent_min}_{bent_max}_{args.trials}", args))
    return [r]


def _run_truncation(args):
    output_dir = Path(args.output_dir)
    fixed_n = args.n if args.n is not None else args.n_min
    r = run_truncation_experiment(
        n=fixed_n,
        num_trials=args.trials,
        base_seed=args.seed,
        max_workers=args.workers,
        **_shard_kwargs(args),
    )
    r.save(_output_path(output_dir, f"truncation_{fixed_n}_{fixed_n}_{args.trials}", args))
    return [r]


def _run_noise(args):
    output_dir = Path(args.output_dir)
    r = run_noise_sweep_experiment(
        n_range=range(args.n_min, args.n_max + 1),
        num_trials=args.trials,
        base_seed=args.seed,
        max_workers=args.workers,
        **_shard_kwargs(args),
    )
    r.save(_output_path(output_dir, f"noise_sweep_{args.n_min}_{args.n_max}_{args.trials}", args))
    return [r]


def _run_soundness(args):
    output_dir = Path(args.output_dir)
    soundness_trials = max(args.trials, 50)
    r = run_soundness_experiment(
        n_range=range(args.n_min, args.n_max + 1),
        num_trials=soundness_trials,
        base_seed=args.seed,
        max_workers=args.workers,
        **_shard_kwargs(args),
    )
    r.save(_output_path(output_dir, f"soundness_{args.n_min}_{args.n_max}_{soundness_trials}", args))
    return [r]


def _run_soundness_multi(args):
    output_dir = Path(args.output_dir)
    soundness_trials = max(args.trials, 50)
    r = run_soundness_multi_experiment(
        n_range=range(args.n_min, args.n_max + 1),
        num_trials=soundness_trials,
        base_seed=args.seed,
        max_workers=args.workers,
        **_shard_kwargs(args),
    )
    r.save(_output_path(output_dir, f"soundness_multi_{args.n_min}_{args.n_max}_{soundness_trials}", args))
    return [r]


def _run_average_case(args):
    output_dir = Path(args.output_dir)
    r = run_average_case_experiment(
        n_range=range(args.n_min, args.n_max + 1),
        num_trials=args.trials,
        base_seed=args.seed,
        max_workers=args.workers,
        **_shard_kwargs(args),
    )
    r.save(_output_path(output_dir, f"average_case_{args.n_min}_{args.n_max}_{args.trials}", args))
    return [r]


def _run_gate_noise(args):
    output_dir = Path(args.output_dir)
    r = run_gate_noise_experiment(
        n_range=range(args.n_min, args.n_max + 1),
        num_trials=args.trials,
        base_seed=args.seed,
        max_workers=args.workers,
        **_shard_kwargs(args),
    )
    r.save(_output_path(output_dir, f"gate_noise_{args.n_min}_{args.n_max}_{args.trials}", args))
    return [r]


def _run_k_sparse(args):
    output_dir = Path(args.output_dir)
    r = run_k_sparse_experiment(
        n_range=range(args.n_min, args.n_max + 1, 2),
        num_trials=args.trials,
        base_seed=args.seed,
        max_workers=args.workers,
        **_shard_kwargs(args),
    )
    r.save(_output_path(output_dir, f"k_sparse_{args.n_min}_{args.n_max}_{args.trials}", args))
    return [r]


def _run_theta_sensitivity(args):
    output_dir = Path(args.output_dir)
    r = run_theta_sensitivity_experiment(
        n_range=range(args.n_min, args.n_max + 1, 2),
        num_trials=args.trials,
        base_seed=args.seed,
        max_workers=args.workers,
        **_shard_kwargs(args),
    )
    r.save(_output_path(output_dir, f"theta_sensitivity_{args.n_min}_{args.n_max}_{args.trials}", args))
    return [r]


def _run_ab_regime(args):
    output_dir = Path(args.output_dir)
    r = run_ab_regime_experiment(
        n_range=range(args.n_min, args.n_max + 1),
        num_trials=args.trials,
        base_seed=args.seed,
        max_workers=args.workers,
        **_shard_kwargs(args),
    )
    r.save(_output_path(output_dir, f"ab_regime_{args.n_min}_{args.n_max}_{args.trials}", args))
    return [r]


def _run_merge(args):
    from experiments.harness.sharding import merge_shard_files

    merge_shard_files(args.shards, args.output)
    return []


def _run_all(args):
    experiments = []
    experiments.extend(_run_scaling(args))
    experiments.extend(_run_bent(args))
    experiments.extend(_run_truncation(args))
    experiments.extend(_run_noise(args))
    experiments.extend(_run_soundness(args))
    experiments.extend(_run_soundness_multi(args))
    experiments.extend(_run_average_case(args))
    experiments.extend(_run_gate_noise(args))
    experiments.extend(_run_k_sparse(args))
    experiments.extend(_run_theta_sensitivity(args))
    experiments.extend(_run_ab_regime(args))
    return experiments


def main():
    parser = argparse.ArgumentParser(
        description="MoS verification protocol experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- scaling ---
    sp = subparsers.add_parser("scaling", help="Scaling experiment: completeness vs n")
    _add_common_args(sp)

    # --- bent ---
    sp = subparsers.add_parser("bent", help="Bent function worst-case experiment")
    _add_common_args(sp)

    # --- truncation ---
    sp = subparsers.add_parser("truncation", help="Verifier truncation tradeoff experiment")
    _add_common_args(sp)
    sp.add_argument(
        "--n",
        type=int,
        default=None,
        help="Fixed n for the truncation experiment, which sweeps a 2-D "
             "grid of (epsilon x verifier_samples) at a single dimension "
             "rather than sweeping n. Defaults to n-min when not specified.",
    )

    # --- noise ---
    sp = subparsers.add_parser("noise", help="Noise sweep experiment")
    _add_common_args(sp)

    # --- soundness ---
    sp = subparsers.add_parser("soundness", help="Soundness against dishonest provers")
    _add_common_args(sp)

    # --- soundness_multi ---
    sp = subparsers.add_parser("soundness_multi", help="Soundness against dishonest provers with multi-element targets")
    _add_common_args(sp)

    # --- average_case ---
    sp = subparsers.add_parser("average_case", help="Average-case performance across function families")
    _add_common_args(sp)

    # --- gate_noise ---
    sp = subparsers.add_parser("gate_noise", help="Gate-level depolarising noise experiment")
    _add_common_args(sp)

    # --- k_sparse ---
    sp = subparsers.add_parser("k_sparse", help="k-Fourier-sparse verification path experiment")
    _add_common_args(sp)

    # --- theta_sensitivity ---
    sp = subparsers.add_parser("theta_sensitivity", help="Theta sensitivity: extraction boundary mapping")
    _add_common_args(sp)

    # --- ab_regime ---
    sp = subparsers.add_parser("ab_regime", help="a^2 != b^2 distributional regime experiment")
    _add_common_args(sp)

    # --- all ---
    sp = subparsers.add_parser("all", help="Run all experiments")
    _add_common_args(sp)
    sp.add_argument(
        "--n",
        type=int,
        default=None,
        help="Fixed n for the truncation experiment. Defaults to n-min.",
    )

    # --- merge ---
    sp = subparsers.add_parser(
        "merge", help="Merge sharded experiment result files"
    )
    sp.add_argument(
        "shards", nargs="+", type=Path, help="Shard .pb files to merge"
    )
    sp.add_argument(
        "-o", "--output", type=Path, required=True, help="Output merged .pb file"
    )

    args = parser.parse_args()

    # Validate shard flags (not applicable to merge)
    if args.command != "merge":
        if (args.shard_index is None) != (args.num_shards is None):
            parser.error("--shard-index and --num-shards must be used together")
        if args.num_shards is not None:
            if args.num_shards < 1:
                parser.error("--num-shards must be >= 1")
            if not (0 <= args.shard_index < args.num_shards):
                parser.error(
                    f"--shard-index must be in [0, {args.num_shards}), "
                    f"got {args.shard_index}"
                )

    # Dispatch table
    dispatch = {
        "scaling": _run_scaling,
        "bent": _run_bent,
        "truncation": _run_truncation,
        "noise": _run_noise,
        "soundness": _run_soundness,
        "soundness_multi": _run_soundness_multi,
        "average_case": _run_average_case,
        "gate_noise": _run_gate_noise,
        "k_sparse": _run_k_sparse,
        "theta_sensitivity": _run_theta_sensitivity,
        "ab_regime": _run_ab_regime,
        "all": _run_all,
        "merge": _run_merge,
    }

    t_total = time.time()
    experiments = dispatch[args.command](args)
    wall_total = time.time() - t_total

    if args.command == "merge":
        return

    output_dir = Path(args.output_dir)
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
