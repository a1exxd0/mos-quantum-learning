# Quickstart for environment setup
Assuming you have [UV](https://docs.astral.sh/uv/) installed:
```sh
uv sync
```

# Developer rules

Always use Context7 when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

If you are logged into thhe `kudu-taught` server, you have access to DCS batch compute. Use `experiments/DCS_BATCH_COMPUTE.md` to find information 
about how to use the batch compute system.

Do not mention Claude Code as a co-author.

# Run tests
Use parallel workers matching the computer's core count:
```sh
uv run pytest -n auto
```

# Run experiments
Use `--workers` to parallelise with the computer's core count:
```sh
uv run python -m experiments.harness {scaling,bent,truncation,noise,soundness,soundness_multi,average_case,gate_noise,k_sparse,theta_sensitivity,ab_regime,all} --workers $(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

# Run experiments on the DCS cluster (SLURM)
Submit a single experiment as a sharded job array, with an automatic merge step:
```sh
bash experiments/slurm/submit.sh <experiment> <n_min> <n_max> <trials> <num_shards>
# Example: 8 shards of the scaling experiment
bash experiments/slurm/submit.sh scaling 4 16 24 8
```
This submits a SLURM array job (one task per shard) to the `tiger` partition, then
queues a dependent merge job that combines shard `.pb` files into the final result.

Monitor and manage jobs:
```sh
squeue --me          # check status
scancel <JOBID>      # cancel a job
```

Manual shard execution (without `submit.sh`):
```sh
mkdir -p slurm_logs results
sbatch --array=0-7 experiments/slurm/run_experiment.sbatch
# After all tasks complete:
uv run python -m experiments.harness merge results/scaling_4_16_24_shard*.pb -o results/scaling_4_16_24.pb
```

# Decode experiment results
```sh
uv run python -m experiments.decode results/scaling_4_10_20.pb
uv run python -m experiments.decode results/scaling_4_10_20.pb -o results/scaling_4_10_20.json
uv run python -m experiments.decode results/*.pb
```

# Local Documentation Build
```sh
uv run sphinx-build docs docs/_build/html
open docs/_build/html/index.html
```
