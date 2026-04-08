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
bash experiments/slurm/submit.sh <experiment> <n_min> <n_max> <trials> <num_shards> [partition] [seed]
# Example: 8 shards of the scaling experiment on tiger (default)
bash experiments/slurm/submit.sh scaling 4 16 24 8
# Example: submit to falcon partition
bash experiments/slurm/submit.sh scaling 4 16 24 8 falcon
```
This submits a SLURM array job (one task per shard) then queues a dependent merge job
that combines shard `.pb` files into the final result. Partition defaults to `tiger`.
CPU count is auto-adjusted for GPU partitions (falcon=12, gecko=16, eagle=6).

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

# Audit and follow-up reruns

A full per-experiment audit of all 11 experiments against
`papers/classical_verification_of_quantum_learning.pdf` lives in
`audit/`. The cross-experiment synthesis is `audit/SUMMARY.md`; each
individual audit file lists MAJOR/MINOR/NIT findings keyed back to
specific lines of code.

The protocol code under `ql/` and `mos/` is **correct** and should not
be modified — every Theorem 12 / Theorem 15 invariant has been
audited line-by-line. The audit findings are about experimental
**framing**, hard-coded sample budgets bypassing analytic formulas,
and a few specific harness bugs (now fixed). Tier 1 / Tier 2 fixes
have been applied; **Tier 3 / Tier 4 reruns** that need to happen on
the DCS cluster are tracked in `audit/FOLLOW_UPS.md`.

When working on an experiment, **always read its audit file**
(`audit/<experiment>.md`) before changing the harness, plot script,
or interpretation — the audit explains what was already known to be
wrong with the previous version and why it was fixed the way it was.

The most important applied fixes:

| Experiment | Fix | Existing `.pb` valid? |
|---|---|---|
| `average_case` | `TrialSpec.k` is now plumbed for `k_sparse_*` and `sparse_plus_noise`; `random_boolean` dropped | NO — needs rerun |
| `soundness_multi` | `classical_samples_verifier` default 3000 → 30000; `_strategy_diluted_list` formula fixed | NO — needs rerun |
| `noise_sweep` | η range extended to cross theoretical breakdown; θ held fixed | NO — needs rerun |
| `scaling`, `theta_sensitivity`, `truncation`, `k_sparse`, `bent`, `ab_regime`, `gate_noise`, `soundness` | docstring + plot caption / interpretation fixes only | yes |

The follow-up reruns are listed in `audit/FOLLOW_UPS.md` with
exact submission commands.
