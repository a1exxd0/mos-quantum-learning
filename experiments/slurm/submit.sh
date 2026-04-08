#!/bin/bash
# Submit a SLURM Job Array for an experiment, then queue a dependent merge job.
#
# Usage:
#   bash experiments/slurm/submit.sh <experiment> <n_min> <n_max> <trials> <num_shards> [partition] [seed]
#
# Examples:
#   bash experiments/slurm/submit.sh scaling 4 16 24 8
#   bash experiments/slurm/submit.sh scaling 4 16 24 8 falcon
#
# This submits:
#   1. An array job (0..num_shards-1) that runs the experiment shards
#   2. A merge job that waits for all shards, then combines them

set -euo pipefail

EXPERIMENT="${1:?Usage: submit.sh <experiment> <n_min> <n_max> <trials> <num_shards>}"
N_MIN="${2:?Missing n_min}"
N_MAX="${3:?Missing n_max}"
TRIALS="${4:?Missing trials}"
NUM_SHARDS="${5:?Missing num_shards}"
PARTITION="${6:-tiger}"
SEED="${7:-42}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATTERN="${EXPERIMENT}_${N_MIN}_${N_MAX}_${TRIALS}"

# For soundness, trials are clamped to min 50
if [ "$EXPERIMENT" = "soundness" ] && [ "$TRIALS" -lt 50 ]; then
    PATTERN="${EXPERIMENT}_${N_MIN}_${N_MAX}_50"
fi

# The 'noise' subcommand writes its output as 'noise_sweep_*'; every other
# experiment uses the CLI subcommand name as the file basename. Without
# this special case, the merge job would glob 'noise_4_16_100_shard*.pb'
# (no match) and silently produce no output.
if [ "$EXPERIMENT" = "noise" ]; then
    PATTERN="noise_sweep_${N_MIN}_${N_MAX}_${TRIALS}"
fi

# SLURM opens --output/--error files before the script body runs,
# so these directories must exist at submission time.
mkdir -p slurm_logs results

echo "Submitting ${EXPERIMENT}: n=[${N_MIN},${N_MAX}], trials=${TRIALS}, shards=${NUM_SHARDS}, partition=${PARTITION}"

# Override cpus-per-task for partitions with fewer cores
CPUS_FLAG=()
case "${PARTITION}" in
    eagle)          CPUS_FLAG=(--cpus-per-task=6)  ;;
    falcon)         CPUS_FLAG=(--cpus-per-task=12) ;;
    gecko)          CPUS_FLAG=(--cpus-per-task=16) ;;
esac

# Submit array job
ARRAY_JOB_ID=$(
    EXPERIMENT="$EXPERIMENT" N_MIN="$N_MIN" N_MAX="$N_MAX" TRIALS="$TRIALS" SEED="$SEED" \
    sbatch --parsable \
        --partition="${PARTITION}" \
        "${CPUS_FLAG[@]}" \
        --array="0-$((NUM_SHARDS - 1))" \
        --job-name="mos-${EXPERIMENT}" \
        "${SCRIPT_DIR}/run_experiment.sbatch"
)
echo "  Array job submitted: ${ARRAY_JOB_ID} (${NUM_SHARDS} tasks)"

# Submit dependent merge job
MERGE_JOB_ID=$(
    PATTERN="$PATTERN" OUTPUT="$PATTERN" \
    sbatch --parsable \
        --partition="${PARTITION}" \
        --dependency="afterok:${ARRAY_JOB_ID}" \
        "${SCRIPT_DIR}/merge_results.sbatch"
)
echo "  Merge job submitted: ${MERGE_JOB_ID} (depends on ${ARRAY_JOB_ID})"
echo ""
echo "Monitor: squeue --me"
echo "Result:  results/${PATTERN}.pb"
