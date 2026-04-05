#!/bin/bash
#SBATCH --job-name=mos-experiment
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
# Set --array when submitting, e.g.: sbatch --array=0-7 run_experiment.sbatch scaling
# NOTE: slurm_logs/ must exist before submission (SLURM opens output files
# before the script runs). Use submit.sh or run: mkdir -p slurm_logs

# --- Derive shard params from SLURM array ---
SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
NUM_SHARDS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
WORKERS="${SLURM_CPUS_PER_TASK}"

# --- Configuration (override via environment or edit here) ---
EXPERIMENT="${EXPERIMENT:-scaling}"
N_MIN="${N_MIN:-4}"
N_MAX="${N_MAX:-16}"
TRIALS="${TRIALS:-24}"
SEED="${SEED:-42}"

# --- Environment setup ---
cd /dcs/23/u5505454/cs310-code
source .venv/bin/activate

mkdir -p results

echo "=== SLURM Array Task $((SHARD_INDEX + 1))/${NUM_SHARDS} ==="
echo "Experiment: ${EXPERIMENT}, n=[${N_MIN},${N_MAX}], trials=${TRIALS}"
echo "Workers: ${WORKERS}, Node: $(hostname)"
echo ""

# --- Run ---
python -m experiments.harness "$EXPERIMENT" \
    --n-min "$N_MIN" \
    --n-max "$N_MAX" \
    --trials "$TRIALS" \
    --seed "$SEED" \
    --workers "$WORKERS" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    --output-dir results
