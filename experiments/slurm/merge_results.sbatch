#!/bin/bash
#SBATCH --job-name=mos-merge
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Usage: sbatch merge_results.sbatch
# Set PATTERN and OUTPUT via environment, e.g.:
#   PATTERN="scaling_4_16_24" OUTPUT="scaling_4_16_24" sbatch merge_results.sbatch
# NOTE: slurm_logs/ must exist before submission. Use submit.sh or run: mkdir -p slurm_logs

cd /dcs/23/u5505454/cs310-code
source .venv/bin/activate

PATTERN="${PATTERN:?Set PATTERN to the base filename, e.g. scaling_4_16_24}"
OUTPUT="${OUTPUT:-$PATTERN}"

echo "Merging results/${PATTERN}_shard*.pb -> results/${OUTPUT}.pb"

python -m experiments.harness merge \
    results/${PATTERN}_shard*.pb \
    -o "results/${OUTPUT}.pb"
