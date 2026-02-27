#!/bin/bash
#SBATCH --job-name=hackathon_notebook
#SBATCH --output=hackathon_notebook.%j.out
#SBATCH --error=hackathon_notebook.%j.err
#SBATCH --partition=GPU-A40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=24:00:00

# ============================================================================
# SEROVA CHALLENGE - Run Jupyter Notebook on GPU Cluster
# ============================================================================
#
# This script runs the hackathon notebook end-to-end on a GPU node
#
# Usage:
#   sbatch run_notebook_gpu.slurm
#
# ============================================================================

set -e

PROJECT_DIR="/home/students/q.abbas/+proj-q.abbas/genome_modeling"
LOG_DIR="${PROJECT_DIR}/logs"
OUTPUT_DIR="${PROJECT_DIR}/results"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Starting Jupyter Notebook Conversion on GPU Node..."
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"

# Run notebook with timeout of 6 hours
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=21600 \
    --output="${OUTPUT_DIR}/hackathon_final_notebook_executed_${SLURM_JOB_ID}.ipynb" \
    "${PROJECT_DIR}/hackathon_final_notebook.ipynb"

echo "Notebook execution completed at: $(date)"
echo "Output saved to: ${OUTPUT_DIR}/hackathon_final_notebook_executed_${SLURM_JOB_ID}.ipynb"
