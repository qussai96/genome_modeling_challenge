#!/bin/bash
#SBATCH --job-name=serova_hackathon
#SBATCH --output=serova_hackathon.%j.out
#SBATCH --error=serova_hackathon.%j.err
#SBATCH --partition=GPU-A40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=24:00:00

# ============================================================================
# SEROVA CHALLENGE - Cell-Type Specific mRNA Sequence Design
# 24-Hour Hackathon Solution - SLURM Submission Script
# ============================================================================
#
# This script submits the complete hackathon pipeline to compute nodes with:
#   - GPU-A40 (40GB VRAM)
#   - 64 CPU cores
#   - 240GB RAM
#   - 24-hour walltime
#
# Usage:
#   sbatch serova_hackathon.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f serova_hackathon.JOBID.out
#
# ============================================================================

set -e  # Exit on error

echo "================================================================================"
echo "  SEROVA CHALLENGE - Hackathon Submission"
echo "  Cell-Type Specific mRNA Sequence Design"
echo "================================================================================"
echo ""
echo "Job Details:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Partition: $SLURM_JOB_PARTITION"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE MB"
echo "  Nodes: $SLURM_NNODES"
echo "  GPUs: $SLURM_GPUS_PER_NODE"
echo ""
echo "Job started at: $(date)"
echo "================================================================================"
echo ""

# Setup paths
PROJECT_DIR="/home/students/q.abbas/+proj-q.abbas/genome_modeling"
LOG_DIR="${PROJECT_DIR}/fibroblast_vs_neurons/logs"
RESULTS_DIR="${PROJECT_DIR}/fibroblast_vs_neurons"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"


# Log file
LOG_FILE="${LOG_DIR}/slurm_${SLURM_JOB_ID}.log"

echo "Project Directory: $PROJECT_DIR" | tee "$LOG_FILE"
echo "Results Directory: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check GPU availability
echo "GPU Information:" | tee -a "$LOG_FILE"
nvidia-smi | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Python environment
echo "Setting up Python environment..." | tee -a "$LOG_FILE"
module load python/3.10 2>/dev/null || true
python --version | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Change to project directory
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# OPTION 1: Run Jupyter Notebook (for interactive development)
# ============================================================================
# Uncomment the following block to run the main notebook
#
# echo "Starting Jupyter Notebook Execution..." | tee -a "$LOG_FILE"
# jupyter nbconvert --to notebook --execute hackathon_final_notebook.ipynb \
#     --output="${RESULTS_DIR}/hackathon_final_notebook_executed_${SLURM_JOB_ID}.ipynb" \
#     --ExecutePreprocessor.timeout=3600 | tee -a "$LOG_FILE"
#
# ============================================================================

# ============================================================================
# OPTION 2: Run Python Pipeline Script (for automated processing)
# ============================================================================

echo "Starting Hackathon Pipeline..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Generate sequences with RiboNN data
python hackathon_pipeline.py \
    --mode generate \
    --data "${PROJECT_DIR}/raw_data/41587_2025_2712_MOESM3_ESM.xlsx" \
    --output-dir "$RESULTS_DIR" \
    --target-cell "TE_neurons" \
    --offtarget-cell "TE_fibroblast" \
    2>&1 | tee -a "$LOG_FILE"

PIPELINE_EXIT_CODE=$?

# Check results
echo "" | tee -a "$LOG_FILE"
echo "Pipeline execution completed with exit code: $PIPELINE_EXIT_CODE" | tee -a "$LOG_FILE"
echo "Checking results..." | tee -a "$LOG_FILE"

if [ -f "${RESULTS_DIR}/top_20_candidates.csv" ]; then
    echo "✓ Top 20 candidates generated" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Results Preview:" | tee -a "$LOG_FILE"
    head -5 "${RESULTS_DIR}/top_20_candidates.csv" | tee -a "$LOG_FILE"
else
    echo "✗ Results file not found" | tee -a "$LOG_FILE"
fi

if [ -f "${RESULTS_DIR}/best_sequence.fasta" ]; then
    echo "✓ Best sequence FASTA generated" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Best Sequence:" | tee -a "$LOG_FILE"
    cat "${RESULTS_DIR}/best_sequence.fasta" | tee -a "$LOG_FILE"
fi

if [ -f "${RESULTS_DIR}/all_candidates_ranked.csv" ]; then
    echo "✓ All candidates ranked and saved" | tee -a "$LOG_FILE"
    CANDIDATE_COUNT=$(wc -l < "${RESULTS_DIR}/all_candidates_ranked.csv")
    echo "  Total candidates: $((CANDIDATE_COUNT - 1))" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "Job completed at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Results location: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "Pipeline exit code: $PIPELINE_EXIT_CODE" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

exit $PIPELINE_EXIT_CODE
