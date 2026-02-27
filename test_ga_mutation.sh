#!/bin/bash

# Test script for Genetic Algorithm-based UTR mutation

echo "================================"
echo "Testing GA-based UTR Mutation"
echo "================================"

cd /home/students/q.abbas/+proj-q.abbas/genome_modeling

# Test 1: Train TE predictor from scratch and use GA for candidate generation
echo ""
echo "[TEST 1] Train TE Predictor + Generate with GA"
echo "=============================================="

python3 hackathon_pipeline.py \
    --mode generate \
    --train-te-predictor \
    --use-ga \
    --ga-training-data raw_data/41587_2025_2712_MOESM3_ESM.xlsx \
    --ga-population-size 50 \
    --ga-generations 20 \
    --target-cell TE_neurons \
    --offtarget-cell TE_fibroblast \
    --output-dir /tmp/test_ga_run \
    --utr5-length 62 \
    --utr3-length 267 \
    2>&1 | head -100

echo ""
echo "[TEST 1] Complete!"
echo ""

# Test 2: Use pre-trained module (if exists) for faster generation
if [ -f "/tmp/test_ga_run/te_predictor_final.pt" ]; then
    echo ""
    echo "[TEST 2] Using Pre-trained TE Predictor"
    echo "======================================"
    
    python3 hackathon_pipeline.py \
        --mode generate \
        --use-ga \
        --te-predictor-model /tmp/test_ga_run/te_predictor_final.pt \
        --ga-population-size 30 \
        --ga-generations 15 \
        --target-cell TE_neurons \
        --offtarget-cell TE_fibroblast \
        --output-dir /tmp/test_ga_run_2 \
        --utr5-length 62 \
        --utr3-length 267 \
        2>&1 | head -100
    
    echo ""
    echo "[TEST 2] Complete!"
fi

echo ""
echo "All tests finished!"
echo "Results in /tmp/test_ga_run* folders"
