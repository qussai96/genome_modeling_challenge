#!/usr/bin/env python3
"""
Standalone TE predictor training script.

Trains a TE predictor once, saves a reusable checkpoint, and stores cell mapping
metadata so generation can run without retraining.
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import torch

from hackathon_pipeline import (
    logger,
    extract_cell_to_idx_mapping,
    load_and_prepare_training_data,
    train_te_predictor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TE predictor model once and save checkpoint")
    parser.add_argument(
        "--training-data",
        type=str,
        required=True,
        help="Path to training data file (xlsx/csv)",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        required=True,
        help="Path to save TE predictor checkpoint (.pt)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory for intermediate embedding/training artifacts (defaults next to output model)",
    )
    parser.add_argument("--target-cell", type=str, default="TE_neurons", help="Target cell type column")
    parser.add_argument("--offtarget-cell", type=str, default="TE_fibroblast", help="Off-target cell type column")
    parser.add_argument("--sample-size", type=int, default=5000, help="Max transcripts to sample for faster training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    args = parser.parse_args()

    training_data_path = Path(args.training_data)
    if not training_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {training_data_path}")

    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = output_model_path.parent / "te_predictor_training_artifacts"
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STANDALONE TE PREDICTOR TRAINING")
    logger.info("=" * 70)
    logger.info(f"Training data: {training_data_path}")
    logger.info(f"Output model: {output_model_path}")
    logger.info(f"Work dir: {work_dir}")

    cell_to_idx = extract_cell_to_idx_mapping(training_data_path)
    num_cells = len(cell_to_idx)
    logger.info(f"Found {num_cells} cell types")

    if args.target_cell not in cell_to_idx:
        raise ValueError(f"Target cell '{args.target_cell}' not found in training data")
    if args.offtarget_cell not in cell_to_idx:
        raise ValueError(f"Off-target cell '{args.offtarget_cell}' not found in training data")

    rna_embeddings, target_te, offtarget_te, cell_indices, _ = load_and_prepare_training_data(
        training_data_path,
        work_dir,
        args.target_cell,
        args.offtarget_cell,
        cell_to_idx=cell_to_idx,
        sample_size=args.sample_size,
    )

    te_predictor = train_te_predictor(
        rna_embeddings,
        target_te,
        offtarget_te,
        cell_indices,
        work_dir,
        num_cells=num_cells,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    checkpoint = {
        "model_state_dict": te_predictor.state_dict(),
        "num_cells": num_cells,
        "cell_to_idx": {k: int(v) for k, v in cell_to_idx.items()},
        "target_cell": args.target_cell,
        "offtarget_cell": args.offtarget_cell,
        "rna_embedding_dim": int(rna_embeddings.shape[1]),
        "trained_at": datetime.now().isoformat(),
    }
    torch.save(checkpoint, output_model_path)

    cell_map_path = output_model_path.with_suffix(output_model_path.suffix + ".cell_to_idx.json")
    with open(cell_map_path, "w") as f:
        json.dump({k: int(v) for k, v in cell_to_idx.items()}, f, indent=2)

    logger.info(f"✓ Saved TE predictor checkpoint to {output_model_path}")
    logger.info(f"✓ Saved cell mapping to {cell_map_path}")
    logger.info("Training complete. You can now reuse this model for generation without retraining.")


if __name__ == "__main__":
    main()
