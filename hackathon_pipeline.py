#!/usr/bin/env python3
"""
Cell-Type Specific mRNA Sequence Design - Training and Generation Pipeline
===========================================================================

Serova Challenge - 24-hour Hackathon
Main Script for training Style and Judge models and generating optimized sequences

Usage:
    sbatch serova_hackathon.sh
    python hackathon_pipeline.py --mode train --data data.xlsx
    python hackathon_pipeline.py --mode generate --base-sequence "..." --base-protein "..."
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from datetime import datetime
import os
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Get SLURM job info if running on SLURM
SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
SLURM_ARRAY_TASK_ID = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
SLURM_SUBMIT_DIR = os.environ.get('SLURM_SUBMIT_DIR', str(Path.cwd()))

# Setup logging with SLURM context
def setup_logger():
    """Configure logging for both console and file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (with SLURM job ID)
    log_file = Path(__file__).parent / f'pipeline_job_{SLURM_JOB_ID}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] [Job: {job}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_format._fmt = file_format._fmt.format(job=SLURM_JOB_ID)
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()
logger.info(f"Pipeline started - SLURM Job ID: {SLURM_JOB_ID}")

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Embedding Generation Functions
# ==============================================================================

def write_fasta(sequences: List[str], output_path: Path, prefix: str = "seq") -> Path:
    """Write sequences to FASTA file."""
    with open(output_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">{prefix}_{i}\n{seq}\n")
    logger.info(f"✓ Wrote {len(sequences)} sequences to {output_path}")
    return output_path


def translate_to_protein(rna_sequence: str, start_pos: int = 0) -> str:
    """
    Translate RNA sequence to protein using standard genetic code.
    
    Args:
        rna_sequence: RNA sequence (A, C, G, U)
        start_pos: Position to start translation (default: 0)
        
    Returns:
        Protein sequence (amino acids)
    """
    # Standard genetic code
    codon_table = {
        'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
        'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
        'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
        'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
        'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
        'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
        'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
        'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }
    
    # Find CDS (assume 5'UTR ~50bp, or start from specified position)
    seq = rna_sequence[start_pos:].upper()
    
    # Translate in frame
    protein = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if len(codon) == 3:
            aa = codon_table.get(codon, 'X')
            if aa == '*':  # Stop codon
                break
            protein.append(aa)
    
    return ''.join(protein)


def split_transcript_regions(sequence: str, utr5_length: Optional[int] = None, 
                           utr3_length: Optional[int] = None) -> Tuple[str, str, str]:
    """Split full transcript into 5'UTR, CDS (with stop), and 3'UTR for RiboNN.
    
    Args:
        sequence: Full mRNA transcript sequence
        utr5_length: Length of 5'UTR (if provided, assumes user knows the structure)
        utr3_length: Length of 3'UTR (if provided, assumes user knows the structure)
        
    Returns:
        Tuple of (5'UTR, CDS, 3'UTR)
    """
    rna_seq = sequence.upper().replace('T', 'U')

    if len(rna_seq) < 6:
        return rna_seq, 'AUGUAA', ''

    # If user provides UTR lengths, use them directly
    if utr5_length is not None and utr3_length is not None:
        utr5 = rna_seq[:utr5_length]
        utr3 = rna_seq[-utr3_length:] if utr3_length > 0 else ''
        cds = rna_seq[utr5_length:len(rna_seq)-utr3_length if utr3_length > 0 else len(rna_seq)]
    else:
        # Auto-detect by searching for start and stop codons
        start_search_from = 50  # Default assumption
        start = rna_seq.find('AUG', start_search_from)
        if start == -1:
            start = rna_seq.find('AUG')
        if start == -1:
            start = min(50, max(len(rna_seq) - 3, 0))

        stop_codons = {'UAA', 'UAG', 'UGA'}
        stop = None
        for idx in range(start + 3, len(rna_seq) - 2, 3):
            codon = rna_seq[idx:idx + 3]
            if codon in stop_codons:
                stop = idx + 3
                break

        if stop is None:
            stop = len(rna_seq)

        utr5 = rna_seq[:start]
        cds = rna_seq[start:stop]
        utr3 = rna_seq[stop:]

    if len(cds) < 3:
        cds = 'AUG'

    if not cds.startswith('AUG'):
        cds = 'AUG' + cds

    frame_len = (len(cds) // 3) * 3
    cds = cds[:max(frame_len, 3)]

    if len(cds) < 3:
        cds = 'AUG'

    stop_codons = ('UAA', 'UAG', 'UGA')
    if cds[-3:] not in stop_codons:
        cds = cds + 'UAA'

    if len(cds) % 3 != 0:
        cds = cds[:(len(cds) // 3) * 3]
        if len(cds) < 6:
            cds = 'AUGUAA'

    if len(utr5) == 0:
        utr5 = 'A'
    if len(utr3) == 0:
        utr3 = 'A'

    return utr5, cds, utr3


def ribonn_prediction_column(cell_name: str) -> str:
    """Map TE cell name to RiboNN prediction column name."""
    cell_name = cell_name.strip()
    if cell_name.startswith('TE_'):
        return f'predicted_{cell_name}'
    return f'predicted_TE_{cell_name}'


def predict_te_with_ribonn(candidates: List[str], output_dir: Path,
                          target_cell: str, offtarget_cell: str,
                          species: str = 'human', top_k_models: int = 5,
                          utr5_length: Optional[int] = None,
                          utr3_length: Optional[int] = None) -> pd.DataFrame:
    """Predict candidate TE values with RiboNN and return ranked results."""
    import subprocess

    ribonn_repo = Path.home() / 'Tools' / 'RiboNN'
    ribonn_env = Path.home() / 'anaconda3' / 'envs' / 'RiboNN'

    if not ribonn_repo.exists():
        raise RuntimeError(f'RiboNN repo not found at {ribonn_repo}')
    if not ribonn_env.exists():
        raise RuntimeError(f'RiboNN env not found at {ribonn_env}')

    logger.info(f'Predicting TE for {len(candidates)} candidates with RiboNN ({species})...')

    input_path = output_dir / 'ribonn_prediction_input.tsv'
    output_path = output_dir / 'ribonn_prediction_output.tsv'
    runner_path = output_dir / 'run_ribonn_predict.py'

    rows = []
    for idx, seq in enumerate(candidates):
        utr5, cds, utr3 = split_transcript_regions(seq, utr5_length, utr3_length)
        rows.append({
            'tx_id': f'candidate_{idx}',
            'utr5_sequence': utr5.replace('U', 'T'),
            'cds_sequence': cds.replace('U', 'T'),
            'utr3_sequence': utr3.replace('U', 'T')
        })

    input_df = pd.DataFrame(rows)
    for col in ['utr5_sequence', 'cds_sequence', 'utr3_sequence']:
        input_df[col] = input_df[col].fillna('A').astype(str)
        input_df.loc[input_df[col].str.len() == 0, col] = 'A'

    input_df.to_csv(input_path, sep='\t', index=False)
    logger.info(f'✓ Wrote RiboNN input: {input_path}')

    with open(runner_path, 'w') as f:
        f.write('''
import sys
from pathlib import Path
import pandas as pd
from src.predict import predict_using_nested_cross_validation_models

input_path = sys.argv[1]
species = sys.argv[2]
output_path = sys.argv[3]
top_k_models = int(sys.argv[4])

run_df = pd.read_csv(f"models/{species}/runs.csv")

predictions = predict_using_nested_cross_validation_models(
    input_path,
    species,
    run_df,
    top_k_models,
    batch_size=32,
    num_workers=4,
)

columns_to_aggregate = [col for col in predictions.columns if col.startswith("predicted_")]
predicted_te = predictions.groupby(
    ["tx_id", "utr5_sequence", "cds_sequence", "utr3_sequence"],
    as_index=False,
)[columns_to_aggregate].agg("mean")

predicted_te["mean_predicted_TE"] = predicted_te[columns_to_aggregate].mean(axis=1)
predicted_te.to_csv(output_path, sep="\t", index=False)
print(f"Predictions written to {output_path}")
''')

    cmd = (
        f'cd {ribonn_repo} && '
        f'PYTHONPATH={ribonn_repo} '
        f'conda run -p {ribonn_env} python {runner_path} {input_path} {species} {output_path} {top_k_models}'
    )

    logger.info(f'Running: {cmd}')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f'RiboNN failed:\n{result.stderr}')
        raise RuntimeError('RiboNN prediction failed')

    logger.info(result.stdout)

    pred_df = pd.read_csv(output_path, sep='\t')
    target_col = ribonn_prediction_column(target_cell)
    offtarget_col = ribonn_prediction_column(offtarget_cell)

    missing = [col for col in [target_col, offtarget_col] if col not in pred_df.columns]
    if missing:
        available = [col for col in pred_df.columns if col.startswith('predicted_TE_')][:20]
        raise RuntimeError(
            f'Missing RiboNN columns: {missing}. Example available columns: {available}'
        )

    pred_df['TE_target'] = pred_df[target_col]
    pred_df['TE_offtarget'] = pred_df[offtarget_col]

    sequence_map = {f'candidate_{i}': seq for i, seq in enumerate(candidates)}
    result_df = pd.DataFrame({
        'seq_id': pred_df['tx_id'],
        'sequence': pred_df['tx_id'].map(sequence_map),
        'TE_target': pred_df['TE_target'].astype(float),
        'TE_offtarget': pred_df['TE_offtarget'].astype(float),
    })

    return result_df


def generate_rna_embeddings(sequences: List[str], output_dir: Path) -> np.ndarray:
    """
    Generate RNA embeddings using RNA-FM foundation model.
    
    Args:
        sequences: List of RNA sequences (A, C, G, U)
        output_dir: Directory to save temporary files
        
    Returns:
        np.ndarray of shape (N, 640) - RNA embeddings
    """
    import subprocess
    
    logger.info(f"Generating RNA embeddings for {len(sequences)} sequences using RNA-FM...")
    
    # Create temp FASTA using windowed embedding for long sequences
    fasta_path = output_dir / 'temp_rna_sequences.fasta'
    max_rnafm_len = 1022
    window_stride = 480
    normalized_sequences = []
    windowed_sequences = []
    window_parent_indices = []
    expanded_count = 0
    invalid_count = 0
    allowed = {'A', 'C', 'G', 'U'}

    for seq in sequences:
        raw = str(seq).upper().replace('T', 'U')
        cleaned_chars = []
        for ch in raw:
            if ch in allowed:
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append('A')
                invalid_count += 1
        cleaned = ''.join(cleaned_chars)
        if len(cleaned) == 0:
            cleaned = 'A'
        normalized_sequences.append(cleaned)

    for parent_idx, seq in enumerate(normalized_sequences):
        if len(seq) <= max_rnafm_len:
            windowed_sequences.append(seq)
            window_parent_indices.append(parent_idx)
            continue

        last_start = len(seq) - max_rnafm_len
        starts = list(range(0, last_start + 1, window_stride))
        if len(starts) == 0 or starts[-1] != last_start:
            starts.append(last_start)

        expanded_count += len(starts) - 1
        for start in starts:
            windowed_sequences.append(seq[start:start + max_rnafm_len])
            window_parent_indices.append(parent_idx)

    if expanded_count > 0:
        logger.info(
            f"RNA-FM windowing expanded {len(normalized_sequences)} sequences into "
            f"{len(windowed_sequences)} windows (max_len={max_rnafm_len}, stride={window_stride})"
        )
    if invalid_count > 0:
        logger.warning(f"RNA-FM input sanitized {invalid_count} non-ACGU characters")

    write_fasta(windowed_sequences, fasta_path, prefix="rna")
    
    # Create a custom script that processes the FASTA file with RNA-FM
    custom_script = output_dir / 'run_rnafm_batch.py'
    with open(custom_script, 'w') as f:
        f.write('''
import os
import torch
import numpy as np
import sys
import fm

model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()

# Read FASTA
fasta_path = sys.argv[1]
output_path = sys.argv[2]

sequences = []
ids = []
with open(fasta_path, 'r') as f:
    current_seq = []
    current_id = None
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if current_id is not None:
                sequences.append(''.join(current_seq))
                ids.append(current_id)
            current_id = line[1:]
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        sequences.append(''.join(current_seq))
        ids.append(current_id)

print(f"Processing {len(sequences)} sequences...")

# Generate embeddings in batches
all_embeddings = []
all_ids = []
batch_size = 8

def sanitize_and_truncate(seq: str, max_len: int = 1022) -> str:
    seq = seq.upper().replace('T', 'U')
    cleaned = ''.join(ch if ch in {'A', 'C', 'G', 'U'} else 'A' for ch in seq)
    if len(cleaned) == 0:
        cleaned = 'A'
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned

sequences = [sanitize_and_truncate(s) for s in sequences]

requested_device = os.environ.get("RNAFM_DEVICE", "auto").lower()
if requested_device == "cpu":
    device = "cpu"
elif requested_device == "cuda":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
else:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = model.to(device=device)
print(f"RNA-FM device: {device}")

for i in range(0, len(sequences), batch_size):
    batch_seqs = sequences[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    data = list(zip(batch_ids, batch_seqs))
    labels, strs, toks = batch_converter(data)

    def embed_with_current_device(token_tensor):
        token_tensor = token_tensor.to(device)
        with torch.no_grad():
            outputs = model(token_tensor, repr_layers=[12])
            token_embeddings = outputs["representations"][12]

            pooled_embeddings = []
            for b_idx, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                seq_emb = token_embeddings[b_idx, 1:seq_len+1].mean(dim=0)
                pooled_embeddings.append(seq_emb)

            return torch.stack(pooled_embeddings, dim=0)

    try:
        batch_embeddings = embed_with_current_device(toks)
    except RuntimeError as e:
        msg = str(e).lower()
        if ("cuda" in msg or "device-side assert" in msg) and device.startswith("cuda"):
            print("CUDA failure in RNA-FM batch; retrying on CPU for stability")
            torch.cuda.empty_cache()
            device = "cpu"
            model_cpu = model.to(device="cpu")
            model_cpu.eval()
            model = model_cpu
            batch_embeddings = embed_with_current_device(toks)
        else:
            raise

    all_embeddings.append(batch_embeddings.cpu().numpy())
    all_ids.extend(batch_ids)
    
    print(f"Processed {min(i+batch_size, len(sequences))}/{len(sequences)} sequences")

# Concatenate all
embeddings = np.vstack(all_embeddings).astype(np.float32)
print(f"Final shape: {embeddings.shape}")

# Save as .npz
np.savez(output_path, ids=np.array(all_ids, dtype=object), embeddings=embeddings)
print(f"Saved to {output_path}")
''')
    
    # Run RNA-FM
    output_npz = output_dir / 'temp_rna_sequences.rnafm_embeddings.npz'
    cmd = (
        f"conda run -p {Path.home() / 'anaconda3' / 'envs' / 'rnafm'} "
        f"python {custom_script} {fasta_path} {output_npz}"
    )
    
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"RNA-FM failed:\n{result.stderr}")
        raise RuntimeError("RNA-FM embedding generation failed")
    
    logger.info(result.stdout)
    
    # Load embeddings
    data = np.load(output_npz, allow_pickle=True)
    window_embeddings = data['embeddings'].astype(np.float32)

    if len(window_parent_indices) != len(window_embeddings):
        raise RuntimeError(
            f"Window mapping mismatch: {len(window_parent_indices)} parents for "
            f"{len(window_embeddings)} window embeddings"
        )

    # Pool window embeddings back to one embedding per original sequence
    n_original = len(normalized_sequences)
    pooled = np.zeros((n_original, window_embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(n_original, dtype=np.int32)

    for w_idx, parent_idx in enumerate(window_parent_indices):
        pooled[parent_idx] += window_embeddings[w_idx]
        counts[parent_idx] += 1

    if np.any(counts == 0):
        missing = int((counts == 0).sum())
        raise RuntimeError(f"Missing pooled embeddings for {missing} sequences")

    embeddings = pooled / counts[:, None]
    
    logger.info(f"✓ Generated RNA embeddings: {embeddings.shape}")
    return embeddings


def generate_protein_embeddings(proteins: List[str], output_dir: Path) -> np.ndarray:
    """
    Generate protein embeddings using ESM2-t6 foundation model.
    
    Args:
        proteins: List of protein sequences (amino acids)
        output_dir: Directory to save temporary files
        
    Returns:
        np.ndarray of shape (N, 320) - Protein embeddings
    """
    import subprocess
    
    logger.info(f"Generating protein embeddings for {len(proteins)} sequences using ESM2...")
    
    # Create temp FASTA
    fasta_path = output_dir / 'temp_protein_sequences.fasta'
    write_fasta(proteins, fasta_path, prefix="prot")
    
    # Run ESM2
    esm2_script = Path.home() / '+proj-q.abbas' / 'final_datasets' / 'scripts' / 'convert_fasta_to_esm2_embedding_320.py'
    output_npz = fasta_path.with_suffix('.fasta.esm2_embeddings_320.npz')
    
    cmd = (
        f"conda run -p {Path.home() / 'anaconda3' / 'envs' / 'esm2'} "
        f"python {esm2_script} {fasta_path} -o {output_npz}"
    )
    
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"ESM2 failed:\n{result.stderr}")
        raise RuntimeError(f"ESM2 embedding generation failed")
    
    logger.info(result.stdout)
    
    # Load embeddings
    data = np.load(output_npz, allow_pickle=True)
    embeddings = data['embeddings'].astype(np.float32)
    
    logger.info(f"✓ Generated protein embeddings: {embeddings.shape}")
    return embeddings


# ==============================================================================
# TE Predictor and Genetic Algorithm for Intelligent UTR Optimization
# ==============================================================================

class TEPredictorModel(nn.Module):
    """Neural network to predict Translation Efficiency from UTR embeddings and cell type."""
    
    def __init__(self, num_cells: int, rna_embedding_dim: int = 640, cell_embedding_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.num_cells = num_cells
        self.cell_embeddings = nn.Embedding(num_cells, cell_embedding_dim)  # Cell type embeddings
        
        # MLP: RNA embedding + cell embedding -> TE prediction
        fusion_dim = rna_embedding_dim + cell_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, rna_emb: torch.Tensor, cell_ids: torch.Tensor) -> torch.Tensor:
        """Predict TE given RNA embedding and cell type."""
        cell_emb = self.cell_embeddings(cell_ids)
        fusion = torch.cat([rna_emb, cell_emb], dim=1)
        te_pred = self.mlp(fusion).squeeze(-1)
        return te_pred


def extract_cell_to_idx_mapping(data_path: Path) -> Dict[str, int]:
    """
    Extract cell type to index mapping from Excel file.
    
    Args:
        data_path: Path to Excel file with TE_* columns
        
    Returns:
        Dictionary mapping cell type names to indices (sorted alphabetically)
    """
    df = pd.read_excel(data_path, sheet_name=0)
    cell_cols = [col for col in df.columns if col.startswith('TE_')]
    cell_to_idx = {col: idx for idx, col in enumerate(sorted(cell_cols))}
    logger.info(f"Extracted {len(cell_to_idx)} cell types from data")
    return cell_to_idx


def load_and_prepare_training_data(data_path: Path, output_dir: Path, 
                                  target_cell: str = 'TE_neurons',
                                  offtarget_cell: str = 'TE_fibroblast',
                                  cell_to_idx: Optional[Dict[str, int]] = None,
                                  sample_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Load training data, extract UTRs, generate embeddings, and prepare tensors for TE predictor training.
    
    Returns:
        (rna_embeddings, target_te_values, offtarget_te_values, cell_indices, cell_to_idx)
    """
    logger.info(f"Loading training data from {data_path}...")
    
    # Extract cell mapping if not provided
    if cell_to_idx is None:
        cell_to_idx = extract_cell_to_idx_mapping(data_path)
    
    # Validate target and offtarget cells exist in mapping
    if target_cell not in cell_to_idx:
        raise ValueError(f"Target cell '{target_cell}' not found in data. Available cells: {list(cell_to_idx.keys())[:10]}...")
    if offtarget_cell not in cell_to_idx:
        raise ValueError(f"Off-target cell '{offtarget_cell}' not found in data. Available cells: {list(cell_to_idx.keys())[:10]}...")
    
    df = pd.read_excel(data_path, sheet_name=0)
    logger.info(f"✓ Loaded {len(df)} transcripts")
    
    # Sample if specified
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {sample_size} transcripts for faster training")
    
    # Extract UTRs from full transcript sequence
    logger.info("Extracting UTRs from transcripts...")
    utrs = []
    valid_indices = []
    
    for pos, row in enumerate(df.itertuples(index=False)):
        try:
            tx_seq = str(row.tx_sequence).upper().replace('T', 'U')
            utr5_size = int(row.utr5_size)
            utr3_size = int(row.utr3_size)
            
            # Extract 5' and 3' UTRs
            utr5 = tx_seq[:utr5_size]
            utr3 = tx_seq[-utr3_size:] if utr3_size > 0 else ''
            
            utrs.append(utr5 + utr3)  # Concatenate UTRs
            valid_indices.append(pos)
        except Exception as e:
            # Skip rows with invalid data
            continue
    
    logger.info(f"✓ Extracted UTRs from {len(utrs)} valid transcripts")
    
    # Generate RNA embeddings for UTRs
    logger.info("Generating RNA-FM embeddings for UTRs...")
    rna_embeddings = generate_rna_embeddings(utrs, output_dir)
    logger.info(f"✓ Generated embeddings: {rna_embeddings.shape}")
    
    # Extract TE values for target and off-target cell types
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    
    target_te = torch.FloatTensor(df_valid[target_cell].values)
    offtarget_te = torch.FloatTensor(df_valid[offtarget_cell].values)
    
    # Create cell type indices using proper mapping
    target_cell_id = cell_to_idx[target_cell]
    offtarget_cell_id = cell_to_idx[offtarget_cell]
    cell_indices_target = torch.LongTensor([target_cell_id] * len(df_valid))
    cell_indices_offtarget = torch.LongTensor([offtarget_cell_id] * len(df_valid))
    
    logger.info(f"Target cell '{target_cell}' → index {target_cell_id}")
    logger.info(f"Off-target cell '{offtarget_cell}' → index {offtarget_cell_id}")
    
    rna_embeddings_torch = torch.FloatTensor(rna_embeddings)
    
    logger.info(f"Target TE range: [{target_te.min():.3f}, {target_te.max():.3f}]")
    logger.info(f"Off-target TE range: [{offtarget_te.min():.3f}, {offtarget_te.max():.3f}]")
    
    logger.info(f"Cell type mapping: {len(cell_to_idx)} unique cell types")
    return rna_embeddings_torch, target_te, offtarget_te, cell_indices_target, cell_to_idx


def train_te_predictor(rna_embeddings: torch.Tensor, target_te: torch.Tensor,
                      offtarget_te: torch.Tensor, cell_indices_target: torch.Tensor,
                      output_dir: Path, num_cells: int, epochs: int = 30, batch_size: int = 64,
                      learning_rate: float = 1e-3) -> TEPredictorModel:
    """Train a TE predictor model on UTR embeddings."""
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING TE PREDICTOR MODEL")
    logger.info("="*70)
    logger.info(f"Number of cell types: {num_cells}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare tensors for both target and offtarget
    cell_indices_offtarget = torch.ones_like(cell_indices_target)  # 1 for offtarget
    cell_indices_offtarget[:] = (cell_indices_target[0] + 1) % num_cells  # Cycle through cell types
    
    # Duplicate embeddings for both cell types
    all_rna_embeddings = torch.cat([rna_embeddings, rna_embeddings], dim=0)
    all_te_values = torch.cat([target_te, offtarget_te], dim=0)
    all_cell_indices = torch.cat([cell_indices_target, cell_indices_offtarget], dim=0)

    # Remove NaN/Inf labels to avoid unstable loss and missing checkpoints
    finite_mask = torch.isfinite(all_te_values)
    dropped = int((~finite_mask).sum().item())
    if dropped > 0:
        logger.warning(f"Dropping {dropped} TE labels with NaN/Inf before training")
    all_rna_embeddings = all_rna_embeddings[finite_mask]
    all_te_values = all_te_values[finite_mask]
    all_cell_indices = all_cell_indices[finite_mask]

    if len(all_te_values) < 10:
        raise RuntimeError("Not enough finite TE labels to train TE predictor")
    
    # Create model with correct num_cells
    model = TEPredictorModel(num_cells=num_cells).to(device)
    all_cell_indices = torch.cat([cell_indices_target, cell_indices_offtarget], dim=0)
    
    # Shuffle
    perm = torch.randperm(len(all_rna_embeddings))
    all_rna_embeddings = all_rna_embeddings[perm]
    all_te_values = all_te_values[perm]
    all_cell_indices = all_cell_indices[perm]
    
    # Split train/val
    split = int(0.8 * len(all_rna_embeddings))
    train_rna = all_rna_embeddings[:split].to(device)
    train_te = all_te_values[:split].to(device)
    train_cells = all_cell_indices[:split].to(device)
    
    val_rna = all_rna_embeddings[split:].to(device)
    val_te = all_te_values[split:].to(device)
    val_cells = all_cell_indices[split:].to(device)
    
    logger.info(f"Train size: {len(train_te)}, Val size: {len(val_te)}")
    logger.info(f"Number of cell types: {num_cells}")
    
    # Initialize model with correct num_cells
    model = TEPredictorModel(num_cells=num_cells, rna_embedding_dim=rna_embeddings.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_path = output_dir / 'te_predictor_best.pt'

    # Ensure a checkpoint always exists even if validation loss is NaN
    torch.save(model.state_dict(), best_model_path)
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for i in range(0, len(train_te), batch_size):
            batch_rna = train_rna[i:i+batch_size]
            batch_te = train_te[i:i+batch_size]
            batch_cells = train_cells[i:i+batch_size]
            
            optimizer.zero_grad()
            pred_te = model(batch_rna, batch_cells)
            loss = criterion(pred_te, batch_te)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_te)
        
        train_loss /= len(train_te)
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(val_rna, val_cells)
            val_loss = criterion(val_pred, val_te).item()

        if not np.isfinite(train_loss):
            logger.warning(f"Epoch {epoch+1}: train loss is non-finite ({train_loss}); stopping early")
            break

        if not np.isfinite(val_loss):
            logger.warning(f"Epoch {epoch+1}: val loss is non-finite ({val_loss}); keeping last best checkpoint")
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            continue
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    logger.info(f"✓ TE Predictor training complete!")
    logger.info("="*70 + "\n")
    
    return model.to('cpu')  # Move to CPU for inference


def _predict_specificity_scores(
    sequences: List[str],
    te_predictor: TEPredictorModel,
    target_cell_id: int,
    offtarget_cell_id: int,
    lambda_offtarget: float,
    embedding_output_dir: Path,
    utr5_length: Optional[int] = None,
    utr3_length: Optional[int] = None,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
) -> List[float]:
    """
    Batch-predict specificity scores for candidate sequences with caching.
    
    Note: To match training distribution, extracts UTRs only (same preprocessing as training)
    and embeds utr5+utr3 concatenation, not full transcript.
    
    Args:
        sequences: Full transcript sequences (5'UTR + CDS + 3'UTR)
        te_predictor: Trained TE predictor (trained on UTR-only embeddings)
        target_cell_id: Cell type to maximize
        offtarget_cell_id: Cell type to minimize
        lambda_offtarget: Weighting for offtarget penalty
        embedding_output_dir: Directory for temp embedding files
        utr5_length: Length of 5'UTR (if None, will try to infer)
        utr3_length: Length of 3'UTR (if None, will try to infer)
        embedding_cache: Optional dict to cache UTR sequence → embedding mappings (modified in-place)
        
    Returns:
        List of specificity scores (higher is better for target cell)
    """
    if len(sequences) == 0:
        return []

    if embedding_cache is None:
        embedding_cache = {}

    # Extract UTRs from full sequences (match training preprocessing)
    logger.debug(f"Extracting UTRs from {len(sequences)} sequences for scoring (training used UTR-only)")
    utr_sequences = []
    for seq in sequences:
        try:
            utr5, cds, utr3 = split_transcript_regions(seq, utr5_length, utr3_length)
            # concatenate UTRs without CDS (same as training)
            utr_seq = utr5 + utr3
            utr_sequences.append(utr_seq)
        except Exception as e:
            logger.warning(f"Failed to extract UTRs from sequence: {e}. Using full sequence.")
            utr_sequences.append(seq)

    # Check cache for embeddings (avoid recomputing)
    cached_count = sum(1 for utr in utr_sequences if utr in embedding_cache)
    logger.debug(f"Embedding cache hit: {cached_count}/{len(utr_sequences)} sequences")
    
    # Identify sequences that need embedding
    sequences_to_embed = []
    indices_to_embed = []
    for idx, utr in enumerate(utr_sequences):
        if utr not in embedding_cache:
            sequences_to_embed.append(utr)
            indices_to_embed.append(idx)
    
    # Generate embeddings only for uncached sequences (saves subprocess calls)
    if sequences_to_embed:
        logger.debug(f"Computing embeddings for {len(sequences_to_embed)} new sequences (cached {cached_count})")
        new_embeddings = generate_rna_embeddings(sequences_to_embed, embedding_output_dir)
        
        # Store in cache
        for idx, new_emb in zip(indices_to_embed, new_embeddings):
            embedding_cache[utr_sequences[idx]] = new_emb
    
    # Build final embeddings array from cache
    embeddings = np.zeros((len(utr_sequences), 640), dtype=np.float32)
    for idx, utr in enumerate(utr_sequences):
        embeddings[idx] = embedding_cache[utr]
    
    device = next(te_predictor.parameters()).device
    rna_emb = torch.FloatTensor(embeddings).to(device)

    target_cell_tensor = torch.full((len(sequences),), target_cell_id, dtype=torch.long, device=device)
    offtarget_cell_tensor = torch.full((len(sequences),), offtarget_cell_id, dtype=torch.long, device=device)

    with torch.no_grad():
        te_target = te_predictor(rna_emb, target_cell_tensor)
        te_offtarget = te_predictor(rna_emb, offtarget_cell_tensor)
        specificity = te_target - lambda_offtarget * te_offtarget

    logger.debug(f"Specificity scores - Target TE range: [{te_target.min():.3f}, {te_target.max():.3f}], " +
                f"Off-target TE range: [{te_offtarget.min():.3f}, {te_offtarget.max():.3f}]")
    return specificity.detach().cpu().numpy().tolist()


def generate_candidates_with_beam_search(base_sequence: str, te_predictor: TEPredictorModel,
                                       target_cell_id: int = 0, offtarget_cell_id: int = 1,
                                       beam_width: int = 100, num_iterations: int = 30,
                               utr5_length: Optional[int] = None,
                               utr3_length: Optional[int] = None,
                               lambda_offtarget: float = 1.0,
                               output_dir: Optional[Path] = None,
                               num_return: Optional[int] = None) -> List[str]:
    """
    Deterministically generate optimized UTR variants using beam search guided by TE predictor.
    
    Note: This is beam search/local search, not a genetic algorithm.
    No crossover or mutation operators; deterministic position-by-position refinement.
    
    Args:
        base_sequence: Full mRNA transcript (5'UTR + CDS + 3'UTR)
        te_predictor: Trained TE predictor model
        target_cell_id: Cell type ID to maximize TE
        offtarget_cell_id: Cell type ID to minimize TE
        beam_width: Number of top candidates to carry to next iteration
        num_iterations: Number of search iterations across UTR positions
        utr5_length: Length of 5'UTR
        utr3_length: Length of 3'UTR
        lambda_offtarget: Weight for off-target penalty
        output_dir: Directory to save search history
        
    Returns:
        List of optimized candidate sequences ranked by predicted specificity
    """
    if num_return is None:
        num_return = beam_width

    logger.info("\n" + "="*70)
    logger.info("BEAM SEARCH UTR OPTIMIZATION")
    logger.info("="*70)
    logger.info(f"Beam width: {beam_width}, Iterations: {num_iterations}")
    logger.info(f"Target cell: {target_cell_id}, Off-target cell: {offtarget_cell_id}")
    
    base_seq = base_sequence.upper().replace('T', 'U')
    utr5, cds, utr3 = split_transcript_regions(base_seq, utr5_length, utr3_length)
    
    logger.info(f"Base UTRs: 5'={len(utr5)}bp, 3'={len(utr3)}bp, CDS={len(cds)}bp")
    
    if len(utr5) == 0 and len(utr3) == 0:
        raise RuntimeError("UTR optimization requires non-empty UTR regions")

    alphabet = ['A', 'C', 'G', 'U'] if ('U' in base_seq and 'T' not in base_seq) else ['A', 'C', 'G', 'T']
    te_predictor.eval()
    beam: List[Tuple[str, str]] = [(utr5, utr3)]
    utr_total_len = len(utr5) + len(utr3)
    window_size = max(1, int(np.ceil(utr_total_len / max(num_iterations, 1))))

    search_history = []
    
    # Initialize embedding cache to avoid recomputing identical UTRs across iterations
    embedding_cache: Dict[str, np.ndarray] = {}
    logger.info(f"Embedding cache enabled for {num_iterations} iterations")

    # Deterministic beam search across UTR positions
    for iteration in range(num_iterations):
        start = iteration * window_size
        positions = list(range(start, min(start + window_size, utr_total_len)))
        if not positions:
            positions = list(range(utr_total_len))

        expanded_variants: List[Tuple[str, str]] = []
        seen = set()

        for cand_utr5, cand_utr3 in beam:
            key = (cand_utr5, cand_utr3)
            if key not in seen:
                expanded_variants.append((cand_utr5, cand_utr3))
                seen.add(key)

            for pos in positions:
                if pos < len(cand_utr5):
                    current_base = cand_utr5[pos]
                    for base in alphabet:
                        if base == current_base:
                            continue
                        new_utr5 = cand_utr5[:pos] + base + cand_utr5[pos + 1:]
                        new_key = (new_utr5, cand_utr3)
                        if new_key not in seen:
                            expanded_variants.append(new_key)
                            seen.add(new_key)
                else:
                    pos3 = pos - len(cand_utr5)
                    if pos3 < 0 or pos3 >= len(cand_utr3):
                        continue
                    current_base = cand_utr3[pos3]
                    for base in alphabet:
                        if base == current_base:
                            continue
                        new_utr3 = cand_utr3[:pos3] + base + cand_utr3[pos3 + 1:]
                        new_key = (cand_utr5, new_utr3)
                        if new_key not in seen:
                            expanded_variants.append(new_key)
                            seen.add(new_key)

        # Cap candidate count deterministically for runtime control
        max_eval = max(beam_width * 6, beam_width)
        expanded_variants = expanded_variants[:max_eval]

        full_sequences = [u5 + cds + u3 for u5, u3 in expanded_variants]
        fitness_scores = _predict_specificity_scores(
            full_sequences,
            te_predictor,
            target_cell_id,
            offtarget_cell_id,
            lambda_offtarget,
            output_dir or Path('/tmp'),
            utr5_length=len(utr5),
            utr3_length=len(utr3),
            embedding_cache=embedding_cache,
        )

        ranked = sorted(
            zip(expanded_variants, full_sequences, fitness_scores),
            key=lambda item: (-item[2], item[1]),
        )

        beam = [item[0] for item in ranked[:beam_width]]
        best_fitness = ranked[0][2]
        avg_fitness = float(np.mean([item[2] for item in ranked[:beam_width]]))
        std_fitness = float(np.std([item[2] for item in ranked[:beam_width]]))
        
        search_history.append({
            'iteration': iteration,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'std_fitness': std_fitness
        })
        
        if (iteration + 1) % 5 == 0 or iteration == 0:
            logger.info(f"Iter {iteration+1:3d} - Best fitness: {best_fitness:7.4f}, Avg: {avg_fitness:7.4f}")
        
    # Save search history if output_dir provided
    if output_dir:
        search_df = pd.DataFrame(search_history)
        search_df.to_csv(output_dir / 'beam_search_history.csv', index=False)
        logger.info(f"✓ Saved beam search history to {output_dir / 'beam_search_history.csv'}")

    # Final scoring and ranking
    final_sequences = [u5 + cds + u3 for u5, u3 in beam]
    final_scores = _predict_specificity_scores(
        final_sequences,
        te_predictor,
        target_cell_id,
        offtarget_cell_id,
        lambda_offtarget,
        output_dir or Path('/tmp'),
        utr5_length=len(utr5),
        utr3_length=len(utr3),
        embedding_cache=embedding_cache,
    )

    ranked_final = sorted(
        zip(final_sequences, final_scores),
        key=lambda item: (-item[1], item[0]),
    )
    candidates = [seq for seq, _ in ranked_final[:num_return]]

    logger.info(f"✓ Deterministic optimization complete!")
    logger.info(f"  Best candidate fitness: {ranked_final[0][1]:.4f}")
    logger.info(f"  Embedding cache: {len(embedding_cache)} unique UTR sequences cached")
    logger.info("="*70 + "\n")


class Config:
    """Configuration class for the pipeline."""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize with default or custom config."""
        
        # Model
        self.rna_embedding_dim = 640
        self.protein_embedding_dim = 320
        self.style_dim = 256
        self.fusion_dim = 256
        self.cell_dim = 128
        self.num_cells = 2  # target and offtarget
        
        # Training
        self.lr_main = 1e-3
        self.lr_finetune = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs = 100
        self.batch_size = 32
        self.early_stopping_patience = 10
        self.alpha_rank = 0.1
        
        # Data
        self.test_split = 0.15
        self.holdout_proteins = True
        self.max_seq_length = 5000
        
        # Generation
        self.num_candidates = 500
        self.beta_style = 0.5
        self.lambda_offtarget = 1.0
        self.K_prefilter = 200
        self.N_return = 20
        
        # Update with custom config if provided
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class RNADataset(torch.utils.data.Dataset):
    """PyTorch Dataset for RNA sequences."""
    
    def __init__(self, rna_embeddings: np.ndarray, protein_embeddings: np.ndarray,
                 cell_ids: np.ndarray, te_values: np.ndarray):
        """
        Args:
            rna_embeddings: (N, rna_dim)
            protein_embeddings: (N, protein_dim)
            cell_ids: (N,) - cell line indices
            te_values: (N,) - translation efficiency values
        """
        self.rna_emb = torch.FloatTensor(rna_embeddings)
        self.protein_emb = torch.FloatTensor(protein_embeddings)
        self.cell_ids = torch.LongTensor(cell_ids)
        self.te = torch.FloatTensor(te_values)
    
    def __len__(self):
        return len(self.rna_emb)
    
    def __getitem__(self, idx):
        return {
            'rna': self.rna_emb[idx],
            'protein': self.protein_emb[idx],
            'cell': self.cell_ids[idx],
            'te': self.te[idx]
        }


class StyleModel(nn.Module):
    """Style Model for cell-type embeddings."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.rna_proj = nn.Sequential(
            nn.Linear(config.rna_embedding_dim, config.style_dim),
            nn.LayerNorm(config.style_dim),
            nn.GELU()
        )
        
        self.cell_embeddings = nn.Embedding(config.num_cells, config.style_dim)
        self.cell_bias = nn.Embedding(config.num_cells, 1)
    
    def forward(self, rna_emb: torch.Tensor, cell_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_rna = self.rna_proj(rna_emb)
        v_cell = self.cell_embeddings(cell_ids)
        te_pred = (z_rna * v_cell).sum(dim=1) + self.cell_bias(cell_ids).squeeze(1)
        
        return te_pred, z_rna, v_cell
    
    def get_style_delta(self, target_cell_id: int, offtarget_cell_id: int) -> torch.Tensor:
        """Compute style direction."""
        device = self.cell_embeddings.weight.device
        target = self.cell_embeddings(torch.tensor([target_cell_id], device=device))
        offtarget = self.cell_embeddings(torch.tensor([offtarget_cell_id], device=device))
        return (target - offtarget).squeeze(0)


class JudgeModel(nn.Module):
    """Judge Model for accurate TE prediction."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Projections
        self.rna_proj = nn.Sequential(
            nn.Linear(config.rna_embedding_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.protein_proj = nn.Sequential(
            nn.Linear(config.protein_embedding_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.cell_embeddings = nn.Embedding(config.num_cells, config.cell_dim)
        self.cell_proj = nn.Linear(config.cell_dim, config.fusion_dim)
        
        # Fusion MLP
        fusion_input_size = config.fusion_dim * 3
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_size, config.fusion_dim * 2),
            nn.LayerNorm(config.fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config.fusion_dim * 2, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.te_head = nn.Linear(config.fusion_dim, 1)
    
    def forward(self, rna_emb: torch.Tensor, protein_emb: torch.Tensor,
                cell_ids: torch.Tensor) -> torch.Tensor:
        z_rna = self.rna_proj(rna_emb)
        z_prot = self.protein_proj(protein_emb)
        z_cell = self.cell_proj(self.cell_embeddings(cell_ids))
        
        fusion_input = torch.cat([z_rna, z_prot, z_cell], dim=1)
        fusion_output = self.fusion_mlp(fusion_input)
        
        te_pred = self.te_head(fusion_output).squeeze(1)
        
        return te_pred


def train_style_model(model: StyleModel, train_loader: torch.utils.data.DataLoader,
                     val_loader: torch.utils.data.DataLoader, config: Config,
                     device: torch.device) -> Dict[str, List[float]]:
    """Train the Style model."""
    
    logger.info("Starting Style Model Training...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_main, weight_decay=config.weight_decay)
    criterion = nn.HuberLoss(delta=1.0)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            rna_emb = batch['rna'].to(device)
            cell_ids = batch['cell'].to(device)
            te_target = batch['te'].to(device)
            
            optimizer.zero_grad()
            te_pred, _, _ = model(rna_emb, cell_ids)
            loss = criterion(te_pred, te_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                rna_emb = batch['rna'].to(device)
                cell_ids = batch['cell'].to(device)
                te_target = batch['te'].to(device)
                
                te_pred, _, _ = model(rna_emb, cell_ids)
                loss = criterion(te_pred, te_target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    logger.info("Style Model training complete!")
    return history


def train_judge_model(model: JudgeModel, train_loader: torch.utils.data.DataLoader,
                     val_loader: torch.utils.data.DataLoader, config: Config,
                     device: torch.device) -> Dict[str, List[float]]:
    """Train the Judge model."""
    
    logger.info("Starting Judge Model Training...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_main, weight_decay=config.weight_decay)
    criterion = nn.HuberLoss(delta=1.0)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            rna_emb = batch['rna'].to(device)
            protein_emb = batch['protein'].to(device)
            cell_ids = batch['cell'].to(device)
            te_target = batch['te'].to(device)
            
            optimizer.zero_grad()
            te_pred = model(rna_emb, protein_emb, cell_ids)
            loss = criterion(te_pred, te_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                rna_emb = batch['rna'].to(device)
                protein_emb = batch['protein'].to(device)
                cell_ids = batch['cell'].to(device)
                te_target = batch['te'].to(device)
                
                te_pred = model(rna_emb, protein_emb, cell_ids)
                loss = criterion(te_pred, te_target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    logger.info("Judge Model training complete!")
    return history


def generate_candidates(base_sequence: str, num_total: int = 500,
                       utr5_length: Optional[int] = None,
                       utr3_length: Optional[int] = None) -> List[str]:
    """Legacy interface retained for compatibility; random mutation path removed."""
    raise RuntimeError(
        "Random candidate generation has been removed. "
        "Use deterministic TE predictor guided optimization instead."
    )


def score_candidates(candidates: List[str], style_model: StyleModel, judge_model: JudgeModel,
                    rna_embeddings: np.ndarray, protein_embeddings: np.ndarray,
                    target_cell_id: int, offtarget_cell_id: int,
                    config: Config, device: torch.device) -> pd.DataFrame:
    """Score candidates with Judge model using actual embeddings."""
    
    results = []
    device_no_grad = device if torch.cuda.is_available() else 'cpu'
    
    judge_model.eval()
    
    with torch.no_grad():
        for i, seq in enumerate(candidates):
            # Use actual candidate-specific embeddings
            rna_emb = torch.FloatTensor(rna_embeddings[i]).unsqueeze(0).to(device_no_grad)
            protein_emb = torch.FloatTensor(protein_embeddings[i]).unsqueeze(0).to(device_no_grad)
            
            target_cell_tensor = torch.tensor([target_cell_id], device=device_no_grad)
            offtarget_cell_tensor = torch.tensor([offtarget_cell_id], device=device_no_grad)
            
            # Predict TE
            te_target = judge_model(rna_emb, protein_emb, target_cell_tensor).item()
            te_offtarget = judge_model(rna_emb, protein_emb, offtarget_cell_tensor).item()
            
            specificity = te_target - config.lambda_offtarget * te_offtarget
            
            results.append({
                'rank': len(results) + 1,
                'seq_id': f'candidate_{i}',
                'sequence': seq,
                'TE_target': te_target,
                'TE_offtarget': te_offtarget,
                'specificity_score': specificity
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values('specificity_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def main():
    """Main pipeline execution."""
    
    parser = argparse.ArgumentParser(description='Serova mRNA Design Challenge Pipeline')
    parser.add_argument('--mode', choices=['train', 'generate', 'both'], default='generate',
                       help='Pipeline mode')
    parser.add_argument('--data', type=str, help='Path to training data (Excel/CSV)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (auto-generated with SLURM job ID if not specified)')
    parser.add_argument('--config', type=str, help='Config JSON file')
    parser.add_argument('--model-style', type=str, help='Path to trained style model')
    parser.add_argument('--model-judge', type=str, help='Path to trained judge model')
    parser.add_argument('--base-sequence', type=str, help='Base mRNA transcript sequence (with UTRs)')
    parser.add_argument('--base-protein', type=str, default=None, 
                       help='Base protein sequence (if not provided, will translate from CDS)')
    parser.add_argument('--utr5-length', type=int, default=None,
                       help='Length of 5\' UTR in base sequence')
    parser.add_argument('--utr3-length', type=int, default=None,
                       help='Length of 3\' UTR in base sequence')
    parser.add_argument('--target-cell', type=str, default='TE_neurons',
                       help='Target cell type column')
    parser.add_argument('--offtarget-cell', type=str, default='TE_fibroblast',
                       help='Off-target cell type column')
    parser.add_argument('--ribonn-species', type=str, default='human', choices=['human', 'mouse'],
                       help='Species model for RiboNN prediction')
    parser.add_argument('--ribonn-topk', type=int, default=5,
                       help='Top-k models per fold to average in RiboNN')
    parser.add_argument('--train-te-predictor', action='store_true',
                       help='Deprecated: generation mode does not train TE predictor; use train_te_predictor.py')
    parser.add_argument('--te-predictor-model', type=str, default=None,
                       help='Path to trained TE predictor checkpoint (.pt)')
    parser.add_argument('--use-ga', action='store_true', default=True,
                       help='Deprecated flag kept for compatibility; deterministic guided optimization is always used')
    parser.add_argument('--beam-width', type=int, default=100,
                       help='Beam width for beam search optimization')
    parser.add_argument('--num-iterations', type=int, default=30,
                       help='Number of iterations for beam search optimization')
    parser.add_argument('--training-data', type=str, default=None,
                       help='Optional fallback training data path to derive cell mapping if missing in checkpoint')
    
    args = parser.parse_args()
    
    
    # Setup output directory with SLURM context
    if args.output_dir is None:
        output_dir = Path(__file__).parent / 'results' / f'job_{SLURM_JOB_ID}'
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load config
    config = Config()
    if args.config:
        try:
            with open(args.config) as f:
                config_dict = json.load(f)
                config = Config(config_dict)
                logger.info(f"Loaded config from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
    
    # GPU setup
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    logger.info(f"Using device: {device}")
    
    if cuda_available:
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    
    # Training mode
    if args.mode in ['train', 'both']:
        if not args.data:
            logger.error("Training requires --data argument")
            sys.exit(1)
        
        try:
            logger.info(f"Loading data from {args.data}")
            if args.data.endswith('.xlsx') or args.data.endswith('.xls'):
                df = pd.read_excel(args.data, sheet_name=0)
                logger.info(f"Loaded Excel file: {args.data} (sheet 0)")
            else:
                df = pd.read_csv(args.data)
                logger.info(f"Loaded CSV file: {args.data}")
            
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()[:10]}...")
            
            # Log data preparation would occur here
            logger.info("Training mode: Models would be trained on actual embeddings")
            logger.info(f"Target cell: {args.target_cell}")
            logger.info(f"Off-target cell: {args.offtarget_cell}")
            logger.info(f"Config: {json.dumps(config.to_dict(), indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            sys.exit(1)
    
    # Generation mode
    if args.mode in ['generate', 'both']:
        if not args.base_sequence:
            fallback_path = Path(__file__).parent / 'raw_data' / 'POU5f1_transcript.txt'
            logger.warning("No --base-sequence provided; using POU5F1 transcript fallback")
            logger.info(f"Fallback transcript file: {fallback_path}")
            try:
                with open(fallback_path, 'r') as f:
                    fallback_lines = [line.strip() for line in f if line.strip() and not line.startswith('>')]
                fallback_sequence = ''.join(fallback_lines).upper()
                fallback_sequence = ''.join(ch for ch in fallback_sequence if ch in {'A', 'C', 'G', 'T', 'U'})

                if not fallback_sequence:
                    raise ValueError("Fallback transcript is empty after FASTA parsing")

                args.base_sequence = fallback_sequence
                if args.utr5_length is None:
                    args.utr5_length = 62
                if args.utr3_length is None:
                    args.utr3_length = 267

                logger.info(f"✓ Loaded fallback transcript ({len(args.base_sequence)} bp)")
                logger.info(f"Using fallback UTR lengths: 5'UTR={args.utr5_length}bp, 3'UTR={args.utr3_length}bp")
            except Exception as e:
                logger.error(f"Failed to load fallback transcript: {e}")
                logger.error("Please provide --base-sequence or fix fallback file")
                sys.exit(1)
        
        try:
            logger.info("Setting up generation pipeline...")
            logger.info(f"Base sequence length: {len(args.base_sequence)} bp")
            
            # Determine UTR lengths and extract CDS
            if args.utr5_length is not None and args.utr3_length is not None:
                logger.info(f"Using provided UTR lengths: 5'UTR={args.utr5_length}bp, 3'UTR={args.utr3_length}bp")
                utr5, cds, utr3 = split_transcript_regions(args.base_sequence, args.utr5_length, args.utr3_length)
            else:
                logger.warning("UTR lengths not provided, attempting auto-detection...")
                logger.warning("For better accuracy, provide --utr5-length and --utr3-length")
                utr5, cds, utr3 = split_transcript_regions(args.base_sequence)
            
            logger.info(f"Transcript regions: 5'UTR={len(utr5)}bp, CDS={len(cds)}bp, 3'UTR={len(utr3)}bp")
            
            # Get base protein (translate if not provided)
            if args.base_protein is None:
                logger.info("No --base-protein provided, translating from CDS...")
                args.base_protein = translate_to_protein(cds, start_pos=0)
                logger.info(f"✓ Translated protein from CDS: {len(args.base_protein)} aa")
            else:
                logger.info(f"Using provided protein: {len(args.base_protein)} aa")
            
            # Step 0: Load TE predictor checkpoint (required, no in-generation training)
            if not args.te_predictor_model:
                raise RuntimeError(
                    "--te-predictor-model is required for generation. "
                    "Train once with train_te_predictor.py and reuse the saved checkpoint."
                )

            model_path = Path(args.te_predictor_model)
            if not model_path.exists():
                raise RuntimeError(f"TE predictor checkpoint not found: {model_path}")

            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    num_cells = int(checkpoint.get('num_cells', state_dict['cell_embeddings.weight'].shape[0]))
                    cell_to_idx = checkpoint.get('cell_to_idx')
                else:
                    state_dict = checkpoint
                    num_cells = state_dict['cell_embeddings.weight'].shape[0]
                    cell_to_idx = None

                logger.info(f"Detected {num_cells} cell types from model checkpoint")

                te_predictor = TEPredictorModel(num_cells=num_cells)
                te_predictor.load_state_dict(state_dict)
                te_predictor = te_predictor.to(device)
                te_predictor.eval()
                logger.info(f"✓ Loaded TE predictor from {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load TE predictor checkpoint '{model_path}': {e}")

            # If checkpoint lacks cell mapping, fallback to training-data-derived mapping
            if cell_to_idx is None:
                training_data_path = args.training_data or (Path(__file__).parent / 'raw_data' / '41587_2025_2712_MOESM3_ESM.xlsx')
                if not training_data_path.exists():
                    raise RuntimeError(
                        "Checkpoint does not include cell mapping and training data is unavailable. "
                        "Provide --training-data or retrain with train_te_predictor.py."
                    )
                logger.warning("Checkpoint missing cell mapping; deriving it from training data")
                cell_to_idx = extract_cell_to_idx_mapping(training_data_path)

            # Step 1: Generate candidate sequences (deterministic guided optimization)
            logger.info("Using deterministic TE-guided UTR optimization for candidate generation...")

            logger.info(f"Available cell types for generation: {len(cell_to_idx)}")
            
            # Get cell IDs from mapping
            if args.target_cell in cell_to_idx and args.offtarget_cell in cell_to_idx:
                target_cell_id = cell_to_idx[args.target_cell]
                offtarget_cell_id = cell_to_idx[args.offtarget_cell]
                logger.info(f"Using cell indices: {args.target_cell}={target_cell_id}, {args.offtarget_cell}={offtarget_cell_id}")
            else:
                logger.warning(f"Requested cells not in mapping. Using alternative cells.")
                available = [k for k in sorted(cell_to_idx.keys())]
                target_cell_id = cell_to_idx.get(available[0], 0) if available else 0
                offtarget_cell_id = cell_to_idx.get(available[1], 1) if len(available) > 1 else 0
                logger.warning(f"Using fallback cells: index {target_cell_id} and {offtarget_cell_id}")


            candidates = generate_candidates_with_beam_search(
                args.base_sequence,
                te_predictor,
                target_cell_id=target_cell_id,
                offtarget_cell_id=offtarget_cell_id,
                beam_width=args.beam_width,
                num_iterations=args.num_iterations,
                utr5_length=args.utr5_length,
                utr3_length=args.utr3_length,
                lambda_offtarget=config.lambda_offtarget,
                output_dir=output_dir,
                num_return=config.num_candidates,
            )
            logger.info(f"✓ Generated {len(candidates)} candidate sequences via beam search optimization")
            proteins = []
            utr5_len = args.utr5_length if args.utr5_length is not None else len(utr5)
            
            for seq in candidates:
                try:
                    # Extract CDS from each candidate using the established UTR lengths
                    if args.utr5_length is not None and args.utr3_length is not None:
                        cand_utr5, cand_cds, cand_utr3 = split_transcript_regions(seq, args.utr5_length, args.utr3_length)
                    else:
                        cand_utr5, cand_cds, cand_utr3 = split_transcript_regions(seq)
                    
                    prot = translate_to_protein(cand_cds, start_pos=0)
                    if len(prot) < 5:  # If translation yielded very short protein, use base
                        prot = args.base_protein
                    proteins.append(prot)
                except Exception as e:
                    logger.warning(f"Translation failed for candidate, using base protein: {e}")
                    proteins.append(args.base_protein)
            logger.info(f"✓ Translated {len(proteins)} proteins (avg length: {np.mean([len(p) for p in proteins]):.1f} aa)")
            
            # Step 3: Predict TE with RiboNN (primary ranking path)
            ribonn_scored = False
            logger.info("\n" + "="*70)
            logger.info("RIBONN PREDICTION")
            logger.info("="*70)

            try:
                results = predict_te_with_ribonn(
                    candidates=candidates,
                    output_dir=output_dir,
                    target_cell=args.target_cell,
                    offtarget_cell=args.offtarget_cell,
                    species=args.ribonn_species,
                    top_k_models=args.ribonn_topk,
                    utr5_length=args.utr5_length,
                    utr3_length=args.utr3_length,
                )
                results['specificity_score'] = results['TE_target'] - config.lambda_offtarget * results['TE_offtarget']
                results = results.sort_values('specificity_score', ascending=False).reset_index(drop=True)
                results['rank'] = range(1, len(results) + 1)
                ribonn_scored = True
                logger.info("✓ Scored candidates with RiboNN predicted TE")
                logger.info("="*70 + "\n")
            except Exception as e:
                logger.error(f"RiboNN scoring failed: {e}")
                logger.warning("Falling back to embedding/model scoring...")

            # Step 4 (fallback): Generate embeddings and score with Judge model/random
            if not ribonn_scored:
                logger.info("\n" + "="*70)
                logger.info("EMBEDDING GENERATION (FALLBACK)")
                logger.info("="*70)
            
                try:
                    # Generate RNA embeddings with RNA-FM
                    rna_embeddings = generate_rna_embeddings(candidates, output_dir)
                    logger.info(f"✓ RNA embeddings shape: {rna_embeddings.shape}")

                    # Generate protein embeddings with ESM2
                    protein_embeddings = generate_protein_embeddings(proteins, output_dir)
                    logger.info(f"✓ Protein embeddings shape: {protein_embeddings.shape}")

                    embeddings_generated = True
                    logger.info("="*70 + "\n")

                except Exception as e:
                    logger.error(f"Embedding generation failed: {e}")
                    logger.warning("Embedding-based fallback unavailable")
                    embeddings_generated = False

                # Step 5 (fallback): Score candidates
                if embeddings_generated and args.model_judge and Path(args.model_judge).exists():
                    logger.info("Loading trained Judge model for scoring...")
                    try:
                        # Load model
                        judge_model = JudgeModel(config).to(device)

                        checkpoint = torch.load(args.model_judge, map_location=device)
                        judge_model.load_state_dict(checkpoint['model_state_dict'])
                        judge_model.eval()

                        logger.info("✓ Judge model loaded successfully")

                        # Cell IDs (neurons=0, fibroblast=1)
                        target_cell_id = 0
                        offtarget_cell_id = 1

                        # Score with model
                        style_model = None
                        results = score_candidates(
                            candidates,
                            style_model,
                            judge_model,
                            rna_embeddings,
                            protein_embeddings,
                            target_cell_id,
                            offtarget_cell_id,
                            config,
                            device
                        )
                        logger.info("✓ Scored candidates with trained model")

                    except Exception as e:
                        logger.error(f"Model scoring failed: {e}")
                        embeddings_generated = False

                # Final fallback removed: do not use random scores
                if not embeddings_generated or not (args.model_judge and Path(args.model_judge).exists()):
                    if not embeddings_generated:
                        raise RuntimeError("Scoring failed: embeddings are not available")
                    else:
                        raise RuntimeError(
                            "Scoring failed: trained Judge model not available. "
                            f"Expected model at: {args.model_judge}"
                        )
            
            # Step 6: Save results
            top_20 = results.head(20)[['rank', 'seq_id', 'TE_target', 'TE_offtarget', 'specificity_score']]
            top_20.to_csv(output_dir / 'top_20_candidates.csv', index=False)
            logger.info(f"✓ Saved top 20 candidates to {output_dir / 'top_20_candidates.csv'}")
            
            results.to_csv(output_dir / 'all_candidates_ranked.csv', index=False)
            logger.info(f"✓ Saved all candidates to {output_dir / 'all_candidates_ranked.csv'}")
            
            # Save best sequence FASTA
            best_seq = results.iloc[0]['sequence']
            with open(output_dir / 'best_sequence.fasta', 'w') as f:
                f.write(f">best_candidate_rank1 specificity={results.iloc[0]['specificity_score']:.4f}\n")
                f.write(f"{best_seq}\n")
            logger.info(f"✓ Saved best sequence to {output_dir / 'best_sequence.fasta'}")
            
            # Summary
            logger.info("\n" + "="*70)
            logger.info("GENERATION SUMMARY")
            logger.info("="*70)
            logger.info(f"Total candidates generated: {len(candidates)}")
            logger.info(f"Best candidate specificity score: {results.iloc[0]['specificity_score']:.4f}")
            logger.info(f"Best target TE: {results.iloc[0]['TE_target']:.4f}")
            logger.info(f"Best off-target TE: {results.iloc[0]['TE_offtarget']:.4f}")
            logger.info(f"Results saved to: {output_dir}")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            sys.exit(1)
    
    logger.info(f"Pipeline completed successfully!")
    logger.info(f"SLURM Job ID: {SLURM_JOB_ID}")
    logger.info(f"Results: {output_dir}")


if __name__ == '__main__':
    main()
