#!/usr/bin/env python3
"""
eToehold mRNA Designer - Serova Challenge
==========================================

Computational design of mRNA sequences with toehold switch logic gates
for cell-type specific translation control.

Objective: Generate an mRNA sequence delivering the POU5F1 gene with an
eToehold logic gate in the 5' UTR that remains closed in liver (off-target)
but opens in neurons (target).

Phases:
    1. Trigger RNA Selection (HPA Dataset)
    2. Sequence Initialization
    3. ViennaRNA Structure Prediction
    4. Evolutionary Search Loop (Genetic Algorithm)

Usage:
    sbatch etoehold_submit.sh
    python etoehold_pipeline.py --output-dir results/run1 --generations 100
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from datetime import datetime
import os
import sys
import random
import requests
import zipfile
import io
import subprocess
from dataclasses import dataclass
from Bio import SeqIO
from Bio.Seq import Seq

# Setup logging
def setup_logger(output_dir: Path):
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
    
    # File handler
    log_file = output_dir / 'etoehold_pipeline.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

# Global seed for reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ==============================================================================
# Phase 1: Trigger RNA Selection (HPA Dataset)
# ==============================================================================

def load_local_transcript_data(data_dir: Path, logger: logging.Logger) -> Path:
    """Load local transcript tissue expression data."""
    logger.info("Phase 1: Loading local transcript tissue expression data...")
    
    tsv_path = data_dir / "transcript_rna_tissue.tsv"
    
    if not tsv_path.exists():
        logger.error(f"Transcript data not found at {tsv_path}")
        raise FileNotFoundError(f"Transcript data file not found: {tsv_path}")
    
    logger.info(f"✓ Loaded transcript data from {tsv_path}")
    return tsv_path


def load_pou5f1_sequence(data_dir: Path, logger: logging.Logger) -> Tuple[str, int, int]:
    """Load POU5F1 transcript sequence with known UTR boundaries.
    
    Returns:
        (full_transcript, utr5_len, utr3_len)
    """
    logger.info("Phase 1: Loading POU5F1 transcript sequence...")
    
    seq_path = data_dir / "POU5f1_transcript.txt"
    
    if not seq_path.exists():
        logger.error(f"POU5F1 sequence not found at {seq_path}")
        raise FileNotFoundError(f"POU5F1 sequence file not found: {seq_path}")
    
    # Read FASTA or plain text
    with open(seq_path, 'r') as f:
        lines = f.readlines()
    
    # If FASTA format, skip header
    if lines[0].startswith('>'):
        sequence = ''.join(line.strip() for line in lines[1:])
    else:
        sequence = ''.join(line.strip() for line in lines)
    
    # Convert to RNA
    sequence = sequence.upper().replace('T', 'U')
    
    logger.info(f"✓ Loaded POU5F1 sequence: {len(sequence)}nt")
    logger.info(f"  5' UTR: 62 bp, 3' UTR: 267 bp")
    
    return sequence, 62, 267


def filter_neuron_specific_transcripts(tsv_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Filter transcript data to find neuron-specific transcripts using only TPM columns.
    
    Returns:
        DataFrame with top candidates sorted by specificity score
    """
    logger.info("Filtering for neuron-specific transcripts (using TPM columns only)...")
    
    df = pd.read_csv(tsv_path, sep='\t')
    logger.info(f"Loaded {len(df)} transcript entries")
    
    # Get column names
    cols = df.columns.tolist()
    logger.info(f"Total columns: {len(cols)}")
    
    # Select only TPM columns (exclude est_counts columns)
    tpm_cols = [col for col in cols if col.startswith('TPM.')]
    
    if not tpm_cols:
        logger.error(f"No TPM columns found. Available: {cols[:10]}")
        raise ValueError("No TPM expression columns found")
    
    logger.info(f"Found {len(tpm_cols)} TPM columns")
    
    # Extract tissue types from TPM column names
    # Column format: "TPM.{tissue}.{sample}"
    def extract_tissue(col_name: str) -> str:
        """Extract tissue name from TPM column name."""
        # Remove 'TPM.' prefix
        parts = col_name[4:].rsplit('.', 1)
        return parts[0] if parts else col_name
    
    tissue_map = {col: extract_tissue(col) for col in tpm_cols}
    unique_tissues = set(tissue_map.values())
    logger.info(f"Found {len(unique_tissues)} unique tissues")
    
    # Identify neuron and liver tissues from TPM columns
    neuron_tpm_cols = [col for col in tpm_cols if any(x in tissue_map[col].lower() for x in ['brain', 'cortex', 'neuron', 'hippocampus'])]
    liver_tpm_cols = [col for col in tpm_cols if 'liver' in tissue_map[col].lower()]
    
    logger.info(f"Neuron TPM columns: {len(neuron_tpm_cols)}")
    logger.info(f"Liver TPM columns: {len(liver_tpm_cols)}")
    
    if neuron_tpm_cols:
        logger.info(f"  Neuron tissues: {sorted(set(tissue_map[col] for col in neuron_tpm_cols))}")
    if liver_tpm_cols:
        logger.info(f"  Liver tissues: {sorted(set(tissue_map[col] for col in liver_tpm_cols))}")
    
    # Calculate max expression in neurons and liver
    df['neuron_expr'] = df[neuron_tpm_cols].max(axis=1) if neuron_tpm_cols else 0
    df['liver_expr'] = df[liver_tpm_cols].max(axis=1) if liver_tpm_cols else 0
    
    # Calculate specificity score
    df['specificity'] = df['neuron_expr'] - df['liver_expr']
    df = df.sort_values('specificity', ascending=False)
    
    logger.info(f"Top 5 candidates:")
    for idx, row in df.head(5).iterrows():
        logger.info(f"  Transcript {row['enstid']}: neuron={row['neuron_expr']:.2f}, liver={row['liver_expr']:.2f}, score={row['specificity']:.2f}")
    
    return df


def calculate_volcano_statistics(tsv_path: Path, output_dir: Path, logger: logging.Logger) -> List[Dict]:
    """
    Calculate volcano plot statistics using proper statistical testing.
    
    Performs independent samples t-tests to calculate p-values for each gene,
    comparing neuron vs liver expression. Results are saved to volcano_data.json
    and returned for use in trigger selection.
    
    Args:
        tsv_path: Path to transcript TSV file
        output_dir: Output directory for results
        logger: Logger instance
    
    Returns:
        List of volcano statistics dictionaries
    """
    from scipy import stats
    
    logger.info("="*70)
    logger.info("Computing Differential Expression Statistics (t-tests)")
    logger.info("="*70)
    
    # Load transcript data
    df = pd.read_csv(tsv_path, sep='\t')
    logger.info(f"Loaded {len(df)} transcripts for analysis")
    
    # Get TPM columns
    tpm_cols = [col for col in df.columns if col.startswith('TPM.')]
    
    # Extract tissue from column name
    def extract_tissue(col_name: str) -> str:
        parts = col_name[4:].rsplit('.', 1)
        return parts[0] if parts else col_name
    
    tissue_map = {col: extract_tissue(col) for col in tpm_cols}
    
    # Identify neuron and liver columns
    neuron_cols = [col for col in tpm_cols if any(x in tissue_map[col].lower() for x in ['brain', 'cortex', 'neuron', 'hippocampus'])]
    liver_cols = [col for col in tpm_cols if 'liver' in tissue_map[col].lower()]
    
    logger.info(f"Using {len(neuron_cols)} neuron samples and {len(liver_cols)} liver samples for t-tests")
    
    # Calculate vectorized statistics
    neuron_expr = df[neuron_cols].values if neuron_cols else np.ones((len(df), 1))
    liver_expr = df[liver_cols].values if liver_cols else np.ones((len(df), 1))
    
    # Handle NaN values
    neuron_expr_clean = np.where(np.isnan(neuron_expr), 0.1, neuron_expr)
    liver_expr_clean = np.where(np.isnan(liver_expr), 0.1, liver_expr)
    
    # Calculate means
    neuron_mean = np.mean(neuron_expr_clean, axis=1)
    liver_mean = np.mean(liver_expr_clean, axis=1)
    
    # Calculate log2 fold change
    pseudo = 0.1
    log2_fc = np.log2((neuron_mean + pseudo) / (liver_mean + pseudo))
    
    # Calculate p-values using vectorized t-test
    pvalues = np.zeros(len(df))
    for i in range(len(df)):
        neuron_vals = neuron_expr_clean[i, :]
        liver_vals = liver_expr_clean[i, :]
        
        # Only perform t-test if we have sufficient data
        # Require at least 2 values per group with variance
        if len(neuron_vals) >= 2 and len(liver_vals) >= 2:
            try:
                # Check for sufficient variance (don't compute if all values are identical)
                if np.std(neuron_vals) > 0 or np.std(liver_vals) > 0:
                    t_stat, pval = stats.ttest_ind(neuron_vals, liver_vals)
                    # Handle edge cases (NaN or inf p-values)
                    if np.isnan(pval) or np.isinf(pval):
                        pvalues[i] = 1.0
                    else:
                        pvalues[i] = pval
                else:
                    # No variance - cannot compute meaningful t-test
                    pvalues[i] = 1.0
            except Exception as e:
                pvalues[i] = 1.0
        else:
            pvalues[i] = 1.0
        
        # Log progress every 50k genes
        if (i + 1) % 50000 == 0:
            logger.info(f"  Processed {i + 1}/{len(df)} genes...")
    
    # Ensure p-values are valid
    pvalues = np.clip(pvalues, 1e-300, 1.0)
    neg_log10_pval = -np.log10(pvalues)
    
    # Build results
    volcano_results = []
    for i in range(len(df)):
        row = df.iloc[i]
        ensgid = row.get('ensgid', f'ENSG_{i}')
        enstid = row.get('enstid', f'ENST_{i}')
        
        is_significant = (np.abs(log2_fc[i]) > 1.0) and (neg_log10_pval[i] > -np.log10(0.05))
        
        volcano_results.append({
            'ensgid': ensgid,
            'enstid': enstid,
            'neuron_mean_tpm': float(neuron_mean[i]),
            'liver_mean_tpm': float(liver_mean[i]),
            'log2_fold_change': float(log2_fc[i]),
            'pvalue': float(pvalues[i]),
            'neg_log10_pvalue': float(neg_log10_pval[i]),
            'significant': bool(is_significant)
        })
    
    # Save results to JSON
    volcano_file = output_dir / 'volcano_data.json'
    with open(volcano_file, 'w') as f:
        json.dump(volcano_results, f, indent=2)
    
    logger.info(f"✓ Saved volcano data: {volcano_file}")
    logger.info(f"  Total genes analyzed: {len(volcano_results)}")
    
    # Summary statistics
    sig_count = sum(1 for r in volcano_results if r['significant'])
    neuron_up = sum(1 for r in volcano_results if r['significant'] and r['log2_fold_change'] > 0)
    liver_up = sum(1 for r in volcano_results if r['significant'] and r['log2_fold_change'] < 0)
    
    logger.info(f"  Significant DETs (|FC|>1, p<0.05): {sig_count}")
    logger.info(f"    Neuron-enriched: {neuron_up}")
    logger.info(f"    Liver-enriched: {liver_up}")
    
    return volcano_results


def fetch_ensembl_sequence(gene_id: str, logger: logging.Logger) -> Optional[str]:
    """Fetch transcript sequence from Ensembl REST API."""
    logger.info(f"Fetching sequence for {gene_id} from Ensembl...")
    
    # Try Ensembl REST API
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{gene_id}?type=cdna"
    
    try:
        response = requests.get(server + ext, 
                              headers={"Content-Type": "application/json"},
                              timeout=30)
        if response.status_code == 200:
            data = response.json()
            sequence = data.get('seq', '')
            logger.info(f"✓ Retrieved sequence of length {len(sequence)}")
            return sequence
        else:
            logger.warning(f"Ensembl API returned status {response.status_code}")
    except Exception as e:
        logger.warning(f"Ensembl API error: {e}")
    
    return None


def extract_trigger_sequence(rna_sequence: str, length: int = 25, 
                            logger: Optional[logging.Logger] = None) -> str:
    """
    Extract a trigger sequence from the RNA.
    
    For simplicity, we'll extract a middle region that could represent
    a single-stranded loop region.
    """
    if logger:
        logger.info(f"Extracting {length}nt trigger sequence...")
    
    # Convert to RNA
    rna_seq = rna_sequence.upper().replace('T', 'U')
    
    # Extract from middle region (often less structured)
    if len(rna_seq) < length:
        trigger = rna_seq
    else:
        mid_point = len(rna_seq) // 2
        start = max(0, mid_point - length // 2)
        trigger = rna_seq[start:start + length]
    
    # Ensure it's exactly the right length
    if len(trigger) < length:
        trigger = trigger + 'A' * (length - len(trigger))
    elif len(trigger) > length:
        trigger = trigger[:length]
    
    if logger:
        logger.info(f"✓ Trigger sequence: {trigger}")
    
    return trigger


def select_trigger_rna(output_dir: Path, data_dir: Path, logger: logging.Logger, 
                      volcano_data: Optional[List[Dict]] = None) -> Tuple[str, str]:
    """
    Phase 1: Select neuron-specific trigger RNA based on volcano statistics.
    
    Uses differential expression analysis (volcano plot stats) to find the most
    significantly neuron-enriched transcript and extracts a 25nt trigger sequence.
    
    NOTE: This trigger will be inserted into the 5'UTR of POU5F1 (not the trigger gene itself).
    The trigger gene is selected only for its neuron-specificity context.
    
    Args:
        output_dir: Output directory for results
        data_dir: Data directory containing sequences
        logger: Logger instance
        volcano_data: Pre-calculated volcano statistics (from calculate_volcano_statistics)
    
    Returns:
        (trigger_gene_name, trigger_sequence_25nt)
    """
    logger.info("="*70)
    logger.info("PHASE 1: Trigger RNA Selection (From Neuron-Enriched Gene)")
    logger.info("="*70)
    
    if volcano_data is None:
        raise ValueError("Volcano statistics required for trigger selection")
    
    # Convert volcano data to DataFrame for easier filtering
    volcano_df = pd.DataFrame(volcano_data)
    
    # Select best trigger candidate:
    # - Must be neuron-enriched (log2_fc > 0)
    # - Must be statistically significant (p < 0.05)
    # - Ranked by combination of fold change and -log10(p-value)
    neuron_enriched = volcano_df[
        (volcano_df['log2_fold_change'] > 0) & 
        (volcano_df['neg_log10_pvalue'] > -np.log10(0.05))
    ].copy()
    
    if len(neuron_enriched) == 0:
        logger.warning("No significantly neuron-enriched genes found. Using top TPM-based candidate...")
        # Fallback to TPM filtering
        tsv_path = data_dir / "transcript_rna_tissue.tsv"
        candidates = filter_neuron_specific_transcripts(tsv_path, logger)
        top_candidate = candidates.iloc[0]
        gene_name = top_candidate['ensgid']
        transcript_id = top_candidate['enstid']
    else:
        # Calculate specificity score: fold change weighted by significance
        neuron_enriched['specificity_score'] = (
            neuron_enriched['log2_fold_change'] * neuron_enriched['neg_log10_pvalue']
        )
        neuron_enriched = neuron_enriched.sort_values('specificity_score', ascending=False)
        
        logger.info(f"Found {len(neuron_enriched)} significantly neuron-enriched transcripts")
        logger.info(f"Top 5 candidates by specificity:")
        for i, row in neuron_enriched.head(5).iterrows():
            logger.info(f"  {row['ensgid']}: FC={row['log2_fold_change']:.2f}, "
                       f"-log10(p)={row['neg_log10_pvalue']:.2f}, "
                       f"specificity={row['specificity_score']:.2f}")
        
        # Get top candidate's Ensembl ID and transcript ID
        top_result = neuron_enriched.iloc[0]
        gene_name = top_result['ensgid']
        transcript_id = top_result['enstid']
    
    logger.info(f"Selected trigger gene: {gene_name}")
    
    # Fetch sequence for the selected candidate using Ensembl
    full_sequence = fetch_ensembl_sequence(transcript_id, logger)
    
    if full_sequence is None:
        logger.warning(f"Failed to fetch {transcript_id} from Ensembl. Using default miR-124 trigger...")
        trigger_seq = "UAAGGCACGCGGUGAAUGCCAAGGG"
        gene_name = "default_miR-124"
    else:
        # Extract trigger from the sequence
        trigger_seq = extract_trigger_sequence(full_sequence, length=25, logger=logger)
    
    # Save trigger info to file
    trigger_file = output_dir / "trigger_sequence.txt"
    with open(trigger_file, 'w') as f:
        f.write(f"Trigger Gene (neuron-specific): {gene_name}\n")
        f.write(f"Trigger sequence (25nt): {trigger_seq}\n")
        f.write(f"Note: This trigger will be inserted into POU5F1 5'UTR via toehold switch\n")
    
    logger.info(f"✓ Trigger sequence: {trigger_seq}")
    logger.info(f"✓ Phase 1 complete. Trigger will be delivered in POU5F1 mRNA.")
    
    return gene_name, trigger_seq


# ==============================================================================
# Phase 2: Sequence Initialization
# ==============================================================================

@dataclass
class mRNADesign:
    """Data class to hold mRNA design components."""
    utr5: str
    cds: str
    utr3: str
    trigger_seq: str
    
    @property
    def full_sequence(self) -> str:
        """Return the full mRNA sequence."""
        return self.utr5 + self.cds + self.utr3
    
    @property
    def atg_position(self) -> int:
        """Find the position of ATG start codon."""
        return len(self.utr5)


def get_pou5f1_cds() -> str:
    """
    Get the POU5F1 (OCT4) coding sequence.
    
    Using the human POU5F1 CDS from NCBI RefSeq.
    """
    # Human POU5F1 CDS (NM_002701.6) - first ~300nt for demonstration
    # In practice, use the full sequence
    pou5f1_cds = (
        "ATGGCGGGACACCTGGCTTCGGATTTCGCCTTCTCGCCCCCTCCAGGTGGTGGAGGTGATCTGCTTGGCGCCGCCAGCAG"
        "CTTGTGCCGCGGCACCCGCTCCCCCCGCCCCCTCCGGGCTCCCCGAGCCGGCAGGGAAGCGGCGGCGGGACCCCGCCGTT"
        "CGCCCCAGCCCACCACCAGCACAGGAGCAGCCACCATGGCGCACCCGCAGCACCTGCTCCACCGACTTGCCGGGGCTCTC"
        "CTGGAGGGCCAGGAATCGGGCCGGGGGTTGGCCCCGCGGTTGGCCCCTGGCTCCCGAGGAGCTCGTGCAGCAGATCACTC"
        "ACATCGCCAATCAGCTTGGGCTCGAGAAGGATGTGGTCCGAGTGTGGTTCTGTAACCGGCGCCAGAAGGGCAAGCGATCA"
        "AGCAGCGACTATGCACAACGAGAGGATTTTGAGGCTGCTGGGTCTCCTTTCTCAGGGGGACCAGTGTCCTTTCCTCTGGG"
        "TCGTTTTCCCTGAACCTGCCCCCATATTTCCCAACCTGGTTTTCCAAAACCCTGGCTGCCCGAAAGGGGGGTTAAAGGCA"
        "GCCTTTGCCTCCCTGCCCTCTCTGGGTCCCCCATGTCTGCCCCCCTGGAGGTGCAGCCCCTGCTCCGGACTGGGCCCGTG"
        "CAGGGTCTGGAGCGGCCCTGATGGGGTGGGCCCTTGGGGCTGGGTGCCCCCACCATTCCTAGGTGGTGGGCTCCGGTCCC"
        "CGGGGCCCCACCCCCGTCCCAGCCTCCCAGTCCCCCTCTCTGCCCCCGTATGAGTTCTGCAGGGCTTTCATGTCCTGGGA"
        "CTCCACCTCCACACCTCTGCATCTGAGAAGAATGAAAATCTTCAGGAGATATGCAAAGCAGAAACCCTCGGGGCTCTGGG"
        "CCGCGGGGAGCGGGGTTGGGCCAGAATGGCTGGACACCTGGCTTCAGATTTGCCTTTCTGCGGCAGGTATGCTGGTCTCT"
        "TCTGGGCTCTCTGGGTGCTGCTCTGCTCTGCCCCATCCCCACCCTGCTGTGCGGGGGACAGCTGCCCCAGACTGCAGACT"
        "CTGCTCTGCTCTAACTCCGAGCTGGCCCGGGGTCTACTTAGTTGCTGCGAAATGA"
    )
    
    # Convert to RNA and ensure it starts with AUG
    rna_cds = pou5f1_cds.upper().replace('T', 'U')
    if not rna_cds.startswith('AUG'):
        # Find AUG
        atg_pos = rna_cds.find('AUG')
        if atg_pos > 0:
            rna_cds = rna_cds[atg_pos:]
    
    # Ensure it ends with a stop codon
    if not rna_cds.endswith(('UAA', 'UAG', 'UGA')):
        rna_cds += 'UAA'
    
    return rna_cds


def generate_complement(sequence: str) -> str:
    """Generate the Watson-Crick complement of an RNA sequence."""
    complement_map = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement_map.get(base, base) for base in sequence)


def design_toehold_utr(trigger_seq: str, logger: logging.Logger) -> str:
    """
    Design a 5' UTR with a toehold switch that buries the start codon.
    
    Structure:
        [Toehold region] [Hairpin stem] [ATG in loop] [Hairpin stem complement]
    
    The toehold region is complementary to the trigger sequence.
    When the trigger binds, it disrupts the hairpin and exposes the ATG.
    """
    logger.info("Designing toehold 5' UTR...")
    
    # Reverse complement of trigger for toehold binding region
    trigger_rc = generate_complement(trigger_seq[::-1])
    
    # Design hairpin that sequesters the ATG
    # Structure: 5'-[toehold]-[stem1]-ATG-[stem1complement]-3'
    
    # Create a stable hairpin stem (6-8 bp)
    # We want: stem1 - loop with ATG - stem1_complement
    stem1 = "GCGCGC"  # 6bp stem
    loop = "UUUU"     # Small loop
    stem1_comp = generate_complement(stem1[::-1])
    
    # Build the hairpin structure
    # The ATG should be buried in the hairpin
    # For a toehold switch, we place the toehold region before the hairpin
    
    # Basic toehold structure:
    # 5'-[toehold complementary to trigger]-[linker]-[stem]-ATG-[stem_complement]-3'
    
    linker = "AAA"
    
    # We'll create a hairpin that includes the ATG in the stem region
    # so it's base-paired and hidden
    
    # More sophisticated design:
    # Create a stem-loop where ATG is base-paired within the stem
    # The trigger binding will compete for these bases
    
    # Simplified approach: 
    # [Toehold (RC of trigger)] - [Hairpin stem] - [ATG sequestered] - [Hairpin complement]
    
    # For the toehold to work, part of the trigger complement should overlap
    # with the hairpin structure
    
    # Let's create a hairpin with the ATG base-paired:
    # We need bases that pair with ATG: CAU (complement of AUG)
    
    # Design: [toehold_region][stem_before]ATG[stem_after]
    # where stem_before + ATG + stem_after forms a hairpin
    
    stem_before = "GCGCGCGC"  # 8 bases before ATG
    atg = "AUG"
    stem_after = "CGCGCGCG"   # Complement of stem_before (reversed)
    
    # The toehold region (complementary to trigger)
    toehold_region = trigger_rc[:15]  # Use first 15nt
    
    # Add a spacer
    spacer = "AAA"
    
    # Combine
    utr5 = toehold_region + spacer + stem_before + atg + stem_after
    
    logger.info(f"✓ Designed 5' UTR ({len(utr5)}nt): {utr5[:50]}...")
    logger.info(f"  Toehold region (15nt): {toehold_region}")
    logger.info(f"  Hairpin with ATG: {stem_before}{atg}{stem_after}")
    
    return utr5


def initialize_mrna_sequence(trigger_seq: str, logger: logging.Logger) -> mRNADesign:
    """
    Phase 2 main function: Design mRNA for POU5F1 delivery with toehold trigger in 5'UTR.
    
    Args:
        trigger_seq: Selected 25nt trigger sequence from Phase 1 (neuron-specific gene)
        logger: Logger instance
    
    Returns:
        mRNADesign object for POU5F1 with toehold-modified 5'UTR
    """
    logger.info("="*70)
    logger.info("PHASE 2: Sequence Initialization (POU5F1 mRNA Design)")
    logger.info("="*70)
    
    # Load POU5F1 with its known UTR lengths
    # POU5F1: 5'UTR=62bp, CDS=1103bp, 3'UTR=267bp
    data_dir = Path(__file__).parent / "data"
    full_sequence, utr5_len, utr3_len = load_pou5f1_sequence(data_dir, logger)
    
    # Extract regions from loaded sequence
    # Structure: [5'UTR: 62 bp][CDS][3'UTR: 267 bp]
    utr5_original = full_sequence[:utr5_len]
    cds_start = utr5_len
    cds_end = len(full_sequence) - utr3_len
    cds = full_sequence[cds_start:cds_end]
    utr3 = full_sequence[cds_end:]
    
    logger.info(f"POU5F1 CDS length: {len(cds)}nt")
    logger.info(f"Original 5' UTR length: {len(utr5_original)}nt")
    logger.info(f"Original 3' UTR length: {len(utr3)}nt")
    
    # Design toehold 5' UTR (replace original)
    utr5 = design_toehold_utr(trigger_seq, logger)
    
    # Ensure CDS starts with ATG
    if not cds.startswith('AUG'):
        atg_pos = cds.find('AUG')
        if atg_pos > 0:
            cds = cds[atg_pos:]
    
    # Ensure CDS ends with stop codon
    if not cds.endswith(('UAA', 'UAG', 'UGA')):
        cds += 'UAA'
    
    # Create mRNA design
    design = mRNADesign(
        utr5=utr5,
        cds=cds,
        utr3=utr3,
        trigger_seq=trigger_seq
    )
    
    logger.info(f"✓ Phase 2 complete.")
    logger.info(f"  Full mRNA length: {len(design.full_sequence)}nt")
    logger.info(f"  ATG position: {design.atg_position}")
    
    return design


# ==============================================================================
# Phase 3: ViennaRNA Structure Prediction
# ==============================================================================

def write_fasta(sequences: List[str], output_path: Path, prefix: str = "seq") -> Path:
    """Write sequences to FASTA file."""
    with open(output_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">{prefix}_{i}\n{seq}\n")
    return output_path


def predict_structure_with_viennarna(sequences: List[str], output_dir: Path, 
                                      logger: logging.Logger) -> List[Dict]:
    """
    Predict RNA secondary structure using ViennaRNA RNAfold.
    
    Returns:
        List of dicts with 'sequence', 'structure', and 'mfe' for each sequence
    """
    logger.info(f"Predicting structures for {len(sequences)} sequences with ViennaRNA...")
    
    # Write sequences to FASTA
    fasta_path = output_dir / "temp_structures.fasta"
    write_fasta(sequences, fasta_path, prefix="mrna")
    
    # Create Python script to run ViennaRNA structure prediction
    script_path = output_dir / "run_viennarna_structure.py"
    output_json = output_dir / "structures.json"
    
    with open(script_path, 'w') as f:
        f.write(f'''
import sys
import json
import RNA

# Read FASTA
fasta_path = "{fasta_path}"
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

print(f"Loaded {{len(sequences)}} sequences")

# Sanitize sequences
def sanitize_seq(seq, max_len=2000):
    seq = seq.upper().replace('T', 'U')
    cleaned = ''.join(ch if ch in {{'A', 'C', 'G', 'U'}} else 'A' for ch in seq)
    if len(cleaned) == 0:
        cleaned = 'A'
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned

sequences = [sanitize_seq(s) for s in sequences]

# Predict structures using ViennaRNA RNAfold
results = []

for idx, seq in enumerate(sequences):
    try:
        # Check if this is a cofold case (two interacting molecules)
        if '&' in seq:
            # Use ViennaRNA's cofold for intermolecular interaction
            structure, mfe = RNA.cofold(seq)
        else:
            # Use ViennaRNA's fold for single molecule
            structure, mfe = RNA.fold(seq)
        
        num_pairs = structure.count('(')
        
        results.append({{
            'id': ids[idx],
            'sequence': seq,
            'structure': structure,
            'mfe': mfe
        }})
        
        print(f"Processed {{idx+1}}/{{len(sequences)}}: {{len(seq)}}nt, {{num_pairs}} bp, MFE={{mfe:.2f}} kcal/mol")
        
    except Exception as e:
        print(f"Error processing {{ids[idx]}}: {{e}}")
        results.append({{
            'id': ids[idx],
            'sequence': seq,
            'structure': '.' * len(seq),
            'mfe': 0.0
        }})

# Save results
with open("{output_json}", 'w') as f:
    json.dump(results, f, indent=2)

print(f"Saved results to {output_json}")
''')
    
    # Run the script
    rnafm_env = Path.home() / "anaconda3" / "envs" / "rnafm"
    cmd = f"conda run -p {rnafm_env} python {script_path}"
    
    logger.info(f"Running ViennaRNA RNAfold for thermodynamic predictions...")
    logger.info(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"ViennaRNA RNAfold prediction failed:\n{result.stderr}")
        raise RuntimeError("ViennaRNA RNAfold prediction failed")
    
    logger.info(result.stdout)
    
    # Load results
    with open(output_json, 'r') as f:
        results = json.load(f)
    
    logger.info(f"✓ Predicted {len(results)} structures")
    
    return results


# ==============================================================================
# Phase 3b: Translation Efficiency Prediction with RiboNN
# ==============================================================================

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


def predict_te_with_ribonn(sequences: List[str], output_dir: Path,
                          target_cell: str = "neuron", offtarget_cell: str = "liver",
                          species: str = 'human', top_k_models: int = 5,
                          utr5_length: Optional[int] = None,
                          utr3_length: Optional[int] = None,
                          logger: logging.Logger = None) -> pd.DataFrame:
    """Predict candidate Translation Efficiency (TE) values with RiboNN and return ranked results.
    
    Predicts TE for the ENTIRE transcript sequence (including all UTRs).
    
    Args:
        sequences: List of FULL mRNA sequences (5'UTR + CDS + 3'UTR)
        output_dir: Output directory for RiboNN files
        target_cell: Target cell type for TE prediction
        offtarget_cell: Off-target cell type for TE prediction
        species: Species for RiboNN models (default: 'human')
        top_k_models: Number of top models to use for ensemble prediction
        utr5_length: Unused (kept for API compatibility)
        utr3_length: Unused (kept for API compatibility)
        logger: Logger instance
        
    Returns:
        DataFrame with prediction results
    """
    import subprocess
    
    if logger is None:
        logger = logging.getLogger(__name__)

    ribonn_repo = Path.home() / 'Tools' / 'RiboNN'
    ribonn_env = Path.home() / 'anaconda3' / 'envs' / 'RiboNN'

    if not ribonn_repo.exists():
        logger.warning(f'RiboNN repo not found at {ribonn_repo}, skipping TE prediction')
        return None
    if not ribonn_env.exists():
        logger.warning(f'RiboNN env not found at {ribonn_env}, skipping TE prediction')
        return None

    logger.info(f'Predicting TE for {len(sequences)} full transcript sequences with RiboNN ({species})...')

    input_path = output_dir / 'ribonn_prediction_input.tsv'
    output_path = output_dir / 'ribonn_prediction_output.tsv'
    runner_path = output_dir / 'run_ribonn_predict.py'

    rows = []
    for idx, full_seq in enumerate(sequences):
        # Use entire sequence as CDS, minimal UTRs
        full_seq_dna = full_seq.replace('U', 'T').upper()
        rows.append({
            'tx_id': f'variant_{idx}',
            'utr5_sequence': 'A',  # Minimal 5' UTR
            'cds_sequence': full_seq_dna,  # Full transcript as CDS
            'utr3_sequence': 'A'  # Minimal 3' UTR
        })

    input_df = pd.DataFrame(rows)
    for col in ['utr5_sequence', 'cds_sequence', 'utr3_sequence']:
        input_df[col] = input_df[col].fillna('A').astype(str)
        input_df.loc[input_df[col].str.len() == 0, col] = 'A'

    input_df.to_csv(input_path, sep='\t', index=False)
    logger.info(f'✓ Wrote RiboNN input: {input_path}')
    logger.info(f'  Predicting TE for full transcript sequences (5\'UTR + CDS + 3\'UTR)')

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

    logger.info(f'Running RiboNN: {cmd}')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f'RiboNN failed:\n{result.stderr}')
        return None

    logger.info(result.stdout)

    try:
        pred_df = pd.read_csv(output_path, sep='\t')
        sequence_map = {f'variant_{i}': seq for i, seq in enumerate(sequences)}
        
        result_df = pd.DataFrame({
            'seq_id': pred_df['tx_id'],
            'sequence': pred_df['tx_id'].map(sequence_map),
            'mean_te': pred_df['mean_predicted_TE'].astype(float),
        })

        logger.info(f"✓ RiboNN predictions completed: {len(result_df)} sequences")
        return result_df
    except Exception as e:
        logger.warning(f"Error parsing RiboNN results: {e}")
        return None


def calculate_fitness(mrna_design: mRNADesign, output_dir: Path, 
                     logger: logging.Logger) -> Dict[str, float]:
    """
    Calculate fitness score for an mRNA design.
    
    Predicts folding in two states:
        State A (Liver/Off-target): mRNA alone, ATG should be paired (hidden)
        State B (Neurons/Target): mRNA + trigger, ATG should be unpaired (exposed)
    
    Fitness = ΔG_alone - ΔG_co-folded + Start_Codon_Exposure_Penalty
    
    Returns:
        Dict with fitness metrics
    """
    # State A: mRNA alone
    seq_alone = mrna_design.full_sequence
    
    # State B: mRNA + trigger (intermolecular co-folding)
    # Use '&' separator for ViennaRNA cofold to treat as two interacting molecules
    seq_cofold = mrna_design.trigger_seq + "&" + seq_alone
    
    # Predict structures
    structures = predict_structure_with_viennarna(
        [seq_alone, seq_cofold],
        output_dir,
        logger
    )
    
    struct_alone = structures[0]
    struct_cofold = structures[1]
    
    dg_alone = struct_alone['mfe']
    dg_cofold = struct_cofold['mfe']
    
    # Check start codon exposure
    atg_pos = mrna_design.atg_position
    
    # In State A (alone), ATG should be paired
    atg_structure_alone = struct_alone['structure'][atg_pos:atg_pos+3]
    atg_paired_alone = '(' in atg_structure_alone or ')' in atg_structure_alone
    
    # In State B (cofold), adjust position due to trigger prefix plus '&' separator
    atg_pos_cofold = len(mrna_design.trigger_seq) + 1 + atg_pos
    atg_structure_cofold = struct_cofold['structure'][atg_pos_cofold:atg_pos_cofold+3]
    atg_unpaired_cofold = atg_structure_cofold == '...'
    
    # Penalties
    penalty = 0.0
    
    # Penalty if ATG is exposed when it shouldn't be (State A)
    if not atg_paired_alone:
        penalty += 10.0
    
    # Penalty if ATG is hidden when it should be exposed (State B)
    if not atg_unpaired_cofold:
        penalty += 10.0
    
    # Calculate fitness (higher is better)
    # We want large difference: ΔG_alone should be very negative (stable closed state)
    # ΔG_cofold should be less negative (trigger binding destabilizes the hairpin)
    thermodynamic_delta = dg_alone - dg_cofold
    fitness = thermodynamic_delta - penalty
    
    metrics = {
        'fitness': fitness,
        'dg_alone': dg_alone,
        'dg_cofold': dg_cofold,
        'thermodynamic_delta': thermodynamic_delta,
        'penalty': penalty,
        'atg_paired_alone': atg_paired_alone,
        'atg_unpaired_cofold': atg_unpaired_cofold
    }
    
    return metrics


# ==============================================================================
# Phase 4: Evolutionary Search Loop (Genetic Algorithm)
# ==============================================================================

class GeneticAlgorithm:
    """Genetic Algorithm for optimizing the 5' UTR sequence."""
    
    def __init__(self, initial_design: mRNADesign, output_dir: Path,
                 logger: logging.Logger, population_size: int = 20,
                 mutation_rate: float = 0.1):
        self.initial_design = initial_design
        self.output_dir = output_dir
        self.logger = logger
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # Initialize population
        self.population = [initial_design]
        for _ in range(population_size - 1):
            mutated = self.mutate_design(initial_design, rate=0.3)
            self.population.append(mutated)
        
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_design = None
        self.fitness_history = []
        self.te_history = []  # Track translation efficiency across generations
    
    def mutate_design(self, design: mRNADesign, rate: float = None) -> mRNADesign:
        """
        Mutate the 5' UTR sequence.
        
        Mutations:
            - Nucleotide substitutions
            - Small insertions
            - Small deletions
        """
        if rate is None:
            rate = self.mutation_rate
        
        utr5 = design.utr5
        nucleotides = ['A', 'U', 'G', 'C']
        
        # Collect mutations to apply (to avoid index issues during iteration)
        mutations = []
        for i in range(len(utr5)):
            if random.random() < rate:
                mutation_type = random.choice(['substitute', 'insert', 'delete'])
                mutations.append((i, mutation_type))
        
        # Apply mutations (process in reverse order to maintain indices)
        utr5_list = list(utr5)
        for i, mut_type in reversed(mutations):
            if i >= len(utr5_list):
                continue  # Skip if index is now out of range
            
            if mut_type == 'substitute':
                utr5_list[i] = random.choice(nucleotides)
            elif mut_type == 'insert' and len(utr5_list) < 200:  # Limit length
                utr5_list.insert(i, random.choice(nucleotides))
            elif mut_type == 'delete' and len(utr5_list) > 30:  # Minimum length
                utr5_list.pop(i)
        
        # Ensure ATG is still at the end of UTR (if it was there)
        new_utr5 = ''.join(utr5_list)
        if not new_utr5.endswith('AUG'):
            # Re-add ATG at the end
            if new_utr5.endswith('AU'):
                new_utr5 = new_utr5[:-2] + 'AUG'
            elif new_utr5.endswith('A'):
                new_utr5 = new_utr5[:-1] + 'AUG'
            else:
                new_utr5 = new_utr5[:-3] + 'AUG' if len(new_utr5) > 3 else new_utr5 + 'AUG'
        
        return mRNADesign(
            utr5=new_utr5,
            cds=design.cds,
            utr3=design.utr3,
            trigger_seq=design.trigger_seq
        )
    
    def crossover(self, parent1: mRNADesign, parent2: mRNADesign) -> mRNADesign:
        """Perform crossover between two parent designs."""
        # Single-point crossover in the 5' UTR
        utr1 = parent1.utr5
        utr2 = parent2.utr5
        
        # Find crossover point
        min_len = min(len(utr1), len(utr2))
        if min_len < 10:
            return parent1
        
        crossover_point = random.randint(5, min_len - 5)
        
        # Create offspring
        new_utr5 = utr1[:crossover_point] + utr2[crossover_point:]
        
        # Ensure it ends with ATG
        if not new_utr5.endswith('AUG'):
            new_utr5 = new_utr5[:-3] + 'AUG' if len(new_utr5) > 3 else 'AUG'
        
        return mRNADesign(
            utr5=new_utr5,
            cds=parent1.cds,
            utr3=parent1.utr3,
            trigger_seq=parent1.trigger_seq
        )
    
    def evaluate_population(self) -> List[Tuple[mRNADesign, Dict]]:
        """Evaluate fitness for all individuals in the population."""
        self.logger.info(f"Evaluating generation {self.generation} "
                        f"({len(self.population)} individuals)...")
        
        results = []
        for i, design in enumerate(self.population):
            try:
                metrics = calculate_fitness(design, self.output_dir, self.logger)
                results.append((design, metrics))
                
                if metrics['fitness'] > self.best_fitness:
                    self.best_fitness = metrics['fitness']
                    self.best_design = design
                    self.logger.info(f"  ★ New best fitness: {self.best_fitness:.2f}")
            except Exception as e:
                self.logger.warning(f"  Error evaluating individual {i}: {e}")
                # Assign poor fitness
                results.append((design, {'fitness': -1000.0}))
        
        # Sort by fitness
        results.sort(key=lambda x: x[1]['fitness'], reverse=True)
        
        # Log statistics
        fitnesses = [r[1]['fitness'] for r in results]
        self.logger.info(f"  Generation {self.generation} fitness: "
                        f"Best={max(fitnesses):.2f}, "
                        f"Mean={np.mean(fitnesses):.2f}, "
                        f"Worst={min(fitnesses):.2f}")
        
        # Predict Translation Efficiency with RiboNN
        self.logger.info(f"  Predicting Translation Efficiency with RiboNN...")
        sequences = [design.full_sequence for design in self.population]
        te_df = predict_te_with_ribonn(
            sequences,
            self.output_dir,
            logger=self.logger
        )
        
        if te_df is not None:
            te_values = te_df['mean_te'].tolist()
            self.logger.info(f"  Translation Efficiency: "
                           f"Best={max(te_values):.4f}, "
                           f"Mean={np.mean(te_values):.4f}, "
                           f"Worst={min(te_values):.4f}")
            
            self.te_history.append({
                'generation': self.generation,
                'te_values': te_values,
                'best_te': max(te_values),
                'mean_te': np.mean(te_values),
                'worst_te': min(te_values)
            })
        else:
            self.logger.info("  RiboNN TE prediction skipped or failed")
        
        self.fitness_history.append({
            'generation': self.generation,
            'best': max(fitnesses),
            'mean': np.mean(fitnesses),
            'worst': min(fitnesses)
        })
        
        return results
    
    def select_parents(self, evaluated_pop: List[Tuple[mRNADesign, Dict]], 
                      k: int = 5) -> List[mRNADesign]:
        """Select parents using tournament selection."""
        parents = []
        for _ in range(self.population_size):
            # Tournament selection
            tournament = random.sample(evaluated_pop, min(k, len(evaluated_pop)))
            winner = max(tournament, key=lambda x: x[1]['fitness'])
            parents.append(winner[0])
        return parents
    
    def evolve_generation(self):
        """Evolve one generation."""
        # Evaluate current population
        evaluated_pop = self.evaluate_population()
        
        # Select parents
        parents = self.select_parents(evaluated_pop)
        
        # Create next generation
        next_generation = []
        
        # Elitism: keep top 2 individuals
        next_generation.extend([evaluated_pop[0][0], evaluated_pop[1][0]])
        
        # Generate offspring
        while len(next_generation) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Crossover
            if random.random() < 0.7:  # Crossover probability
                offspring = self.crossover(parent1, parent2)
            else:
                offspring = parent1
            
            # Mutation
            offspring = self.mutate_design(offspring)
            
            next_generation.append(offspring)
        
        self.population = next_generation
        self.generation += 1
    
    def run(self, num_generations: int) -> Tuple[mRNADesign, Dict]:
        """
        Run the genetic algorithm for a specified number of generations.
        
        Returns:
            (best_design, best_metrics)
        """
        self.logger.info("="*70)
        self.logger.info(f"PHASE 4: Evolutionary Search ({num_generations} generations)")
        self.logger.info("="*70)
        
        for gen in range(num_generations):
            self.evolve_generation()
            
            # Save checkpoint every 10 generations
            if (gen + 1) % 10 == 0:
                checkpoint_file = self.output_dir / f"checkpoint_gen_{self.generation}.json"
                self.save_checkpoint(checkpoint_file)
        
        # Final evaluation of best design
        self.logger.info("="*70)
        self.logger.info("Evolution complete. Evaluating best design...")
        final_metrics = calculate_fitness(self.best_design, self.output_dir, self.logger)
        
        self.logger.info(f"✓ Best fitness: {final_metrics['fitness']:.2f}")
        self.logger.info(f"  ΔG_alone: {final_metrics['dg_alone']:.2f} kcal/mol")
        self.logger.info(f"  ΔG_cofold: {final_metrics['dg_cofold']:.2f} kcal/mol")
        self.logger.info(f"  Thermodynamic Δ: {final_metrics['thermodynamic_delta']:.2f}")
        self.logger.info(f"  ATG paired (alone): {final_metrics['atg_paired_alone']}")
        self.logger.info(f"  ATG unpaired (cofold): {final_metrics['atg_unpaired_cofold']}")
        
        return self.best_design, final_metrics
    
    def save_checkpoint(self, filepath: Path):
        """Save checkpoint with current best design and history."""
        checkpoint = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_design': {
                'utr5': self.best_design.utr5,
                'cds': self.best_design.cds[:100] + '...',  # Truncate for readability
                'utr3': self.best_design.utr3,
                'trigger_seq': self.best_design.trigger_seq,
                'full_sequence': self.best_design.full_sequence[:200] + '...'
            },
            'fitness_history': self.fitness_history,
            'te_history': self.te_history  # Include translation efficiency history
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        self.logger.info(f"  Saved checkpoint to {filepath}")


def save_final_results(best_design: mRNADesign, metrics: Dict, 
                      output_dir: Path, logger: logging.Logger, te_history: List = None):
    """Save final optimized sequences and results."""
    logger.info("Saving final results...")
    
    # Save sequences
    sequences_file = output_dir / "optimized_sequences.fasta"
    with open(sequences_file, 'w') as f:
        f.write(f">optimized_mrna_POU5F1_eToehold\n")
        f.write(f"{best_design.full_sequence}\n")
        f.write(f"\n>5_prime_UTR\n")
        f.write(f"{best_design.utr5}\n")
        f.write(f"\n>CDS_POU5F1\n")
        f.write(f"{best_design.cds}\n")
        f.write(f"\n>3_prime_UTR\n")
        f.write(f"{best_design.utr3}\n")
        f.write(f"\n>trigger_sequence\n")
        f.write(f"{best_design.trigger_seq}\n")
    
    logger.info(f"✓ Saved sequences to {sequences_file}")
    
    # Save TE history if available
    if te_history:
        te_history_file = output_dir / "te_history.json"
        with open(te_history_file, 'w') as f:
            json.dump(te_history, f, indent=2)
        logger.info(f"✓ Saved TE history to {te_history_file}")
    
    # Save metrics
    metrics_file = output_dir / "optimization_metrics.json"
    results = {
        'final_metrics': metrics,
        'sequence_info': {
            'full_length': len(best_design.full_sequence),
            'utr5_length': len(best_design.utr5),
            'cds_length': len(best_design.cds),
            'utr3_length': len(best_design.utr3),
            'atg_position': best_design.atg_position
        },
        'sequences': {
            'full_mrna': best_design.full_sequence,
            'utr5': best_design.utr5,
            'cds': best_design.cds,
            'utr3': best_design.utr3,
            'trigger': best_design.trigger_seq
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Saved metrics to {metrics_file}")
    
    # Create summary report
    report_file = output_dir / "RESULTS.txt"
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ETOEHOLD mRNA OPTIMIZATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write("Objective: Design mRNA with eToehold logic gate for neuron-specific\n")
        f.write("           POU5F1 expression (ON in neurons, OFF in liver)\n\n")
        f.write("-"*70 + "\n")
        f.write("FINAL METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Fitness Score:          {metrics['fitness']:.2f}\n")
        f.write(f"ΔG alone (liver):       {metrics['dg_alone']:.2f} kcal/mol\n")
        f.write(f"ΔG cofold (neuron):     {metrics['dg_cofold']:.2f} kcal/mol\n")
        f.write(f"Thermodynamic Δ:        {metrics['thermodynamic_delta']:.2f} kcal/mol\n")
        f.write(f"Penalty:                {metrics['penalty']:.2f}\n")
        f.write(f"ATG paired (liver):     {metrics['atg_paired_alone']}\n")
        f.write(f"ATG unpaired (neuron):  {metrics['atg_unpaired_cofold']}\n\n")
        f.write("-"*70 + "\n")
        f.write("SEQUENCE INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Total mRNA length:      {len(best_design.full_sequence)} nt\n")
        f.write(f"5' UTR length:          {len(best_design.utr5)} nt\n")
        f.write(f"CDS length:             {len(best_design.cds)} nt\n")
        f.write(f"3' UTR length:          {len(best_design.utr3)} nt\n")
        f.write(f"ATG position:           {best_design.atg_position}\n\n")
        f.write("-"*70 + "\n")
        f.write("OPTIMIZED 5' UTR SEQUENCE\n")
        f.write("-"*70 + "\n")
        f.write(f"{best_design.utr5}\n\n")
        f.write("-"*70 + "\n")
        f.write("TRIGGER SEQUENCE (neuron-specific)\n")
        f.write("-"*70 + "\n")
        f.write(f"{best_design.trigger_seq}\n\n")
        f.write("-"*70 + "\n")
        f.write("MECHANISM\n")
        f.write("-"*70 + "\n")
        f.write("In liver (off-target):  Toehold hairpin sequesters ATG → No translation\n")
        f.write("In neurons (target):    Trigger RNA binds toehold → Opens hairpin → \n")
        f.write("                        ATG exposed → Translation of POU5F1\n\n")
    
    logger.info(f"✓ Saved summary report to {report_file}")


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='eToehold mRNA Designer - Serova Challenge',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations for genetic algorithm (default: 50)')
    parser.add_argument('--population-size', type=int, default=20,
                       help='Population size for genetic algorithm (default: 20)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                       help='Mutation rate (default: 0.1)')
    parser.add_argument('--skip-phase1', action='store_true',
                       help='Skip Phase 1 (trigger selection) and use default trigger')
    parser.add_argument('--trigger-seq', type=str, default=None,
                       help='Manually specify trigger sequence (25nt)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(output_dir)
    
    # Data directory
    data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    logger.info("="*70)
    logger.info("ETOEHOLD mRNA DESIGNER - SEROVA CHALLENGE")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Generations: {args.generations}")
    logger.info(f"Population size: {args.population_size}")
    logger.info(f"Mutation rate: {args.mutation_rate}")
    
    try:
        # STEP 0: Calculate volcano plot statistics (before trigger selection)
        # This provides the statistical foundation for selecting the best trigger
        logger.info("\n" + "="*70)
        logger.info("STEP 0: Volcano Plot Analysis (Differential Expression Statistics)")
        logger.info("="*70)
        
        tsv_path = load_local_transcript_data(data_dir, logger)
        volcano_results = calculate_volcano_statistics(tsv_path, output_dir, logger)
        
        # Phase 1: Select trigger RNA based on volcano statistics
        if args.trigger_seq:
            logger.info("Using manually specified trigger sequence")
            trigger_seq = args.trigger_seq.upper().replace('T', 'U')
            trigger_gene = "manual"
        elif args.skip_phase1:
            logger.info("Skipping Phase 1, using default miR-124 trigger")
            trigger_seq = "UAAGGCACGCGGUGAAUGCCAAGGG"
            trigger_gene = "miR-124"
        else:
            # Use volcano statistics for trigger selection
            trigger_gene, trigger_seq = select_trigger_rna(
                output_dir, data_dir, logger, volcano_data=volcano_results
            )
        
        # Phase 2: Initialize mRNA sequence (POU5F1 with toehold in 5'UTR)
        initial_design = initialize_mrna_sequence(trigger_seq, logger)
        
        # Save initial design
        initial_file = output_dir / "initial_design.json"
        with open(initial_file, 'w') as f:
            json.dump({
                'trigger_gene': trigger_gene,
                'trigger_seq': trigger_seq,
                'delivery_gene': 'POU5F1',
                'utr5': initial_design.utr5,
                'cds': initial_design.cds[:100] + '...',
                'utr3': initial_design.utr3,
                'full_length': len(initial_design.full_sequence)
            }, f, indent=2)
        
        # Phase 3 & 4: Optimize with genetic algorithm
        ga = GeneticAlgorithm(
            initial_design=initial_design,
            output_dir=output_dir,
            logger=logger,
            population_size=args.population_size,
            mutation_rate=args.mutation_rate
        )
        
        best_design, final_metrics = ga.run(args.generations)
        
        # Save final results
        save_final_results(best_design, final_metrics, output_dir, logger, te_history=ga.te_history)
        
        logger.info("="*70)
        logger.info("✓ PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"See RESULTS.txt for summary")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
