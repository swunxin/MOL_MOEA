#!/usr/bin/env python
"""
Decode PlatEMO final-population latent vectors to SMILES.

Reads the CSV produced by compute_momo_task1_metrics_from_platemo.m
(SaveDecs=true), decodes each latent vector through the ReLSO model,
and writes a new CSV with a 'smiles' column appended.

Usage:
    python decode_platemo_population.py \
        --decsv outputs/momo_task1_FRCSO_N100/Data/FRCSO_N100/metrics/final_population_decs_20250101_120000.csv \
        --output outputs/momo_task1_FRCSO_N100/final_population_smiles.csv

Environment variables (or edit defaults below):
    LATENT_CKPT   : path to ReLSO checkpoint (default: runs/relso_selfies/version_0/last.ckpt)
    VOCAB_FILE    : path to SELFIES vocab file (default: selfies_vocab.txt)
    DEVICE        : 'cuda' or 'cpu' (default: auto-detect)
    BATCH_SIZE    : number of rows to decode per batch (default: 32)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import selfies as sf

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import build_vocab
from model import ReLSO


def load_model(ckpt_path: str, device: str):
    """Load the ReLSO model from a checkpoint."""
    model = ReLSO.load_from_checkpoint(ckpt_path, map_location=torch.device(device))
    model.eval()
    model.to(device)
    return model


def load_vocab_mappings(vocab_file: str):
    """Load SELFIES vocabulary and build encode/decode mappings."""
    vocab = build_vocab.load_vocab_from_file(vocab_file)
    model_str2num, vocab_str2num = build_vocab.get_single_encoder(vocab)
    model_num2str, vocab_num2str = build_vocab.get_decoders(model_str2num, vocab_str2num)
    return model_str2num, vocab_str2num, model_num2str, vocab_num2str


def decode_latent_vectors(
    model: ReLSO,
    z_batch: np.ndarray,
    model_num2str: dict,
    vocab_num2str: dict,
    max_seq_len: int = 200,
    temperature: float = 1.0,
    beam_top_k: int = 5,
) -> list:
    """
    Decode a batch of latent vectors to SMILES using beam search.

    Args:
        model: ReLSO model
        z_batch: (batch_size, latent_dim) numpy array of latent vectors
        model_num2str: model token ID -> token string
        vocab_num2str: vocab token ID -> token string
        max_seq_len: maximum sequence length for decoding
        temperature: softmax temperature
        beam_top_k: number of top-k candidates per position

    Returns:
        list of SMILES strings (one per input vector)
    """
    device = next(model.parameters()).device
    z_tensor = torch.tensor(z_batch, dtype=torch.float32).to(device)
    softmax = nn.Softmax(dim=2)

    with torch.no_grad():
        logits = model.decode(z_tensor)  # (batch, seq_len, vocab_size)
        if temperature != 1.0:
            logits = logits / temperature
        probs = softmax(logits)
        # Argmax decoding (deterministic)
        token_ids = torch.argmax(probs, dim=2).cpu().tolist()

    smiles_list = []
    for token_seq in token_ids:
        try:
            decoded = build_vocab.selfies_decode_molecule(
                token_seq, model_num2str, vocab_num2str
            )
            smi = sf.decoder(decoded)
            smiles_list.append(smi if smi else '')
        except Exception:
            smiles_list.append('')

    return smiles_list


def main():
    parser = argparse.ArgumentParser(
        description='Decode PlatEMO final-population latent vectors to SMILES'
    )
    parser.add_argument(
        '--decsv', required=True,
        help='Path to final_population_decs_*.csv from compute_momo_task1_metrics_from_platemo'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output CSV path (will include SMILES column)'
    )
    parser.add_argument(
        '--ckpt',
        default=os.getenv('LATENT_CKPT', str(PROJECT_ROOT / 'runs' / 'relso_selfies' / 'version_0' / 'last.ckpt')),
        help='Path to ReLSO checkpoint'
    )
    parser.add_argument(
        '--vocab',
        default=os.getenv('VOCAB_FILE', str(PROJECT_ROOT / 'selfies_vocab.txt')),
        help='Path to SELFIES vocabulary file'
    )
    parser.add_argument(
        '--device',
        default=os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'),
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for decoding'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Softmax temperature (1.0 = deterministic argmax)'
    )
    parser.add_argument(
        '--max-seq-len', type=int, default=200,
        help='Maximum sequence length'
    )
    args = parser.parse_args()

    # Validate input
    decsv_path = Path(args.decsv)
    if not decsv_path.exists():
        print(f"ERROR: Input CSV not found: {decsv_path}")
        sys.exit(1)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: ReLSO checkpoint not found: {ckpt_path}")
        print(f"  Set LATENT_CKPT environment variable or use --ckpt")
        sys.exit(1)

    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        print(f"ERROR: Vocab file not found: {vocab_path}")
        sys.exit(1)

    print(f"Loading ReLSO model from: {ckpt_path}")
    print(f"Device: {args.device}")
    model = load_model(str(ckpt_path), args.device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"Loading vocabulary from: {vocab_path}")
    model_str2num, vocab_str2num, model_num2str, vocab_num2str = load_vocab_mappings(str(vocab_path))
    print(f"Vocabulary size: {len(model_str2num) + len(vocab_str2num)}")

    # Read the decision variables CSV
    print(f"\nReading: {decsv_path}")
    df = pd.read_csv(decsv_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # Identify dec columns (dec_1, dec_2, ..., dec_N)
    dec_cols = [c for c in df.columns if c.startswith('dec_')]
    if not dec_cols:
        print("ERROR: No 'dec_N' columns found in CSV")
        sys.exit(1)

    n_dec_dims = len(dec_cols)
    print(f"  Latent dimension: {n_dec_dims}")
    print(f"  Decoding with temperature={args.temperature}, batch_size={args.batch_size}")

    # Decode in batches
    all_smiles = []
    n_total = len(df)

    for batch_start in range(0, n_total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n_total)
        batch_df = df.iloc[batch_start:batch_end]
        z_batch = batch_df[dec_cols].values.astype(np.float32)

        smiles_batch = decode_latent_vectors(
            model, z_batch,
            model_num2str, vocab_num2str,
            max_seq_len=args.max_seq_len,
            temperature=args.temperature,
        )
        all_smiles.extend(smiles_batch)

        n_valid = sum(1 for s in smiles_batch if s)
        print(f"  Batch {batch_start//args.batch_size + 1}: "
              f"{batch_start}-{batch_end-1} | "
              f"valid SMILES: {n_valid}/{len(smiles_batch)} "
              f"({100*n_valid/len(smiles_batch):.1f}%)")

    # Add SMILES column and save
    df['smiles'] = all_smiles
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_valid_total = sum(1 for s in all_smiles if s)
    print(f"\nDone. Saved to: {output_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Valid SMILES: {n_valid_total}/{len(df)} ({100*n_valid_total/len(df):.1f}%)")

    # Quick summary per mol_id
    if 'mol_id' in df.columns:
        print(f"\nPer-lead summary:")
        for mol_id, group in df.groupby('mol_id'):
            n_valid = group['smiles'].apply(lambda x: bool(x)).sum()
            n_total = len(group)
            print(f"  mol_id={mol_id}: {n_valid}/{n_total} valid SMILES")


if __name__ == '__main__':
    main()
