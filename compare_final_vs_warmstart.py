#!/usr/bin/env python
"""
Compare FRCSO_N100 final population SMILES vs MOMO warm-start SMILES.

For each lead (mol_id), answers:
  - Was there a successful warm-start molecule?
  - Did the optimizer find it (or a duplicate) in the final population?
  - What's the overlap between final pop and warm-start?

Usage (on AutoDL server):
    cd /root/autodl-tmp/model_combine/ManyObjectiveDrugDesign
    python compare_final_vs_warmstart.py \
        --decoded-csv outputs/momo_task1_FRCSO_N100/final_population_smiles.csv \
        --warmstart-csv MOMO_qed_mol800.csv \
        --output outputs/momo_task1_FRCSO_N100/comparison_report.csv

Or for a quick analysis without the decoded file (just metrics):
    python compare_final_vs_warmstart.py --quick-summary
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np


def canonicalize(smi: str) -> str:
    """Canonicalize SMILES via RDKit. Returns empty string on failure."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def load_warmstart(path: str):
    """Load MOMO warm-start CSV. Returns dict: mol_id -> list of (smiles, qed, sim).
    
    Supports two formats:
      1. With qed/sim columns: MOMO_qed_mol800.csv (SMILES,mol_id,qed,sim)
      2. Without qed/sim: QMO_qed_mol800_optsmiles.csv (SMILES,mol_id,...)
         In format 2, qed/sim are computed on the fly via RDKit.
    """
    df = pd.read_csv(path)
    warm = defaultdict(list)
    
    has_qed = 'qed' in df.columns
    has_sim = 'sim' in df.columns
    
    for _, row in df.iterrows():
        mid = int(row['mol_id'])
        smi = str(row['SMILES']).strip()
        
        if has_qed and has_sim:
            qed = float(row['qed'])
            sim = float(row['sim'])
        else:
            # Compute on the fly
            try:
                from rdkit import Chem
                from rdkit.Chem import QED, AllChem, DataStructs
                mol = Chem.MolFromSmiles(smi)
                qed = QED.qed(mol) if mol else 0.0
                sim = 0.0  # Can't compute sim without lead reference
            except Exception:
                qed = 0.0
                sim = 0.0
        
        warm[mid].append({
            'smiles': smi,
            'qed': qed,
            'sim': sim,
            'success': qed >= 0.9 and sim >= 0.4,
        })
    
    if not has_qed and not has_sim:
        print("  [WARN] No qed/sim columns found; computed QED, sim set to 0")
    return warm


def load_decoded_final(path: str):
    """Load decoded final population CSV. Returns dict: mol_id -> list of (smiles, obj1, obj2)."""
    df = pd.read_csv(path)
    final = defaultdict(list)
    for _, row in df.iterrows():
        mid = int(row['mol_id'])
        smi = str(row.get('smiles', '')).strip()
        if not smi:
            continue
        final[mid].append({
            'smiles': smi,
            'obj1': float(row.get('obj1', np.nan)),
            'obj2': float(row.get('obj2', np.nan)),
            # In task1, -obj1 = QED, -obj2 = sim
            'qed': -float(row.get('obj1', np.nan)),
            'sim': -float(row.get('obj2', np.nan)),
        })
    return final


def main():
    parser = argparse.ArgumentParser(
        description='Compare final population vs warm-start'
    )
    parser.add_argument('--decoded-csv', default='',
                        help='Path to final_population_smiles.csv (from decode_platemo_population.py)')
    parser.add_argument('--warmstart-csv',
                        default='MOMO-master-main/momo/data/oripops_qed/QMO_qed_mol800_optsmiles.csv',
                        help='Path to MOMO warm-start CSV (optsmiles)')
    parser.add_argument('--output', default='comparison_report.csv',
                        help='Output CSV path')
    parser.add_argument('--quick-summary', action='store_true',
                        help='Print quick summary from existing metrics CSV')
    parser.add_argument('--canonicalize', action='store_true',
                        help='Canonicalize SMILES before comparing (slower but more accurate)')
    parser.add_argument('--mol-id-range', default='0,199',
                        help='mol_id range (0-based), e.g. "0,199" for first 200')
    args = parser.parse_args()

    if args.quick_summary:
        print("Quick summary mode: no decoded SMILES file needed.")
        print("Run with --decoded-csv to get full per-lead comparison.")
        return

    if not args.decoded_csv:
        print("ERROR: --decoded-csv is required (or use --quick-summary)")
        sys.exit(1)

    # Parse mol_id range
    parts = args.mol_id_range.split(',')
    mol_min, mol_max = int(parts[0]), int(parts[1])

    # Load warm-start
    print(f"Loading warm-start: {args.warmstart_csv}")
    warm = load_warmstart(args.warmstart_csv)
    print(f"  {len(warm)} unique mol_ids with warm-start data")

    # Load decoded final population
    print(f"Loading decoded final population: {args.decoded_csv}")
    final = load_decoded_final(args.decoded_csv)
    print(f"  {len(final)} mol_ids with decoded final population")

    # Canonicalize if requested
    if args.canonicalize:
        print("Canonicalizing SMILES (this may take a while)...")
        for mid in warm:
            for entry in warm[mid]:
                entry['smiles'] = canonicalize(entry['smiles'])
        for mid in final:
            for entry in final[mid]:
                entry['smiles'] = canonicalize(entry['smiles'])

    # Per-lead comparison
    print(f"\nComparing mol_id {mol_min}-{mol_max}...")
    rows = []

    for mid_1b in range(mol_min + 1, mol_max + 2):  # 1-based for warm-start
        mid_0b = mid_1b - 1
        warm_entries = warm.get(mid_1b, [])
        final_entries = final.get(mid_0b, [])

        # Warm-start stats
        warm_smiles = [e['smiles'] for e in warm_entries]
        warm_success_count = sum(1 for e in warm_entries if e['success'])
        warm_has_success = warm_success_count > 0

        # Final population stats
        final_smiles = [e['smiles'] for e in final_entries]
        final_n = len(final_entries)
        final_qed_max = max([e['qed'] for e in final_entries], default=np.nan)
        final_sim_max = max([e['sim'] for e in final_entries], default=np.nan)

        # Check if final pop has any molecule meeting success criteria
        final_success_count = sum(
            1 for e in final_entries
            if e['qed'] >= 0.9 and e['sim'] >= 0.4
        )
        final_has_success = final_success_count > 0

        # Overlap: any final SMILES also in warm-start?
        warm_smiles_set = set(w for w in warm_smiles if w)
        final_smiles_set = set(f for f in final_smiles if f)
        overlap = final_smiles_set & warm_smiles_set
        n_overlap = len(overlap)

        # Overlap with warm-start SUCCESS molecules specifically
        warm_success_smiles_set = set(
            e['smiles'] for e in warm_entries if e['success'] and e['smiles']
        )
        overlap_success = final_smiles_set & warm_success_smiles_set
        n_overlap_success = len(overlap_success)

        rows.append({
            'mol_id': mid_0b,
            'warm_has_success': warm_has_success,
            'warm_n_entries': len(warm_entries),
            'warm_n_success': warm_success_count,
            'final_n': final_n,
            'final_has_success': final_has_success,
            'final_n_success': final_success_count,
            'final_qed_max': final_qed_max,
            'final_sim_max': final_sim_max,
            'overlap_any': n_overlap,
            'overlap_with_ws_success': n_overlap_success,
            'rediscovered_ws_success': n_overlap_success > 0,
        })

    report = pd.DataFrame(rows)
    report.to_csv(args.output, index=False)
    print(f"\nReport saved: {args.output}")

    # Summary statistics
    n_total = len(report)
    n_warm_success = report['warm_has_success'].sum()
    n_final_success = report['final_has_success'].sum()
    n_rediscovered = report['rediscovered_ws_success'].sum()
    n_warm_not_final = report[
        report['warm_has_success'] & ~report['final_has_success']
    ].shape[0]
    n_final_not_warm = report[
        ~report['warm_has_success'] & report['final_has_success']
    ].shape[0]

    # Among leads that had warm-start success, how many final pops also succeeded?
    if n_warm_success > 0:
        n_final_given_warm = report[report['warm_has_success'] & report['final_has_success']].shape[0]
        pct_final_given_warm = 100 * n_final_given_warm / n_warm_success
    else:
        n_final_given_warm = 0
        pct_final_given_warm = 0

    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY (mol_id {mol_min}-{mol_max})")
    print(f"{'='*60}")
    print(f"  Total leads analyzed:          {n_total}")
    print(f"")
    print(f"  Warm-start already successful: {n_warm_success}/{n_total} ({100*n_warm_success/n_total:.1f}%)")
    print(f"  Final pop successful:          {n_final_success}/{n_total} ({100*n_final_success/n_total:.1f}%)")
    print(f"")
    print(f"  Rediscovered WS success in final: {n_rediscovered}/{n_warm_success}")
    print(f"    (final pop contains >=1 SMILES from warm-start success set)")
    print(f"")
    print(f"  Warm success, final NOT success:  {n_warm_not_final}")
    print(f"    (leads where warm-start was good but optimizer failed)")
    print(f"  Warm NOT success, final success:  {n_final_not_warm}")
    print(f"    (leads where optimizer improved beyond warm-start)")
    print(f"")
    print(f"  If warm-start succeeded, final also succeeded: "
          f"{n_final_given_warm}/{n_warm_success} ({pct_final_given_warm:.1f}%)")
    print(f"{'='*60}")

    # Detailed: which leads had warm-start success but final did NOT
    if n_warm_not_final > 0:
        bad_leads = report[report['warm_has_success'] & ~report['final_has_success']]
        print(f"\nLeads where warm-start succeeded but final pop FAILED:")
        for _, row in bad_leads.iterrows():
            print(f"  mol_id={int(row['mol_id'])}: "
                  f"warm={int(row['warm_n_success'])} successes, "
                  f"final qed_max={row['final_qed_max']:.3f}, "
                  f"sim_max={row['final_sim_max']:.3f}, "
                  f"overlap_any={int(row['overlap_any'])}")

    # Detailed: which leads did optimizer succeed where warm-start didn't
    if n_final_not_warm > 0:
        new_leads = report[~report['warm_has_success'] & report['final_has_success']]
        print(f"\nLeads where warm-start FAILED but final pop SUCCEEDED (new discoveries):")
        for _, row in new_leads.iterrows():
            print(f"  mol_id={int(row['mol_id'])}: "
                  f"final_success={int(row['final_n_success'])}, "
                  f"qed_max={row['final_qed_max']:.3f}, "
                  f"sim_max={row['final_sim_max']:.3f}")


if __name__ == '__main__':
    main()
