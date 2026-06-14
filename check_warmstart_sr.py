"""
Cross-check MOMO Task1 lead SMILES and warm-start success rates.

1. Compare mol_id→SMILES in mapping file vs qed_test.csv (lead SMILES)
2. Count warm-start solutions already meeting QED>=0.9 AND sim>=0.4
"""
import pandas as pd
from pathlib import Path

BASE = Path(r"E:\model_two_combine")

# --- Load mapping file (lead SMILES from optimization) ---
map_path = BASE / "ManyObjectiveDrugDesign" / "log" / "mol_id_to_mat_mapping.csv"
map_df = pd.read_csv(map_path)
print(f"Mapping file: {len(map_df)} rows, mol_id 0-{map_df['mol_id'].max()}")

# --- Load qed_test.csv (lead SMILES source) ---
qed_test = []
with open(BASE / "MOMO-master-main" / "momo" / "data" / "qed_test.csv", 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('SMILES'):
            # First field before comma is SMILES
            smi = line.split(',')[0].strip()
            qed_test.append(smi)
print(f"qed_test.csv: {len(qed_test)} lead SMILES")

# --- Cross-check: mapping mol_id vs qed_test line index ---
print("\n" + "=" * 60)
print("CROSS-CHECK: mapping mol_id vs qed_test.csv leads")
print("=" * 60)

mismatches = []
for _, row in map_df.iterrows():
    mol_id = int(row['mol_id'])
    map_smi = str(row['lead_smiles']).strip()
    
    # qed_test is 0-indexed, mapping mol_id is 0-based
    if mol_id >= len(qed_test):
        mismatches.append((mol_id, map_smi, "OUT_OF_RANGE"))
        continue
    
    expected_smi = qed_test[mol_id]
    if map_smi != expected_smi:
        mismatches.append((mol_id, map_smi, expected_smi))

if mismatches:
    print(f"\n  MISMATCHES: {len(mismatches)} / {len(map_df)}")
    for mol_id, map_smi, exp in mismatches[:5]:
        print(f"  mol_id={mol_id}: MAP='{map_smi[:60]}...' vs EXPECTED='{exp[:60]}...'")
else:
    print(f"\n  ALL {len(map_df)} SMILES MATCH! [PASS]")

# --- Load MOMO warm-start (optimized molecules) ---
momo_path = BASE / "MOMO_qed_mol800.csv"
momo_df = pd.read_csv(momo_path)
print(f"\nMOMO warm-start: {len(momo_df)} rows, mol_id 1-{momo_df['mol_id'].max()}")

# --- Warm-start success rate (first 200 leads) ---
print("\n" + "=" * 60)
print("WARM-START SUCCESS (first 200 leads, mol_id 0-199)")
print("  Condition: QED >= 0.9 AND sim >= 0.4")
print("=" * 60)

# MOMO dataset uses 1-based mol_id, our mapping uses 0-based
momo_first200 = momo_df[momo_df['mol_id'].between(1, 200)]
n_entries = len(momo_first200)
n_leads_with_data = momo_first200['mol_id'].nunique()

success_1based = []
for mol_id_1b in range(1, 201):
    group = momo_first200[momo_first200['mol_id'] == mol_id_1b]
    if group.empty:
        continue
    if ((group['qed'] >= 0.9) & (group['sim'] >= 0.4)).any():
        success_1based.append(mol_id_1b)

n_success = len(success_1based)
n_no_warmstart = 200 - n_leads_with_data

print(f"  Warm-start entries (mol_id 1-200): {n_entries}")
print(f"  Leads with >=1 warm-start solution: {n_leads_with_data}/200")
print(f"  Leads WITHOUT warm-start data:      {n_no_warmstart}/200")
print(f"  Leads already successful:           {n_success}/200 ({100*n_success/200:.1f}%)")

# Which leads lack warm-start?
missing = [i for i in range(1, 201) if i not in momo_first200['mol_id'].values]
if missing:
    print(f"  mol_ids WITHOUT warm-start (1-based): {missing}")

# Best QED+sim per lead among those NOT yet successful
non_success = [i for i in range(1, 201) if i not in success_1based and i in momo_first200['mol_id'].values]
non_success_details = []
for mid in non_success:
    g = momo_first200[momo_first200['mol_id'] == mid]
    best_qed = g['qed'].max()
    best_sim = g['sim'].max()
    best_qed_row = g.loc[g['qed'].idxmax()]
    best_sim_row = g.loc[g['sim'].idxmax()]
    non_success_details.append({
        'mol_id': mid,
        'best_qed': best_qed,
        'best_sim': best_sim,
        'best_combined_qed': best_qed_row['qed'] if best_qed >= 0.9 else best_sim_row['qed'],
        'best_combined_sim': best_sim_row['sim'] if best_qed >= 0.9 else best_qed_row['sim'],
    })

# Categorize non-success leads
qed_ok_sim_low = [d for d in non_success_details if d['best_qed'] >= 0.9 and d['best_sim'] < 0.4]
sim_ok_qed_low = [d for d in non_success_details if d['best_sim'] >= 0.4 and d['best_qed'] < 0.9]
both_low = [d for d in non_success_details if d['best_qed'] < 0.9 and d['best_sim'] < 0.4]
no_data = [i for i in range(1, 201) if i not in momo_first200['mol_id'].values]

print(f"\n  Breakdown of {200 - n_success} non-success leads:")
print(f"    QED>=0.9 but sim<0.4:  {len(qed_ok_sim_low)}")
print(f"    sim>=0.4 but QED<0.9:  {len(sim_ok_qed_low)}")
print(f"    Both below threshold:  {len(both_low)}")
print(f"    No warm-start data:    {len(no_data)}")

if qed_ok_sim_low:
    ids = [d['mol_id'] for d in qed_ok_sim_low[:10]]
    print(f"    (QED-OK mol_ids): {ids}")
if sim_ok_qed_low:
    ids = [d['mol_id'] for d in sim_ok_qed_low[:10]]
    print(f"    (Sim-OK mol_ids): {ids}")

print(f"\n{'='*60}")
print(f"SUMMARY:")
print(f"  SMILES cross-check (lead SMILES): {'PASS' if not mismatches else f'{len(mismatches)} MISMATCHES'}")
print(f"  Warm-start success rate: {n_success}/200 ({100*n_success/200:.1f}%) already meet QED>=0.9 & sim>=0.4")
print(f"  Leads needing optimization: {200 - n_success}")
print(f"{'='*60}")
