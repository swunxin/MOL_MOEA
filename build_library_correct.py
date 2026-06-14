"""
正确顺序的全量建库脚本：先按 Lead 判定成败，后全局 SMILES 去重。

与错误版本的区别：
  错误：先全局去重 → 同分子被一个Lead"抢走" → 其他Lead丢成功证据
  正确：先逐Lead判定QED≥0.9且sim≥0.4 → 记录成败 → 再收集所有分子去重

用法：
  python build_library_correct.py \
    --input MOMO_qed_mol800.csv \
    --output unique_library.csv \
    --summary sr_report.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_library(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    qed_threshold: float = 0.9,
    sim_threshold: float = 0.4,
) -> dict:
    """
    Returns dict with keys:
      total_leads, successful_leads, sr, total_unique_smiles,
      unique_successful_smiles, total_molecules
    """
    # ── Step 1: 按分子读入，不丢任何行 ──────────────────────────
    lead_success: dict[int, bool] = {}       # mol_id → has any success
    lead_mols: dict[int, list[str]] = {}      # mol_id → list of SMILES
    all_molecules_smiles: list[str] = []

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row["mol_id"])
            smi = row["SMILES"].strip()
            qed = float(row["qed"])
            sim = float(row["sim"])
            success = qed >= qed_threshold and sim >= sim_threshold

            lead_success[mid] = lead_success.get(mid, False) or success
            lead_mols.setdefault(mid, []).append(smi)
            all_molecules_smiles.append(smi)

    # ── Step 2: 逐 Lead 计算成功率 ──────────────────────────────
    total_leads = len(lead_success)
    successful_leads = sum(1 for v in lead_success.values() if v)
    sr = successful_leads / max(total_leads, 1)

    # ── Step 3: 全局去重，建唯一 SMILES 库 ───────────────────────
    all_unique = sorted(set(all_molecules_smiles))

    successful_smiles: set[str] = set()
    for mid, success in lead_success.items():
        if success:
            successful_smiles.update(lead_mols[mid])
    successful_unique = sorted(set(successful_smiles))

    # ── 写入 ────────────────────────────────────────────────────
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SMILES"])
        for smi in all_unique:
            writer.writerow([smi])

    lines = [
        "=" * 50,
        "Unique SMILES Library Build (Correct Order)",
        "=" * 50,
        f"Input:          {input_path}",
        f"Total leads:    {total_leads}",
        f"Successful:     {successful_leads}",
        f"SR:             {sr*100:.1f}%",
        f"",
        f"Library:        {output_path}",
        f"Total molecules (pre-dedup): {len(all_molecules_smiles)}",
        f"Unique SMILES (all):        {len(all_unique)}",
        f"Unique SMILES (successful): {len(successful_unique)}",
    ]
    report = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)

    return {
        "total_leads": total_leads,
        "successful_leads": successful_leads,
        "sr": sr,
        "total_unique_smiles": len(all_unique),
        "unique_successful_smiles": len(successful_unique),
        "total_molecules": len(all_molecules_smiles),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="MOMO_qed_mol800.csv",
                        help="Path to optimization results CSV")
    parser.add_argument("--output", default="unique_library.csv",
                        help="Output path for unique SMILES library")
    parser.add_argument("--summary", default="sr_report.txt",
                        help="Summary report path")
    parser.add_argument("--qed-threshold", type=float, default=0.9)
    parser.add_argument("--sim-threshold", type=float, default=0.4)
    args = parser.parse_args()

    build_library(
        input_path=Path(args.input),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        qed_threshold=args.qed_threshold,
        sim_threshold=args.sim_threshold,
    )


if __name__ == "__main__":
    main()
