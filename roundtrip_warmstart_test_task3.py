#!/usr/bin/env python
"""
Round-trip fidelity test —— Task3 (QED + DRD2 + similarity) warm-start good points.

和 roundtrip_warmstart_test.py(Task1)同思路,只是 Task3 是 3 目标:
  - lead     = qeddrd_test.csv（有表头；像 no_gui_task3_momo.m 那样跳过 'SMILES' 表头）
  - oripops  = QMO_qeddrd_mol200_optsmiles.csv（列 SMILES,mol_id,qed,sim,drd；mol_id 0-based）
  - 达标     = QED>=0.8 AND DRD2>=0.3 AND sim>=0.4
  - 对齐     = mol_id==lead_id  =>  lead = leads[mol_id]（跳表头后的第 mol_id 行, 0-based）
              (本地已验 |QED-csv|=|sim-csv|=0.0000)

复用 optimizer1.py 完全相同的 encode/decode(beam)/打分:
  seq_to_emb / decode_with_validity_fallback / QED.qed / tanimoto_similarity / drd2_oracle(TDC)

用法(服务器, 激活 xin38, 项目根):
  python roundtrip_warmstart_test_task3.py --out outputs/roundtrip_warmstart_task3.csv
  # 默认: lead=../MOMO-master-main/momo/data/qeddrd_test.csv ; oripops=两个 QMO_qeddrd 文件(0-199 + 200-699)拼接, 共 0-699
  # 可用 --leads-csv / --warmstart-csv(逗号分隔多文件) 覆盖
  # 期望: SANITY 三项误差≈0, rt_pass 高 => 热启好点对齐 + 往返无损 => 注入后能被 BANK 保住
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

from optimizer1 import get_optimizer, tanimoto_similarity, morgan_fingerprint, drd2_oracle


def load_leads_task3(path):
    """像 no_gui_task3_momo.m: 逐行取第一段(逗号前), 跳过 'SMILES' 表头。leads[k]=去表头后的第k行(0-based)。"""
    leads = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        first = line.split(',')[0].strip()
        if first.lower() == 'smiles':
            continue
        leads.append(first)
    return leads


def _drd2(smi):
    try:
        v = float(drd2_oracle(smi))
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warmstart-csv',
                    default='../MOMO-master-main/momo/data/oripops_qeddrd/QMO_qeddrd_mol200_optsmiles.csv,'
                            '../MOMO-master-main/momo/data/oripops_qeddrd/QMO_qeddrd_mol200700_optsmiles.csv',
                    help='QMO_qeddrd oripops（逗号分隔多个文件，会拼接；mol_id 0-699，列 SMILES,mol_id,qed,sim,drd）')
    ap.add_argument('--leads-csv',
                    default='../MOMO-master-main/momo/data/qeddrd_test.csv',
                    help='qeddrd_test.csv（有表头；脚本会跳 SMILES 表头）')
    ap.add_argument('--mol-id-range', default='0,699', help='0-based inclusive (两个oripops合起来覆盖0-699)')
    ap.add_argument('--tre-qed', type=float, default=0.8)
    ap.add_argument('--tre-drd', type=float, default=0.3)
    ap.add_argument('--tre-sim', type=float, default=0.4)
    ap.add_argument('--num-top', type=int, default=5)
    ap.add_argument('--out', default='outputs/roundtrip_warmstart_task3.csv')
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.mol_id_range.split(','))
    TQ, TD, TS = args.tre_qed, args.tre_drd, args.tre_sim

    if drd2_oracle is None:
        print("ERROR: drd2_oracle 未加载（Task3 需 TDC: pip install PyTDC）")
        sys.exit(1)

    print("Loading model (ReLSO) via get_optimizer() ...", flush=True)
    opt = get_optimizer()

    leads = load_leads_task3(args.leads_csv)
    print(f"leads(qeddrd 去表头) = {len(leads)}", flush=True)

    ws_paths = [p.strip() for p in args.warmstart_csv.split(',') if p.strip()]
    w = pd.concat([pd.read_csv(p) for p in ws_paths], ignore_index=True)
    print(f"oripops 文件 {len(ws_paths)} 个, 合并 {len(w)} 行, mol_id {w.mol_id.min()}..{w.mol_id.max()}", flush=True)
    for c in ('qed', 'sim', 'drd'):
        w[c] = pd.to_numeric(w[c], errors='coerce')
    good = w[(w.mol_id >= lo) & (w.mol_id <= hi) &
             (w.qed >= TQ) & (w.drd >= TD) & (w.sim >= TS)].copy()
    print(f"达标好点 (QED>={TQ} & DRD2>={TD} & sim>={TS}, mol_id {lo}-{hi}): "
          f"{len(good)} 个 (覆盖 {good.mol_id.nunique()} 个 lead)\n", flush=True)

    rows = []
    for i, r in enumerate(good.itertuples(), 1):
        mid = int(r.mol_id); smi = str(r.SMILES)
        if mid >= len(leads):
            continue
        lead_smi = leads[mid]                       # 对齐: lead = leads[mol_id]
        lead_mol = Chem.MolFromSmiles(lead_smi)
        if lead_mol is None:
            continue
        lead_fp = morgan_fingerprint(lead_mol)
        m0 = Chem.MolFromSmiles(smi)
        if m0 is None:
            continue
        # sanity: 原分子用我们的打分重算（应≈CSV）
        qed0 = float(QED.qed(m0)); sim0 = float(tanimoto_similarity(m0, lead_fp)); drd0 = _drd2(smi)

        # ===== 往返: encode -> decode(beam) -> 重打分 =====
        z = opt.seq_to_emb(smi)
        dec = opt.decode_with_validity_fallback(z, num_top=args.num_top)
        mol_rt, smi_rt = dec[0]
        if mol_rt is None:
            qed_rt = sim_rt = drd_rt = 0.0; valid = False; same = False
        else:
            qed_rt = float(QED.qed(mol_rt)); sim_rt = float(tanimoto_similarity(mol_rt, lead_fp))
            drd_rt = _drd2(smi_rt); valid = True
            same = (Chem.MolToSmiles(mol_rt) == Chem.MolToSmiles(m0))

        rows.append(dict(mol_id=mid, qed_csv=float(r.qed), sim_csv=float(r.sim), drd_csv=float(r.drd),
                         qed_orig=qed0, sim_orig=sim0, drd_orig=drd0,
                         qed_rt=qed_rt, sim_rt=sim_rt, drd_rt=drd_rt, rt_valid=valid, rt_same=same,
                         rt_pass=bool(valid and qed_rt >= TQ and drd_rt >= TD and sim_rt >= TS)))
        if i % 5 == 0:
            print(f"  {i}/{len(good)} ...", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n = len(df)

    print("\n" + "=" * 64)
    print("SANITY（原分子重打分 vs CSV，应≈0）:")
    print(f"  |qed_orig - qed_csv| 中位={np.median((df.qed_orig-df.qed_csv).abs()):.4f}")
    print(f"  |sim_orig - sim_csv| 中位={np.median((df.sim_orig-df.sim_csv).abs()):.4f}")
    print(f"  |drd_orig - drd_csv| 中位={np.median((df.drd_orig-df.drd_csv).abs()):.4f}")
    print("=" * 64)
    print(f"往返结果 ({n} 个达标好点):")
    print(f"  往返后仍达标 (rt_pass)        : {df.rt_pass.sum()}/{n} = {100*df.rt_pass.sum()/max(n,1):.1f}%")
    print(f"  往返解码无效                  : {(~df.rt_valid).sum()}/{n}")
    print(f"  往返解码出同一分子 (rt_same)  : {df.rt_same.sum()}/{n}")
    print(f"  QED 掉幅 中位={np.median(df.qed_orig-df.qed_rt):+.4f}  "
          f"DRD2 掉幅 中位={np.median(df.drd_orig-df.drd_rt):+.4f}  "
          f"sim 掉幅 中位={np.median(df.sim_orig-df.sim_rt):+.4f}")
    fail = df[~df.rt_pass & df.rt_valid]
    if len(fail):
        print(f"  未通过的 {len(fail)} 个里: QED<{TQ} {int((fail.qed_rt<TQ).sum())}; "
              f"DRD2<{TD} {int((fail.drd_rt<TD).sum())}; sim<{TS} {int((fail.sim_rt<TS).sum())}")
    print("=" * 64)
    print(f"saved: {args.out}")

    rt_rate = df.rt_pass.mean() if n else 0.0
    sim_err = float(np.median((df.sim_orig - df.sim_csv).abs())) if n else 0.0
    print("\n---------------- 判读 ----------------")
    if sim_err > 0.05:
        print(f"[!] SANITY sim 误差={sim_err:.3f} 偏大 => lead 配对/对齐有问题, 先修对齐, rt_pass 不可信。")
    elif rt_rate >= 0.8:
        print(f"rt_pass={rt_rate*100:.0f}% 高 => Task3 热启好点对齐且往返基本无损 => 注入后能被 BANK 保住。")
    else:
        print(f"rt_pass={rt_rate*100:.0f}% 低 => 往返把好点改坏了(看上面 QED/DRD2/sim 哪个掉破线)。")
    print("=" * 38)


if __name__ == '__main__':
    main()
