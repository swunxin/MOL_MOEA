#!/usr/bin/env python
"""
Round-trip fidelity test for warm-start "good points" (Task1 QED).

问题：热启 CSV 里的达标分子(QED>=0.9 & sim>=0.4)以 SMILES 给出，但进种群后只以
*潜向量* 存在；MOEA 看到的目标值是 decode(encode(smiles)) 解码分子的 QED/sim。
若自编码器往返把分子改坏，好点在搜索空间里就不再达标 -> BANK 没东西可保 -> SR 上不去。

本脚本直接量化这件事：把热启的达标好点过滤出来，用 optimizer1.py **完全相同** 的
encode/decode/打分，往返一遍再打分，看有多少还达标、QED/sim 掉了多少。

复用 optimizer1.py（其 run 逻辑在 __main__ 下，import 不会触发跑实验）：
    get_optimizer()         -> 同一套 ReLSO 权重/词表/device
    seq_to_emb(smi)         -> 编码（与初始化注入热启种子时一致, optimizer1.py:1462）
    decode_with_validity_fallback(z, num_top) -> beam 解码（与 CalObj 评估一致, BEAM_SEARCH_ENABLED=1）
    QED.qed / tanimoto_similarity(mol, lead_fp) -> 与 py_OBJ 打分一致

用法（服务器，激活 xin38，在项目根目录）：
    cd /root/autodl-tmp/model_combine/ManyObjectiveDrugDesign
    python roundtrip_warmstart_test.py --mol-id-range 1,200 --out outputs/roundtrip_warmstart_task1.csv
    # 默认已用项目内【对齐后】数据 momo_data/task1_oripops.csv + momo_data/task1_leads.csv
    # 对齐后期望: SANITY 的 sim 误差≈0, 且 rt_pass 应很高(往返无损 + lead 配对正确)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

# 复用实际跑实验的同一套编码/解码/打分（optimizer1.py 的 run 在 __main__ 下，import 安全）
from optimizer1 import get_optimizer, tanimoto_similarity, morgan_fingerprint


def load_leads(leads_csv: str):
    """qed_test.csv: 每行 1 个 lead SMILES（可能逗号分隔，取第一段）。1-based: leads[m-1]=mol_id m 的 lead。"""
    leads = []
    with open(leads_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('smiles'):
                continue
            leads.append(line.split(',')[0].strip())
    return leads


def main():
    ap = argparse.ArgumentParser()
    # 默认用项目内【已对齐】的 Task1 数据(momo_data/);lead 第 i 行 = mol_id (i-1) 的 lead。
    ap.add_argument('--warmstart-csv', default='momo_data/task1_oripops.csv', help='oripops (SMILES,mol_id,sim,qed)')
    ap.add_argument('--leads-csv', default='momo_data/task1_leads.csv', help='对齐后的 lead(纯SMILES,每行1个;mol_id==行号-1)')
    ap.add_argument('--mol-id-range', default='1,200', help='1-based inclusive, e.g. "1,200"')
    ap.add_argument('--tre-qed', type=float, default=0.9)
    ap.add_argument('--tre-sim', type=float, default=0.4)
    ap.add_argument('--num-top', type=int, default=5, help='beam top-k (与 run 默认一致)')
    ap.add_argument('--out', default='outputs/roundtrip_warmstart_task1.csv')
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.mol_id_range.split(','))
    TQ, TS = args.tre_qed, args.tre_sim

    print(f"Loading model (ReLSO) via get_optimizer() ...", flush=True)
    opt = get_optimizer()

    leads = load_leads(args.leads_csv)
    print(f"leads: {len(leads)}", flush=True)

    warm = pd.read_csv(args.warmstart_csv)
    warm['qed'] = pd.to_numeric(warm['qed'], errors='coerce')
    warm['sim'] = pd.to_numeric(warm['sim'], errors='coerce')
    good = warm[(warm.mol_id >= lo) & (warm.mol_id <= hi) & (warm.qed >= TQ) & (warm.sim >= TS)].copy()
    print(f"达标好点 (QED>={TQ} & sim>={TS}, mol_id {lo}-{hi}): {len(good)} 个 "
          f"(覆盖 {good.mol_id.nunique()} 个 lead)\n", flush=True)

    rows = []
    for i, r in enumerate(good.itertuples(), 1):
        mid = int(r.mol_id)
        smi = str(r.SMILES)
        # 对齐口径：lead 文件第 i 行 = mol_id (i-1) 的 lead  =>  mol_id m 的 lead = leads[m]（0-based）
        if mid >= len(leads):
            continue
        lead_smi = leads[mid]
        lead_mol = Chem.MolFromSmiles(lead_smi)
        if lead_mol is None:
            continue
        lead_fp = morgan_fingerprint(lead_mol)

        m0 = Chem.MolFromSmiles(smi)
        if m0 is None:
            continue
        # sanity: 用我们的打分重算原分子（应≈CSV值，验证 lead 对齐 + 打分口径）
        qed0 = float(QED.qed(m0)); sim0 = float(tanimoto_similarity(m0, lead_fp))

        # ===== 往返：encode -> decode(beam) -> 重打分 =====
        z = opt.seq_to_emb(smi)                                  # [1, D]
        dec = opt.decode_with_validity_fallback(z, num_top=args.num_top)
        mol_rt, smi_rt = dec[0]
        if mol_rt is None:
            qed_rt = sim_rt = 0.0; valid = False; same = False
        else:
            qed_rt = float(QED.qed(mol_rt)); sim_rt = float(tanimoto_similarity(mol_rt, lead_fp))
            valid = True
            same = (Chem.MolToSmiles(mol_rt) == Chem.MolToSmiles(m0))

        rows.append(dict(mol_id=mid, lead=lead_smi, warm_smi=smi, rt_smi=smi_rt,
                         qed_csv=float(r.qed), sim_csv=float(r.sim),
                         qed_orig=qed0, sim_orig=sim0,
                         qed_rt=qed_rt, sim_rt=sim_rt, rt_valid=valid, rt_same=same,
                         rt_pass=bool(valid and qed_rt >= TQ and sim_rt >= TS)))
        if i % 25 == 0:
            print(f"  {i}/{len(good)} ...", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n = len(df)

    print("\n" + "=" * 64)
    print("SANITY（原分子重打分 vs CSV，应≈0）:")
    print(f"  |qed_orig - qed_csv| 中位={np.median((df.qed_orig-df.qed_csv).abs()):.4f}")
    print(f"  |sim_orig - sim_csv| 中位={np.median((df.sim_orig-df.sim_csv).abs()):.4f}")
    print("=" * 64)
    print(f"往返结果 ({n} 个达标好点):")
    print(f"  往返后仍达标 (rt_pass)        : {df.rt_pass.sum()}/{n} = {100*df.rt_pass.sum()/n:.1f}%")
    print(f"  往返解码无效                  : {(~df.rt_valid).sum()}/{n}")
    print(f"  往返解码出同一分子 (rt_same)  : {df.rt_same.sum()}/{n} = {100*df.rt_same.sum()/n:.1f}%")
    print(f"  QED 掉幅 (qed_orig-qed_rt): 中位={np.median(df.qed_orig-df.qed_rt):+.4f}  均值={np.mean(df.qed_orig-df.qed_rt):+.4f}")
    print(f"  sim 掉幅 (sim_orig-sim_rt): 中位={np.median(df.sim_orig-df.sim_rt):+.4f}  均值={np.mean(df.sim_orig-df.sim_rt):+.4f}")
    # 失败拆类：往返后是 QED 掉破线、还是 sim 掉破线
    fail = df[~df.rt_pass & df.rt_valid]
    if len(fail):
        qed_break = (fail.qed_rt < TQ).sum(); sim_break = (fail.sim_rt < TS).sum()
        print(f"  未通过的 {len(fail)} 个里: QED<{TQ} 的 {qed_break} ; sim<{TS} 的 {sim_break}")
    print("=" * 64)
    print(f"saved: {args.out}")
    print("\n判读：rt_pass 很低 => 往返损失实锤，好点在搜索空间里就不达标，BANK/保存无能为力，")
    print("      主线得转向 ① 提升解码保真 / ② 让目标值不依赖往返(直接用注入分子的真值)。")


if __name__ == '__main__':
    main()
