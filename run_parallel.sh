#!/bin/bash
# ============================================================
# MOMO 并行优化启动脚本（4 worker，每个处理 200 个 lead）
#
# 用法（在 AutoDL 的 tmux 中运行）：
#   cd /root/autodl-tmp/model_combine/ManyObjectiveDrugDesign
#   chmod +x run_parallel.sh
#   ./run_parallel.sh
#
# 每个 worker 使用独立通信目录（outputs/w{N}/），互不干扰。
# 运行完毕后自动合并输出文件。
# ============================================================

set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/model_combine/ManyObjectiveDrugDesign"
PLATEMO_SRC="${PROJECT_DIR}/PlatEMO 4.2"
RUN_TAG="MOL_MOEA_v10_bank_256"
OUT_DIR="${PROJECT_DIR}/outputs/${RUN_TAG}"

WORKERS=4
TOTAL_LEADS=200
LEADS_PER_WORKER=$(( TOTAL_LEADS / WORKERS ))   # = 200

mkdir -p "${OUT_DIR}"

echo "========================================"
echo " MOMO 并行优化：${WORKERS} workers x ${LEADS_PER_WORKER} leads each"
echo "========================================"

declare -a MATLAB_PIDS
declare -a PYTHON_PIDS

for i in $(seq 0 $(( WORKERS - 1 ))); do
    LEAD_START=$(( i * LEADS_PER_WORKER + 1 ))
    LEAD_END=$(( (i + 1) * LEADS_PER_WORKER ))
    COMM_DIR="${OUT_DIR}/w${i}"
    mkdir -p "${COMM_DIR}"

    echo ""
    echo "--- Worker ${i}: leads ${LEAD_START}-${LEAD_END}"
    echo "    comm_dir = ${COMM_DIR}"

    # 1. 先启动 Python optimizer（在后台等待 MATLAB 发来信号）
    OBJECTIVE_MODE=momo_task1 \
    REPAIR_MODE=none \
    LATENT_NORMALIZE_TO_UNIT=0 \
    BEAM_SEARCH_ENABLED=1 \
    SIGMA_MODE=z0_std \
    LEAD_INIT_SIGMA=0.38 \
    DISCRETE_REFINE_ENABLED=0 \
    DISCRETE_REFINE_TOP_K=5 \
    DISCRETE_REFINE_NEIGHBORS=30 \
    DISCRETE_REFINE_MIN_QED=0.6 \
    STAGE2_FRAG_MODE=hiqed \
    STAGE2_HIQED_THRESHOLD=0.85 \
    STAGE2_QED_MONOTONIC=1 \
    STAGE2_HARD_MIN_QED=0.90 \
    STAGE2_HARD_MIN_SIM=0.20 \
    STAGE2_SELECT_MODE=pareto \
    WARM_START_CSV="/root/autodl-tmp/model_combine/MOMO-master-main/momo/data/oripops_qed/QMO_qed_mol800_optsmiles.csv" \
    PLATEMO_COMM_DIR="${COMM_DIR}" \
    WORKER_ID="${i}" \
        python "${PROJECT_DIR}/optimizer1.py" \
        > "${OUT_DIR}/optimizer_w${i}.log" 2>&1 &
    PYTHON_PIDS[$i]=$!
    echo "    Python PID: ${PYTHON_PIDS[$i]}"

    # 等待 Python 加载模型（约 8 秒），再启动 MATLAB
    sleep 8

    # 2. 启动 MATLAB
    # 强制告诉 MATLAB 读哪个源文件数据集（支持全量 800 分子）
    PLATEMO_COMM_DIR="${COMM_DIR}" \
    TASK1_LEAD_FILE="/root/autodl-tmp/model_combine/MOMO-master-main/momo/data/qed_test.csv" \
    PLATEMO_MOL_TASK=1 \
    matlab -nodesktop -nosplash -r \
        "comm_dir_override='${COMM_DIR}'; lead_start=${LEAD_START}; lead_end=${LEAD_END}; run('${PLATEMO_SRC}/no_gui_task1.m'); exit" \
        > "${OUT_DIR}/matlab_w${i}.log" 2>&1 &
    MATLAB_PIDS[$i]=$!
    echo "    MATLAB PID: ${MATLAB_PIDS[$i]}"
done

echo ""
echo "全部 ${WORKERS} 个 worker 已启动。"
echo ""
echo "实时查看日志（新开 tmux 窗格）："
for i in $(seq 0 $(( WORKERS - 1 ))); do
    echo "  tail -f ${OUT_DIR}/optimizer_w${i}.log"
done
echo ""
echo "等待所有进程完成..."

# 等待所有进程结束
wait

echo ""
echo "========================================"
echo " 所有 worker 完成，开始合并输出..."
echo "========================================"

python3 - <<'PYEOF'
import pandas as pd, glob, os

d = "/root/autodl-tmp/model_combine/ManyObjectiveDrugDesign"
out = d + "/outputs/E10_maxfe25k_hiqed"

def merge(pattern, out_path, dedup_cols):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[MERGE] 未找到文件: {pattern}")
        return
    merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    merged = merged.drop_duplicates(subset=dedup_cols, keep='first')
    merged.to_csv(out_path, index=False)
    print(f"[MERGE] {len(files)} 个文件 -> {len(merged)} 行 -> {out_path}")

merge(f"{out}/MOMO_qed_mol800_w*.csv",
      f"{out}/MOMO_qed_mol800.csv",
      ['SMILES', 'mol_id'])

merge(f"{out}/task1_final_population_w*.csv",
      f"{out}/task1_final_population.csv",
      ['smiles', 'mol_id'])

merge(f"{out}/results_stage2_discrete_w*.csv",
      f"{out}/results_stage2_discrete.csv",
      ['smiles'])

print("[MERGE] 完成！")
PYEOF

echo ""
echo "========================================"
echo " 并行优化运行完毕"
echo "========================================"
