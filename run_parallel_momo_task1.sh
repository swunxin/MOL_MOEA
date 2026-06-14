#!/bin/bash
# ============================================================
# MOMO Task1 并行优化脚本
#
# 用法（在 AutoDL 的 tmux 中运行）：
#   cd /root/autodl-tmp/model_combine/ManyObjectiveDrugDesign
#   chmod +x run_parallel_momo_task1.sh
#   ./run_parallel_momo_task1.sh
#
# 换算法：改下面 ALGO 这一个变量即可。
# 每个 worker 使用独立通信目录，互不干扰。
# 运行完毕后自动合并 mol_id_to_mat_mapping.csv 文件。
# ============================================================

set -euo pipefail

# ======================== 唯一需要改的配置 ========================
ALGO="MOL_MOEA_v10_bank"          # PlatEMO 算法名（对应 Algorithms/ 下的目录名）
                                  #   切回基线对比就改成 FRCSO_N100
                                  #   注意：MOL_MOEA 的 task=1 由 no_gui_task1_momo.m 自动补（见该文件 platemo 调用处）
TOTAL_LEADS=200                   # 总 lead 数
WORKERS=4                         # 并行 worker 数
# =================================================================

LEADS_PER_WORKER=$(( TOTAL_LEADS / WORKERS ))

PROJECT_DIR="/root/autodl-tmp/model_combine/ManyObjectiveDrugDesign"
PLATEMO_SRC="${PROJECT_DIR}/PlatEMO 4.2"
RUN_TAG="momo_task1_${ALGO}"
OUT_DIR="${PROJECT_DIR}/outputs/${RUN_TAG}"

mkdir -p "${OUT_DIR}"

echo "========================================"
echo " MOMO Task1 并行优化"
echo "   Algorithm : ${ALGO}"
echo "   Workers   : ${WORKERS} x ${LEADS_PER_WORKER} leads"
echo "   Data dir  : ${PLATEMO_SRC}/Data/${ALGO}"
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

    # 1. 启动 Python optimizer（MOMO Task1 模式）
    OBJECTIVE_MODE=momo_task1 \
    REPAIR_MODE=none \
    LATENT_NORMALIZE_TO_UNIT=1 \
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
    PLATEMO_COMM_DIR="${COMM_DIR}" \
    PLATEMO_ALGORITHM="${ALGO}" \
    TASK1_LEAD_FILE="${PROJECT_DIR}/momo_data/task1_leads.csv" \
    matlab -nodesktop -nosplash -r \
        "comm_dir_override='${COMM_DIR}'; lead_start=${LEAD_START}; lead_end=${LEAD_END}; run('${PLATEMO_SRC}/no_gui_task1_momo.m'); exit" \
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
echo " 所有 worker 完成，开始合并映射文件..."
echo "========================================"

# 合并 mol_id_to_mat_mapping.csv
python3 - <<PYEOF
import pandas as pd, glob, os

out_dir = "${OUT_DIR}"

# 合并 mapping 文件
mapping_files = sorted(glob.glob(f"{out_dir}/w*/mol_id_to_mat_mapping.csv"))
if mapping_files:
    merged = pd.concat([pd.read_csv(f) for f in mapping_files], ignore_index=True)
    merged = merged.sort_values('mol_id').drop_duplicates(subset=['mol_id'], keep='first')
    merged.to_csv(f"{out_dir}/mol_id_to_mat_mapping.csv", index=False)
    print(f"[MERGE] {len(mapping_files)} mapping files -> {len(merged)} rows -> mol_id_to_mat_mapping.csv")
else:
    print("[MERGE] 未找到 mapping 文件（可能是早期成功跳过了所有 PlatEMO 运行）")

# 合并 retry_summary.csv（如果存在）
retry_files = sorted(glob.glob(f"{out_dir}/w*/retry_summary.csv"))
if retry_files:
    merged = pd.concat([pd.read_csv(f) for f in retry_files], ignore_index=True)
    merged = merged.sort_values('mol_id')
    merged.to_csv(f"{out_dir}/retry_summary.csv", index=False)
    print(f"[MERGE] {len(retry_files)} retry summary files -> {len(merged)} rows")

print("[MERGE] 完成！")
PYEOF

echo ""
echo "========================================"
echo " MOMO Task1 并行优化运行完毕"
echo "   Algorithm : ${ALGO}"
echo "   输出目录  : ${OUT_DIR}"
echo "   映射文件  : ${OUT_DIR}/mol_id_to_mat_mapping.csv"
echo ""
echo "计算 SR/HV 指标 + 导出 final population latent vectors："
echo ""
echo "  cd('${PLATEMO_SRC}');"
echo "  compute_momo_task1_metrics_from_platemo(..."
echo "      'DataDir',   '${PLATEMO_SRC}/Data/${ALGO}',..."
echo "      'Pattern',   '${ALGO}_DDProblem1_M2_D*_*.mat');"
echo ""
echo "解码 SMILES："
echo "  cd('${PROJECT_DIR}');"
echo "  python decode_platemo_population.py \\"
echo "      --decsv  '${PLATEMO_SRC}/Data/${ALGO}/metrics/final_population_decs_YYYYMMDD_HHMMSS.csv' \\"
echo "      --output '${OUT_DIR}/final_population_smiles.csv'"
echo "========================================"
