#!/bin/bash
# ============================================================
# 单 lead Pop_Pool trace（分子 Task1）
#
# 用途：只跑【一个】lead，用 MOL_MOEA_v11_pooltrace 打印每代 [PT]
#       (OFF/SEL/POOL 的 N / 唯一目标点 / 唯一潜向量)，定位分子上
#       Pop_Pool 是不是"复制凑数"。== run_parallel_momo_task1.sh 砍成
#       1 个 worker、1 个 lead、算法换成 v11_pooltrace。
#
# 用法（AutoDL tmux）：
#   cd /root/autodl-tmp/model_combine/ManyObjectiveDrugDesign
#   chmod +x run_single_lead_trace.sh
#   ./run_single_lead_trace.sh 5        # 跑第 5 个 lead（不给参数默认第 1 个）
#
# 跑完看 [PT] 日志：
#   outputs/momo_task1_trace/single/matlab_lead5.log   (grep '^\[PT\]')
# 把这个 .log 发回来分析。
# ============================================================

set -euo pipefail

# ======================== 配置 ========================
ALGO="MOL_MOEA_v11_pooltrace"     # 分子版 pooltrace（已在 Algorithms/ 下）
LEAD="${1:-1}"                    # 要 trace 的 lead 序号（命令行第 1 个参数，默认 1）
# =====================================================

PROJECT_DIR="/root/autodl-tmp/model_combine/ManyObjectiveDrugDesign"
PLATEMO_SRC="${PROJECT_DIR}/PlatEMO 4.2"
OUT_DIR="${PROJECT_DIR}/outputs/momo_task1_trace/single"
COMM_DIR="${OUT_DIR}/lead${LEAD}"

mkdir -p "${COMM_DIR}"

echo "========================================"
echo " 单 lead Pop_Pool trace"
echo "   Algorithm : ${ALGO}"
echo "   Lead      : ${LEAD} (lead_start=lead_end=${LEAD})"
echo "   comm_dir  : ${COMM_DIR}"
echo "   MATLAB log: ${OUT_DIR}/matlab_lead${LEAD}.log   <- [PT] 在这里"
echo "========================================"

# 1. 启动 Python optimizer（与批量脚本同一组 momo_task1 环境）
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
WORKER_ID="0" \
    python "${PROJECT_DIR}/optimizer1.py" \
    > "${OUT_DIR}/optimizer_lead${LEAD}.log" 2>&1 &
PY_PID=$!
echo "Python PID: ${PY_PID}"

# 等 Python 加载模型
sleep 8

# 2. 启动 MATLAB —— 只跑这一个 lead；[PT] 打到 matlab 日志
PLATEMO_COMM_DIR="${COMM_DIR}" \
PLATEMO_ALGORITHM="${ALGO}" \
TASK1_LEAD_FILE="${PROJECT_DIR}/momo_data/task1_leads.csv" \
matlab -nodesktop -nosplash -r \
    "comm_dir_override='${COMM_DIR}'; lead_start=${LEAD}; lead_end=${LEAD}; run('${PLATEMO_SRC}/no_gui_task1_momo.m'); exit" \
    > "${OUT_DIR}/matlab_lead${LEAD}.log" 2>&1 &
ML_PID=$!
echo "MATLAB PID: ${ML_PID}"

echo ""
echo "实时看 [PT]：  tail -f ${OUT_DIR}/matlab_lead${LEAD}.log | grep --line-buffered '\[PT\]'"
echo "等待完成..."
wait

echo ""
echo "========================================"
echo " 完成。提取 [PT] 日志："
echo "   grep '^\[PT\]' '${OUT_DIR}/matlab_lead${LEAD}.log' > '${OUT_DIR}/PT_lead${LEAD}.log'"
echo " 把 PT_lead${LEAD}.log 发回来分析。"
echo "========================================"
