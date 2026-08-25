#!/bin/bash
# RDR v2 直接质量评分奖励模型训练 - 全参数 + DeepSpeed ZeRO-3

set -e

ROOT_DIR="/mmu_vcg2_wjc_ssd/sunbingqian"
CHAPTERRL="${ROOT_DIR}/ChapterRL"
LOG_DIR="${ROOT_DIR}/logs"
CKPT_DIR="${ROOT_DIR}/checkpoints/rdr_rm_v2_direct_quality_ft"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/rdr_v2_direct_quality_ft_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/rdr_v2_direct_quality_ft.pid"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

# ============================================================
# NCCL 调试与超时配置（解决之前 600s reduce_scatter 卡死问题）
# ============================================================
# 启用 FlightRecorder：卡死时自动 dump 每 rank 的 collective trace
export NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${LOG_DIR}/nccl_trace_${TIMESTAMP}"

# 增大默认超时到 30 分钟（原 10 分钟）
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800

# 让 NCCL 错误立刻抛出，避免卡 600 秒
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# IB 网络容错（防抖动）
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=10

# 防止 DataLoader worker 跟主进程抢 CPU
export OMP_NUM_THREADS=4

# PyTorch 内存碎片整理（可选）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ============================================================

cd "${ROOT_DIR}"

nohup torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29502 \
  "${CHAPTERRL}/rdr/train_rdr_rm_continuous.py" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "RDR 全参数 ZeRO-3 训练已启动"
echo "PID : ${PID}"
echo "LOG : ${LOG_FILE}"
echo "CKPT: ${CKPT_DIR}"
echo ""
echo "查看实时日志: tail -f ${LOG_FILE}"
echo "查看每 rank 进度: grep '\[Rank' ${LOG_FILE} | tail -50"
echo "查看 sampler 配置: grep '\[Sampler\]' ${LOG_FILE}"
