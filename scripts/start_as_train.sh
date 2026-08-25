#!/bin/bash
# AS 模型训练 - 全参数 + DeepSpeed ZeRO-3
# 对齐 RDR 训练的最终成熟方案（NCCL 调试 + 30min 超时 + 防卡死配置）

set -e

ROOT_DIR="/mmu_vcg2_wjc_ssd/sunbingqian"
MODEL_PATH="${ROOT_DIR}/Qwen3-8B"
CHAPTERRL="${ROOT_DIR}/ChapterRL"
LOG_DIR="${ROOT_DIR}/logs"
OUTPUT_DIR="${ROOT_DIR}/checkpoints/as_model_ft"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRAIN_LOG="${LOG_DIR}/as_train_ft_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/as_train_ft.pid"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ============================================================
# NCCL 调试与超时配置
# ============================================================
# 启用 FlightRecorder：卡死时自动 dump 每 rank 的 collective trace
export NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${LOG_DIR}/nccl_trace_as_${TIMESTAMP}"

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

# PyTorch 内存碎片整理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ============================================================

cd "${ROOT_DIR}"

nohup torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29503 \
  "${CHAPTERRL}/drt/train_as_model.py" \
  --arousal_jsonl "${CHAPTERRL}/data/arousal_train.jsonl" \
  --novel_json    "${CHAPTERRL}/data/rdr_novels.json" \
  --split_dir     "${CHAPTERRL}/data/splits" \
  --model_path    "${MODEL_PATH}" \
  --output_dir    "${OUTPUT_DIR}" \
  --epochs        5 \
  --batch_size    8 \
  --lr            5e-6 \
  --max_length    2048 \
  --seed          42 \
  --log_interval  50 \
  --save_every_n_epochs 1 \
  --deepspeed_config "ChapterRL/configs/ds_config_8b_zero3.json" \
  > "${TRAIN_LOG}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "✅ AS 全参数 ZeRO-3 训练已启动"
echo "   PID : ${PID}"
echo "   LOG : ${TRAIN_LOG}"
echo "   CKPT: ${OUTPUT_DIR}"
echo ""
echo "实时查看日志：tail -f ${TRAIN_LOG}"
echo "查看每 rank 进度：grep '\[Rank' ${TRAIN_LOG} | tail -50"
echo "终止训练：kill -9 ${PID} && pkill -9 -f train_as_model"
