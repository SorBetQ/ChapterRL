
set -e

export PYTHON=python3.10

ROOT_DIR="/mmu_vcg2_wjc_ssd/sunbingqian"
MODEL_PATH="${ROOT_DIR}/Qwen3-8B"
AS_BASE_MODEL="${ROOT_DIR}/Qwen3-8B"
AS_CKPT="${ROOT_DIR}/checkpoints/as_model_ft/best"

CHAPTERRL="${ROOT_DIR}/ChapterRL"
SWIFT_DIR="${ROOT_DIR}/ms-swift"
LOG_DIR="${ROOT_DIR}/logs"
OUTPUT_DIR="${ROOT_DIR}/output/grpo_as_only_ft"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

REWARD_PORT=29601
export CHAPTERRL_REWARD_MODE=rpc
export CHAPTERRL_REWARD_SERVER="http://localhost:${REWARD_PORT}"
export CHAPTERRL_REWARD_TIMEOUT=600

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
unset NCCL_SOCKET_IFNAME

NCCL_LIB=$(python3.10 -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])" 2>/dev/null || echo "")
if [ -n "${NCCL_LIB}" ]; then
    export LD_LIBRARY_PATH="${NCCL_LIB}/lib:${LD_LIBRARY_PATH}"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================="
echo "GRPO 消融 - 仅 AS 奖励"
echo "========================================="
echo "  策略模型    : Qwen3-8B"
echo "  AS 奖励     : ${AS_CKPT}"
echo "  RDR         : 禁用"
echo "  训练卡      : GPU 1-7（7卡，num_generations=7）"
echo "========================================="

# ── Step 1: 检查 AS checkpoint ───────────────────────────────
echo "[Step 1] 检查 AS checkpoint..."
if [ ! -f "${AS_CKPT}/score_head.pt" ]; then
    echo "  [ERROR] AS checkpoint 缺少: ${AS_CKPT}/score_head.pt"
    exit 1
fi
echo "  ✅ AS: ${AS_CKPT} (head=$(du -sh ${AS_CKPT}/score_head.pt | cut -f1))"

# ── Step 2: 检查依赖 ──────────────────────────────────────────
echo "[Step 2] 检查关键依赖..."
python3.10 -c "import trl; trl.__version__" 2>/dev/null || { echo "  [ERROR] trl 未安装到 python3.10"; exit 1; }
python3.10 -c "import swift" 2>/dev/null || { echo "  [ERROR] ms-swift 未安装到 python3.10"; exit 1; }
echo "  ✅ trl: $(python3.10 -c 'import trl; print(trl.__version__)')"
echo "  ✅ swift: $(python3.10 -c 'import swift; print(swift.__version__)')"

# ── Step 3: 清理旧进程 ────────────────────────────────────────
echo "[Step 3] 清理旧进程..."
pkill -f "reward_server.py" 2>/dev/null || true
sleep 2

# ── Step 4: 启动 AS 奖励服务器（GPU 0 独占）──────────────────
echo "[Step 4] 启动 AS 奖励服务器 (port=${REWARD_PORT}, GPU=0)..."
REWARD_LOG="${LOG_DIR}/reward_server_as_only_${TIMESTAMP}.log"
cd "${CHAPTERRL}"
CUDA_VISIBLE_DEVICES=0 nohup python3.10 reward/reward_server.py \
    --port "${REWARD_PORT}" \
    --as_checkpoint "${AS_CKPT}" \
    --as_base_model "${AS_BASE_MODEL}" \
    --drt_enabled true \
    > "${REWARD_LOG}" 2>&1 &
REWARD_PID=$!
echo "${REWARD_PID}" > "${LOG_DIR}/reward_server.pid"
echo "  PID  : ${REWARD_PID}"
echo "  日志 : ${REWARD_LOG}"

echo ""
echo "[等待] AS 模型加载中（最长 3 分钟）..."
for i in $(seq 1 90); do
    if curl -sf "http://localhost:${REWARD_PORT}/health" 2>/dev/null | grep -q '"healthy"'; then
        echo "  ✓ 奖励服务器已就绪（${i}×2s）"
        break
    fi
    [ "${i}" -eq 90 ] && { echo "  [ERROR] 超时，查看: tail -50 ${REWARD_LOG}"; exit 1; }
    sleep 2
done

# ── Step 5: 修复 DeepSpeed / trl 元数据 ──────────────────────
python3.10 -c "import importlib.metadata; importlib.metadata.version('deepspeed')" 2>/dev/null || {
    DS_VERSION=$(python3.10 -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null || echo "0.14.0")
    DS_SITE=$(python3.10 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "/usr/local/lib/python3.10/dist-packages")
    DS_DIST="${DS_SITE}/deepspeed-${DS_VERSION}.dist-info"
    mkdir -p "${DS_DIST}"
    printf "Metadata-Version: 2.1\nName: deepspeed\nVersion: ${DS_VERSION}\n" > "${DS_DIST}/METADATA"
    echo "" > "${DS_DIST}/INSTALLER"
}

# ── Step 6: 启动 GRPO 训练（GPU 1-7，7卡）──────────────────
echo ""
echo "[Step 6] 启动 GRPO 训练（GPU 1-7，7卡）..."
TRAIN_LOG="${LOG_DIR}/grpo_as_only_${TIMESTAMP}.log"
cd "${SWIFT_DIR}"

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --dataset "${CHAPTERRL}/data/grpo_prompts_swift.json" \
    --output_dir "${OUTPUT_DIR}" \
    --external_plugins "${CHAPTERRL}/grpo/swift_reward_plugin.py" \
    --reward_funcs chapterrl_unified \
    --num_generations 7 \
    --max_length 32768 \
    --max_completion_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 7 \
    --num_train_epochs 3 \
    --learning_rate 5e-6 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --beta 0.04 \
    --deepspeed "${CHAPTERRL}/configs/ds_config_8b_zero3.json" \
    --bf16 true \
    --gradient_checkpointing true \
    --save_steps 100 \
    --logging_steps 10 \
    --split_dataset_ratio 0.05 \
    --dataloader_num_workers 2 \
    --save_only_model true \
    --report_to wandb \
    --run_name "grpo_as_only_ft_${TIMESTAMP}" \
    > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${LOG_DIR}/grpo_train.pid"
echo "  PID  : ${TRAIN_PID}"
echo "  日志 : ${TRAIN_LOG}"

echo ""
echo "========================================="
echo "GRPO AS-Only 启动完成！"
echo "  监控: tail -f ${TRAIN_LOG}"
echo "  停止: kill \$(cat ${LOG_DIR}/grpo_train.pid)"
echo "         kill \$(cat ${LOG_DIR}/reward_server.pid)"
echo "========================================="
