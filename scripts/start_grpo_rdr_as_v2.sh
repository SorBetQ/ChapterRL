
set -e

export PYTHON=python3.10

ROOT_DIR="/mmu_vcg2_wjc_ssd/sunbingqian"
MODEL_PATH="${ROOT_DIR}/Qwen3-8B"
RDR_BASE_MODEL="${ROOT_DIR}/Qwen3-8B"
AS_BASE_MODEL="${ROOT_DIR}/Qwen3-8B"
RDR_CKPT="${ROOT_DIR}/checkpoints/rdr_rm_v2_direct_quality_ft/best"
AS_CKPT="${ROOT_DIR}/checkpoints/as_model/best"

CHAPTERRL="${ROOT_DIR}/ChapterRL"
SWIFT_DIR="${ROOT_DIR}/ms-swift"
LOG_DIR="${ROOT_DIR}/logs"
OUTPUT_DIR="${ROOT_DIR}/output/grpo_rdr_as_v2_ft"
DS_CONFIG="${CHAPTERRL}/configs/ds_config_grpo_8b_zero3.json"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ── 分布式环境变量 ───────────────────────────────────────────
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE=7
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# ── Reward RPC 配置 ──────────────────────────────────────────
REWARD_PORT=29601
export CHAPTERRL_REWARD_MODE=rpc
export CHAPTERRL_REWARD_SERVER="http://localhost:${REWARD_PORT}"
export CHAPTERRL_REWARD_TIMEOUT=600

# ── 关键：DS_ACCELERATOR 必须是 cuda（不能是 false） ─────────
export DS_ACCELERATOR=cuda
export ACCELERATE_USE_DEEPSPEED=false

# ── NCCL ─────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset NCCL_SOCKET_IFNAME

NCCL_LIB=$(python3.10 -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])" 2>/dev/null || echo "")
if [ -n "${NCCL_LIB}" ]; then
    export LD_LIBRARY_PATH="${NCCL_LIB}/lib:${LD_LIBRARY_PATH}"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "==========================================================="
echo "  GRPO RDR+AS v2 (Full-Param policy + RDR full + AS LoRA)"
echo "  🎯 v6 配置 - 2 epoch + 12K 生成长度 + 调优超参"
echo "==========================================================="
echo "  策略模型     : ${MODEL_PATH}"
echo "  RDR 奖励     : ${RDR_CKPT} (full-param)"
echo "  AS 奖励      : ${AS_CKPT} (LoRA)"
echo "  DRT          : 启用"
echo "  Reward GPU   : 0"
echo "  Train  GPU   : 1-7（7卡，num_generations=7）"
echo "  Output       : ${OUTPUT_DIR}"
echo "  ⚠️ Eval     : 已禁用（避免 ZeRO-3 + GRPO eval deque bug）"
echo "  Epochs       : 2"
echo "  LR           : 1e-5"
echo "  KL beta      : 0.02"
echo "  Max comp len : 12288"
echo "==========================================================="

# ── Step 1: 检查 checkpoint 完整性 ───────────────────────────
echo ""
echo "[Step 1] 检查 checkpoint 完整性..."
if [ ! -f "${RDR_CKPT}/model.pt" ]; then
    echo "  [ERROR] RDR checkpoint 缺少: ${RDR_CKPT}/model.pt"
    exit 1
fi
RDR_SIZE=$(du -sh "${RDR_CKPT}/model.pt" | cut -f1)
echo "  ✅ RDR: ${RDR_CKPT}/model.pt (${RDR_SIZE}, 期望 ~16G)"

echo "  AS  LoRA 文件检查："
for f in adapter_config.json adapter_model.safetensors score_head.pt as_meta.json; do
    if [ ! -f "${AS_CKPT}/${f}" ]; then
        echo "  [ERROR] AS checkpoint 缺少: ${AS_CKPT}/${f}"
        exit 1
    fi
    SIZE=$(du -sh "${AS_CKPT}/${f}" | cut -f1)
    echo "    ✅ ${f} (${SIZE})"
done
echo "  AS  meta: $(cat ${AS_CKPT}/as_meta.json)"

# ── Step 2: 检查 DS config ───────────────────────────────────
echo ""
echo "[Step 2] 检查 DeepSpeed config..."
if [ ! -f "${DS_CONFIG}" ]; then
    echo "  [ERROR] DS config 不存在: ${DS_CONFIG}"
    exit 1
fi
if ! grep -q '"train_micro_batch_size_per_gpu":\s*"auto"' "${DS_CONFIG}"; then
    echo "  [ERROR] DS config 中 train_micro_batch_size_per_gpu 不是 'auto'"
    exit 1
fi
echo "  ✅ DS config: ${DS_CONFIG}"

# ── Step 3: 检查关键依赖 ──────────────────────────────────────
echo ""
echo "[Step 3] 检查关键依赖..."
python3.10 -c "import trl" 2>/dev/null || { echo "  [ERROR] trl 未安装"; exit 1; }
python3.10 -c "import swift" 2>/dev/null || { echo "  [ERROR] ms-swift 未安装"; exit 1; }
python3.10 -c "import peft" 2>/dev/null || { echo "  [ERROR] peft 未安装"; exit 1; }
python3.10 -c "import flask" 2>/dev/null || { echo "  [ERROR] flask 未安装"; exit 1; }
echo "  ✅ trl  : $(python3.10 -c 'import trl; print(trl.__version__)')"
echo "  ✅ swift: $(python3.10 -c 'import swift; print(swift.__version__)')"
echo "  ✅ peft : $(python3.10 -c 'import peft; print(peft.__version__)')"

# ── Step 4: 验证 rdr_reward_model.py ─────────────────────────
echo ""
echo "[Step 4] 验证关键代码补丁..."
if grep -q 'os.environ\["DS_ACCELERATOR"\] = "false"' "${CHAPTERRL}/rdr/rdr_reward_model.py"; then
    echo "  [ERROR] rdr_reward_model.py 仍包含 DS_ACCELERATOR=false"
    exit 1
fi
echo "  ✅ rdr_reward_model.py 已修复"

# ── Step 5: 清理旧进程 ────────────────────────────────────────
echo ""
echo "[Step 5] 清理旧进程..."
pkill -f "reward_server.py" 2>/dev/null && echo "  killed old reward_server" || echo "  no old reward_server"
pkill -f "swift rlhf" 2>/dev/null && echo "  killed old swift" || echo "  no old swift"
sleep 3

if ss -lntp 2>/dev/null | grep -q ":${REWARD_PORT} "; then
    echo "  [WARN] port ${REWARD_PORT} still in use, force killing..."
    fuser -k "${REWARD_PORT}/tcp" 2>/dev/null
    sleep 2
fi

# ── Step 6: 启动 reward_server（GPU 0 独占）─────────────────
echo ""
echo "[Step 6] 启动 RDR+AS 奖励服务器 (port=${REWARD_PORT}, GPU=0)..."
REWARD_LOG="${LOG_DIR}/reward_server_rdr_as_v2_${TIMESTAMP}.log"
cd "${CHAPTERRL}"
CUDA_VISIBLE_DEVICES=0 nohup python3.10 reward/reward_server.py \
    --port "${REWARD_PORT}" \
    --rdr_checkpoint "${RDR_CKPT}" \
    --rdr_base_model "${RDR_BASE_MODEL}" \
    --as_checkpoint "${AS_CKPT}" \
    --as_base_model "${AS_BASE_MODEL}" \
    --drt_enabled true \
    > "${REWARD_LOG}" 2>&1 &
REWARD_PID=$!
echo "${REWARD_PID}" > "${LOG_DIR}/reward_server.pid"
echo "  PID  : ${REWARD_PID}"
echo "  日志 : ${REWARD_LOG}"

# ── Step 7: 等待 reward_server 健康检查通过 ─────────────────
echo ""
echo "[Step 7] 等待 reward_server 模型加载（最长 8 分钟）..."
HEALTH_CHECK_OK=0
HEALTH_RESPONSE=""

for i in $(seq 1 240); do
    HEALTH_RESPONSE=$(curl -sf "http://localhost:${REWARD_PORT}/health" 2>/dev/null || echo "")

    if echo "${HEALTH_RESPONSE}" | grep -qE '"ready_for_compute"\s*:\s*true'; then
        echo "  ✓ 奖励服务器已就绪 (${i}*2s = $((i*2))s)"
        HEALTH_CHECK_OK=1
        break
    fi

    if [ $((i % 30)) -eq 0 ]; then
        echo "    ...等待中 ($((i*2))s/480s)..."
        if [ $((i % 60)) -eq 0 ] && [ -n "${HEALTH_RESPONSE}" ]; then
            echo "    Latest health response: ${HEALTH_RESPONSE:0:300}"
        fi
    fi
    sleep 2
done

if [ ${HEALTH_CHECK_OK} -ne 1 ]; then
    echo "  [ERROR] reward_server 启动超时"
    echo "  Last health response: ${HEALTH_RESPONSE}"
    echo "  最后 30 行日志："
    tail -30 "${REWARD_LOG}"
    exit 1
fi

# ── Step 8: reward_server 冒烟测试 ──────────────────────────
echo ""
echo "[Step 8] reward_server 冒烟测试..."
SMOKE_TEST=$(curl -s -X POST "http://localhost:${REWARD_PORT}/compute" \
    -H "Content-Type: application/json" \
    -d '{"completions":["第一章\n初战告捷，将军意气风发。\n\n第二章\n敌军反攻，形势紧张。"]}' 2>&1)

SMOKE_REWARD=$(echo "${SMOKE_TEST}" | grep -oE '"rewards":\s*\[[-0-9.,e ]+\]' | head -c 200)
if [ -z "${SMOKE_REWARD}" ]; then
    echo "  [ERROR] 冒烟测试失败"
    echo "  返回内容: ${SMOKE_TEST:0:500}"
    exit 1
fi
echo "  ✅ Smoke test passed: ${SMOKE_REWARD}"

# ── Step 9: 修复 deepspeed 元数据 ────────────────────────────
python3.10 -c "import importlib.metadata; importlib.metadata.version('deepspeed')" 2>/dev/null || {
    DS_VERSION=$(python3.10 -c "import deepspeed; print(deepspeed.__version__)" 2>/dev/null || echo "0.14.0")
    DS_SITE=$(python3.10 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "/usr/local/lib/python3.10/dist-packages")
    DS_DIST="${DS_SITE}/deepspeed-${DS_VERSION}.dist-info"
    mkdir -p "${DS_DIST}"
    printf "Metadata-Version: 2.1\nName: deepspeed\nVersion: ${DS_VERSION}\n" > "${DS_DIST}/METADATA"
    echo "" > "${DS_DIST}/INSTALLER"
    echo "  ✅ 修补 deepspeed 元数据 (${DS_VERSION})"
}

# ── Step 10: 启动 GRPO 训练（GPU 1-7，7卡，全参数）──────────
# ⚠️ 关键参数：
#   --split_dataset_ratio 0   不切验证集，避免 eval 阶段崩溃
#   --eval_strategy no        显式禁用 evaluation
#   --save_steps 50           缩短为每 ~3.5 小时保存一次
#   --save_total_limit 3      限制 checkpoint 数量，控制磁盘
# 🎯 v6 调优超参：
#   --num_train_epochs 2      按要求改为 2 epoch
#   --learning_rate 1e-5      加速全参收敛
#   --beta 0.02               降 KL 约束
#   --warmup_steps 30         总步数少，warmup 缩短
#   --max_completion_length 12288   与 reward v6 上限 12000 对齐
#   --max_length 14336        prompt ~2K + completion 12K + buffer
#   --temperature 0.9         避免默认 1.0 过度发散
#   --top_p 0.95              核采样
echo ""
echo "[Step 10] 启动 GRPO 训练（GPU 1-7，7卡，全参数 + ZeRO-3，v6 超参）..."
TRAIN_LOG="${LOG_DIR}/grpo_rdr_as_v2_${TIMESTAMP}.log"
cd "${SWIFT_DIR}"

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --dataset "${CHAPTERRL}/data/grpo_prompts_swift.json" \
    --output_dir "${OUTPUT_DIR}" \
    --external_plugins "${CHAPTERRL}/grpo/swift_reward_plugin.py" \
    --reward_funcs chapterrl_unified \
    --num_generations 7 \
    --max_length 14336 \
    --max_completion_length 12288 \
    --temperature 0.9 \
    --top_p 0.95 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 7 \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --warmup_steps 30 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --beta 0.02 \
    --deepspeed "${DS_CONFIG}" \
    --bf16 true \
    --gradient_checkpointing true \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --eval_strategy no \
    --split_dataset_ratio 0 \
    --dataloader_num_workers 2 \
    --save_only_model true \
    --report_to wandb \
    --run_name "grpo_rdr_as_v2_ft_v6_${TIMESTAMP}" \
    > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${LOG_DIR}/grpo_train.pid"
echo "  PID  : ${TRAIN_PID}"
echo "  日志 : ${TRAIN_LOG}"

# ── Step 11: 总结 ──────────────────────────────────────────
echo ""
echo "==========================================================="
echo "  ✅ GRPO RDR+AS v2 启动完成（主实验，全参数训练，v6 配置）"
echo "==========================================================="
echo "  Reward Server PID : ${REWARD_PID} (GPU 0)"
echo "  GRPO Train    PID : ${TRAIN_PID} (GPU 1-7)"
echo "  ⚠️ Eval         : 已禁用"
echo "  Save interval     : 每 50 步（约 2.5 小时，v6 12K 生成）"
echo "  Save limit        : 最多保留 3 个 checkpoint"
echo ""
echo "  关键超参（v6）："
echo "    epochs                : 2"
echo "    learning_rate         : 1e-5"
echo "    beta (KL)             : 0.02"
echo "    warmup_steps          : 30"
echo "    max_completion_length : 12288"
echo "    temperature / top_p   : 0.9 / 0.95"
echo ""
echo "  日志监控："
echo "    tail -f ${TRAIN_LOG}"
echo "    tail -f ${REWARD_LOG}"
echo ""
echo "  停止命令："
echo "    kill \$(cat ${LOG_DIR}/grpo_train.pid)"
echo "    kill \$(cat ${LOG_DIR}/reward_server.pid)"
echo ""
echo "  健康检查："
echo "    curl -s http://localhost:${REWARD_PORT}/health | python3.10 -m json.tool"
echo "==========================================================="
