# ChapterRL

ChapterRL is a reinforcement learning framework for long-form fiction generation. It combines chapter-level quality reward and rhythm-tension reward to improve story structure, reading quality, and narrative pacing.

## Project Structure

```text
ChapterRL/
├── configs/
│   ├── train_config.yaml
│   └── ds_config_h200.json
│
├── rdr/
│   ├── rdr_reward_model.py
│   ├── train_rdr_rm_continuous.py
│   └── build_rdr_seq_dataset_discrete.py
│
├── drt/
│   ├── as_model.py
│   └── compute_drt_reward.py
│
├── grpo/
│   └── train_rdr_rm_continuous.py
│
├── reward/
│   └── unified_reward.py
│
├── scripts/
│
└── data/
```

## Modules

### RDR

RDR is the chapter-level reading quality reward model.

Main files:

```text
rdr/rdr_reward_model.py
rdr/train_rdr_rm_continuous.py
rdr/build_rdr_seq_dataset_discrete.py
```

It uses a sliding window of chapters, usually `K=3`, to predict a continuous chapter quality score.

### DRT

DRT is the dynamic rhythm-tension reward.

Main files:

```text
drt/as_model.py
drt/compute_drt_reward.py
```

It uses chapter-level arousal scores to detect rhythm collapse.

### Unified Reward

The unified reward combines RDR and DRT.

Main file:

```text
reward/unified_reward.py
```

Reward form:

```text
Reward = 0.5 * RDR + 0.5 * DRT
```

### GRPO

GRPO is used for policy optimization.

Main file:

```text
grpo/train_rdr_rm_continuous.py
```

## Installation

```bash
pip install torch transformers peft deepspeed numpy scipy tqdm flask requests pyyaml
```

## Configuration

Main config file:

```text
configs/train_config.yaml
```

DeepSpeed config file:

```text
configs/ds_config_h200.json
```

## Build RDR Dataset

```bash
python rdr/build_rdr_seq_dataset_discrete.py \
  --novel_json data/rdr_novels.json \
  --rdr_labels data/rdr_labels_all.jsonl \
  --split_dir data/splits \
  --output_dir data \
  --K 3 \
  --min_N_t 1 \
  --min_chapters 10 \
  --max_chapters 300
```

## Train RDR Reward Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --master_addr 127.0.0.1 \
  --master_port 29502 \
  rdr/train_rdr_rm_continuous.py
```

## Train AS Model

```bash
python drt/train_as_model.py \
  --arousal_jsonl data/arousal.jsonl \
  --novel_json data/rdr_novels.json \
  --split_dir data/splits \
  --model_path /path/to/base_model \
  --output_dir checkpoints/as_model \
  --epochs 5 \
  --batch_size 8 \
  --lr 2e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_length 8192
```

## Start Reward Server

### RDR Only

```bash
python scripts/reward_server.py \
  --host 0.0.0.0 \
  --port 29601 \
  --rdr_checkpoint checkpoints/rdr_rm/best \
  --rdr_base_model /path/to/base_model \
  --drt_enabled false
```

### RDR + DRT

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/reward_server.py \
  --host 0.0.0.0 \
  --port 29601 \
  --rdr_checkpoint checkpoints/rdr_rm/best \
  --rdr_base_model /path/to/base_model \
  --as_checkpoint checkpoints/as_model/best \
  --as_base_model /path/to/base_model \
  --drt_enabled true \
  --format_penalty -0.5
```


## Train GRPO

Set reward server environment variables:

```bash
export CHAPTERRL_REWARD_MODE=rpc
export CHAPTERRL_REWARD_SERVER=http://127.0.0.1:29601
export CHAPTERRL_REWARD_TIMEOUT=300
```

Start GRPO training:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun \
  --nnodes 1 \
  --nproc_per_node 7 \
  --master_addr 127.0.0.1 \
  --master_port 29503 \
  grpo/train_rdr_rm_continuous.py
```
