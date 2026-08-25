import os
import sys
import json
import yaml
import time
import random
import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoModel, AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class RDRRegressionModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        use_lora: bool = False,
    ):
        super().__init__()
        self.use_lora = use_lora

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )
        if hasattr(self.backbone.config, 'use_cache'):
            self.backbone.config.use_cache = False
        if hasattr(self.backbone, 'gradient_checkpointing_enable'):
            self.backbone.gradient_checkpointing_enable()

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.enable_input_require_grads()
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.backbone.enable_input_require_grads()

        hidden_size = self.backbone.config.hidden_size
        self.regression = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        seq_lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), seq_lengths]
        pred = self.regression(pooled).squeeze(-1)
        return pred


class RDRDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _build_text_from_window(ex: Dict) -> str:
        texts = ex.get('texts', [])
        titles = ex.get('titles', [])
        if not isinstance(texts, list) or len(texts) == 0:
            raise KeyError("missing 'texts' list in windowed RDR sample")

        chunks = []
        for i, txt in enumerate(texts):
            if txt is None or txt == '<empty>':
                continue
            title = titles[i] if isinstance(titles, list) and i < len(titles) else ''
            title = (title or '').strip()
            txt = str(txt).strip()
            if title:
                chunks.append(f"{title}\n{txt}")
            else:
                chunks.append(txt)

        if not chunks:
            last_txt = texts[-1]
            if last_txt is None:
                raise ValueError('all texts in window are empty')
            return str(last_txt)
        return "\n\n".join(chunks)

    @staticmethod
    def _get_last_valid_numeric(values, field_name: str) -> float:
        if not isinstance(values, list) or len(values) == 0:
            raise KeyError(f"missing '{field_name}' list in windowed RDR sample")
        for v in reversed(values):
            if v is not None:
                return float(v)
        raise ValueError(f"all entries in '{field_name}' are null")

    def __getitem__(self, idx):
        ex = self.samples[idx]
        try:
            if 'texts' in ex:
                text = self._build_text_from_window(ex)
                q_hat = self._get_last_valid_numeric(ex.get('labels'), 'labels')
                log_n_t = self._get_last_valid_numeric(ex.get('weights'), 'weights')
                book_id = ex.get('book_id', 'unknown')
                q_t_variance = float(ex.get('q_t_variance', 0.0))
                raw_chapter_count = len([t for t in ex.get('texts', []) if t not in [None, '<empty>']])
            elif 'text' in ex:
                text = ex['text']
                q_hat = float(ex['Q_hat'])
                log_n_t = float(ex.get('log_N_t', 1.0))
                book_id = ex.get('book_id', 'unknown')
                q_t_variance = float(ex.get('q_t_variance', 0.0))
                raw_chapter_count = 1
            else:
                raise KeyError(f"sample has neither 'texts' nor 'text', keys={list(ex.keys())}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse RDR sample at idx={idx}, keys={list(ex.keys())}, "
                f"book_id={ex.get('book_id', 'unknown')}: {e}"
            ) from e

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            'input_ids': torch.tensor(enc['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(enc['attention_mask'], dtype=torch.long),
            'Q_hat': torch.tensor(q_hat, dtype=torch.float32),
            'log_N_t': torch.tensor(log_n_t, dtype=torch.float32),
            'book_id': book_id,
            'q_t_variance': torch.tensor(q_t_variance, dtype=torch.float32),
            'raw_chapter_count': raw_chapter_count,
        }


class GroupBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: RDRDataset,
        batch_size: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        max_chapters_per_book_in_batch: int = 4,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        self.max_chapters_per_book_in_batch = max(1, max_chapters_per_book_in_batch)

        self.book_to_indices = defaultdict(list)
        for idx, ex in enumerate(self.dataset.samples):
            book_id = ex.get('book_id', 'unknown')
            self.book_to_indices[book_id].append(idx)
        self.all_book_ids = sorted(self.book_to_indices.keys())

        self._num_batches_per_rank = self._compute_num_batches_per_rank()
        self._calibrated = False

    def _compute_num_batches_per_rank(self) -> int:
        step = self.max_chapters_per_book_in_batch
        total_chunks = 0
        for book_id in self.all_book_ids:
            n = len(self.book_to_indices[book_id])
            total_chunks += (n + step - 1) // step
        chunks_per_rank = total_chunks // self.world_size
        chunks_per_batch = max(1, self.batch_size // step)
        num_batches = chunks_per_rank // chunks_per_batch
        return max(1, num_batches)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _build_batches(self) -> List[List[int]]:
        g = random.Random(self.seed + self.epoch)

        books = self.all_book_ids[:]
        g.shuffle(books)
        all_chunks = []
        step = self.max_chapters_per_book_in_batch
        for book_id in books:
            idxs = self.book_to_indices[book_id][:]
            g.shuffle(idxs)
            for i in range(0, len(idxs), step):
                chunk = idxs[i:i + step]
                if chunk:
                    all_chunks.append(chunk)

        g.shuffle(all_chunks)

        chunks_per_rank = len(all_chunks) // self.world_size
        start = self.rank * chunks_per_rank
        end = start + chunks_per_rank
        local_chunks = all_chunks[start:end]

        batches = []
        batch = []
        for chunk in local_chunks:
            if len(batch) + len(chunk) > self.batch_size:
                if len(batch) > 0:
                    batches.append(batch)
                batch = []
            batch.extend(chunk)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._build_batches()
        target = self._num_batches_per_rank

        if len(batches) >= target:
            batches = batches[:target]
        else:
            if len(batches) == 0:
                raise RuntimeError(
                    f"Rank {self.rank} epoch {self.epoch}: produced 0 batches "
                    f"but target={target}. Check sampler config (batch_size too large?)."
                )
            deficit = target - len(batches)
            base_n = len(batches)
            extra = [batches[i % base_n] for i in range(deficit)]
            batches = batches + extra

        assert len(batches) == target, \
            f"Rank {self.rank}: got {len(batches)} batches, expected {target}"

        for b in batches:
            yield b

    def __len__(self):
        return self._num_batches_per_rank


def calibrate_sampler_length(sampler: GroupBatchSampler, world_size: int, device, name: str = "") -> int:
    try:
        actual_batches = sampler._build_batches()
        actual = len(actual_batches)
    except Exception as e:
        print(f"[Sampler:{name}] _build_batches failed on rank {sampler.rank}: {e}", flush=True)
        actual = 0

    if dist.is_initialized() and world_size > 1:
        t = torch.tensor([actual], dtype=torch.long, device=device)
        gathered = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(gathered, t)
        all_counts = [g.item() for g in gathered]
        global_min = min(all_counts)
        if is_main_process():
            print(f"[Sampler:{name}] per_rank actual batches = {all_counts}, "
                  f"using global_min = {global_min}", flush=True)
    else:
        global_min = actual
        print(f"[Sampler:{name}] single-process: batches = {global_min}", flush=True)

    if global_min < 1:
        raise RuntimeError(f"[Sampler:{name}] global_min batches = {global_min}, "
                           f"cannot train. Check data and batch_size.")

    sampler._num_batches_per_rank = global_min
    sampler._calibrated = True
    return global_min


def collate_fn(batch):
    input_ids = [x['input_ids'] for x in batch]
    attention_mask = [x['attention_mask'] for x in batch]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'Q_hats': torch.stack([x['Q_hat'] for x in batch]),
        'log_N_t': torch.stack([x['log_N_t'] for x in batch]),
        'book_ids': [x['book_id'] for x in batch],
        'q_t_variances': torch.stack([x['q_t_variance'] for x in batch]),
        'raw_chapter_counts': torch.tensor([x['raw_chapter_count'] for x in batch], dtype=torch.long),
    }


def huber_loss_weighted(preds, targets, weights, delta=0.5):
    err = preds - targets
    abs_err = err.abs()
    quadratic = torch.minimum(abs_err, torch.tensor(delta, device=err.device))
    linear = abs_err - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    w = weights.clamp_min(0)
    denom = w.sum().clamp_min(1e-8)
    w = w / denom
    return (loss * w).sum()


def intra_book_rank_loss(preds, targets, book_ids, q_t_variances=None):
    unique_books = sorted(set(book_ids))
    losses = []
    pair_count = 0

    for book in unique_books:
        idxs = [i for i, b in enumerate(book_ids) if b == book]
        if len(idxs) < 2:
            continue
        idxs_t = torch.tensor(idxs, device=preds.device, dtype=torch.long)
        p = preds.index_select(0, idxs_t)
        t = targets.index_select(0, idxs_t)
        pdiff = p[:, None] - p[None, :]
        tdiff = t[:, None] - t[None, :]

        mask = torch.triu(torch.ones_like(pdiff, dtype=torch.bool), diagonal=1)
        diff_loss = (pdiff - tdiff) ** 2

        if q_t_variances is not None:
            v = q_t_variances.index_select(0, idxs_t)
            pair_w = 1.0 + 0.5 * (v[:, None] + v[None, :])
            diff_loss = diff_loss * pair_w

        sel = diff_loss[mask]
        if sel.numel() > 0:
            losses.append(sel.mean())
            pair_count += sel.numel()

    if len(losses) == 0:
        return torch.tensor(0.0, device=preds.device), 0
    return torch.stack(losses).mean(), pair_count


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = world_size > 1
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(minutes=30),
        )
        torch.cuda.set_device(local_rank)
    return local_rank, world_size, is_distributed


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def all_reduce_mean(value: float, device):
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t.item()


def assert_dataloader_len_equal(loader, device, name=""):
    rank_id = dist.get_rank() if dist.is_initialized() else 0
    local_len = len(loader)
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered = [torch.zeros(1, dtype=torch.long, device=device)
                    for _ in range(dist.get_world_size())]
        local_t = torch.tensor([local_len], dtype=torch.long, device=device)
        dist.all_gather(gathered, local_t)
        lens = [t.item() for t in gathered]
        if len(set(lens)) > 1:
            raise RuntimeError(
                f"[{name}] DataLoader length mismatch across ranks: {lens}. "
                f"This will cause NCCL deadlock under ZeRO-3. Check sampler calibration."
            )
        if is_main_process():
            print(f"[{name}] All ranks have {local_len} batches (synchronized)", flush=True)
    return local_len


def load_config(config_path=None):
    cfg_path = config_path if config_path and os.path.isabs(config_path) else os.path.join(_PROJECT_ROOT, config_path)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    mode = cfg.get('mode')
    overrides = cfg.get('mode_overrides', {}).get(mode, {})
    merged = json.loads(json.dumps(cfg))
    for section, section_dict in overrides.items():
        if isinstance(section_dict, dict):
            merged.setdefault(section, {})
            merged[section].update(section_dict)
        else:
            merged[section] = section_dict
    return merged


def prepare_paths(config: Dict):
    root_dir = config['paths'].get('root_dir', str(_PROJECT_ROOT))
    paths = {
        'root_dir': root_dir,
        'model_path': os.path.join(root_dir, ''),
        'train_jsonl': os.path.join(root_dir, config['paths']['rdr_train_jsonl']),
        'val_jsonl': os.path.join(root_dir, config['paths']['rdr_val_jsonl']),
        'output_dir': os.path.join(root_dir, config['paths']['rdr_output_dir']),
    }
    os.makedirs(paths['output_dir'], exist_ok=True)
    return paths


def save_consolidated_bf16(model_engine, save_dir: str):
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()

    state = model_engine._zero3_consolidated_16bit_state_dict()

    if is_main_process():
        try:
            os.makedirs(save_dir, exist_ok=True)
            tmp_path = os.path.join('', f'rdr_model_{os.getpid()}.pt')
            torch.save(state, tmp_path)
            target_path = os.path.join(save_dir, 'model.pt')
            import shutil
            shutil.move(tmp_path, target_path)
            print(f"  ✓ Saved consolidated bf16 to {target_path}", flush=True)
        except Exception as e:
            print(f"  ✗ Save failed: {e}", flush=True)

    del state
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()


@torch.no_grad()
def evaluate(model_engine, val_loader, device, huber_delta=0.5, rank_loss_weight=0.0):
    model_engine.eval()
    total_huber = 0.0
    total_rank = 0.0
    total_loss = 0.0
    total_pairs = 0
    n_batches = 0
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        Q_hats = batch['Q_hats'].to(device, non_blocking=True)
        weights = batch['log_N_t'].to(device, non_blocking=True)
        book_ids = batch['book_ids']
        q_t_variances = batch['q_t_variances'].to(device, non_blocking=True)

        preds = model_engine(input_ids, attention_mask)
        huber = huber_loss_weighted(preds, Q_hats, weights, delta=huber_delta)
        rank, pair_count = intra_book_rank_loss(preds, Q_hats, book_ids, q_t_variances)
        loss = huber + rank_loss_weight * rank

        total_huber += huber.item()
        total_rank += rank.item()
        total_loss += loss.item()
        total_pairs += pair_count
        n_batches += 1

    return {
        'total_loss': total_loss / max(n_batches, 1),
        'huber_loss': total_huber / max(n_batches, 1),
        'rank_loss': total_rank / max(n_batches, 1),
        'same_book_pairs': total_pairs / max(n_batches, 1),
    }


def train_epoch(model_engine, train_loader, device, epoch,
                rank_loss_weight=0.0, huber_delta=0.5, log_interval=50):
    model_engine.train()
    total_loss = 0.0
    total_huber = 0.0
    total_rank = 0.0
    total_pairs = 0
    n_batches = 0

    rank_id = dist.get_rank() if dist.is_initialized() else 0

    total_batches_in_loader = assert_dataloader_len_equal(train_loader, device, name=f"train-epoch{epoch}")

    t_iter_start = time.time()
    for batch_idx, batch in enumerate(train_loader):
        t_load = time.time() - t_iter_start

        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        Q_hats = batch['Q_hats'].to(device, non_blocking=True)
        weights = batch['log_N_t'].to(device, non_blocking=True)
        book_ids = batch['book_ids']
        q_t_variances = batch['q_t_variances'].to(device, non_blocking=True)
        seq_len = input_ids.shape[1]
        unique_books_in_batch = len(set(book_ids))

        t_step_start = time.time()
        preds = model_engine(input_ids, attention_mask)
        huber = huber_loss_weighted(preds, Q_hats, weights, delta=huber_delta)
        if rank_loss_weight > 0:
            rank, pair_count = intra_book_rank_loss(preds, Q_hats, book_ids, q_t_variances)
            loss = huber + rank_loss_weight * rank
        else:
            rank = torch.tensor(0.0, device=device)
            pair_count = 0
            loss = huber

        model_engine.backward(loss)
        model_engine.step()
        torch.cuda.synchronize()
        t_step = time.time() - t_step_start

        total_loss += loss.item()
        total_huber += huber.item()
        total_rank += rank.item() if rank_loss_weight > 0 else 0.0
        total_pairs += pair_count
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"[Rank {rank_id}] Epoch {epoch} batch {batch_idx + 1}/{total_batches_in_loader} "
                f"seq_len={seq_len} books_in_batch={unique_books_in_batch} pairs={pair_count} "
                f"load={t_load:.2f}s step={t_step:.2f}s "
                f"loss={total_loss / n_batches:.4f} "
                f"huber={total_huber / n_batches:.4f} "
                f"rank={total_rank / n_batches:.4f}",
                flush=True,
            )

        if batch_idx < 5:
            print(
                f"[Rank {rank_id}] init batch {batch_idx} "
                f"books_in_batch={unique_books_in_batch} pairs={pair_count}",
                flush=True,
            )

        t_iter_start = time.time()

    if dist.is_initialized():
        if is_main_process():
            print(f"[Epoch {epoch}] all ranks reached end of epoch, waiting at barrier ...", flush=True)
        dist.barrier()
        if is_main_process():
            print(f"[Epoch {epoch}] passed barrier, continuing to validation", flush=True)

    return {
        'total_loss': total_loss / max(n_batches, 1),
        'huber_loss': total_huber / max(n_batches, 1),
        'rank_loss': total_rank / max(n_batches, 1) if rank_loss_weight > 0 else 0.0,
        'same_book_pairs': total_pairs / max(n_batches, 1),
    }


def main():
    config = load_config()
    local_rank, world_size, is_distributed = init_distributed()
    device = torch.device(f'cuda:{local_rank}')

    seed = config.get('distributed', {}).get('seed', 42)
    set_seed(seed + local_rank)

    paths = prepare_paths(config)
    tokenizer = AutoTokenizer.from_pretrained(paths['model_path'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rdr_cfg = config['rdr']
    train_dataset = RDRDataset(paths['train_jsonl'], tokenizer, max_length=rdr_cfg.get('max_length', 2048))
    val_dataset = RDRDataset(paths['val_jsonl'], tokenizer, max_length=rdr_cfg.get('max_length', 2048))

    train_batch_sampler = GroupBatchSampler(
        train_dataset,
        batch_size=rdr_cfg['per_device_batch_size'],
        world_size=world_size,
        rank=local_rank,
        seed=42,
        max_chapters_per_book_in_batch=rdr_cfg.get('max_chapters_per_book_in_batch', 4),
        drop_last=True,
    )
    val_batch_sampler = GroupBatchSampler(
        val_dataset,
        batch_size=rdr_cfg['per_device_batch_size'],
        world_size=world_size,
        rank=local_rank,
        seed=142,
        max_chapters_per_book_in_batch=rdr_cfg.get('max_chapters_per_book_in_batch', 4),
        drop_last=True,
    )

    n_train = calibrate_sampler_length(train_batch_sampler, world_size, device, name="train")
    n_val = calibrate_sampler_length(val_batch_sampler, world_size, device, name="val")

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=rdr_cfg.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=rdr_cfg.get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if is_main_process():
        print(f"[Sampler] FINAL train batches/rank={n_train}, val batches/rank={n_val}", flush=True)

    use_lora = config.get('model', {}).get('use_lora', False)
    model = RDRRegressionModel(
        paths['model_path'],
        lora_r=config['model'].get('lora_r', 64),
        lora_alpha=config['model'].get('lora_alpha', 128),
        lora_dropout=config['model'].get('lora_dropout', 0.05),
        use_lora=use_lora,
    )

    if is_main_process():
        mode_name = 'LoRA' if use_lora else 'Full-param'
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Model] {mode_name} | trainable: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.2f}%)", flush=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=rdr_cfg['lr'],
        weight_decay=rdr_cfg.get('weight_decay', 0.01),
    )

    ds_config = config.get('deepspeed', {}).get('config_file', '')
    ds_config = ds_config if os.path.isabs(ds_config) else os.path.join(_PROJECT_ROOT, ds_config)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
        config=ds_config,
    )

    best_val = float('inf')
    epochs = rdr_cfg['epochs']
    rank_loss_weight = rdr_cfg.get('rank_loss_weight', 0.0)
    huber_delta = rdr_cfg.get('huber_delta', 0.5)
    log_interval = rdr_cfg.get('log_interval', 50)
    save_full_ckpt_every_n = rdr_cfg.get('save_full_ckpt_every_n_epochs', 0)

    for epoch in range(1, epochs + 1):
        train_batch_sampler.set_epoch(epoch)
        val_batch_sampler.set_epoch(epoch)

        if is_main_process():
            print(f"\n{'=' * 80}\nEpoch {epoch}/{epochs}\n{'=' * 80}", flush=True)

        train_metrics = train_epoch(
            model_engine,
            train_loader,
            device,
            epoch,
            rank_loss_weight=rank_loss_weight,
            huber_delta=huber_delta,
            log_interval=log_interval,
        )

        assert_dataloader_len_equal(val_loader, device, name=f"val-epoch{epoch}")

        val_metrics = evaluate(
            model_engine,
            val_loader,
            device,
            huber_delta=huber_delta,
            rank_loss_weight=rank_loss_weight,
        )

        train_loss = all_reduce_mean(train_metrics['total_loss'], device)
        train_pairs = all_reduce_mean(train_metrics['same_book_pairs'], device)
        val_loss = all_reduce_mean(val_metrics['total_loss'], device)
        val_huber = all_reduce_mean(val_metrics['huber_loss'], device)
        val_rank = all_reduce_mean(val_metrics['rank_loss'], device)
        val_pairs = all_reduce_mean(val_metrics['same_book_pairs'], device)

        if is_main_process():
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} train_pairs={train_pairs:.1f} | "
                f"val_loss={val_loss:.4f} val_huber={val_huber:.4f} "
                f"val_rank={val_rank:.4f} val_pairs={val_pairs:.1f}",
                flush=True,
            )

        if val_loss < best_val:
            best_val = val_loss
            best_dir = os.path.join(paths['output_dir'], 'best')
            save_consolidated_bf16(model_engine, best_dir)
            if is_main_process():
                print(f"[Best] saved | val_loss={best_val:.4f}", flush=True)

        if save_full_ckpt_every_n > 0 and (epoch % save_full_ckpt_every_n == 0):
            epoch_dir = os.path.join(paths['output_dir'], f'epoch_{epoch}')
            save_consolidated_bf16(model_engine, epoch_dir)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()