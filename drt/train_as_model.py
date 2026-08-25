import argparse
import json
import logging
import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig, AutoModel, AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau as scipy_kendalltau


class ArousalDataset(Dataset):
    def __init__(
        self,
        arousal_jsonl: str,
        split: str = "train",
        split_dir: Optional[str] = None,
        novel_json: Optional[str] = None,
        max_chars: int = 6000,
    ):
        self.max_chars = max_chars
        self.items: List[dict] = []

        split_ids = None
        if split_dir:
            fname = f"{split}_ids.json"
            path = os.path.join(split_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                ids = data["book_ids"] if isinstance(data, dict) and "book_ids" in data else data
                split_ids = {str(x) for x in ids}

        novel_texts: dict = {}
        if novel_json and os.path.exists(novel_json):
            with open(novel_json, "r", encoding="utf-8") as f:
                novels = json.load(f)
            for b in novels:
                bid = str(b["book_id"])
                novel_texts[bid] = [
                    ch.get("text", "") if isinstance(ch, dict) else str(ch)
                    for ch in b.get("chapters", [])
                ]

        skipped = 0
        with open(arousal_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    bid = str(rec["book_id"])
                    if split_ids is not None and bid not in split_ids:
                        continue

                    arousal_seq = rec["arousal_sequence"]
                    chapters_text = novel_texts.get(bid, [])

                    for t, scores in enumerate(arousal_seq):
                        if not scores:
                            continue
                        a_t = float(np.mean(scores))
                        if t < len(chapters_text):
                            text = chapters_text[t][:max_chars]
                        else:
                            text = ""
                        if not text.strip():
                            skipped += 1
                            continue
                        self.items.append({
                            "text": text,
                            "arousal": a_t,
                            "book_id": bid,
                            "chapter_idx": t,
                        })
                except Exception:
                    skipped += 1

        print(f"[ArousalDataset:{split}] {len(self.items)} samples (skipped={skipped})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, tokenizer, max_length: int = 2048):
    texts = [item["text"] for item in batch]
    arousals = torch.tensor([item["arousal"] for item in batch], dtype=torch.float32)

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "arousal": arousals,
        "book_ids": [item["book_id"] for item in batch],
    }


class ArousalScorer(nn.Module):
    def __init__(
        self,
        model_path: str,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
        )
        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.enable_input_require_grads()
        self.backbone.gradient_checkpointing_enable()

        hidden_size = config.hidden_size
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        for layer in self.score_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden = outputs.last_hidden_state[:, -1, :]
        scores = self.score_head(last_hidden.to(torch.float32)).squeeze(-1)
        return scores

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.backbone.save_pretrained(save_dir)
        torch.save(self.score_head.state_dict(), os.path.join(save_dir, "score_head.pt"))
        import json as _json
        meta = {"lora_r": self.backbone.peft_config["default"].r,
                "lora_alpha": self.backbone.peft_config["default"].lora_alpha}
        with open(os.path.join(save_dir, "as_meta.json"), "w") as f:
            _json.dump(meta, f)

    @classmethod
    def load_pretrained(cls, save_dir: str, base_model_path: str, **kwargs):
        from peft import PeftModel
        import json as _json
        meta_path = os.path.join(save_dir, "as_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = _json.load(f)
            kwargs.setdefault("lora_r", meta.get("lora_r", 32))
            kwargs.setdefault("lora_alpha", meta.get("lora_alpha", 64))

        model = cls(base_model_path, **kwargs)
        model.backbone = PeftModel.from_pretrained(model.backbone.base_model.model, save_dir)
        state = torch.load(os.path.join(save_dir, "score_head.pt"), map_location="cpu")
        model.score_head.load_state_dict(state)
        return model


def train_epoch(model, loader, optimizer, scheduler, scaler, device,
                logger, epoch, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc=f"[train epoch {epoch+1}]")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["arousal"].to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            preds = model(input_ids, attention_mask)
            loss = nn.functional.mse_loss(preds, targets)

        loss_scaled = loss / gradient_accumulation_steps
        scaler.scale(loss_scaled).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        if (step + 1) % 100 == 0:
            logger.info(f"  epoch={epoch+1} step={step+1} loss={loss.item():.6f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_targets, all_book_ids = [], [], []

    for batch in tqdm(loader, desc="[val]"):
        preds = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        ).cpu().float()
        all_preds.extend(preds.tolist())
        all_targets.extend(batch["arousal"].tolist())
        all_book_ids.extend(batch["book_ids"])

    p = np.array(all_preds)
    t = np.array(all_targets)

    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    pearson = float(pearsonr(p, t)[0]) if len(p) > 1 else 0.0
    spearman = float(spearmanr(p, t)[0]) if len(p) > 1 else 0.0

    book_taus = []
    books = {}
    for pred, tgt, bid in zip(all_preds, all_targets, all_book_ids):
        books.setdefault(bid, {"p": [], "t": []})
        books[bid]["p"].append(pred)
        books[bid]["t"].append(tgt)
    for bid, v in books.items():
        if len(v["p"]) >= 3:
            tau, _ = scipy_kendalltau(v["p"], v["t"])
            if not np.isnan(tau):
                book_taus.append(tau)
    kendall_book = float(np.mean(book_taus)) if book_taus else 0.0

    return {
        "val_mse": mse,
        "val_mae": mae,
        "val_pearson": pearson,
        "val_spearman": spearman,
        "kendall_book": kendall_book,
        "n_books_eval": len(book_taus),
    }


def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("as_train")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = setup_logger(args.output_dir)

    logger.info("=" * 60)
    logger.info("AS Model Training (ChapterRL, single-GPU A800/H200)")
    logger.info("=" * 60)
    logger.info(f"  arousal_jsonl : {args.arousal_jsonl}")
    logger.info(f"  novel_json    : {args.novel_json}")
    logger.info(f"  model_path    : {args.model_path}")
    logger.info(f"  output_dir    : {args.output_dir}")
    logger.info(f"  epochs        : {args.epochs}")
    logger.info(f"  batch_size    : {args.batch_size}")
    logger.info(f"  lr            : {args.lr}")
    logger.info(f"  lora_r        : {args.lora_r}")
    logger.info(f"  lora_alpha    : {args.lora_alpha}")
    logger.info(f"  max_length    : {args.max_length}")
    logger.info(f"  grad_accum    : {args.gradient_accumulation_steps}")
    logger.info(f"  device        : {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = ArousalDataset(
        args.arousal_jsonl, split="train",
        split_dir=args.split_dir, novel_json=args.novel_json,
    )
    val_ds = ArousalDataset(
        args.arousal_jsonl, split="val",
        split_dir=args.split_dir, novel_json=args.novel_json,
    )

    if len(train_ds) == 0:
        logger.warning("训练集为空！请检查 split_dir 或 arousal_jsonl 路径。")
        full_ds = ArousalDataset(
            args.arousal_jsonl, split="all",
            split_dir=None, novel_json=args.novel_json,
        )
        n = len(full_ds)
        n_train = int(n * 0.8)
        indices = list(range(n))
        random.shuffle(indices)
        from torch.utils.data import Subset
        train_ds = Subset(full_ds, indices[:n_train])
        val_ds = Subset(full_ds, indices[n_train:])
        logger.info(f"  随机切分: train={len(train_ds)}, val={len(val_ds)}")

    logger.info(f"  train_samples : {len(train_ds)}")
    logger.info(f"  val_samples   : {len(val_ds)}")

    _collate = lambda b: collate_fn(b, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=_collate,
    )

    model = ArousalScorer(
        args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    effective_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_optimizer_steps = effective_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_optimizer_steps)),
        num_training_steps=total_optimizer_steps,
    )
    scaler = torch.cuda.amp.GradScaler()

    best_spearman = -1.0
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, logger, epoch, args.gradient_accumulation_steps,
        )
        logger.info(f"[epoch {epoch+1}] avg_train_loss={train_loss:.6f}")

        metrics = eval_epoch(model, val_loader, device)
        logger.info(
            f"[epoch {epoch+1}] "
            f"val_mse={metrics['val_mse']:.6f}  "
            f"val_mae={metrics['val_mae']:.4f}  "
            f"pearson={metrics['val_pearson']:.4f}  "
            f"spearman={metrics['val_spearman']:.4f}  "
            f"kendall_book={metrics['kendall_book']:.4f}"
            f"  (eval on {metrics['n_books_eval']} books)"
        )

        if metrics["val_spearman"] > best_spearman:
            best_spearman = metrics["val_spearman"]
            model.save_pretrained(os.path.join(args.output_dir, "best"))
            logger.info(f"  ✓ best model saved (val_spearman={best_spearman:.4f})")

        model.save_pretrained(os.path.join(args.output_dir, f"epoch{epoch+1}"))

    logger.info(f"Training complete. Best val Spearman: {best_spearman:.4f}")
    logger.info(f"Best model saved at: {os.path.join(args.output_dir, 'best')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AS model (ChapterRL, single-GPU)")
    parser.add_argument("--arousal_jsonl", type=str, default=None)
    parser.add_argument("--novel_json", type=str, default=None)
    parser.add_argument("--split_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()
    main(args)