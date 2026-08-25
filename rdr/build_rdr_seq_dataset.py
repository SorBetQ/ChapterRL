import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Optional


def load_split_ids(split_dir: Optional[str]) -> tuple:
    if not split_dir:
        return None, None, None
    results = []
    for fname in ("train_ids.json", "val_ids.json", "test_ids.json"):
        path = os.path.join(split_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and "book_ids" in data:
                ids = data["book_ids"]
            else:
                ids = data
            results.append(set(str(x) for x in ids))
        else:
            results.append(None)
    return tuple(results)


def build_dataset(
    novel_json: str,
    rdr_labels: str,
    split_dir: Optional[str],
    output_dir: str,
    output_suffix: str = "",
    K: int = 3,
    min_N_t: int = 1,
    min_chapters: int = 10,
    max_chapters: int = 300,
    seed: int = 42,
    no_split: bool = False,
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading RDR continuous labels (Q_hat)...")
    rdr_map = {}
    with open(rdr_labels) as f:
        for line in f:
            rec = json.loads(line)
            rdr_map[str(rec["book_id"])] = rec
    print(f"  Loaded {len(rdr_map)} books from rdr_labels")

    sample_rec = list(rdr_map.values())[0]
    has_Q_hat = "Q_hat" in sample_rec
    has_level = "level" in sample_rec
    print(f"  Label format: Q_hat={has_Q_hat}, level={has_level}")

    if has_Q_hat:
        label_key = "Q_hat"
        label_type_str = "continuous"
        print(f"  → Using Q_hat (continuous) labels")
    elif has_level:
        label_key = "level"
        label_type_str = "discrete"
        print(f"  → Using level (discrete) labels (fallback)")
    else:
        print(f"  ❌ No Q_hat or level field found in labels!")
        return

    print("Loading novel texts...")
    with open(novel_json) as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            novels = json.load(f)
        else:
            novels = []
            for line in f:
                line = line.strip()
                if line:
                    novels.append(json.loads(line))
    print(f"  Loaded {len(novels)} books from novel_json")

    if no_split:
        all_ids = [str(b["book_id"]) for b in novels if str(b["book_id"]) in rdr_map]
        train_ids = set(all_ids)
        val_ids = set()
        test_ids = set()
        print(f"  No split: all {len(train_ids)} books → train")
    else:
        train_ids, val_ids, test_ids = load_split_ids(split_dir)
        if train_ids is not None:
            print(f"  Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
        else:
            all_ids = [str(b["book_id"]) for b in novels if str(b["book_id"]) in rdr_map]
            random.shuffle(all_ids)
            n = len(all_ids)
            n_val = max(1, n // 10)
            n_test = max(1, n // 10)
            test_ids = set(all_ids[:n_test])
            val_ids = set(all_ids[n_test:n_test + n_val])
            train_ids = set(all_ids[n_test + n_val:])
            print(f"  Auto split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    split_data = {"train": [], "val": [], "test": []}
    stats = defaultdict(int)

    for book in novels:
        book_id = str(book["book_id"])
        if book_id not in rdr_map:
            continue

        if book_id in train_ids:
            split = "train"
        elif book_id in val_ids:
            split = "val"
        elif book_id in test_ids:
            split = "test"
        else:
            continue

        label_rec = rdr_map[book_id]
        chapters = book.get("chapters", [])
        n_chapters = len(chapters)

        if n_chapters < min_chapters or n_chapters > max_chapters:
            stats["filtered_chapters"] += 1
            continue

        label_list = label_rec.get(label_key, [])
        N_t_list = label_rec.get("N_t", [])

        for t in range(1, n_chapters + 1):
            window_texts = []
            window_titles = []
            window_labels = []
            window_weights = []
            window_positions = []
            is_prefix_padding = []

            for k in range(K):
                pos = t - K + 1 + k
                window_positions.append(pos)

                if pos < 1:
                    window_texts.append("<empty>")
                    window_titles.append("")
                    window_labels.append(None)
                    window_weights.append(0.0)
                    is_prefix_padding.append(True)
                else:
                    ch = chapters[pos - 1]
                    window_texts.append(ch.get("text", ""))
                    window_titles.append(ch.get("title", ""))

                    if k == K - 1:
                        if pos - 1 < len(label_list):
                            label = label_list[pos - 1]
                            N_t = N_t_list[pos - 1] if pos - 1 < len(N_t_list) else 0
                            if N_t >= min_N_t and label is not None:
                                weight = math.log(N_t) if N_t > 0 else 0.0
                                window_labels.append(label)
                                window_weights.append(weight)
                            else:
                                window_labels.append(None)
                                window_weights.append(0.0)
                        else:
                            window_labels.append(None)
                            window_weights.append(0.0)
                    else:
                        window_labels.append(None)
                        window_weights.append(0.0)
                    is_prefix_padding.append(False)

            if window_labels[-1] is not None:
                record = {
                    "book_id": book_id,
                    "window_start": max(0, t - K),
                    "chapter_positions": window_positions,
                    "texts": window_texts,
                    "titles": window_titles,
                    "labels": window_labels,
                    "weights": window_weights,
                    "is_prefix_padding": is_prefix_padding,
                    "n_valid": 1,
                    "label_type": label_type_str,
                }
                split_data[split].append(record)
                stats[f"windows_{split}"] += 1

    for split_name in ["train", "val", "test"]:
        if split_data[split_name]:
            out_path = os.path.join(output_dir, f"rdr_seq_{split_name}{output_suffix}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in split_data[split_name]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  Saved {len(split_data[split_name])} windows → {out_path}")

    print("\n[stats]")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RDR sequence dataset (continuous Q_hat labels)")
    parser.add_argument("--novel_json", type=str, default=None)
    parser.add_argument("--rdr_labels", type=str, default=None)
    parser.add_argument("--split_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--min_N_t", type=int, default=1)
    parser.add_argument("--min_chapters", type=int, default=10)
    parser.add_argument("--max_chapters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--no_split", action="store_true")

    args = parser.parse_args()
    build_dataset(
        novel_json=args.novel_json,
        rdr_labels=args.rdr_labels,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        output_suffix=args.output_suffix,
        K=args.K,
        min_N_t=args.min_N_t,
        min_chapters=args.min_chapters,
        max_chapters=args.max_chapters,
        seed=args.seed,
        no_split=args.no_split,
    )