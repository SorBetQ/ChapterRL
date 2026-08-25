import argparse
import json
import os
import re
import time
import fcntl
import multiprocessing
from tqdm import tqdm
import vertexai
from vertexai.generative_models import GenerativeModel


def load_vad_lexicon(vad_path: str) -> dict:
    lexicon = {}
    with open(vad_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i == 0:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                english_word = parts[0].strip().lower()
                arousal = float(parts[2].strip())
                if english_word:
                    lexicon[english_word] = arousal
            except (ValueError, IndexError):
                continue
    return lexicon


def words_to_arousal(words: list, lexicon: dict, max_words: int = 20) -> list:
    scores = []
    for w in words:
        w = w.strip().lower()
        if w in lexicon:
            scores.append(lexicon[w])
        if len(scores) >= max_words:
            break
    return scores if scores else [0.5]


def _gemini_call_with_retry(model, prompt: str, tag: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("busy", "unavailable", "quota", "resource", "429", "503", "繁忙")):
                wait = 50 * (attempt + 1)
                print(f"[{tag}] 繁忙，等待 {wait}s ({attempt+1}/{max_retries}): {e}", flush=True)
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[{tag}] 全部重试失败: {e}", flush=True)
                raise
    return ""


def detect_protagonist(model, chapter_text: str) -> str:
    prompt = (
        "以下是一段中文小说章节。本章的主角是谁？"
        "（即视角最集中、经历与情感最突出的人物）\n"
        "输出应只包含人物姓名，不要输出任何其他内容。\n\n"
        + chapter_text[:4000]
    )
    try:
        return _gemini_call_with_retry(model, prompt, "Gemini-主角")
    except Exception:
        return "主角"


def extract_emotion_words(model, protagonist: str, chapter_text: str, max_words: int = 20) -> list:
    prompt = (
        f"以下是一段中文小说章节，主角是「{protagonist}」。\n\n"
        f"请追踪主角「{protagonist}」的情感变化，每当情感发生明显转变时记录一个新阶段，"
        f"你的任务：为每个阶段输出**一个英文形容词**\n\n"
        f"要求：\n"
        f"1. 只描述主角「{protagonist}」的情感，忽略其他人物。\n"
        f"2. 选词需同时体现情感**类别**和**强度**——\n"
        f"   例如同是「悲」：grieving（极度）/ sorrowful（中度）/ melancholy（轻度）\n"
        f"   例如同是「怒」：furious（极度）/ angry（中度）/ irritated（轻度）\n"
        f"3. 优先选常见英文情感形容词。\n"
        f"4. 只输出一个 JSON 数组，不含任何解释例如：\n"
        f'   ["calm", "curious", "anxious", "angry", "sad"]\n\n'
        + chapter_text[:6000]
    )
    try:
        text = _gemini_call_with_retry(model, prompt, "Gemini-情感词")
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            words = json.loads(match.group())
            if isinstance(words, list) and len(words) > 0:
                return [str(w).strip().lower() for w in words if str(w).strip()]
    except Exception:
        pass
    return []


def chapter_arousal(chapter_text: str, model, lexicon: dict, max_words: int = 20) -> list:
    if not chapter_text.strip():
        return [0.5]
    protagonist = detect_protagonist(model, chapter_text)
    time.sleep(0.3)
    words = extract_emotion_words(model, protagonist, chapter_text, max_words=max_words)
    time.sleep(0.3)
    if not words:
        return [0.5]
    return [round(s, 6) for s in words_to_arousal(words, lexicon, max_words=max_words)]


def load_books_from_novel_json(novel_json_path: str, min_chapters: int = 10, max_chapters: int = 300) -> list:
    with open(novel_json_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            novel_data = json.load(f)
        else:
            novel_data = [json.loads(line) for line in f if line.strip()]
    books = []
    for rec in novel_data:
        bid = str(rec.get("book_id", "")).strip()
        name = rec.get("book_name", "")
        chapters = rec.get("chapters", [])
        if not bid:
            continue
        texts = []
        for ch in chapters:
            if isinstance(ch, dict):
                texts.append(ch.get("text", ""))
            else:
                texts.append(str(ch))
        if min_chapters <= len(texts) <= max_chapters:
            books.append({"book_id": bid, "book_name": name, "chapters": texts})
    return books


load_books_from_novel_jsonl = load_books_from_novel_json


def worker(worker_id: int, book_shard: list, output_path: str, vad_path: str, novel_json_path: str, max_words: int = 20):
    print(f"[worker {worker_id}] 启动，负责 {len(book_shard)} 本书", flush=True)

    vertexai.init(project="mmu-jichu-shangyehua-gemini", location="us-east1")
    model = GenerativeModel("gemini-2.5-pro")

    lexicon = load_vad_lexicon(vad_path)

    for book in tqdm(book_shard, desc=f"worker-{worker_id}", position=worker_id):
        bid = book["book_id"]
        chapters = book["chapters"]
        n_ch = len(chapters)

        arousal_seq = []
        for ch_idx, ch_text in enumerate(chapters, start=1):
            print(f"  [w{worker_id}|{bid}] ch{ch_idx:03d}/{n_ch} 处理中...", flush=True)
            try:
                a_t_seq = chapter_arousal(ch_text, model, lexicon, max_words=max_words)
                arousal_mean = round(sum(a_t_seq) / len(a_t_seq), 4)
                print(f"  [w{worker_id}|{bid}] ch{ch_idx:03d}/{n_ch} ✓  mean={arousal_mean}", flush=True)
            except Exception as e:
                print(f"  [w{worker_id}|{bid}] ch{ch_idx:03d}/{n_ch} ✗  {e}", flush=True)
                a_t_seq = [0.5]
            arousal_seq.append(a_t_seq)

        result = {
            "book_id": bid,
            "book_name": book["book_name"],
            "chapter_count": len(arousal_seq),
            "arousal_sequence": arousal_seq,
        }
        line = json.dumps(result, ensure_ascii=False) + "\n"

        with open(output_path, "a", encoding="utf-8") as fout:
            fcntl.flock(fout, fcntl.LOCK_EX)
            fout.write(line)
            fout.flush()
            fcntl.flock(fout, fcntl.LOCK_UN)

        print(f"  [w{worker_id}|{bid}] ✅ 写入完成", flush=True)

    print(f"[worker {worker_id}] 全部完成", flush=True)


def main():
    parser = argparse.ArgumentParser(description="多进程并行提取章节级 Arousal")
    parser.add_argument("--novel_json", default=None)
    parser.add_argument("--novel_jsonl", default=None)
    parser.add_argument("--vad", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--max_words", type=int, default=20)
    parser.add_argument("--min_chapters", type=int, default=10)
    parser.add_argument("--max_chapters", type=int, default=300)
    args = parser.parse_args()

    if args.novel_jsonl is None and args.novel_json is None:
        parser.error("请指定 --novel_json")

    print("[main] 加载书籍列表...")
    novel_path = args.novel_json or args.novel_jsonl
    all_books = load_books_from_novel_json(
        novel_path, min_chapters=args.min_chapters, max_chapters=args.max_chapters)
    print(f"[main] 共 {len(all_books)} 本书（{args.min_chapters} ≤ 章节数 ≤ {args.max_chapters}），来源：{novel_path}")

    done_ids = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(str(json.loads(line)["book_id"]))
                except Exception:
                    pass
    print(f"[main] 已完成 {len(done_ids)} 本，跳过")

    todo = sorted(
        [b for b in all_books if b["book_id"] not in done_ids],
        key=lambda b: b["book_id"]
    )
    print(f"[main] 待处理 {len(todo)} 本书，将分给 {args.workers} 个进程")

    if not todo:
        print("[main] 无需处理，退出")
        return

    n = args.workers
    shards = [todo[i::n] for i in range(n)]
    for i, shard in enumerate(shards):
        print(f"  [shard {i}] {len(shard)} 本书")

    processes = []
    for worker_id, shard in enumerate(shards):
        if not shard:
            continue
        novel_path = args.novel_json or args.novel_jsonl
        p = multiprocessing.Process(
            target=worker,
            args=(worker_id, shard, args.output, args.vad, novel_path, args.max_words),
        )
        p.start()
        processes.append(p)
        print(f"[main] worker {worker_id} 启动 (PID={p.pid}), {len(shard)} 本书", flush=True)

    for p in processes:
        p.join()

    final_count = 0
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            final_count = sum(1 for _ in f)
    print(f"\n[main] 全部进程完成！arousal.jsonl 共 {final_count} 本书")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()