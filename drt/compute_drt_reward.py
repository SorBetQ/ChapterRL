import argparse
from typing import List

import numpy as np


def r_drt(arousal_seq: List[float], chapter_idx: int, w: int = 3, tau: float = 0.05) -> float:
    t = chapter_idx
    if t < 2 or t > len(arousal_seq):
        return 0.0

    a_t = arousal_seq[t - 1]

    start = max(1, t - w)
    window = arousal_seq[start - 1 : t - 1]
    window_mean = float(np.mean(window)) if window else 0.5

    penalty = max(0.0, window_mean - a_t - tau)
    return penalty


def compute_drt_reward(
    arousal_sequence: List[float],
    chapter_idx: int,
    w: int = 3,
    tau: float = 0.05,
) -> dict:
    t = chapter_idx
    rd = r_drt(arousal_sequence, t, w, tau)

    a_t = arousal_sequence[t - 1] if t <= len(arousal_sequence) else None

    start = max(1, t - w)
    window = arousal_sequence[start - 1 : t - 1]
    window_mean = float(np.mean(window)) if window else None

    return {
        "R_DRT": rd,
        "a_t": a_t,
        "window_mean": window_mean,
        "chapter_idx": t,
        "w": w,
    }


def compute_drt_sequence(
    arousal_sequence: List[float],
    w: int = 3,
    tau: float = 0.05,
) -> List[dict]:
    results = []
    for t in range(1, len(arousal_sequence) + 1):
        results.append(compute_drt_reward(arousal_sequence, t, w, tau))
    return results


def compute_drt_mean_reward(
    arousal_sequence: List[float],
    w: int = 3,
    tau: float = 0.05,
) -> float:
    if len(arousal_sequence) < 2:
        return 0.0

    rewards = compute_drt_sequence(arousal_sequence, w, tau)
    valid_rewards = [r["R_DRT"] for r in rewards[1:]]
    return -float(np.mean(valid_rewards)) if valid_rewards else 0.0


def flatten_arousal_sequence(arousal_seq_of_seqs: List[List[float]]) -> List[float]:
    return [float(np.mean(seq)) if seq else 0.5 for seq in arousal_seq_of_seqs]


def compute_drt_reward_nested(
    arousal_seq_of_seqs: List[List[float]],
    chapter_idx: int,
    w: int = 3,
) -> dict:
    flat_seq = flatten_arousal_sequence(arousal_seq_of_seqs)
    return compute_drt_reward(flat_seq, chapter_idx, w)


if __name__ == "__main__":
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("--arousal", type=str, required=True)
    parser.add_argument("--chapter_idx", type=int, default=None)
    parser.add_argument("--w", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()

    seq = ast.literal_eval(args.arousal.strip())
    if args.chapter_idx:
        result = compute_drt_reward(seq, args.chapter_idx, args.w, args.tau)
        print(result)
    else:
        results = compute_drt_sequence(seq, args.w, args.tau)
        mean_reward = compute_drt_mean_reward(seq, args.w, args.tau)
        for r in results:
            print(r)
        print(f"\nMean R_DRT = {mean_reward:.4f}")