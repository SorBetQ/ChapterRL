import re
import os
import sys
from typing import List, Optional

import torch


_REWARD_DIR = os.path.dirname(os.path.abspath(__file__))
_CHAPTERRL_ROOT = os.path.dirname(_REWARD_DIR)
if _CHAPTERRL_ROOT not in sys.path:
    sys.path.insert(0, _CHAPTERRL_ROOT)


def group_normalise(values: torch.Tensor, eps: float = 1e-8) -> tuple:
    mu = values.mean()
    sigma = values.std(unbiased=False)
    normalised = (values - mu) / (sigma + eps)
    return normalised, mu, sigma


def compute_unified_reward(
    rdr_scores: torch.Tensor,
    drt_scores: torch.Tensor,
    eps: float = 1e-8,
    drt_enabled: bool = True,
) -> dict:
    rdr_norm, mu_rdr, sigma_rdr = group_normalise(rdr_scores, eps)

    if drt_enabled:
        drt_norm, mu_drt, sigma_drt = group_normalise(drt_scores, eps)
        unified = (rdr_norm + drt_norm) * 0.5
        lambda_eff = 0.5
    else:
        drt_norm = torch.zeros_like(rdr_norm)
        unified = rdr_norm
        mu_drt, sigma_drt = 0.0, 1.0
        lambda_eff = 1.0

    return {
        "unified_reward": unified,
        "rdr_norm": rdr_norm,
        "drt_norm": drt_norm,
        "lambda_eff": lambda_eff,
        "mu_rdr": mu_rdr.item(),
        "sigma_rdr": sigma_rdr.item(),
        "mu_drt": mu_drt if isinstance(mu_drt, float) else mu_drt.item(),
        "sigma_drt": sigma_drt if isinstance(sigma_drt, float) else sigma_drt.item(),
    }


def compute_unified_reward_batch(
    rdr_scores_list: List[torch.Tensor],
    drt_scores_list: List[torch.Tensor],
    eps: float = 1e-8,
) -> List[dict]:
    return [
        compute_unified_reward(rdr, drt, eps)
        for rdr, drt in zip(rdr_scores_list, drt_scores_list)
    ]


class UnifiedRewardCalculator:
    CHAPTER_PATTERN = re.compile(r'第\s*\d+\s*章')

    LONG_MIN_CHAPTERS = 5

    LONG_CRITICAL_SHORT = 1500
    LONG_PARTIAL_SHORT = 3000
    LONG_IDEAL_MAX = 9000
    LONG_PARTIAL_LONG = 12000

    SHORT_MIN_CHARS = 300
    SHORT_MAX_CHARS = 2500

    PENALTY_NO_CHAPTERS = -0.5
    PENALTY_TOO_FEW_CH = -0.3
    PENALTY_FEW_CHAPTERS = -0.10
    PENALTY_CRITICAL_SHORT = -0.30
    PENALTY_PARTIAL_SHORT = -0.10
    PENALTY_PARTIAL_LONG = -0.05
    PENALTY_TOO_LONG = -0.20

    SHORT_TYPE_RDR_CAP = 0.2

    def __init__(
        self,
        rdr_checkpoint: str,
        rdr_base_model: str,
        as_checkpoint: str = None,
        as_base_model: str = None,
        format_penalty: float = -0.5,
        drt_enabled: bool = True,
        device: Optional[str] = None,
        lambda_rdr: float = 0.5,
    ):
        self.format_penalty = format_penalty
        self.drt_enabled = drt_enabled and (as_checkpoint is not None)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.rdr_infer = None
        if rdr_checkpoint and rdr_base_model:
            self._load_rdr(rdr_checkpoint, rdr_base_model)

        self.as_model = None
        self.as_tokenizer = None
        if self.drt_enabled and as_checkpoint and as_base_model:
            self._load_as(as_checkpoint, as_base_model)

    def _load_rdr(self, rdr_checkpoint: str, rdr_base_model: str):
        try:
            from rdr.rdr_reward_model import load_rdr_inference

            self.rdr_infer = load_rdr_inference(
                base_model_path=rdr_base_model,
                checkpoint_path=rdr_checkpoint,
                device=self.device,
                max_length=8192,
            )
        except Exception as e:
            print(f"[UnifiedRewardCalculator] Warning: failed to load RDR model: {e}")
            import traceback
            traceback.print_exc()

    def _load_as(self, as_checkpoint: str, as_base_model: str):
        try:
            from drt.as_model import ArousalScorer
            from transformers import AutoTokenizer

            import torch.nn as nn
            _old = {}
            for attr in ["_ds_init_empty_weights_ctx", "_empty_weights_ctx", "_accelerate_init_empty"]:
                if hasattr(nn.Module, attr):
                    _old[attr] = getattr(nn.Module, attr)
                    delattr(nn.Module, attr)
            _old_env = {}
            for k in ["DEVICE_MAP", "ACCELERATE_USE_DEEPSPEED", "DS_ACCELERATOR"]:
                _old_env[k] = os.environ.get(k)
            os.environ["DEVICE_MAP"] = "cpu"
            os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
            os.environ.setdefault("DS_ACCELERATOR", "cuda")

            try:
                self.as_model = ArousalScorer.load_pretrained(
                    as_checkpoint,
                    base_model_path=as_base_model,
                )
                self.as_model = self.as_model.to(self.device).eval()
            finally:
                for attr, val in _old.items():
                    setattr(nn.Module, attr, val)
                for k, val in _old_env.items():
                    if val is not None:
                        os.environ[k] = val
                    elif k in os.environ:
                        del os.environ[k]

            self.as_tokenizer = AutoTokenizer.from_pretrained(
                as_base_model, trust_remote_code=True, padding_side='left'
            )
            if self.as_tokenizer.pad_token is None:
                self.as_tokenizer.pad_token = self.as_tokenizer.eos_token

            print(f"[UnifiedRewardCalculator] Loaded AS model from {as_checkpoint}")

        except Exception as e:
            print(f"[UnifiedRewardCalculator] Warning: failed to load AS model: {e}")
            import traceback
            traceback.print_exc()
            self.drt_enabled = False

    def _parse_chapters(self, text: str) -> List[str]:
        splits = self.CHAPTER_PATTERN.split(text)
        return [s.strip() for s in splits if s.strip()]

    @torch.no_grad()
    def _compute_rdr_score(self, chapters: List[str]) -> float:
        if self.rdr_infer is None or not chapters:
            return 0.0
        return self.rdr_infer.score_story(chapters)

    @torch.no_grad()
    def _compute_drt_score_with_arousal(self, chapters: List[str]) -> tuple:
        if self.as_model is None or not chapters:
            return 0.0, []

        from drt.compute_drt_reward import compute_drt_mean_reward

        model_device = next(self.as_model.parameters()).device
        arousal_seq = []

        for ch in chapters:
            if not ch.strip():
                arousal_seq.append(0.5)
                continue
            encoded = self.as_tokenizer(
                ch[:6000],
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False,
            )
            encoded = {k: v.to(model_device) for k, v in encoded.items()}
            a_t = self.as_model(**encoded).item()
            arousal_seq.append(float(a_t))

        drt_val = compute_drt_mean_reward(arousal_seq, w=3, tau=0.05)
        return drt_val, arousal_seq

    def _apply_long_length_penalty(self, total_chars: int) -> tuple:
        if total_chars < self.LONG_CRITICAL_SHORT:
            return self.PENALTY_CRITICAL_SHORT, "CRITICAL_SHORT"
        elif total_chars < self.LONG_PARTIAL_SHORT:
            return self.PENALTY_PARTIAL_SHORT, "PARTIAL_SHORT"
        elif total_chars <= self.LONG_IDEAL_MAX:
            return 0.0, "IDEAL"
        elif total_chars <= self.LONG_PARTIAL_LONG:
            return self.PENALTY_PARTIAL_LONG, "PARTIAL_LONG"
        else:
            return self.PENALTY_TOO_LONG, "TOO_LONG"

    def __call__(
        self,
        generated_texts: List[str],
        metadata: Optional[List[dict]] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        import logging
        _log = logging.getLogger(__name__)
        import re as _re

        G = len(generated_texts)
        rdr_scores = torch.zeros(G)
        drt_scores = torch.zeros(G)
        is_penalty_sample = [False] * G

        for i, text in enumerate(generated_texts):
            text_clean = _re.sub(r'<div class="mc-think-block">.*?</div>', '', text, flags=_re.DOTALL).strip()

            meta = (metadata[i] if metadata else {})
            story_type = meta.get("type", "long")
            is_long = meta.get("is_long", story_type != "short")
            total_chars = len(text_clean)

            if story_type == "short":
                if total_chars < self.SHORT_MIN_CHARS:
                    rdr_scores[i] = self.PENALTY_TOO_FEW_CH
                    drt_scores[i] = 0.0
                    is_penalty_sample[i] = True
                    if verbose:
                        _log.info(
                            f"[Reward] sample={i} | type=short | n_chars={total_chars} | "
                            f"SHORT_TOO_SHORT | penalty={self.PENALTY_TOO_FEW_CH:.3f}"
                        )
                    continue

                if total_chars > self.SHORT_MAX_CHARS:
                    rdr_scores[i] = self.PENALTY_TOO_LONG
                    drt_scores[i] = 0.0
                    is_penalty_sample[i] = True
                    if verbose:
                        _log.info(
                            f"[Reward] sample={i} | type=short | n_chars={total_chars} | "
                            f"SHORT_TOO_LONG (pseudo-long) | penalty={self.PENALTY_TOO_LONG:.3f}"
                        )
                    continue

                rdr_val_raw = self._compute_rdr_score([text_clean])
                rdr_val = min(rdr_val_raw, self.SHORT_TYPE_RDR_CAP)
                rdr_scores[i] = rdr_val
                drt_scores[i] = 0.0
                if verbose:
                    cap_marker = " (CAPPED)" if rdr_val_raw > self.SHORT_TYPE_RDR_CAP else ""
                    _log.info(
                        f"[Reward] sample={i} | type=short | n_chars={total_chars} | "
                        f"rdr_raw={rdr_val_raw:.4f} → rdr={rdr_val:.4f}{cap_marker} | drt=0.0000"
                    )
                continue

            chapters = self._parse_chapters(text_clean)
            n_chapters = len(chapters)

            if n_chapters == 0:
                rdr_scores[i] = self.PENALTY_NO_CHAPTERS
                drt_scores[i] = 0.0
                is_penalty_sample[i] = True
                if verbose:
                    _log.info(
                        f"[Reward] sample={i} | chapters=0 | total_chars={total_chars} | "
                        f"NO_CHAPTERS | penalty={self.PENALTY_NO_CHAPTERS:.3f}"
                    )
                continue

            if n_chapters < 3:
                rdr_scores[i] = self.PENALTY_TOO_FEW_CH
                drt_scores[i] = 0.0
                is_penalty_sample[i] = True
                if verbose:
                    _log.info(
                        f"[Reward] sample={i} | chapters={n_chapters} | total_chars={total_chars} | "
                        f"TOO_FEW_CHAPTERS | penalty={self.PENALTY_TOO_FEW_CH:.3f}"
                    )
                continue

            if n_chapters < self.LONG_MIN_CHAPTERS:
                rdr_val = self._compute_rdr_score(chapters)
                rdr_scores[i] = rdr_val + self.PENALTY_FEW_CHAPTERS

                drt_val = 0.0
                arousal_seq = []
                if self.drt_enabled and self.as_model is not None:
                    drt_val, arousal_seq = self._compute_drt_score_with_arousal(chapters)
                drt_scores[i] = drt_val

                if verbose:
                    arousal_str = (
                        "[" + ", ".join(f"{a:.3f}" for a in arousal_seq) + "]"
                        if arousal_seq else "N/A"
                    )
                    _log.info(
                        f"[Reward] sample={i} | chapters={n_chapters} | total_chars={total_chars} "
                        f"| FEW_CHAPTERS_PENALTY | rdr_raw={rdr_val:.4f} → rdr={rdr_scores[i].item():.4f} "
                        f"| drt={drt_val:.4f} | arousal={arousal_str}"
                    )
                continue

            length_penalty, length_tag = self._apply_long_length_penalty(total_chars)

            rdr_val = self._compute_rdr_score(chapters)
            rdr_scores[i] = rdr_val + length_penalty

            drt_val = 0.0
            arousal_seq = []
            if self.drt_enabled and self.as_model is not None:
                drt_val, arousal_seq = self._compute_drt_score_with_arousal(chapters)
            drt_scores[i] = drt_val

            if verbose:
                arousal_str = (
                    "[" + ", ".join(f"{a:.3f}" for a in arousal_seq) + "]"
                    if arousal_seq else "N/A"
                )
                avg = total_chars / max(n_chapters, 1)
                if length_tag == "IDEAL":
                    _log.info(
                        f"[Reward] sample={i} | chapters={n_chapters} | total_chars={total_chars} "
                        f"(avg={avg:.0f}/章) | OK | rdr={rdr_val:.4f} | drt={drt_val:.4f} "
                        f"| arousal={arousal_str}"
                    )
                else:
                    _log.info(
                        f"[Reward] sample={i} | chapters={n_chapters} | total_chars={total_chars} "
                        f"(avg={avg:.0f}/章) | LONG_{length_tag} "
                        f"| rdr_raw={rdr_val:.4f} → rdr={rdr_scores[i].item():.4f} "
                        f"| drt={drt_val:.4f} | arousal={arousal_str}"
                    )

        if self.drt_enabled:
            rewards = (rdr_scores + drt_scores) * 0.5
        else:
            rewards = rdr_scores

        if verbose:
            penalty_mask = torch.tensor(is_penalty_sample, dtype=torch.bool)
            n_format = int(penalty_mask.sum().item())
            valid_mask = ~penalty_mask

            if valid_mask.sum().item() > 0:
                valid_rdr = rdr_scores[valid_mask]
                rdr_mean = valid_rdr.mean().item()
                rdr_std = valid_rdr.std().item() if valid_rdr.numel() > 1 else 0.0

                if self.drt_enabled:
                    valid_drt = drt_scores[valid_mask]
                    drt_mean = valid_drt.mean().item()
                    drt_std = valid_drt.std().item() if valid_drt.numel() > 1 else 0.0
                    drt_str = f"| drt_mean={drt_mean:.4f} | drt_std={drt_std:.4f} "
                else:
                    drt_str = ""
            else:
                rdr_mean, rdr_std = float('nan'), float('nan')
                drt_str = ""

            unified_mean = rewards.mean().item()
            unified_std = rewards.std().item() if rewards.numel() > 1 else 0.0

            _log.info(
                f"[Reward] GROUP SUMMARY | n={G} | format_penalty={n_format} "
                f"| rdr_mean={rdr_mean:.4f} | rdr_std={rdr_std:.4f} "
                + drt_str
                + f"| unified_mean={unified_mean:.4f} | unified_std={unified_std:.4f}"
            )

        return rewards