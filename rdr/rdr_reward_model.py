import os
import sys
import importlib.util

import torch
import torch.nn as nn
from transformers import AutoTokenizer


_RDR_DIR = os.path.dirname(os.path.abspath(__file__))
_CHAPTERRL_ROOT = os.path.dirname(_RDR_DIR)
if _CHAPTERRL_ROOT not in sys.path:
    sys.path.insert(0, _CHAPTERRL_ROOT)


def build_sliding_window_prompt(chapter_texts: list) -> str:
    parts = []
    for k, text in enumerate(chapter_texts, start=1):
        formatted = "" if (text == "<empty>" or not text) else text.strip()
        parts.append(f"[CHAPTER {k}]\n{formatted}\n")
    return "".join(parts)


def build_prefix_filled_window(texts: list, K: int = 3) -> list:
    t = len(texts)
    window = []
    for i in range(K):
        pos = t - K + 1 + i
        if pos < 1:
            window.append("<empty>")
        else:
            window.append(texts[pos - 1])
    return window


def _push_clean_init_env():
    _old_attrs = {}
    for attr in ["_ds_init_empty_weights_ctx",
                 "_empty_weights_ctx",
                 "_accelerate_init_empty"]:
        if hasattr(nn.Module, attr):
            _old_attrs[attr] = getattr(nn.Module, attr)
            delattr(nn.Module, attr)

    _old_env = {}
    for k in ["DEVICE_MAP", "ACCELERATE_USE_DEEPSPEED", "DS_ACCELERATOR"]:
        _old_env[k] = os.environ.get(k)
    os.environ["DEVICE_MAP"] = "cpu"
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    os.environ.setdefault("DS_ACCELERATOR", "cuda")

    return _old_attrs, _old_env


def _pop_clean_init_env(_old_attrs, _old_env):
    for attr, val in _old_attrs.items():
        setattr(nn.Module, attr, val)
    for k, val in _old_env.items():
        if val is not None:
            os.environ[k] = val
        elif k in os.environ:
            del os.environ[k]


class RDROrdinalInference:
    def __init__(
        self,
        base_model_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        max_length: int = 8192,
        n_classes: int = 5,
    ):
        self.base_model_path = base_model_path
        self.max_length = max_length
        self.n_classes = n_classes
        self.device = torch.device(device)

        self.model = self._load_model(base_model_path, checkpoint_path, n_classes)
        self.tokenizer = self._load_tokenizer(base_model_path)

    def _load_model(self, base_model_path: str, checkpoint_path: str, n_classes: int):
        _script = os.path.join(_RDR_DIR, "train_rdr_rm_multinode.py")
        spec = importlib.util.spec_from_file_location("rdr_train", _script)
        rdr_train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rdr_train)
        RDROrdinalModel = rdr_train.RDROrdinalModel

        ckpt_file = checkpoint_path if os.path.isfile(checkpoint_path) \
                    else os.path.join(checkpoint_path, "model.pt")
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"RDR ordinal checkpoint not found: {ckpt_file}")

        raw = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        if (isinstance(raw, dict) and "model" in raw
            and not any(k.startswith("backbone") or k.startswith("ordinal") for k in raw)):
            state = raw["model"]
        else:
            state = raw

        _old_attrs, _old_env = _push_clean_init_env()
        try:
            model = RDROrdinalModel(
                model_path=base_model_path,
                n_classes=n_classes,
                lora_r=64,
                lora_alpha=128,
            )
            missing, unexpected = model.load_state_dict(state, strict=False)
            n_total = len(model.state_dict())

            print(f"[RDROrdinalInference] state_dict load: "
                  f"missing={len(missing)}, unexpected={len(unexpected)}, total={n_total}")
            if missing:
                print(f"  missing (first 5): {missing[:5]}")
            if unexpected:
                print(f"  unexpected (first 5): {unexpected[:5]}")

            if len(missing) > n_total * 0.5:
                raise RuntimeError(
                    f"RDR ordinal architecture mismatch: missing {len(missing)}/{n_total} "
                    f"keys (>50%). Sample: {missing[:3]}"
                )
            if len(unexpected) > n_total * 0.5:
                raise RuntimeError(
                    f"RDR ordinal architecture mismatch: {len(unexpected)} unexpected keys. "
                    f"Sample: {unexpected[:3]}"
                )
        finally:
            _pop_clean_init_env(_old_attrs, _old_env)

        model = model.to(self.device).eval()
        model = model.to(torch.bfloat16)

        dtype_str = str(next(model.parameters()).dtype)
        print(f"[RDROrdinalInference] Loaded RDROrdinalModel from {ckpt_file}, dtype={dtype_str}")
        return model

    def _load_tokenizer(self, model_path: str):
        tok = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    @torch.no_grad()
    def score_chapter(self, chapters_so_far: list) -> float:
        window = build_prefix_filled_window(chapters_so_far, K=3)
        prompt = build_sliding_window_prompt(window)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        score = self.model.predict_scores(input_ids, attention_mask).item()
        return float(score)

    @torch.no_grad()
    def score_chapters(self, chapters: list) -> list:
        scores = []
        for j in range(1, len(chapters) + 1):
            s = self.score_chapter(chapters[:j])
            scores.append(s)
        return scores

    def score_story(self, chapters: list) -> float:
        if not chapters:
            return 0.0
        scores = self.score_chapters(chapters)
        return float(sum(scores) / len(scores))


class RDRRegressionInference:
    def __init__(
        self,
        base_model_path: str,
        checkpoint_path: str,
        device: str = "cpu",
        max_length: int = 8192,
    ):
        self.base_model_path = base_model_path
        self.max_length = max_length
        self.device = torch.device(device)

        self.model = self._load_model(base_model_path, checkpoint_path)
        self.tokenizer = self._load_tokenizer(base_model_path)

    def _load_model(self, base_model_path: str, checkpoint_path: str):
        _script = os.path.join(_RDR_DIR, "train_rdr_rm_continuous.py")
        spec = importlib.util.spec_from_file_location("rdr_continuous", _script)
        rdr_continuous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rdr_continuous)
        RDRRegressionModel = rdr_continuous.RDRRegressionModel

        ckpt_file = checkpoint_path if os.path.isfile(checkpoint_path) \
                    else os.path.join(checkpoint_path, "model.pt")
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"RDR regression checkpoint not found: {ckpt_file}")

        raw = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        if (isinstance(raw, dict) and "model" in raw
            and not any(k.startswith(("backbone", "regression")) for k in raw)):
            state = raw["model"]
        else:
            state = raw

        is_lora = any(("lora_A" in k) or ("lora_B" in k) or ("base_model.model" in k)
                      for k in state.keys())

        _old_attrs, _old_env = _push_clean_init_env()
        try:
            if is_lora:
                model = RDRRegressionModel(
                    model_path=base_model_path,
                    lora_r=64,
                    lora_alpha=128,
                    use_lora=True,
                )
            else:
                model = RDRRegressionModel(
                    model_path=base_model_path,
                    use_lora=False,
                )
            missing, unexpected = model.load_state_dict(state, strict=False)
            n_total = len(model.state_dict())

            print(f"[RDRRegressionInference] state_dict load: "
                  f"missing={len(missing)}, unexpected={len(unexpected)}, total={n_total}")
            if missing:
                print(f"  missing (first 5): {missing[:5]}")
            if unexpected:
                print(f"  unexpected (first 5): {unexpected[:5]}")

            if len(missing) > n_total * 0.5:
                raise RuntimeError(
                    f"RDR regression architecture mismatch: missing {len(missing)}/{n_total} "
                    f"keys (>50%). use_lora={is_lora}, likely wrong. "
                    f"Sample missing: {missing[:3]}"
                )
            if len(unexpected) > n_total * 0.5:
                raise RuntimeError(
                    f"RDR regression architecture mismatch: {len(unexpected)} unexpected keys. "
                    f"use_lora={is_lora}, likely wrong. "
                    f"Sample unexpected: {unexpected[:3]}"
                )
        finally:
            _pop_clean_init_env(_old_attrs, _old_env)

        model = model.to(self.device).eval()
        model = model.to(torch.bfloat16)

        mode_str = "LoRA" if is_lora else "Full-param"
        dtype_str = str(next(model.parameters()).dtype)
        print(f"[RDRRegressionInference] Loaded RDRRegressionModel ({mode_str}) from {ckpt_file}, "
              f"dtype={dtype_str}")
        return model

    def _load_tokenizer(self, model_path: str):
        tok = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    @torch.no_grad()
    def score_chapter(self, chapters_so_far: list) -> float:
        window = build_prefix_filled_window(chapters_so_far, K=3)
        prompt = build_sliding_window_prompt(window)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        score_tensor = self.model(input_ids, attention_mask)
        score = score_tensor.float().squeeze().item()
        return float(score)

    @torch.no_grad()
    def score_chapters(self, chapters: list) -> list:
        scores = []
        for j in range(1, len(chapters) + 1):
            s = self.score_chapter(chapters[:j])
            scores.append(s)
        return scores

    def score_story(self, chapters: list) -> float:
        if not chapters:
            return 0.0
        scores = self.score_chapters(chapters)
        return float(sum(scores) / len(scores))


def load_rdr_inference(
    base_model_path: str,
    checkpoint_path: str,
    device: str = "cpu",
    max_length: int = 8192,
):
    ckpt_file = checkpoint_path if os.path.isfile(checkpoint_path) \
                else os.path.join(checkpoint_path, "model.pt")
    raw = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if (isinstance(raw, dict) and "model" in raw
        and not any(k.startswith(("backbone", "regression", "ordinal")) for k in raw)):
        state = raw["model"]
    else:
        state = raw

    has_regression = any(
        k.startswith("regression.") or k.startswith("regression_head.")
        for k in state
    )
    has_ordinal = any(
        k.startswith("ordinal_head.") or k.startswith("ordinal.")
        for k in state
    )

    if has_regression and not has_ordinal:
        is_regression = True
    elif has_ordinal and not has_regression:
        is_regression = False
    elif has_regression and has_ordinal:
        print(f"[load_rdr_inference] Warning: both regression and ordinal heads detected, "
              f"defaulting to regression")
        is_regression = True
    else:
        sample_keys = list(state.keys())[:10]
        raise RuntimeError(
            f"Cannot detect RDR checkpoint type from {ckpt_file}.\n"
            f"Neither 'regression.*' / 'regression_head.*' nor 'ordinal_head.*' found.\n"
            f"First 10 keys: {sample_keys}"
        )

    if is_regression:
        print(f"[load_rdr_inference] Detected RDRRegressionModel (continuous), "
              f"loading RDRRegressionInference")
        return RDRRegressionInference(
            base_model_path=base_model_path,
            checkpoint_path=checkpoint_path,
            device=device,
            max_length=max_length,
        )
    else:
        print(f"[load_rdr_inference] Detected RDROrdinalModel (discrete), "
              f"loading RDROrdinalInference")
        return RDROrdinalInference(
            base_model_path=base_model_path,
            checkpoint_path=checkpoint_path,
            device=device,
            max_length=max_length,
        )