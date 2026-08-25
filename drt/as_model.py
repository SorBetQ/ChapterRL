import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from peft import LoraConfig, TaskType, get_peft_model


class ArousalScorer(nn.Module):
    def __init__(
        self,
        model_path: str,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list = None,
    ):
        super().__init__()
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
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
        torch.save(
            self.score_head.state_dict(),
            os.path.join(save_dir, "score_head.pt")
        )

    @classmethod
    def load_pretrained(cls, save_dir: str, base_model_path: str, **kwargs):
        from peft import PeftModel

        model = cls(base_model_path, **kwargs)
        model.backbone = PeftModel.from_pretrained(
            model.backbone.base_model.model,
            save_dir
        )

        state_dict = torch.load(
            os.path.join(save_dir, "score_head.pt"),
            map_location="cpu"
        )
        model.score_head.load_state_dict(state_dict)
        return model


def estimate_arousal_from_text(text: str, vad_lexicon: dict) -> float:
    if not text:
        return 0.5

    scores = [vad_lexicon[w] for w in text if w in vad_lexicon]
    return sum(scores) / len(scores) if scores else 0.5