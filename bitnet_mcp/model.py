from __future__ import annotations

from typing import Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "BitNetModel requires the 'transformers' and 'torch' packages."
    ) from exc


class BitNetModel:
    """Wrapper around HuggingFace transformers for BitNet models."""

    def __init__(self, model_name_or_path: str,
                 device: Optional[str] = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
