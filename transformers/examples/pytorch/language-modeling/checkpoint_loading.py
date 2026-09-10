
"""Load untouched models, full checkpoints, or PEFT adapters uniformly."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from transformers import AutoModelForCausalLM


def is_peft_adapter(checkpoint: str) -> bool:
    path = Path(checkpoint)
    return path.is_dir() and (path / "adapter_config.json").is_file()


def adapter_base_model(checkpoint: str) -> Optional[str]:
    path = Path(checkpoint) / "adapter_config.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle).get("base_model_name_or_path")


def load_causal_lm(
    checkpoint: str,
    *,
    dtype: Any,
    device: Any,
    base_model: Optional[str] = None,
):
    """Return an eval-mode model for any supported checkpoint representation.

    `base_model` is optional for adapters because PEFT normally records it in
    adapter_config.json.  Passing it explicitly is useful when the recorded
    path was an ephemeral Colab directory.
    """
    if is_peft_adapter(checkpoint):
        try:
            from peft import PeftModel
        except ImportError as error:
            raise ImportError("PEFT is required to evaluate an adapter checkpoint") from error
        resolved_base = base_model or adapter_base_model(checkpoint)
        if not resolved_base:
            raise ValueError(
                f"adapter {checkpoint} does not identify its base model; pass --base_model"
            )
        base = AutoModelForCausalLM.from_pretrained(resolved_base, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype)
    return model.to(device).eval()
