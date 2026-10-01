#!/usr/bin/env python3
"""Frozen-model sentence embeddings, identical to the STS evaluation's.

This mirrors `learning-concepts/eval/eval_mteb.py` (the code behind every STS number
in the paper) so the new probes see the same representation Zhang et al. evaluate:
the base model in 4-bit NF4 with fp16 compute (as trained), the PEFT adapter on top,
right padding with pad = EOS, the tokenizer's default special tokens, the final
hidden layer, a mask-aware mean over real tokens, then L2 normalisation.

`pool_special=False` is the one deviation, used only for single words: a lone word
is one to three tokens, so a BOS state would dominate the mean.  Everything else
calls `encode` with the STS defaults.

Checkpoints are named by label and resolved through the run manifests the notebooks
write (label -> adapter directory); "pretrained" is the untouched base model.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def resolve_checkpoints(labels, manifests, allow_full=False):
    """label -> adapter path (None for the untouched model), from manifest JSONs.

    allow_full also accepts a fully fine-tuned model directory (config.json, no adapter),
    for callers that load such a directory as the base model themselves."""
    table = {}
    for path in manifests:
        for key, value in json.loads(Path(path).read_text()).items():
            table.setdefault(key.replace(" ", "_"), value)
    out = {}
    for label in labels:
        if label == "pretrained":
            out[label] = None
            continue
        if label not in table:
            raise SystemExit(f"{label} is in none of the manifests: {manifests}")
        adapter = Path(table[label])
        full = allow_full and (adapter / "config.json").is_file()
        if not (adapter / "adapter_config.json").is_file() and not full:
            raise SystemExit(f"{label}: no adapter_config.json under {adapter}")
        out[label] = str(adapter)
    return out


class FrozenEncoder:
    def __init__(self, base_model, adapter=None, *, quantize=True, max_length=256,
                 batch_size=64, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        kwargs = {"device_map": {"": self.device}}
        if quantize:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=False)
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        # As eval_mteb.py does: inputs follow the weights (PEFT may place them).
        self.device = next(self.model.parameters()).device
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, texts, *, pool_special=True):
        """(N, H) float32, L2-normalised.  Sorted by length internally for speed;
        mean pooling is mask-aware, so padding never changes a row's value."""
        texts = list(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt",
                                 return_special_tokens_mask=not pool_special)
            special = enc.pop("special_tokens_mask", None)
            enc = enc.to(self.device)
            hidden = self.model(**enc, output_hidden_states=True).hidden_states[-1]
            mask = enc["attention_mask"]
            if special is not None:
                mask = mask * (1 - special.to(self.device))
                empty = mask.sum(dim=1) == 0          # a string that is only special tokens
                mask[empty] = enc["attention_mask"][empty]
            mask = mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = F.normalize(pooled.float(), p=2, dim=1).cpu().numpy()
            for row, i in enumerate(idx):
                out[i] = pooled[row]
        return np.stack(out) if out else np.zeros((0, 1), dtype=np.float32)

    def close(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
