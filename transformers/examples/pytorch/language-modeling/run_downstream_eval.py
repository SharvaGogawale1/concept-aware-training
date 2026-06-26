#!/usr/bin/env python3
"""
Task 7: Fine-tune model checkpoints on downstream NLU tasks and compare.

Wraps a causal LM with a linear classification head pooled over the last
non-padded token's hidden state, which is the standard approach for
decoder-only models (Llama, Pythia, GPT family).

Supported tasks:
  snli  — Stanford NLI, stanfordnlp/snli (3-class: entailment/neutral/contradiction)
  spam  — SpamAssassin email spam detection, talby/spamassassin "text" config (binary)
          Dataset has only a train split; we create an 80/20 train/val split automatically.

Two evaluation modes:
  --freeze_base   Linear probe: only the classification head is trained.
                  Cleanest measure of pre-training representation quality.
  (default)       Full fine-tune: all parameters trained.
                  Measures best achievable downstream performance.

Usage:
    python run_downstream_eval.py \
        --checkpoints /path/to/ckpt1 /path/to/ckpt2 \
        --task snli \
        --output_dir ./downstream_outputs \
        --max_train_samples 20000 \
        --results_json snli_results.json

    # Linear probe only:
    python run_downstream_eval.py \
        --checkpoints /path/to/ckpt1 \
        --task snli \
        --freeze_base \
        --output_dir ./downstream_probe \
        --results_json snli_probe_results.json
"""

import argparse
import json
import os

# datasets' torch formatter unconditionally imports torchvision.io.VideoReader,
# which is missing in some Colab torchvision builds. Stub it out before any
# dataset formatting call triggers the import.
try:
    from torchvision.io import VideoReader  # noqa: F401
except ImportError:
    import torchvision.io as _tv_io
    class _StubVideoReader:
        pass
    _tv_io.VideoReader = _StubVideoReader

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import evaluate as hf_evaluate

# ── Task configuration ────────────────────────────────────────────────────────

TASK_CONFIG = {
    "snli": {
        "hf_path": "stanfordnlp/snli",
        "hf_config": None,                  # default config
        "num_labels": 3,
        "label_col": "label",
        "text_fn": lambda ex: f"{ex['premise']} [SEP] {ex['hypothesis']}",
        "filter_fn": lambda ex: ex["label"] in (0, 1, 2),  # drop -1 unlabeled
        "val_split": "validation",          # has train / validation / test
        "auto_split": False,
        "metric_for_best": "accuracy",
        "f1_average": "macro",
    },
    "spam": {
        # SpamAssassin public corpus — talby/spamassassin, "text" config
        # Fields: label (ClassLabel: ham=0, spam=1), group, text
        # Only has a "train" split (10.7k rows); we auto-split 80/20.
        "hf_path": "talby/spamassassin",
        "hf_config": "text",
        "num_labels": 2,
        "label_col": "label",
        "text_fn": lambda ex: ex["text"],
        "filter_fn": None,
        "val_split": "validation",          # created by auto_split
        "auto_split": True,                 # split train → 80% train / 20% val
        "auto_split_test_size": 0.2,
        "metric_for_best": "f1",
        "f1_average": "binary",
    },
}


# ── Classification wrapper ───────────────────────────────────────────────────

class CausalLMForClassification(nn.Module):
    """
    Decoder-only LM + linear classification head.

    Pools the last non-padded token's hidden state. For causal LMs that attend
    left-to-right, this position has seen the full input context.
    """

    def __init__(self, base_model, num_labels: int, freeze_base: bool):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad_(False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [B, L, H]

        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_idx, seq_lens]   # [B, H]
        else:
            pooled = hidden[:, -1]

        logits = self.classifier(pooled.float())   # [B, num_labels]

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


# ── Data loading ─────────────────────────────────────────────────────────────

def load_task_data(task: str, tokenizer, max_train_samples: int = None, max_length: int = 128):
    cfg = TASK_CONFIG[task]

    # Load — some datasets require a named config (e.g. "text" for spamassassin)
    if cfg["hf_config"]:
        raw = load_dataset(cfg["hf_path"], cfg["hf_config"])
    else:
        raw = load_dataset(cfg["hf_path"])

    # SpamAssassin (and any dataset with only a "train" split): auto-split into train/val
    if cfg.get("auto_split") and "validation" not in raw and "test" not in raw:
        test_size = cfg.get("auto_split_test_size", 0.2)
        split = raw["train"].train_test_split(test_size=test_size, seed=42)
        raw = {"train": split["train"], "validation": split["test"]}

    if cfg["filter_fn"]:
        raw = {split: ds.filter(cfg["filter_fn"]) for split, ds in raw.items()}

    if max_train_samples and len(raw["train"]) > max_train_samples:
        raw["train"] = raw["train"].shuffle(seed=42).select(range(max_train_samples))

    text_fn = cfg["text_fn"]
    label_col = cfg["label_col"]
    original_cols = raw["train"].column_names

    def tokenize(examples):
        texts = [
            text_fn({c: examples[c][i] for c in original_cols})
            for i in range(len(examples[label_col]))
        ]
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
        enc["labels"] = examples[label_col]
        return enc

    tokenized = {
        split: ds.map(tokenize, batched=True, remove_columns=original_cols)
        for split, ds in raw.items()
    }
    for split in tokenized:
        tokenized[split].set_format("torch")

    return tokenized, cfg["num_labels"], cfg["val_split"]


# ── Single checkpoint run ────────────────────────────────────────────────────

def run_one_checkpoint(
    checkpoint_path: str,
    task: str,
    output_dir: str,
    freeze_base: bool,
    max_train_samples: int,
    num_epochs: int,
    max_length: int,
    lr: float,
) -> dict:
    cfg = TASK_CONFIG[task]

    print(f"  Loading tokenizer + model: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    if len(tokenizer) > base.get_input_embeddings().weight.shape[0]:
        base.resize_token_embeddings(len(tokenizer))

    print(f"  Loading {task} dataset ...")
    tokenized, num_labels, val_split = load_task_data(task, tokenizer, max_train_samples, max_length)

    model = CausalLMForClassification(base, num_labels, freeze_base=freeze_base)

    accuracy_metric = hf_evaluate.load("accuracy")
    f1_metric = hf_evaluate.load("f1")
    f1_avg = cfg["f1_average"]

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = logits.argmax(axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(predictions=preds, references=labels, average=f1_avg)
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    mode = "probe" if freeze_base else "finetune"
    ckpt_name = os.path.basename(checkpoint_path.rstrip("/"))
    run_dir = os.path.join(output_dir, f"{ckpt_name}_{task}_{mode}")

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=16,
        learning_rate=lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best"],
        greater_is_better=True,
        report_to="none",
        logging_steps=100,
        save_total_limit=1,
    )

    val_dataset = (
        tokenized.get(val_split)
        or tokenized.get("validation")
        or tokenized.get("test")
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"  Training ({'linear probe' if freeze_base else 'full fine-tune'}) ...")
    trainer.train()
    metrics = trainer.evaluate()

    return {
        "checkpoint": checkpoint_path,
        "task": task,
        "mode": mode,
        "accuracy": round(metrics.get("eval_accuracy", float("nan")), 4),
        "f1": round(metrics.get("eval_f1", float("nan")), 4),
        "eval_loss": round(metrics.get("eval_loss", float("nan")), 4),
        "run_dir": run_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--task", choices=list(TASK_CONFIG.keys()), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--freeze_base", action="store_true",
                        help="Linear probe: freeze LM, train only classifier head")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--results_json", default="downstream_results.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for ckpt in args.checkpoints:
        print(f"\n{'='*70}")
        print(f"Checkpoint: {ckpt}")
        print("=" * 70)
        try:
            r = run_one_checkpoint(
                ckpt,
                task=args.task,
                output_dir=args.output_dir,
                freeze_base=args.freeze_base,
                max_train_samples=args.max_train_samples,
                num_epochs=args.num_epochs,
                max_length=args.max_length,
                lr=args.lr,
            )
            all_results.append(r)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            all_results.append({"checkpoint": ckpt, "task": args.task, "error": str(exc)})

    mode_str = "linear probe" if args.freeze_base else "full fine-tune"
    print(f"\n{'='*72}")
    print(f"Task: {args.task}  |  Mode: {mode_str}")
    print(f"{'Checkpoint':<38} {'Accuracy':>10} {'F1':>10}")
    print("-" * 72)
    for r in all_results:
        name = os.path.basename(r["checkpoint"].rstrip("/"))
        if "error" in r:
            print(f"{name:<38}  ERROR")
        else:
            print(f"{name:<38} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")
    print("=" * 72)

    with open(args.results_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.results_json}")


if __name__ == "__main__":
    main()
