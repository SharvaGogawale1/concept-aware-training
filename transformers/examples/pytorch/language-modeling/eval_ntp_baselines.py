#!/usr/bin/env python3
"""
Task 1: Evaluate model checkpoints using standard NTP loss on vanilla validation data.

Loads each checkpoint and runs eval-only with a vanilla CLM objective on plain text,
producing a comparison table of eval_loss / perplexity / accuracy.

Usage:
    python eval_ntp_baselines.py \
        --checkpoints /path/to/vanilla_ckpt /path/to/syn_ckpt /path/to/hyp_ckpt \
        --validation_file /path/to/vanilla_val.txt \
        --results_json ntp_results.json
"""

import argparse
import json
import math
import os

import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def eval_checkpoint(checkpoint_path: str, validation_file: str, block_size: int = 128) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    ext = validation_file.split(".")[-1]
    dataset_ext = "text" if ext == "txt" else ext
    raw = load_dataset(dataset_ext, data_files={"validation": validation_file}, trust_remote_code=False)

    col = "text" if "text" in raw["validation"].column_names else raw["validation"].column_names[0]

    def tokenize_fn(example):
        ids = tokenizer(
            str(example[col]),
            return_tensors="pt",
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=block_size,
        )["input_ids"].squeeze().tolist()
        if isinstance(ids, int):
            ids = [ids]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids.copy()}

    tokenized = raw.map(tokenize_fn, batched=False, remove_columns=raw["validation"].column_names)

    accuracy_metric = evaluate.load("accuracy")

    def preprocess_logits(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        mask = labels != -100
        return accuracy_metric.compute(predictions=preds[mask], references=labels[mask])

    tmp_out = os.path.join(checkpoint_path, "_ntp_eval_tmp")
    training_args = TrainingArguments(
        output_dir=tmp_out,
        per_device_eval_batch_size=8,
        do_eval=True,
        do_train=False,
        bf16=True,
        report_to="none",
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
    )

    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", float("nan"))
    perplexity = math.exp(eval_loss) if eval_loss < 300 else float("inf")
    accuracy = metrics.get("eval_accuracy", float("nan"))

    return {
        "checkpoint": checkpoint_path,
        "eval_loss": round(eval_loss, 4),
        "perplexity": round(perplexity, 2),
        "accuracy": round(accuracy, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint directories to evaluate (in order)")
    parser.add_argument("--validation_file", required=True,
                        help="Path to vanilla_val.txt (plain text, no augmentation)")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--results_json", default="ntp_baseline_results.json")
    args = parser.parse_args()

    all_results = []
    for ckpt in args.checkpoints:
        print(f"\nEvaluating: {ckpt}")
        try:
            r = eval_checkpoint(ckpt, args.validation_file, args.block_size)
            all_results.append(r)
            print(f"  loss={r['eval_loss']}  ppl={r['perplexity']}  acc={r['accuracy']}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_results.append({"checkpoint": ckpt, "error": str(exc)})

    print("\n" + "=" * 82)
    print(f"{'Checkpoint':<46} {'Loss':>8} {'Perplexity':>12} {'Accuracy':>10}")
    print("-" * 82)
    for r in all_results:
        name = os.path.basename(r["checkpoint"].rstrip("/")) or r["checkpoint"]
        if "error" in r:
            print(f"{name:<46}  ERROR: {r['error']}")
        else:
            print(f"{name:<46} {r['eval_loss']:>8.4f} {r['perplexity']:>12.2f} {r['accuracy']:>10.4f}")
    print("=" * 82)

    with open(args.results_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.results_json}")


if __name__ == "__main__":
    main()
