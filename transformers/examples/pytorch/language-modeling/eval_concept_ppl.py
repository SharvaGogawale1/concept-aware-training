#!/usr/bin/env python3
"""
Task 5: Dual evaluation — Set-Marginal Concept PPL + NTP PPL.

For each checkpoint, computes two complementary metrics:

  1. NTP PPL / Acc  — standard CLM loss on vanilla_val.txt
     (same protocol as eval_ntp_baselines.py; included here for one-stop reporting)

  2. Set-Marginal Concept PPL — exp( mean( -log Σ_{c ∈ C} p(c | context) ) )
     Evaluates on context_loss_val.csv. For each concept slot, takes the model's
     log_softmax distribution at the concept-prediction position, then scores the
     full valid concept set via logsumexp. PPL < NTP PPL is the desired direction:
     the model should assign higher total mass to the concept set than to any single
     gold token.

Usage:
    python eval_concept_ppl.py \
        --checkpoints /path/to/ckpt1 /path/to/ckpt2 \
        --concept_csv ../../../data/syn/youtube/context_loss_val.csv \
        --vanilla_val ../../../data/hyp/youtube/vanilla_val.txt \
        --results_json dual_eval_results.json
"""

import argparse
import ast
import json
import math
import os

import pandas as pd
import torch
import torch.nn.functional as F
import evaluate as hf_evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)


# ── NTP eval (identical logic to eval_ntp_baselines.py) ─────────────────────

def _eval_ntp(model, tokenizer, vanilla_val: str, block_size: int = 128) -> dict:
    ext = vanilla_val.rsplit(".", 1)[-1]
    dataset_ext = "text" if ext == "txt" else ext
    raw = load_dataset(dataset_ext, data_files={"validation": vanilla_val}, trust_remote_code=False)
    col = "text" if "text" in raw["validation"].column_names else raw["validation"].column_names[0]

    def tokenize_fn(examples):
        return {"input_ids": tokenizer(examples[col], add_special_tokens=False)["input_ids"]}

    tokenized_flat = raw.map(tokenize_fn, batched=True, remove_columns=raw["validation"].column_names)

    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total = (len(concatenated) // block_size) * block_size
        chunks = [concatenated[i: i + block_size] for i in range(0, total, block_size)]
        return {"input_ids": chunks, "labels": chunks.copy()}

    tokenized = tokenized_flat.map(group_texts, batched=True)
    accuracy_metric = hf_evaluate.load("accuracy")

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

    training_args = TrainingArguments(
        output_dir="/tmp/_ntp_eval_tmp",
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
        processing_class=tokenizer,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
    )
    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss", float("nan"))
    ppl = math.exp(loss) if loss < 300 else float("inf")
    return {
        "ntp_loss": round(loss, 4),
        "ntp_ppl": round(ppl, 2),
        "ntp_acc": round(metrics.get("eval_accuracy", float("nan")), 4),
    }


# ── Set-marginal concept PPL ─────────────────────────────────────────────────

def _eval_concept_ppl(model, tokenizer, concept_csv: str, block_size: int = 128) -> dict:
    """
    For each row in the concept CSV:
      - Tokenize the context prefix (text column)
      - Run one forward pass
      - Take logits at the last real token position (predicts the concept slot)
      - Compute log Σ_{c ∈ C} p(c | context) via logsumexp over valid concept IDs
      - Accumulate negative log-probability

    Only single-token concepts are scored (multi-token concepts are skipped,
    matching the training-time filter in all NCP trainers).

    Returns:
      concept_ppl   — exp(mean(-log p_set)), the set-marginal PPL
      n_evaluated   — number of concept slots scored
      n_skipped     — slots skipped (no single-token concepts or parse failure)
      coverage_pct  — fraction of rows that had ≥1 scoreable concept
    """
    device = next(model.parameters()).device
    df = pd.read_csv(concept_csv)

    # Support both context_loss CSVs (context_syn) and contrastive CSVs (positives)
    if "context_syn" in df.columns:
        concept_col = "context_syn"
    elif "positives" in df.columns:
        concept_col = "positives"
    else:
        raise ValueError(f"No 'context_syn' or 'positives' column found in {concept_csv}")

    token_cache: dict = {}

    def get_token_id(word: str):
        if word not in token_cache:
            enc = tokenizer(word, return_tensors="pt", add_special_tokens=False)["input_ids"]
            token_cache[word] = enc[0][0].item() if enc.size(1) == 1 else None
        return token_cache[word]

    neg_log_p_list = []
    n_skipped = 0
    n_total = 0

    model.eval()
    with torch.no_grad():
        for _, row in df.iterrows():
            n_total += 1
            text = str(row["text"])

            try:
                raw_concepts = ast.literal_eval(str(row[concept_col]))
            except Exception:
                n_skipped += 1
                continue

            if not isinstance(raw_concepts, list) or not raw_concepts:
                n_skipped += 1
                continue

            concepts = [str(c).strip() for c in raw_concepts if str(c).strip()]
            concept_ids = [tid for c in concepts if (tid := get_token_id(c)) is not None]

            if not concept_ids:
                n_skipped += 1
                continue

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=block_size - 1,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()  # [1, L, V]

            # Predict concept at the last attended position (next token = concept)
            last_pos = attention_mask[0].nonzero(as_tuple=True)[0][-1].item()
            log_probs = F.log_softmax(logits[0, last_pos], dim=-1)  # [V]

            ids_t = torch.tensor(concept_ids, device=device, dtype=torch.long)
            log_p_set = torch.logsumexp(log_probs[ids_t], dim=0)
            neg_log_p_list.append(-log_p_set.item())

    n_evaluated = len(neg_log_p_list)
    if n_evaluated > 0:
        mean_nlp = sum(neg_log_p_list) / n_evaluated
        concept_ppl = math.exp(mean_nlp) if mean_nlp < 300 else float("inf")
        coverage_pct = round(100.0 * n_evaluated / n_total, 1)
    else:
        concept_ppl = float("inf")
        mean_nlp = float("nan")
        coverage_pct = 0.0

    return {
        "concept_ppl": round(concept_ppl, 2),
        "concept_neg_log_p": round(mean_nlp, 4),
        "concept_n_evaluated": n_evaluated,
        "concept_n_skipped": n_skipped,
        "concept_coverage_pct": coverage_pct,
    }


# ── Per-checkpoint driver ────────────────────────────────────────────────────

def eval_checkpoint(checkpoint_path: str, concept_csv: str, vanilla_val: str, block_size: int = 128) -> dict:
    print(f"  Loading tokenizer + model: {checkpoint_path}")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("  [1/2] NTP eval on vanilla text ...")
    ntp_metrics = _eval_ntp(model, tokenizer, vanilla_val, block_size)

    print("  [2/2] Set-marginal concept PPL eval ...")
    concept_metrics = _eval_concept_ppl(model, tokenizer, concept_csv, block_size)

    return {
        "checkpoint": checkpoint_path,
        **ntp_metrics,
        **concept_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Dual eval: concept set-marginal PPL + NTP PPL")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint directories to evaluate")
    parser.add_argument("--concept_csv", required=True,
                        help="context_loss_val.csv with 'text' and 'context_syn' columns")
    parser.add_argument("--vanilla_val", required=True,
                        help="vanilla_val.txt for NTP eval")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--results_json", default="dual_eval_results.json")
    args = parser.parse_args()

    all_results = []
    for ckpt in args.checkpoints:
        print(f"\n{'='*70}")
        print(f"Evaluating: {ckpt}")
        print("=" * 70)
        try:
            r = eval_checkpoint(ckpt, args.concept_csv, args.vanilla_val, args.block_size)
            all_results.append(r)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            all_results.append({"checkpoint": ckpt, "error": str(exc)})

    # Print comparison table
    print("\n" + "=" * 110)
    print(f"{'Checkpoint':<40} {'NTP PPL':>9} {'NTP Acc':>9} {'Concept PPL':>13} {'Coverage':>10} {'#Slots':>8}")
    print("-" * 110)
    for r in all_results:
        name = os.path.basename(r["checkpoint"].rstrip("/")) or r["checkpoint"]
        if "error" in r:
            print(f"{name:<40}  ERROR: {r['error']}")
        else:
            print(
                f"{name:<40}"
                f" {r['ntp_ppl']:>9.2f}"
                f" {r['ntp_acc']:>9.4f}"
                f" {r['concept_ppl']:>13.2f}"
                f" {r['concept_coverage_pct']:>9.1f}%"
                f" {r['concept_n_evaluated']:>8}"
            )
    print("=" * 110)

    with open(args.results_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.results_json}")


if __name__ == "__main__":
    main()
