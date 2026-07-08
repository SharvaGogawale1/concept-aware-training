#!/usr/bin/env python3
"""
Dual evaluation v2 — set-marginal Concept PPL + NTP PPL (July 2026 rewrite).

Fixes over eval_concept_ppl.py (v1):

 1. CANONICAL TOKENIZER — one tokenizer (--tokenizer_path) is used for every
    checkpoint. v1 loaded each checkpoint's own tokenizer, added [PAD] and
    resized embeddings at eval time; with Llama-3.2's tied embeddings that
    injects a randomly-initialised row into the output softmax, so checkpoints
    were not compared under identical distributions. v2 never resizes.

 2. IN-CONTEXT CONTINUATION SCORING — each candidate is scored as an actual
    continuation of its context: log p(candidate tokens | context), using the
    tokenization of (context + " " + candidate). The space-prefixed BPE form is
    therefore used automatically. v1 scored the bare-word token id — the same
    convention the concept trainers optimise, making the v1 metric partially
    circular in favour of concept-trained models.

 3. MULTI-TOKEN CONCEPTS — any candidate can be scored (sum of token
    log-probs), removing v1's single-token filter (72–78% slot coverage).
    The single-token subset is still reported separately for continuity.

 4. UNCERTAINTY — mean set-NLL with a 95% bootstrap CI over rows; the PPL CI
    is exp() of the NLL CI. Per-row NLLs are saved for downstream analysis.

 5. COVERAGE DISAMBIGUATION — reports eval_slot_coverage_pct (fraction of rows
    scoreable). This is a DIFFERENT statistic from the negative-mining
    coverage printed by build_contrastive_dataset.py; do not conflate them.

Set-marginal definition per row (concept set C, context s):

    NLL(s, C) = -log Σ_{c ∈ C} exp( Σ_t log p(c_t | s, c_<t) )

Candidates with identical tokenizations are deduplicated. The marginal sums
continuation probabilities of distinct strings; if one candidate is a strict
prefix of another the events overlap slightly — reported as-is (rare).
Rows whose text contains "<mask>" (concept slot at position 0) cannot be
scored as continuations and are skipped (counted in n_skipped).

Usage:
    python eval_concept_ppl_v2.py \
        --checkpoints /path/ckpt1 /path/ckpt2 \
        --tokenizer_path /path/to/base/Llama-3.2-1B \
        --concept_csv ../../../data/syn/youtube_clean/context_loss_val.csv \
        --vanilla_val ../../../data/hyp/youtube_clean/vanilla_val.txt \
        --results_json dual_eval_v2.json
"""

import argparse
import ast
import json
import math
import os
import random

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

MASK_TOKEN = "<mask>"


# ── NTP eval (vanilla CLM loss on plain text; fixed-length blocks, no padding) ─

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
        output_dir="/tmp/_ntp_eval_v2_tmp",
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


# ── Set-marginal concept eval ────────────────────────────────────────────────

def _continuation_ids(tokenizer, context: str, candidate: str):
    """
    Tokenize candidate as an in-context continuation.

    Returns (full_ids, k) where full_ids is the tokenization of
    context+" "+candidate and k is the length of the shared prefix with the
    context tokenization (continuation tokens are full_ids[k:]). Comparing
    token prefixes (not string lengths) makes this robust to BPE boundary
    re-tokenization.
    """
    ctx = context.rstrip()
    if not ctx:
        return None, None
    full = ctx + " " + candidate.strip()
    ctx_ids = tokenizer(ctx, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=True)["input_ids"]
    k = 0
    for a, b in zip(ctx_ids, full_ids):
        if a != b:
            break
        k += 1
    if k == 0 or k >= len(full_ids):
        return None, None
    return full_ids, k


@torch.no_grad()
def _score_candidate(model, full_ids, k, block_size, device) -> float:
    """Sum of log p(token_j | tokens_<j) for j in [k, len(full_ids))."""
    ids = full_ids
    if len(ids) > block_size:
        drop = len(ids) - block_size
        if k - drop < 1:  # would truncate into the continuation (or lose BOS)
            return None
        ids = ids[drop:]
        k = k - drop
    input_ids = torch.tensor([ids], device=device)
    logits = model(input_ids=input_ids).logits.float()
    log_probs = F.log_softmax(logits[0], dim=-1)  # [L, V]
    total = 0.0
    for j in range(k, len(ids)):
        total += log_probs[j - 1, ids[j]].item()
    return total


def _bootstrap_ci(values, n_boot: int, seed: int, alpha: float = 0.05):
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return (lo, hi)


def _safe_exp(x):
    if x != x:  # nan
        return float("nan")
    return math.exp(x) if x < 300 else float("inf")


def _eval_concept_v2(model, tokenizer, concept_csv: str, block_size: int,
                     n_bootstrap: int, seed: int) -> dict:
    device = next(model.parameters()).device
    df = pd.read_csv(concept_csv)

    if "context_syn" in df.columns:
        concept_col = "context_syn"
    elif "positives" in df.columns:
        concept_col = "positives"
    elif "dict_syn" in df.columns:
        concept_col = "dict_syn"
    else:
        raise ValueError(f"No concept column found in {concept_csv}")

    per_row = []
    n_total = 0
    n_skipped = 0
    n_cand_total = 0
    n_cand_single = 0
    cont_len_total = 0

    model.eval()
    for _, row in df.iterrows():
        n_total += 1
        text = str(row["text"])
        if MASK_TOKEN in text:
            n_skipped += 1  # slot at position 0 — not scoreable as continuation
            continue

        try:
            raw_concepts = ast.literal_eval(str(row[concept_col]))
        except Exception:
            n_skipped += 1
            continue
        if not isinstance(raw_concepts, list) or not raw_concepts:
            n_skipped += 1
            continue
        concepts = [str(c).strip() for c in raw_concepts if str(c).strip()]

        cand_scores = []       # (logp, n_cont_tokens)
        seen_conts = set()
        for c in concepts:
            full_ids, k = _continuation_ids(tokenizer, text, c)
            if full_ids is None:
                continue
            key = tuple(full_ids[k:])
            if key in seen_conts:
                continue
            seen_conts.add(key)
            lp = _score_candidate(model, full_ids, k, block_size, device)
            if lp is None:
                continue
            cand_scores.append((lp, len(key)))

        if not cand_scores:
            n_skipped += 1
            continue

        logps = torch.tensor([lp for lp, _ in cand_scores])
        nll = -torch.logsumexp(logps, dim=0).item()

        single = [lp for lp, n in cand_scores if n == 1]
        nll_single = -torch.logsumexp(torch.tensor(single), dim=0).item() if single else None

        n_cand_total += len(cand_scores)
        n_cand_single += len(single)
        cont_len_total += sum(n for _, n in cand_scores)
        per_row.append({"nll": nll, "nll_single": nll_single, "n_cands": len(cand_scores)})

    n_scored = len(per_row)
    if n_scored == 0:
        return {"concept_error": "no scoreable rows", "concept_n_rows_total": n_total}

    nlls = [r["nll"] for r in per_row]
    mean_nll = sum(nlls) / n_scored
    ci_lo, ci_hi = _bootstrap_ci(nlls, n_bootstrap, seed)

    single_nlls = [r["nll_single"] for r in per_row if r["nll_single"] is not None]
    mean_nll_single = sum(single_nlls) / len(single_nlls) if single_nlls else float("nan")

    return {
        # full set-marginal (all candidates, multi-token included)
        "concept_nll_mean": round(mean_nll, 4),
        "concept_nll_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "concept_ppl": round(_safe_exp(mean_nll), 2),
        "concept_ppl_ci95": [round(_safe_exp(ci_lo), 2), round(_safe_exp(ci_hi), 2)],
        "concept_set_mass_mean": round(sum(math.exp(-v) for v in nlls) / n_scored, 6),
        # single-token subset (continuity with v1 — but in-context tokenization)
        "concept_nll_mean_single_token": round(mean_nll_single, 4),
        "concept_ppl_single_token": round(_safe_exp(mean_nll_single), 2),
        "n_rows_with_single_token_cand": len(single_nlls),
        # coverage statistics — NOT the negative-mining coverage
        "concept_n_rows_total": n_total,
        "concept_n_rows_scored": n_scored,
        "concept_n_rows_skipped": n_skipped,
        "eval_slot_coverage_pct": round(100.0 * n_scored / n_total, 1),
        "avg_candidates_per_row": round(n_cand_total / n_scored, 2),
        "pct_candidates_single_token": round(100.0 * n_cand_single / max(n_cand_total, 1), 1),
        "avg_continuation_tokens": round(cont_len_total / max(n_cand_total, 1), 2),
        "per_row_nll": [round(v, 4) for v in nlls],
    }


# ── Per-checkpoint driver ────────────────────────────────────────────────────

def eval_checkpoint(checkpoint_path: str, tokenizer, concept_csv: str,
                    vanilla_val: str, block_size: int,
                    n_bootstrap: int, seed: int) -> dict:
    print(f"  Loading model: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    n_embed = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > n_embed:
        # NEVER resize at eval time — that injects random rows into a tied softmax.
        raise ValueError(
            f"Canonical tokenizer has {len(tokenizer)} tokens but model embedding "
            f"has {n_embed} rows. Use a tokenizer the model can score."
        )
    if n_embed > len(tokenizer):
        print(f"  NOTE: model has {n_embed - len(tokenizer)} extra embedding rows "
              f"(e.g. a [PAD] added at training time). They stay in the softmax "
              f"denominator — that is the model as trained.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("  [1/2] NTP eval on vanilla text ...")
    ntp_metrics = _eval_ntp(model, tokenizer, vanilla_val, block_size)

    print("  [2/2] Set-marginal concept eval (in-context continuation scoring) ...")
    concept_metrics = _eval_concept_v2(model, tokenizer, concept_csv, block_size,
                                       n_bootstrap, seed)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint": checkpoint_path,
        "model_embedding_rows": n_embed,
        "tokenizer_vocab": len(tokenizer),
        **ntp_metrics,
        **concept_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Dual eval v2: concept set-marginal PPL + NTP PPL")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--tokenizer_path", required=True,
                        help="Canonical tokenizer used for EVERY checkpoint (base model path)")
    parser.add_argument("--concept_csv", required=True,
                        help="context_loss_val.csv (clean split) with 'text' + concept column")
    parser.add_argument("--vanilla_val", required=True, help="vanilla_val.txt (clean split)")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_json", default="dual_eval_v2_results.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"Canonical tokenizer: {args.tokenizer_path} (vocab {len(tokenizer)})")

    all_results = []
    for ckpt in args.checkpoints:
        print(f"\n{'=' * 70}\nEvaluating: {ckpt}\n{'=' * 70}")
        try:
            r = eval_checkpoint(ckpt, tokenizer, args.concept_csv, args.vanilla_val,
                                args.block_size, args.n_bootstrap, args.seed)
            all_results.append(r)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            all_results.append({"checkpoint": ckpt, "error": str(exc)})

    print("\n" + "=" * 128)
    print(f"{'Checkpoint':<34} {'NTP PPL':>8} {'NTP Acc':>8} {'Concept PPL':>12} "
          f"{'[95% CI]':>20} {'SetMass':>8} {'Slots':>6} {'Cov':>6}")
    print("-" * 128)
    for r in all_results:
        name = os.path.basename(r["checkpoint"].rstrip("/")) or r["checkpoint"]
        if "error" in r or "concept_error" in r:
            print(f"{name:<34}  ERROR: {r.get('error', r.get('concept_error'))}")
            continue
        ci = r["concept_ppl_ci95"]
        print(f"{name:<34} {r['ntp_ppl']:>8.2f} {r['ntp_acc']:>8.4f} {r['concept_ppl']:>12.2f} "
              f"{f'[{ci[0]}, {ci[1]}]':>20} {r['concept_set_mass_mean']:>8.4f} "
              f"{r['concept_n_rows_scored']:>6} {r['eval_slot_coverage_pct']:>5.1f}%")
    print("=" * 128)

    with open(args.results_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.results_json}")


if __name__ == "__main__":
    main()
