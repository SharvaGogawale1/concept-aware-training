#!/usr/bin/env python3
"""Canonical exact-sequence evaluation for the controlled Tasks-13 audit."""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import math
import os
import random
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sequence_ncp_trainer import (
    MASK_TOKEN,
    encode_candidate_continuation,
    has_strict_prefix_collision,
    parse_candidate_list,
    sequence_log_probs_from_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument(
        "--concept_csv",
        nargs="+",
        required=True,
        help="One or more TAG=PATH entries, for example syn=data/syn/...csv",
    )
    parser.add_argument(
        "--vanilla_val", nargs="+", required=True,
        help=(
            "one PATH (backward-compatible) or multiple TAG=PATH entries, e.g. "
            "syn=data/syn/.../vanilla_val.txt hyp=data/hyp/.../vanilla_val.txt"
        ),
    )
    parser.add_argument("--results_json", required=True)
    parser.add_argument(
        "--gold_column",
        default="gold_surface",
        help=(
            "CSV column holding the observed continuation.  When present, the set marginal is "
            "decomposed into a gold component and an ALTERNATIVES-ONLY component.  The "
            "alternatives-only figure is the decisive metric for gold-inclusive training: a model "
            "scored on a gold-inclusive set can otherwise satisfy the marginal through p(T) alone, "
            "since -log sum_{c in C u {T}} p(c) <= -log p(T) holds before any training."
        ),
    )
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def _dtype(name: str, device: torch.device):
    if device.type == "cpu":
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _bootstrap_ci(values: Sequence[float], n_bootstrap: int, seed: int) -> List[float]:
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_bootstrap))
    return [means[int(0.025 * n_bootstrap)], means[min(int(0.975 * n_bootstrap), n_bootstrap - 1)]]


def _safe_exp(value: float) -> float:
    return math.exp(value) if value < 300 else float("inf")


def _pad(rows: Sequence[Sequence[int]], value: int, device: torch.device) -> torch.Tensor:
    width = max(len(row) for row in rows)
    return torch.tensor([list(row) + [value] * (width - len(row)) for row in rows], device=device)


@torch.no_grad()
def _score_encoded_candidates(
    model: Any,
    encoded: Sequence[Dict[str, Any]],
    pad_token_id: int,
    batch_size: int,
    device: torch.device,
) -> List[float]:
    scores: List[float] = []
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        ids = _pad([item["input_ids"] for item in batch], pad_token_id, device)
        masks = _pad([item["attention_mask"] for item in batch], 0, device)
        labels = _pad([item["labels"] for item in batch], -100, device)
        logits = model(input_ids=ids, attention_mask=masks, use_cache=False).logits
        scores.extend(sequence_log_probs_from_logits(logits, labels).float().cpu().tolist())
    return scores


@torch.no_grad()
def evaluate_ntp(
    model: Any,
    tokenizer: Any,
    validation_file: str,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    with open(validation_file, encoding="utf-8") as handle:
        texts = [line.rstrip("\n") for line in handle if line.strip()]
    flat_ids: List[int] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        # Keep sentence boundaries explicit. The previous evaluator concatenated raw token lists
        # with no EOS/BOS separator, thereby scoring artificial last-word -> next-sentence links.
        if tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id):
            ids = list(ids) + [tokenizer.eos_token_id]
        flat_ids.extend(ids)
    chunks = [flat_ids[index : index + block_size]
              for index in range(0, len(flat_ids), block_size)
              if len(flat_ids[index : index + block_size]) >= 2]
    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    for start in range(0, len(chunks), batch_size):
        rows = chunks[start : start + batch_size]
        input_ids = _pad(rows, tokenizer.pad_token_id, device)
        attention_mask = _pad([[1] * len(row) for row in rows], 0, device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        labels = input_ids[:, 1:]
        valid = attention_mask[:, 1:].bool()
        predictions = logits[:, :-1].argmax(dim=-1)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels.masked_fill(~valid, -100).reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_nll += float(loss.cpu())
        total_tokens += int(valid.sum().cpu())
        total_correct += int((predictions.eq(labels) & valid).sum().cpu())
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "ntp_nll_mean": mean_nll,
        "ntp_ppl": _safe_exp(mean_nll),
        "ntp_accuracy": total_correct / max(total_tokens, 1),
        "ntp_tokens": total_tokens,
    }


def _candidate_column(fieldnames: Iterable[str]) -> str:
    names = set(fieldnames)
    for name in ("context_syn", "positives", "dict_syn"):
        if name in names:
            return name
    raise ValueError("concept CSV has no candidate column")


@torch.no_grad()
def evaluate_concepts(
    model: Any,
    tokenizer: Any,
    concept_csv: str,
    block_size: int,
    batch_size: int,
    n_bootstrap: int,
    seed: int,
    device: torch.device,
    gold_column: Optional[str] = None,
) -> Dict[str, Any]:
    with open(concept_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        concept_column = _candidate_column(fieldnames)
        rows = list(reader)
    has_gold = bool(gold_column) and gold_column in fieldnames
    has_row_id = "row_id" in fieldnames

    per_row: List[Dict[str, Any]] = []
    skipped = {"mask_row": 0, "no_candidates": 0, "unscoreable": 0, "strict_token_prefix_collision": 0}
    all_single_logps: List[float] = []
    all_multi_logps: List[float] = []
    model.eval()

    for index, row in enumerate(rows):
        context = str(row["text"])
        if MASK_TOKEN in context:
            skipped["mask_row"] += 1
            continue
        candidates = parse_candidate_list(row[concept_column])
        if not candidates:
            skipped["no_candidates"] += 1
            continue

        encoded = []
        strings = []
        seen = set()
        for candidate in candidates:
            item = encode_candidate_continuation(tokenizer, context, candidate, block_size)
            if item is None:
                continue
            key = tuple(item["continuation_ids"])
            if key in seen:
                continue
            seen.add(key)
            encoded.append(item)
            strings.append(candidate)
        if not encoded:
            skipped["unscoreable"] += 1
            continue
        token_sequences = [item["continuation_ids"] for item in encoded]
        if has_strict_prefix_collision(token_sequences):
            skipped["strict_token_prefix_collision"] += 1
            continue

        logps = _score_encoded_candidates(model, encoded, tokenizer.pad_token_id, batch_size, device)
        logps_tensor = torch.tensor(logps, dtype=torch.float64)
        nll = float(-torch.logsumexp(logps_tensor, dim=0))
        normalized = torch.softmax(logps_tensor, dim=0)
        entropy = float(-(normalized * torch.log(normalized.clamp_min(1e-300))).sum())
        normalized_entropy = entropy / math.log(len(logps)) if len(logps) > 1 else 0.0
        lengths = [len(tokens) for tokens in token_sequences]
        single = [score for score, length in zip(logps, lengths) if length == 1]
        multi = [score for score, length in zip(logps, lengths) if length > 1]
        all_single_logps.extend(single)
        all_multi_logps.extend(multi)

        # Gold / alternatives decomposition.  `strings` is aligned with `logps`; the rebuild
        # writes the gold first, but match on the string so the order is not load-bearing.
        gold_surface = str(row.get(gold_column, "")).strip() if has_gold else ""
        gold_index = None
        if gold_surface:
            folded = gold_surface.casefold()
            gold_index = next(
                (i for i, candidate in enumerate(strings) if candidate.strip().casefold() == folded),
                None,
            )
        if gold_index is None:
            nll_gold = None
            alt_logps = logps
        else:
            nll_gold = -logps[gold_index]
            alt_logps = [score for i, score in enumerate(logps) if i != gold_index]
        nll_alternatives = (
            float(-torch.logsumexp(torch.tensor(alt_logps, dtype=torch.float64), dim=0))
            if alt_logps
            else None
        )

        per_row.append(
            {
                "row_id": str(row["row_id"]) if has_row_id else f"eval:{index}",
                "nll": nll,
                "nll_set_mean": nll + math.log(len(logps)),
                "set_mass": _safe_exp(-nll),
                "gold_surface": gold_surface or None,
                "gold_in_set": gold_index is not None,
                "nll_gold": nll_gold,
                "gold_mass": _safe_exp(-nll_gold) if nll_gold is not None else None,
                "nll_alternatives": nll_alternatives,
                "nll_alternatives_mean": (
                    nll_alternatives + math.log(len(alt_logps))
                    if nll_alternatives is not None and alt_logps else None
                ),
                "alternatives_mass": (
                    _safe_exp(-nll_alternatives) if nll_alternatives is not None else None
                ),
                "n_alternatives": len(alt_logps),
                "n_candidates": len(logps),
                "candidate_strings": strings,
                "candidate_token_ids": token_sequences,
                "candidate_lengths": lengths,
                "candidate_logps": logps,
                "candidate_entropy": entropy,
                "candidate_entropy_normalized": normalized_entropy,
                "min_candidate_logp": min(logps),
                "nll_single": float(-torch.logsumexp(torch.tensor(single), dim=0)) if single else None,
                "nll_multi": float(-torch.logsumexp(torch.tensor(multi), dim=0)) if multi else None,
            }
        )

    nlls = [item["nll"] for item in per_row]
    if not nlls:
        return {"error": "no scoreable concept rows", "rows_total": len(rows), "skipped": skipped}
    ci = _bootstrap_ci(nlls, n_bootstrap, seed)
    mean_nll = statistics.fmean(nlls)

    # ALTERNATIVES-ONLY aggregates -- the decisive figures when the eval set is gold-inclusive.
    alt_rows = [item for item in per_row if item["nll_alternatives"] is not None]
    alt_nlls = [item["nll_alternatives"] for item in alt_rows]
    alt_mean_nlls = [item["nll_alternatives_mean"] for item in alt_rows]
    gold_rows = [item for item in per_row if item["nll_gold"] is not None]
    gold_nlls = [item["nll_gold"] for item in gold_rows]

    return {
        "rows_total": len(rows),
        "rows_scored": len(per_row),
        "eval_slot_coverage_pct": 100.0 * len(per_row) / max(len(rows), 1),
        "skipped": skipped,
        "gold_column_present": has_gold,
        "rows_with_gold_scored": len(gold_rows),
        "concept_nll_mean": mean_nll,
        "concept_mean_nll_mean": statistics.fmean(item["nll_set_mean"] for item in per_row),
        "concept_nll_median": statistics.median(nlls),
        "concept_nll_ci95": ci,
        "concept_ppl": _safe_exp(mean_nll),
        "alt_nll_mean": statistics.fmean(alt_nlls) if alt_nlls else None,
        "alt_mean_nll_mean": statistics.fmean(alt_mean_nlls) if alt_mean_nlls else None,
        "alt_nll_median": statistics.median(alt_nlls) if alt_nlls else None,
        "alt_nll_ci95": _bootstrap_ci(alt_nlls, n_bootstrap, seed) if alt_nlls else None,
        "alt_ppl": _safe_exp(statistics.fmean(alt_nlls)) if alt_nlls else None,
        "alt_mass_mean": (
            statistics.fmean(item["alternatives_mass"] for item in alt_rows) if alt_rows else None
        ),
        "gold_nll_mean": statistics.fmean(gold_nlls) if gold_nlls else None,
        "gold_nll_median": statistics.median(gold_nlls) if gold_nlls else None,
        "gold_mass_mean": (
            statistics.fmean(item["gold_mass"] for item in gold_rows) if gold_rows else None
        ),
        "set_mass_mean": statistics.fmean(item["set_mass"] for item in per_row),
        "candidate_entropy_normalized_mean": statistics.fmean(
            item["candidate_entropy_normalized"] for item in per_row
        ),
        "min_candidate_logp_mean": statistics.fmean(item["min_candidate_logp"] for item in per_row),
        "single_candidate_logp_mean": statistics.fmean(all_single_logps) if all_single_logps else None,
        "multi_candidate_logp_mean": statistics.fmean(all_multi_logps) if all_multi_logps else None,
        "per_row": per_row,
    }


def _parse_concept_paths(entries: Sequence[str]) -> Dict[str, str]:
    result = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"expected TAG=PATH, got {entry}")
        tag, path = entry.split("=", 1)
        result[tag] = path
    return result


def _parse_vanilla_paths(entries: Sequence[str]) -> Optional[Dict[str, str]]:
    """Return named paths, or ``None`` for the legacy single-path output schema."""
    if len(entries) == 1 and "=" not in entries[0]:
        return None
    return _parse_concept_paths(entries)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("canonical tokenizer must define eos_token_id")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    concept_paths = _parse_concept_paths(args.concept_csv)
    vanilla_paths = _parse_vanilla_paths(args.vanilla_val)

    results = []
    for checkpoint in args.checkpoints:
        print(f"\n=== Evaluating {checkpoint} ===")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=_dtype(args.torch_dtype, device)
        ).to(device).eval()
        model.config.pad_token_id = tokenizer.pad_token_id
        if vanilla_paths is None:
            ntp: Dict[str, Any] = evaluate_ntp(
                model, tokenizer, args.vanilla_val[0], args.block_size, args.batch_size, device
            )
        else:
            ntp = {
                tag: evaluate_ntp(model, tokenizer, path, args.block_size, args.batch_size, device)
                for tag, path in vanilla_paths.items()
            }
        record: Dict[str, Any] = {"checkpoint": checkpoint, "ntp": ntp, "concept": {}}
        for offset, (tag, path) in enumerate(concept_paths.items()):
            record["concept"][tag] = evaluate_concepts(
                model,
                tokenizer,
                path,
                args.block_size,
                args.batch_size,
                args.n_bootstrap,
                args.seed + offset,
                device,
                gold_column=args.gold_column,
            )
            metric = record["concept"][tag]
            print(
                f"{tag}: set NLL={metric.get('concept_nll_mean')} "
                f"| ALT-ONLY NLL={metric.get('alt_nll_mean')} (decisive) "
                f"| gold NLL={metric.get('gold_nll_mean')} "
                f"| median={metric.get('concept_nll_median')} "
                f"coverage={metric.get('eval_slot_coverage_pct')}"
            )
        results.append(record)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
    with open(args.results_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved {args.results_json}")


if __name__ == "__main__":
    main()
