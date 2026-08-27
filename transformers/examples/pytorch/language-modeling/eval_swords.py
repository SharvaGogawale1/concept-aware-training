#!/usr/bin/env python3
"""SWORDS (Stanford Word Substitution Benchmark) evaluation for concept-trained causal LMs.

Why this benchmark
------------------
SWORDS is the external version of the question this project has been answering with its own
unvalidated candidate sets: *given a context and a target word, which substitutes are actually
acceptable?*  Every substitute carries 3 or 10 human TRUE/FALSE/UNSURE judgements, so the label
is graded and human, not WordNet- or LLM-generated. This makes SWORDS a substantially stronger
external test, without making unmatched SWORDS-versus-YouTube comparisons causal.

Two scoring modes
-----------------
``left``  scores ``log p(substitute | context[:offset])`` using ``encode_candidate_continuation``
          -- byte-for-byte the encoder used by ``sequence_ncp_trainer`` and ``eval_concept_ppl_v3``.
          This is exactly what the concept objective optimizes, so it is the mode where a gain is
          attributable to training.

``full``  substitutes the word in place and scores the whole sentence.  This is the standard
          lexical-substitution protocol and uses the right context too.  A causal LM trained only
          on left-context concept slots should be worse here; the interesting quantity is the
          *gap* between the two modes, and whether concept training narrows it.

Report both.  ``--length_normalize`` divides by continuation token count; multi-token substitutes
are otherwise penalized, which confounds substitute length with acceptability.

Metrics (macro-averaged over targets, then broken out by POS)
-------------------------------------------------------------
spearman  rank correlation between model score and human acceptability -- uses the full graded signal
gap       generalized average precision (Kishida 2005), the standard graded lexsub ranking metric
p_at_1    is the model's top-ranked substitute acceptable at the conservative threshold?
auroc     separation of acceptable from unacceptable substitutes

Usage
-----
  python eval_swords.py --checkpoints ckpt_a ckpt_b --tokenizer_path /path/to/Llama-3.2-1B \\
      --swords_json data/swords/swords-v1.1_dev.json.gz --results_json out/swords.json
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sequence_ncp_trainer import (
    encode_candidate_continuation,
    has_strict_prefix_collision,
    sequence_log_probs_from_logits,
)

CONSERVATIVE = 0.5  # SWORDS "conservative" acceptability threshold
LENIENT = 0.1       # SWORDS "lenient" threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--swords_json", required=True, help="swords-v1.1_{dev,test}.json.gz")
    parser.add_argument("--results_json", required=True)
    parser.add_argument("--modes", nargs="+", default=["left", "full"], choices=["left", "full"])
    parser.add_argument("--length_normalize", action="store_true", default=True)
    parser.add_argument("--no_length_normalize", dest="length_normalize", action="store_false")
    parser.add_argument("--max_targets", type=int, default=None, help="smoke-test subset")
    parser.add_argument("--target_ids_file", default=None,
                        help="optional allowlist used for the leak-free SWORDS-dev validation side")
    parser.add_argument("--max_context_chars", type=int, default=1200,
                        help="left-truncate very long Enron/CoInCo contexts before tokenizing")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


# ------------------------------------------------------------------ benchmark loading


def load_swords(path: str, max_targets: Optional[int], max_context_chars: int,
                target_ids_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Flatten SWORDS into one record per target, with graded human scores per substitute."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)

    allowlist = None
    if target_ids_file:
        with open(target_ids_file, encoding="utf-8") as handle:
            allowlist = {line.strip() for line in handle if line.strip()}

    by_target: Dict[str, List[Tuple[str, float, float, int]]] = defaultdict(list)
    for substitute_id, substitute in data["substitutes"].items():
        labels = data["substitute_labels"][substitute_id]
        if not labels:
            continue
        # The README diagnostic counts UNSURE against; the official evaluator removes UNSURE
        # before its ratio-weighted GAP. Retain both plus the raw TRUE count.
        score = sum(label == "TRUE" for label in labels) / len(labels)
        confident = [label for label in labels if label != "UNSURE"]
        score_no_unsure = (
            sum(label == "TRUE" for label in confident) / len(confident) if confident else score
        )
        by_target[substitute["target_id"]].append(
            (substitute["substitute"], score, score_no_unsure,
             sum(label == "TRUE" for label in labels))
        )

    records: List[Dict[str, Any]] = []
    for target_id, target in data["targets"].items():
        if allowlist is not None and target_id not in allowlist:
            continue
        substitutes = by_target.get(target_id, [])
        if len(substitutes) < 2:
            continue
        context = data["contexts"][target["context_id"]]["context"]
        offset = int(target["offset"])
        word = target["target"]
        if context[offset : offset + len(word)] != word:
            continue  # offset does not line up; skip rather than silently mis-score
        # Long mail threads blow past block_size; keep the tail nearest the target.
        trimmed_start = max(0, offset - max_context_chars)
        records.append(
            {
                "target_id": target_id,
                "pos": target.get("pos", "?"),
                "target": word,
                "left": context[trimmed_start:offset],
                "right": context[offset + len(word) : offset + len(word) + max_context_chars],
                "substitutes": [s for s, _, _, _ in substitutes],
                "human": [h for _, h, _, _ in substitutes],
                "human_no_unsure": [h for _, _, h, _ in substitutes],
                "human_true_count": [h for _, _, _, h in substitutes],
            }
        )
    records.sort(key=lambda r: r["target_id"])
    return records[:max_targets] if max_targets else records


# ------------------------------------------------------------------------ metrics


def _rank(values: Sequence[float]) -> List[float]:
    """Average ranks, ties shared (needed for a correct Spearman on tied human scores)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def official_gap(model: Sequence[float], human: Sequence[float]) -> Optional[float]:
    """Melamud et al. GAP, matching SWORDS' checksum-pinned reference implementation.

    SWORDS reports two variants: raw TRUE counts (``gap``) and TRUE-rater ratios
    (``gap_rat``).  The algebra is identical; callers select the corresponding ``human`` weights.
    """
    if not any(h > 0 for h in human):
        return None
    predicted = [human[i] for i in sorted(range(len(model)), key=lambda i: -model[i])]
    ideal = sorted(human, reverse=True)

    def accumulate(weights: Sequence[float]) -> float:
        total, running = 0.0, 0.0
        for position, weight in enumerate(weights, start=1):
            running += weight
            if weight > 0:
                total += running / position
        return total

    denominator = accumulate(ideal)
    return accumulate(predicted) / denominator if denominator else None


def bootstrap_mean_ci(values: Sequence[float], n_bootstrap: int, seed: int) -> Optional[List[float]]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.tensor(clean, dtype=torch.float64)
    means = []
    for _ in range(n_bootstrap):
        index = torch.randint(len(clean), (len(clean),), generator=generator)
        means.append(float(tensor[index].mean()))
    means.sort()
    low = means[max(int(0.025 * len(means)), 0)]
    high = means[min(int(0.975 * len(means)), len(means) - 1)]
    return [low, high]


def auroc(scores: Sequence[float], positive: Sequence[bool]) -> Optional[float]:
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return None
    ranks = _rank(scores)
    rank_sum = sum(r for r, p in zip(ranks, positive) if p)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ------------------------------------------------------------------------ scoring


@torch.no_grad()
def _score_batch(model: Any, encoded: Sequence[Dict[str, Any]], pad_id: int,
                 batch_size: int, device: torch.device) -> List[float]:
    scores: List[float] = []
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        width = max(len(item["input_ids"]) for item in batch)

        def pad(key: str, value: int) -> torch.Tensor:
            return torch.tensor(
                [item[key] + [value] * (width - len(item[key])) for item in batch], device=device
            )

        logits = model(input_ids=pad("input_ids", pad_id),
                       attention_mask=pad("attention_mask", 0), use_cache=False).logits
        scores.extend(sequence_log_probs_from_logits(logits, pad("labels", -100)).float().cpu().tolist())
    return scores


@torch.no_grad()
def score_record(model: Any, tokenizer: Any, record: Dict[str, Any], mode: str,
                 block_size: int, batch_size: int, length_normalize: bool,
                 device: torch.device) -> Optional[Dict[str, Any]]:
    """Return model scores aligned with record['substitutes'], or None if nothing is scoreable."""
    encoded, keep = [], []
    for index, substitute in enumerate(record["substitutes"]):
        if mode == "left":
            context = record["left"]
            candidate = substitute
        else:
            # Substitute in place, then score the substituted word *plus its right context*.
            context = record["left"]
            candidate = f'{substitute}{record["right"]}'
        item = encode_candidate_continuation(tokenizer, context, candidate, block_size)
        if item is None:
            continue
        encoded.append(item)
        keep.append(index)
    if len(keep) < 2:
        return None

    exact = _score_batch(model, encoded, tokenizer.pad_token_id, batch_size, device)
    ranking = (
        [score / max(len(item["continuation_ids"]), 1) for score, item in zip(exact, encoded)]
        if length_normalize else list(exact)
    )
    return {
        "indices": keep,
        "scores": ranking,
        "exact_logps": exact,
        "lengths": [len(item["continuation_ids"]) for item in encoded],
        "token_sequences": [item["continuation_ids"] for item in encoded],
    }


def evaluate_checkpoint(model: Any, tokenizer: Any, records: Sequence[Dict[str, Any]],
                        args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for mode in args.modes:
        per_target, skipped = [], 0
        for record in records:
            scored = score_record(model, tokenizer, record, mode, args.block_size,
                                  args.batch_size, args.length_normalize, device)
            if scored is None:
                skipped += 1
                continue
            # Match the official ranker preprocessing at least at the surface level: the observed
            # target is not an alternative. Lemma-level merging remains the official evaluator's
            # responsibility and is checked by fixtures in test_task14_external.py.
            keep = [j for j, original in enumerate(scored["indices"])
                    if record["substitutes"][original].casefold() != record["target"].casefold()]
            if len(keep) < 2:
                skipped += 1
                continue
            model_scores = [scored["scores"][j] for j in keep]
            exact_logps = [scored["exact_logps"][j] for j in keep]
            lengths = [scored["lengths"][j] for j in keep]
            token_sequences = [scored["token_sequences"][j] for j in keep]
            original_indices = [scored["indices"][j] for j in keep]
            human = [record["human"][i] for i in original_indices]
            human_official = [record["human_no_unsure"][i] for i in original_indices]
            human_counts = [record["human_true_count"][i] for i in original_indices]
            acceptable = [h >= CONSERVATIVE for h in human]
            # This is explicitly an oracle-k diagnostic: k comes from the gold labels.
            k = sum(acceptable)
            if k:
                top_k = sorted(range(len(model_scores)), key=lambda i: -model_scores[i])[:k]
                hits = sum(acceptable[i] for i in top_k)
                oracle_precision = oracle_recall = hits / k
                oracle_f1 = oracle_precision
            else:
                oracle_precision = oracle_recall = oracle_f1 = None
            predicted = set(sorted(range(len(model_scores)), key=lambda i: -model_scores[i])[:k])
            oracle_accuracy = statistics.fmean(
                float((i in predicted) == acceptable[i]) for i in range(len(model_scores))
            )

            gold_candidate = (record["target"] if mode == "left"
                              else f'{record["target"]}{record["right"]}')
            gold_encoded = encode_candidate_continuation(
                tokenizer, record["left"], gold_candidate, args.block_size
            )
            gold_logp = None
            gold_length = None
            if gold_encoded is not None:
                gold_logp = _score_batch(
                    model, [gold_encoded], tokenizer.pad_token_id, 1, device
                )[0]
                gold_length = len(gold_encoded["continuation_ids"])

            positive_logps = [value for value, positive in zip(exact_logps, acceptable) if positive]
            positive_sequences = [seq for seq, positive in zip(token_sequences, acceptable) if positive]
            mass_prefix_collision = has_strict_prefix_collision(
                positive_sequences + ([gold_encoded["continuation_ids"]]
                                      if gold_encoded is not None else [])
            ) if positive_sequences else False
            alt_nll = inclusive_nll = None
            if positive_logps and not mass_prefix_collision:
                alt_tensor = torch.tensor(positive_logps, dtype=torch.float64)
                alt_nll = float(-torch.logsumexp(alt_tensor, dim=0))
                if gold_logp is not None:
                    inclusive_nll = float(-torch.logsumexp(
                        torch.cat([torch.tensor([gold_logp], dtype=torch.float64), alt_tensor]), dim=0
                    ))

            # Human-REJECTED substitutes: below the lenient threshold, i.e. unacceptable even
            # under the most permissive reading.  This is the set contrastive training claims to
            # suppress.  `auroc` says whether they RANK below acceptable ones; these say whether
            # their probability was actually pushed down, which is a different question and the
            # one an InfoNCE arm is making a claim about.
            rejected = [h < LENIENT for h in human]
            rejected_logps = [value for value, drop in zip(exact_logps, rejected) if drop]
            rejected_sequences = [seq for seq, drop in zip(token_sequences, rejected) if drop]
            rejected_collision = (
                has_strict_prefix_collision(rejected_sequences) if rejected_sequences else False
            )
            rejected_nll = rejected_mass_share = None
            if rejected_logps and not rejected_collision:
                rejected_tensor = torch.tensor(rejected_logps, dtype=torch.float64)
                rejected_nll = float(-torch.logsumexp(rejected_tensor, dim=0))
                if alt_nll is not None:
                    # p_rejected / (p_rejected + p_acceptable), computed as a sigmoid of the NLL
                    # difference so no probability is ever materialized.  Scale-free, and LOWER is
                    # better -- the same direction as every NLL here, so a paired delta reads the
                    # same way.  Raw `rejected_nll` is the opposite: higher means less mass.
                    rejected_mass_share = float(
                        torch.sigmoid(torch.tensor(alt_nll - rejected_nll, dtype=torch.float64))
                    )

            acceptable_single = [value for value, positive, length in zip(exact_logps, acceptable, lengths)
                                 if positive and length == 1]
            acceptable_multi = [value for value, positive, length in zip(exact_logps, acceptable, lengths)
                                if positive and length > 1]
            per_target.append(
                {
                    "oracle_f1_at_k": oracle_f1,
                    "oracle_precision_at_k": oracle_precision,
                    "oracle_recall_at_k": oracle_recall,
                    "oracle_accuracy_at_k": oracle_accuracy,
                    "target_id": record["target_id"],
                    "pos": record["pos"],
                    "n": len(model_scores),
                    "n_acceptable": sum(acceptable),
                    "spearman": spearman(model_scores, human_official),
                    "gap": official_gap(model_scores, human_counts),
                    "gap_rat": official_gap(model_scores, human_official),
                    "auroc": auroc(model_scores, acceptable),
                    "p_at_1": float(
                        human[max(range(len(model_scores)), key=lambda i: model_scores[i])] >= CONSERVATIVE
                    ),
                    "p_at_1_lenient": float(
                        human[max(range(len(model_scores)), key=lambda i: model_scores[i])] >= LENIENT
                    ),
                    "gold_logp": gold_logp,
                    "gold_nll": -gold_logp if gold_logp is not None else None,
                    "gold_length": gold_length,
                    "alternatives_nll": alt_nll,
                    "inclusive_nll": inclusive_nll,
                    "mass_prefix_collision": mass_prefix_collision,
                    "n_rejected": sum(rejected),
                    "rejected_nll": rejected_nll,
                    "rejected_mass_share": rejected_mass_share,
                    "acceptable_single_logp_mean": (
                        statistics.fmean(acceptable_single) if acceptable_single else None
                    ),
                    "acceptable_multi_logp_mean": (
                        statistics.fmean(acceptable_multi) if acceptable_multi else None
                    ),
                }
            )

        def macro(field: str, rows: Sequence[Dict[str, Any]]) -> Optional[float]:
            values = [r[field] for r in rows if r[field] is not None]
            return statistics.fmean(values) if values else None

        headline_fields = [
            "spearman", "gap", "gap_rat", "auroc", "p_at_1", "p_at_1_lenient",
            "oracle_f1_at_k", "oracle_precision_at_k", "oracle_recall_at_k",
            "oracle_accuracy_at_k", "gold_nll", "alternatives_nll", "inclusive_nll",
            "acceptable_single_logp_mean", "acceptable_multi_logp_mean",
            "rejected_nll", "rejected_mass_share",
        ]
        summary = {
            "targets_scored": len(per_target),
            "targets_skipped": skipped,
            "substitutes_scored": sum(r["n"] for r in per_target),
            "mass_targets_excluded_prefix_collision": sum(
                int(r["mass_prefix_collision"]) for r in per_target
            ),
            **{field: macro(field, per_target) for field in headline_fields},
            "ci95": {
                field: bootstrap_mean_ci(
                    [r[field] for r in per_target if r[field] is not None],
                    args.n_bootstrap, args.seed,
                )
                for field in headline_fields
            },
            "by_pos": {
                pos: {field: macro(field, [r for r in per_target if r["pos"] == pos])
                      for field in ["spearman", "gap", "gap_rat", "auroc", "p_at_1",
                                    "oracle_f1_at_k", "gold_nll", "alternatives_nll"]}
                for pos in sorted({r["pos"] for r in per_target})
            },
            "per_target": per_target,
        }
        out[mode] = summary
    # The left/full gap: how much does the model rely on right context it never trained to use?
    if "left" in out and "full" in out:
        for field in ["spearman", "gap", "gap_rat", "auroc"]:
            left, full = out["left"][field], out["full"][field]
            out.setdefault("mode_gap", {})[field] = (
                full - left if left is not None and full is not None else None
            )
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.float32 if device.type == "cpu"
             else {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[args.torch_dtype])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    records = load_swords(
        args.swords_json, args.max_targets, args.max_context_chars, args.target_ids_file
    )
    print(f"SWORDS: {len(records)} targets, "
          f"{sum(len(r['substitutes']) for r in records)} substitutes, "
          f"modes={args.modes}, length_normalize={args.length_normalize}")

    results = []
    for checkpoint in args.checkpoints:
        print(f"\n=== {checkpoint} ===")
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype).to(device).eval()
        model.config.pad_token_id = tokenizer.pad_token_id
        record = {"checkpoint": checkpoint, **evaluate_checkpoint(model, tokenizer, records, args, device)}
        for mode in args.modes:
            summary = record[mode]
            print(f"  {mode:5s} spearman={summary['spearman']:.4f} "
                  f"GAP={summary['gap']:.4f} GAP-ratio={summary['gap_rat']:.4f} "
                  f"auroc={summary['auroc']:.4f} oracle-F1@k={summary['oracle_f1_at_k']:.4f} "
                  f"rejected-share={summary['rejected_mass_share']} "
                  f"p@1={summary['p_at_1']:.4f} "
                  f"(n={summary['targets_scored']})")
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
