#!/usr/bin/env python3
"""HyperLex evaluation for concept-trained causal LMs: graded, *directional* lexical entailment.

Why this benchmark
------------------
The only clean effect this project has is on the hypernym relation, and HyperLex is the standard
external test of exactly that.  It is also the one benchmark whose structure matches the failure
modes we care about:

  hyp-1..hyp-4   hypernym at increasing taxonomic distance   mean rating 4.72 - 5.00
  syn            synonymy                                    4.10
  r-hyp-1..4     the SAME pairs REVERSED                     2.85 down to 1.71
  cohyp          co-hyponyms (our hard negatives)            2.13
  mero           meronymy                                    1.89
  ant / no-rel   antonymy / unrelated                        0.88 / 0.51

Three things fall out of that:

1. **Directionality.** ``hyp`` and ``r-hyp`` are the same word pairs in opposite order. Any
   symmetric scorer -- cosine similarity, co-occurrence, most sentence embeddings -- is at chance
   on this contrast by construction. A causal LM scored with directional templates is not, so
   `directionality_accuracy` is the sharpest single number here.
2. **Co-hyponym discrimination.** `cohyp` is precisely the hard-negative class mined by
   `build_contrastive_dataset.py`. HyperLex says humans rate it 2.13 against 4.9 for true
   hypernyms, so it is separable -- the question is whether *our* models separate it.
3. **Gradedness.** Ratings are continuous, matching an objective that allocates probability mass
   by degree rather than by a binary in/out decision.

Frequency control (do not skip)
-------------------------------
`log p(y | template(x))` is dominated by how frequent `y` is, so an uncorrected score largely
measures unigram frequency. Every score here is PMI-corrected against a neutral template:

    s(x -> y) = mean_t [ logp_norm(y | t(x)) - logp_norm(y | neutral) ]

This is the same surface-form-competition correction that makes MCQA scoring meaningful, and
without it HyperLex correlations are uninterpretable.

Caveat to state in any writeup: HyperLex pairs are context-free, while this project's thesis is
about context-dependent concept slots. Treat it as an external anchor for the hypernym claim, not
as the primary evaluation. Absolute Spearman for an unsupervised 1B decoder will be modest
(distributional unsupervised methods sit around 0.3); only the ordering across arms is evidence.

Usage
-----
  python eval_hyperlex.py --checkpoints ckpt_a ckpt_b --tokenizer_path /path/to/Llama-3.2-1B \\
      --hyperlex data/hyperlex-data/splits/random/hyperlex_test_all_random.txt \\
      --results_json out/hyperlex.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sequence_ncp_trainer import encode_candidate_continuation, sequence_log_probs_from_logits

# Directional templates. Each maps (hyponym x, hypernym y) -> (context, continuation).
# Mixed Hearst-style and copular frames; the mean over templates is the score.
TEMPLATES: List[Tuple[str, str]] = [
    ("A {x} is a type of", "{y}"),
    ("A {x} is a kind of", "{y}"),
    ("{x} and other", "{y}"),
    ("{x} or some other", "{y}"),
    ("The {x} is a", "{y}"),
]
NEUTRAL_CONTEXT = "The word is"  # frequency baseline for the PMI correction

RELATION_GROUPS = {
    "hyp": ("hyp-1", "hyp-2", "hyp-3", "hyp-4"),
    "r-hyp": ("r-hyp-1", "r-hyp-2", "r-hyp-3", "r-hyp-4"),
    "syn": ("syn",),
    "cohyp": ("cohyp",),
    "mero": ("mero",),
    "ant": ("ant",),
    "no-rel": ("no-rel",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--hyperlex", required=True, help="a hyperlex-*.txt / split file")
    parser.add_argument("--results_json", required=True)
    parser.add_argument("--pos", nargs="+", default=["N"], choices=["N", "V"],
                        help="HyperLex POS filter; nouns by default (verbs are only 453 pairs)")
    parser.add_argument("--no_pmi", action="store_true", help="disable the frequency correction")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def load_hyperlex(path: str, keep_pos: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().split()
        assert header[:5] == ["WORD1", "WORD2", "POS", "TYPE", "AVG_SCORE"], f"unexpected header: {header}"
        for line in handle:
            fields = line.split()
            if len(fields) < 5:
                continue
            word1, word2, pos, relation, score = fields[0], fields[1], fields[2], fields[3], float(fields[4])
            if pos not in keep_pos:
                continue
            rows.append({"x": word1, "y": word2, "pos": pos, "type": relation, "gold": score})
    return rows


def _render(template: Tuple[str, str], x: str, y: str) -> Tuple[str, str]:
    """Every template keeps y as the continuation, so the score is directional by construction."""
    context, continuation = template
    return context.format(x=x, y=y), continuation.format(x=x, y=y)


@torch.no_grad()
def _score_pairs(model: Any, tokenizer: Any, pairs: Sequence[Tuple[str, str]],
                 block_size: int, batch_size: int, device: torch.device) -> List[Optional[float]]:
    """Length-normalized log p(continuation | context) for each (context, continuation) pair."""
    encoded, index_map = [], []
    for position, (context, continuation) in enumerate(pairs):
        item = encode_candidate_continuation(tokenizer, context, continuation, block_size)
        if item is not None:
            encoded.append(item)
            index_map.append(position)

    scores: List[Optional[float]] = [None] * len(pairs)
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start : start + batch_size]
        width = max(len(item["input_ids"]) for item in batch)

        def pad(key: str, value: int) -> torch.Tensor:
            return torch.tensor(
                [item[key] + [value] * (width - len(item[key])) for item in batch], device=device
            )

        logits = model(input_ids=pad("input_ids", tokenizer.pad_token_id),
                       attention_mask=pad("attention_mask", 0), use_cache=False).logits
        values = sequence_log_probs_from_logits(logits, pad("labels", -100)).float().cpu().tolist()
        for offset, value in enumerate(values):
            item = batch[offset]
            scores[index_map[start + offset]] = value / max(len(item["continuation_ids"]), 1)
    return scores


def directional_scores(model: Any, tokenizer: Any, rows: Sequence[Dict[str, Any]],
                       use_pmi: bool, block_size: int, batch_size: int,
                       device: torch.device) -> Tuple[List[Optional[float]], List[List[Optional[float]]]]:
    """s(x -> y): mean over templates of the PMI-corrected, length-normalized log-probability."""
    per_template: List[List[Optional[float]]] = []
    for template in TEMPLATES:
        rendered = [_render(template, row["x"], row["y"]) for row in rows]
        values = _score_pairs(model, tokenizer, rendered, block_size, batch_size, device)
        if use_pmi:
            baseline_pairs = [(NEUTRAL_CONTEXT, continuation) for _, continuation in rendered]
            baseline = _score_pairs(model, tokenizer, baseline_pairs, block_size, batch_size, device)
            values = [None if v is None or b is None else v - b for v, b in zip(values, baseline)]
        per_template.append(values)

    combined: List[Optional[float]] = []
    for position in range(len(rows)):
        usable = [t[position] for t in per_template if t[position] is not None]
        combined.append(statistics.fmean(usable) if usable else None)
    return combined, per_template


# ------------------------------------------------------------------------ metrics


def _rank(values: Sequence[float]) -> List[float]:
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


def auroc(scores: Sequence[float], positive: Sequence[bool]) -> Optional[float]:
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return None
    ranks = _rank(scores)
    return (sum(r for r, p in zip(ranks, positive) if p) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_ci(size: int, metric: Any, n_bootstrap: int, seed: int) -> Optional[List[float]]:
    if size <= 1:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(size) for _ in range(size)]
        value = metric(indices)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    values.sort()
    return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]


def evaluate_checkpoint(model: Any, tokenizer: Any, rows: Sequence[Dict[str, Any]],
                        args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    forward, forward_templates = directional_scores(
        model, tokenizer, rows, not args.no_pmi, args.block_size, args.batch_size, device
    )
    # The same pairs scored in the opposite direction -- this is what makes the test directional.
    swapped = [{"x": r["y"], "y": r["x"]} for r in rows]
    backward, backward_templates = directional_scores(
        model, tokenizer, swapped, not args.no_pmi, args.block_size, args.batch_size, device
    )

    usable = [i for i, value in enumerate(forward) if value is not None]
    gold = [rows[i]["gold"] for i in usable]
    score = [forward[i] for i in usable]
    asymmetry = [
        forward[i] - backward[i] if backward[i] is not None else None for i in usable
    ]

    result: Dict[str, Any] = {
        "n_pairs": len(usable),
        "n_dropped": len(rows) - len(usable),
        "spearman_gold": spearman(score, gold),
        "spearman_gold_asymmetry": spearman(
            [a for a in asymmetry if a is not None],
            [g for g, a in zip(gold, asymmetry) if a is not None],
        ),
        "by_type": {},
        "template_metrics": [],
    }

    grouped: Dict[str, List[int]] = defaultdict(list)
    for position, i in enumerate(usable):
        for group, members in RELATION_GROUPS.items():
            if rows[i]["type"] in members:
                grouped[group].append(position)
    for group, positions in sorted(grouped.items()):
        result["by_type"][group] = {
            "n": len(positions),
            "mean_score": statistics.fmean(score[p] for p in positions),
            "mean_gold": statistics.fmean(gold[p] for p in positions),
        }

    # Directionality: on a true hypernym pair, does x -> y outscore y -> x?
    hyp_positions = grouped.get("hyp", [])
    wins = [asymmetry[p] > 0 for p in hyp_positions if asymmetry[p] is not None]
    result["directionality_accuracy"] = statistics.fmean(map(float, wins)) if wins else None
    result["directionality_n"] = len(wins)

    # Discrimination: hypernyms vs each confusable class, and vs everything non-entailing.
    for negative in ["cohyp", "mero", "ant", "no-rel", "r-hyp", "syn"]:
        positions = hyp_positions + grouped.get(negative, [])
        if hyp_positions and grouped.get(negative):
            result["by_type"].setdefault("_auroc", {})[f"hyp_vs_{negative}"] = auroc(
                [score[p] for p in positions],
                [p in set(hyp_positions) for p in positions],
            )
    non_entailing = [p for group in ["cohyp", "mero", "ant", "no-rel"] for p in grouped.get(group, [])]
    if hyp_positions and non_entailing:
        positions = hyp_positions + non_entailing
        result["auroc_hyp_vs_non_entailing"] = auroc(
            [score[p] for p in positions], [p in set(hyp_positions) for p in positions]
        )

    for template_index, template in enumerate(TEMPLATES):
        template_score = [forward_templates[template_index][i] for i in usable]
        template_asymmetry = [
            (forward_templates[template_index][i] - backward_templates[template_index][i])
            if forward_templates[template_index][i] is not None
            and backward_templates[template_index][i] is not None else None
            for i in usable
        ]
        template_wins = [
            template_asymmetry[p] > 0 for p in hyp_positions
            if template_asymmetry[p] is not None
        ]
        valid_positions = [p for p, value in enumerate(template_score) if value is not None]
        result["template_metrics"].append({
            "template": template,
            "spearman_gold": spearman(
                [template_score[p] for p in valid_positions],
                [gold[p] for p in valid_positions],
            ),
            "directionality_accuracy": (
                statistics.fmean(map(float, template_wins)) if template_wins else None
            ),
        })

    result["per_pair"] = [
        {
            "row_index": i,
            "x": rows[i]["x"], "y": rows[i]["y"], "type": rows[i]["type"],
            "gold": rows[i]["gold"], "score": forward[i], "reverse_score": backward[i],
            "asymmetry": forward[i] - backward[i] if backward[i] is not None else None,
        }
        for i in usable
    ]

    result["ci95"] = {
        "spearman_gold": bootstrap_ci(
            len(score), lambda idx: spearman([score[i] for i in idx], [gold[i] for i in idx]),
            args.n_bootstrap, args.seed,
        ),
        "directionality_accuracy": bootstrap_ci(
            len(wins), lambda idx: statistics.fmean(float(wins[i]) for i in idx),
            args.n_bootstrap, args.seed + 1,
        ) if wins else None,
    }
    if hyp_positions and non_entailing:
        positions = hyp_positions + non_entailing
        auc_scores = [score[p] for p in positions]
        auc_labels = [p in set(hyp_positions) for p in positions]
        result["ci95"]["auroc_hyp_vs_non_entailing"] = bootstrap_ci(
            len(positions),
            lambda idx: auroc([auc_scores[i] for i in idx], [auc_labels[i] for i in idx]),
            args.n_bootstrap, args.seed + 2,
        )
    return result


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

    rows = load_hyperlex(args.hyperlex, args.pos)
    counts = defaultdict(int)
    for row in rows:
        counts[row["type"]] += 1
    print(f"HyperLex: {len(rows)} pairs (POS={args.pos}), pmi={'off' if args.no_pmi else 'on'}")
    print("  types:", dict(sorted(counts.items())))

    results = []
    for checkpoint in args.checkpoints:
        print(f"\n=== {checkpoint} ===")
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype).to(device).eval()
        model.config.pad_token_id = tokenizer.pad_token_id
        record = {"checkpoint": checkpoint, **evaluate_checkpoint(model, tokenizer, rows, args, device)}
        print(f"  spearman(gold)          = {record['spearman_gold']}")
        print(f"  spearman(gold, x->y minus y->x) = {record['spearman_gold_asymmetry']}")
        print(f"  directionality accuracy = {record['directionality_accuracy']} "
              f"(n={record['directionality_n']}, chance=0.5)")
        print(f"  AUROC hyp vs non-entailing = {record.get('auroc_hyp_vs_non_entailing')}")
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
