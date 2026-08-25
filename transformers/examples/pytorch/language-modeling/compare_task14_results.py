#!/usr/bin/env python3
"""Paired bootstrap comparisons for Task-14 external benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_json", required=True)
    parser.add_argument("--treatment_json", required=True)
    parser.add_argument("--benchmark", choices=["swords", "hyperlex", "bm_semlex"], required=True)
    parser.add_argument("--mode", default="left")
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_json", required=True)
    return parser.parse_args()


def load_first(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{path} must contain exactly one checkpoint result")
    return value[0]


def ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        rank = (start + stop) / 2 + 1
        for position in range(start, stop + 1):
            result[order[position]] = rank
        start = stop + 1
    return result


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = (
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    ) ** 0.5
    return numerator / denominator if denominator else None


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    score_ranks = ranks(scores)
    rank_sum = sum(rank for rank, label in zip(score_ranks, labels) if label)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def percentile_interval(values: List[float]) -> List[float]:
    values = sorted(values)
    return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]


def paired_rows(left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]],
                key: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    a = {str(row[key]): row for row in left}
    b = {str(row[key]): row for row in right}
    if set(a) != set(b):
        raise ValueError(
            f"paired comparison requires identical rows: baseline={len(a)} treatment={len(b)} "
            f"intersection={len(set(a) & set(b))}"
        )
    return [(a[row_id], b[row_id]) for row_id in sorted(a)]


def bootstrap_mean_deltas(pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
                          fields: Sequence[str], n_bootstrap: int,
                          seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    output: Dict[str, Any] = {}
    for field in fields:
        clean = [(float(a[field]), float(b[field])) for a, b in pairs
                 if a.get(field) is not None and b.get(field) is not None]
        if not clean:
            continue
        observed = statistics.fmean(b - a for a, b in clean)
        samples = []
        for _ in range(n_bootstrap):
            draw = [clean[rng.randrange(len(clean))] for _ in range(len(clean))]
            samples.append(statistics.fmean(b - a for a, b in draw))
        output[field] = {
            "n": len(clean), "treatment_minus_baseline": observed,
            "ci95": percentile_interval(samples),
        }
    return output


def compare_swords(base: Dict[str, Any], treatment: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    pairs = paired_rows(base[args.mode]["per_target"], treatment[args.mode]["per_target"], "target_id")
    fields = ["gap", "gap_rat", "spearman", "auroc", "p_at_1", "oracle_f1_at_k",
              "gold_nll", "alternatives_nll", "inclusive_nll",
              "acceptable_single_logp_mean", "acceptable_multi_logp_mean"]
    return {"n_paired": len(pairs), "metrics": bootstrap_mean_deltas(
        pairs, fields, args.n_bootstrap, args.seed
    )}


def compare_bm(base: Dict[str, Any], treatment: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    pairs = paired_rows(base[args.mode]["per_row"], treatment[args.mode]["per_row"], "row_id")
    for a, b in pairs:
        a["correct_float"], b["correct_float"] = float(a["correct"]), float(b["correct"])
    return {"n_paired": len(pairs), "metrics": bootstrap_mean_deltas(
        pairs, ["correct_float", "margin"], args.n_bootstrap, args.seed
    )}


def compare_hyperlex(base: Dict[str, Any], treatment: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    pairs = paired_rows(base["per_pair"], treatment["per_pair"], "row_index")
    rng = random.Random(args.seed)

    def metrics(draw: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Optional[float]]:
        gold = [float(a["gold"]) for a, _ in draw]
        base_score = [float(a["score"]) for a, _ in draw]
        treatment_score = [float(b["score"]) for _, b in draw]
        hyp = [(a, b) for a, b in draw if str(a["type"]).startswith("hyp-")]
        non_entailing = [(a, b) for a, b in draw if a["type"] in {"cohyp", "mero", "ant", "no-rel"}]
        auc_rows = hyp + non_entailing
        labels = [True] * len(hyp) + [False] * len(non_entailing)
        return {
            "spearman_gold": (spearman(treatment_score, gold) or 0.0) - (spearman(base_score, gold) or 0.0),
            "directionality_accuracy": (
                statistics.fmean(float(b["asymmetry"] > 0) for _, b in hyp)
                - statistics.fmean(float(a["asymmetry"] > 0) for a, _ in hyp)
            ) if hyp else None,
            "auroc_hyp_vs_non_entailing": (
                (auroc([float(b["score"]) for _, b in auc_rows], labels) or 0.0)
                - (auroc([float(a["score"]) for a, _ in auc_rows], labels) or 0.0)
            ) if hyp and non_entailing else None,
        }

    observed = metrics(pairs)
    samples: Dict[str, List[float]] = {key: [] for key in observed}
    for _ in range(args.n_bootstrap):
        draw = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        for key, value in metrics(draw).items():
            if value is not None:
                samples[key].append(float(value))
    return {
        "n_paired": len(pairs),
        "metrics": {
            key: {"treatment_minus_baseline": value,
                  "ci95": percentile_interval(samples[key]) if samples[key] else None}
            for key, value in observed.items()
        },
    }


def main() -> None:
    args = parse_args()
    baseline, treatment = load_first(args.baseline_json), load_first(args.treatment_json)
    if args.benchmark == "swords":
        result = compare_swords(baseline, treatment, args)
    elif args.benchmark == "bm_semlex":
        result = compare_bm(baseline, treatment, args)
    else:
        result = compare_hyperlex(baseline, treatment, args)
    result.update({
        "benchmark": args.benchmark, "mode": args.mode,
        "baseline": args.baseline_json, "treatment": args.treatment_json,
        "n_bootstrap": args.n_bootstrap, "seed": args.seed,
    })
    with open(args.results_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
