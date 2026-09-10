#!/usr/bin/env python3
"""Paired bootstrap confidence intervals from SWORDS or HyperLex result JSON."""

import argparse
import json
import random
import statistics
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    return values[int(q * (len(values) - 1))]


def rank(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        shared = (start + stop) / 2 + 1
        for offset in range(start, stop + 1):
            ranks[order[offset]] = shared
        start = stop + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    a, b = rank(x), rank(y)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    numerator = sum((u - ma) * (v - mb) for u, v in zip(a, b))
    denominator = (
        sum((u - ma) ** 2 for u in a) * sum((v - mb) ** 2 for v in b)
    ) ** 0.5
    return numerator / denominator if denominator else None


def bootstrap(values, fn, samples, seed):
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        indices = [rng.randrange(len(values)) for _ in values]
        estimate = fn(indices)
        if estimate is not None:
            estimates.append(float(estimate))
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)] if estimates else None


def swords_pair(baseline, candidate, mode, samples, seed):
    left = {row["target_id"]: row for row in baseline[mode]["per_target"]}
    right = {row["target_id"]: row for row in candidate[mode]["per_target"]}
    ids = sorted(left.keys() & right.keys())
    fields = ["gap", "gap_rat", "auroc", "p_at_1", "alternatives_nll",
              "gold_nll", "rejected_mass_share"]
    result = {"n_common": len(ids), "metrics": {}}
    for field in fields:
        pairs = [(left[i].get(field), right[i].get(field)) for i in ids]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if not pairs:
            continue
        deltas = [b - a for a, b in pairs]
        result["metrics"][field] = {
            "candidate_minus_baseline": statistics.fmean(deltas),
            "ci95": bootstrap(deltas, lambda idx: statistics.fmean(deltas[i] for i in idx), samples, seed),
            "n": len(deltas),
        }
    return result


def hyperlex_pair(baseline, candidate, samples, seed):
    left = {row["row_index"]: row for row in baseline["per_pair"]}
    right = {row["row_index"]: row for row in candidate["per_pair"]}
    ids = sorted(left.keys() & right.keys())

    def statistic(rows, indices, field):
        return spearman([rows[i][field] for i in indices], [rows[i]["gold"] for i in indices])

    left_rows, right_rows = [left[i] for i in ids], [right[i] for i in ids]
    result = {"n_common": len(ids), "metrics": {}}
    for field in ["score", "asymmetry"]:
        observed = statistic(right_rows, range(len(ids)), field) - statistic(left_rows, range(len(ids)), field)
        ci = bootstrap(ids, lambda idx: statistic(right_rows, idx, field) - statistic(left_rows, idx, field), samples, seed)
        result["metrics"][f"spearman_{field}"] = {
            "candidate_minus_baseline": observed, "ci95": ci, "n": len(ids)
        }
    hyp = [i for i in range(len(ids)) if left_rows[i]["type"].startswith("hyp-")]
    if hyp:
        deltas = [
            float(right_rows[i]["asymmetry"] > 0) - float(left_rows[i]["asymmetry"] > 0)
            for i in hyp
        ]
        result["metrics"]["directionality_accuracy"] = {
            "candidate_minus_baseline": statistics.fmean(deltas),
            "ci95": bootstrap(deltas, lambda idx: statistics.fmean(deltas[i] for i in idx), samples, seed),
            "n": len(deltas),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["swords", "hyperlex"], required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-index", type=int, default=0)
    parser.add_argument("--mode", default="left")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = json.loads(Path(args.results_json).read_text())
    baseline = records[args.baseline_index]
    comparisons = []
    for index, candidate in enumerate(records):
        if index == args.baseline_index:
            continue
        paired = (
            swords_pair(baseline, candidate, args.mode, args.samples, args.seed)
            if args.kind == "swords"
            else hyperlex_pair(baseline, candidate, args.samples, args.seed)
        )
        comparisons.append({
            "baseline": baseline["checkpoint"], "candidate": candidate["checkpoint"], **paired
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparisons, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
