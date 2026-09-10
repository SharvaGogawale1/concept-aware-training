#!/usr/bin/env python3
"""Build one readable CSV table from Task 15/16 metric artifacts."""

import argparse
import csv
import json
from pathlib import Path


def load_records(path):
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else []


def by_checkpoint(path):
    return {str(row["checkpoint"]): row for row in load_records(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON mapping readable method name to checkpoint")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.result_dir)
    perplexity = by_checkpoint(root / "perplexity.json")
    concepts = by_checkpoint(root / "concept_sets.json")
    swords = by_checkpoint(root / "swords.json") or by_checkpoint(root / "swords_zero_shot.json")
    bm = by_checkpoint(root / "bm_semlex.json")
    hyperlex = by_checkpoint(root / "hyperlex.json")

    rows = []
    for method, checkpoint in manifest.items():
        checkpoint = str(checkpoint)
        row = {"method": method, "checkpoint": checkpoint}
        if checkpoint in perplexity:
            record = perplexity[checkpoint]
            row.update({
                "global_nll": record["global"]["mean_nll"],
                "global_ppl": record["global"]["perplexity"],
                "content_nll": record["content_words"]["mean_nll"],
                "content_ppl": record["content_words"]["perplexity"],
            })
        if checkpoint in concepts:
            record = concepts[checkpoint]
            row.update({
                "concept_set_mass": record.get("set_mass"),
                "alternative_nll": record.get("alternative_nll"),
                "observed_target_nll": record.get("observed_target_nll"),
                "candidate_entropy": record.get("candidate_entropy"),
                "minimum_candidate_probability": record.get("minimum_candidate_probability"),
            })
        if checkpoint in swords:
            record = swords[checkpoint].get("left", {})
            row.update({
                "swords_gap": record.get("gap"),
                "swords_gap_ratio": record.get("gap_rat"),
                "swords_auroc": record.get("auroc"),
                "swords_p_at_1": record.get("p_at_1"),
                "swords_alternative_nll": record.get("alternatives_nll"),
                "swords_observed_target_nll": record.get("gold_nll"),
                "swords_rejected_mass_share": record.get("rejected_mass_share"),
            })
        if checkpoint in bm:
            record = bm[checkpoint].get("left", {})
            row["bm_semlex_accuracy"] = record.get("accuracy")
        if checkpoint in hyperlex:
            record = hyperlex[checkpoint]
            row.update({
                "hyperlex_spearman": record.get("spearman_gold"),
                "hyperlex_directionality": record.get("directionality_accuracy"),
                "hyperlex_hyp_vs_nonentailing_auroc": record.get("auroc_hyp_vs_non_entailing"),
            })
        sts_path = root / f"sts_{method}.csv"
        if sts_path.exists():
            with sts_path.open() as handle:
                scores = [float(r["main_score"]) for r in csv.DictReader(handle)]
            row["sts_mean"] = sum(scores) / len(scores) if scores else None
            row["sts_tasks"] = len(scores)
        rows.append(row)

    fields = ["method", "checkpoint"] + sorted(
        {key for row in rows for key in row} - {"method", "checkpoint"}
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output.read_text())


if __name__ == "__main__":
    main()
