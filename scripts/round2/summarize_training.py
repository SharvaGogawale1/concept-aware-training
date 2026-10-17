"""Training-history summary of every finished round-2 run: the manipulation check.

For each run, the logged R (mass_term), B (balance_term), S (uncapped_term), CE and concept
loss at the first logged step and at the end (mean of the last 5 logged steps), plus the
recorded train_runtime.  Grouped by model x objective x gamma with each seed shown, and
flagged when a value is non-finite or one seed sits far from the other two.

    python scripts/round2/summarize_training.py
Writes results/round2/training_summary.csv and prints the table.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import r2common as C

TERMS = ("ce_loss", "concept_loss", "mass_term", "balance_term", "uncapped_term")
TAIL = 5


def runs():
    for marker in sorted(C.CHECKPOINTS.glob("*/*/g*/seed_*/round2_run.json")):
        rec = json.loads(marker.read_text())
        if rec.get("status") != "ok":
            continue
        yield marker.parent, rec


def history(path):
    rows, runtime = [], None
    for line in (path / "training_history.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "mass_term" in row:
            rows.append(row)
        if "train_runtime" in row:
            runtime = row["train_runtime"] / 60
    return rows, runtime


def far_seed(values):
    """Index of a seed far from the other two: its gap to their mean exceeds 3x their own
    gap and 5% of the mean.  None with fewer than three seeds."""
    if len(values) < 3:
        return None
    for i, v in enumerate(values):
        rest = [x for j, x in enumerate(values) if j != i]
        centre, spread = statistics.fmean(rest), abs(rest[0] - rest[1])
        if abs(v - centre) > max(3 * spread, 0.05 * abs(centre)):
            return i
    return None


def main():
    table = defaultdict(list)
    out_rows = []
    for path, rec in runs():
        rows, runtime = history(path)
        tag, objective, gamma, seed = path.parts[-4], path.parts[-3], path.parts[-2], path.parts[-1]
        if not rows:
            out_rows.append({"model": tag, "objective": objective, "gamma": gamma, "seed": seed,
                             "flag": "no logged terms"})
            continue
        first = {k: rows[0].get(k) for k in TERMS}
        last = {k: statistics.fmean(r[k] for r in rows[-TAIL:] if k in r) for k in TERMS}
        nonfinite = [k for k in TERMS for v in (first[k], last[k])
                     if v is None or not math.isfinite(v)]
        row = {"model": tag, "objective": objective, "gamma": gamma, "seed": seed,
               "logged_steps": len(rows), "train_minutes": round(runtime, 1) if runtime else "",
               **{f"first_{k}": first[k] for k in TERMS}, **{f"last_{k}": last[k] for k in TERMS},
               "flag": ("non-finite " + ",".join(sorted(set(nonfinite)))) if nonfinite else ""}
        out_rows.append(row)
        table[(tag, objective, gamma)].append(row)
    for key, rows in table.items():
        rows.sort(key=lambda r: int(r["seed"].split("_")[1]))
        for k in TERMS:
            i = far_seed([r[f"last_{k}"] for r in rows])
            if i is not None:
                rows[i]["flag"] = (rows[i]["flag"] + f"; seed far from the others on last {k}").strip("; ")
    out = C.R2 / "training_summary.csv"
    fields = sorted({k for r in out_rows for k in r}, key=lambda k: (k not in ("model", "objective", "gamma", "seed"), k))
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"{len(out_rows)} runs -> {out}\n")
    print(f"{'model':16s} {'objective':12s} {'gamma':7s} {'seed':9s} "
          + " ".join(f"{k.replace('_term', '').replace('_loss', ''):>16s}" for k in TERMS) + "  flag")
    print(" " * 46 + " ".join(f"{'first -> last':>16s}" for _ in TERMS))
    for key in sorted(table):
        for r in table[key]:
            cells = " ".join(f"{r[f'first_{k}']:7.3f}->{r[f'last_{k}']:7.3f}" for k in TERMS)
            print(f"{r['model']:16s} {r['objective']:12s} {r['gamma']:7s} {r['seed']:9s} {cells}  {r['flag']}")
    flagged = [r for r in out_rows if r.get("flag")]
    print(f"\nflags: {len(flagged)}" + "".join(f"\n  {r['model']} {r['objective']} {r['gamma']} {r['seed']}: {r['flag']}"
                                        for r in flagged))


if __name__ == "__main__":
    main()
