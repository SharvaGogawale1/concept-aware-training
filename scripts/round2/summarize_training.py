"""Training-history summary of every finished round-2 run: the manipulation check.

For each run, the logged R (mass_term), B (balance_term), S (uncapped_term), CE and concept
loss at the first logged step and at the end (mean of the last 5 logged steps), plus the
recorded train_runtime; non-finite values are flagged.  Logged values are single-batch losses
and the batch order depends only on the seed, so they are compared across OBJECTIVES within a
seed (identical batches), never across seeds: the paired table gives, per model x seed x gamma,
R(Alt-Mass) - R(Alt-Balance) and B(Alt-Balance) - B(Alt-Mass), both expected below zero.

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
    for rows in table.values():
        rows.sort(key=lambda r: int(r["seed"].split("_")[1]))
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
    by = {(r["model"], r["objective"], r["gamma"], r["seed"]): r for r in out_rows if "last_mass_term" in r}
    pairs = sorted({(m, g, sd) for (m, o, g, sd) in by if (m, "alt_mass", g, sd) in by and (m, "alt_balance", g, sd) in by})
    print(f"\npaired on identical batches (same seed): end-of-training differences")
    print(f"{'model':16s} {'gamma':7s} {'seed':9s} {'R(mass)-R(bal)':>15s} {'B(bal)-B(mass)':>15s}")
    hits = [0, 0]
    for m, g, sd in pairs:
        dr = by[(m, "alt_mass", g, sd)]["last_mass_term"] - by[(m, "alt_balance", g, sd)]["last_mass_term"]
        db = by[(m, "alt_balance", g, sd)]["last_balance_term"] - by[(m, "alt_mass", g, sd)]["last_balance_term"]
        hits[0] += dr < 0
        hits[1] += db < 0
        print(f"{m:16s} {g:7s} {sd:9s} {dr:15.3f} {db:15.3f}")
    print(f"expected sign (< 0): mass {hits[0]}/{len(pairs)}, balance {hits[1]}/{len(pairs)}")
    ws = sorted({(m, g, sd) for (m, o, g, sd) in by if o == "concept_uncapped" and (m, "alt_mass", g, sd) in by})
    if ws:
        print(f"\nwhole-set control vs Alt-Mass, same seed and batches: S(whole-set) - S(Alt-Mass)")
        n_ok = 0
        for m, g, sd in ws:
            ds = by[(m, "concept_uncapped", g, sd)]["last_uncapped_term"] - by[(m, "alt_mass", g, sd)]["last_uncapped_term"]
            n_ok += ds < 0
            print(f"{m:16s} {g:7s} {sd:9s} {ds:15.3f}")
        print(f"expected sign (< 0): {n_ok}/{len(ws)}")
    flagged = [r for r in out_rows if r.get("flag")]
    print(f"\nflags: {len(flagged)}" + "".join(f"\n  {r['model']} {r['objective']} {r['gamma']} {r['seed']}: {r['flag']}"
                                        for r in flagged))


if __name__ == "__main__":
    main()
