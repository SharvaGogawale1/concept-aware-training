"""Additional CoInCo evaluation: the analysis fixed by sealed/EVALUATION_PLAN.md §4–5, run once.

    python scripts/round2/analyze_additional_coinco.py

Reads the three models' completion records (additional_coinco_eval.py), checks every output against
its SHA-256, its checkpoint and the 3,370 IDs, and writes results/round2/additional_coinco/analysis/:
cells.csv, comparisons.csv, readings.json, table.md and report.md.  Refuses to overwrite.

Comparisons (plan §4): D1 Alt-Mass - Concept loss, D2 Alt-Mass - whole-set, D3 Alt-Mass+Balance -
Alt-Mass, M1/M2 the main effects; on GAP (the only metric with a verdict), annotated-substitute NLL
and observed-word share; left and full scorers.  Bootstrap: analyze_round2.bootstrap, seeds then
sentences (a drawn sentence brings all its targets), 10,000 replicates, one draw shared by every
configuration and seed of a design.  Holm across the three models for each comparison, scorer and
metric; the family size is always 3, so a model whose comparison cannot be computed enters with p = 1.

Missing runs (plan §9 step 5): each comparison uses the seeds that all of its configurations have.
SAFEGUARD: a comparison without every matched seed, or (on GAP) without all 3,370 targets, is
"incomplete".  It keeps its estimate and interval but gets no verdict, so it is never confirmed or
adverse and counts as not confirmed in the across-model readings.  The safeguard is checked again on
the finished rows before anything is written.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import additional_coinco_eval as R
import analyze_round2 as A
import r2common as C

REPS = 10_000
ALPHA = 0.05
GAMMA = C.PRIMARY_GAMMA
REQUIRED_SEEDS = tuple(C.SEEDS)
SCORERS = R.MODES
# metric -> (label, display scale)
METRICS = {"gap": ("GAP", 100), "alternatives_nll": ("annotated-substitute NLL", 1),
           "observed_share": ("observed-word share", 100)}
VERDICT_METRIC = "gap"
# id -> (name, weights over configurations, existing-benchmark (family, contrast) in analysis_t2c)
COMPARISONS = {
    "D1": ("Alt-Mass − Concept loss", {"alt_mass": 1, "concept": -1}, ("F1", "mass_vs_concept")),
    "D2": ("Alt-Mass − whole-set", {"alt_mass": 1, A.WHOLESET: -1}, ("F3", "mass_vs_wholeset")),
    "D3": ("Alt-Mass+Balance − Alt-Mass", {"alt_mass_balance": 1, "alt_mass": -1}, ("F2", "mb_vs_mass")),
    "M1": ("Mass main effect", {"alt_mass": .5, "alt_mass_balance": .5, "concept": -.5, "alt_balance": -.5},
           ("F5", "mass_main")),
    "M2": ("Balance main effect", {"alt_balance": .5, "alt_mass_balance": .5, "concept": -.5, "alt_mass": -.5},
           ("F5", "balance_main")),
}
TABLE_ROWS = ("D1", "D2", "D3")
CONFIRMED, INCONCLUSIVE, ADVERSE = "confirmed", "inconclusive", "adverse"
INCOMPLETE, NOT_COMPUTED, DESCRIPTIVE = "incomplete", "not computed", "descriptive"
EXISTING = ("swords_test", "coinco_test")


def arm_key(config, seed):
    return (config, None if config == "concept" else GAMMA, seed)


def sentence_map(eval_file):
    with gzip.open(eval_file, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    return {t: rec["context_id"] for t, rec in data["targets"].items()}


def load_model(root, tag, ids, sentence_of, src=None):
    """(scorer, metric) -> {arm: {("", sentence, target): value}}, the record, provenance lines."""
    rec_file = Path(root) / tag / "COMPLETE.json"
    if not rec_file.is_file():
        sys.exit(f"no completion record {rec_file}: every model must be scored (or its runs recorded missing) first")
    rec = json.loads(rec_file.read_text())
    flats = {A.flat(*a[:3]): a for a in R.plan_arms()}
    if set(rec["outputs"]) & set(rec["missing"]) or set(rec["outputs"]) | set(rec["missing"]) != set(flats):
        sys.exit(f"{rec_file}: outputs and missing runs must cover the 15 checkpoints exactly once")
    data = defaultdict(dict)
    for name, out in sorted(rec["outputs"].items()):
        f = Path(root) / tag / "outputs" / f"{name}.json"
        if not f.is_file() or C.sha256(f) != out["sha256"]:
            sys.exit(f"{f} does not match the SHA-256 in its completion record")
        if not R.output_ok(f, out["checkpoint"], ids):
            sys.exit(f"{f} does not hold exactly the {len(ids)} targets for {out['checkpoint']}")
        objective, gamma, seed, source = flats[name]
        if src is not None and source == "october" and out["checkpoint"] != src.locked(tag)[A.october_label(objective, gamma, seed)]:
            sys.exit(f"{f}: not the locked October checkpoint")
        row = json.loads(f.read_text())[0]
        arm = (objective, gamma, seed)
        for scorer in SCORERS:
            per = row[scorer]["per_target"]
            item = lambda t: ("", sentence_of[t["target_id"]], t["target_id"])
            for metric in ("gap", "alternatives_nll"):
                vals = {item(t): t[metric] for t in per}
                A.check_aggregate(list(vals.values()), row[scorer][metric], f"{f} {scorer} {metric}")
                data[(scorer, metric)][arm] = vals
            data[(scorer, "observed_share")][arm] = {
                item(t): (math.exp(t["inclusive_nll"] - t["gold_nll"])
                          if t["inclusive_nll"] is not None and t["gold_nll"] is not None else None) for t in per}
    return data, rec


def verdict(row):
    """The plan §5 outcome for one model and comparison; the safeguard comes first."""
    if not row["computed"]:
        return NOT_COMPUTED
    if tuple(row["seeds"]) != REQUIRED_SEEDS or (row["metric"] == VERDICT_METRIC and row["n_items"] != row["n_expected"]):
        return INCOMPLETE
    if row["metric"] != VERDICT_METRIC:
        return DESCRIPTIVE
    if row["p_holm"] < ALPHA and row["estimate"] > 0:
        return CONFIRMED
    if row["p_holm"] < ALPHA and row["estimate"] < 0:
        return ADVERSE
    return INCONCLUSIVE


def check_safeguard(rows):
    """No confirmatory verdict on an incomplete comparison, whatever produced the rows."""
    for r in rows:
        if r["verdict"] in (CONFIRMED, ADVERSE) and not (
                r["computed"] and tuple(r["seeds"]) == REQUIRED_SEEDS and r["n_items"] == r["n_expected"]
                and r["metric"] == VERDICT_METRIC and r["p_holm"] < ALPHA):
            raise RuntimeError(f"safeguard: {r['id']} {r['model']} {r['scorer']} has verdict {r['verdict']} "
                               f"with seeds {r['seeds']} and {r['n_items']} of {r['n_expected']} items")


def cells(tag, data):
    out = []
    for (scorer, metric), per_arm in sorted(data.items()):
        for config in R.CONFIGS:
            seeds = [s for s in C.SEEDS if arm_key(config, s) in per_arm]
            if not seeds:
                continue
            X, n, strata, n_items = A.design(per_arm, GAMMA, seeds, (config,))
            per_seed = [float(X[0, i].sum() / n.sum()) for i in range(len(seeds))]
            out.append({"model": tag, "scorer": scorer, "metric": metric, "objective": config,
                        "configuration": C.DISPLAY[config], "mean": float(np.mean(per_seed)),
                        "sd_seeds": float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else "",
                        "per_seed": " ".join(f"{s}:{v:.6g}" for s, v in zip(seeds, per_seed)),
                        "n_seeds": len(seeds), "n_items": n_items})
    return out


def comparisons(tag, data, reps, n_expected):
    rows = []
    for (scorer, metric), per_arm in sorted(data.items()):
        groups = defaultdict(list)
        for cid, (name, weights, _) in COMPARISONS.items():
            seeds = tuple(s for s in C.SEEDS if all(arm_key(c, s) in per_arm for c in weights))
            base = {"id": cid, "comparison": name, "model": tag, "scorer": scorer, "metric": metric,
                    "seeds": seeds, "n_seeds": len(seeds), "n_expected": n_expected}
            if seeds:
                groups[seeds].append((cid, base))
            else:
                rows.append({**base, "computed": False, "estimate": None, "ci_low": None, "ci_high": None,
                             "p": None, "n_items": 0, "n_sentences": 0})
        for seeds, members in groups.items():
            order = tuple(c for c in R.CONFIGS if any(c in COMPARISONS[cid][1] for cid, _ in members))
            X, n, strata, n_items = A.design(per_arm, GAMMA, list(seeds), order)
            W = np.array([[COMPARISONS[cid][1].get(c, 0) for c in order] for cid, _ in members], dtype=float)
            D = np.einsum("kc,csu->ksu", W, X)
            est = A.point(D, n, strata)
            key = f"additional_coinco|{tag}|{scorer}|{metric}|seeds={'-'.join(map(str, seeds))}"
            hier, _ = A.bootstrap(D, n, strata, reps, key)
            for k, (_, base) in enumerate(members):
                lo, hi = np.percentile(hier[:, k], [2.5, 97.5])
                rows.append({**base, "computed": True, "estimate": float(est[k]), "ci_low": float(lo),
                             "ci_high": float(hi), "p": A.p_two_sided(hier[:, k]), "n_items": n_items,
                             "n_sentences": int(len(n))})
    return rows


def adjust(rows, models):
    """Holm across the three models per (comparison, scorer, metric); then the verdicts."""
    fam = defaultdict(dict)
    for r in rows:
        fam[(r["id"], r["scorer"], r["metric"])][r["model"]] = r
    for key, members in fam.items():
        if set(members) != set(models):
            raise RuntimeError(f"Holm family {key} lacks a model: {sorted(members)}")
        ordered = [members[m] for m in models]
        adj = A.holm([r["p"] if r["computed"] else 1.0 for r in ordered])
        for r, p in zip(ordered, adj):
            r["p_holm"], r["holm_m"] = (p if r["computed"] else None), len(models)
    for r in rows:
        r["verdict"] = verdict({**r, "p_holm": 1.0 if r["p_holm"] is None else r["p_holm"]})
    check_safeguard(rows)
    return rows


def readings(rows, models):
    out = {}
    for cid in COMPARISONS:
        by = {r["model"]: r["verdict"] for r in rows
              if r["id"] == cid and r["scorer"] == "left" and r["metric"] == VERDICT_METRIC}
        confirmed = [m for m in models if by[m] == CONFIRMED]
        out[cid] = {
            "comparison": COMPARISONS[cid][0],
            "confirmed": confirmed,
            "inconclusive": [m for m in models if by[m] == INCONCLUSIVE],
            "adverse": [m for m in models if by[m] == ADVERSE],
            "incomplete": [m for m in models if by[m] in (INCOMPLETE, NOT_COMPUTED)],
            "summary": ("confirmed in all three" if len(confirmed) == len(models) else
                        f"confirmed in {len(confirmed)} of {len(models)}" if confirmed else "confirmed in none"),
            "general_claim": len(confirmed) == len(models),
        }
    return out


def attach_existing(rows, effects_csv):
    if not Path(effects_csv).is_file():
        return f"existing-benchmark values not attached: {effects_csv} not found"
    idx = {}
    with open(effects_csv, newline="") as handle:
        for e in csv.DictReader(handle):
            idx[(e["family"], e["contrast"], e["model"], e["benchmark"], e["scorer"], e["metric"], e["gamma"])] = e
    for r in rows:
        family, contrast = COMPARISONS[r["id"]][2]
        for bench in EXISTING:
            e = idx.get((family, contrast, r["model"], bench, r["scorer"], r["metric"], f"{GAMMA:g}"))
            r[f"existing_{bench}"] = float(e["estimate"]) if e else None
            r[f"existing_{bench}_p_holm"] = float(e["p_holm"]) if e and e["p_holm"] not in ("", None) else None
    return f"existing-benchmark values from {effects_csv} (their own protocol §5 families; targets resampled)"


def fmt(v, scale, d=2, sign=True):
    return "" if v is None else f"{v * scale:{'+' if sign else ''}.{d}f}"


def compact_table(cell_rows, rows, models, missing):
    """The plan §8 main-text table: five configurations' left GAP, the key paired differences below."""
    names = {"llama-3.2-1b": "Llama-3.2-1B", "qwen3-1.7b-base": "Qwen3-1.7B", "falcon3-1b-base": "Falcon3-1B"}
    lines = ["| | " + " | ".join(names.get(m, m) for m in models) + " |", "|---" * (len(models) + 1) + "|"]
    cell = {(c["model"], c["objective"]): c for c in cell_rows if c["scorer"] == "left" and c["metric"] == "gap"}
    for config in R.CONFIGS:
        vals = []
        for m in models:
            c = cell.get((m, config))
            vals.append("—" if c is None else fmt(c["mean"], 100, sign=False)
                        + ("" if c["n_seeds"] == len(C.SEEDS) else f" ({c['n_seeds']} seeds)"))
        lines.append(f"| {C.DISPLAY[config]} | " + " | ".join(vals) + " |")
    lines.append("| *Paired differences, GAP points [95% interval]* |" + " |" * len(models))
    comp = {(r["id"], r["model"]): r for r in rows if r["scorer"] == "left" and r["metric"] == "gap"}
    for cid in TABLE_ROWS:
        vals = []
        for m in models:
            r = comp[(cid, m)]
            if not r["computed"]:
                vals.append("not computed ‡")
                continue
            mark = "†" if r["verdict"] in (CONFIRMED, ADVERSE) else ("‡" if r["verdict"] == INCOMPLETE else "")
            vals.append(f"{fmt(r['estimate'], 100)} [{fmt(r['ci_low'], 100)}, {fmt(r['ci_high'], 100)}]{mark}")
        lines.append(f"| {COMPARISONS[cid][0]} | " + " | ".join(vals) + " |")
    note = "† Holm-adjusted p < 0.05 across the three models within a row, all three seeds matched."
    if any("‡" in line for line in lines):
        note += " ‡ A run is missing: no verdict (plan §5)."
    if missing:
        note += " Missing runs: " + "; ".join(missing) + "."
    return "\n".join(lines) + "\n\n" + note + "\n"


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (" ".join(map(str, v)) if isinstance(v, tuple) else v) for k, v in r.items()})


def run(root, eval_file, ids_file, out_dir, reps, effects_csv, models=tuple(C.FAMILIES), src=None, plan=None):
    out_dir = Path(out_dir)
    if out_dir.exists():
        sys.exit(f"refusing to overwrite {out_dir}: the analysis runs once (plan §9 step 7)")
    ids = R.target_ids(ids_file)
    sentence_of = sentence_map(eval_file)
    if set(sentence_of) != ids:
        sys.exit(f"{eval_file} does not hold exactly the targets of {ids_file}")
    loaded = {tag: load_model(root, tag, ids, sentence_of, src) for tag in models}
    records = [rec for _, rec in loaded.values()]
    for field in ("plan_sha256", "pins", "correction"):
        if len({json.dumps(r[field], sort_keys=True) for r in records}) != 1:
            sys.exit(f"the completion records disagree on {field}")
    if plan is not None and C.sha256(plan) != records[0]["plan_sha256"]:
        sys.exit(f"{plan} has changed since the evaluation was scored")
    cell_rows, rows = [], []
    for tag, (data, _) in loaded.items():
        cell_rows += cells(tag, data)
        rows += comparisons(tag, data, reps, len(ids))
    adjust(rows, list(models))
    reading = readings(rows, list(models))
    existing_note = attach_existing(rows, effects_csv)
    missing = [f"{tag} {name}" for tag, (_, rec) in loaded.items() for name in sorted(rec["missing"])]
    table = compact_table(cell_rows, rows, list(models), missing)

    out_dir.mkdir(parents=True)
    write_csv(out_dir / "cells.csv", cell_rows)
    write_csv(out_dir / "comparisons.csv", rows)
    C.write_json(out_dir / "readings.json", reading)
    (out_dir / "table.md").write_text(table, encoding="utf-8")
    report = [
        "# Additional CoInCo evaluation: analysis", "",
        f"Written {C.now_utc()} at git {C.git_commit()} from {root}; {reps:,} bootstrap replicates; "
        f"{len(ids):,} targets in {len(set(sentence_of.values())):,} sentences.",
        f"Plan SHA-256 {records[0]['plan_sha256']}; correction: {records[0]['correction'] or 'none'}.",
        f"Missing runs: {', '.join(missing) if missing else 'none'}.", existing_note, "",
        "## Main table (plan §8)", "", table, "## Readings, left-scorer GAP (plan §5)", ""]
    for cid, r in reading.items():
        report.append(f"- **{cid} {r['comparison']}**: {r['summary']}; confirmed {r['confirmed'] or '—'}, "
                      f"inconclusive {r['inconclusive'] or '—'}, adverse {r['adverse'] or '—'}, "
                      f"incomplete (no verdict) {r['incomplete'] or '—'}.")
    report += ["", "## Every comparison", "",
               "| id | model | scorer | metric | estimate [95%] | p | Holm p | seeds | targets | sentences | verdict |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        scale = METRICS[r["metric"]][1]
        est = (f"{fmt(r['estimate'], scale, 3)} [{fmt(r['ci_low'], scale, 3)}, {fmt(r['ci_high'], scale, 3)}]"
               if r["computed"] else "—")
        p = "" if r["p"] is None else f"{r['p']:.4f}"
        p_holm = "" if r["p_holm"] is None else f"{r['p_holm']:.4f}"
        seeds = " ".join(map(str, r["seeds"]))
        report.append(f"| {r['id']} | {r['model']} | {r['scorer']} | {METRICS[r['metric']][0]} | {est} | {p} | "
                      f"{p_holm} | {seeds} | {r['n_items']} | {r['n_sentences']} | {r['verdict']} |")
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return reading, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(R.OUT_ROOT), help="the scoring root, or a correction_<note> root")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--out", default=None, help="default <root>/analysis; required with --reps other than 10,000")
    ap.add_argument("--existing", default=str(C.R2 / "analysis_t2c" / "effects.csv"))
    args = ap.parse_args()
    if args.reps != REPS and args.out is None:
        sys.exit(f"--reps {args.reps} differs from the plan's {REPS:,}; pass --out to write somewhere else")
    problems = R.data_problems()
    if problems:
        sys.exit("refusing: " + "; ".join(problems))
    out = Path(args.out) if args.out else Path(args.root) / "analysis"
    reading, _ = run(Path(args.root), R.EVAL_FILE, R.IDS, out, args.reps, args.existing,
                     src=A.Sources(C.OUTPUTS, C.R2), plan=R.PLAN)
    print(f"[{C.now_utc()}] wrote {out}")
    for cid, r in reading.items():
        print(f"  {cid} {r['comparison']}: {r['summary']}" + (f"; incomplete in {', '.join(r['incomplete'])}"
                                                              if r["incomplete"] else ""))


if __name__ == "__main__":
    main()
