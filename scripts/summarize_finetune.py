#!/usr/bin/env python3
"""Progress and accuracy tables for the downstream fine-tune runs, readable mid-run.

Reads the JSON that eval_iyer_finetune.py (generated answer) or eval_seqcls_finetune.py
(classification head) writes after every task x arm, so it shows how far a run has got
and what it has found so far.  Paired differences against --ref are computed here from
the stored per-item results, so they are available before the run's own end-of-run block.

    python scripts/summarize_finetune.py outputs/llama-3.2-1b/iyer_finetune.json
    python scripts/summarize_finetune.py outputs/*/seqcls_finetune.json --ref zhang_seed42
"""
import argparse, json
from pathlib import Path

import numpy as np

TASKS = ["mnli", "snli", "hate", "spam", "fake", "logic", "cola", "sst2", "mrpc", "qqp", "qnli", "rte"]
GLUE = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]


def paired(a, b, reps=10_000):
    x = np.asarray(b, float) - np.asarray(a, float)
    boots = x[np.random.RandomState(42).randint(0, len(x), (reps, len(x)))].mean(1)
    return x.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def summarize(path, ref):
    data = json.loads(Path(path).read_text())
    meta = data.get("_meta", {})
    arms = list(dict.fromkeys(a for t in TASKS for a in data.get(t, {})))
    if not arms:
        print(f"\n{path}: nothing finished yet"); return
    seqcls = "eval" in data[next(t for t in TASKS if data.get(t))][arms[0]]
    if seqcls:
        cell = lambda r: f"{100 * r['train']['accuracy']:5.1f}/{100 * r['eval']['accuracy']:5.1f}"
        test = lambda r: r["eval"]["accuracy"]
        items = lambda r: r["per_item_test"]
        legend = "train/test accuracy (%), classification head"
    else:
        cell = lambda r: (f"{100 * r['match_accuracy']:5.1f}/{100 * r['ranked_accuracy']:5.1f}"
                          + ("!" if r["unparseable_rate"] > 0.05 else " "))
        test = lambda r: r["match_accuracy"]
        items = lambda r: r["per_item_match"]
        legend = "match/ranked accuracy (%), generated answer; ! = over 5% unparseable"
    done = sum(len(data.get(t, {})) for t in TASKS)
    secs = [r["_seconds"] for t in TASKS for r in data.get(t, {}).values()]
    print(f"\n{path}\n  {meta.get('base_model')}: {done}/{len(TASKS) * len(arms)} task x arm runs finished "
          f"for {len(arms)} arms, {np.mean(secs) / 60:.1f} min each on average\n  {legend}")
    width = max(15, *(len(a) for a in arms))
    print("  " + "task".ljust(7) + "".join(a.rjust(width + 2) for a in arms))
    for t in TASKS:
        row = data.get(t, {})
        print("  " + t.ljust(7) + "".join((cell(row[a]) if a in row else "-").rjust(width + 2) for a in arms))
    glue = [np.mean([test(data[t][a]) for t in GLUE]) * 100 if all(a in data.get(t, {}) for t in GLUE) else None
            for a in arms]
    print("  " + "GLUE-7".ljust(7) + "".join(("-" if g is None else f"{g:5.1f}").rjust(width + 2) for g in glue))
    unparse = [r["unparseable_rate"] for t in TASKS for r in data.get(t, {}).values() if "unparseable_rate" in r]
    if unparse and max(unparse) > 0.05:
        bad = [(t, a, r["unparseable_rate"]) for t in TASKS for a, r in data.get(t, {}).items()
               if r.get("unparseable_rate", 0) > 0.05]
        print("  unparseable > 5%: " + ", ".join(f"{t}/{a} {u:.0%}" for t, a, u in bad))
    if ref not in arms:
        return
    print(f"  paired test difference vs {ref}, points [95% CI]  (* = interval excludes zero)")
    for a in arms:
        if a == ref:
            continue
        parts = []
        for t in TASKS:
            row = data.get(t, {})
            if a in row and ref in row:
                d, lo, hi = paired(items(row[ref]), items(row[a]))
                parts.append(f"{t} {100 * d:+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}]{'*' if lo > 0 or hi < 0 else ''}")
        print(f"    {a}: " + ("; ".join(parts) if parts else "no shared tasks yet"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--ref", default="zhang_seed42", help="arm the paired differences are taken against")
    a = p.parse_args()
    for f in a.files:
        summarize(f, a.ref)


if __name__ == "__main__":
    main()
