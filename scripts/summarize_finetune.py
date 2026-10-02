#!/usr/bin/env python3
"""Progress and accuracy tables for the downstream fine-tune runs, readable mid-run.

Reads the JSON that eval_iyer_finetune.py (generated answer) or eval_seqcls_finetune.py
(classification head) writes after every task x arm, so it shows how far a run has got
and what it has found so far.  Paired differences against --ref are computed here from
the stored per-item results, so they are available before the run's own end-of-run block.

    python scripts/summarize_finetune.py outputs/llama-3.2-1b/iyer_finetune.json
    python scripts/summarize_finetune.py outputs/*/seqcls_finetune.json --ref zhang_seed42
    python scripts/summarize_finetune.py outputs/*/iyer_finetune.json --baselines   # + label shares
    python scripts/summarize_finetune.py outputs/llama-3.2-1b/seqcls_finetune{,_replica}.json --merge

--baselines loads each task's evaluation subset (a minute or two, from the HF cache) and
adds the majority-class row.  A cell whose accuracy equals one label's share of the
evaluation set is marked "=": the fine-tune most likely answered that label for every
item (for the classification head this is read off the stored predictions instead, and
marked when one label takes over 95% of them).  Such a cell cannot rank the systems.
"""
import argparse, json
from pathlib import Path

import numpy as np

TASKS = ["emo", "mnli", "snli", "hate", "spam", "fake", "logic", "cola", "sst2", "mrpc", "qqp", "qnli", "rte"]
GLUE = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]


def paired(a, b, reps=10_000):
    x = np.asarray(b, float) - np.asarray(a, float)
    boots = x[np.random.RandomState(42).randint(0, len(x), (reps, len(x)))].mean(1)
    return x.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def label_shares(tasks):
    """task -> share of each label in the evaluation subset the fine-tunes score."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_iyer_finetune as iyer
    out = {}
    for t in tasks:
        ys = np.array(iyer.load(t)["eval"][1])
        out[t] = np.bincount(ys) / len(ys)
    return out


def merged(paths):
    """One result dict from several files of one protocol and base model, e.g. our arms and
    the Iyer-replication arms, which run in separate processes and so write separate files.
    Refuses files whose evaluation data differ (per-task fingerprints), since the paired
    differences need identical items."""
    out, prints = {"_meta": {}}, {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        meta = data.get("_meta", {})
        base = out["_meta"].setdefault("base_model", meta.get("base_model"))
        if meta.get("base_model") != base:
            raise SystemExit(f"{path}: base model {meta.get('base_model')} differs from {base}")
        for t, fp in meta.get("fingerprints", {}).items():
            if prints.setdefault(t, fp) != fp:
                raise SystemExit(f"{path}: {t} was scored on different data; cannot merge")
        for t in TASKS:
            for arm, r in data.get(t, {}).items():
                out.setdefault(t, {}).setdefault(arm, r)
    return out


def summarize(path, ref, shares=None, data=None):
    data = data if data is not None else json.loads(Path(path).read_text())
    meta = data.get("_meta", {})
    arms = list(dict.fromkeys(a for t in TASKS for a in data.get(t, {})))
    if not arms:
        print(f"\n{path}: nothing finished yet"); return
    seqcls = "eval" in data[next(t for t in TASKS if data.get(t))][arms[0]]
    def one_label(t, r):
        if seqcls:
            share = np.array(r["eval"]["pred_label_share"], float)
            return share.max() > 0.95 * share.sum()
        if shares is None or t not in shares:
            return False
        return bool(np.isclose(r["match_accuracy"], shares[t], atol=5e-4).any())
    if seqcls:
        cell = lambda t, r: (f"{100 * r['train']['accuracy']:5.1f}/{100 * r['eval']['accuracy']:5.1f}"
                             + ("=" if one_label(t, r) else " "))
        test = lambda r: r["eval"]["accuracy"]
        items = lambda r: r["per_item_test"]
        legend = "train/test accuracy (%), classification head"
    else:
        cell = lambda t, r: (f"{100 * r['match_accuracy']:5.1f}/{100 * r['ranked_accuracy']:5.1f}"
                             + ("!" if r["unparseable_rate"] > 0.05 else "=" if one_label(t, r) else " "))
        test = lambda r: r["match_accuracy"]
        items = lambda r: r["per_item_match"]
        legend = "match/ranked accuracy (%), generated answer; ! = over 5% unparseable"
    legend += "; = one label for (nearly) every item" if seqcls or shares else ""
    done = sum(len(data.get(t, {})) for t in TASKS)
    secs = [r["_seconds"] for t in TASKS for r in data.get(t, {}).values()]
    print(f"\n{path}\n  {meta.get('base_model')}: {done}/{len(TASKS) * len(arms)} task x arm runs finished "
          f"for {len(arms)} arms, {np.mean(secs) / 60:.1f} min each on average\n  {legend}")
    width = max(15, *(len(a) for a in arms))
    print("  " + "task".ljust(7) + "".join(a.rjust(width + 2) for a in arms))
    for t in TASKS:
        row = data.get(t, {})
        majority = "" if shares is None or t not in shares else f"   majority {100 * shares[t].max():5.1f}"
        print("  " + t.ljust(7) + "".join((cell(t, row[a]) if a in row else "-").rjust(width + 2) for a in arms)
              + majority)
    glue = [np.mean([test(data[t][a]) for t in GLUE]) * 100 if all(a in data.get(t, {}) for t in GLUE) else None
            for a in arms]
    glue_majority = "" if shares is None or not all(t in shares for t in GLUE) else \
        f"   majority {100 * np.mean([shares[t].max() for t in GLUE]):5.1f}"
    print("  " + "GLUE-7".ljust(7) + "".join(("-" if g is None else f"{g:5.1f}").rjust(width + 2) for g in glue)
          + glue_majority)
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
    p.add_argument("--baselines", action="store_true",
                   help="load the evaluation subsets: majority-class row, and flag one-label answers")
    p.add_argument("--merge", action="store_true",
                   help="one table from all FILES (same protocol and base model), e.g. ours + the replication")
    a = p.parse_args()
    shares = label_shares(TASKS) if a.baselines else None
    if a.merge:
        summarize(" + ".join(a.files), a.ref, shares, data=merged(a.files))
        return
    for f in a.files:
        summarize(f, a.ref, shares)


if __name__ == "__main__":
    main()
