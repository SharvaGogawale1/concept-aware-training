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
    python scripts/summarize_finetune.py --pool-seeds \
        outputs/llama-3.2-1b/seqcls_finetune.json,outputs/llama-3.2-1b/seqcls_finetune_replica.json \
        outputs/llama-3.2-1b/seqcls_finetune_ftseed{1,2}.json --ref iyer_ntp_syn

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


def pooled(groups, ref):
    """Mean +- sd over fine-tune seeds.  Each group is one seed's file(s), comma-separated
    (e.g. our arms + the replication).  Only arms and tasks present in every seed are used.
    The evaluation items are the same in every seed, so the difference against `ref` is
    averaged over seeds item by item and bootstrapped over items; the per-seed differences
    show whether the seeds agree."""
    datas = [merged(g.split(",")) for g in groups]
    first = datas[0]
    seqcls = any("eval" in r for t in TASKS for r in first.get(t, {}).values())
    acc = (lambda r: r["eval"]["accuracy"]) if seqcls else (lambda r: r["match_accuracy"])
    items = (lambda r: r["per_item_test"]) if seqcls else (lambda r: r["per_item_match"])
    arms = [a for a in dict.fromkeys(a for t in TASKS for a in first.get(t, {}))
            if all(a in d.get(t, {}) for d in datas for t in TASKS if t in first)]
    tasks = [t for t in TASKS if all(all(a in d.get(t, {}) for a in arms) for d in datas)]
    print(f"\n{len(groups)} fine-tune seeds pooled ({'classification head' if seqcls else 'generated answer'}), "
          f"{len(arms)} systems, {len(tasks)} tasks: test accuracy (%), mean +- sd over seeds")
    width = max(13, *(len(a) for a in arms))
    print("  " + "task".ljust(7) + "".join(a.rjust(width + 2) for a in arms))
    for t in tasks:
        cells = []
        for a in arms:
            v = 100 * np.array([acc(d[t][a]) for d in datas])
            cells.append(f"{v.mean():5.1f} +- {v.std(ddof=1):3.1f}")
        print("  " + t.ljust(7) + "".join(c.rjust(width + 2) for c in cells))
    if all(t in tasks for t in GLUE):
        cells = []
        for a in arms:
            v = 100 * np.array([np.mean([acc(d[t][a]) for t in GLUE]) for d in datas])
            cells.append(f"{v.mean():5.1f} +- {v.std(ddof=1):3.1f}")
        print("  " + "GLUE-7".ljust(7) + "".join(c.rjust(width + 2) for c in cells))
    if ref not in arms:
        print(f"  (no paired differences: {ref} is not in every seed)")
        return
    print(f"  difference vs {ref}, points: seed-averaged [95% CI over items] (per-seed differences)  * = CI excludes 0")
    for a in arms:
        if a == ref:
            continue
        parts = []
        for t in tasks:
            per_item = np.mean([np.asarray(items(d[t][a]), float) - np.asarray(items(d[t][ref]), float)
                                for d in datas], axis=0)
            per_seed = [100 * (np.mean(items(d[t][a])) - np.mean(items(d[t][ref]))) for d in datas]
            boots = per_item[np.random.RandomState(42).randint(0, len(per_item), (10_000, len(per_item)))].mean(1)
            lo, hi = np.percentile(boots, [2.5, 97.5]) * 100
            parts.append(f"{t} {100 * per_item.mean():+.1f} [{lo:+.1f},{hi:+.1f}]{'*' if lo > 0 or hi < 0 else ''}"
                         f" ({'/'.join(f'{x:+.1f}' for x in per_seed)})")
        print(f"    {a}: " + "; ".join(parts))


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
    p.add_argument("--pool-seeds", action="store_true",
                   help="each FILES argument is one fine-tune seed (comma-separate files to merge within a "
                        "seed): mean +- sd over seeds, and seed-averaged paired differences vs --ref")
    a = p.parse_args()
    if a.pool_seeds:
        pooled(a.files, a.ref)
        return
    shares = label_shares(TASKS) if a.baselines else None
    if a.merge:
        summarize(" + ".join(a.files), a.ref, shares, data=merged(a.files))
        return
    for f in a.files:
        summarize(f, a.ref, shares)


if __name__ == "__main__":
    main()
