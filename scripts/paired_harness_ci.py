#!/usr/bin/env python3
"""Paired bootstrap over lm-eval-harness questions: is arm B better than arm A?

The harness table reports one accuracy per task, and the arms differ by tenths of a
point -- far inside one task's standard error (~1.4 pp on ARC-Challenge).  Every arm
answers the SAME questions, though, so the right test pairs them question by
question, as SWORDS is analysed.  Needs a run of scripts/run_harness_eval.py with
--log-samples.

    python scripts/paired_harness_ci.py --harness outputs/llama-3.2-1b/harness_verified.json \
        --pair verified_zhang:verified_hybrid --pair verified_zhang_seed123:verified_hybrid_seed123

Per task: accuracy of each arm, B - A, and a 95% interval from resampling
questions.  Per group (knowledge, commonsense): the mean of the task means, with
questions resampled independently inside each task.  Metric is acc_norm where the
task defines it (Zhang report length-normalised accuracy), acc otherwise.
"""
import argparse, json, random
from pathlib import Path

KNOWLEDGE = ["arc_challenge", "arc_easy", "openbookqa"]
COMMONSENSE = ["hellaswag", "piqa", "winogrande"]


def outcomes(raw_dir, task):
    """doc_id -> 0/1 for one arm on one task, from the newest samples file."""
    files = sorted(Path(raw_dir).rglob(f"samples_{task}_*.jsonl"), key=lambda q: q.stat().st_mtime)
    if not files:
        raise SystemExit(f"no samples for {task} under {raw_dir}; rerun the harness with --log-samples")
    out = {}
    for line in files[-1].read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("acc_norm", row.get("acc"))
        if value is None:
            raise SystemExit(f"{files[-1]}: a sample has neither acc_norm nor acc")
        out[row["doc_id"]] = float(value)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harness", required=True, help="the json written by run_harness_eval.py")
    p.add_argument("--pair", action="append", required=True, help="A:B, reports B - A")
    p.add_argument("--tasks", nargs="*", default=KNOWLEDGE + COMMONSENSE)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="json (default: beside --harness)")
    a = p.parse_args()

    results = json.loads(Path(a.harness).read_text())
    report = {}
    for pair in a.pair:
        left, right = pair.split(":")
        for arm in (left, right):
            if not results.get(arm, {}).get("_log_samples"):
                raise SystemExit(f"{arm} was not scored with --log-samples")
        per_task = {}
        for task in a.tasks:
            x = outcomes(results[left]["_raw_dir"], task)
            y = outcomes(results[right]["_raw_dir"], task)
            common = sorted(set(x) & set(y))
            if len(common) != len(x) or len(common) != len(y):
                raise SystemExit(f"{task}: the two arms scored different questions")
            per_task[task] = [(x[d], y[d]) for d in common]

        rng = random.Random(a.seed)
        # One set of question resamples per task, shared by every statistic below,
        # so task and group intervals come from the same bootstrap draws.
        draws = {t: [[rng.randrange(len(v)) for _ in v] for _ in range(a.samples)]
                 for t, v in per_task.items()}

        def delta(task, idx=None):
            rows = per_task[task] if idx is None else [per_task[task][i] for i in idx]
            return sum(b - a_ for a_, b in rows) / len(rows)

        def interval(stat):
            vals = sorted(stat(k) for k in range(a.samples))
            return vals[int(0.025 * a.samples)], vals[int(0.975 * a.samples) - 1]

        out = {}
        for task in a.tasks:
            rows = per_task[task]
            lo, hi = interval(lambda k, t=task: delta(t, draws[t][k]))
            out[task] = {"n": len(rows), left: sum(r[0] for r in rows) / len(rows),
                         right: sum(r[1] for r in rows) / len(rows),
                         "delta": delta(task), "ci95": [lo, hi]}
        for group, tasks in (("knowledge", KNOWLEDGE), ("commonsense", COMMONSENSE)):
            tasks = [t for t in tasks if t in per_task]
            if not tasks:
                continue
            point = sum(delta(t) for t in tasks) / len(tasks)
            lo, hi = interval(lambda k: sum(delta(t, draws[t][k]) for t in tasks) / len(tasks))
            out[f"{group}_mean"] = {"delta": point, "ci95": [lo, hi]}
        report[f"{right} - {left}"] = out

        print(f"\n{right} - {left}")
        for key, row in out.items():
            lo, hi = row["ci95"]
            sig = "*" if lo > 0 or hi < 0 else " "
            print(f"  {key:18s} {100 * row['delta']:+6.2f} pp  [{100 * lo:+6.2f}, {100 * hi:+6.2f}] {sig}"
                  + (f"  n={row['n']}" if "n" in row else ""))

    output = Path(a.output) if a.output else Path(a.harness).with_name(Path(a.harness).stem + "_paired.json")
    output.write_text(json.dumps(report, indent=2))
    print("\nwrote", output)


if __name__ == "__main__":
    main()
