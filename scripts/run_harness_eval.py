#!/usr/bin/env python3
"""Zhang's six lm-eval-harness benchmarks, on adapters that already exist.

Zhang et al. §4.3 evaluate ARC-Challenge, ARC-Easy and OpenBookQA (knowledge
reasoning) and HellaSwag, PIQA and WinoGrande (commonsense) through the Language
Model Evaluation Harness, at the same training scale we use, and report that
concept models beat every baseline on the knowledge group and tie on commonsense.
So this is a live comparison, not a formality -- and it is the only downstream
evidence available without training anything.

Deliberately NOT a notebook cell: it is eval-only, so it must not be able to take
down a six-hour training run that has already finished its expensive part.

    pip install lm-eval
    python scripts/run_harness_eval.py \
        --screen outputs/llama-3.2-1b/task15b_screen_2026-09-21 \
        --base-model meta-llama/Llama-3.2-1B \
        --out outputs/llama-3.2-1b/harness.json
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

KNOWLEDGE = ["arc_challenge", "arc_easy", "openbookqa"]
COMMONSENSE = ["hellaswag", "piqa", "winogrande"]
# The arms the paper compares. Anything absent from the manifest is skipped with a
# warning rather than silently dropped -- a missing baseline changes the claim.
DEFAULT_ARMS = ["ntp_seed42", "zhang_seed42", "zhang_lambda1.0_seed42",
                "randomized_seed42", "alternative_uniform_seed42",
                "zhang_plus_uniform_g0.25_seed42", "calibrated_seed42"]

p = argparse.ArgumentParser()
p.add_argument("--screen", required=True, help="dir holding manifest.json")
p.add_argument("--base-model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
p.add_argument("--tasks", nargs="*", default=KNOWLEDGE + COMMONSENSE)
p.add_argument("--batch-size", default="auto")
p.add_argument("--limit", type=int, default=None, help="smoke test: items per task")
p.add_argument("--dtype", default="bfloat16")
a = p.parse_args()

manifest = json.loads((Path(a.screen) / "manifest.json").read_text())
by_label = {k.replace(" ", "_"): v for k, v in manifest.items()}
out_path = Path(a.out)
results = json.loads(out_path.read_text()) if out_path.is_file() else {}

# Pretrained first: it is the reference row and needs no adapter.
todo = [("pretrained", None)] + [(arm, by_label[arm]) for arm in a.arms if arm in by_label]
for arm in a.arms:
    if arm not in by_label:
        print(f"  ! {arm} is not in the manifest; the table will lack it", file=sys.stderr)

for label, adapter in todo:
    if label in results and not a.limit:
        print(f"resume: {label} already scored, skipping")
        continue
    model_args = f"pretrained={a.base_model},dtype={a.dtype}"
    if adapter:
        model_args += f",peft={adapter}"
    argv = [sys.executable, "-m", "lm_eval", "--model", "hf",
            "--model_args", model_args, "--tasks", ",".join(a.tasks),
            "--batch_size", str(a.batch_size), "--output_path", str(out_path.parent / f"harness_raw_{label}")]
    if a.limit:
        argv += ["--limit", str(a.limit)]
    print("+", " ".join(argv), flush=True)
    started = time.time()
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode:
        print(proc.stdout[-3000:], proc.stderr[-3000:], sep="\n", file=sys.stderr)
        raise SystemExit(f"lm_eval failed for {label}")
    # lm_eval writes a timestamped json under output_path; take the newest.
    raw = sorted((out_path.parent / f"harness_raw_{label}").rglob("*.json"),
                 key=lambda q: q.stat().st_mtime)
    scores = json.loads(raw[-1].read_text())["results"]
    # acc_norm where the task defines it (Zhang report length-normalized accuracy);
    # WinoGrande has none, where normalized and raw accuracy coincide anyway.
    results[label] = {task: {k: v for k, v in vals.items() if k.startswith(("acc,", "acc_norm,"))}
                      for task, vals in scores.items()}
    results[label]["_seconds"] = round(time.time() - started, 1)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  {label}: {results[label]['_seconds']}s -> {out_path}")

def mean(label, tasks):
    vals = []
    for task in tasks:
        row = results.get(label, {}).get(task, {})
        pick = next((v for k, v in row.items() if k.startswith("acc_norm,")), None)
        pick = pick if pick is not None else next((v for k, v in row.items() if k.startswith("acc,")), None)
        if pick is not None:
            vals.append(pick)
    return sum(vals) / len(vals) if vals else float("nan")

print(f"\n{'arm':34s} {'knowledge':>10s} {'commonsense':>12s}")
for label, _ in todo:
    if label in results:
        print(f"{label[:34]:34s} {mean(label, KNOWLEDGE):10.4f} {mean(label, COMMONSENSE):12.4f}")
