"""Task 5 (existing evaluators, unchanged): score every finished round-2 checkpoint of one model.

    python scripts/round2/evaluate_round2.py --family llama-3.2-1b --gpu 1 --pass test

test pass, per checkpoint: SWORDS test, CoInCo-800 test, SemEval-07 (left + full scorers),
  C4 test NLL / word-level PPL, bm-semlex, STS (9 MTEB tasks); then the six-task harness.
dev pass, per checkpoint: SWORDS dev, CoInCo-800 dev, C4 validation NLL.
The October arms are not re-scored: Task 0 reproduced their stored numbers exactly, so the
round-2 tables read them from outputs/<model>/.  A test pass refuses to run until
ROUND2_PROTOCOL.md is committed and approved, and every ledger row it writes carries
added_after_first_test_read=true (the October test sets were read before these arms existed).
One output file per checkpoint and benchmark under results/round2/eval/<model>/, so an
interrupted pass resumes at the checkpoint it was on.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import r2common as C

EVAL_DIR = C.MAIN / "transformers/examples/pytorch/language-modeling"
LEXSUB = {
    "swords_test": C.MAIN / "data/swords/swords-v1.1_test.json.gz",
    "coinco_test": C.MAIN / "data/lexsub/coinco_test_sub800.json.gz",
    "semeval07_test": C.MAIN / "data/lexsub/semeval07_test.json.gz",
    "swords_dev": C.MAIN / "data/swords/swords-v1.1_dev.json.gz",
    "coinco_dev": C.MAIN / "data/lexsub/coinco_dev_sub800.json.gz",
}
STEPS = {"test": ("swords_test", "coinco_test", "semeval07_test", "c4_test", "bm_semlex", "sts"),
         "dev": ("swords_dev", "coinco_dev", "c4_val")}


def protocol_ok():
    f = C.MAIN / "ROUND2_PROTOCOL.md"
    text = f.read_text() if f.is_file() else ""
    tracked = subprocess.run(["git", "-C", str(C.MAIN), "ls-files", "--error-unmatch", f.name],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(C.MAIN), "diff", "--quiet", "HEAD", "--", f.name],
                           capture_output=True).returncode == 0
    approved = next((l for l in text.splitlines() if l.startswith("- Approved by:")), "")
    problems = [p for p, bad in (("not committed", not tracked), ("has uncommitted edits", not clean),
                                 ("still carries the TEMPLATE banner", "TEMPLATE, not yet approved" in text),
                                 ("names no approver", "AUTHORS" in approved or not approved)) if bad]
    return problems


def finished_runs(tag):
    out = []
    for marker in sorted((C.CHECKPOINTS / tag).glob("*/g*/seed_*/round2_run.json")):
        if json.loads(marker.read_text()).get("status") == "ok":
            out.append(marker.parent)
    return out


def flat(path):
    objective, gamma, seed = path.parts[-3], path.parts[-2], path.parts[-1]
    return f"{objective}_{gamma}_{seed}"


def env(gpu):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CONCEPT_DATA_ROOT": str(C.DATA),
            "CONCEPT_RESULTS_ROOT": str(C.R2 / "eval" / "_scratch"), "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
            "HF_HOME": os.environ.get("HF_HOME", str(C.WORK / "hf_cache"))}


def run(argv, cwd, gpu, log):
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(f"[{C.now_utc()}] + " + " ".join(map(str, argv)) + "\n")
        handle.flush()
        return subprocess.run(list(map(str, argv)), cwd=cwd, env=env(gpu), stdout=handle,
                              stderr=subprocess.STDOUT).returncode


def command(step, tag, ckpt, out, py):
    model = C.FAMILIES[tag]
    if step in LEXSUB:
        return [py, EVAL_DIR / "eval_swords.py", "--checkpoints", ckpt, "--tokenizer_path", model,
                "--base_model", model, "--swords_json", LEXSUB[step], "--results_json", out,
                "--modes", "left", "full"], C.MAIN
    if step in ("c4_test", "c4_val"):
        split = "test" if step == "c4_test" else "val"
        return [py, "eval/eval_perplexity_explicit.py", "--checkpoints", ckpt, "--base-model", model,
                "--test-jsonl", C.leaf(tag) / f"synonyms_{split}.jsonl", "--output", out], C.EXT
    if step == "bm_semlex":
        return [py, EVAL_DIR / "eval_bm_semlex.py", "--checkpoints", ckpt, "--tokenizer_path", model,
                "--base_model", model, "--data", C.MAIN / "data/bm_semlex/curated_200.tsv",
                "--results_json", out], C.MAIN
    if step == "sts":
        return [py, "eval/eval_mteb.py", "--base-model", model, "--dataset", "c4",
                "--dataset-type", "embedding", "--tasks", "sts", "--run-label", flat(Path(ckpt)),
                "--csv-output", out, "--mteb-output-root", out.parent / "mteb_raw",
                "--adapter-path", ckpt], C.EXT
    raise ValueError(step)


def done(step, out):
    if not out.is_file():
        return False
    if step == "sts":
        return sum(1 for line in out.read_text().splitlines() if line.strip()) >= 10   # header + 9 tasks
    try:
        return bool(json.loads(out.read_text()))
    except json.JSONDecodeError:
        return False


def ledger(tag, ckpt, step, pass_, status, started, minutes, gpu, note=""):
    p = Path(ckpt)
    C.append_ledger({"run_id": C.run_id(tag, p.parts[-3], p.parts[-2].lstrip("g"), p.parts[-1].split("_")[1]),
                     "pass": f"{pass_}:{step}", "model_tag": tag, "objective": p.parts[-3],
                     "gamma": p.parts[-2].lstrip("g"), "seed": p.parts[-1].split("_")[1],
                     "status": status, "added_after_first_test_read": str(pass_ == "test").lower(),
                     "checkpoint": str(ckpt), "started_utc": started, "ended_utc": C.now_utc(),
                     "wall_minutes": f"{minutes:.1f}", "gpu": str(gpu), "git_commit": C.git_commit(),
                     "patch_sha256": C.sha256(C.PATCH), "note": note})


def harness(tag, ckpts, gpu, harness_py, log):
    root = C.R2 / "eval" / tag
    screen = root / "harness_manifest"
    screen.mkdir(parents=True, exist_ok=True)
    labels = {flat(p): str(p) for p in ckpts}
    (screen / "manifest.json").write_text(json.dumps(labels, indent=2))
    print(f"[{C.now_utc()}] harness: {len(labels)} checkpoints + untouched", flush=True)
    started, t0 = C.now_utc(), time.time()
    code = run([harness_py, C.MAIN / "scripts/run_harness_eval.py", "--screen", screen,
                "--base-model", C.FAMILIES[tag], "--out", root / "harness.json", "--log-samples",
                "--arms", *labels], C.BASE, gpu, log)
    C.append_ledger({"run_id": f"{tag}/harness", "pass": "test:harness", "model_tag": tag,
                     "status": "ok" if code == 0 else "failed", "added_after_first_test_read": "true",
                     "started_utc": started, "ended_utc": C.now_utc(),
                     "wall_minutes": f"{(time.time() - t0) / 60:.1f}", "gpu": str(gpu),
                     "git_commit": C.git_commit(), "note": f"{len(labels)} arms; exit {code}"})
    print(f"  harness {'ok' if code == 0 else 'FAILED (see ' + str(log) + ')'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", required=True, choices=sorted(C.FAMILIES))
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--pass", dest="pass_", required=True, choices=sorted(STEPS))
    ap.add_argument("--steps", default=None, help="comma-separated subset of this pass's steps")
    ap.add_argument("--no-harness", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--harness-python", default=str(Path.home() / "miniconda3/envs/harness/bin/python"))
    args = ap.parse_args()
    if args.pass_ == "test":
        problems = protocol_ok()
        if problems:
            sys.exit("refusing the test pass: ROUND2_PROTOCOL.md " + "; ".join(problems))
    tag = args.family
    ckpts = finished_runs(tag)
    if not ckpts:
        sys.exit(f"no finished round-2 checkpoints for {tag}")
    steps = args.steps.split(",") if args.steps else list(STEPS[args.pass_])
    root = C.R2 / "eval" / tag
    log = root / f"{args.pass_}_evaluators.log"
    root.mkdir(parents=True, exist_ok=True)
    print(f"{tag}: {len(ckpts)} checkpoints, {args.pass_} pass, steps {', '.join(steps)}; "
          f"evaluator output in {log}", flush=True)
    failures = 0
    for step in [s for s in steps if s != "harness"]:
        for ckpt in ckpts:
            suffix = ".csv" if step == "sts" else ".json"
            out = root / step / (flat(ckpt) + suffix)
            if done(step, out):
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            argv, cwd = command(step, tag, ckpt, out, args.python)
            started, t0 = C.now_utc(), time.time()
            code = run(argv, cwd, args.gpu, log)
            minutes = (time.time() - t0) / 60
            status = "ok" if code == 0 and done(step, out) else "failed"
            failures += status != "ok"
            ledger(tag, ckpt, step, args.pass_, status, started, minutes, args.gpu,
                   "" if status == "ok" else f"exit {code}; see {log.name}")
            print(f"  {status:6s} {step:15s} {flat(ckpt):34s} {minutes:5.1f} min", flush=True)
    if args.pass_ == "test" and not args.no_harness and (not args.steps or "harness" in steps):
        harness(tag, ckpts, args.gpu, args.harness_python, root / "harness.log")
    print(f"[{C.now_utc()}] {tag} {args.pass_} pass done; {failures} failed step(s)", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
