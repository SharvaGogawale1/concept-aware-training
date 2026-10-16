"""Task 0.5-0.6: reproduce stored numbers from existing checkpoints, or stop.

Re-scores, with the evaluators exactly as they are, and compares to the October files:
  SWORDS test  GAP and accepted-alternative NLL (left scorer)   Concept loss + balanced g=.125, seed 42, all 3 models
  CoInCo test  GAP (left)                                       same two checkpoints, Llama
  C4 test      global NLL                                       Concept loss seed 42, Llama
Tolerance: |delta GAP| <= 0.02 points (0.0002 raw), |delta NLL| <= 0.005 nats.
Writes results/round2/repro/ and exits non-zero if anything fails; nothing new may run then.

    python scripts/round2/reproduce_gate.py --gpu 0
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

GAP_TOL = 0.0002          # 0.02 GAP points
NLL_TOL = 0.005
EVAL_DIR = C.MAIN / "transformers/examples/pytorch/language-modeling"
REPRO = C.R2 / "repro"
ARMS = {"concept": "zhang_seed42", "alt_mass_balance_g0.125": "zhang_plus_uniform_g0.125_seed42"}


def stored(tag, bench):
    path = C.OUTPUTS / tag / "task15b_screen_test" / f"{bench}.json"
    rows = json.loads(path.read_text())
    return path, {r["checkpoint"]: r for r in rows}


def locked_arms(tag):
    marker = C.OUTPUTS / tag / "task15b_screen_test" / "swords_test_locked.json"
    return json.loads(marker.read_text())["arms"]


def run(argv, cwd, gpu, log):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUTF8": "1",
           "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "HF_HOME": os.environ.get("HF_HOME", str(C.WORK / "hf_cache"))}
    start = time.time()
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("+ " + " ".join(map(str, argv)) + "\n")
        handle.flush()
        code = subprocess.run(list(map(str, argv)), cwd=cwd, env=env, stdout=handle,
                              stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f"failed ({code}): {' '.join(map(str, argv))}; see {log}")
    return (time.time() - start) / 60


def check(rows, label, new, old, field, tol, unit):
    a, b = new.get(field), old.get(field)
    ok = a is not None and b is not None and abs(a - b) <= tol
    rows.append({"check": label, "field": field, "stored": b, "reproduced": a,
                 "delta": None if a is None or b is None else a - b, "tolerance": tol,
                 "unit": unit, "pass": ok})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--families", default=",".join(C.FAMILIES))
    ap.add_argument("--coinco-family", default="llama-3.2-1b")
    ap.add_argument("--c4-family", default="llama-3.2-1b")
    args = ap.parse_args()
    REPRO.mkdir(parents=True, exist_ok=True)
    log = REPRO / "repro.log"
    checks, minutes = [], 0.0
    started = C.now_utc()
    for tag in args.families.split(","):
        model = C.FAMILIES[tag]
        arms = locked_arms(tag)
        ckpts = [arms[ARMS[k]] for k in ARMS]
        benches = ["swords_test"] + (["coinco_test"] if tag == args.coinco_family else [])
        sources = {"swords_test": C.MAIN / "data/swords/swords-v1.1_test.json.gz",
                   "coinco_test": C.MAIN / "data/lexsub/coinco_test_sub800.json.gz"}
        for bench in benches:
            out = REPRO / tag / f"{bench}.json"
            if not out.is_file():
                print(f"[{C.now_utc()}] scoring {bench} for {tag}: {len(ckpts)} checkpoints "
                      f"(progress in {log})", flush=True)
                minutes += run([args.python, EVAL_DIR / "eval_swords.py", "--checkpoints", *ckpts,
                                "--tokenizer_path", model, "--base_model", model,
                                "--swords_json", sources[bench], "--results_json", out,
                                "--modes", "left", "full"], C.MAIN, args.gpu, log)
            new = {r["checkpoint"]: r for r in json.loads(out.read_text())}
            _, old = stored(tag, bench)
            for arm, label in ARMS.items():
                ck = arms[label]
                for mode in ("left", "full"):
                    check(checks, f"{tag} {bench} {arm} {mode}", new[ck][mode], old[ck][mode],
                          "gap", GAP_TOL, "raw (x100 = points)")
                    if bench == "swords_test":
                        check(checks, f"{tag} {bench} {arm} {mode}", new[ck][mode], old[ck][mode],
                              "alternatives_nll", NLL_TOL, "nats")
        if tag == args.c4_family:
            out = REPRO / tag / "perplexity.json"
            ck = arms[ARMS["concept"]]
            if not out.is_file():
                print(f"[{C.now_utc()}] C4 test NLL for {tag}", flush=True)
                minutes += run([args.python, "eval/eval_perplexity_explicit.py", "--checkpoints", ck,
                                "--base-model", model, "--test-jsonl", C.leaf(tag) / "synonyms_test.jsonl",
                                "--output", out], C.EXT, args.gpu, log)
            new = {r["checkpoint"]: r for r in json.loads(out.read_text())}
            old = {}
            for d in ("task15b_screen_test", "task15b_screen_shared_gamma", "task15b_screen_confirm"):
                f = C.OUTPUTS / tag / d / "perplexity.json"
                if f.is_file():
                    old.update({r["checkpoint"]: r for r in json.loads(f.read_text())
                                if r["checkpoint"] not in old})
            check(checks, f"{tag} c4_test concept", new[ck]["global"], old[ck]["global"],
                  "mean_nll", NLL_TOL, "nats")
    passed = all(c["pass"] for c in checks)
    report = {"passed": passed, "checks": checks, "started_utc": started, "ended_utc": C.now_utc(),
              "gpu_minutes": round(minutes, 1), "git_commit": C.git_commit()}
    C.write_json(REPRO / "reproduction_report.json", report)
    C.append_ledger({"run_id": "task0/reproduction_gate", "pass": "repro", "status":
                     "ok" if passed else "failed", "started_utc": started, "ended_utc": C.now_utc(),
                     "wall_minutes": f"{minutes:.1f}", "gpu": args.gpu, "git_commit": C.git_commit(),
                     "note": f"{sum(c['pass'] for c in checks)}/{len(checks)} checks within tolerance"})
    for c in checks:
        d = "n/a" if c["delta"] is None else f"{c['delta']:+.6f}"
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['check']:48s} {c['field']:17s} "
              f"stored={c['stored']}  new={c['reproduced']}  delta={d}")
    print("\nREPRODUCTION", "PASSED" if passed else "FAILED -- stop; run nothing new")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
