"""Tier 2c on one GPU: train one model's whole-set control runs, retry a failed run once, then test-pass score.

    bash scripts/round2/r2tmux.sh r2wsllama scripts/round2/train_eval_2c.py --family llama-3.2-1b --gpu 1

Training goes through run_round2.py (its gates, ledger rows and checks unchanged).  A run that
fails or diverges is retried once with the same seed (ROUND2_PROTOCOL.md section 7); if it fails
again it stays failed.  The test pass then scores every finished checkpoint of the model that
has no output yet: evaluate_round2.py skips files that exist, and the harness keeps the arms
already in harness.json, so the Tier-1 results are only read.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import r2common as C
import run_round2

HERE = Path(__file__).resolve().parent


def to_train(family):
    return [r for r in run_round2.arms("2c", [family]) if r["action"].startswith("TRAIN")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", required=True, choices=sorted(C.FAMILIES))
    ap.add_argument("--gpu", required=True)
    args = ap.parse_args()
    train = [sys.executable, HERE / "run_round2.py", "train", "--tier", "2c", "--queue", "A",
             "--families", args.family, "--gpu", args.gpu, "--approved"]
    for attempt in (1, 2):
        if not to_train(args.family):
            break
        print(f"[{C.now_utc()}] training attempt {attempt}: {len(to_train(args.family))} run(s)", flush=True)
        code = subprocess.run(list(map(str, train))).returncode
        if code:
            sys.exit(f"run_round2.py train exited {code} (a gate refused, or it crashed); not evaluating")
    failed = to_train(args.family)
    if failed:
        print(f"[{C.now_utc()}] failed twice, reported as failed: " +
              ", ".join(f"seed {r['seed']}" for r in failed), flush=True)
    evaluate = [sys.executable, HERE / "evaluate_round2.py", "--family", args.family, "--gpu", args.gpu,
                "--pass", "test"]
    print(f"[{C.now_utc()}] test pass", flush=True)
    sys.exit(subprocess.run(list(map(str, evaluate))).returncode or (1 if failed else 0))


if __name__ == "__main__":
    main()
