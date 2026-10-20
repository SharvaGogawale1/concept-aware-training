"""C4 training-distribution diagnostic (ROUND2_PROTOCOL.md §0, §5): concept-set metrics on held-out C4.

    bash scripts/round2/r2tmux.sh r2c4llama scripts/round2/c4_diagnostic.py --family llama-3.2-1b --gpu 1

Scores the four 2x2 cells at both gammas and Concept loss + whole-set, seeds 42/123/2024, with
eval/eval_concept_sets.py on the model's held-out C4 test file, one output per checkpoint:
results/round2/eval/<model>/concept_sets/<arm>.json.  Per slot that holds M(C) (set_mass),
-log M(A) (alternative_nll), p(y) (observed_target_nll), |A| and KL(u_A || q_A).
The October arms are re-scored, not read: their stored files predate the KL field.  Every field
those files already hold must reproduce, which is checked slot by slot and written to
reproduction.json next to the outputs.  Evaluation only; 4-bit base, as in October.
An interrupted run resumes at the first checkpoint without an output.
Exit 1 if a checkpoint is unavailable or fails, or a stored field does not reproduce.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import analyze_round2 as A
import evaluate_round2 as E
import r2common as C

OLD_FIELDS = ("set_mass", "set_nll", "observed_target_nll", "alternative_nll",
              "candidate_entropy", "minimum_candidate_probability")
LABEL_FIELDS = ("hierarchy_coverage", "pos", "frequency_bucket", "set_size_bucket", "wordnet_depth")
TOL = 1e-6     # float32 arithmetic; the C4 NLL reproduced with delta 0.0 in the Task 0.5 gate


def arms(tag, src):
    """(objective, gamma, seed, source, checkpoint or None) for every arm the diagnostic covers."""
    out = []
    for objective, gamma, seed, source in A.needed_arms() + A.wholeset_arms():
        if source == "october":
            ckpt = src.locked(tag).get(A.october_label(objective, gamma, seed))
        else:
            d = C.checkpoint_dir(tag, objective, gamma, seed)
            marker = d / "round2_run.json"
            ok = marker.is_file() and json.loads(marker.read_text()).get("status") == "ok"
            ckpt = str(d) if ok else None
        out.append((objective, gamma, seed, source, ckpt))
    return out


def scored(out, ckpt):
    try:
        rows = json.loads(out.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return len(rows) == 1 and rows[0]["checkpoint"] == ckpt


def compare(new, old):
    """Every field of the October record against the re-scored one, slot by slot."""
    problems, worst, exact = [], 0.0, 0
    a, b = new["per_slot"], old["per_slot"]
    if len(a) != len(b):
        return {"n_slots": len(a), "n_slots_stored": len(b), "problems": ["slot counts differ"]}
    for x, y in zip(a, b):
        where = f"row {y['row_id']} slot {y['slot_id']}"
        if (x["row_id"], x["slot_id"]) != (y["row_id"], y["slot_id"]):
            problems.append(f"{where}: slots out of order")
            break
        same = True
        for f in OLD_FIELDS:
            u, v = x[f], y[f]
            if (u is None) != (v is None):
                problems.append(f"{where} {f}: {u} vs stored {v}")
                same = False
            elif u is not None:
                worst, same = max(worst, abs(u - v)), same and u == v
                if abs(u - v) > TOL:
                    problems.append(f"{where} {f}: {u} vs stored {v}")
        for f in LABEL_FIELDS:
            if x.get(f) != y.get(f):
                problems.append(f"{where} {f}: {x.get(f)} vs stored {y.get(f)}")
                same = False
        exact += same
    for f in OLD_FIELDS:
        if (new[f] is None) != (old[f] is None) or (new[f] is not None and abs(new[f] - old[f]) > TOL):
            problems.append(f"aggregate {f}: {new[f]} vs stored {old[f]}")
    return {"n_slots": len(a), "n_exact_slots": exact, "max_abs_delta": worst,
            "problems": problems[:20], "n_problems": len(problems)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", required=True, choices=sorted(C.FAMILIES))
    ap.add_argument("--gpu", required=True)
    args = ap.parse_args()
    tag, model = args.family, C.FAMILIES[args.family]
    problems = E.protocol_ok()
    if problems:
        sys.exit("refusing: ROUND2_PROTOCOL.md " + "; ".join(problems))
    applied = C.EXT / ".round2_patch_sha256"
    evaluator = C.EXT / "eval" / "eval_concept_sets.py"
    if (not applied.is_file() or applied.read_text().strip() != C.sha256(C.PATCH)
            or "alternative_balance_kl" not in evaluator.read_text()):
        sys.exit(f"refusing: {C.EXT} does not carry this repo's patch (run prepare_ext.sh)")
    test_file = C.leaf(tag) / "synonyms_test.jsonl"
    root = C.R2 / "eval" / tag / "concept_sets"
    root.mkdir(parents=True, exist_ok=True)
    log = root / "concept_sets.log"
    src = A.Sources(C.OUTPUTS, C.R2)
    todo = arms(tag, src)
    failed = [f"{A.flat(*a[:3])}: no finished checkpoint" for a in todo if a[4] is None]
    print(f"[{C.now_utc()}] {tag}: {len(todo)} arms, {len(failed)} without a checkpoint (progress in {log})",
          flush=True)
    for objective, gamma, seed, source, ckpt in todo:
        out = root / f"{A.flat(objective, gamma, seed)}.json"
        if ckpt is None or scored(out, ckpt):
            continue
        started, t0 = C.now_utc(), time.time()
        code = E.run([sys.executable, "eval/eval_concept_sets.py", "--checkpoints", ckpt, "--base-model", model,
                      "--test-jsonl", test_file, "--output", out], C.EXT, args.gpu, log)
        ok = code == 0 and scored(out, ckpt)
        minutes = (time.time() - t0) / 60
        C.append_ledger({"run_id": C.run_id(tag, objective, gamma, seed), "pass": "diagnostic:c4_concept_sets",
                         "model_tag": tag, "objective": objective,
                         "gamma": "" if gamma is None else f"{gamma:g}", "seed": str(seed),
                         "status": "ok" if ok else "failed", "added_after_first_test_read": "true",
                         "checkpoint": ckpt, "started_utc": started, "ended_utc": C.now_utc(),
                         "wall_minutes": f"{minutes:.1f}", "gpu": str(args.gpu), "git_commit": C.git_commit(),
                         "patch_sha256": C.sha256(C.PATCH), "note": f"trained in {source}; exit {code}"})
        print(f"[{C.now_utc()}] {'ok' if ok else 'FAILED'}  {out.name}  {minutes:.1f} min", flush=True)
        if not ok:
            failed.append(f"{out.name}: evaluator exit {code}")

    checks = []
    for objective, gamma, seed, source, ckpt in todo:
        out = root / f"{A.flat(objective, gamma, seed)}.json"
        if source != "october" or not scored(out, ckpt):
            continue
        new = json.loads(out.read_text())[0]
        stored = []
        for d in A.OCT_DIRS:
            f = C.OUTPUTS / tag / d / "concept_sets.json"
            if f.is_file() and ckpt in src.by_checkpoint(f):
                stored.append((f, src.by_checkpoint(f)[ckpt]))
        if not stored:
            checks.append({"arm": out.stem, "pass": False, "problems": ["no October concept_sets.json scores it"]})
        for f, old in stored:
            result = compare(new, old)
            checks.append({"arm": out.stem, "stored_file": str(f), **result, "pass": not result["problems"]})
    october = sum(1 for a in todo if a[3] == "october")
    passed = len({c["arm"] for c in checks if c["pass"]}) == october and all(c["pass"] for c in checks)
    C.write_json(root / "reproduction.json", {
        "model": tag, "passed": passed, "tolerance": TOL, "fields": OLD_FIELDS + LABEL_FIELDS,
        "october_arms": october, "checks": checks, "written_utc": C.now_utc(),
        "git_commit": C.git_commit(), "patch_sha256": C.sha256(C.PATCH)})
    exact = sum(c.get("n_exact_slots", 0) for c in checks)
    slots = sum(c.get("n_slots", 0) for c in checks)
    worst = max((c.get("max_abs_delta", 0.0) for c in checks), default=0.0)
    print(f"[{C.now_utc()}] reproduction of the October fields: {'PASSED' if passed else 'FAILED'}; "
          f"{len(checks)} comparisons, {exact}/{slots} slots exact, max |delta| {worst:.3g}", flush=True)
    for line in failed:
        print("  " + line, flush=True)
    sys.exit(0 if passed and not failed else 1)


if __name__ == "__main__":
    main()
