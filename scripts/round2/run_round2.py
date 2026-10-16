"""Task 3: the round-2 training matrix -- plan (dry run), then train one GPU queue.

    python scripts/round2/run_round2.py plan  --tier 1            # arm table, estimates, path check
    python scripts/round2/run_round2.py train --tier 1 --queue A --gpu 0 --approved
    python scripts/round2/run_round2.py smoke --family llama-3.2-1b --gpu 0

Tiers: 1 = T1-a alt_mass missing seeds + T1-b alt_balance; 2c = concept_uncapped g=.125;
2b = coefficient sweep (dev only).  Member-wise (2a) has its own script (Task 4).
Training refuses unless: the reproduction gate passed, the Task 1 tests passed on this
commit, the patch on disk is the one in this repo, and --approved is given.  Every run
writes round2_run.json beside its adapter and one ledger row; failures are recorded,
never dropped.  Existing (October) checkpoints are only read.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import r2common as C

QUEUES = {"A": ("qwen3-1.7b-base",), "B": ("llama-3.2-1b", "falcon3-1b-base")}
STARTUP_MIN = 4.0           # model load + data tokenization, measured on Colab and the A40
OLD_KIND = {"alt_mass": "mass", "alt_balance": "within_kl", "alt_mass_balance": "uniform"}


def manifest(tag, name):
    path = C.OUTPUTS / tag / "run_manifests" / f"{name}.json"
    return {k.replace(" ", "_"): v for k, v in json.loads(path.read_text()).items()} if path.is_file() else {}


def read_config(path):
    f = Path(path) / "concept_training_config.json"
    return json.loads(f.read_text()) if f.is_file() else None


def train_file_hash(tag):
    f = C.leaf(tag) / "synonyms_train.jsonl"
    return C.sha256(f) if f.is_file() else None


def audited_train_hash(tag):
    """SHA-256 of synonyms_train.jsonl recorded by audit_concept_data.py when the data was built."""
    f = C.OUTPUTS / tag / "data_audit.json"
    if not f.is_file():
        return None
    for split in json.loads(f.read_text()).get("splits", []):
        if split.get("split") == "train":
            return split.get("sha256")
    return None


def existing_equivalent(tag, objective, gamma, seed, data_hash):
    """(path, note) of an October checkpoint that trained exactly this objective (Task 1
    test H), or (None, reason).

    The adapter path is written by the notebooks from the exact training arguments
    (seed_<s>/lambda_<l>_beta_<b>_aux_<kind>_<gamma>), so it must match; recorded config
    fields must agree with it.  Runs from before 2026-09-22 recorded no data hash: they are
    accepted only if the family's training file today still has the hash audited when it was
    built, i.e. the file never changed in between.
    """
    if objective not in OLD_KIND:
        return None, ""
    kind = OLD_KIND[objective]
    label = f"zhang_plus_{kind}_g{gamma:g}_seed{seed}"
    path = manifest(tag, "task15b").get(label)
    if not path or not (Path(path) / "adapter_config.json").is_file():
        return None, "no October checkpoint"
    if Path(path).name != f"lambda_1.0_beta_0.0_aux_{kind}_{gamma}" or Path(path).parent.name != f"seed_{seed}":
        return None, f"path does not encode {kind} {gamma} seed {seed}: {path}"
    cfg = read_config(path) or {}
    fields_ok = (cfg.get("alt_aux", kind) == kind
                 and math.isclose(float(cfg.get("alt_aux_weight", gamma)), gamma)
                 and int(cfg.get("seed", seed)) == seed
                 and float(cfg.get("concept_loss_weight", 1.0)) == 1.0
                 and float(cfg.get("target_mass", 0.6)) == 0.6
                 and float(cfg.get("contrast_beta", 0.0) or 0.0) == 0.0
                 and not cfg.get("exclude_target", False))
    if not fields_ok:
        return None, "recorded config disagrees with the plan"
    recorded = cfg.get("train_file_sha256")
    if data_hash is None:
        return path, "reuse"
    if recorded:
        return (path, "reuse") if recorded == data_hash else (None, "trained on a different data file")
    audited = audited_train_hash(tag)
    if audited and audited == data_hash:
        return path, "reuse (no data hash recorded; training file unchanged since it was audited)"
    return None, "no data hash recorded and the training file cannot be shown unchanged"


def arms(tier, families):
    rows = []
    for tag in families:
        data_hash = train_file_hash(tag)
        plan = []
        if tier == "1":
            plan += [("T1-a", "alt_mass", g, s) for g in C.MAIN_GAMMAS for s in C.SEEDS]
            plan += [("T1-b", "alt_balance", g, s) for g in C.MAIN_GAMMAS for s in C.SEEDS]
            plan += [("reuse", "alt_mass_balance", g, s) for g in C.MAIN_GAMMAS for s in C.SEEDS]
        elif tier == "2c":
            plan += [("T2-c", "concept_uncapped", C.PRIMARY_GAMMA, s) for s in C.SEEDS]
        elif tier == "2b":
            plan += [("T2-b", obj, g, C.SEEDS[0]) for obj in ("alt_mass", "alt_balance", "alt_mass_balance")
                     for g in C.SWEEP_GAMMAS]
        else:
            raise ValueError(tier)
        for block, objective, gamma, seed in plan:
            out = C.checkpoint_dir(tag, objective, gamma, seed)
            old, why = existing_equivalent(tag, objective, gamma, seed, data_hash)
            done = (out / "adapter_config.json").is_file() and _completed(out)
            if done:
                action = "done (round 2)"
            elif old:
                action = why
            elif block == "reuse":
                action = f"TRAIN (October checkpoint not reusable: {why})"
            else:
                action = "TRAIN"
            rows.append({"block": block, "family": tag, "objective": objective, "gamma": gamma,
                         "seed": seed, "action": action, "reuse_path": old or "",
                         "out": str(out), "passes": "dev only" if block == "T2-b" else "all"})
    return rows


def _completed(out):
    f = Path(out) / "round2_run.json"
    return f.is_file() and json.loads(f.read_text()).get("status") == "ok"


def family_minutes(tag):
    """Mean measured train_runtime of this family's concept-loss runs, plus startup."""
    mins = []
    for label, path in manifest(tag, "task15").items():
        if not label.startswith("zhang_seed"):
            continue
        hist = Path(path) / "training_history.jsonl"
        if hist.is_file():
            for line in hist.read_text().splitlines():
                if '"train_runtime"' in line:
                    mins.append(json.loads(line)["train_runtime"] / 60)
    return (sum(mins) / len(mins) if mins else float("nan")) + STARTUP_MIN, len(mins)


def cmd_plan(args):
    families = [f for q in QUEUES.values() for f in q]
    rows = arms(args.tier, families)
    est = {tag: family_minutes(tag) for tag in families}
    print(f"{'block':6s} {'family':16s} {'objective':18s} {'gamma':>8s} {'seed':>5s}  action")
    for r in rows:
        print(f"{r['block']:6s} {r['family']:16s} {r['objective']:18s} {r['gamma']:8g} {r['seed']:5d}  "
              f"{r['action']}" + (f"  <- {r['reuse_path'].split('/runs/')[-1]}" if r["reuse_path"] else ""))
    print()
    for q, fams in QUEUES.items():
        n = sum(1 for r in rows if r["family"] in fams and r["action"].startswith("TRAIN"))
        minutes = sum(est[r["family"]][0] for r in rows if r["family"] in fams and r["action"].startswith("TRAIN"))
        print(f"queue {q} ({', '.join(fams)}): {n} runs to train, ~{minutes / 60:.1f} h "
              f"(per-run estimate: " + ", ".join(f"{f} {est[f][0]:.0f} min from {est[f][1]} runs" for f in fams) + ")")
    clash = [r["out"] for r in rows if r["action"].startswith("TRAIN") and Path(r["out"]).exists()]
    # Lexical, not resolved: results/round2/checkpoints may be a symlink to the NAS.
    outside = [r["out"] for r in rows if not C.inside_r2(r["out"])]
    print(f"\npath check: {len(clash)} TRAIN outputs already exist (partial runs, renamed aside before "
          f"retraining); {len(outside)} outputs outside results/round2; every existing checkpoint is read-only.")
    C.write_json(C.R2 / "plans" / f"tier{args.tier}.json", {"rows": rows, "estimates_min": est,
                                                            "queues": QUEUES, "written": C.now_utc()})
    if outside:
        sys.exit("refusing: an output path escapes results/round2")


def gates(args):
    problems = []
    repro = C.R2 / "repro" / "reproduction_report.json"
    if not repro.is_file() or not json.loads(repro.read_text()).get("passed"):
        problems.append("Task 0 reproduction gate has not passed (results/round2/repro/reproduction_report.json)")
    tests = C.R2 / "task1_tests.json"
    if not tests.is_file() or not json.loads(tests.read_text()).get("passed"):
        problems.append("Task 1 tests have not passed on the server (results/round2/task1_tests.json)")
    elif json.loads(tests.read_text()).get("patch_sha256") != C.sha256(C.PATCH):
        problems.append("Task 1 tests ran against a different learning-concepts.patch")
    applied = C.EXT / ".round2_patch_sha256"
    if not applied.is_file() or applied.read_text().strip() != C.sha256(C.PATCH):
        problems.append("the learning-concepts checkout is not patched with this repo's patch "
                        "(run scripts/round2/prepare_ext.sh)")
    if not args.approved:
        problems.append("--approved not given (the dry-run table needs the authors' approval)")
    if problems:
        sys.exit("refusing to train:\n  - " + "\n  - ".join(problems))


def gpu_name(gpu):
    try:
        return subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=name,uuid",
                                        "--format=csv,noheader"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def train_one(row, gpu, python):
    tag, objective, gamma, seed = row["family"], row["objective"], row["gamma"], row["seed"]
    out = Path(row["out"])
    rid = C.run_id(tag, objective, gamma, seed)
    if out.exists() and not _completed(out):
        aside = out.with_name(out.name + f".partial-{int(time.time())}")
        out.rename(aside)
        print(f"  moved an unfinished attempt aside: {aside}")
    out.parent.mkdir(parents=True, exist_ok=True)
    log = C.R2 / "logs" / (rid.replace("/", "__") + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = [python, "train.py", "--model-name", C.FAMILIES[tag], "--dataset", "c4",
            "--dataset-type", "embedding", "--concept-loss-weight", "1.0",
            "--concept-objective", "set_marginal", "--contrast-beta", "0.0", "--seed", str(seed),
            "--num-train-epochs", "5", "--output-dir", str(out), "--save-strategy", "no",
            "--report-to", "none", "--per-device-train-batch-size", "8",
            "--gradient-accumulation-steps", "2", *C.gamma_flags(objective, gamma)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CONCEPT_DATA_ROOT": str(C.DATA),
           "CONCEPT_CHECKPOINT_ROOT": str(C.R2 / "ckpt_scratch"), "PYTHONUTF8": "1",
           "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "HF_HOME": os.environ.get("HF_HOME", str(C.WORK / "hf_cache"))}
    started, t0 = C.now_utc(), time.time()
    with open(log, "w", encoding="utf-8") as handle:
        handle.write("+ " + " ".join(argv) + "\n")
        handle.flush()
        code = subprocess.run(argv, cwd=C.EXT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    minutes = (time.time() - t0) / 60
    text = log.read_text(errors="ignore")
    status, note = "ok", ""
    if code:
        status = "oom" if "CUDA out of memory" in text or "OutOfMemoryError" in text else "failed"
        note = f"exit {code}; see {log.relative_to(C.MAIN)}"
    cfg = read_config(out)
    if status == "ok":
        want = dict(zip(("gamma_mass", "gamma_balance", "gamma_uncapped"),
                        (gamma * on for on in C.OBJECTIVES[objective])))
        bad = [k for k, v in want.items() if not cfg or not math.isclose(float(cfg.get(k, -1)), v)]
        hist = (out / "training_history.jsonl")
        nonfinite = hist.is_file() and any(
            isinstance(v, (int, float)) and not math.isfinite(v)
            for line in hist.read_text().splitlines() if line.strip()
            for v in json.loads(line).values())
        data_hash = train_file_hash(tag)
        if bad:
            status, note = "failed", f"recorded gammas differ from the plan: {bad}"
        elif nonfinite:
            status, note = "diverged", "non-finite value in training_history.jsonl"
        elif cfg.get("train_file_sha256") != data_hash:
            status, note = "failed", "trained on a file whose hash differs from the family's synonyms_train.jsonl"
    record = {"run_id": rid, "status": status, "note": note, "objective": objective, "gamma": gamma,
              "seed": seed, "model": C.FAMILIES[tag], "argv": argv, "gpu": str(gpu),
              "gpu_name": gpu_name(gpu), "started_utc": started, "ended_utc": C.now_utc(),
              "wall_minutes": round(minutes, 2), "git_commit": C.git_commit(),
              "patch_sha256": C.sha256(C.PATCH), "train_file_sha256": (cfg or {}).get("train_file_sha256"),
              "concept_training_config": cfg, "log": str(log)}
    if out.is_dir():
        C.write_json(out / "round2_run.json", record)
    C.append_ledger({"run_id": rid, "pass": "train", "model_tag": tag, "objective": objective,
                     "gamma": gamma, "seed": seed, "status": status, "checkpoint": str(out),
                     "started_utc": started, "ended_utc": record["ended_utc"],
                     "wall_minutes": f"{minutes:.1f}", "gpu": f"{gpu} {record['gpu_name']}",
                     "git_commit": record["git_commit"], "patch_sha256": record["patch_sha256"],
                     "train_file_sha256": record["train_file_sha256"] or "",
                     "concept_file_sha256": record["train_file_sha256"] or "",
                     "config_json": json.dumps(cfg, sort_keys=True) if cfg else "", "note": note})
    print(f"  {status:8s} {rid}  {minutes:.1f} min" + (f"  ({note})" if note else ""), flush=True)
    return status


def cmd_train(args):
    gates(args)
    families = QUEUES[args.queue] if not args.families else tuple(args.families.split(","))
    rows = [r for r in arms(args.tier, families) if r["action"].startswith("TRAIN")]
    print(f"queue {args.queue} on GPU {args.gpu}: {len(rows)} runs", flush=True)
    for row in rows[: args.max_runs or None]:
        # A notebook pass re-patches the shared checkout at its start; never train on
        # whatever it left behind.
        applied = C.EXT / ".round2_patch_sha256"
        if not applied.is_file() or applied.read_text().strip() != C.sha256(C.PATCH):
            sys.exit("the learning-concepts checkout changed under this queue; stopping")
        train_one(row, args.gpu, args.python)


def cmd_smoke(args):
    """Eight sequences, one epoch, every objective: the full path on the GPU, then deleted."""
    tag = args.family
    for objective in ("alt_mass", "alt_balance", "alt_mass_balance", "concept_uncapped"):
        out = C.R2 / "smoke" / tag / objective
        if out.exists():
            shutil.rmtree(out)
        argv = [args.python, "train.py", "--model-name", C.FAMILIES[tag], "--dataset", "c4",
                # 48 rows = 3 optimizer steps at 8 x 2; the trainer logs from step 1 on,
                # so fewer than 2 steps would print no term values at all.
                "--dataset-type", "embedding", "--concept-loss-weight", "1.0", "--max-train-samples", "48",
                "--num-train-epochs", "1", "--output-dir", str(out), "--save-strategy", "no",
                "--report-to", "none", "--logging-steps", "1", "--seed", "42",
                *C.gamma_flags(objective, C.PRIMARY_GAMMA)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "CONCEPT_DATA_ROOT": str(C.DATA),
               "CONCEPT_CHECKPOINT_ROOT": str(C.R2 / "ckpt_scratch"), "PYTHONUTF8": "1",
               "HF_HOME": os.environ.get("HF_HOME", str(C.WORK / "hf_cache"))}
        code = subprocess.run(argv, cwd=C.EXT, env=env, capture_output=True, text=True)
        hist = out / "training_history.jsonl"
        rows = [json.loads(l) for l in hist.read_text().splitlines()] if hist.is_file() else []
        terms = [{k: r[k] for k in r if k.endswith("_term")} for r in rows if any(k.endswith("_term") for k in r)]
        print(f"{objective:18s} exit={code.returncode}  first logged terms: {terms[:1]}")
        if code.returncode:
            print(code.stdout[-3000:])
        shutil.rmtree(out, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--tier", required=True, choices=["1", "2b", "2c"])
    t = sub.add_parser("train")
    t.add_argument("--tier", required=True, choices=["1", "2b", "2c"])
    t.add_argument("--queue", required=True, choices=sorted(QUEUES))
    t.add_argument("--families", default=None)
    t.add_argument("--gpu", required=True)
    t.add_argument("--approved", action="store_true")
    t.add_argument("--max-runs", type=int, default=0)
    t.add_argument("--python", default=sys.executable)
    s = sub.add_parser("smoke")
    s.add_argument("--family", required=True, choices=sorted(C.FAMILIES))
    s.add_argument("--gpu", required=True)
    s.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    {"plan": cmd_plan, "train": cmd_train, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
