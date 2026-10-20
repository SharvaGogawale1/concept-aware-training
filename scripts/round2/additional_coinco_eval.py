"""Additional CoInCo evaluation (sealed/EVALUATION_PLAN.md): score the 45 fixed checkpoints once.

    python scripts/round2/additional_coinco_eval.py check --family llama-3.2-1b
    bash scripts/round2/r2tmux.sh r2acpre scripts/round2/additional_coinco_eval.py preflight --family llama-3.2-1b --gpu 1
    bash scripts/round2/r2tmux.sh r2acllama scripts/round2/additional_coinco_eval.py score --family llama-3.2-1b --gpu 1

check      every gate below and the model's 15 checkpoints; scores nothing.
preflight  re-scores CoInCo-800 test for the model's October Concept loss seed-42 checkpoint through
           the same command builder, and requires every per-target value to equal the stored October
           value exactly (plan §9 step 3).  Reads nothing from the additional set.
score      the model's 15 checkpoints of plan §3 on the additional set, left and full scorers, after a
           passed preflight.

Gates (plan §9 step 2).  The plan gates apply to score only; preflight reads nothing from the
additional set, so it may run before the plan is approved.  check reports both groups.
  - plan gates: ROUND2_PROTOCOL.md committed, clean and approved (evaluate_round2.protocol_ok), with
    a line "Analysis plan SHA-256: <hex>"; sealed/EVALUATION_PLAN.md committed, clean, equal to that
    hash, and both approval lines in its section 11 ticked and dated;
  - the ID file, the evaluation file and eval_swords.py equal their pinned SHA-256;
  - all 15 checkpoints present, every round-2 run marked ok;
  - no completion record for this model, unless --correction names a committed correction note
    (plan §9 step 8), which writes to its own directory and never touches the original outputs.

Technical failures (plan §9 step 5): an evaluator that exits non-zero is retried at most twice more.
After three failed attempts the checkpoint is recorded as missing.  An attempt cut off by an
interruption is not counted.  An evaluator that exits 0 but writes an output without exactly the
3,370 targets is not a technical failure: the run stops, keeps the file, and asks for investigation.
When every checkpoint is scored or missing, COMPLETE.json records the SHA-256 of every output, and
any later score of this model is refused.

The console shows progress only.  The evaluator prints scores, so its logs go to evaluator_logs/ and
are never echoed; a failure prints the traceback alone.  Outputs: results/round2/additional_coinco/
<model>/ (or correction_<note>/<model>/).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import analyze_round2 as A
import evaluate_round2 as E
import r2common as C

PLAN = C.SEALED / "EVALUATION_PLAN.md"
PROTOCOL = C.MAIN / "ROUND2_PROTOCOL.md"
OUT_ROOT = C.R2 / "additional_coinco"
IDS = C.SEALED / "sealed_holdout_ids.json"
EVAL_FILE = C.R2 / "sealed_data" / "coinco_sealed_v2.json.gz"
EVALUATOR = E.EVAL_DIR / "eval_swords.py"
PREFLIGHT_FILE = E.LEXSUB["coinco_test"]
PINS = {                                   # path -> SHA-256 (plan §2, §3)
    IDS: "2820863846d6cbe7d9d0adf3943c269c7fedf5c21f2e424be8152a6395355c08",        # commit 863cfa8
    EVAL_FILE: "db8e2973f0bd1daf95d1d85fa713158768f03e8e35ae0d5a8ff60cc2d71348a2",
    EVALUATOR: "005e1a7c0e71828422b7fb764518202b948395d13100a7223d039efb40df996f",  # unchanged since 5b7f8b7
    PREFLIGHT_FILE: "4400bf20e4ede042446743c08e536d3d4479a38515d92e3331e2ab34745c2bf2",
}
N_TARGETS = 3370
MODES = ("left", "full")
MAX_ATTEMPTS = 3                          # one attempt and at most two retries
PREFLIGHT_ARM = ("concept", None, 42)
CONFIGS = ("concept", "alt_mass", "alt_balance", "alt_mass_balance", A.WHOLESET)
APPROVERS = ("Sharva Gogawale", "Chen Shani")
PLAN_HASH_LINE = re.compile(r"Analysis plan SHA-256:\s*`?([0-9a-f]{64})`?")
CORRECTION_NOTE = re.compile(r"correction_\d{4}-\d{2}-\d{2}_[A-Za-z0-9_-]+\.md")


class StopRun(Exception):
    """A problem that retrying cannot fix: stop, keep every file, investigate."""


# ---------- gates ----------

def plan_arms():
    """(objective, gamma, seed, source) for the 15 checkpoints of plan §3, in a fixed order."""
    arms = [a for a in A.needed_arms() if a[1] in (None, C.PRIMARY_GAMMA)] + A.wholeset_arms()
    if len(arms) != 15 or {a[0] for a in arms} != set(CONFIGS):
        raise RuntimeError("plan §3 names 15 checkpoints per model over five configurations")
    return arms


def git_state(path, repo=C.MAIN):
    """(tracked, clean) for one file."""
    rel = os.path.relpath(path, repo)
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", rel],
                           capture_output=True).returncode == 0
    return tracked, clean


def approvals(text):
    """{name: date or None}: a line counts only when it is ticked and carries a date."""
    section = text.split("## 11. Approval", 1)
    body = section[1] if len(section) == 2 else ""
    out = {}
    for name in APPROVERS:
        m = re.search(rf"^- \[([ xX])\] {re.escape(name)}, date:[ \t]*(\d{{4}}-\d{{2}}-\d{{2}})?", body, re.M)
        out[name] = m.group(2) if m and m.group(1) in "xX" and m.group(2) else None
    return out


def plan_problems(plan=PLAN, protocol=PROTOCOL, repo=C.MAIN, protocol_check=E.protocol_ok):
    if not Path(plan).is_file():
        return [f"{plan} does not exist"]
    problems = []
    tracked, clean = git_state(plan, repo)
    if not tracked:
        problems.append("the plan is not committed")
    elif not clean:
        problems.append("the plan has uncommitted edits")
    for name, date in approvals(Path(plan).read_text()).items():
        if not date:
            problems.append(f"the plan's approval line for {name} is not ticked and dated")
    text = Path(protocol).read_text() if Path(protocol).is_file() else ""
    pinned = PLAN_HASH_LINE.search(text)
    if not pinned:
        problems.append("ROUND2_PROTOCOL.md does not record the plan's SHA-256 ('Analysis plan SHA-256: <hex>')")
    elif pinned.group(1) != C.sha256(plan):
        problems.append("the plan differs from the version whose SHA-256 ROUND2_PROTOCOL.md records")
    problems += [f"ROUND2_PROTOCOL.md {p}" for p in protocol_check()]
    return problems


def data_problems(pins=None, ids_file=IDS, eval_file=EVAL_FILE):
    pins = PINS if pins is None else pins
    problems = []
    for path, want in pins.items():
        if not Path(path).is_file():
            problems.append(f"{path} does not exist")
        elif C.sha256(path) != want:
            problems.append(f"{path} does not match its pinned SHA-256")
    if not problems:
        meta = json.loads(Path(ids_file).read_text())
        if meta.get("eval_file_sha256") != pins[eval_file] or len(meta.get("ids", [])) != N_TARGETS:
            problems.append(f"{ids_file} does not describe {N_TARGETS} targets of the pinned evaluation file")
    return problems


def resolve_checkpoints(tag, src=None):
    """[(arm, source, checkpoint)] in plan order, and the problems found."""
    src = src or A.Sources(C.OUTPUTS, C.R2)
    out, problems = [], []
    try:
        locked = src.locked(tag)
    except A.Missing as e:
        locked, problems = {}, [f"no locked October arms file: {e}"]
    for objective, gamma, seed, source in plan_arms():
        arm = (objective, gamma, seed)
        if source == "october":
            ckpt = locked.get(A.october_label(*arm))
            ok, why = bool(ckpt) and Path(ckpt).is_dir(), f"October checkpoint {ckpt} is not a directory"
        else:
            d = C.checkpoint_dir(tag, objective, gamma, seed)
            marker = d / "round2_run.json"
            ok = marker.is_file() and json.loads(marker.read_text()).get("status") == "ok"
            ckpt, why = str(d), f"{marker} is not status ok"
        if not ok:
            problems.append(f"{A.flat(*arm)}: {why}")
        out.append((arm, source, ckpt))
    return out, problems


def output_root(tag, correction=None, base=None):
    base = OUT_ROOT if base is None else base
    return (base if correction is None else base / f"correction_{Path(correction).stem}") / tag


def correction_problems(note, repo=C.MAIN, sealed=C.SEALED):
    if note is None:
        return []
    note = Path(note)
    if not CORRECTION_NOTE.fullmatch(note.name) or note.resolve().parent != Path(sealed).resolve():
        return [f"a correction note must be sealed/correction_YYYY-MM-DD_<slug>.md, not {note}"]
    if not note.is_file():
        return [f"{note} does not exist"]
    tracked, clean = git_state(note, repo)
    return [] if tracked and clean else [f"{note} is not committed, or has uncommitted edits"]


def gates(tag, correction=None):
    """(checkpoints, plan problems, every other problem)."""
    problems = data_problems() + correction_problems(correction)
    ckpts, missing = resolve_checkpoints(tag)
    problems += missing
    if (output_root(tag, correction) / "COMPLETE.json").exists():
        problems.append("a completion record already exists for this model; a second scoring is refused "
                        "(plan §9 step 6; a documented correction uses --correction)")
    return ckpts, plan_problems(), problems


def eta_hours(minutes_done, remaining):
    """Hours left at the mean pace of the checkpoints scored in this session (None before the first)."""
    return None if not minutes_done else sum(minutes_done) / len(minutes_done) * remaining / 60


# ---------- scoring ----------

def target_ids(ids_file=IDS):
    return set(json.loads(Path(ids_file).read_text())["ids"])


def command(tag, ckpt, data_file, out, py=sys.executable):
    """The round-2 lexsub command of evaluate_round2.py, on another data file."""
    model = C.FAMILIES[tag]
    return [py, EVALUATOR, "--checkpoints", ckpt, "--tokenizer_path", model, "--base_model", model,
            "--swords_json", data_file, "--results_json", out, "--modes", *MODES], C.MAIN


def output_ok(path, ckpt, ids):
    """Structure only: one record for this checkpoint holding exactly `ids` in both scorers.
    No score is read or returned."""
    try:
        rows = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if not (isinstance(rows, list) and len(rows) == 1 and rows[0].get("checkpoint") == ckpt):
        return False
    for mode in MODES:
        seen = [t.get("target_id") for t in (rows[0].get(mode) or {}).get("per_target") or []]
        if len(seen) != len(ids) or set(seen) != ids:
            return False
    return True


def read_attempts(root):
    f = Path(root) / "attempts.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.is_file() else []


def append_attempt(root, row):
    f = Path(root) / "attempts.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def failed_attempts(rows, name):
    return sum(1 for r in rows if r["flat"] == name and r["event"] == "end" and not r["ok"])


def traceback_tail(log, limit=20):
    """The last traceback in an evaluator log, and nothing else (the log also holds scores)."""
    lines = Path(log).read_text(errors="replace").splitlines() if Path(log).is_file() else []
    starts = [i for i, line in enumerate(lines) if line.startswith("Traceback")]
    return lines[starts[-1]:][:limit] if starts else []


def ledger(tag, arm, ckpt, step, status, started, minutes, gpu, note):
    C.append_ledger({"run_id": C.run_id(tag, arm[0], arm[1], arm[2]), "pass": f"additional_coinco:{step}",
                     "model_tag": tag, "objective": arm[0], "gamma": "" if arm[1] is None else f"{arm[1]:g}",
                     "seed": str(arm[2]), "status": status, "added_after_first_test_read": "false",
                     "checkpoint": str(ckpt), "started_utc": started, "ended_utc": C.now_utc(),
                     "wall_minutes": f"{minutes:.1f}", "gpu": str(gpu), "git_commit": C.git_commit(),
                     "note": note})


def score_one(tag, arm, ckpt, root, ids, gpu, py, runner=E.run, data_file=EVAL_FILE, record=ledger):
    """'ok' or 'missing'.  Raises StopRun on a problem that retrying cannot fix."""
    name = A.flat(*arm)
    final = root / "outputs" / f"{name}.json"
    if final.exists():
        if output_ok(final, ckpt, ids):
            return "ok"
        raise StopRun(f"{final} exists but does not hold exactly the {len(ids)} targets for {ckpt}")
    rows = read_attempts(root)
    while failed_attempts(rows, name) < MAX_ATTEMPTS:
        k = 1 + sum(1 for r in rows if r["flat"] == name and r["event"] == "start")
        attempt = root / "attempts" / f"{name}.attempt{k}.json"
        log = root / "evaluator_logs" / f"{name}.attempt{k}.log"
        for d in (attempt.parent, log.parent, final.parent):
            d.mkdir(parents=True, exist_ok=True)
        started, t0 = C.now_utc(), time.time()
        append_attempt(root, {"flat": name, "event": "start", "attempt": k, "checkpoint": ckpt, "utc": started})
        argv, cwd = command(tag, ckpt, data_file, attempt, py)
        code = runner(argv, cwd, gpu, log)
        minutes = (time.time() - t0) / 60
        if code == 0 and not output_ok(attempt, ckpt, ids):
            append_attempt(root, {"flat": name, "event": "end", "attempt": k, "ok": False, "exit_code": code,
                                  "utc": C.now_utc(), "stopped": True})
            raise StopRun(f"{attempt}: the evaluator exited 0 but the output does not hold exactly the "
                          f"{len(ids)} targets; this is not a technical failure (plan §9 step 8)")
        ok = code == 0
        sha = None
        if ok:
            os.replace(attempt, final)
            sha = C.sha256(final)
        append_attempt(root, {"flat": name, "event": "end", "attempt": k, "ok": ok, "exit_code": code,
                              "utc": C.now_utc(), "minutes": round(minutes, 2), "sha256": sha})
        record(tag, arm, ckpt, "score", "ok" if ok else "failed", started, minutes, gpu,
               f"attempt {k}; exit {code}; root {root}")
        if ok:
            return "ok"
        print(f"  attempt {k} of {name} failed (exit {code}); evaluator log {log}", flush=True)
        for line in traceback_tail(log):
            print("    " + line, flush=True)
        rows = read_attempts(root)
    return "missing"


def completion_record(tag, root, resolved, correction=None):
    """resolved: [(arm, source, checkpoint, 'ok' | 'missing')]."""
    rows = read_attempts(root)
    outputs, missing = {}, {}
    for arm, source, ckpt, status in resolved:
        name = A.flat(*arm)
        mine = [r for r in rows if r["flat"] == name and r["event"] == "end"]
        if status == "ok":
            outputs[name] = {"checkpoint": ckpt, "source": source,
                             "sha256": C.sha256(root / "outputs" / f"{name}.json"), "attempts": len(mine)}
        else:
            missing[name] = {"checkpoint": ckpt, "source": source, "attempts": len(mine),
                             "exit_codes": [r["exit_code"] for r in mine]}
    return {"model": tag, "written_utc": C.now_utc(), "git_commit": C.git_commit(),
            "plan_sha256": C.sha256(PLAN) if PLAN.is_file() else None,
            "pins": {str(p): h for p, h in PINS.items()}, "correction": None if correction is None else str(correction),
            "max_attempts": MAX_ATTEMPTS, "outputs": outputs, "missing": missing}


def preflight_status(root):
    """The latest preflight result of this root, or None."""
    results = sorted((Path(root) / "preflight").glob("run*.result.json"),
                     key=lambda p: int(re.search(r"run(\d+)", p.name).group(1)))
    return json.loads(results[-1].read_text()) if results else None


def same(u, v):
    if isinstance(u, float) and isinstance(v, float) and math.isnan(u) and math.isnan(v):
        return True
    return u == v


def compare_records(new, old):
    """Every per-target field of both scorers, exactly."""
    problems, n = [], 0
    for mode in MODES:
        a, b = new[mode]["per_target"], old[mode]["per_target"]
        if len(a) != len(b):
            problems.append(f"{mode}: {len(a)} targets vs {len(b)} stored")
            continue
        for x, y in zip(a, b):
            n += 1
            fields = sorted(f for f in set(x) | set(y) if not same(x.get(f), y.get(f)))
            if fields:
                problems.append(f"{mode} {y.get('target_id')}: {', '.join(fields)}")
    return {"n_target_records": n, "n_mismatched": len(problems), "problems": problems[:20],
            "passed": not problems and n > 0}


def preflight(tag, gpu, py, correction=None, runner=E.run, src=None):
    src = src or A.Sources(C.OUTPUTS, C.R2)
    root = output_root(tag, correction)
    label = A.october_label(*PREFLIGHT_ARM)
    stored, stored_file = src.october_record(tag, "coinco_test", label)
    ckpt = stored["checkpoint"]
    k = 1 + len(list((root / "preflight").glob("run*.result.json")))
    out = root / "preflight" / f"run{k}.coinco_test_sub800.json"
    log = root / "preflight" / f"run{k}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    started, t0 = C.now_utc(), time.time()
    argv, cwd = command(tag, ckpt, PREFLIGHT_FILE, out, py)
    code = runner(argv, cwd, gpu, log)
    minutes = (time.time() - t0) / 60
    if code == 0 and out.is_file():
        result = compare_records(json.loads(out.read_text())[0], stored)
    else:
        result = {"passed": False, "problems": [f"evaluator exit {code}"], "traceback": traceback_tail(log)}
    result.update({"model": tag, "checkpoint": ckpt, "stored_file": str(stored_file), "data_file": str(PREFLIGHT_FILE),
                   "evaluator_sha256": C.sha256(EVALUATOR), "git_commit": C.git_commit(),
                   "written_utc": C.now_utc(), "minutes": round(minutes, 2)})
    C.write_json(root / "preflight" / f"run{k}.result.json", result)
    ledger(tag, PREFLIGHT_ARM, ckpt, "preflight", "ok" if result["passed"] else "failed", started, minutes, gpu,
           f"CoInCo-800 test re-score; {result.get('n_mismatched', '?')} mismatched target records")
    return result


# ---------- commands ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("check", "preflight", "score"))
    ap.add_argument("--family", required=True, choices=sorted(C.FAMILIES))
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--correction", default=None,
                    help="sealed/correction_YYYY-MM-DD_<slug>.md: a committed note of a genuine bug (plan §9 step 8)")
    args = ap.parse_args()
    tag = args.family
    if args.command != "check" and args.gpu is None:
        sys.exit("--gpu is required")
    ckpts, plan_gate, other = gates(tag, args.correction)
    root = output_root(tag, args.correction)
    if args.command == "check":
        for arm, source, ckpt in ckpts:
            print(f"  {A.flat(*arm):34s} {source:7s} {ckpt}")
        state = preflight_status(root)
        print(f"preflight: {'none yet' if state is None else ('passed' if state['passed'] else 'FAILED')}")
        for title, group in (("data, evaluator, checkpoints (needed by preflight and score)", other),
                             ("plan approval (needed by score)", plan_gate)):
            print(f"{title}: {'ok' if not group else f'{len(group)} problem(s)'}")
            for p in group:
                print("  - " + p)
        sys.exit(1 if other or plan_gate else 0)
    problems = other + (plan_gate if args.command == "score" else [])
    if problems:
        print(f"{tag}: refusing to {args.command}:", flush=True)
        for p in problems:
            print("  - " + p, flush=True)
        sys.exit(1)
    if args.command == "preflight":
        result = preflight(tag, args.gpu, args.python, args.correction)
        print(f"[{C.now_utc()}] {tag} preflight {'PASSED' if result['passed'] else 'FAILED'}: "
              f"{result.get('n_target_records', 0)} target records, {result.get('n_mismatched', '?')} mismatched",
              flush=True)
        for p in result["problems"]:
            print("  " + p, flush=True)
        sys.exit(0 if result["passed"] else 1)

    state = preflight_status(root)
    if not state or not state["passed"] or state["evaluator_sha256"] != C.sha256(EVALUATOR):
        sys.exit(f"{tag}: refusing to score: no passed preflight for this evaluator in {root / 'preflight'}")
    ids = target_ids()
    print(f"[{C.now_utc()}] {tag}: scoring {len(ckpts)} checkpoints on {len(ids)} targets, GPU {args.gpu}; "
          f"outputs in {root}", flush=True)
    resolved = []
    try:
        paced = []
        for i, (arm, source, ckpt) in enumerate(ckpts, 1):
            t0 = time.time()
            status = score_one(tag, arm, ckpt, root, ids, args.gpu, args.python)
            resolved.append((arm, source, ckpt, status))
            minutes = (time.time() - t0) / 60
            if status == "ok" and minutes > 0.5:          # scored now, not found already done
                paced.append(minutes)
            left = eta_hours(paced, len(ckpts) - i)
            print(f"[{C.now_utc()}] {i:2d}/{len(ckpts)} {status:7s} {A.flat(*arm)}  {minutes:.1f} min"
                  + ("" if left is None else f"  (~{left:.1f} h left)"), flush=True)
    except StopRun as e:
        sys.exit(f"{tag}: STOPPED, nothing recorded as complete: {e}")
    if (root / "COMPLETE.json").exists():
        sys.exit(f"{tag}: a completion record appeared while scoring; not overwriting it")
    C.write_json(root / "COMPLETE.json", completion_record(tag, root, resolved, args.correction))
    missing = [A.flat(*a) for a, _, _, s in resolved if s == "missing"]
    print(f"[{C.now_utc()}] {tag}: done; {len(resolved) - len(missing)} scored, {len(missing)} missing"
          + (f" ({', '.join(missing)})" if missing else "") + f"; completion record {root / 'COMPLETE.json'}",
          flush=True)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
