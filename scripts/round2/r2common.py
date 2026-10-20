"""Shared constants and helpers for the round-2 factorial experiments.

Every round-2 script imports this module, so names, seeds, weights and paths are
declared once.  Server layout (CONCEPT_BASE = the task15 directory):

    $CONCEPT_BASE/concept_aware/concept-aware-training   this repo (MAIN)
    $CONCEPT_BASE/concept_aware/learning-concepts        pinned upstream + patch (EXT)
    $CONCEPT_BASE/concept_aware/data/c4/<tag>/embedding  concept data
    $CONCEPT_BASE/concept_aware/runs/<tag>/...           existing adapters (read-only here)
    MAIN/results/round2/                                 everything round 2 writes
"""
from __future__ import annotations

import csv
import datetime as _dt
import fcntl
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

MAIN = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get("CONCEPT_BASE", MAIN.parents[1] if MAIN.parent.name == "concept_aware"
                           else MAIN)).resolve()
WORK = BASE / "concept_aware"
EXT = WORK / "learning-concepts"
DATA = WORK / "data"
RUNS = WORK / "runs"
OUTPUTS = BASE / "outputs" if (BASE / "outputs").is_dir() else MAIN / "outputs"
R2 = MAIN / "results" / "round2"
CHECKPOINTS = R2 / "checkpoints"
LEDGER = R2 / "ledger.csv"
SEALED = MAIN / "sealed"
UPSTREAM_COMMIT = "b1d414143d11c8ed988b4cccbb06626cc8272bbe"
PATCH = MAIN / "external" / "learning-concepts.patch"

FAMILIES = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "qwen3-1.7b-base": "Qwen/Qwen3-1.7B-Base",
    "falcon3-1b-base": "tiiuae/Falcon3-1B-Base",
}
SEEDS = (42, 123, 2024)
PRIMARY_GAMMA = 0.125
SENSITIVITY_GAMMA = 0.0625
MAIN_GAMMAS = (SENSITIVITY_GAMMA, PRIMARY_GAMMA)
SWEEP_GAMMAS = (0.03125, 0.25, 0.5)      # T2-b, first seed, dev only

# objective -> which of (gamma_mass, gamma_balance, gamma_uncapped) carry gamma
OBJECTIVES = {
    "concept": (0, 0, 0),
    "alt_mass": (1, 0, 0),
    "alt_balance": (0, 1, 0),
    "alt_mass_balance": (1, 1, 0),
    "concept_uncapped": (0, 0, 1),
}
# The 2x2: (mass on?, balance on?) -> objective
CELLS = {(0, 0): "concept", (1, 0): "alt_mass", (0, 1): "alt_balance", (1, 1): "alt_mass_balance"}

# One mapping for every table, figure and legend.
DISPLAY = {
    "concept": "Concept loss",
    "alt_mass": "Alt-Mass",
    "alt_balance": "Alt-Balance",
    "alt_mass_balance": "Alt-Mass+Balance",
    "concept_uncapped": "Concept loss + whole-set",
    "ntp": "NTP",
    "untouched": "Untouched",
    "augmented_ntp": "Augmented NTP",
    "randomized": "Randomized",
    "memberwise": "Member-wise (Iyer et al.)",
}
# Never in any asset (checked by scan_assets()).
FORBIDDEN_TERMS = ("set marginal", "alternative-uniform", "alternative uniform",
                   "alternative-mass", "concept baseline", "λ=.25", "lambda=.25")

LEDGER_COLUMNS = (
    "run_id", "pass", "model_tag", "objective", "gamma", "seed", "variant", "status",
    "added_after_first_test_read", "checkpoint", "started_utc", "ended_utc",
    "wall_minutes", "gpu", "host", "git_commit", "patch_sha256", "train_file_sha256",
    "concept_file_sha256", "config_json", "note",
)


def run_id(tag, objective, gamma, seed, variant=""):
    g = "0" if not gamma else f"{float(gamma):g}"
    return "/".join(x for x in (tag, objective, f"g{g}", f"seed_{seed}", variant) if x)


def checkpoint_dir(tag, objective, gamma, seed, variant=""):
    return CHECKPOINTS / run_id(tag, objective, gamma, seed, variant)


def gamma_flags(objective, gamma):
    on = OBJECTIVES[objective]
    names = ("--gamma-mass", "--gamma-balance", "--gamma-uncapped")
    flags = []
    for name, active in zip(names, on):
        if active:
            flags += [name, str(gamma)]
    return flags


def leaf(tag):
    return DATA / "c4" / tag / "embedding"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(repo=MAIN):
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--"],
                               capture_output=True).returncode
        return out + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def now_utc():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def append_ledger(row: dict):
    """Append one row; never rewrites.  Unknown keys are refused, missing ones blank."""
    unknown = set(row) - set(LEDGER_COLUMNS)
    if unknown:
        raise KeyError(f"not ledger columns: {sorted(unknown)}")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {**{c: "" for c in LEDGER_COLUMNS}, "host": socket.gethostname(), **row}
    with open(LEDGER, "a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        empty = handle.read(1) == ""
        handle.seek(0, os.SEEK_END)
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS)
        if empty:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def read_ledger():
    if not LEDGER.is_file():
        return []
    with open(LEDGER, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def inside_r2(path):
    """True when `path` is under results/round2, compared lexically so that a
    subdirectory symlinked to the NAS (checkpoints/, per_target/) still counts."""
    return os.path.normpath(os.path.abspath(path)).startswith(os.path.normpath(os.path.abspath(R2)) + os.sep)


def refuse_overwrite(path):
    """New outputs only: a path that exists outside results/round2 is never written."""
    if not inside_r2(path) and Path(path).exists():
        raise FileExistsError(f"refusing to write an existing path outside results/round2: {path}")


def scan_assets(root):
    """Return (file, term) for every forbidden term found in a generated asset."""
    hits = []
    for path in Path(root).rglob("*"):
        if path.suffix.lower() not in {".tex", ".csv", ".md", ".json", ".txt", ".svg"}:
            continue
        text = path.read_text(errors="ignore").lower()
        for term in FORBIDDEN_TERMS:
            if term.lower() in text:
                hits.append((str(path), term))
        if "precision" in text and any(k in path.name.lower() for k in ("coinco", "semeval", "openvocab", "wordnet")):
            hits.append((str(path), "precision"))
    return hits
