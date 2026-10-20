"""Task 0: enumerate every existing run and recover its settings from code and configs.

Reads, never writes, the run directories.  Sources, in order of authority:
  concept_training_config.json   what train.py recorded (objective, lambda, tau, gamma, ...)
  adapter_config.json            LoRA rank / alpha / modules / base model
  training_history.jsonl         steps, epochs, train_runtime (wall-clock back-fill), NaNs
  run manifests                  label -> path, written by the notebooks
Settings train.py does NOT record (quantization, LoRA scaling, checkpoint choice) are
taken from the pinned trainer's code defaults and marked as such.

    python scripts/round2/audit_existing_runs.py                 # server: scans RUNS
    python scripts/round2/audit_existing_runs.py --include-local-logs   # + Colab-only copies
Writes artifacts/run_manifest_existing.csv and artifacts/run_audit_report.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import r2common as C

# From external/learning-concepts/conceptlib/config.py and train.py (pinned + patch):
# use_4bit=True, nf4, double quant off, compute dtype fp16; no CLI flag changes it.
CODE_DEFAULTS = {
    "quantization": "4-bit NF4, no double quant, fp16 compute (code default; not recorded per run)",
    "checkpoint_used": "final (save_strategy=no, load_best_model_at_end=False)",
}
PAPER = {"lora_r": 4, "lora_scaling_as_written": 4, "learning_rate": 7e-5,
         "effective_batch_size": 16, "num_train_epochs": 5, "target_mass": 0.6}


PATH_NAMES = (("augmented_ntp", "augmented_ntp"), ("randomized", "randomized"),
              ("alternative_contrastive", "alternative_only:uniform+contrast"),
              ("contrast", "alternative_only:uniform+contrast"),
              ("alternative_uniform", "alternative_only:uniform"),
              ("inclusive_uniform", "inclusive_uniform"), ("zhang_plus_aux", "zhang_plus_aux"),
              ("zhang", "concept"), ("ntp", "ntp"))


def objective_from_path(path):
    p = str(path)
    for key, name in PATH_NAMES:
        if key in p:
            return name
    return "unknown"


def objective_name(cfg, path):
    """(objective, source): from the recorded config, or from the path when the config
    predates the field that decides it (older runs recorded no exclude_target/train_file)."""
    p = str(path)
    if cfg is None:
        return objective_from_path(p), "path (no concept_training_config.json)"
    if "exclude_target" not in cfg or "train_file" not in cfg:
        return objective_from_path(p), "path (config predates exclude_target/train_file)"
    return _objective_from_config(cfg, p), "config"


def _objective_from_config(cfg, p):
    lam = float(cfg.get("concept_loss_weight") or 0)
    kind = cfg.get("alt_aux", "none")
    gamma = float(cfg.get("alt_aux_weight") or 0)
    gm, gb, gu = (float(cfg.get(k) or 0) for k in ("gamma_mass", "gamma_balance", "gamma_uncapped"))
    obj = cfg.get("concept_objective", "set_marginal")
    train_file = str(cfg.get("train_file", ""))
    if "aug" in Path(train_file).name:
        return "augmented_ntp"
    if "randomized" in p and lam > 0:
        return "randomized"
    if lam == 0 and not any((gamma, gm, gb, gu)) and float(cfg.get("contrast_beta") or 0) == 0:
        return "ntp"
    if "verified" in train_file or "/verified_" in p:
        return f"verified:{obj}" + (f"+{kind}" if kind != "none" else "")
    if cfg.get("exclude_target"):
        return f"alternative_only:{obj}" + ("+contrast" if float(cfg.get("contrast_beta") or 0) else "")
    if obj == "uniform":
        return "inclusive_uniform"
    if obj == "set_marginal" and any((gm, gb, gu)):
        on = tuple(int(bool(x)) for x in (gm, gb, gu))
        for name, pattern in C.OBJECTIVES.items():
            if pattern == on:
                return name
    if obj == "set_marginal" and kind != "none":
        return {"mass": "alt_mass", "uniform": "alt_mass_balance",
                "within_kl": "alt_balance"}[kind] + "(alt_aux)"
    if obj == "set_marginal":
        return "concept"
    return obj


def history_stats(path):
    rows, nan, runtime, last_step, last_epoch = 0, False, None, None, None
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            for key in ("loss", "ce_loss", "concept_loss", "alt_aux_loss", "eval_loss",
                        "mass_term", "balance_term", "uncapped_term"):
                v = row.get(key)
                if isinstance(v, (int, float)) and not math.isfinite(v):
                    nan = True
            if "train_runtime" in row:
                runtime = float(row["train_runtime"])
            last_step = row.get("step", last_step)
            last_epoch = row.get("epoch", last_epoch)
    return {"history_rows": rows, "history_has_nonfinite": nan, "last_step": last_step,
            "last_epoch": last_epoch,
            "train_runtime_min": round(runtime / 60, 2) if runtime is not None else ""}


def load_manifests(outputs):
    """path -> sorted labels, from every outputs/<tag>/run_manifests/*.json (not .bak)."""
    labels = defaultdict(set)
    for path in sorted(Path(outputs).glob("*/run_manifests/*.json")):
        if "shards" in path.name:
            continue
        try:
            for label, run in json.loads(path.read_text()).items():
                labels[str(Path(run))].add(f"{path.stem}:{label}")
        except (json.JSONDecodeError, AttributeError):
            continue
    return labels


def find_runs(roots):
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for cfg in root.rglob("adapter_config.json"):
            run = cfg.parent
            if run.resolve() in seen:
                continue
            seen.add(run.resolve())
            yield run


def describe(run, manifests, data_hashes):
    cfg_path = run / "concept_training_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else None
    adapter = json.loads((run / "adapter_config.json").read_text())
    r, alpha = adapter.get("r"), adapter.get("lora_alpha")
    tag = next((t for t in C.FAMILIES if f"/{t}/" in f"{run}/"), "")
    if not tag:
        base = adapter.get("base_model_name_or_path", "")
        tag = next((t for t, m in C.FAMILIES.items() if m == base), "")
    seed = (cfg or {}).get("seed")
    if seed is None:
        m = re.search(r"seed_?(\d+)", str(run))
        seed = int(m.group(1)) if m else ""
    row = {
        "checkpoint_path": str(run),
        "location": "server" if str(run).startswith(str(C.WORK)) else "local-copy",
        "labels": ";".join(sorted(manifests.get(str(run), []))),
        "model_tag": tag,
        "base_model": adapter.get("base_model_name_or_path", ""),
        "objective": objective_name(cfg, run)[0],
        "objective_source": objective_name(cfg, run)[1],
        "seed": seed,
        "lambda": (cfg or {}).get("concept_loss_weight", ""),
        "tau": (cfg or {}).get("target_mass", ""),
        "concept_objective": (cfg or {}).get("concept_objective", ""),
        "alt_aux": (cfg or {}).get("alt_aux", ""),
        "gamma": (cfg or {}).get("alt_aux_weight", ""),
        "gamma_mass": (cfg or {}).get("gamma_mass", ""),
        "gamma_balance": (cfg or {}).get("gamma_balance", ""),
        "gamma_uncapped": (cfg or {}).get("gamma_uncapped", ""),
        "exclude_target": (cfg or {}).get("exclude_target", ""),
        "contrast_beta": (cfg or {}).get("contrast_beta", ""),
        "slot_ntp_weight": (cfg or {}).get("slot_ntp_weight", ""),
        "learning_rate": (cfg or {}).get("learning_rate", ""),
        "epochs": (cfg or {}).get("num_train_epochs", ""),
        "per_device_batch": (cfg or {}).get("per_device_train_batch_size", ""),
        "grad_accum": (cfg or {}).get("gradient_accumulation_steps", ""),
        "effective_batch": (cfg or {}).get("effective_batch_size", ""),
        "lora_r": r,
        "lora_alpha": alpha,
        "lora_scaling_alpha_over_r": (alpha / r) if r and alpha else "",
        "lora_dropout": adapter.get("lora_dropout", ""),
        "lora_targets": ",".join(sorted(adapter.get("target_modules") or [])),
        "quantization": CODE_DEFAULTS["quantization"],
        "checkpoint_used": CODE_DEFAULTS["checkpoint_used"]
        if (cfg or {}).get("save_strategy", "no") == "no" else f"save_strategy={cfg.get('save_strategy')}",
        "train_file": (cfg or {}).get("train_file", ""),
        "train_file_sha256": (cfg or {}).get("train_file_sha256", ""),
        "train_file_matches_disk": "",
        "upstream_commit": (cfg or {}).get("upstream_commit", ""),
        "repo_git_commit": "not recorded by train.py",
        "has_concept_config": cfg is not None,
    }
    disk = data_hashes.get(row["train_file"])
    if disk is not None and row["train_file_sha256"]:
        row["train_file_matches_disk"] = disk == row["train_file_sha256"]
    row.update(history_stats(run / "training_history.jsonl"))
    return row


def mismatches(rows):
    """Shared settings that differ from the paper, per run (Task 0.4).  Smoke runs (a few
    steps, never evaluated) are listed separately by the caller, not here."""
    out = []
    for row in rows:
        if "/smoke" in row["checkpoint_path"]:
            continue
        if not row["has_concept_config"]:
            out.append((row["checkpoint_path"], "no concept_training_config.json"))
            continue
        problems = []
        if row["lora_r"] != PAPER["lora_r"]:
            problems.append(f"lora_r={row['lora_r']}")
        if row["learning_rate"] not in ("", None) and abs(float(row["learning_rate"]) - 7e-5) > 1e-12:
            problems.append(f"lr={row['learning_rate']}")
        if row["effective_batch"] not in ("", None) and int(row["effective_batch"]) != 16:
            problems.append(f"effective_batch={row['effective_batch']}")
        epochs_expected = 1 if row["objective"] == "augmented_ntp" else 5
        if row["epochs"] not in ("", None) and float(row["epochs"]) != epochs_expected:
            problems.append(f"epochs={row['epochs']} (expected {epochs_expected})")
        if row["tau"] not in ("", None) and float(row["tau"]) != 0.6:
            problems.append(f"tau={row['tau']}")
        if row.get("last_epoch") not in ("", None) and float(row["last_epoch"]) + 1e-6 < epochs_expected:
            problems.append(f"history ends at epoch {row['last_epoch']}")
        if row.get("history_has_nonfinite"):
            problems.append("non-finite loss in history")
        if problems:
            out.append((row["checkpoint_path"], "; ".join(problems)))
    return out


def seed_report(rows):
    """Seeds per (model, objective, gamma), for Task 0.2 and 0.3."""
    table = defaultdict(set)
    for row in rows:
        gamma = row["gamma"] or row["gamma_mass"] or row["gamma_balance"] or row["gamma_uncapped"] or 0
        lam = row["lambda"]
        table[(row["model_tag"], row["objective"], f"lambda={lam}", f"gamma={gamma}")].add(row["seed"])
    return {" | ".join(map(str, k)): sorted(v, key=str) for k, v in sorted(table.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", action="append", default=None,
                    help="directories to scan for adapters (default: the server runs/)")
    ap.add_argument("--include-local-logs", action="store_true",
                    help="also scan outputs/<tag>/logs (copies of Colab-trained runs)")
    ap.add_argument("--out-dir", default=str(C.MAIN / "artifacts"))
    args = ap.parse_args()
    roots = args.runs_root or [str(C.RUNS)]
    if args.include_local_logs:
        roots += [str(p) for p in Path(C.OUTPUTS).glob("*/logs")]
    manifests = load_manifests(C.OUTPUTS)
    data_hashes = {}
    for tag in C.FAMILIES:
        for f in C.leaf(tag).glob("synonyms_train*.jsonl"):
            data_hashes[str(f.resolve())] = C.sha256(f)
    rows = [describe(run, manifests, data_hashes) for run in find_runs(roots)]
    rows.sort(key=lambda r: (r["model_tag"], r["objective"], str(r["gamma"]), str(r["seed"]),
                             r["checkpoint_path"]))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(out / "run_manifest_existing.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    labelled = {p for p in manifests}
    found = {r["checkpoint_path"] for r in rows}
    report = {
        "runs_found": len(rows),
        "roots": roots,
        "by_model_objective": Counter(f"{r['model_tag']} {r['objective']}" for r in rows),
        "seeds_by_model_objective_gamma": seed_report(rows),
        "settings_mismatch_vs_paper": mismatches(rows),
        "smoke_runs_ignored": sorted(r["checkpoint_path"] for r in rows if "/smoke" in r["checkpoint_path"]),
        # A manifest path with no adapter_config.json is either gone, or a FULL-model
        # checkpoint (the Iyer replication trains whole models): say which.
        "manifest_paths_missing_on_disk": sorted(p for p in labelled - found if not Path(p).exists())
        if args.runs_root is None else "n/a (custom roots)",
        "manifest_paths_full_model_not_lora": sorted(
            p for p in labelled - found if (Path(p) / "config.json").is_file())
        if args.runs_root is None else "n/a (custom roots)",
        "train_file_hash_differs_from_disk": [r["checkpoint_path"] for r in rows
                                              if r["train_file_matches_disk"] is False],
        "lora_scaling_values": Counter(str(r["lora_scaling_alpha_over_r"]) for r in rows),
        "code_defaults_not_recorded_per_run": CODE_DEFAULTS,
        "gpu_minutes_backfilled_training_only": round(sum(
            float(r["train_runtime_min"]) for r in rows if r.get("train_runtime_min") not in ("", None)), 1),
    }
    C.write_json(out / "run_audit_report.json", report)
    print(f"{len(rows)} runs -> {out / 'run_manifest_existing.csv'}")
    print(json.dumps({k: v for k, v in report.items() if k != "seeds_by_model_objective_gamma"},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
