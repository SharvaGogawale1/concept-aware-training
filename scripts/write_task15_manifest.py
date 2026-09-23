#!/usr/bin/env python3
"""Write outputs/<tag>/run_manifests/task15.json for a model whose Task 15 adapters were
copied onto this machine rather than trained here (Llama's live on Drive).

The 15b notebooks read this manifest for the NTP, Zhang and randomized rows every table
is compared against.  Each adapter directory must hold adapter_config.json; a missing one
is an error, not a silent gap, because a table without its baseline changes the claim.

    python scripts/write_task15_manifest.py --model-tag llama-3.2-1b --base /path/to/task15
"""
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model-tag", required=True)
p.add_argument("--base", default=".")
p.add_argument("--force", action="store_true", help="replace a manifest that has other entries (backed up first)")
a = p.parse_args()
runs = Path(a.base).resolve() / "concept_aware" / "runs" / a.model_tag
wanted = {"ntp_seed42": ("ntp", 42, "lambda_0.0_beta_0.0"),
          "zhang_seed42": ("zhang_marginal", 42, "lambda_1.0_beta_0.0"),
          "randomized_seed42": ("randomized", 42, "lambda_0.25_beta_0.0")}
# Present when copied, absent otherwise: the seed-matched Zhang references the 15b
# confirmation pairs against, and the other three-seed baselines if they came too.
optional = {f"{family}_seed{seed}": (method, seed, value)
            for seed in (123, 2024)
            for family, method, value in (("zhang", "zhang_marginal", "lambda_1.0_beta_0.0"),
                                          ("ntp", "ntp", "lambda_0.0_beta_0.0"),
                                          ("randomized", "randomized", "lambda_0.25_beta_0.0"),
                                          ("randomized_lambda1.0", "randomized", "lambda_1.0_beta_0.0"),
                                          ("augmented_ntp", "augmented_ntp", "lambda_0.0_beta_0.0"))}
def present(path):
    return (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file()
manifest, missing = {}, []
for label, (method, seed, value) in wanted.items():
    path = runs / method / f"seed_{seed}" / value
    if present(path):
        manifest[label] = str(path)
    else:
        missing.append(str(path))
for label, (method, seed, value) in optional.items():
    path = runs / method / f"seed_{seed}" / value
    if present(path):
        manifest[label] = str(path)
if missing:
    raise SystemExit("missing adapters (need adapter_config.json + adapter_model.safetensors):\n  " + "\n  ".join(missing))
out = Path(a.base).resolve() / "outputs" / a.model_tag / "run_manifests" / "task15.json"
out.parent.mkdir(parents=True, exist_ok=True)
if out.is_file():
    existing = json.loads(out.read_text())
    extra = sorted(set(existing) - set(manifest))
    if extra and not a.force:
        raise SystemExit(f"{out} already lists {extra}; this would drop them.  Pass --force (a backup is kept).")
    if existing != manifest:
        import time
        backup = out.with_name(f"task15.json.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        backup.write_text(json.dumps(existing, indent=2))
        print("backed up the previous manifest to", backup)
out.write_text(json.dumps(manifest, indent=2))
print(f"wrote {out}\n" + "\n".join(f"  {k}: {v}" for k, v in manifest.items()))
