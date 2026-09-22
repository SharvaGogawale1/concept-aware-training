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
a = p.parse_args()
runs = Path(a.base).resolve() / "concept_aware" / "runs" / a.model_tag
wanted = {"ntp_seed42": ("ntp", "lambda_0.0_beta_0.0"),
          "zhang_seed42": ("zhang_marginal", "lambda_1.0_beta_0.0"),
          "randomized_seed42": ("randomized", "lambda_0.25_beta_0.0")}
manifest, missing = {}, []
for label, (method, value) in wanted.items():
    path = runs / method / "seed_42" / value
    if (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file():
        manifest[label] = str(path)
    else:
        missing.append(str(path))
if missing:
    raise SystemExit("missing adapters (need adapter_config.json + adapter_model.safetensors):\n  " + "\n  ".join(missing))
out = Path(a.base).resolve() / "outputs" / a.model_tag / "run_manifests" / "task15.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2))
print(f"wrote {out}\n" + "\n".join(f"  {k}: {v}" for k, v in manifest.items()))
