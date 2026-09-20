#!/usr/bin/env python3
"""Set the Task 15b control-cell flags for one stage, and check what resume needs.

The control cell is edited in place, idempotently, so the notebook file records
what ran.  RUN_SCREEN is not a preference: the eight screen arms re-enter this
pass through `load_runs("task15b")` whether or not it is set, so it is False when
every adapter that manifest names is on disk and True when one is missing and
must be retrained.  The check is printed either way.

    python set_15b_flags.py reproducibilty_15b.ipynb --gpu 3 [--base .]
"""
import argparse, json, re
from pathlib import Path

FLAGS = {"RUN_DATA": "False", "RUN_SCREEN": "False", "RUN_HYBRID": "True",
         "RUN_EVAL": "True", "RUN_CONFIRM": "False", "RUN_MULTISEED": "False",
         "RUN_NEGATIVE_CONTROLS": "False", "SELECTED_HYBRID": "None",
         "SELECTED_ALPHA": "0.5", "SELECTED_BETA": "1.0"}

parser = argparse.ArgumentParser()
parser.add_argument("notebook")
parser.add_argument("--gpu")
parser.add_argument("--base", default=".", help="CONCEPT_BASE: holds outputs/")
parser.add_argument("--model-tag", default="qwen3-1.7b-base")
args = parser.parse_args()

# ---- what is already trained -------------------------------------------------
manifests = Path(args.base) / "outputs" / args.model_tag / "run_manifests"
missing_any = False
for name in ("task15", "task15b"):
    path = manifests / f"{name}.json"
    if not path.is_file():
        print(f"{name}.json: ABSENT at {path}")
        # Task 15b's absence means the screen arms cannot be reloaded by manifest.
        missing_any = missing_any or name == "task15b"
        continue
    runs = json.loads(path.read_text())
    gone = [k for k, v in runs.items() if not (Path(v) / "adapter_config.json").is_file()]
    print(f"{name}.json: {len(runs)} arms, {len(gone)} adapters missing"
          + ("" if not gone else "  -> " + ", ".join(gone)))
    missing_any = missing_any or (bool(gone) and name == "task15b")

if missing_any:
    FLAGS["RUN_SCREEN"] = "True"
    print("\n=> RUN_SCREEN=True: a screen arm must be retrained to re-enter the table.")
else:
    print("\n=> RUN_SCREEN=False: every screen arm reloads from the manifest.")

# ---- edit the control cell ---------------------------------------------------
notebook = json.loads(Path(args.notebook).read_text())
control = [c for c in notebook["cells"] if c["cell_type"] == "code"
           and "THE ONLY CELL YOU EDIT" in "".join(c["source"])]
assert len(control) == 1, f"expected one control cell, found {len(control)}"
cell = control[0]

changed = []
for i, line in enumerate(cell["source"]):
    match = re.match(r"^(\s*)([A-Z_]+)(\s*)=(\s*)(\S+)", line)
    if not match:
        continue
    _, name, _, gap, old = match.groups()
    new = f'"{args.gpu}"' if (args.gpu and name == "GPU_ID") else FLAGS.get(name)
    if new is None or old == new:
        continue
    cell["source"][i] = line.replace(f"={gap}{old}", f"={gap}{new}", 1)
    changed.append(f"  {name}: {old} -> {new}")

Path(args.notebook).write_text(json.dumps(notebook, indent=1) + "\n")
print("\nchanged:" if changed else "\nnothing to change (already set)")
print("\n".join(changed))
print("\nflags now:")
for line in cell["source"]:
    if re.match(r"^\s*(GPU_ID|MODEL|HF_TOKEN|RUN_|SELECTED_|SPACY_)", line):
        print("   " + line.rstrip())
