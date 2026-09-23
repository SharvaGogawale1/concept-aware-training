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
parser.add_argument("--stage", choices=["hybrid", "verified", "confirm15"], default="hybrid",
                    help="hybrid: the frontier screen (default); verified: the verified-supervision arms; "
                         "confirm15: Task 15's other seeds (use on reproducibilty_15_<tag>.ipynb)")
parser.add_argument("--hybrid-arms", default="",
                    help='hybrid stage only: train only these grid arms, e.g. "uniform:0.125,mass:0.125"')
parser.add_argument("--selected-hybrid", default=None,
                    help='hybrid stage only: the selected arm, e.g. "uniform:0.125"; with --multiseed trains its seeds')
parser.add_argument("--multiseed", action="store_true", help="add seeds 123 and 2024 (hybrid / confirm15)")
parser.add_argument("--result-tag", default="",
                    help='hybrid stage only: score into task15b_screen<tag>, e.g. "_confirm"')
parser.add_argument("--eval-arms", default="",
                    help="hybrid stage only: with --result-tag, only these 15b arm labels enter the table")
parser.add_argument("--no-screen", action="store_true",
                    help="hybrid stage only: never turn RUN_SCREEN on (a machine without the screen arms)")
parser.add_argument("--confirm-arms", default="",
                    help='confirm15 only: families that get the extra seeds, e.g. "zhang" ("" = all)')
parser.add_argument("--smoke-steps", type=int, default=0,
                    help="verified stage only: >0 trains ~that many steps per arm and prints magnitudes")
parser.add_argument("--smoke-only", action="store_true",
                    help="verified stage only: stop after the smoke run instead of training")
parser.add_argument("--gamma", type=float, default=None,
                    help="verified stage only: continuity arm's gamma (default per model: llama .125, qwen .0625)")
parser.add_argument("--arms", default="",
                    help='verified stage only: train only these arms (comma list; "" = all)')
parser.add_argument("--lambda", dest="lam", type=float, default=1.0,
                    help="verified stage only: weight on the slot objective (smoke rule may change it)")
parser.add_argument("--no-eval", action="store_true",
                    help="train and stop (a parallel training process); evaluate in a later pass")
args = parser.parse_args()
if args.stage == "hybrid":
    FLAGS.update({"HYBRID_ARMS": f'"{args.hybrid_arms}"',
                  "RESULT_TAG": f'"{args.result_tag}"', "EVAL_ARMS": f'"{args.eval_arms}"',
                  "SELECTED_HYBRID": f'"{args.selected_hybrid}"' if args.selected_hybrid else "None",
                  "RUN_MULTISEED": "True" if args.multiseed else "False",
                  "RUN_EVAL": "False" if args.no_eval else "True"})
if args.stage == "confirm15":
    # Task 15's control cell: only the extra seeds of the chosen families, no screen,
    # no evaluation here (15b scores them as baselines of its own table).
    FLAGS.clear()
    FLAGS.update({"RUN_DATA": "False", "RUN_SMOKE": "False", "RUN_SCREEN": "False",
                  "RUN_CONFIRM": "True", "RUN_MULTISEED": "True",
                  "CONFIRM_ARMS": f'"{args.confirm_arms}"',
                  "RUN_EVAL": "False" if args.no_eval else "True"})
if args.stage == "verified":
    # The verified pass must not relaunch the hybrid screen: those arms resume by
    # manifest, and RUN_HYBRID=True would otherwise retrain nothing but still add
    # the five hybrid checkpoints to a table this stage keeps separate on purpose.
    # And never the screen either: on a machine that has no screen arms for this
    # model (Llama on the server) the manifest check would otherwise turn
    # RUN_SCREEN on and spend hours training eight arms this stage never reads.
    FLAGS.update({"RUN_HYBRID": "False", "RUN_SCREEN": "False", "RUN_VERIFIED": "True",
                  "VERIFIED_SMOKE_STEPS": str(args.smoke_steps),
                  "VERIFIED_SMOKE_ONLY": "True" if args.smoke_only else "False",
                  "VERIFIED_ARMS": f'"{args.arms}"',
                  "VERIFIED_LAMBDA": str(args.lam),
                  "RUN_EVAL": "False" if args.no_eval else "True",
                  "VERIFIED_GAMMA": str(args.gamma if args.gamma is not None
                                        else (0.125 if "llama" in args.model_tag else 0.0625))})

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

if missing_any and args.stage == "hybrid" and not args.no_screen:
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
    if re.match(r"^\s*(GPU_ID|MODEL|HF_TOKEN|RUN_|SELECTED_|SPACY_|VERIFIED_|HYBRID_|CONFIRM_|RESULT_|EVAL_)", line):
        print("   " + line.rstrip())
