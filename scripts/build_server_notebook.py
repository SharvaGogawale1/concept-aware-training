#!/usr/bin/env python3
"""Derive the server notebook from the Colab one.

Everything outside the first two cells refers to Drive only through function and
variable NAMES, never through hardcoded paths.  So the server variant replaces
the setup and bootstrap cells, renames those symbols, and reuses the experiment
body verbatim -- which keeps the two notebooks from drifting apart.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
SOURCE = NB_DIR / "research_tasks_15_zhang_reproduction.ipynb"
TARGET = NB_DIR / "reproducibilty_15.ipynb"

def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text): return {"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}

INSTALL = r'''
# Install once per environment.  transformers is pinned LAST and BELOW 4.58 on
# purpose: upstream's extractor reuses a prefix KV cache through
# DynamicCache.from_legacy_cache, which transformers removed in v5.
%pip install -q accelerate peft bitsandbytes datasets spacy "mteb>=1.12" nltk scipy scikit-learn seaborn pandas pytest wandb
%pip install -q "transformers>=4.51,<4.58"
!python -m spacy download en_core_web_sm
import transformers, torch
print("transformers", transformers.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available())
'''

SETUP = r'''
import os
# Pick a free GPU BEFORE torch initialises CUDA.  nvidia-smi showed 2 and 3 busy.
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from pathlib import Path
import hashlib, json, re, shutil, subprocess, sys, torch

# Every clone, dataset, checkpoint and result lives under BASE, so the whole
# experiment is one directory to archive or copy off the server.
BASE = Path(os.environ.get("CONCEPT_BASE", Path.cwd())).resolve()
WORK = BASE / "concept_aware"
MAIN = WORK / "concept-aware-training"     # our repo: patch, evaluators, scripts
EXT = WORK / "learning-concepts"           # upstream, pinned
DATA = WORK / "data"                       # CONCEPT_DATA_ROOT
RUNS = WORK / "runs"                       # adapters (small at r=4, kept)
OUTPUTS = BASE / "outputs"                 # everything you download for analysis

BASE_MODEL = "meta-llama/Llama-3.2-1B"
PRIMARY_SEED = 42
# One seed screens the pipeline and shows the direction of every effect, but it
# CANNOT support a claim: the pre-registered rule needs all three to agree in
# sign.  Flip to True for the reportable run; finished arms are skipped.
RUN_MULTISEED = False
SEEDS = [PRIMARY_SEED] + ([123, 2024] if RUN_MULTISEED else [])
UPSTREAM_COMMIT = "b1d414143d11c8ed988b4cccbb06626cc8272bbe"

os.environ["HF_HOME"] = str(WORK / "hf_cache")

RUN_DATA = True
RUN_SMOKE = True
RUN_SCREEN = True
RUN_CONFIRM = False
RUN_EVAL = True
# Skip any run whose artefacts already exist.  A killed job resumes from here.
RESUME_FINISHED_RUNS = True

# Extraction precision.  Upstream inherits use_4bit=True from TrainingConfig,
# where it exists for QLoRA TRAINING.  Extraction runs ~94 small forwards per
# sequence; measured on an L4, bf16 was ~25% faster and avoids quantisation
# noise in the top-100 pool and the 0.75 cosine threshold the method depends on.
EXTRACT_4BIT = False
# Full paper spec: 10,000 sequences split 8000/1000/1000.  merge_synonym_parts
# hard-fails unless the split sizes sum to the rows the shards actually cover.
EXTRACT_SEQUENCES = 10000
SPLIT_TRAIN = int(EXTRACT_SEQUENCES * 0.8)
SPLIT_VAL = SPLIT_TEST = int(EXTRACT_SEQUENCES * 0.1)

_BAR = re.compile(r"\b(\d+)/(\d+)\s*\[")   # tqdm counter, e.g. "  200/1000 ["
PROGRESS_EVERY = 100

def run(argv, cwd=None, env=None):
    """Run a child process, streaming its output and keeping the tail on failure."""
    argv = list(map(str, argv))
    print("+", " ".join(argv), flush=True)
    merged = os.environ.copy()
    merged.update({"CONCEPT_DATA_ROOT": str(DATA),
                   "CONCEPT_CHECKPOINT_ROOT": str(RUNS),
                   "CONCEPT_RESULTS_ROOT": str(OUTPUTS)})
    if env: merged.update(env)
    process = subprocess.Popen(argv, cwd=cwd, env=merged, text=True, bufsize=1,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail = []
    for line in process.stdout:
        line = line.replace("\r", "")
        tail.append(line)
        del tail[:-40]
        hit = _BAR.search(line)
        if hit:
            done, total = int(hit.group(1)), int(hit.group(2))
            if done % PROGRESS_EVERY and done != total:
                continue
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(
            f"command failed with exit code {code}\n  {' '.join(argv)}\n"
            f"--- last {len(tail)} lines of its output ---\n{''.join(tail)}")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sync_small_artifacts(source, label):
    destination = OUTPUTS / label
    destination.mkdir(parents=True, exist_ok=True)
    for path in Path(source).rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".png", ".log"}:
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

def audit_outputs():
    """Keep OUTPUTS downloadable: reports only, no multi-GB weights."""
    forbidden = {"pytorch_model.bin", "model.safetensors", "optimizer.pt",
                 "scheduler.pt", "scaler.pt", "rng_state.pth"}
    found = [str(p) for p in OUTPUTS.rglob("*")
             if p.name in forbidden or p.name.startswith("checkpoint-")]
    assert not found, f"full-weight/optimizer artifacts reached OUTPUTS: {found}"

# The server filesystem persists, so the Colab Drive round-trip is unnecessary.
# These keep the same names and call sites as the Colab notebook, which is what
# lets the two share every experiment cell below.
def cache_dataset(leaf): pass

def restore_dataset(leaf):
    return (Path(leaf) / "synonyms_train.jsonl").is_file()

def cache_adapter(path): return Path(path)

def restore_adapter(path):
    return (Path(path) / "adapter_config.json").is_file()

RUN_MANIFEST = OUTPUTS / "run_manifests"

def save_runs(runs, name):
    RUN_MANIFEST.mkdir(parents=True, exist_ok=True)
    (RUN_MANIFEST / f"{name}.json").write_text(
        json.dumps({k: str(v) for k, v in runs.items()}, indent=2))

def load_runs(name):
    path = RUN_MANIFEST / f"{name}.json"
    return {} if not path.is_file() else {k: Path(v) for k, v in json.loads(path.read_text()).items()}

def restore_all(runs):
    live = {}
    for label, path in runs.items():
        if restore_adapter(path):
            live[label] = Path(path)
        else:
            print("missing adapter, dropping from this pass:", label)
    return live

def eval_done(marker):
    return Path(marker).is_file() and RESUME_FINISHED_RUNS

def assert_under_base(path):
    resolved = Path(path).resolve()
    assert str(resolved).startswith(str(BASE)), f"{resolved} escapes {BASE}"

for directory in (WORK, DATA, RUNS, OUTPUTS):
    directory.mkdir(parents=True, exist_ok=True)
print("BASE    ", BASE)
print("OUTPUTS ", OUTPUTS)
print("GPU     ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
'''

BOOTSTRAP = r'''
# Both clones are required: MAIN carries the patch, the evaluators and the
# summary script; EXT is the pinned upstream the patch applies to.
if not MAIN.exists():
    run(["git", "clone", "https://github.com/SharvaGogawale1/concept-aware-training.git", MAIN])
else:
    run(["git", "-C", str(MAIN), "pull", "--ff-only"])
if not EXT.exists():
    run(["git", "clone", "https://github.com/christine-zhang1/learning-concepts.git", EXT])
run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=EXT)

patch_file = MAIN / "external" / "learning-concepts.patch"
assert patch_file.exists(), f"{patch_file} missing; push it before running here."
check = subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=EXT)
if check.returncode == 0:
    run(["git", "apply", str(patch_file)], cwd=EXT)
else:
    reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch_file)], cwd=EXT)
    assert reverse.returncode == 0, "External checkout is neither clean nor exactly patched."

run([sys.executable, "-m", "pip", "install", "-q", "-e", str(EXT), "--no-deps"])

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

run([sys.executable, MAIN / "builddataset/verify_task14_data.py",
     "--repo_root", MAIN, "--download_missing",
     "--report_json", OUTPUTS / "external_benchmark_integrity.json"], cwd=MAIN)
(OUTPUTS / "environment_freeze.txt").write_text(
    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))

# Gated model.  Either export HF_TOKEN before launching Jupyter, or run
# `huggingface-cli login` once on this machine.
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    print("logged in from HF_TOKEN")
else:
    print("No HF_TOKEN set; relying on a previous `huggingface-cli login`.")
'''

RENAME = [
    ("DRIVE_RESULTS", "OUTPUTS"),
    ("cache_dataset_to_drive", "cache_dataset"),
    ("restore_dataset_from_drive", "restore_dataset"),
    ("cache_adapter_to_drive", "cache_adapter"),
    ("restore_adapter_from_drive", "restore_adapter"),
    ("audit_no_drive_weights", "audit_outputs"),
    ("assert_ephemeral", "assert_under_base"),
    ("Colab disconnect", "killed job"),
    ("Drive-side manifest", "manifest on disk"),
    ("/content is wiped between sessions; pull the generated concept data back rather\n"
     "# than paying the multi-hour regeneration again.",
     "The server filesystem persists, so this is a cheap existence check."),
    ("adapters remain under /content", "adapters stay under RUNS"),
]

source = json.loads(SOURCE.read_text())
cells = [md("# Task 15 (server) — reproduce concept training before extending it\n\n"
            "Everything lives under the directory this notebook runs from: clones in\n"
            "`concept_aware/`, results in `outputs/`.  Set `GPU_ID` in the setup cell to a\n"
            "free GPU before running."),
         code(INSTALL), code(SETUP), code(BOOTSTRAP)]

for index, cell in enumerate(source["cells"]):
    if index in (0, 1, 2):          # title, Colab setup, Colab bootstrap
        continue
    text = "".join(cell["source"])
    for old, new in RENAME:
        text = text.replace(old, new)
    cells.append(code(text) if cell["cell_type"] == "code" else md(text))

TARGET.write_text(json.dumps({
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}, indent=1) + "\n")
print(f"wrote {TARGET} ({len(cells)} cells)")
