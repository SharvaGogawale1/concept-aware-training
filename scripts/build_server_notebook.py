#!/usr/bin/env python3
"""Derive the server notebooks from the Colab ones.

Everything outside the first two cells refers to Drive only through function and
variable NAMES, never through hardcoded paths.  So each server variant replaces
the setup and bootstrap cells, renames those symbols, and reuses the experiment
body verbatim -- which keeps the twins from drifting apart.

Two notebooks are emitted, and they are meant to run against the SAME
CONCEPT_BASE: 15b reads 15's manifest and results directory for its comparators,
so pointing them at one directory is what lets the objective arms be scored
against reproduction arms that are already trained.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"

def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text): return {"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}

INSTALL = r'''
# Install once per environment.  transformers is pinned LAST and BELOW 4.58 on
# purpose: upstream's extractor reuses a prefix KV cache through
# DynamicCache.from_legacy_cache, which transformers removed in v5.
%pip install -q accelerate peft bitsandbytes datasets spacy "mteb>=1.12" nltk scipy scikit-learn seaborn pandas pytest wandb
%pip install -q "transformers>=4.51,<4.58"
# torchao is a Colab-era fix: peft raises on torchao < 0.16 from inside
# PeftModel.from_pretrained.  But a CURRENT torchao evaluates torch.int1 at
# import, which torch < 2.6 does not define, and transformers imports torchao
# unconditionally whenever it is installed -- so on an older torch a mismatched
# torchao makes EVERY model unloadable, with the traceback pointing at the model
# class rather than at torchao.  Install it only where it helps; remove it
# otherwise, which is safe because peft only needs it when it is present.
import subprocess, sys, torch
_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
if _version >= (2, 6):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao>=0.16"], check=False)
    print("torchao: installed for torch", torch.__version__)
else:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"], check=False)
    print("torchao: removed, torch", torch.__version__, "predates torch.int1")
# cupy backs spacy.require_gpu(); without it thinc raises and SPACY_GPU must be
# set False.  thinc reads cupy's presence at import, so install before spacy loads.
%pip install -q cupy-cuda12x
!python -m spacy download en_core_web_sm
import transformers, torch
print("transformers", transformers.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available())
'''

GPU_PICK = r'''
# Which cards are free.  Pick one with spare memory and no other process, then
# name it in the control cell below -- BEFORE anything here touches CUDA.
!nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
'''

# One control cell per notebook.  They are written out rather than generated
# from a shared template: a reader of the notebook has to be able to see every
# knob and what it costs, and the gates mean different things in the two files.
CONTROL_15 = r'''
# ==== THE ONLY CELL YOU EDIT.  Set these, then Run All. ======================
import os

GPU_ID   = "1"                 # from the table above
MODEL    = "Qwen/Qwen3-1.7B-Base"   # ONE model per pass; then "Qwen/Qwen3-4B-Base".
                               # The -Base suffix is NOT cosmetic: a bare Qwen3 name is the
                               # post-trained chat model, and this study continues PRE-training
                               # and compares against Llama-3.2-1B, a base model.  (Qwen2.5 named
                               # them the other way round, which is how this gets picked wrong.)
HF_TOKEN = ""                  # gated models only (Llama).  Qwen3 is open: leave empty.
                               # If you do paste one, CLEAR IT BEFORE SAVING THIS FILE.

RUN_DATA    = True    # extract concept sets from C4 and build the splits (the long stage)
RUN_SMOKE   = True    # ~20-step objective check before any long training starts
RUN_SCREEN  = True    # train the seven arms at seed 42
RUN_CONFIRM = False   # retrain the headline arms across SEEDS; Qwen is a seed-42
                      # replication, so off.  It needs RUN_MULTISEED too -- on its own
                      # it loops over [42] alone and every arm is already finished.
RUN_MULTISEED = False # add seeds 123 and 2024 to SEEDS
RUN_EVAL    = True    # score every checkpoint: SWORDS, STS, perplexity, bm-semlex
SPACY_GPU   = True    # 8.5x faster extraction; needs cupy-cuda12x, installed below

# Everything below reads these from the environment, which is also how they reach
# each child process.  Set any value to None to defer to a variable exported in
# the shell instead -- that is what the nbconvert commands in the cell above use.
for _name, _value in {"GPU_ID": GPU_ID, "CONCEPT_MODEL": MODEL, "HF_TOKEN": HF_TOKEN or None,
                      "RUN_DATA": RUN_DATA, "RUN_SMOKE": RUN_SMOKE, "RUN_SCREEN": RUN_SCREEN,
                      "RUN_CONFIRM": RUN_CONFIRM, "RUN_EVAL": RUN_EVAL,
                      "RUN_MULTISEED": RUN_MULTISEED, "SPACY_GPU": SPACY_GPU}.items():
    if _value is not None:
        os.environ[_name] = ("1" if _value else "0") if isinstance(_value, bool) else str(_value)

# CUDA_VISIBLE_DEVICES has to be set before any torch import initialises the
# driver, and the install cell below imports torch to decide about torchao.  The
# setup cell sets it too, from GPU_ID -- but by then the choice can already be
# locked in, and the run would quietly land on card 0.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.setdefault("GPU_ID", "1")
print("GPU", os.environ["CUDA_VISIBLE_DEVICES"],
      "|", os.environ.get("CONCEPT_MODEL", "(default)"),
      "| HF token", "set" if os.environ.get("HF_TOKEN") else "none",
      "| stages:", " ".join(stage for stage in ("RUN_DATA", "RUN_SMOKE", "RUN_SCREEN",
                                                "RUN_CONFIRM", "RUN_EVAL")
                            if os.environ.get(stage) == "1"))
'''

CONTROL_15B = r'''
# ==== THE ONLY CELL YOU EDIT.  Set these, then Run All. ======================
import os

GPU_ID   = "1"                      # from the table above
MODEL    = "Qwen/Qwen3-1.7B-Base"   # must be a model whose Task 15 pass already
                                    # finished: the arms here are scored against its
                                    # NTP, Zhang and randomized adapters.
HF_TOKEN = ""                       # gated models only (Llama).  Qwen3 is open.
                                    # If you do paste one, CLEAR IT BEFORE SAVING.

# Alpha was selected on Llama-3.2-1B's C4 validation and beta on SWORDS dev.
# Setting them here TRANSFERS those values instead of tuning again on this model.
# That is the intended replication: re-tuning per model would read this model's
# data for selection and make the second family a second tuning round rather
# than a test.  Say so in the write-up.  Leave both None to run the alpha screen
# and read the decision cell's gate, the way the first model did it.
SELECTED_ALPHA = 0.5
SELECTED_BETA  = 1.0

RUN_DATA    = True    # mine WordNet hard negatives and write the negatives view (minutes)
RUN_SCREEN  = True    # alpha sweep {0.25,0.5,1.0}; then, because SELECTED_ALPHA is set,
                      # the inclusive-set ablation, the beta screen {0.25,0.5,1.0} and the
                      # alpha=0.75 frontier point -- 8 arms, ~30 min each on an A40
RUN_CONFIRM = False   # the locked arms across SEEDS.  Needs RUN_MULTISEED for a real
                      # three-seed confirmation; on its own it loops over [42], which the
                      # screen has already trained.
RUN_MULTISEED = False # add seeds 123 and 2024 to SEEDS
RUN_HYBRID  = False   # Zhang + alternative-only auxiliary term (5 arms) and the two
                      # negative-quality controls, seed 42.  Off until the 1B gate on
                      # Colab has picked one; then set SELECTED_HYBRID and run only that.
SELECTED_HYBRID = None  # e.g. "within_kl:0.5" -- transferred from the 1B gate, never tuned here
RUN_NEGATIVE_CONTROLS = False  # clean/fragments negative diagnostic; not method selection
RUN_VERIFIED = False  # six arms from synonyms_train_verified.jsonl, one slot objective each,
                      # scored in their own directory against Task 15's NTP and Zhang
VERIFIED_SMOKE_STEPS = 0     # >0: ~that many steps per arm, print first-step magnitudes, train nothing else
VERIFIED_LAMBDA = 1.0        # weight on the slot objective for pool/rank/list arms -- declared, not tuned
VERIFIED_GAMMA  = 0.0625     # the continuity arm's gamma from the dev frontier: Qwen .0625, Llama .125
RUN_EVAL    = True    # score these arms AND the Task 15 comparators: SWORDS with paired
                      # bootstrap intervals, STS, perplexity, concept sets, bm-semlex
SPACY_GPU   = True    # only matters if this model still needs extraction

# Everything below reads these from the environment, which is also how they reach
# each child process.  Set any value to None to defer to a variable exported in
# the shell instead -- that is what the nbconvert commands in the cell above use.
for _name, _value in {"GPU_ID": GPU_ID, "CONCEPT_MODEL": MODEL, "HF_TOKEN": HF_TOKEN or None,
                      "SELECTED_ALPHA": SELECTED_ALPHA, "SELECTED_BETA": SELECTED_BETA,
                      "RUN_DATA": RUN_DATA, "RUN_SCREEN": RUN_SCREEN,
                      "RUN_HYBRID": RUN_HYBRID, "SELECTED_HYBRID": SELECTED_HYBRID,
                      "RUN_NEGATIVE_CONTROLS": RUN_NEGATIVE_CONTROLS,
                      "RUN_VERIFIED": RUN_VERIFIED, "VERIFIED_SMOKE_STEPS": VERIFIED_SMOKE_STEPS,
                      "VERIFIED_LAMBDA": VERIFIED_LAMBDA, "VERIFIED_GAMMA": VERIFIED_GAMMA,
                      "RUN_CONFIRM": RUN_CONFIRM, "RUN_EVAL": RUN_EVAL,
                      "RUN_MULTISEED": RUN_MULTISEED, "SPACY_GPU": SPACY_GPU}.items():
    if _value is not None:
        os.environ[_name] = ("1" if _value else "0") if isinstance(_value, bool) else str(_value)

# CUDA_VISIBLE_DEVICES has to be set before any torch import initialises the
# driver, and the install cell below imports torch to decide about torchao.  The
# setup cell sets it too, from GPU_ID -- but by then the choice can already be
# locked in, and the run would quietly land on card 0.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.setdefault("GPU_ID", "1")
print("GPU", os.environ["CUDA_VISIBLE_DEVICES"],
      "|", os.environ.get("CONCEPT_MODEL", "(default)"),
      "| alpha", os.environ.get("SELECTED_ALPHA", "(screen)"),
      "| beta", os.environ.get("SELECTED_BETA", "(screen)"),
      "| stages:", " ".join(stage for stage in ("RUN_DATA", "RUN_SCREEN", "RUN_HYBRID", "RUN_VERIFIED",
                                                "RUN_CONFIRM", "RUN_EVAL")
                            if os.environ.get(stage) == "1"))
'''

SETUP = r'''
import os
# Set in the control cell above; re-applied here so this cell stands alone when
# it is re-run on its own, and so the headless path (GPU_ID=2 jupyter nbconvert
# --execute ...) works with the control cell's GPU_ID set to None.
GPU_ID = os.environ.get("GPU_ID", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from pathlib import Path
import datetime, hashlib, json, re, shutil, subprocess, sys, torch

# Every clone, dataset, checkpoint and result lives under BASE, so the whole
# experiment is one directory to archive or copy off the server.
def _notebook_base():
    """Directory the notebook itself lives in, so `outputs/` lands beside it.

    CONCEPT_BASE wins when set.  Otherwise prefer JPY_SESSION_NAME, which Jupyter
    sets to the notebook's own path: the working directory is the notebook's
    directory only when the kernel happened to start there, so `jupyter nbconvert
    --execute` invoked from anywhere else would scatter a second outputs/ tree
    next to wherever it was run from.
    """
    override = os.environ.get("CONCEPT_BASE")
    if override:
        return Path(override)
    session = os.environ.get("JPY_SESSION_NAME", "")
    if session.endswith(".ipynb") and Path(session).parent.is_dir():
        return Path(session).parent
    return Path.cwd()

BASE = _notebook_base().resolve()
WORK = BASE / "concept_aware"
MAIN = WORK / "concept-aware-training"     # our repo: patch, evaluators, scripts
EXT = WORK / "learning-concepts"           # upstream, pinned
DATA = WORK / "data"                       # CONCEPT_DATA_ROOT
RUNS = WORK / "runs"                       # adapters (small at r=4, kept)
OUTPUTS = BASE / "outputs"                 # everything you download for analysis

# ONE model per pass.  Every artefact below is keyed on the model tag and the
# resume logic reads finished work back by path, so the model is selected in the
# control cell rather than looped over here -- which is also what lets two models
# share the machine without sharing a GPU.
BASE_MODEL = os.environ.get("CONCEPT_MODEL", "Qwen/Qwen3-1.7B-Base")
# Every per-model artifact is keyed on this tag.  Two models must never share a
# path: adapters resume by path, so a collision hands one model's weights to
# another and the run still looks like it succeeded.  conceptlib.paths derives
# the same tag the same way, so the data directories line up with these.
MODEL_TAG = BASE_MODEL.split("/")[-1].lower()
# combined.jsonl depends only on the tokenizer, so a model sharing a vocabulary
# with one already extracted produces a byte-identical file.  Naming the donor
# skips the spaCy POS pass; the vocabularies are compared before the copy and the
# run aborts if they differ.  Each pair shares a tokenizer WITHIN its family and
# never across families, which is why this is a table and not a size heuristic.
VOCAB_DONOR = {"Qwen/Qwen2.5-3B": "Qwen/Qwen2.5-1.5B",
               "Qwen/Qwen3-4B-Base": "Qwen/Qwen3-1.7B-Base",
               "Qwen/Qwen3-4B": "Qwen/Qwen3-1.7B",
               "meta-llama/Llama-3.2-3B": "meta-llama/Llama-3.2-1B"}
REUSE_CONTENT_WORDS_FROM = os.environ.get("CONCEPT_REUSE_FROM") or VOCAB_DONOR.get(BASE_MODEL)

# outputs/<model tag>/{data_audit.json,results,logs,run_manifests}, plus one
# shared directory for the two model-independent reports.  Unlike the Colab
# notebook there is no legacy unsuffixed path to preserve here, so EVERY model
# gets its own directory -- including the first one.
MODEL_OUT = OUTPUTS / MODEL_TAG
RESULT_DIR = MODEL_OUT / "results"
LOG_DIR = MODEL_OUT / "logs"
SHARED = OUTPUTS / "shared"
DATA_AUDIT = MODEL_OUT / "data_audit.json"
PRIMARY_SEED = 42
# One seed screens the pipeline and shows the direction of every effect, but it
# CANNOT support a claim: the pre-registered rule needs all three to agree in
# sign.  Flip to True for the reportable run; finished arms are skipped.
def _flag(name, default):
    """Read a run flag from the environment, defaulting to the value here.

    The control cell writes the flags into the environment rather than binding
    them here, so the same file also runs headless -- `CONCEPT_MODEL=...
    RUN_DATA=1 jupyter nbconvert --execute` drives one stage under tmux, which
    survives a dropped VPN, and the notebook stays reproducible either way.
    """
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")

# Requires cupy (installed below).  Verified output-identical to CPU spaCy.
SPACY_GPU = _flag("SPACY_GPU", True)
RUN_MULTISEED = _flag("RUN_MULTISEED", False)
SEEDS = [PRIMARY_SEED] + ([123, 2024] if RUN_MULTISEED else [])
UPSTREAM_COMMIT = "b1d414143d11c8ed988b4cccbb06626cc8272bbe"

# Capture an existing `huggingface-cli login` BEFORE redirecting HF_HOME.  The
# token lives under the DEFAULT HF_HOME, so once we move HF_HOME into the project
# directory a freshly spawned child finds no token and gated downloads 401 --
# even though the parent, which imported huggingface_hub earlier, looks fine.
# Exporting HF_TOKEN makes auth explicit and inherited by every subprocess.  A
# token pasted into the control cell is already in the environment and wins here.
_cli_token = Path.home() / ".cache" / "huggingface" / "token"
if not os.environ.get("HF_TOKEN") and _cli_token.is_file():
    os.environ["HF_TOKEN"] = _cli_token.read_text().strip()
os.environ["HF_HOME"] = str(WORK / "hf_cache")

# Defaults for the Qwen3-1.7B-Base server pass; the control cell sets all of them, so
# these apply only when a flag is left None there or this cell is re-run alone.
# The Colab notebook keeps them False because a stray Run All there costs money
# and a session slot.  Here the whole point is an unattended pass, and every
# stage is resumable, so an accidental start costs the shard in flight.
RUN_DATA = _flag("RUN_DATA", True)
RUN_SMOKE = _flag("RUN_SMOKE", True)
RUN_SCREEN = _flag("RUN_SCREEN", True)
# Three seeds are a Colab job for the headline arms only; Qwen is a second-family
# replication at seed 42, so this stays off.
RUN_CONFIRM = _flag("RUN_CONFIRM", False)
# Task 15b only, and only after the 1B gate has chosen an arm.
RUN_HYBRID = _flag("RUN_HYBRID", False)
RUN_NEGATIVE_CONTROLS = _flag("RUN_NEGATIVE_CONTROLS", False)
RUN_VERIFIED = _flag("RUN_VERIFIED", False)
VERIFIED_SMOKE_STEPS = int(float(os.environ.get("VERIFIED_SMOKE_STEPS", "0")))
RUN_EVAL = _flag("RUN_EVAL", True)
# Skip any run whose artefacts already exist.  A killed job resumes from here.
RESUME_FINISHED_RUNS = True

# Extraction precision.  Upstream inherits use_4bit=True from TrainingConfig,
# where it exists for QLoRA TRAINING.  Extraction runs ~94 small forwards per
# sequence; measured on an L4, bf16 was ~25% faster and avoids quantisation
# noise in the top-100 pool and the 0.75 cosine threshold the method depends on.
EXTRACT_4BIT = False
# Must match the Colab notebook: the two variants feed one study, and a model
# extracted here on 10,000 sequences could not be compared with one extracted
# there on 4,000.  Zhang et al. Fig. 8 reports STS unchanged at a quarter of the
# data; 4,000 keeps the 80/10/10 ratio.  Raise BOTH to 10000 for the strict
# reproduction.  merge_synonym_parts hard-fails unless the split sizes sum to the
# rows the shards actually cover.
EXTRACT_SEQUENCES = 4000
# Sequences per extraction shard, and so exactly what a killed job costs: a shard
# is only recorded as finished once it completes.
EXTRACT_SHARD = 500
SPLIT_TRAIN = int(EXTRACT_SEQUENCES * 0.8)
SPLIT_VAL = SPLIT_TEST = int(EXTRACT_SEQUENCES * 0.1)

_BAR = re.compile(r"\b(\d+)/(\d+)\s*\[")     # bounded tqdm: "  200/1000 ["
_BAR_OPEN = re.compile(r"\b(\d+)it\s*\[")     # unbounded tqdm: "  3200it [00:49"
PROGRESS_EVERY = 100

def run(argv, cwd=None, env=None):
    """Run a child process, streaming its output and keeping the tail on failure."""
    argv = list(map(str, argv))
    print("+", " ".join(argv), flush=True)
    merged = os.environ.copy()
    merged.update({"CONCEPT_DATA_ROOT": str(DATA),
                   "CONCEPT_CHECKPOINT_ROOT": str(RUNS),
                   "CONCEPT_RESULTS_ROOT": str(OUTPUTS),
                   # The POS filter is ~90% of extraction; on GPU it ran 8.5x
                   # faster on an A40 (42.6 -> 5.0 s/seq) with 12400/12400 POS
                   # tags identical to CPU.  SPACY_GPU gates it because it needs
                   # cupy, and because the Colab-produced Llama data did not use it.
                   "CONCEPT_SPACY_GPU": "1" if SPACY_GPU else "0",
                   # This machine has no locale set, so Python defaults to ASCII
                   # and both the child and its captured output die on C4's
                   # non-ASCII text, partway through a long run.
                   "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
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
        else:
            # Dataset loading has no total ("3200it [00:49"), so there is no final
            # count to anchor on.  Thin it ten times harder; it is pure noise and
            # a full sweep would otherwise emit tens of thousands of lines.
            loose = _BAR_OPEN.search(line)
            if loose and int(loose.group(1)) % (PROGRESS_EVERY * 10):
                continue
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(
            f"command failed with exit code {code}\n  {' '.join(argv)}\n"
            f"--- last {len(tail)} lines of its output ---\n{''.join(tail)}")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sync_small_artifacts(source, destination):
    destination = Path(destination)
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

# Inside THIS model's output directory.  A manifest shared between models would
# name another model's adapters, and restore_all would load them without
# complaint -- the resume path has no way to tell whose weights it just read.
RUN_MANIFEST = MODEL_OUT / "run_manifests"

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

def eval_covered(path, checkpoints):
    """True when `path` already scores every checkpoint of THIS pass.

    Coverage, not mere existence: adding a seed grows `checkpoints`, the old file
    stops covering it, and the evaluator reruns.  A plain existence check would
    report the previous pass's table as if it were this one's.
    """
    if not (RESUME_FINISHED_RUNS and Path(path).is_file()):
        return False
    try:
        rows = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return False              # truncated by a killed job mid-write; redo it
    if not isinstance(rows, list):
        return False
    scored = {str(row.get("checkpoint")) for row in rows if isinstance(row, dict)}
    return set(map(str, checkpoints)) <= scored

def sts_covered(path):
    """True when one STS pass already wrote its nine task rows to `path`."""
    if not (RESUME_FINISHED_RUNS and Path(path).is_file()):
        return False
    with open(path, encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip()) >= 10   # header + 9 tasks

def guarded(path, checkpoints, argv, cwd, what):
    """Run one evaluator unless its output already covers every checkpoint."""
    if eval_covered(path, checkpoints):
        print(f"resume: {what} already covers {len(checkpoints)} checkpoints, skipping")
        return
    run(argv, cwd=cwd)

def assert_under_base(path):
    resolved = Path(path).resolve()
    assert str(resolved).startswith(str(BASE)), f"{resolved} escapes {BASE}"

for directory in (WORK, DATA, RUNS, OUTPUTS, MODEL_OUT, RESULT_DIR, LOG_DIR, SHARED):
    directory.mkdir(parents=True, exist_ok=True)
print("BASE      ", BASE)
print("MODEL     ", BASE_MODEL, "->", MODEL_TAG)
print("REUSE FROM", REUSE_CONTENT_WORDS_FROM or "(nothing: full extraction)")
print("OUTPUTS   ", MODEL_OUT)
print("GPU       ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
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
# Reset to the pinned commit and wipe every patch artefact before re-applying.
# Testing "does it apply, else does it reverse-apply" only worked while the patch
# never changed: once MAIN pulls a newer one, the old patch is applied, neither
# direction matches, and the run dies on an assertion.  Resetting is idempotent
# and always ends in the same state.  EXT holds upstream code only -- the corpus
# lives in DATA, outside it -- so clean -fd is safe.
run(["git", "-C", str(EXT), "reset", "--hard", UPSTREAM_COMMIT])
run(["git", "-C", str(EXT), "clean", "-fdq"])

patch_file = MAIN / "external" / "learning-concepts.patch"
assert patch_file.exists(), f"{patch_file} missing; push it before running here."
run(["git", "apply", str(patch_file)], cwd=EXT)
print("patch applied onto", UPSTREAM_COMMIT[:7])

run([sys.executable, "-m", "pip", "install", "-q", "-e", str(EXT), "--no-deps"])

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

run([sys.executable, MAIN / "builddataset/verify_task14_data.py",
     "--repo_root", MAIN, "--download_missing",
     "--report_json", SHARED / "external_benchmark_integrity.json"], cwd=MAIN)
(SHARED / "environment_freeze.txt").write_text(
    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))

# Prove the model is reachable FROM A SUBPROCESS, which is where every download
# actually happens.  The parent can hold credentials a freshly spawned child does
# not inherit, and the failure then surfaces 20 minutes later as a bare 401 on
# config.json whose traceback says nothing about authentication.
# The token is optional because gating is: Qwen2.5 is openly licensed, Llama-3.2
# is gated.  Demanding a token unconditionally would block a Qwen-only run for no
# reason, so let the reachability probe be the thing that decides.
from huggingface_hub import login, whoami
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    print("Hugging Face:", whoami()["name"])
else:
    print("No HF_TOKEN set; continuing on the assumption this model is ungated.")
_probe = subprocess.run(
    [sys.executable, "-c",
     "import sys; from transformers import AutoConfig;"
     "AutoConfig.from_pretrained(sys.argv[1]);"
     "print('model repo reachable from a subprocess')", BASE_MODEL],
    env={**os.environ}, capture_output=True, text=True)
assert _probe.returncode == 0, (
    f"{BASE_MODEL} is not reachable from a child process.  If it is a gated repo, "
    "run `huggingface-cli login` on this machine or export HF_TOKEN before "
    f"starting Jupyter, then re-run this cell.\n{_probe.stderr[-2000:]}")
print(_probe.stdout.strip())
'''

HEADER = r'''
# Task 15 (server) — reproduce concept training before extending it

Server twin of `research_tasks_15_zhang_reproduction.ipynb` (Colab). Same
objective, same schedule, same evaluators; the difference is that the filesystem
persists, so nothing round-trips through Drive.

**How to run it.** Upload this one `.ipynb` to the server, open it, read the GPU
table in §1, edit the single control cell in §2 — GPU, model, HF token, which
stages run — and Run All. Nothing else is uploaded and no path is edited: the
notebook clones both repositories itself and keeps every artefact under its own
directory, so moving the file moves the whole experiment.

**One model per pass.** Every artefact is keyed on the model tag and the resume
logic reads finished work back by path, so `MODEL` in the control cell selects
the pass instead of a loop over models. Qwen3-1.7B-Base first; then change that
one line to `Qwen/Qwen3-4B-Base`, which reuses 1.7B's content words and skips the spaCy
POS pass entirely — the two tokenizers are compared before the copy and the run
aborts if they differ. Running them in parallel on two cards is also fine; 4B
then simply pays that pass itself, so either order is safe.

**Interruptions are cheap.** Extraction shards, trained adapters and finished
evaluation tables are all skipped on a re-run. If the kernel dies, reopen the
notebook and Run All: it picks up at the shard or arm that was in flight.

### Headless alternative, for an unattended pass

A kernel survives a dropped VPN — it runs on the server, not in the browser —
but the output stream does not always reattach, so a multi-hour pass is easier to
follow as a log file under `tmux`:

```bash
export CONCEPT_BASE=$PWD
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output executed_qwen17.ipynb reproducibilty_15.ipynb 2>&1 | tee qwen17.log
```

The control cell's values win over the shell, so set one to `None` there to pass
it in from outside instead — which is how two models share the machine on
separate cards:

```bash
CONCEPT_MODEL=Qwen/Qwen3-1.7B-Base GPU_ID=1 jupyter nbconvert ... &
CONCEPT_MODEL=Qwen/Qwen3-4B-Base   GPU_ID=2 jupyter nbconvert ... &
```

## Layout

Everything — both clones, the HF cache, the corpus, the adapters and every
report — lands under the directory this notebook is sitting in. Nothing is
written to your home directory, and nothing outside this tree is touched:

```
<the directory holding this notebook>/
├── reproducibilty_15.ipynb
├── concept_aware/
│   ├── concept-aware-training/   our repo, cloned and pulled: patch + evaluators
│   ├── learning-concepts/        upstream, reset to the pinned commit, patched
│   ├── hf_cache/                 HF_HOME: model weights and the C4 shards
│   ├── data/                     extracted concept sets and the splits
│   └── runs/                     QLoRA adapters (small at r=4, kept for resume)
└── outputs/                      <- the only directory you copy back
    ├── qwen3-1.7b-base/
    │   ├── data_audit.json
    │   ├── results/       flat_main_table.csv, sts_*.csv, *_ci_*.json, mteb_raw/
    │   ├── logs/          per-arm training_history.jsonl
    │   └── run_manifests/ label -> adapter path, extraction shard completion
    ├── qwen3-4b-base/     same shape
    └── shared/            pip freeze, benchmark integrity (model-independent)
```

`outputs/` holds reports only — an assertion fails the run if model weights or
optimizer state ever reach it — so it stays small enough to `rsync` back. It is
also the layout the local analysis copy uses, so a finished model directory is
copied off the server as-is with no renaming.

The base directory is the notebook's own, found through `JPY_SESSION_NAME`, so a
kernel started somewhere else cannot scatter a second tree. Setting
`CONCEPT_BASE` overrides it.

## Before the long run

The control cell in §2 is the whole configuration; everything below it reads from
the environment it sets. Watch the **first extraction shard** anyway: on the A40
`SPACY_GPU` measured **5.0 s/sequence** against 42.6 on CPU, so at 5 s/seq the
4,000 sequences take about 5.5 h and at 40 they take days. A rate near 40 means
`CONCEPT_SPACY_GPU` did not reach the child process — the flag says GPU while
the work ran on CPU. (Missing cupy is the loud failure, not the slow one:
`spacy.require_gpu()` raises `No GPU devices detected` rather than falling back.)
Shards are recorded only once they finish, so interrupting after the first costs
nothing.
'''

HEADER_15B = r'''
# Task 15b (server) — alternative-only supervision and contrastive calibration

Server twin of `research_tasks_15b_objective_and_contrastive.ipynb` (Colab). These
are the arms the paper contributes: **alternative-only** concept supervision
(`--exclude-target`, uniform over the alternatives) and the same with
**hard-negative contrastive** calibration.

**Run it in the SAME directory as the Task 15 pass for this model.** Every arm
here is compared against that pass's NTP, Zhang and randomized adapters, and it
reads them through `run_manifests/task15.json` and `results/` under the same
`CONCEPT_BASE`. Point it somewhere else and there is nothing to compare against.
Nothing is re-extracted: the concept data is already on disk.

```bash
cd <the Task 15 directory for this model>     # e.g. ~/fos_retrieval/task15
export CONCEPT_BASE=$PWD
papermill reproducibilty_15b.ipynb executed_15b.ipynb -k concept --log-output 2>&1 | tee -a 15b.log
```

## What it trains

With `SELECTED_ALPHA` set in the control cell (the replication path), `RUN_SCREEN`
trains eight arms at seed 42:

| arms | why |
|---|---|
| uniform α ∈ {0.25, 0.5, 1.0} | the objective, at three weights |
| uniform α = 0.75 | frontier control: a contrastive arm must beat the uniform *curve*, not just the same-α point |
| inclusive uniform α | identical loss with the observed target back in the set — isolates the exclusion |
| contrastive β ∈ {0.25, 0.5, 1.0} | hard negatives at the locked α |

The α = 0.75 point exists because contrastive arms land between α = 0.5 and
α = 1.0 on both axes, so two uniform points cannot say whether they sit above the
uniform curve or merely on it. Any contrastive claim is tested against a HIGHER-α
uniform arm, never only against the same α.

## Costs, measured on an A40 with Qwen3-1.7B-Base

Roughly 30 min per training arm, so ~4 h for the eight. Evaluation then scores
these plus the Task 15 comparators — about 17 checkpoints — at ~4.5 min for
SWORDS and ~5 min for STS each, so ~4-5 h. Call it 9-10 h end to end, and every
stage resumes.
'''

RENAME = [
    # Layout: outputs/<model tag>/{data_audit.json,results,logs,run_manifests}.  The
    # Colab body keys everything on _TAG_SUFFIX because its 1B run predates the
    # per-model layout; the server has no such legacy, so every model gets its own
    # directory and the suffix has no meaning.  These must run BEFORE the generic
    # DRIVE_RESULTS rename below, since they match the original spelling.
    ('DRIVE_RESULTS / f"data_audit{_TAG_SUFFIX}.json"', "DATA_AUDIT"),
    ('load_runs(f"task15_shards{_TAG_SUFFIX}")', 'load_runs("task15_shards")'),
    ('save_runs(shards_done, f"task15_shards{_TAG_SUFFIX}")', 'save_runs(shards_done, "task15_shards")'),
    ('load_runs(f"task15{_TAG_SUFFIX}")', 'load_runs("task15")'),
    ('save_runs(REPRO_RUNS, f"task15{_TAG_SUFFIX}")', 'save_runs(REPRO_RUNS, "task15")'),
    ("result_dir = DRIVE_RESULTS / RESULT_DIR", "result_dir = RESULT_DIR"),
    ('sync_small_artifacts(path, f"{LOG_DIR}/{label}")', "sync_small_artifacts(path, LOG_DIR / label)"),
    ('plt.savefig(DRIVE_RESULTS / RESULT_DIR / "training_curves.png"', 'plt.savefig(RESULT_DIR / "training_curves.png"'),
    # Task 15b keys its artifacts the same way: inside this model's own output
    # directory, with no tag suffix, because the server gives every model its own.
    ('SCREEN_DIR = DRIVE_RESULTS / f"task15b_screen{_TAG_SUFFIX}"',
     'SCREEN_DIR = MODEL_OUT / "task15b_screen"'),
    ('DRIVE_RESULTS / f"contrastive_negative_report{_TAG_SUFFIX}.json"',
     'MODEL_OUT / "contrastive_negative_report.json"'),
    ('DRIVE_RESULTS / f"contrastive_negative_report_{name}{_TAG_SUFFIX}.json"',
     'MODEL_OUT / f"contrastive_negative_report_{name}.json"'),
    ('load_runs(f"task15b{_TAG_SUFFIX}")', 'load_runs("task15b")'),
    ('save_runs(OBJECTIVE_RUNS, f"task15b{_TAG_SUFFIX}")', 'save_runs(OBJECTIVE_RUNS, "task15b")'),
    ('sync_small_artifacts(path, f"task15b_logs{_TAG_SUFFIX}/{label}")',
     'sync_small_artifacts(path, LOG_DIR / "task15b_logs" / label)'),
    # The verified-supervision stage: same per-model layout, no tag suffix.
    ('VERIFIED_DIR = DRIVE_RESULTS / f"task15b_verified{_TAG_SUFFIX}"',
     'VERIFIED_DIR = MODEL_OUT / "task15b_verified"'),
    ('load_runs(f"task15b_verified{_TAG_SUFFIX}")', 'load_runs("task15b_verified")'),
    ('save_runs(VERIFIED_RUNS, f"task15b_verified{_TAG_SUFFIX}")', 'save_runs(VERIFIED_RUNS, "task15b_verified")'),
    ('sync_small_artifacts(path, f"task15b_verified_logs{_TAG_SUFFIX}/{label}")',
     'sync_small_artifacts(path, LOG_DIR / "task15b_verified_logs" / label)'),
    # 15b scores its arms against Task 15's, so it reads that notebook's result
    # directory -- which is why both must run with the same CONCEPT_BASE.
    ("task15_dir = DRIVE_RESULTS / RESULT_DIR", "task15_dir = RESULT_DIR"),
    # Wording that is true of Colab and false of a persistent filesystem.
    ("# Unconditional: /content is wiped between sessions and the smoke run in the next\n"
     "# section reads synonyms_train.jsonl directly.  Whenever the data was generated in\n"
     "# an EARLIER session -- the normal case for every model after the first -- RUN_DATA\n"
     "# is off, and restoring only inside that branch left the smoke run to die on a\n"
     "# missing file before any training started.",
     "# A cheap existence check here; kept unconditional so the body stays identical\n"
     "# to the Colab notebook's, where it is a real restore from Drive."),
    ("Each 1k shard is cached to Drive once\n    # it completes, so a disconnect costs at most the shard in progress.",
     "A shard is recorded as finished only\n    # once it completes, so a killed job costs at most the shard in progress."),
    ("# One manifest per model: a shared name would let the 1B shards mark the 3B\n"
     "    # ones as finished, and extraction would be skipped entirely.",
     "# The manifest lives in this model's own output directory, so one model's\n"
     "    # finished shards can never mark another's as done and skip extraction."),
    ("# Per model: a shared name means the second model's audit silently\n"
     "         # overwrites the first's, and the provenance of the finished data is\n"
     "         # exactly what an audit report exists to preserve.",
     "# Inside this model's directory: a shared path means the second model's\n"
     "         # audit silently overwrites the first's, and the provenance of the\n"
     "         # finished data is exactly what an audit report exists to preserve."),
    ("exists locally or on Drive", "is on disk"),
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

def build(source_name, target_name, header, control):
    """Assemble one server notebook from its Colab source."""
    source = json.loads((NB_DIR / source_name).read_text())
    # The GPU table and the control cell come FIRST, before the install cell
    # imports torch: CUDA_VISIBLE_DEVICES chosen after the driver initialises is
    # ignored, and the run would land on card 0 while reporting the card asked for.
    cells = [md(header.strip("\n")),
             md("## 1. Which GPUs are free\n\n"
                "Run this first, then name one in the control cell below. Getting it wrong "
                "means restarting the kernel, not just re-running a cell."),
             code(GPU_PICK),
             md("## 2. The one cell to edit\n\n"
                "GPU, model, HF token and the stage gates. Everything after this reads them "
                "from the environment, including every child process."),
             code(control),
             md("## 3. Dependencies\n\n"
                "Once per environment. The `transformers` pin matters: upstream's extractor "
                "reuses a prefix KV cache through an API removed in v5."),
             code(INSTALL), code(SETUP), code(BOOTSTRAP)]

    for index, cell in enumerate(source["cells"]):
        if index in (0, 1, 2):          # title, Colab setup, Colab bootstrap
            continue
        text = "".join(cell["source"])
        for old, new in RENAME:
            text = text.replace(old, new)
        cells.append(code(text) if cell["cell_type"] == "code" else md(text))
    return cells, NB_DIR / target_name

def undefined_names(notebook_cells):
    """Names the notebook loads but never binds.

    The server setup cell is written by hand while the experiment body is copied
    from the Colab notebook, so a helper added to the Colab setup silently fails
    to reach here and the copied body NameErrors at runtime.  Locals are
    over-approximated (every parameter and assignment target anywhere counts as
    bound) so this reports only names that are genuinely nowhere.
    """
    import ast, builtins
    bound, loaded = set(dir(builtins)), set()
    trees = []
    for cell in notebook_cells:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        # %pip / ! lines are notebook magics, not Python.
        text = "\n".join("" if line.lstrip().startswith(("%", "!")) else line
                          for line in text.splitlines())
        trees.append(ast.parse(text))
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
                bound.update(a.arg for a in getattr(node, "args", ast.arguments(
                    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])).args)
                for extra in ("posonlyargs", "kwonlyargs"):
                    bound.update(a.arg for a in getattr(getattr(node, "args", None), extra, []) or [])
                for attr in ("vararg", "kwarg"):
                    arg = getattr(getattr(node, "args", None), attr, None)
                    if arg: bound.add(arg.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bound.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, ast.Global):
                bound.update(node.names)
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
    return sorted(loaded - bound)


NOTEBOOKS = [
    ("research_tasks_15_zhang_reproduction.ipynb", "reproducibilty_15.ipynb", HEADER, CONTROL_15),
    ("research_tasks_15b_objective_and_contrastive.ipynb", "reproducibilty_15b.ipynb",
     HEADER_15B, CONTROL_15B),
]

for source_name, target_name, header, control in NOTEBOOKS:
    cells, target = build(source_name, target_name, header, control)
    missing = undefined_names(cells)
    if missing:
        raise SystemExit(
            f"refusing to write {target_name}, which would NameError at runtime.\n"
            "These names are used but never bound -- the hand-written server setup\n"
            "has drifted behind the Colab COMMON_SETUP it mirrors:\n  "
            + "\n  ".join(missing))
    target.write_text(json.dumps({
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }, indent=1) + "\n")
    print(f"wrote {target} ({len(cells)} cells)")
