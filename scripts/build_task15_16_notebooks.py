"""Generate the three canonical Colab notebooks for Tasks 15--16."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.strip() + "\n",
    }


def write(name, cells):
    payload = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    (NB_DIR / name).write_text(json.dumps(payload, indent=1) + "\n")


COMMON_SETUP = r'''
from pathlib import Path
from getpass import getpass
import datetime, hashlib, json, os, re, shutil, subprocess, sys, torch

BASE_MODEL = "meta-llama/Llama-3.2-1B"
# Every per-model artifact -- data, adapters, manifests, results -- is keyed on
# this tag.  Nothing about a model may share a path with another model: adapters
# are restored from Drive by path, so a collision would silently hand one model's
# weights to another and the run would look like it succeeded.
MODEL_TAG = BASE_MODEL.split("/")[-1].lower()
# combined.jsonl depends only on the tokenizer (plus spaCy), so a model that
# shares a vocabulary with one already extracted produces a byte-identical file.
# Llama-3.2 1B and 3B do.  Naming the donor here skips the POS pass, which is the
# most expensive stage of extraction; the vocabularies are compared before the
# copy and the run aborts if they differ.
REUSE_CONTENT_WORDS_FROM = None
PRIMARY_SEED = 42
# One seed first.  Seed 42 alone screens the pipeline and shows the direction of
# every effect, but it CANNOT support a claim: the pre-registered rule needs all
# three seeds to agree in sign.  Flip to True for the reportable run; resume makes
# the seed-42 arms free the second time.
RUN_MULTISEED = False
SEEDS = [PRIMARY_SEED] + ([123, 2024] if RUN_MULTISEED else [])
UPSTREAM_COMMIT = "b1d414143d11c8ed988b4cccbb06626cc8272bbe"
MAIN = Path("/content/concept-aware-training")
EXT = Path("/content/learning-concepts")
DATA = Path("/content/concept_data")
RUNS = Path("/content/concept_runs")
DRIVE_ROOT = Path("/content/drive")
DRIVE_PROJECT = DRIVE_ROOT / "MyDrive/concept_training"
DRIVE_RESULTS = DRIVE_PROJECT / "task15_16_results"
os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/content/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/content/hf_cache/transformers"

# Turn on one gate at a time.  Defaults are safe and do not start a GPU matrix.
RUN_DATA = False
RUN_SMOKE = False
RUN_SCREEN = False
RUN_CONFIRM = False
# Task 15b only: Zhang's set marginal plus an alternative-only auxiliary term, and
# the two negative-quality controls.  Seed 42 unless SELECTED_HYBRID is set.
#   os.environ["HYBRID_ARMS"]     = "uniform:0.125,mass:0.125"  train only these grid arms
#       (empty = all); a second process on another GPU can take the rest.
#   os.environ["SELECTED_HYBRID"] = "uniform:0.125"  with RUN_MULTISEED: that arm's seeds
#       123 and 2024, on the same original file as Task 15's three-seed baselines.
RUN_HYBRID = False
# The two negative-quality controls (clean / fragments).  They are a diagnostic
# for the demoted contrastive term, NOT a method-selection run, so they are off
# by default and should be run AFTER the H1/H2 screen has been gated.
RUN_NEGATIVE_CONTROLS = False
# Task 15b only: the verified-supervision comparison.  Six arms trained from
# synonyms_train_verified.jsonl (pruned positives + verifier-mined negatives), one
# slot objective each, scored in their own result directory against Task 15's
# NTP and Zhang.  VERIFIED_SMOKE_STEPS > 0 trains ~that many steps per arm,
# prints first-step loss magnitudes for the pre-declared lambda rule, and
# trains nothing else.  Weights are read from the environment so a Colab session
# and the server twin declare them the same way:
#   os.environ["VERIFIED_LAMBDA"] = "1.0"    weight on the slot objective (pool/rank/list, alt-only control)
#   os.environ["VERIFIED_GAMMA"]  = "0.125"  the continuity arm's gamma: Llama .125, Qwen .0625
#   os.environ["VERIFIED_ARMS"]   = "verified_alt_uniform"  train only these arms (empty =
#       every arm); a final pass with it empty resumes them all and evaluates.
RUN_VERIFIED = False
VERIFIED_SMOKE_STEPS = 0
# With VERIFIED_SMOKE_STEPS > 0: True = smoke, print, stop (read it by hand);
# False = smoke, apply the declared lambda rule automatically, record the
# decision, then train and evaluate in the same pass -- the unattended path.
VERIFIED_SMOKE_ONLY = False
RUN_EVAL = False
# Skip any run whose adapter is already on Drive.  /content is wiped between
# Colab sessions, so without this the screen -> gate -> confirm sequence has to
# finish in one sitting.  Set False to force a full retrain.
RESUME_FINISHED_RUNS = True
# Precision for CONCEPT EXTRACTION only; training is QLoRA-4bit either way.
# Upstream inherits use_4bit=True from TrainingConfig, but extraction runs ~94
# small forwards per sequence and NF4 dequantization dominates them: 8.0 s/seq
# at 4-bit against roughly a third of that in bf16, i.e. 22 h against ~8 h for
# ten shards.  Quantization also perturbs the top-100 pool and the 0.75 cosine
# threshold the method depends on.  Use ONE setting for all ten shards.
EXTRACT_4BIT = False
# How many C4 sequences to extract concept sets for.  The paper uses 10,000 split
# 8000/1000/1000, but extraction measured 7.45 s/sequence on an L4 (spaCy 32%,
# GPU the rest; bf16 moved it only 7%), so the full set is ~21 h for one model.
# Zhang et al. Fig. 8 ablates exactly this and reports STS unchanged at a quarter
# of the training data, so 4,000 keeps the 80/10/10 ratio at ~8 h.  Set 10000 for
# the strict reproduction.  merge_synonym_parts hard-fails unless the split sizes
# sum to the number of sequences the shards actually cover, and the 3B scaling
# stage uses the same count so size comparisons are not confounded by data volume.
EXTRACT_SEQUENCES = 4000
# Sequences per extraction shard.  A shard is the unit of resume: it is cached to
# Drive only once it completes, so this is exactly what a disconnect costs.  At
# 3B's measured 10 s/sequence a 1000-shard is nearly three hours of exposure;
# 500 halves that for one extra model load per shard, about 30 seconds.
EXTRACT_SHARD = 500
SPLIT_TRAIN = int(EXTRACT_SEQUENCES * 0.8)
SPLIT_VAL = SPLIT_TEST = int(EXTRACT_SEQUENCES * 0.1)

_BAR = re.compile(r"\b(\d+)/(\d+)\s*\[")     # bounded tqdm: "  200/1000 ["
_BAR_OPEN = re.compile(r"\b(\d+)it\s*\[")     # unbounded tqdm: "  3200it [00:49"
PROGRESS_EVERY = 100                          # one line per this many items

def run(argv, cwd=None, env=None):
    """Run a child process, streaming its output into the cell.

    subprocess.run() writes the child's stdout to the kernel's file descriptor,
    which Colab does not route into the cell, so a failing command used to raise
    CalledProcessError with no diagnostic at all.  Stream it line by line and put
    the tail into the exception message.
    """
    argv = list(map(str, argv))
    print("+", " ".join(argv), flush=True)
    merged = os.environ.copy()
    merged.update({"CONCEPT_DATA_ROOT": str(DATA),
                   "CONCEPT_CHECKPOINT_ROOT": str(RUNS),
                   "CONCEPT_RESULTS_ROOT": str(DRIVE_RESULTS)})
    if env: merged.update(env)
    process = subprocess.Popen(argv, cwd=cwd, env=merged, text=True, bufsize=1,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail = []
    for line in process.stdout:
        line = line.replace("\r", "")
        tail.append(line)
        del tail[:-40]
        hit = _BAR.search(line)       # tqdm writes "c4:  12%| | 120/1000 [..."
        if hit:
            done, total = int(hit.group(1)), int(hit.group(2))
            if done % PROGRESS_EVERY and done != total:
                continue
        else:
            # Dataset loading has no total ("3200it [00:49"), so there is no final
            # count to anchor on.  Thin it ten times harder; it is pure noise and
            # seven training runs would otherwise emit tens of thousands of lines.
            loose = _BAR_OPEN.search(line)
            if loose and int(loose.group(1)) % (PROGRESS_EVERY * 10):
                continue
        print(line, end="", flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(
            f"command failed with exit code {code}\n  {' '.join(argv)}\n"
            f"--- last {len(tail)} lines of its output ---\n{''.join(tail)}")

def assert_ephemeral(path):
    resolved = str(Path(path).resolve())
    assert resolved.startswith("/content/") and not resolved.startswith("/content/drive/"), resolved

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sync_small_artifacts(source, label):
    destination = DRIVE_RESULTS / label
    destination.mkdir(parents=True, exist_ok=True)
    for path in Path(source).rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".png", ".log"}:
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

def audit_no_drive_weights():
    # LoRA adapters at r=4 are ~10 MB and ARE cached to Drive on purpose: /content is
    # ephemeral, and without them a disconnect between the screen and confirm phases
    # discards every trained arm.  Full weights and optimizer state stay off Drive.
    forbidden = {"pytorch_model.bin", "model.safetensors", "optimizer.pt",
                 "scheduler.pt", "scaler.pt", "rng_state.pth"}
    found = [str(p) for p in DRIVE_RESULTS.rglob("*")
             if p.name in forbidden or p.name.startswith("checkpoint-")]
    assert not found, f"full-weight/optimizer artifacts reached Drive: {found}"

DATA_CACHE = DRIVE_PROJECT / "task15_16_data"

def cache_dataset_to_drive(leaf):
    """Cache JSONL data needed by later notebooks, including top-k shards."""
    model_root = Path(leaf).parent
    copied = 0
    for source in (model_root, model_root / "embedding", model_root / "prompting"):
        destination = DATA_CACHE / source.relative_to(DATA)
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.glob("*.jsonl"):
            shutil.copy2(path, destination / path.name)
            copied += 1
    print(f"cached {copied} dataset files under", DATA_CACHE / model_root.relative_to(DATA))

def restore_dataset_from_drive(leaf):
    model_root = Path(leaf).parent
    embedding_source = DATA_CACHE / Path(leaf).relative_to(DATA)
    copied = 0
    for destination in (model_root, model_root / "embedding", model_root / "prompting"):
        source = DATA_CACHE / destination.relative_to(DATA)
        if not source.exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        for path in source.glob("*.jsonl"):
            target = destination / path.name
            if not target.is_file():
                shutil.copy2(path, target)
                copied += 1
    print(f"restored {copied} dataset files from", DATA_CACHE / model_root.relative_to(DATA))
    return (embedding_source / "synonyms_train.jsonl").is_file()

# The 1B reproduction wrote these names before the study covered more than one
# model; they stay as they are so that work still resolves, and every other model
# gets its own suffix rather than overwriting it.
_TAG_SUFFIX = "" if MODEL_TAG == "llama-3.2-1b" else f"_{MODEL_TAG}"
RESULT_DIR = f"task15_reproduction{_TAG_SUFFIX}"
LOG_DIR = f"task15_logs{_TAG_SUFFIX}"

# HyperLex is split train/dev/test; the lexical split shares no lemma between
# them, which is the stricter generalisation setting.  Every diagnostic in these
# notebooks reads DEV.  The test file is deliberately not referenced anywhere:
# the hierarchy claim in Task 16 is scored on it exactly once, after the method
# is locked, and a diagnostic that peeks at it would spend that.
HYPERLEX_DEV = MAIN / "data/hyperlex-data/splits/lexical/hyperlex_dev_all_lexical.txt"
HYPERLEX_TEST = MAIN / "data/hyperlex-data/splits/lexical/hyperlex_test_all_lexical.txt"  # locked
RUN_MANIFEST = DRIVE_RESULTS / "run_manifests"

def save_runs(runs, name):
    """Persist label -> adapter path so a later session can evaluate earlier phases."""
    RUN_MANIFEST.mkdir(parents=True, exist_ok=True)
    (RUN_MANIFEST / f"{name}.json").write_text(
        json.dumps({k: str(v) for k, v in runs.items()}, indent=2))

def load_runs(name):
    path = RUN_MANIFEST / f"{name}.json"
    if not path.is_file():
        return {}
    return {k: Path(v) for k, v in json.loads(path.read_text()).items()}

def restore_all(runs):
    """Pull every adapter in `runs` back onto /content; drop any that is missing."""
    live = {}
    for label, path in runs.items():
        if (Path(path) / "adapter_config.json").is_file() or restore_adapter_from_drive(path):
            live[label] = Path(path)
        else:
            print("missing adapter, dropping from this pass:", label)
    return live

def eval_done(marker):
    """True when a completed evaluation artifact is already on Drive."""
    return Path(marker).is_file() and RESUME_FINISHED_RUNS

def eval_covered(path, checkpoints):
    """True when `path` already scores every checkpoint of THIS pass.

    Each JSON evaluator writes a list of {"checkpoint": ..., ...}.  Testing
    coverage rather than mere existence is what makes this safe to resume:
    adding a seed grows `checkpoints`, the old file no longer covers it, and the
    evaluator reruns.  A plain "file exists" check would instead report the
    previous pass's table as if it were this one's.
    """
    if not (RESUME_FINISHED_RUNS and Path(path).is_file()):
        return False
    try:
        rows = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return False              # truncated by a disconnect mid-write; redo it
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

ADAPTER_CACHE = DRIVE_PROJECT / "task15_16_adapters"
# train.py writes concept_training_config.json (the objective metadata: lambda,
# alpha, gamma, exclude_target, seed).  It was missing here, so a session that
# died after training cached the WEIGHTS but not the description of what they
# were trained with -- recoverable only from the directory name.  run_config.json
# is kept for older runs that wrote it.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors",
                 "training_history.jsonl", "run_config.json",
                 "concept_training_config.json")

def adapter_cache_dirs(path):
    """Drive locations for one adapter, most current first.

    The second entry is the layout used before adapters were keyed on the model,
    so the seed-42 arms trained then still resume instead of silently retraining.
    """
    relative = Path(path).relative_to(RUNS)
    locations = [ADAPTER_CACHE / relative]
    # The pre-tag layout was written by the 1B reproduction and by nothing else,
    # so only that model may look there.  Offering it to every model would let 3B
    # restore 1B's weights and report them as a 3B result.
    if not _TAG_SUFFIX:
        locations.append(ADAPTER_CACHE / Path(*relative.parts[1:]))
    return locations

def cache_adapter_to_drive(path):
    """Copy one finished adapter to Drive so a later session can resume."""
    destination = adapter_cache_dirs(path)[0]
    destination.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        source = Path(path) / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    return destination

def restore_adapter_from_drive(path):
    """Return True when a completed adapter for `path` was restored from Drive."""
    for source in adapter_cache_dirs(path):
        if not (source / "adapter_config.json").is_file():
            continue
        Path(path).mkdir(parents=True, exist_ok=True)
        for name in ADAPTER_FILES:
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, Path(path) / name)
        return True
    return False

DATA.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)
# DRIVE_RESULTS is deliberately NOT created here.  Creating any path under
# /content/drive before drive.mount() makes the mountpoint non-empty, and the
# mount then fails with "Mountpoint must not already contain files".  The next
# cell creates it immediately after mounting.
'''


BOOTSTRAP = r'''
from google.colab import drive
if DRIVE_ROOT.is_dir() and not (DRIVE_ROOT / "MyDrive").is_dir():
    # A previous cell (or a failed run) left plain directories at the mountpoint.
    shutil.rmtree(DRIVE_ROOT)
drive.mount(str(DRIVE_ROOT))
DRIVE_RESULTS.mkdir(parents=True, exist_ok=True)

if not MAIN.exists():
    run(["git", "clone", "https://github.com/SharvaGogawale1/concept-aware-training.git", MAIN])
else:
    run(["git", "-C", str(MAIN), "pull", "--ff-only"])
if not EXT.exists():
    run(["git", "clone", "https://github.com/christine-zhang1/learning-concepts.git", EXT])
run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=EXT)
patch_file = MAIN / "external" / "learning-concepts.patch"
assert patch_file.exists(), "Commit external/learning-concepts.patch before running Colab."
# Reset to the pinned commit and wipe every patch artefact before re-applying.
# Testing "does it apply, else does it reverse-apply" only held while the patch
# never changed: once a newer patch lands, the old one is applied, neither
# direction matches, and the run dies on an assertion.  Resetting is idempotent.
# EXT holds upstream code only -- the corpus lives in DATA -- so clean is safe.
run(["git", "-C", str(EXT), "reset", "--hard", UPSTREAM_COMMIT])
run(["git", "-C", str(EXT), "clean", "-fdq"])
run(["git", "apply", str(patch_file)], cwd=EXT)
print("patch applied onto", UPSTREAM_COMMIT[:7])

run([sys.executable, "-m", "pip", "install", "-q", "-e", str(EXT), "--no-deps"])
run([sys.executable, "-m", "pip", "install", "-q", "accelerate",
     "peft", "bitsandbytes", "datasets", "spacy", "mteb>=1.12", "wandb",
     "nltk", "scipy", "scikit-learn", "seaborn", "pytest"])
# Pinned LAST so nothing above can pull it forward.  Upstream pins no versions,
# but their extraction reuses a prefix KV cache through
# DynamicCache.from_legacy_cache, which transformers removed in v5; the current
# Colab image installs v5 and the first shard dies with AttributeError.  The
# 4.5x line keeps that API and still satisfies our Task-14 evaluators.
run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.51,<4.58"])
# Colab ships torchao 0.10, but peft requires >=0.16 and RAISES from
# is_torchao_available() rather than degrading.  That call sits inside
# PeftModel.from_pretrained, which every evaluator uses to load an adapter, so
# without this the failure lands hours later at evaluation rather than here.
# get_peft_model takes a different path, which is why training itself succeeds.
run([sys.executable, "-m", "pip", "install", "-q", "-U", "torchao>=0.16"])
import transformers as _tf
print("transformers", _tf.__version__)
run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
run([sys.executable, MAIN / "builddataset/verify_task14_data.py",
     "--repo_root", MAIN, "--download_missing",
     "--report_json", DRIVE_RESULTS / "external_benchmark_integrity.json"], cwd=MAIN)
# SemEval-07 and the fixed CoInCo subsets, derived from the pinned raw files above;
# the script pins the derived digests and stops if a rebuild differs.
run([sys.executable, MAIN / "builddataset/build_lexsub_benchmarks.py", "--repo_root", MAIN], cwd=MAIN)
(DRIVE_RESULTS / "environment_freeze.txt").write_text(
    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
)

from huggingface_hub import login
login(token=getpass("Hugging Face token (input hidden): "), add_to_git_credential=False)
'''


TRAIN_HELPER = r'''
MODEL_TAG = BASE_MODEL.split("/")[-1].lower()
LEAF = DATA / "c4" / MODEL_TAG / "embedding"
# /content is wiped between sessions; pull the generated concept data back rather
# than paying the multi-hour regeneration again.
restore_dataset_from_drive(LEAF)

def adapter_path(method, seed, value):
    path = RUNS / MODEL_TAG / method / f"seed_{seed}" / str(value)
    assert_ephemeral(path)
    return path

def finished(path):
    """A run counts as finished when its adapter exists locally or on Drive."""
    if (Path(path) / "adapter_config.json").is_file():
        return True
    return restore_adapter_from_drive(path)

def train_flat(method, seed, concept_weight, *, objective="set_marginal",
               slot_ntp_weight=None, contrast_beta=0.0, exclude_target=False,
               randomized=False, data_augmentation=False, epochs=5, train_file=None,
               max_samples=None, batch=8, accum=2, alt_aux="none", alt_aux_weight=0.0,
               logging_steps=None, force=False):
    # The aux suffix is added only when the term is on, so every adapter trained
    # before it existed keeps its path and still resumes.
    aux_tag = "" if alt_aux == "none" else f"_aux_{alt_aux}_{alt_aux_weight}"
    out = adapter_path(method, seed, f"lambda_{concept_weight}_beta_{contrast_beta}{aux_tag}")
    args = [sys.executable, "train.py", "--model-name", BASE_MODEL, "--dataset", "c4",
            "--dataset-type", "embedding", "--concept-loss-weight", concept_weight,
            "--concept-objective", objective, "--contrast-beta", contrast_beta,
            "--seed", seed, "--num-train-epochs", epochs, "--output-dir", out,
            "--save-strategy", "no", "--report-to", "none",
            "--per-device-train-batch-size", batch,
            "--gradient-accumulation-steps", accum]
    if slot_ntp_weight is not None: args += ["--slot-ntp-weight", slot_ntp_weight]
    if alt_aux != "none": args += ["--alt-aux", alt_aux, "--alt-aux-weight", alt_aux_weight]
    # Drops the observed target from the concept set, so the loss cannot be paid
    # with the mass NTP already put there.  Pair it with slot_ntp_weight=1.0 or
    # the observed target is pushed down.  adapter_path() does not encode this
    # flag, so the METHOD name must differ from the target-inclusive arm's.
    if exclude_target: args += ["--exclude-target"]
    if randomized: args += ["--randomized-synonyms"]
    if data_augmentation: args += ["--use-data-augmentation"]
    if train_file: args += ["--train-file", train_file]
    if max_samples: args += ["--max-train-samples", max_samples]
    if logging_steps: args += ["--logging-steps", logging_steps]
    # Released effective batch is 8 x 2 = 16; it is recorded in every config.
    if force:
        # A measurement run (the verified smoke) must never be answered from an old
        # adapter: its training_history.jsonl is appended to, not replaced.
        shutil.rmtree(out, ignore_errors=True)
    elif RESUME_FINISHED_RUNS and finished(out):
        print("resume: already trained, skipping", out)
        return out
    run(args, cwd=EXT)
    cache_adapter_to_drive(out)
    return out
'''


def task15():
    cells = [
        md('''# Task 15 — Reproduce concept training before extending it

This notebook answers one question first: **can we reproduce Zhang, Jurafsky, and Shani on the released C4 setup?** It keeps the released target-inclusive set-marginal objective unchanged. Our uniform and contrastive losses are not run here.

| Readable method | What changes |
|---|---|
| Pretrained | No post-training |
| NTP fine-tuning | Ordinary next-token training on the same C4 rows |
| Augmented NTP | Synonym substitutions are written into training text |
| Randomized concepts | Same-sized incoherent sets; semantic control |
| Zhang concept marginal | Raises total probability of the observed word plus accepted alternatives |

Primary reproduction evidence is nine-task mean STS, content-word NTP, and global NTP. The untouched model need not lose to NTP on every STS task.'''),
        code(COMMON_SETUP), code(BOOTSTRAP),
        md('''## Rebuild and audit the concept data for this model

The audit hard-fails on split overlap, target misalignment, empty sets, or any concept that is not a complete single token. The observed target is part of every set by construction.'''),
        code(r'''
LEAF = DATA / "c4" / MODEL_TAG / "embedding"
# Unconditional: /content is wiped between sessions and the smoke run in the next
# section reads synonyms_train.jsonl directly.  Whenever the data was generated in
# an EARLIER session -- the normal case for every model after the first -- RUN_DATA
# is off, and restoring only inside that branch left the smoke run to die on a
# missing file before any training started.
restore_dataset_from_drive(LEAF)
if RUN_DATA:
    # The extraction is the longest stage.  Each 1k shard is cached to Drive once
    # it completes, so a disconnect costs at most the shard in progress.
    # get_content_words.py streams into combined.jsonl, so an interrupted pass
    # leaves a SHORT file behind and existence alone does not mean completion.
    # Test for "enough rows", not "exactly EXTRACT_SEQUENCES": MAX_SAMPLES is
    # hardcoded to 10000 in that script, so the file legitimately holds 10000 rows
    # even when we only extract the first 4000, and an equality test would delete
    # and regenerate it on every resume.
    combined = LEAF.parent / "combined.jsonl"
    if not combined.is_file() and REUSE_CONTENT_WORDS_FROM:
        donor_tag = REUSE_CONTENT_WORDS_FROM.split("/")[-1].lower()
        donor = DATA / "c4" / donor_tag / "combined.jsonl"
        if not donor.is_file():
            restore_dataset_from_drive(DATA / "c4" / donor_tag / "embedding")
        # The donor is a shortcut, never a requirement.  If it has not been
        # extracted yet, fall through and generate this model's own content words
        # rather than dying in shutil.copy2 -- which is also what lets two models
        # of one family run concurrently on separate GPUs: whichever starts first
        # simply pays the POS pass itself.
        if not donor.is_file():
            print("donor", donor, "not extracted yet; generating content words here")
        else:
            from transformers import AutoTokenizer
            assert (AutoTokenizer.from_pretrained(BASE_MODEL).get_vocab()
                    == AutoTokenizer.from_pretrained(REUSE_CONTENT_WORDS_FROM).get_vocab()), (
                f"{BASE_MODEL} and {REUSE_CONTENT_WORDS_FROM} do not share a vocabulary, "
                "so their content words differ and must be regenerated")
            combined.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(donor, combined)
            print("reused content words from", REUSE_CONTENT_WORDS_FROM)
    if combined.is_file():
        rows = sum(1 for _ in combined.open())
        if rows < EXTRACT_SEQUENCES:
            print(f"combined.jsonl has {rows} rows, need {EXTRACT_SEQUENCES}; regenerating")
            combined.unlink()
        else:
            print(f"combined.jsonl has {rows} rows, using the first {EXTRACT_SEQUENCES}")
    if not combined.is_file():
        run([sys.executable, "data/get_content_words.py", "--model", BASE_MODEL,
             "--dataset", "c4", "--max_length", "256"], cwd=EXT)
        cache_dataset_to_drive(LEAF)
    # An interrupted shard leaves PARTIAL synonyms_/topk_ files behind, so their
    # mere existence does not mean the shard finished.  Record completion in a
    # Drive-side manifest instead; embedding_synonyms.py truncates both files when
    # it restarts a shard, so a re-run is always clean.
    # One manifest per model: a shared name would let the 1B shards mark the 3B
    # ones as finished, and extraction would be skipped entirely.
    shards_done = load_runs(f"task15_shards{_TAG_SUFFIX}")
    for start in range(0, EXTRACT_SEQUENCES, EXTRACT_SHARD):
        end = start + EXTRACT_SHARD
        key = f"{start}_{end}"
        synonym_part = LEAF / f"synonyms_{start}_{end}.jsonl"
        topk_part = LEAF.parent / "prompting" / f"topk_{start}_{end}.jsonl"
        if (RESUME_FINISHED_RUNS and key in shards_done
                and synonym_part.is_file() and topk_part.is_file()):
            print("resume: extraction shard already complete", start, end)
            continue
        run([sys.executable, "data/embedding_synonyms.py", "c4",
             "--start", start, "--end", end, "--model", BASE_MODEL,
             *([] if EXTRACT_4BIT else ["--no-4bit"])], cwd=EXT)
        cache_dataset_to_drive(LEAF)
        shards_done[key] = synonym_part
        save_runs(shards_done, f"task15_shards{_TAG_SUFFIX}")
    run([sys.executable, "data/merge_synonym_parts.py", "--train-size", SPLIT_TRAIN,
         "--val-size", SPLIT_VAL, "--test-size", SPLIT_TEST, "--expected-count", "2", "--force"], cwd=EXT)
    run([sys.executable, "data/augment_synonyms.py", "--base-dir", DATA,
         "--num-augmentations", "4", "--seed", "42", "--overwrite"], cwd=EXT)
    run([sys.executable, "data/randomize_synonyms.py", "--split", "train", "--overwrite"], cwd=EXT)

if RUN_DATA:
    cache_dataset_to_drive(LEAF)
if RUN_DATA:
    run([sys.executable, "data/audit_concept_data.py",
         "--train", LEAF / "synonyms_train.jsonl",
         "--validation", LEAF / "synonyms_val.jsonl",
         "--test", LEAF / "synonyms_test.jsonl",
         "--tokenizer", BASE_MODEL,
         "--expected-train", SPLIT_TRAIN, "--expected-validation", SPLIT_VAL,
         "--expected-test", SPLIT_TEST,
         # Per model: a shared name means the second model's audit silently
         # overwrites the first's, and the provenance of the finished data is
         # exactly what an audit report exists to preserve.
         "--report", DRIVE_RESULTS / f"data_audit{_TAG_SUFFIX}.json"], cwd=EXT)
'''),
        md('''## Unit tests and eight-row GPU smoke run

The smoke run checks the complete QLoRA path before any sweep. It is deleted immediately. The tests cover the released loss, our optional objectives, gradients, and hierarchy sequence scoring.'''),
        code(r'''
if RUN_SMOKE:
    run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=EXT)
    # No map-style preprocessing cache is used. Two independent loads must
    # still produce identical candidate supervision.
    # pip install -e EXT only installs the conceptlib PACKAGE (all pyproject
    # declares); train.py is a loose top-level module, so importing it in-process
    # needs EXT on sys.path.  The subprocess calls are unaffected -- they pass
    # cwd=EXT -- which is why this only bites the in-notebook import.
    if str(EXT) not in sys.path:
        sys.path.insert(0, str(EXT))
    from train import ConceptDataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    first = ConceptDataset(LEAF / "synonyms_train.jsonl", tokenizer, max_samples=8)
    second = ConceptDataset(LEAF / "synonyms_train.jsonl", tokenizer, max_samples=8)
    assert [x["content_words"] for x in first.data] == [x["content_words"] for x in second.data]
    smoke = RUNS / "smoke"
    assert_ephemeral(smoke)
    run([sys.executable, "train.py", "--model-name", BASE_MODEL, "--dataset", "c4",
         "--dataset-type", "embedding", "--concept-loss-weight", "1.0",
         "--max-train-samples", "8", "--num-train-epochs", "1",
         "--output-dir", smoke, "--save-strategy", "no", "--report-to", "none"], cwd=EXT)
    assert (smoke / "adapter_config.json").exists()
    # Required one-model equivalence test: adapter logits and merged logits.
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    adapted = PeftModel.from_pretrained(base, smoke).eval()
    probe = tokenizer("A dog is an animal.", return_tensors="pt")
    with torch.no_grad(): adapter_logits = adapted(**probe).logits
    merged = adapted.merge_and_unload().eval()
    with torch.no_grad(): merged_logits = merged(**probe).logits
    assert torch.allclose(adapter_logits, merged_logits, atol=2e-4, rtol=2e-4)
    del merged, adapted, base
    shutil.rmtree(smoke)
'''),
        md(r'''## Exact reproduction schedule

Seed 42 gets the complete $\lambda\in\{0.25,0.5,0.75,1\}$ curve. The headline NTP, one-epoch augmented NTP, randomized $\lambda=.25$, and concept-marginal $\lambda=1$ settings are then confirmed with seeds 42, 123, and 2024. Released effective batch size is logged. If the headline fails, only seed 42 is rerun with the paper-stated batch before any diagnosis.'''),
        code(TRAIN_HELPER),
        code(r'''
REPRO_RUNS = load_runs(f"task15{_TAG_SUFFIX}")
if RUN_SCREEN:
    REPRO_RUNS["ntp_seed42"] = train_flat("ntp", 42, 0.0)
    for lam in [0.25, 0.5, 0.75, 1.0]:
        REPRO_RUNS[f"zhang_lambda{lam}_seed42"] = train_flat("zhang_marginal", 42, lam)
    REPRO_RUNS["augmented_ntp_seed42"] = train_flat("augmented_ntp", 42, 0.0, epochs=1,
        train_file=LEAF / "synonyms_train_aug5x.jsonl", data_augmentation=True)
    REPRO_RUNS["randomized_seed42"] = train_flat("randomized", 42, 0.25, randomized=True)

# Which families get the extra seeds ("zhang,ntp"; empty = all five).  A machine
# that only needs Zhang's other seeds as seed-matched references for 15b need
# not spend four arms' worth of GPU on the rest.
CONFIRM_FAMILIES = [x.strip() for x in os.environ.get("CONFIRM_ARMS", "").split(",") if x.strip()]
def _confirm(family):
    return not CONFIRM_FAMILIES or family in CONFIRM_FAMILIES
if RUN_CONFIRM:
    for seed in SEEDS:
        if _confirm("ntp"):
            REPRO_RUNS[f"ntp_seed{seed}"] = train_flat("ntp", seed, 0.0)
        if _confirm("augmented_ntp"):
            REPRO_RUNS[f"augmented_ntp_seed{seed}"] = train_flat("augmented_ntp", seed, 0.0,
                epochs=1, train_file=LEAF / "synonyms_train_aug5x.jsonl", data_augmentation=True)
        if _confirm("randomized"):
            REPRO_RUNS[f"randomized_seed{seed}"] = train_flat("randomized", seed, 0.25, randomized=True)
            # Zhang runs the semantic control at lambda=0.25 while the headline
            # concept arm runs at lambda=1.0.  Any claim that concept training does
            # or does not beat the control needs it at the SAME weight, otherwise the
            # comparison confounds the candidate sets with the mixing weight.
            REPRO_RUNS[f"randomized_lambda1.0_seed{seed}"] = train_flat(
                "randomized", seed, 1.0, randomized=True)
        if _confirm("zhang"):
            REPRO_RUNS[f"zhang_seed{seed}"] = train_flat("zhang_marginal", seed, 1.0)
    # The lambda sweep already trained zhang_marginal at lambda=1.0 on seed 42, and
    # adapter_path() maps both calls to the same directory.  Two dict keys pointing at
    # one adapter would score it twice and report it as two arms.
    REPRO_RUNS.pop("zhang_lambda1.0_seed42", None)
save_runs(REPRO_RUNS, f"task15{_TAG_SUFFIX}")

# Use only if the released-batch seed-42 reproduction misses the stated trend.
# This is a named sensitivity run, never a replacement or a cherry-picked row.
RERUN_PAPER_STATED_BATCH = False
if RERUN_PAPER_STATED_BATCH:
    REPRO_RUNS["zhang_seed42_paper_batch"] = train_flat(
        "zhang_marginal_paper_batch", 42, 1.0, batch=2, accum=1)
'''),
        md('''## Evaluation and learning curves

Every checkpoint is scored by the same evaluators. “Global NLL” covers every next-token position; “content-word NLL” covers the semantic slots; “set mass” is total probability assigned to the target-inclusive valid set. SWORDS and bm-semlex are zero-shot here.'''),
        code(r'''
if RUN_EVAL:
    # A fresh session has the manifest but not the weights; pull them back first.
    REPRO_RUNS = restore_all(REPRO_RUNS)
    checkpoints = [BASE_MODEL, *map(str, REPRO_RUNS.values())]
    result_dir = DRIVE_RESULTS / RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)
    # Each evaluator is skipped only when its own output already scores every
    # checkpoint of this pass, so a disconnect costs at most one evaluator
    # instead of the whole section.
    guarded(result_dir / "perplexity.json", checkpoints,
            [sys.executable, "eval/eval_perplexity_explicit.py", "--checkpoints", *checkpoints,
             "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_test.jsonl",
             "--output", result_dir / "perplexity.json"], EXT, "perplexity")
    guarded(result_dir / "concept_sets.json", checkpoints,
            [sys.executable, "eval/eval_concept_sets.py", "--checkpoints", *checkpoints,
             "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_test.jsonl",
             "--output", result_dir / "concept_sets.json"], EXT, "concept sets")
    for label, checkpoint in {"pretrained": BASE_MODEL, **REPRO_RUNS}.items():
        display_label = label.replace("_", " ")
        csv_output = result_dir / f"sts_{display_label}.csv"
        # STS writes one CSV per checkpoint, so it resumes per checkpoint.
        if sts_covered(csv_output):
            print("resume: STS already scored, skipping", display_label)
            continue
        mteb_args = [sys.executable, "eval/eval_mteb.py", "--base-model", BASE_MODEL,
                     "--dataset", "c4", "--dataset-type", "embedding", "--tasks", "sts",
                     "--run-label", display_label,
                     "--csv-output", csv_output,
                     "--mteb-output-root", result_dir / "mteb_raw"]
        mteb_args += ["--no-adapter"] if checkpoint == BASE_MODEL else ["--adapter-path", checkpoint]
        run(mteb_args, cwd=EXT)
    guarded(result_dir / "swords_zero_shot.json", checkpoints,
            [sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_swords.py",
             "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
             "--swords_json", MAIN / "data/swords/swords-v1.1_dev.json.gz",
             "--results_json", result_dir / "swords_zero_shot.json", "--modes", "left", "full"],
            MAIN, "SWORDS")
    # Seconds to run, and each must always match the SWORDS file it reads, so
    # these are never skipped.  Pretrained is the reference row; NTP and
    # augmented NTP are the matched controls the causal claims are stated
    # against.  The index is looked up by label rather than hardcoded: RUN_CONFIRM
    # pops zhang_lambda1.0_seed42, so positions shift once seeds are added and a
    # literal index would quietly compare against the wrong arm.
    # randomized is the one that decides whether the SWORDS gain is semantic:
    # it is trained with the same objective on meaningless candidate sets, so a
    # concept arm that does not separate from it there is buying its GAP with
    # distributional smoothing rather than with concept content.
    baselines = {"pretrained": BASE_MODEL}
    for label in ("ntp_seed42", "augmented_ntp_seed42", "randomized_seed42"):
        if label in REPRO_RUNS:
            baselines[label] = str(REPRO_RUNS[label])
    for label, reference in baselines.items():
        run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/paired_benchmark_ci.py",
             "--kind", "swords", "--results-json", result_dir / "swords_zero_shot.json",
             "--baseline-index", checkpoints.index(reference),
             "--output", result_dir / f"swords_paired_ci_vs_{label}.json"], cwd=MAIN)
    guarded(result_dir / "bm_semlex.json", checkpoints,
            [sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_bm_semlex.py",
             "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
             "--data", MAIN / "data/bm_semlex/curated_200.tsv",
             "--results_json", result_dir / "bm_semlex.json"], MAIN, "bm-semlex")
    manifest = {"pretrained": BASE_MODEL, **{label.replace("_", " "): str(path) for label, path in REPRO_RUNS.items()}}
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    run([sys.executable, MAIN / "scripts/summarize_concept_experiments.py",
         "--manifest", result_dir / "manifest.json", "--result-dir", result_dir,
         "--output", result_dir / "flat_main_table.csv"], cwd=MAIN)
    for label, path in REPRO_RUNS.items(): sync_small_artifacts(path, f"{LOG_DIR}/{label}")
    audit_no_drive_weights()
'''),
        code(r'''
# Plot only scalar learning curves; adapters remain under /content.
import pandas as pd
from matplotlib import pyplot as plt
curves = []
for label, path in REPRO_RUNS.items():
    history = Path(path) / "training_history.jsonl"
    if history.exists():
        frame = pd.read_json(history, lines=True)
        frame["method"] = label
        curves.append(frame)
if not curves:
    print("no training_history.jsonl found; nothing to plot")
else:
    frame = pd.concat(curves, ignore_index=True)
    # The trainer logs ce_loss and concept_loss once per EVAL BATCH, two or three
    # rows sharing one global_step.  Only the row that also carries eval_loss is
    # the aggregate over the validation set; plotting the rest draws intra-eval
    # scatter as if it were training dynamics.
    train_rows = frame.dropna(subset=["loss"])
    eval_rows = frame.dropna(subset=["eval_loss"])
    panels = [(train_rows, "step", "loss", "training loss (NOT comparable across arms:\n"
               "each minimises a different mix of the two terms)"),
              (eval_rows, "epoch", "ce_loss", "validation NTP cross-entropy"),
              (eval_rows, "epoch", "concept_loss", "validation concept loss")]
    # One fixed colour per method.  Letting matplotlib assign them per panel makes
    # the colours shift wherever an arm is dropped for having no concept loss,
    # while the legend sits on the first panel only -- so the same colour means
    # different arms in different panels.
    methods = sorted(frame["method"].unique())
    colours = dict(zip(methods, plt.cm.tab10.colors * (1 + len(methods) // 10)))
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, (rows, x, y, title) in zip(axes, panels):
        if y not in rows:
            continue
        for label in methods:
            group = rows[rows["method"] == label].dropna(subset=[y])
            if not group.empty and group[y].abs().sum() > 0:
                axis.plot(group[x], group[y], marker="o", ms=3, lw=1.4,
                          color=colours[label], label=label)
        axis.set_xlabel(x); axis.set_title(title, fontsize=9); axis.grid(alpha=.25, lw=.5)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(DRIVE_RESULTS / RESULT_DIR / "training_curves.png", dpi=180)
    plt.show()
'''),
        md(r'''## Reproduction gate

Proceed only if Zhang concept marginal beats NTP and augmented NTP on mean STS, improves content-word perplexity over NTP, stays near pretrained global perplexity, and the $\lambda$ curve has the reported direction. If seed 42 fails, rerun that one setting with the paper-stated effective batch and report both configurations—do not silently substitute it.'''),
    ]
    write("research_tasks_15_zhang_reproduction.ipynb", cells)


def task15b():
    cells = [
        md(r"""# Task 15b — Alternative-only concept supervision and contrastive calibration

This notebook starts **only after Task 15 passes**. It adds two objectives to the same Zhang C4 pipeline and scores them against every Task 15 arm.

Terminology used throughout (the word "gold" is not used):

- **observed target** — the token that appeared in the C4 sentence (`chaos` in *"The room descended into chaos."*)
- **generated alternatives** — the extractor's substitute candidates (`turmoil, mayhem, disorder`)
- **human-accepted alternatives** — SWORDS substitutes accepted by annotators
- **target-inclusive set** — `{chaos, turmoil, mayhem, disorder}`; what Zhang's set-marginal supervises
- **alternative-only set** — `{turmoil, mayhem, disorder}`; what our concept term supervises. The observed target is still trained, by ordinary NTP at the slot (`--slot-ntp-weight 1.0`).

| Method | Question |
|---|---|
| Alternative-only uniform | Raise every generated alternative equally, with the observed target trained only by NTP: does probability transfer to *human-accepted* alternatives without the entropy cost the randomized control pays? |
| Target-inclusive uniform (ablation) | Same loss with the observed target inside the concept set: does excluding it matter? |
| Alternative-only + contrastive | Add InfoNCE over alternatives ∪ conservative same-POS negatives: does the model learn which plausible candidates should *not* receive that probability (SWORDS AUROC)? |

$$L = L_{\mathrm{NTP}} + \alpha\Big(-\tfrac{1}{|A(x)|}\sum_{a\in A(x)}\log p(a\mid x)\Big) + \beta\Big[\log\!\!\sum_{c\in A(x)\cup N(x)}\!\!e^{s(c)} - \log\!\sum_{a\in A(x)}e^{s(a)}\Big]$$

The contrastive term is joint token-level training; it is not DPO and not sentence-level SimCSE. Its positive set is the same filtered candidate set the concept term uses, so `--exclude-target` makes both alternative-only at once.

Why this and not more set-marginal: at 3B, Zhang's set-marginal matches a low-weight *randomized* control on SWORDS GAP and AUROC (paired CI crosses zero) and does not lower NLL on human-accepted alternatives at all (−0.05, n.s.), while the randomized arm lowers it by a full nat by flattening everything. The metrics that separate them are STS and perplexity, not SWORDS ranking. The method here has to move human-accepted alternatives without that entropy cost, and has to beat the randomized control, not merely NTP.

Every table carries Task 15's arms — pretrained, NTP, augmented NTP, randomized at $\lambda=.25$ and at the matched $\lambda=1$, and set-marginal at $\lambda=1$ — read from that notebook's manifest."""),
        code(COMMON_SETUP), code(BOOTSTRAP), code(TRAIN_HELPER),
        md("""## Build conservative negatives, and a 50-row sample to read by hand

Candidates must occur in the model’s top-100 next-token pool, match POS, lie outside the alternative set, share no WordNet synset with any alternative, not be a morphological variant, and fall below the contextual-similarity ceiling. Coverage and every rejection reason are reported. The 50-row sample is an error analysis, not an annotation project: read it before trusting any contrastive number."""),
        code(r"""
MODEL_TAG = BASE_MODEL.split("/")[-1].lower()
LEAF = DATA / "c4" / MODEL_TAG / "embedding"
NEG_TRAIN = LEAF / "synonyms_train_conservative_negatives.jsonl"
SCREEN_DIR = DRIVE_RESULTS / f"task15b_screen{_TAG_SUFFIX}"
SCREEN_DIR.mkdir(parents=True, exist_ok=True)
if RUN_DATA:
    run([sys.executable, "data/build_contrastive_negatives.py",
         "--source", LEAF / "synonyms_train.jsonl",
         "--topk", DATA / "c4" / MODEL_TAG / "prompting" / "topk_*.jsonl",
         "--output", NEG_TRAIN,
         # Per model: a shared name would let the 3B pass overwrite the 1B report.
         "--report", DRIVE_RESULTS / f"contrastive_negative_report{_TAG_SUFFIX}.json",
         "--max-cosine", "0.35", "--max-negatives", "20"], cwd=EXT)
    cache_dataset_to_drive(LEAF)
    # Stratified by POS and alternative-set size so the sample cannot be all easy
    # nouns with large sets.  Fixed seed: the same 50 rows every time it is rerun.
    import csv, random
    buckets = {}
    with NEG_TRAIN.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for target in row.get("content_word_responses", []):
                if not target.get("negatives"):
                    continue
                key = (target.get("pos") or "?", "small" if len(target.get("synonyms", [])) <= 2 else "large")
                buckets.setdefault(key, []).append({
                    "pos": key[0], "set_size": key[1], "context": row["input_sequence"],
                    "observed_target": target["word"],
                    "alternatives": " | ".join(target.get("synonyms", [])),
                    "negatives": " | ".join(target["negatives"]),
                    "negative_scores": " | ".join(f"{s:.3f}" for s in target.get("negative_scores", []))})
    rng = random.Random(42)
    picked, per_bucket = [], max(1, 50 // max(1, len(buckets)))
    for key in sorted(buckets):
        picked += rng.sample(buckets[key], min(per_bucket, len(buckets[key])))
    remainder = [item for key in sorted(buckets) for item in buckets[key] if item not in picked]
    picked += rng.sample(remainder, min(50 - len(picked), len(remainder)))
    sample_path = SCREEN_DIR / "negative_sample_50.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(picked[0].keys()) if picked else ["context"])
        writer.writeheader(); writer.writerows(picked)
    print(f"wrote {len(picked)} rows across {len(buckets)} strata to", sample_path)
"""),
        md(r"""## Screen on seed 42

$\alpha\in\{.25,.5,1\}$ controls concept pressure on the alternative-only set; the concept-slot NTP weight is 1 in every arm, so the observed target is always trained. Uniform (mean-log) losses are roughly an order of magnitude larger per slot than set-marginal at the same weight, which is why this grid starts lower than Zhang's.

$\alpha$ is selected on **C4 validation** (lowest alternative NLL subject to the global-NLL, observed-target-NLL and collapse gates printed by the decision cell). Only then is one target-inclusive ablation trained at that $\alpha$, and $\beta\in\{.25,.5,1\}$ screened on **SWORDS dev**. SWORDS test is never read during tuning."""),
        code(r"""
OBJECTIVE_RUNS = load_runs(f"task15b{_TAG_SUFFIX}")
if RUN_SCREEN:
    for alpha in [0.25, 0.5, 1.0]:
        OBJECTIVE_RUNS[f"alternative_uniform_alpha{alpha}_seed42"] = train_flat(
            "alternative_uniform", 42, alpha, objective="uniform", slot_ntp_weight=1.0,
            exclude_target=True)

# Set this ONLY from the alpha gate printed by the decision cell below -- or,
# when REPLICATING on another model, to the value that gate already selected
# elsewhere, passed in as SELECTED_ALPHA=0.5 rather than edited in here.  A
# transferred value is a deliberate choice and belongs in the write-up: it means
# alpha was not tuned on this model, which is the stronger claim and also the
# only honest one once the test set has been read for the first model.
SELECTED_ALPHA = float(os.environ["SELECTED_ALPHA"]) if os.environ.get("SELECTED_ALPHA") else None
if RUN_SCREEN and SELECTED_ALPHA is not None:
    # The one ablation that isolates the exclusion: identical loss, observed
    # target back inside the concept set.  Not a headline method.
    OBJECTIVE_RUNS[f"inclusive_uniform_alpha{SELECTED_ALPHA}_seed42"] = train_flat(
        "inclusive_uniform", 42, SELECTED_ALPHA, objective="uniform", slot_ntp_weight=1.0)
    for beta in [0.25, 0.5, 1.0]:
        OBJECTIVE_RUNS[f"contrast_alpha{SELECTED_ALPHA}_beta{beta}_seed42"] = train_flat(
            "alternative_contrastive", 42, SELECTED_ALPHA, objective="uniform",
            slot_ntp_weight=1.0, exclude_target=True, contrast_beta=beta, train_file=NEG_TRAIN)
    # Frontier point.  Contrastive arms land between alpha=.5 and alpha=1 on BOTH
    # axes, so two uniform points cannot say whether they sit above the alpha
    # curve or merely on it.  Not an alpha-selection candidate: alpha was chosen
    # on validation from {.25,.5,1} before this ran, and this is scored after.
    OBJECTIVE_RUNS["alternative_uniform_alpha0.75_seed42"] = train_flat(
        "alternative_uniform", 42, 0.75, objective="uniform", slot_ntp_weight=1.0,
        exclude_target=True)
"""),
        md("""## Calibrated concept marginalization, and what the negatives were really doing

**Why.** Decompose the SWORDS ranking gain. GAP rises either by pushing rejected candidates down or by pulling accepted ones up, and set-marginal training does only the first. Across Llama-1B (3 seeds), Llama-3B and Qwen3-1.7B it lowers rejected-mass share every time (−.014, −.014, −.021) and never lowers NLL on human-accepted alternatives (−.012 n.s., −.046 n.s., **+.088 worse**). Alternative-only supervision moves precisely that second axis, by the same amount in both families (−.50 nats on Llama, −.485 [−.568,−.403] on Qwen). So the two objectives are not rivals; one contains the other. With $q = p / P(S)$ the model's own distribution inside the set,

$$-\\tfrac{1}{n}\\sum_{c\\in S}\\log p_c \\;=\\; \\underbrace{-\\log P(S)}_{\\text{Zhang}} \\;+\\; \\underbrace{\\mathrm{KL}(u\\,\\|\\,q)}_{\\text{within-set}} \\;+\\; \\log n ,$$

and $\\nabla_z[-\\log P(S)] = p - q\\,\\mathbb{1}_S$: the set marginal is self-distillation toward the model's current within-set distribution, so nothing in it says *which* members deserve mass. A semantically randomized control is a second, weaker probe of the same point: on Llama it reproduces most of the ranking gain (+.008 of Zhang's +.011), on Qwen it reproduces none (−.001). That comparison is therefore **model-dependent and reported as such**; the decomposition above is what replicates. These arms keep Zhang's term exactly as released (target-inclusive, $\\lambda=1$, 0.6 cutoff) and add one term on the alternatives only:

- `uniform` — $-\\tfrac1n\\sum\\log p_a$: raises the alternatives and spreads them.
- `within_kl` — $\\mathrm{KL}(u\\|q_A)$ alone. Its logit gradient is zero outside the alternatives and sums to zero inside them, so it moves mass *between* alternatives and leaves the observed word and the set's total mass to Zhang's term. If SWORDS moves under this arm, the gain is within-set calibration; if only `uniform` moves it, the gain is mass, and the decomposition says so.

**Gate, fixed before any of these is scored (SWORDS dev only):** STS $\\ge .5469$; GAP and AUROC above Zhang with paired intervals excluding zero; above randomized $\\lambda=.25$ on GAP and AUROC; global NLL $\\le$ NTP $+.20$; observed-target NLL $\\le$ NTP $+.10$. The smallest weight that passes is the method. If none passes, the result is the STS-vs-GAP frontier these arms trace, reported as a trade-off.

- `mass` (declared 2026-09-23) — $-\\log P(A)$ alone: the uniform term minus its within-set KL half. At the same $\\gamma$ the two apply the same total push $1-P(A)$ to the alternative set and the same gradient $p_t$ to the observed token; they differ only in allocating that push by $q_a$ (the model's preference) instead of $1/|A|$ (hardest on the least likely alternative, which in this data is enriched for wrong ones). The screen showed `within_kl` alone paying nearly all of `uniform`'s STS cost for a fifth of its GAP, in both families. **Mass rule, declared before any result:** at the locked $\\gamma$ it joins the seed confirmation only if (1) its paired GAP interval against Zhang is above zero, (2) its paired GAP interval against `uniform` at the same $\\gamma$ reaches zero or above, and (3) $\\mathrm{STS}_{mass}-\\mathrm{STS}_{uniform}\\ge\\tfrac12(\\mathrm{STS}_{Zhang}-\\mathrm{STS}_{uniform})$. If it passes, the two larger weights are scored as a curve, never as a second selection. Its gradient on the observed token equals `uniform`'s, so it is not expected to recover observed-target NLL or content PPL.

**Confirmation.** The selected arm at seeds 123 and 2024 on the same original file as Task 15's three-seed baselines, each seed gated against Zhang of the same seed. CoInCo dev (a fixed 800-target subset) is scored beside SWORDS dev for breadth; it is reported, not gated on.

**The negatives.** A hand read of `negative_sample_50.csv` (2026-09-18): the contrastive loss only fires where a slot has an alternative (31/50), and in 23 of those 31 every negative is a letter or a word-prefix token; 2 of 31 have a semantically meaningful negative. The 0.35 similarity ceiling rejects every real word in a slot with rich alternatives, so only non-words survive, and WordNet lists them. Two controls settle what the published +.005 GAP was: `clean` (complete words in their dominant POS, plus antonyms of the observed word) and `fragments` (only what `clean` throws away). Effective coverage — slots with an alternative AND a negative — is printed for both; the raw 45.6% is not the supervised fraction."""),
        code(r"""
NEG_VARIANTS = {"clean": ["--strict-lexical", "--antonyms"], "fragments": ["--fragments-only"]}
NEG_FILES = {name: LEAF / f"synonyms_train_negatives_{name}.jsonl" for name in NEG_VARIANTS}
# The grid the gate was pre-registered over on 2026-09-18, before any of it was run.
PREREGISTERED_GRID = [("uniform", 0.25), ("uniform", 0.5),
                      ("within_kl", 0.25), ("within_kl", 0.5), ("within_kl", 1.0)]
# Added 2026-09-21, AFTER that grid was screened and every arm failed on STS alone --
# Llama uniform g0.25 missed the floor by .0004 (the three-seed STS sd is .0005), Qwen
# by .0063.  These two resolve the knee of the frontier between Zhang and g=0.25.  They
# are frontier points, NOT gate candidates: hybrid_gate() scores and prints them but
# refuses to select them, because extending a grid downward after reading a near-miss is
# exactly how a pre-registration gets spent.  The curve was the pre-registered outcome
# when nothing passed; adding points to a curve is resolution, not a second attempt.
POSTHOC_GRID = [("uniform", 0.125), ("uniform", 0.0625)]
# Declared 2026-09-23, before any of it was trained.  `mass` is the uniform auxiliary
# with its within-set KL half removed (uniform = mass + within_kl + log n, exactly):
# same total push on the alternative set, same gradient on the observed token and on
# every other token, differing ONLY in how the push is split among the alternatives
# (q_a, the model's own preference, instead of 1/n).  The first weight is the family's
# locked gamma from the frontier; the two above it are the curve, run only if the
# first passes the rule in the decision cell.  Neither pre-registered nor post-hoc:
# the mass rule below is its own, declared gate.
LOCKED_GAMMA = 0.125 if MODEL_TAG.startswith("llama") else 0.0625
MASS_GRID = [("mass", LOCKED_GAMMA * m) for m in (1, 2, 4)]
HYBRID_GRID = PREREGISTERED_GRID + POSTHOC_GRID + MASS_GRID
# "uniform:0.125" -- set ONLY from the gate below, then rerun with RUN_MULTISEED.
SELECTED_HYBRID = os.environ.get("SELECTED_HYBRID")
# Train only these grid arms in THIS process ("uniform:0.125,mass:0.125"; empty =
# all).  Two processes on two GPUs can then split the arms; the manifest merges.
HYBRID_ONLY = [(k, float(w)) for k, w in
               (x.split(":") for x in os.environ.get("HYBRID_ARMS", "").split(",") if x.strip())]
assert all(arm in HYBRID_GRID for arm in HYBRID_ONLY), f"HYBRID_ARMS not in the grid: {HYBRID_ONLY}"

def hybrid_label(kind, weight, seed=42):
    return f"zhang_plus_{kind}_g{weight}_seed{seed}"

def dedupe_runs(runs):
    # One label per adapter.  Older passes labelled the selected arm's seeds
    # `calibrated_seed*` and popped the grid label; the grid label is the descriptive
    # one, so a `calibrated_*` alias of an adapter that has another label is dropped.
    by_path = {}
    for label, path in runs.items():
        by_path.setdefault(str(path), []).append(label)
    keep = {}
    for label, path in runs.items():
        aliases = by_path[str(path)]
        if label.startswith("calibrated_seed") and any(not a.startswith("calibrated_seed") for a in aliases):
            continue
        keep[label] = path
    return keep

def _merge_save(runs, name):
    # Merge into what is on disk, never overwrite: another process may be adding
    # disjoint arms to the same manifest right now.  Deduped AFTER the merge, or
    # an alias dropped here would come straight back from disk.
    merged = dedupe_runs({**load_runs(name), **runs})
    save_runs(merged, name)
    return merged

OBJECTIVE_RUNS = dedupe_runs(OBJECTIVE_RUNS)

if RUN_HYBRID:
    for kind, weight in HYBRID_GRID:
        if HYBRID_ONLY and (kind, weight) not in HYBRID_ONLY:
            continue
        OBJECTIVE_RUNS[hybrid_label(kind, weight)] = train_flat(
            "zhang_plus_aux", 42, 1.0, alt_aux=kind, alt_aux_weight=weight)
        OBJECTIVE_RUNS = _merge_save(OBJECTIVE_RUNS, f"task15b{_TAG_SUFFIX}")
    # Confirmation of the arm the gate below selected, at the other seeds, on the
    # SAME original data as the Task 15 baselines, so the main table is mean +- sd
    # over three seeds for every arm on one file.  RUN_CONFIRM is NOT used for this:
    # that flag relaunches the alternative-uniform and legacy contrastive arms.
    if SELECTED_HYBRID:
        kind, weight = SELECTED_HYBRID.split(":"); weight = float(weight)
        assert (kind, weight) in HYBRID_GRID, "SELECTED_HYBRID must be one of the screened arms"
        for seed in SEEDS:
            if seed == 42 or (HYBRID_ONLY and (kind, weight) not in HYBRID_ONLY):
                continue                      # seed 42 IS the grid arm above
            OBJECTIVE_RUNS[hybrid_label(kind, weight, seed)] = train_flat(
                "zhang_plus_aux", seed, 1.0, alt_aux=kind, alt_aux_weight=weight)
            OBJECTIVE_RUNS = _merge_save(OBJECTIVE_RUNS, f"task15b{_TAG_SUFFIX}")

if RUN_NEGATIVE_CONTROLS:
    topk_glob = DATA / "c4" / MODEL_TAG / "prompting" / "topk_*.jsonl"
    if not list(topk_glob.parent.glob(topk_glob.name)):
        print("no top-k shards restored; the negative controls are skipped, not faked")
    else:
        assert SELECTED_ALPHA is not None, "the negative controls reuse the locked alpha"
        for name, flags in NEG_VARIANTS.items():
            report = DRIVE_RESULTS / f"contrastive_negative_report_{name}{_TAG_SUFFIX}.json"
            if not NEG_FILES[name].is_file():
                run([sys.executable, "data/build_contrastive_negatives.py",
                     "--source", LEAF / "synonyms_train.jsonl", "--topk", topk_glob,
                     "--output", NEG_FILES[name], "--report", report,
                     "--max-cosine", "0.35", "--max-negatives", "20",
                     # Coverage counted the way the TRAINER counts: a multi-token
                     # candidate never reaches the loss, so the raw fraction
                     # overstates what is actually supervised.
                     "--tokenizer", BASE_MODEL, *flags], cwd=EXT)
            if report.is_file():
                stats = json.loads(report.read_text())
                print(f"{name}: raw coverage {stats['negative_coverage']:.3f}, "
                      f"EFFECTIVE contrastive coverage {stats['effective_contrastive_coverage']:.3f}")
            # A distinct method name per file: adapter_path() does not see train_file.
            OBJECTIVE_RUNS[f"contrast_{name}_negatives_seed42"] = train_flat(
                f"alternative_contrastive_{name}", 42, SELECTED_ALPHA, objective="uniform",
                slot_ntp_weight=1.0, exclude_target=True, contrast_beta=1.0,
                train_file=NEG_FILES[name])

if RUN_HYBRID or RUN_NEGATIVE_CONTROLS:
    OBJECTIVE_RUNS = _merge_save(OBJECTIVE_RUNS, f"task15b{_TAG_SUFFIX}")
"""),
        md("""## Evaluation

Every checkpoint of this pass — Task 15's arms and this notebook's — is scored by the same evaluators. A validation pass of the perplexity and concept-set evaluators exists only for choosing $\\alpha$; every other number is C4 test, SWORDS dev, the nine STS tasks and bm-semlex."""),
        code(r"""
if RUN_EVAL and not OBJECTIVE_RUNS:
    print("no screen arms for this model here; the screen evaluation is skipped, "
          "not run on the baselines alone")
if RUN_EVAL and OBJECTIVE_RUNS:
    OBJECTIVE_RUNS = restore_all(OBJECTIVE_RUNS)
    # The question is "does this beat Zhang", so Zhang's arms sit IN this table:
    # pretrained as the reference row, NTP and augmented NTP as matched controls,
    # randomized at both weights as the semantic control, and set-marginal at
    # lambda=1 as the method being improved on.  They come from Task 15's manifest;
    # an arm Task 15 never trained is reported as absent, never retrained here.
    task15_runs = restore_all(load_runs(f"task15{_TAG_SUFFIX}"))
    BASELINE_LABELS = ("ntp_seed42", "augmented_ntp_seed42", "randomized_seed42",
                       "randomized_lambda1.0_seed42", "zhang_seed42", "zhang_lambda1.0_seed42")
    baseline_runs = {label: task15_runs[label] for label in BASELINE_LABELS if label in task15_runs}
    # Zhang's other seeds, when Task 15 trained them: the seed-matched references
    # for the selected hybrid's confirmation.
    for label in ("zhang_seed123", "zhang_seed2024"):
        if label in task15_runs:
            baseline_runs[label] = task15_runs[label]
    # RUN_CONFIRM in Task 15 pops zhang_lambda1.0_seed42 in favour of zhang_seed42;
    # both name ONE adapter, so keep whichever exists and never both.
    if "zhang_seed42" in baseline_runs:
        baseline_runs.pop("zhang_lambda1.0_seed42", None)
    for label in BASELINE_LABELS[:-1]:
        if label not in baseline_runs and not (label == "zhang_seed42" and "zhang_lambda1.0_seed42" in baseline_runs):
            print("Task 15 never trained this baseline; the table will lack it:", label)
    all_runs = {**baseline_runs, **OBJECTIVE_RUNS}
    checkpoints = [BASE_MODEL, *map(str, all_runs.values())]
    result_dir = SCREEN_DIR
    # Each evaluator is skipped only when its own output already scores every
    # checkpoint of this pass, so a disconnect costs at most one evaluator.
    # VALIDATION pass first: this is the only thing the alpha choice may read.
    guarded(result_dir / "val_perplexity.json", checkpoints,
            [sys.executable, "eval/eval_perplexity_explicit.py", "--checkpoints", *checkpoints,
             "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_val.jsonl",
             "--output", result_dir / "val_perplexity.json"], EXT, "validation perplexity")
    guarded(result_dir / "val_concept_sets.json", checkpoints,
            [sys.executable, "eval/eval_concept_sets.py", "--checkpoints", *checkpoints,
             "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_val.jsonl",
             "--output", result_dir / "val_concept_sets.json"], EXT, "validation concept sets")
    guarded(result_dir / "perplexity.json", checkpoints,
            [sys.executable, "eval/eval_perplexity_explicit.py", "--checkpoints", *checkpoints,
             "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_test.jsonl",
             "--output", result_dir / "perplexity.json"], EXT, "perplexity")
    guarded(result_dir / "concept_sets.json", checkpoints,
            [sys.executable, "eval/eval_concept_sets.py", "--checkpoints", *checkpoints,
             "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_test.jsonl",
             "--output", result_dir / "concept_sets.json"], EXT, "concept sets")
    task15_dir = DRIVE_RESULTS / RESULT_DIR
    for label, checkpoint in {"pretrained": BASE_MODEL, **all_runs}.items():
        display_label = label.replace("_", " ")
        csv_output = result_dir / f"sts_{display_label}.csv"
        # STS is deterministic per checkpoint and Task 15 already scored the
        # baselines, so reuse its file rather than spending 3.5 min re-deriving it.
        previous = task15_dir / csv_output.name
        if not csv_output.is_file() and previous.is_file():
            shutil.copy2(previous, csv_output)
        if sts_covered(csv_output):
            print("resume: STS already scored, skipping", display_label)
            continue
        mteb_args = [sys.executable, "eval/eval_mteb.py", "--base-model", BASE_MODEL,
                     "--dataset", "c4", "--dataset-type", "embedding", "--tasks", "sts",
                     "--run-label", display_label, "--csv-output", csv_output,
                     "--mteb-output-root", result_dir / "mteb_raw"]
        mteb_args += ["--no-adapter"] if checkpoint == BASE_MODEL else ["--adapter-path", checkpoint]
        run(mteb_args, cwd=EXT)
    guarded(result_dir / "swords.json", checkpoints,
            [sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_swords.py",
             "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
             "--swords_json", MAIN / "data/swords/swords-v1.1_dev.json.gz",
             "--results_json", result_dir / "swords.json", "--modes", "left", "full"], MAIN, "SWORDS")
    # A second human-labelled substitution benchmark, same evaluator, same metrics:
    # a fixed 800-target subset of CoInCo dev (builddataset/build_lexsub_benchmarks.py).
    # Reported for breadth beside SWORDS dev; no gate reads it.
    guarded(result_dir / "coinco_dev.json", checkpoints,
            [sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_swords.py",
             "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
             "--swords_json", MAIN / "data/lexsub/coinco_dev_sub800.json.gz",
             "--results_json", result_dir / "coinco_dev.json", "--modes", "left", "full"], MAIN, "CoInCo dev")
    # One paired interval per reference, looked up by label so an index can never
    # drift onto the wrong arm.  vs zhang is the headline; vs randomized is what
    # says whether a gain is semantic rather than distributional; vs the selected
    # alternative-uniform arm is the only fair test of the contrastive term itself.
    references = {"pretrained": BASE_MODEL, **{label: str(path) for label, path in baseline_runs.items()}}
    # The uniform hybrid at the locked gamma: the reference the mass rule pairs against.
    if hybrid_label("uniform", LOCKED_GAMMA) in OBJECTIVE_RUNS:
        references[hybrid_label("uniform", LOCKED_GAMMA)] = str(OBJECTIVE_RUNS[hybrid_label("uniform", LOCKED_GAMMA)])
    # EVERY alternative-only uniform arm is a reference, not just the selected one.
    # Contrastive has to beat the cheaper way of buying the same concept pressure,
    # which is turning alpha up.  A same-alpha comparison alone cannot show that:
    # it credits the negatives with a gain a larger alpha also delivers, which is
    # the identical error this paper accuses set-marginal training of.
    # Includes the confirmation arms (alternative_uniform_seed123, ...), so each
    # contrastive seed has a paired interval against the uniform arm of the SAME
    # seed -- the one comparison that isolates the contrastive term across seeds.
    for key, path in OBJECTIVE_RUNS.items():
        if key.startswith("alternative_uniform"):
            references[key] = str(path)
    for label, reference in references.items():
        for bench in ("swords", "coinco_dev"):
            run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/paired_benchmark_ci.py",
                 "--kind", "swords", "--results-json", result_dir / f"{bench}.json",
                 "--baseline-index", checkpoints.index(reference),
                 "--output", result_dir / f"{bench}_paired_ci_vs_{label}.json"], cwd=MAIN)
    guarded(result_dir / "bm_semlex.json", checkpoints,
            [sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_bm_semlex.py",
             "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
             "--data", MAIN / "data/bm_semlex/curated_200.tsv",
             "--results_json", result_dir / "bm_semlex.json"], MAIN, "bm-semlex")
    manifest = {"pretrained": BASE_MODEL, **{label.replace("_", " "): str(path) for label, path in all_runs.items()}}
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    run([sys.executable, MAIN / "scripts/summarize_concept_experiments.py",
         "--manifest", result_dir / "manifest.json", "--result-dir", result_dir,
         "--output", result_dir / "flat_extension_table.csv"], cwd=MAIN)
"""),
        md("""## Decision cell

Reads only what the evaluation cell wrote and prints every gate with its number, so the choice is auditable from the notebook output alone. First pass: the $\\alpha$ gate (validation only). After `SELECTED_ALPHA` is set and the $\\beta$ arms are scored: the promotion gates and the headline.

Headline rule: contrastive if it passes every promotion gate; otherwise alternative-only uniform if it passes the same gates minus the vs-itself clause; otherwise this is a negative result and is reported as one."""),
        code(r"""
import csv
GLOBAL_NLL_SLACK, OBSERVED_NLL_SLACK, STS_SLACK, BM_SLACK, MIN_PROB_RATIO = 0.20, 0.10, 0.005, 0.02, 0.5
result_dir = SCREEN_DIR

def _by_ckpt(name):
    path = result_dir / name
    return {str(r["checkpoint"]): r for r in json.loads(path.read_text())} if path.is_file() else {}

def _sts(label):
    path = result_dir / f"sts_{label.replace('_', ' ')}.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        scores = [float(r["main_score"]) for r in csv.DictReader(handle)]
    return sum(scores) / len(scores) if scores else None

def _paired(reference_label, candidate):
    path = result_dir / f"swords_paired_ci_vs_{reference_label}.json"
    if not path.is_file():
        return {}
    for entry in json.loads(path.read_text()):
        if str(entry["candidate"]) == str(candidate):
            return entry["metrics"]
    return {}

def _sig(metrics, key, better):
    # (candidate - reference, True when the 95% CI excludes zero on the good side)
    value = metrics.get(key)
    if not value or value.get("candidate_minus_baseline") is None:
        return None, None
    lo, hi = value["ci95"]
    return value["candidate_minus_baseline"], (lo > 0) if better == "up" else (hi < 0)

def _fmt(x):
    return "n/a" if x is None else f"{x:.4f}"

manifest_path = result_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
runs = {label.replace(" ", "_"): str(path) for label, path in manifest.items()}
ntp = runs.get("ntp_seed42")
zhang_label = "zhang_seed42" if "zhang_seed42" in runs else "zhang_lambda1.0_seed42"
zhang, randomized = runs.get(zhang_label), runs.get("randomized_seed42")
val_c, val_p = _by_ckpt("val_concept_sets.json"), _by_ckpt("val_perplexity.json")
tst_p, bm = _by_ckpt("perplexity.json"), _by_ckpt("bm_semlex.json")

print("== alpha gate: C4 VALIDATION only ==")
g_ntp = val_p.get(ntp, {}).get("global", {}).get("mean_nll")
# The observed target is referenced against NTP, never against Zhang.  Zhang's
# supervised set CONTAINS the observed token, so its objective actively drives
# that NLL below NTP's; asking an alternative-only arm to match it would be
# asking it to do the one thing it exists not to do.  NTP is the matched control,
# and the question here is only "is the observed target damaged?" -- which
# --slot-ntp-weight 1.0 is what prevents.
o_ntp = val_c.get(ntp, {}).get("observed_target_nll")
mp_zhang = val_c.get(zhang, {}).get("minimum_candidate_probability")
alpha_rows = []
for label, ck in runs.items():
    if not label.startswith("alternative_uniform_alpha"):
        continue
    c, p = val_c.get(ck, {}), val_p.get(ck, {})
    alt, g = c.get("alternative_nll"), p.get("global", {}).get("mean_nll")
    o, mp = c.get("observed_target_nll"), c.get("minimum_candidate_probability")
    gates = {"global<=ntp+.20": None if None in (g, g_ntp) else g <= g_ntp + GLOBAL_NLL_SLACK,
             "observed<=ntp+.10": None if None in (o, o_ntp) else o <= o_ntp + OBSERVED_NLL_SLACK,
             "min_prob>=.5*zhang": None if None in (mp, mp_zhang) else mp >= MIN_PROB_RATIO * mp_zhang}
    alpha_rows.append((label, alt, gates))
    print(f"  {label:38} alt_nll={_fmt(alt)} global={_fmt(g)} observed={_fmt(o)} min_prob={_fmt(mp)}  "
          + "  ".join(f"{k}:{'n/a' if v is None else ('PASS' if v else 'FAIL')}" for k, v in gates.items()))
passing = [r for r in alpha_rows if r[1] is not None and all(r[2].values())]
if passing:
    best = min(passing, key=lambda r: r[1])
    print("  -> SELECTED_ALPHA =", best[0].split("alpha", 1)[1].split("_", 1)[0])
elif alpha_rows:
    print("  -> no alpha passes every gate.  Report that; do not loosen the gates.")
else:
    print("  (no alternative-uniform arms scored yet)")

print("== promotion gates: SWORDS DEV paired intervals, C4 test, STS, bm-semlex ==")
g_ntp_test = tst_p.get(ntp, {}).get("global", {}).get("mean_nll")
sts_zhang = _sts(zhang_label)
acc_zhang = bm.get(zhang, {}).get("left", {}).get("accuracy")

def promotion(label, ck, self_label=None):
    vz, vr = _paired(zhang_label, ck), _paired("randomized_seed42", ck)
    vs = _paired(self_label, ck) if self_label else {}
    g = tst_p.get(ck, {}).get("global", {}).get("mean_nll")
    sts, acc = _sts(label), bm.get(ck, {}).get("left", {}).get("accuracy")
    d_gap_z, s_gap_z = _sig(vz, "gap", "up")
    d_auroc_s, s_auroc_s = _sig(vs, "auroc", "up")
    d_gap_r, _ = _sig(vr, "gap", "up")
    _, s_auroc_r = _sig(vr, "auroc", "up")
    _, s_rms_r = _sig(vr, "rejected_mass_share", "down")
    d_alt_z, s_alt_z = _sig(vz, "alternatives_nll", "down")
    # Frontier gate.  An arm is DOMINATED when some uniform arm already matches or
    # beats its ranking at no greater language-modelling cost -- that arm's gain is
    # the alpha curve, not the negatives.  Uniform arms that cost MORE global NLL
    # are excluded: losing to a costlier arm is a trade, not a domination.
    dominated = []
    for u_label, u_ck in (runs.items() if self_label else ()):
        u_global = tst_p.get(u_ck, {}).get("global", {}).get("mean_nll")
        if not u_label.startswith("alternative_uniform_alpha") or str(u_ck) == str(ck):
            continue
        if None in (u_global, g) or u_global > g:
            continue
        _, beats = _sig(_paired(u_label, ck), "gap", "up")
        if beats is not None:
            dominated.append(not beats)
    on_frontier = None if not dominated else not any(dominated)
    # Two groups, reported and decided separately.  DISCRIMINATION is the claim:
    # does this arm rank human-labelled substitutes better than Zhang, and better
    # than the randomized control Zhang cannot separate from?  RETENTION is the
    # price: are STS and language modelling preserved?  Collapsing them into one
    # verdict turns "wins the claim, pays on STS" -- a reportable trade-off and
    # the most likely real outcome -- into the same output as "nothing worked".
    discrimination = {
        f"GAP > zhang (CI excl 0)  [{_fmt(d_gap_z)}]": None if s_gap_z is None else bool(s_gap_z),
        f"AUROC > same non-contrastive arm (CI excl 0)  [{_fmt(d_auroc_s)}]": (None if not self_label or s_auroc_s is None else bool(s_auroc_s)),
        f"GAP >= randomized lambda.25  [{_fmt(d_gap_r)}]": None if d_gap_r is None else d_gap_r >= 0,
        "AUROC or rejected-mass share better than randomized (CI excl 0)": (None if s_auroc_r is None and s_rms_r is None else bool(s_auroc_r or s_rms_r)),
        f"human-accepted alt NLL < zhang (CI excl 0)  [{_fmt(d_alt_z)}]": None if s_alt_z is None else bool(s_alt_z),
        "GAP > every uniform arm costing no more global NLL (CI excl 0)": on_frontier,
    }
    retention = {
        f"global NLL <= ntp+.20  [{_fmt(g)} vs {_fmt(g_ntp_test)}]": None if None in (g, g_ntp_test) else g <= g_ntp_test + GLOBAL_NLL_SLACK,
        f"STS >= zhang-.005  [{_fmt(sts)} vs {_fmt(sts_zhang)}]": None if None in (sts, sts_zhang) else sts >= sts_zhang - STS_SLACK,
        f"bm-semlex >= zhang-2pp  [{_fmt(acc)} vs {_fmt(acc_zhang)}]": None if None in (acc, acc_zhang) else acc >= acc_zhang - BM_SLACK,
    }
    print(f"  {label}")
    for heading, group in (("discrimination (the claim)", discrimination), ("retention (the price)", retention)):
        print(f"    {heading}")
        for name, ok in group.items():
            print(f"      {'n/a ' if ok is None else ('PASS' if ok else 'FAIL')}  {name}")
    def _all(group):
        known = [ok for ok in group.values() if ok is not None]
        return bool(known) and all(known)
    return _all(discrimination), _all(retention)

winners = {"contrastive": [], "alternative_uniform": []}
retained = {}
for label, ck in runs.items():
    if label.startswith("contrast_alpha"):
        alpha = label.split("alpha", 1)[1].split("_", 1)[0]
        discriminates, retains = promotion(label, ck, self_label=f"alternative_uniform_alpha{alpha}_seed42")
        if discriminates:
            winners["contrastive"].append(label)
        retained[label] = retains
    elif label.startswith("alternative_uniform_alpha"):
        discriminates, retains = promotion(label, ck)
        if discriminates:
            winners["alternative_uniform"].append(label)
        retained[label] = retains

def announce(kind, labels):
    for label in labels:
        price = ("and pays nothing: every retention gate holds" if retained.get(label)
                 else "but FAILS a retention gate above -- that trade-off is a result, report it")
        print(f"  {kind} wins every discrimination gate: {label} {price}")

print("== headline ==")
if winners["contrastive"]:
    announce("contrastive", winners["contrastive"])
    print("  -> beta is TUNED on SWORDS dev (test is never read here), so any passing")
    print("     beta is admissible.  Take the smallest unless the frontier margin is")
    print("     monotone in beta, in which case take the largest that still retains,")
    print("     and record the choice and its reason before the seeds are run.")
elif winners["alternative_uniform"]:
    announce("alternative-only uniform", winners["alternative_uniform"])
    print("  -> contrastive did not separate from it; the uniform arm is the headline candidate")
else:
    print("  no arm wins every discrimination gate: this is a negative result and is reported as one")
"""),
        md("""## Hybrid gate (automatic)

Applied to the H1/H2 arms only, on SWORDS **dev**, and evaluated before any of these numbers is read by hand. Thresholds are derived from this notebook's own Zhang and NTP rows rather than hardcoded, so the same cell gates a second model without edits.

An arm passes only if it beats Zhang on the axis Zhang provably does not move (human-accepted alternative NLL) **and** on ranking, stays within the retention allowances, and also beats the randomized control. The smallest weight that passes is selected; ties go to the smaller weight. If nothing passes, that is the result and the paper reports the trade-off curve — the cell does not relax a threshold to manufacture a winner."""),
        code(r"""
STS_ALLOWANCE = 0.005        # vs Zhang
GLOBAL_NLL_ALLOWANCE = 0.20  # vs NTP
OBSERVED_NLL_ALLOWANCE = 0.10

def _gate_rows():
    table = SCREEN_DIR / "flat_extension_table.csv"
    if not table.is_file():
        print("no flat_extension_table.csv yet; run RUN_EVAL first"); return None
    import csv
    return {r["method"]: r for r in csv.DictReader(table.open())}

def _ci(path, run_path, metric):
    # Returns (delta, significant) for one candidate in one paired-CI file.
    if not Path(path).is_file():
        return None
    for entry in json.loads(Path(path).read_text()):
        if str(entry["candidate"]) != str(run_path):
            continue
        if metric not in entry["metrics"]:
            return None
        d = entry["metrics"][metric]
        low, high = d["ci95"]
        return d["candidate_minus_baseline"], (low * high > 0)
    return None

def _ci_bounds(path, run_path, metric):
    # The (low, high) 95% interval for one candidate in one paired-CI file.
    if not Path(path).is_file():
        return None
    for entry in json.loads(Path(path).read_text()):
        if str(entry["candidate"]) == str(run_path) and metric in entry["metrics"]:
            return tuple(entry["metrics"][metric]["ci95"])
    return None

def hybrid_gate(verbose=True):
    rows = _gate_rows()
    if rows is None: return None
    zhang = rows.get("zhang seed42") or rows.get("zhang lambda1.0 seed42")
    ntp = rows.get("ntp seed42")
    if not zhang or not ntp:
        print("Task 15 baselines missing from the table; cannot gate"); return None
    sts_floor = float(zhang["sts_mean"]) - STS_ALLOWANCE
    nll_ceiling = float(ntp["global_nll"]) + GLOBAL_NLL_ALLOWANCE
    obs_ceiling = float(ntp["swords_observed_target_nll"]) + OBSERVED_NLL_ALLOWANCE
    print(f"thresholds -> STS >= {sts_floor:.4f} | global NLL <= {nll_ceiling:.4f} "
          f"| observed-target NLL <= {obs_ceiling:.4f}")

    vs_zhang = SCREEN_DIR / ("swords_paired_ci_vs_zhang_seed42.json"
                             if (SCREEN_DIR / "swords_paired_ci_vs_zhang_seed42.json").is_file()
                             else "swords_paired_ci_vs_zhang_lambda1.0_seed42.json")
    vs_rand = SCREEN_DIR / "swords_paired_ci_vs_randomized_seed42.json"

    passing = []
    for kind, weight in HYBRID_GRID:
        label = f"zhang_plus_{kind}_g{weight}_seed42"
        run_path = OBJECTIVE_RUNS.get(label)
        row = rows.get(label.replace("_", " "))
        if row is None or run_path is None:
            if verbose: print(f"  {label:34} not trained/scored yet")
            continue
        checks = {
            "STS": float(row["sts_mean"]) >= sts_floor,
            "globalNLL": float(row["global_nll"]) <= nll_ceiling,
            "obsNLL": float(row["swords_observed_target_nll"]) <= obs_ceiling,
        }
        for metric, key, want_negative in (("alternatives_nll", "altNLL<Zhang", True),
                                           ("gap", "GAP>Zhang", False),
                                           ("auroc", "AUROC>Zhang", False)):
            got = _ci(vs_zhang, run_path, metric)
            checks[key] = bool(got and got[1] and ((got[0] < 0) == want_negative))
        got = _ci(vs_rand, run_path, "gap")
        checks["GAP>random"] = bool(got and got[1] and got[0] > 0)

        ok = all(checks.values())
        posthoc = (kind, weight) in POSTHOC_GRID or (kind, weight) in MASS_GRID
        if verbose:
            failed = [k for k, v in checks.items() if not v]
            print(f"  {label:36} {'PASS' if ok else 'FAIL'}"
                  f"{'  [post-hoc, not selectable]' if (kind, weight) in POSTHOC_GRID else ''}"
                  f"{'  [mass: own rule below]' if (kind, weight) in MASS_GRID else ''}"
                  f"  STS {float(row['sts_mean']):.4f}"
                  f"  gNLL {float(row['global_nll']):.4f}"
                  f"  altNLL {float(row['swords_alternative_nll']):.3f}"
                  + ("" if ok else f"   failed: {', '.join(failed)}"))
        # A post-hoc point never enters `passing`: it is reported on the frontier curve
        # and cannot become the method by passing a gate it was added after reading.
        if ok and not posthoc: passing.append((weight, kind))

    mass_rule(rows, vs_zhang)
    confirmation(rows)
    breadth(rows)

    if not passing:
        print("\nNo hybrid arm passes every gate. That is the result: report the "
              "STS-vs-substitution trade-off curve, do not relax a threshold.")
        return None
    weight, kind = sorted(passing)[0]
    print(f"\nSELECTED_HYBRID = {kind}:{weight}   (smallest passing weight of "
          f"{len(passing)}; set it in the environment and rerun with RUN_MULTISEED)")
    return f"{kind}:{weight}"

def mass_rule(rows, vs_zhang):
    # The declared rule for the mass arm at the locked gamma (see the markdown above).
    print("\n== mass rule: uniform minus its within-set KL half, at the locked gamma ==")
    mass_label, uni_label = hybrid_label("mass", LOCKED_GAMMA), hybrid_label("uniform", LOCKED_GAMMA)
    zhang = rows.get("zhang seed42") or rows.get("zhang lambda1.0 seed42")
    mass_row, uni_row = rows.get(mass_label.replace("_", " ")), rows.get(uni_label.replace("_", " "))
    mass_path = OBJECTIVE_RUNS.get(mass_label)
    if not (mass_row and uni_row and zhang and mass_path):
        print("  not scored yet (needs", mass_label, "and", uni_label, "in this table)"); return
    vs_uni = _ci_bounds(SCREEN_DIR / f"swords_paired_ci_vs_{uni_label}.json", mass_path, "gap")
    got = _ci(vs_zhang, mass_path, "gap")
    sts = {k: float(r["sts_mean"]) for k, r in (("mass", mass_row), ("uniform", uni_row), ("zhang", zhang))}
    need = sts["uniform"] + 0.5 * (sts["zhang"] - sts["uniform"])
    rule = {
        "(1) GAP above Zhang, paired interval > 0": bool(got and got[1] and got[0] > 0),
        "(2) GAP vs uniform at the same gamma, interval reaches >= 0": bool(vs_uni and vs_uni[1] >= 0),
        f"(3) STS {sts['mass']:.4f} >= {need:.4f} (half of uniform's STS cost given back)": sts["mass"] >= need,
    }
    for k, v in rule.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("  -> " + (f"mass passes: confirm it across seeds (SELECTED_HYBRID=mass:{LOCKED_GAMMA}) and "
                     f"score the curve (HYBRID_ARMS=mass:{LOCKED_GAMMA * 2},mass:{LOCKED_GAMMA * 4})"
                     if all(rule.values()) else "mass does not pass; uniform stays the method"))

def confirmation(rows):
    # The selected arm's other seeds, each against Zhang of the SAME seed.
    if not SELECTED_HYBRID:
        return
    kind, weight = SELECTED_HYBRID.split(":"); weight = float(weight)
    print(f"\n== confirmation: {kind}:{weight} vs Zhang, seed-matched ==")
    for seed in (42, 123, 2024):
        label = hybrid_label(kind, weight, seed)
        zlabel = "zhang_seed42" if seed == 42 and "zhang seed42" in rows else f"zhang_seed{seed}"
        if seed == 42 and zlabel not in rows and "zhang lambda1.0 seed42" in rows:
            zlabel = "zhang_lambda1.0_seed42"
        run_path, row = OBJECTIVE_RUNS.get(label), rows.get(label.replace("_", " "))
        zrow = rows.get(zlabel.replace("_", " "))
        path = SCREEN_DIR / f"swords_paired_ci_vs_{zlabel}.json"
        if not (run_path and row and zrow and path.is_file()):
            print(f"  seed {seed:5d}  not scored yet ({label} vs {zlabel})"); continue
        checks = {}
        for metric, key, want_negative in (("gap", "GAP", False), ("auroc", "AUROC", False),
                                           ("alternatives_nll", "acceptedNLL", True)):
            got = _ci(path, run_path, metric)
            checks[key] = bool(got and got[1] and ((got[0] < 0) == want_negative))
        checks["STS"] = float(row["sts_mean"]) >= float(zrow["sts_mean"]) - STS_ALLOWANCE
        failed = [k for k, v in checks.items() if not v]
        print(f"  seed {seed:5d}  {'PASS' if not failed else 'FAIL'}"
              f"  STS {float(row['sts_mean']):.4f} vs {float(zrow['sts_mean']):.4f}"
              + ("" if not failed else f"   failed: {', '.join(failed)}"))

def breadth(rows):
    # SWORDS dev and CoInCo dev side by side for every hybrid arm, vs Zhang.  Reported, not gated.
    zlabel = "zhang_seed42" if "zhang seed42" in rows else "zhang_lambda1.0_seed42"
    print(f"\n== breadth: GAP vs {zlabel}, SWORDS dev | CoInCo dev (800 targets) ==")
    for label, run_path in OBJECTIVE_RUNS.items():
        if not label.startswith("zhang_plus_"):
            continue
        cells = []
        for bench in ("swords", "coinco_dev"):
            got = _ci(SCREEN_DIR / f"{bench}_paired_ci_vs_{zlabel}.json", run_path, "gap")
            cells.append(f"{got[0]:+.4f}{'*' if got[1] else ' '}" if got else "   n/a  ")
        print(f"  {label:40s} {cells[0]} | {cells[1]}")

GATE_CHOICE = hybrid_gate()
"""),
        md("""## Locked confirmation

Lock $\\alpha$ and $\\beta$ from the seed-42 screen's decision cell further down; never choose them per seed. This cell sits before evaluation on purpose, so one Run All trains the new seeds and then scores them. Seeds 42, 123, 2024 for the alternative-only uniform arm and, if promoted, the contrastive arm. The seed-42 adapters already exist and resume for free."""),
        code(r"""
SELECTED_BETA = float(os.environ["SELECTED_BETA"]) if os.environ.get("SELECTED_BETA") else None
if RUN_CONFIRM:
    assert SELECTED_ALPHA is not None, "set SELECTED_ALPHA from the alpha gate first"
    for seed in SEEDS:
        OBJECTIVE_RUNS[f"alternative_uniform_seed{seed}"] = train_flat(
            "alternative_uniform", seed, SELECTED_ALPHA, objective="uniform", slot_ntp_weight=1.0,
            exclude_target=True)
        if SELECTED_BETA is not None:
            OBJECTIVE_RUNS[f"contrastive_seed{seed}"] = train_flat(
                "alternative_contrastive", seed, SELECTED_ALPHA, objective="uniform",
                slot_ntp_weight=1.0, exclude_target=True, contrast_beta=SELECTED_BETA, train_file=NEG_TRAIN)
    # The screen already trained seed 42 at the locked values and adapter_path()
    # maps both calls to one directory; two keys on one adapter would score it twice.
    OBJECTIVE_RUNS.pop(f"alternative_uniform_alpha{SELECTED_ALPHA}_seed42", None)
    if SELECTED_BETA is not None:
        OBJECTIVE_RUNS.pop(f"contrast_alpha{SELECTED_ALPHA}_beta{SELECTED_BETA}_seed42", None)
    for label, path in OBJECTIVE_RUNS.items(): sync_small_artifacts(path, f"task15b_logs{_TAG_SUFFIX}/{label}")
    audit_no_drive_weights()
save_runs(OBJECTIVE_RUNS, f"task15b{_TAG_SUFFIX}")
"""),
        md(r"""## Verified supervision: one slot objective per arm

The question this stage answers, with everything else held equal:

> Does teaching every valid alternative to outrank verifier-filtered negatives improve contextual substitution over maximising the total probability of a concept set?

Same training file for every arm (`synonyms_train_verified.jsonl`: positives pruned by the verifier's low tail, negatives mined from the model's own top-$k$ pool through the same tail), same schedule and initialisation, NTP everywhere else. What differs is **only** what sits at the concept slot:

| arm | at the slot | positives | negatives |
|---|---|---|---|
| `zhang_seed42` (Task 15) | released set marginal | original | — |
| `verified_zhang` | released set marginal | pruned | — |
| `verified_hybrid` | set marginal $+\gamma\,$uniform | pruned | — |
| `verified_pool` | $-\log\frac{P(A)}{P(A)+P(N)}$ | pruned | yes |
| `verified_rank` | $-\frac1{|A|}\sum_a\log\frac{p_a}{p_a+P(N)}$ | pruned | yes |
| `verified_list_uniform` | $-\frac1{|A|}\sum_a\log\frac{p_a}{P(A)+P(N)}$ | pruned | yes |
| `verified_list_verifier` | $-\sum_a w_a\log\frac{p_a}{P(A)+P(N)},\ w_a\propto\max(s_a-\theta,0)$ | pruned | yes |
| `verified_alt_uniform` | $-\frac1{|A|}\sum_a\log p_a$ (full softmax) | pruned | — |

The four discriminative arms exclude the observed token from $A$ and keep slot NTP at 1.0 (the trainer refuses any other combination, and refuses `--contrast-beta` or `--alt-aux` on top of them). A slot carries the term only when at least one alternative **and** one negative survive tokenization; otherwise it contributes exactly zero and stays in the denominator.

Two claims, gated separately below and never merged: **contrast helps** (a discriminative arm beats `verified_zhang` on GAP and AUROC with paired intervals excluding zero, inside the STS and observed-target allowances) and **per-positive helps** (`verified_rank` beats `verified_pool` the same way). Beating the old noisy baseline establishes neither.

The comparison against `verified_zhang` is confounded: the discriminative arms also drop the target-inclusive marginal, which the screen already showed is load-bearing. `verified_alt_uniform` is the matched control for them -- same file, same $\lambda$, observed token excluded, slot NTP 1.0, no negatives. It still differs in two ways besides the negatives, and the paper must say so: it supervises every slot with an alternative (not only the ~30% with a negative), and it asks for absolute probability rather than probability relative to $N$.

The method itself (the mass arm and the selected hybrid's seeds) is trained and confirmed in the hybrid section above, on Zhang's original file, so the main table never depends on this verifier. This stage only asks whether the verified negatives add anything.

Adapters live under a method name that carries the training file's SHA-256, and a resume is refused if the recorded hash differs, so a pruned-positive run can never reuse an older adapter."""),
        code(r"""
VERIFIED_DIR = DRIVE_RESULTS / f"task15b_verified{_TAG_SUFFIX}"
VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
VERIFIED_TRAIN = Path(os.environ.get("VERIFIED_TRAIN_FILE") or (LEAF / "synonyms_train_verified.jsonl"))
VERIFIED_LAMBDA = float(os.environ.get("VERIFIED_LAMBDA", "1.0"))
VERIFIED_GAMMA = float(os.environ.get("VERIFIED_GAMMA") or (0.125 if MODEL_TAG == "llama-3.2-1b" else 0.0625))
VERIFIED_RUNS = load_runs(f"task15b_verified{_TAG_SUFFIX}")
DISCRIMINATIVE = ("pool", "rank", "list_uniform", "list_verifier")
VERIFIED_ONLY = [x.strip() for x in os.environ.get("VERIFIED_ARMS", "").split(",") if x.strip()]

if RUN_VERIFIED:
    assert VERIFIED_TRAIN.is_file(), f"{VERIFIED_TRAIN} missing: run scripts/build_verified_negatives.py first"
    # ---- data-sanity gate: the mechanical version of "read the 50-row sample" ----
    # Ranges come from the Qwen pilot (coverage .35, 9.8% pruned, 0 unscored).  Out of
    # range stops the pass here, before a GPU-hour is spent; the sample is still read
    # by a person afterwards, this only catches a run that went visibly wrong.
    n_rows = sum(1 for _ in VERIFIED_TRAIN.open(encoding="utf-8"))
    assert n_rows >= 3000, f"{VERIFIED_TRAIN} has {n_rows} rows; the merged file should have ~3200"
    _report = None
    for _cand in (DRIVE_RESULTS / f"verified_report{_TAG_SUFFIX}.json",
                  VERIFIED_TRAIN.parent / "verified_report.json"):
        if _cand.is_file():
            _report = json.loads(_cand.read_text()); break
    # Missing or old reports are errors, not reasons to skip the gate: the data is
    # only trusted when its report was written by the current miner FOR THIS FILE.
    assert _report is not None, "no miner report beside the verified data; refusing to train on it"
    _want = re.search(r'MINER_VERSION = "([^"]+)"',
                      (MAIN / "scripts/build_verified_negatives.py").read_text()).group(1)
    assert _report.get("miner_version") == _want, (
        f"verified data was written by miner {_report.get('miner_version')!r}, current is {_want!r}; re-mine")
    assert _report.get("output_sha256") == sha256(VERIFIED_TRAIN), (
        "the miner report describes a different file than the one on disk; re-mine")
    assert _report.get("complete"), "the report is from a partial (pilot or shard) run; re-mine the full file"
    assert "positives_unscored" in _report, "report lacks the unscored-positive count; re-mine"
    if True:
        _checks = {
            "negative_coverage in [.15,.70]": .15 <= _report["negative_coverage"] <= .70,
            "effective_contrastive_coverage >= .15": _report["effective_contrastive_coverage"] >= .15,
            "positives pruned share in [.02,.30]":
                .02 <= _report["positives_pruned"] / max(_report["positives_seen"], 1) <= .30,
            "positives unscored share < .05":
                _report["positives_unscored"] / max(_report["positives_seen"], 1) < .05,
            "rows >= 3000": _report["rows"] >= 3000,
        }
        for _k, _v in _checks.items():
            print(f"  data gate  {'PASS' if _v else 'FAIL'}  {_k}")
        assert all(_checks.values()), "verified data is outside the piloted ranges; stopping before training"
    DATA_HASH = sha256(VERIFIED_TRAIN)
    DATA_TAG = f"data_{DATA_HASH[:8]}"
    print("verified train file:", VERIFIED_TRAIN, "| sha256", DATA_HASH[:16] + "...",
          "| lambda", VERIFIED_LAMBDA, "| gamma", VERIFIED_GAMMA)

    ARMS = {
        "verified_zhang":         dict(objective="set_marginal", weight=1.0),
        "verified_hybrid":        dict(objective="set_marginal", weight=1.0,
                                       alt_aux="uniform", alt_aux_weight=VERIFIED_GAMMA),
        "verified_pool":          dict(objective="pool"),
        "verified_rank":          dict(objective="rank"),
        "verified_list_uniform":  dict(objective="list_uniform"),
        "verified_list_verifier": dict(objective="list_verifier"),
        # Matched control for the four above: same lambda, same exclusion and slot
        # NTP, no negatives.
        "verified_alt_uniform":   dict(objective="uniform", alt_only=True),
    }
    unknown = set(VERIFIED_ONLY) - set(ARMS)
    assert not unknown, f"VERIFIED_ARMS names unknown arms: {sorted(unknown)}"

    def launch(label, spec, *, seed=42, prefix="", epochs=5, max_samples=None,
               logging_steps=None, force=False):
        objective = spec["objective"]
        weight = spec.get("weight", VERIFIED_LAMBDA)
        alt_aux, alt_w = spec.get("alt_aux", "none"), spec.get("alt_aux_weight", 0.0)
        kw = dict(objective=objective, train_file=VERIFIED_TRAIN, epochs=epochs,
                  max_samples=max_samples, alt_aux=alt_aux, alt_aux_weight=alt_w,
                  logging_steps=logging_steps, force=force)
        if objective in DISCRIMINATIVE or spec.get("alt_only"):
            kw.update(exclude_target=True, slot_ntp_weight=1.0)
        # The data hash is part of the METHOD name, so this path can only ever hold
        # an adapter trained on this exact file -- and the recorded hash is checked
        # anyway before a resume is accepted.
        method = f"{prefix}{label}_{DATA_TAG}"
        aux_tag = "" if alt_aux == "none" else f"_aux_{alt_aux}_{alt_w}"
        expected = adapter_path(method, seed, f"lambda_{weight}_beta_0.0{aux_tag}")
        if not force and RESUME_FINISHED_RUNS and finished(expected):
            recorded = json.loads((expected / "concept_training_config.json").read_text()) \
                if (expected / "concept_training_config.json").is_file() else {}
            if recorded.get("train_file_sha256") != DATA_HASH:
                raise RuntimeError(f"{expected} was trained on a different file "
                                   f"({recorded.get('train_file_sha256')}); refusing to resume it")
        return train_flat(method, seed, weight, **kw)

    if VERIFIED_SMOKE_STEPS:
        print(f"== smoke: ~{VERIFIED_SMOKE_STEPS} steps per arm, first logged magnitudes ==")
        magnitudes = {}
        for label in ("verified_zhang", "verified_pool", "verified_rank",
                      "verified_list_uniform", "verified_list_verifier"):
            # force: always a fresh run (never an old adapter's history), and a
            # logging cadence well inside the smoke so it writes TRAINING rows.
            out = launch(label, ARMS[label], prefix="smoke_", epochs=1,
                         max_samples=VERIFIED_SMOKE_STEPS * 16,
                         logging_steps=max(1, VERIFIED_SMOKE_STEPS // 8), force=True)
            history = out / "training_history.jsonl"
            rows = [json.loads(l) for l in history.read_text().splitlines() if l.strip()] \
                if history.is_file() else []
            # Training rows only.  The end-of-epoch EVAL row also carries concept_loss,
            # computed on synonyms_val.jsonl, which has no verified negatives -- so for
            # every discriminative arm it reads exactly 0.0.  Reading it is how the
            # 2026-09-23 pass silently skipped this rule.  Training rows are the ones
            # that carry contrast_loss; eval rows never do.
            train_rows = [r for r in rows if "contrast_loss" in r and "eval_loss" not in r
                          and r.get("concept_loss") is not None]
            if not train_rows:
                raise RuntimeError(f"{label}: the smoke wrote no training row to {history}; "
                                   "the lambda rule cannot be applied")
            first = train_rows[0]
            magnitudes[label] = {k: first[k] for k in
                                 ("step", "ce_loss", "concept_loss", "concept_eligible_share") if k in first}
            magnitudes[label]["smoke_mean_concept_loss"] = \
                sum(r["concept_loss"] for r in train_rows) / len(train_rows)
            print(f"  {label:24s}", {k: round(v, 4) for k, v in magnitudes[label].items()})
        # The rule, declared before any result was read: lambda = 1.0 unless the median
        # first-step concept_loss of the four discriminative arms is more than 3x or less
        # than 1/3 of verified_zhang's, in which case the nearest power of two of the
        # ratio, clamped to [1/16, 16].  One lambda for all four arms.
        import math, statistics
        ref = magnitudes.get("verified_zhang", {}).get("concept_loss")
        disc = [magnitudes[k]["concept_loss"] for k in ("verified_pool", "verified_rank",
                "verified_list_uniform", "verified_list_verifier") if magnitudes[k].get("concept_loss")]
        # A zero or missing magnitude is a broken measurement, never "no change".
        if not ref or len(disc) != 4:
            raise RuntimeError(f"lambda rule needs five non-zero first-step magnitudes, got "
                               f"reference={ref}, discriminative={disc}")
        decision = {"rule": "1.0 unless median(discriminative)/verified_zhang outside [1/3, 3]; then 2^round(log2(zhang/median)) clamped to [1/16,16]; first TRAINING row of each smoke",
                    "magnitudes": magnitudes, "reference": ref, "median_discriminative": None,
                    "lambda_before": VERIFIED_LAMBDA, "lambda_after": VERIFIED_LAMBDA}
        if ref and disc:
            med = statistics.median(disc)
            decision["median_discriminative"] = med
            if med > 3 * ref or med < ref / 3:
                decision["lambda_after"] = float(min(16, max(1 / 16, 2 ** round(math.log2(ref / med)))))
        VERIFIED_LAMBDA = decision["lambda_after"]
        (VERIFIED_DIR / "verified_lambda_decision.json").write_text(json.dumps(decision, indent=2))
        print(f"lambda rule -> {VERIFIED_LAMBDA}  (recorded in {VERIFIED_DIR / 'verified_lambda_decision.json'})")
        if VERIFIED_SMOKE_ONLY:
            print("VERIFIED_SMOKE_ONLY: stopping here; set it False (or VERIFIED_SMOKE_STEPS = 0) to train.")
    if not (VERIFIED_SMOKE_STEPS and VERIFIED_SMOKE_ONLY):
        for label, spec in ARMS.items():
            if VERIFIED_ONLY and label not in VERIFIED_ONLY:
                continue
            VERIFIED_RUNS[label] = launch(label, spec)
            # Merge into what is on disk, not overwrite it: a second process may be
            # training disjoint arms against the same manifest right now.
            VERIFIED_RUNS = {**load_runs(f"task15b_verified{_TAG_SUFFIX}"), **VERIFIED_RUNS}
            save_runs(VERIFIED_RUNS, f"task15b_verified{_TAG_SUFFIX}")
        for label, path in VERIFIED_RUNS.items():
            sync_small_artifacts(path, f"task15b_verified_logs{_TAG_SUFFIX}/{label}")

if RUN_VERIFIED and RUN_EVAL and not (VERIFIED_SMOKE_STEPS and VERIFIED_SMOKE_ONLY):
    # Its OWN result directory: the baselines plus these six, about ten checkpoints,
    # instead of re-scoring the whole screen every time an arm is added.
    VERIFIED_RUNS = restore_all(VERIFIED_RUNS)
    task15_runs = restore_all(load_runs(f"task15{_TAG_SUFFIX}"))
    baseline_runs = {label: task15_runs[label] for label in
                     ("ntp_seed42", "randomized_seed42", "zhang_seed42", "zhang_lambda1.0_seed42")
                     if label in task15_runs}
    if "zhang_seed42" in baseline_runs:
        baseline_runs.pop("zhang_lambda1.0_seed42", None)
    all_runs = {**baseline_runs, **VERIFIED_RUNS}
    checkpoints = [BASE_MODEL, *map(str, all_runs.values())]
    result_dir = VERIFIED_DIR
    guarded(result_dir / "val_perplexity.json", checkpoints,
            [sys.executable, "eval/eval_perplexity_explicit.py", "--checkpoints", *checkpoints,
             "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_val.jsonl",
             "--output", result_dir / "val_perplexity.json"], EXT, "validation perplexity")
    guarded(result_dir / "val_concept_sets.json", checkpoints,
            [sys.executable, "eval/eval_concept_sets.py", "--checkpoints", *checkpoints,
             "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_val.jsonl",
             "--output", result_dir / "val_concept_sets.json"], EXT, "validation concept sets")
    guarded(result_dir / "perplexity.json", checkpoints,
            [sys.executable, "eval/eval_perplexity_explicit.py", "--checkpoints", *checkpoints,
             "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_test.jsonl",
             "--output", result_dir / "perplexity.json"], EXT, "perplexity")
    guarded(result_dir / "concept_sets.json", checkpoints,
            [sys.executable, "eval/eval_concept_sets.py", "--checkpoints", *checkpoints,
             "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_test.jsonl",
             "--output", result_dir / "concept_sets.json"], EXT, "concept sets")
    task15_dir = DRIVE_RESULTS / RESULT_DIR
    for label, checkpoint in {"pretrained": BASE_MODEL, **all_runs}.items():
        display_label = label.replace("_", " ")
        csv_output = result_dir / f"sts_{display_label}.csv"
        # STS is deterministic per checkpoint: reuse the baselines' files from Task 15
        # or the screen rather than re-deriving them.
        for previous in (task15_dir / csv_output.name, SCREEN_DIR / csv_output.name):
            if not csv_output.is_file() and previous.is_file():
                shutil.copy2(previous, csv_output)
        if sts_covered(csv_output):
            print("resume: STS already scored, skipping", display_label)
            continue
        mteb_args = [sys.executable, "eval/eval_mteb.py", "--base-model", BASE_MODEL,
                     "--dataset", "c4", "--dataset-type", "embedding", "--tasks", "sts",
                     "--run-label", display_label, "--csv-output", csv_output,
                     "--mteb-output-root", result_dir / "mteb_raw"]
        mteb_args += ["--no-adapter"] if checkpoint == BASE_MODEL else ["--adapter-path", checkpoint]
        run(mteb_args, cwd=EXT)
    guarded(result_dir / "swords.json", checkpoints,
            [sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_swords.py",
             "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
             "--swords_json", MAIN / "data/swords/swords-v1.1_dev.json.gz",
             "--results_json", result_dir / "swords.json", "--modes", "left", "full"], MAIN, "SWORDS")
    # Paired intervals against every reference a claim below needs: the released
    # baseline, its pruned-data twin, the continuity arm, and pooled contrast.
    references = {"pretrained": BASE_MODEL, **{label: str(path) for label, path in baseline_runs.items()}}
    for key in ("verified_zhang", "verified_hybrid", "verified_pool", "verified_alt_uniform"):
        if key in VERIFIED_RUNS:
            references[key] = str(VERIFIED_RUNS[key])
    for label, reference in references.items():
        run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/paired_benchmark_ci.py",
             "--kind", "swords", "--results-json", result_dir / "swords.json",
             "--baseline-index", checkpoints.index(reference),
             "--output", result_dir / f"swords_paired_ci_vs_{label}.json"], cwd=MAIN)
    guarded(result_dir / "bm_semlex.json", checkpoints,
            [sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_bm_semlex.py",
             "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
             "--data", MAIN / "data/bm_semlex/curated_200.tsv",
             "--results_json", result_dir / "bm_semlex.json"], MAIN, "bm-semlex")
    manifest = {"pretrained": BASE_MODEL, **{label.replace("_", " "): str(path) for label, path in all_runs.items()}}
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    run([sys.executable, MAIN / "scripts/summarize_concept_experiments.py",
         "--manifest", result_dir / "manifest.json", "--result-dir", result_dir,
         "--output", result_dir / "flat_extension_table.csv"], cwd=MAIN)
"""),
        md("""## Verified gate (automatic, two claims, no selection)

Reads only what the cell above wrote. Thresholds come from this table's own rows: STS within .005 of the arm being compared against, observed-target NLL within .10 of NTP, global NLL within .20 of NTP. It prints every check and its number; it selects nothing."""),
        code(r"""
def verified_gate():
    table = VERIFIED_DIR / "flat_extension_table.csv"
    if not table.is_file():
        print("no verified table yet; run RUN_VERIFIED with RUN_EVAL first"); return
    import csv
    rows = {r["method"]: r for r in csv.DictReader(table.open())}
    ntp = rows.get("ntp seed42")
    if ntp is None:
        print("NTP row missing from the verified table; cannot gate"); return
    nll_ceiling = float(ntp["global_nll"]) + 0.20
    obs_ceiling = float(ntp["swords_observed_target_nll"]) + 0.10
    print(f"retention ceilings -> global NLL <= {nll_ceiling:.4f} | observed-target NLL <= {obs_ceiling:.4f}")

    def beats(candidate, reference_label, sts_reference):
        # (all_pass, checks) for candidate vs one reference, paired on SWORDS dev.
        path = VERIFIED_DIR / f"swords_paired_ci_vs_{reference_label}.json"
        run_path = VERIFIED_RUNS.get(candidate)
        row = rows.get(candidate.replace("_", " "))
        if row is None or run_path is None or not path.is_file():
            return None, {}
        checks = {}
        for metric, key, want_negative in (("gap", "GAP", False), ("auroc", "AUROC", False),
                                           ("alternatives_nll", "acceptedNLL", True)):
            got = _ci(path, run_path, metric)
            checks[key] = bool(got and got[1] and ((got[0] < 0) == want_negative))
        checks["STS"] = float(row["sts_mean"]) >= float(sts_reference["sts_mean"]) - 0.005
        checks["globalNLL"] = float(row["global_nll"]) <= nll_ceiling
        checks["obsNLL"] = float(row["swords_observed_target_nll"]) <= obs_ceiling
        return all(checks.values()), checks

    def show(title, candidate, reference_label):
        ref_row = rows.get(reference_label.replace("_", " "))
        if ref_row is None:
            print(f"  {title:44s} reference row missing"); return
        ok, checks = beats(candidate, reference_label, ref_row)
        if ok is None:
            print(f"  {title:44s} not scored yet"); return
        failed = [k for k, v in checks.items() if not v]
        print(f"  {title:44s} {'PASS' if ok else 'FAIL'}" + ("" if ok else f"   failed: {', '.join(failed)}"))

    print("\n== data effect: pruned positives alone ==")
    zhang_label = "zhang_seed42" if "zhang_seed42" in rows or "zhang seed42" in rows else "zhang_lambda1.0_seed42"
    show("verified_zhang vs zhang (original data)", "verified_zhang", zhang_label)
    print("\n== claim 1: discrimination replaces Zhang's marginal (vs verified_zhang, same data) ==")
    print("   confounded: these arms also drop the target-inclusive term; the matched test is next")
    for arm in ("verified_pool", "verified_rank", "verified_list_uniform", "verified_list_verifier"):
        show(f"{arm} vs verified_zhang", arm, "verified_zhang")
    print("\n== claim 1, matched: negatives vs alternative-only (vs verified_alt_uniform) ==")
    print("   same data, lambda, exclusion and slot NTP; still differs in eligibility (~30% of slots)")
    print("   and in asking for relative rather than absolute probability -- not a pure negatives on/off")
    for arm in ("verified_pool", "verified_rank", "verified_list_uniform", "verified_list_verifier"):
        show(f"{arm} vs verified_alt_uniform", arm, "verified_alt_uniform")
    print("\n== claim 1, against the released baseline on original data ==")
    for arm in ("verified_pool", "verified_rank", "verified_list_uniform", "verified_list_verifier"):
        show(f"{arm} vs zhang", arm, zhang_label)
    print("\n== claim 2: per-positive ranking helps (vs verified_pool) ==")
    show("verified_rank vs verified_pool", "verified_rank", "verified_pool")
    print("\n== decompositions ==")
    show("verified_list_uniform vs verified_pool  (within-positive calibration)", "verified_list_uniform", "verified_pool")
    show("verified_hybrid vs verified_zhang        (continuity arm on clean data)", "verified_hybrid", "verified_zhang")

    print("\nNothing is selected here.  Confirm across seeds only what passes its own claim; "
          "SWORDS test stays locked until then.")

verified_gate()
"""),
        md("""## Reporting

Main table: NTP, augmented NTP, Zhang set-marginal, alternative-only uniform, contrastive (if promoted). Control table: pretrained, randomized $\\lambda=.25$, randomized $\\lambda=1$ (matched weight), target-inclusive uniform ablation. Paired bootstrap intervals on every SWORDS comparison; mean ± sd over three seeds everywhere else. Do not expand to 3B from this notebook until the three-seed 1B result is in; the hierarchy experiment stays deferred."""),
        md("""## SWORDS test — one locked invocation

Everything above is SWORDS **dev**: $\\alpha$, $\\beta$ and $\\gamma$ were all chosen on it, so it is development data and cannot support the headline number. This cell scores test **once**, over every locked arm in a single call, so the method and its baselines are measured on identical rows with identical code.

It refuses to run until `SELECTED_HYBRID` is set, and it writes `swords_test_locked.json` recording the arms and the commit. If that file already exists the cell stops: a second test pass with a changed method is the one thing this protocol exists to prevent. Nothing here may be re-run after reading the result."""),
        code(r"""
RUN_SWORDS_TEST = False   # set True exactly once, after the method is locked
if RUN_SWORDS_TEST:
    assert SELECTED_HYBRID, "lock the method first: the gate sets SELECTED_HYBRID"
    marker = SCREEN_DIR / "swords_test_locked.json"
    assert not marker.is_file(), (
        f"SWORDS test has already been run: {marker}. Re-running after seeing the "
        "result invalidates it. Delete the marker ONLY if the previous run crashed.")

    # One invocation, every locked arm, in a fixed order.  Baselines come from
    # Task 15's manifest so the test table cannot quietly use a different NTP
    # than the dev table did.
    task15_runs = restore_all(load_runs(f"task15{_TAG_SUFFIX}"))
    locked = {"pretrained": BASE_MODEL}
    for family in ("ntp", "augmented_ntp", "zhang", "randomized"):
        for seed in (42, 123, 2024):
            key = f"{family}_seed{seed}"
            if key in task15_runs:
                locked[key] = str(task15_runs[key])
            else:
                print("MISSING baseline, test table will be incomplete:", key)
    OBJECTIVE_RUNS = restore_all(OBJECTIVE_RUNS)
    kind, weight = SELECTED_HYBRID.split(":"); weight = float(weight)
    for seed in (42, 123, 2024):
        key = hybrid_label(kind, weight, seed)
        if key not in OBJECTIVE_RUNS and f"calibrated_seed{seed}" in OBJECTIVE_RUNS:
            key = f"calibrated_seed{seed}"          # label used by passes before 2026-09-23
        if key in OBJECTIVE_RUNS:
            locked[key] = str(OBJECTIVE_RUNS[key])
        else:
            print("MISSING method seed:", key)
    # The two ablations the paper reports beside the method.
    for key in ("alternative_uniform_seed42", "inclusive_uniform_alpha0.5_seed42"):
        if key in OBJECTIVE_RUNS:
            locked[key] = str(OBJECTIVE_RUNS[key])

    checkpoints = list(locked.values())
    # Three held-out substitution benchmarks, one invocation each, same evaluator:
    # SWORDS test, the fixed 800-target CoInCo test subset, and all of SemEval-2007
    # Task 10 (see data/lexsub/SOURCE.md).  None of them was read during development.
    TEST_BENCHMARKS = {"swords_test": MAIN / "data/swords/swords-v1.1_test.json.gz",
                       "coinco_test": MAIN / "data/lexsub/coinco_test_sub800.json.gz",
                       "semeval07_test": MAIN / "data/lexsub/semeval07_test.json.gz"}
    for bench, source in TEST_BENCHMARKS.items():
        print(f"scoring {len(checkpoints)} locked checkpoints on {bench}")
        run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_swords.py",
             "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL,
             "--base_model", BASE_MODEL, "--swords_json", source,
             "--results_json", SCREEN_DIR / f"{bench}.json",
             "--modes", "left", "full"], cwd=MAIN)

    # Seed-matched paired intervals: each method seed against the SAME seed of
    # each baseline.  Averaging over mismatched seeds would fold seed variance
    # into the effect.
    for bench in TEST_BENCHMARKS:
        for family in ("zhang", "ntp", "augmented_ntp"):
            for seed in (42, 123, 2024):
                key = f"{family}_seed{seed}"
                if key not in locked:
                    continue
                run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/paired_benchmark_ci.py",
                     "--kind", "swords", "--results-json", SCREEN_DIR / f"{bench}.json",
                     "--baseline-index", checkpoints.index(locked[key]),
                     "--output", SCREEN_DIR / f"{bench}_paired_ci_vs_{key}.json"], cwd=MAIN)

    marker.write_text(json.dumps({
        "selected_hybrid": SELECTED_HYBRID,
        "arms": locked,
        "upstream_commit": UPSTREAM_COMMIT,
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, indent=2))
    print("locked:", marker)
"""),
        md("""### Reporting the test table

For each metric report (a) mean ± sd over the three seeds of each arm, (b) the seed-matched difference method$_s$ − baseline$_s$ for $s \\in \\{42,123,2024\\}$, and (c) a paired bootstrap over per-target scores, averaged across seeds — not a bootstrap over the seed means, which has three points and no power. A claim needs all three seeds to agree in sign with intervals excluding zero."""),
    ]
    # Confirmation must TRAIN before evaluation SCORES.  Written in narrative order
    # -- screen, evaluate, decide, confirm -- a single Run All trained the new seeds
    # AFTER every evaluator had finished, so they were never scored and the table
    # silently held seed 42 alone.  The decision cell still reads only the screen
    # arms, so moving confirmation up changes what gets scored, not what is chosen.
    def _heading(cell, text):
        return cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith(text)
    confirm_at = next(i for i, c in enumerate(cells) if _heading(c, "## Locked confirmation"))
    confirm = cells[confirm_at:confirm_at + 2]
    del cells[confirm_at:confirm_at + 2]
    eval_at = next(i for i, c in enumerate(cells) if _heading(c, "## Evaluation"))
    cells[eval_at:eval_at] = confirm
    write("research_tasks_15b_objective_and_contrastive.ipynb", cells)


def task16():
    cells = [
        md('''# Task 16 — From flat concepts to directional hierarchies

This is the proposed paper contribution after the flat objective is locked. Synonym equivalence remains available for nouns, verbs, and adjectives. Directional hypernym supervision is restricted to nouns and verbs.

| Readable method | Purpose |
|---|---|
| NTP fine-tuning | Ordinary control |
| Augmented NTP | Iyer-style baseline |
| Flat concept marginal | Zhang baseline |
| Independent hypernym prediction | Direct prior-style hierarchy auxiliary |
| Conditioned hierarchy | Predict hypernyms through the contextual synonym distribution |
| Shuffled hierarchy | Tests whether correct relations matter |

HCP anneals from WordNet hypernym-class prediction back to token prediction ([Bai et al., 2022](https://aclanthology.org/2022.acl-long.96/)). BALAUR learns relation-specific lexical-semantic transformations during pretraining and evaluates hypernym-sensitive behavior ([Lauscher et al., 2023](https://aclanthology.org/2023.findings-emnlp.674/)). Our independent auxiliary is the direct hierarchy baseline; reproduce HCP only if it does not adequately cover the curriculum question.'''),
        code(COMMON_SETUP), code(BOOTSTRAP), code(TRAIN_HELPER),
        md('''## Construct hierarchy data and quantify selection bias

Sense selection requires one target/synonym WordNet-synset intersection. Every retained/excluded slot stores its reason, frequency, concept-set size, POS, and WordNet depth. HyperLex is evaluation-only.'''),
        code(r'''
MODEL_TAG = BASE_MODEL.split("/")[-1].lower()
LEAF = DATA / "c4" / MODEL_TAG / "embedding"
HIER_TRAIN = LEAF / "synonyms_train_hierarchy.jsonl"
HIER_VAL = LEAF / "synonyms_val_hierarchy.jsonl"
# Copy these once from the locked Task-15/15b winner.  Defaults reproduce
# Zhang's flat marginal; do not tune them again on HyperLex.
FLAT_OBJECTIVE = "set_marginal"
FLAT_WEIGHT = 1.0
FLAT_SLOT_NTP_WEIGHT = None
FLAT_CONTRAST_BETA = 0.0
FLAT_SOURCE = LEAF / ("synonyms_train_conservative_negatives.jsonl"
                      if FLAT_CONTRAST_BETA > 0 else "synonyms_train.jsonl")
RUN_REFERENCES = False
REFERENCE_RUNS = load_runs("task16_references")
if RUN_REFERENCES:
    REFERENCE_RUNS["ntp"] = train_flat("task16_ntp_reference", 42, 0.0)
    REFERENCE_RUNS["augmented_ntp"] = train_flat(
        "task16_augmented_reference", 42, 0.0, epochs=1,
        train_file=LEAF / "synonyms_train_aug5x.jsonl", data_augmentation=True)
    REFERENCE_RUNS["locked_flat"] = train_flat(
        "task16_flat_reference", 42, FLAT_WEIGHT,
        objective=FLAT_OBJECTIVE, slot_ntp_weight=FLAT_SLOT_NTP_WEIGHT,
        contrast_beta=FLAT_CONTRAST_BETA, train_file=FLAT_SOURCE)
if RUN_DATA:
    for split, source, output in [
        ("train", FLAT_SOURCE, HIER_TRAIN),
        ("validation", LEAF / "synonyms_val.jsonl", HIER_VAL),
    ]:
        run([sys.executable, "data/build_hierarchy_dataset.py", "--input", source,
             "--output", output, "--report", DRIVE_RESULTS / f"hierarchy_coverage_{split}.json",
             "--seed", "42"], cwd=EXT)
    cache_dataset_to_drive(LEAF)
'''),
        md('''## Three-prompt robustness and exact sequence probabilities

Training samples these prompts uniformly: “In this context, *s* is a type of …”, “*s* and other …”, and “A more general term for *s* is …”. Evaluation reports each separately and their average. Hypernyms may be multi-token and are always scored by the full teacher-forced sequence; context is left-truncated before any candidate token.'''),
        code(r'''
# Two cost knobs, both defaulting to the exact objective on full data.  They are
# applied to the screen AND the confirmed runs together, because a gamma locked
# under one setting does not transfer to another.
#
# HIERARCHY_SURFACES = 0 scores every member of the equivalence set.  A positive K
#   truncates the bridge to the top-K by q and renormalises (~1.9x faster at K=3).
#   It is approximately unbiased when tail members have similar P(H|s); watch
#   hierarchy_bridge_mass_retained in the training log and treat anything below
#   ~0.9 as too aggressive.
# TRAIN_SAMPLES = None uses all 8,000 sequences.  Reducing it cuts hierarchy
#   supervision proportionally, because only one slot per batch is supervised, so
#   2,000 sequences means a quarter of the hierarchy updates, not just a quarter of
#   the flat data.  Zhang et al. Fig. 8 licenses the flat-data reduction on STS; it
#   says nothing about hierarchy sample efficiency.
HIERARCHY_SURFACES = 0
TRAIN_SAMPLES = None

def train_hierarchy(label, mode, gamma, seed, max_samples=None):
    out = RUNS / label / f"seed_{seed}" / f"gamma_{gamma}" / f"n_{max_samples or 'full'}"
    assert_ephemeral(out)
    if RESUME_FINISHED_RUNS and finished(out):
        print("resume: already trained, skipping", out)
        return out
    run([sys.executable, "train_hierarchy.py", "--model-name", BASE_MODEL,
         "--train-file", HIER_TRAIN, "--validation-file", HIER_VAL,
         "--output-dir", out, "--hierarchy-mode", mode, "--gamma", gamma,
         "--concept-loss-weight", FLAT_WEIGHT, "--concept-objective", FLAT_OBJECTIVE,
         "--contrast-beta", FLAT_CONTRAST_BETA,
         "--seed", seed, "--epochs", "5", "--learning-rate", "7e-5",
         "--candidate-microbatch-size", "8", "--max-hierarchy-slots-per-batch", "1",
         "--max-hierarchy-surfaces", HIERARCHY_SURFACES, "--report-to", "none",
         *(["--max-train-samples", max_samples] if max_samples else []),
         *(["--slot-ntp-weight", FLAT_SLOT_NTP_WEIGHT]
           if FLAT_SLOT_NTP_WEIGHT is not None else [])], cwd=EXT)
    cache_adapter_to_drive(out)
    return out

HIERARCHY_RUNS = load_runs("task16")
if RUN_SCREEN:
    for gamma in [0.1, 0.25, 0.5]:
        HIERARCHY_RUNS[f"conditioned_gamma{gamma}_seed42"] = train_hierarchy(
            "conditioned_hierarchy", "conditioned", gamma, 42, max_samples=TRAIN_SAMPLES)
    HIERARCHY_RUNS["independent_seed42"] = train_hierarchy(
        "independent_hypernym", "independent", 0.25, 42, max_samples=TRAIN_SAMPLES)
    HIERARCHY_RUNS["shuffled_seed42"] = train_hierarchy(
        "shuffled_hierarchy", "shuffled", 0.25, 42, max_samples=TRAIN_SAMPLES)
'''),
        md('''## Evaluation and gamma lock

Primary hierarchy metrics are HyperLex Spearman, forward-versus-reverse accuracy, and hypernym-versus-co-hyponym AUROC, stratified by relation depth and POS. SWORDS, bm-semlex, STS, and global/content NTP are retention tests. Select gamma on seed 42; then freeze it.'''),
        code(r'''
if RUN_EVAL:
    REFERENCE_RUNS = restore_all(REFERENCE_RUNS)
    HIERARCHY_RUNS = restore_all(HIERARCHY_RUNS)
    result_dir = DRIVE_RESULTS / "task16_screen"; result_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [str(x) for x in {**REFERENCE_RUNS, **HIERARCHY_RUNS}.values()]
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_hyperlex.py",
         "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
         "--hyperlex", HYPERLEX_DEV,
         "--pos", "N", "V", "--results_json", result_dir / "hyperlex.json"], cwd=MAIN)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/paired_benchmark_ci.py",
         "--kind", "hyperlex", "--results-json", result_dir / "hyperlex.json",
         "--baseline-index", "0", "--output", result_dir / "hyperlex_paired_ci.json"], cwd=MAIN)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_swords.py",
         "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
         "--swords_json", MAIN / "data/swords/swords-v1.1_dev.json.gz",
         "--results_json", result_dir / "swords.json", "--modes", "left", "full"], cwd=MAIN)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/paired_benchmark_ci.py",
         "--kind", "swords", "--results-json", result_dir / "swords.json",
         "--baseline-index", "0", "--output", result_dir / "swords_paired_ci.json"], cwd=MAIN)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_bm_semlex.py",
         "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
         "--data", MAIN / "data/bm_semlex/curated_200.tsv",
         "--results_json", result_dir / "bm_semlex.json"], cwd=MAIN)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_hierarchy_consistency.py",
         "--checkpoints", *checkpoints, "--base_model", BASE_MODEL,
         "--tokenizer_path", BASE_MODEL, "--hierarchy_jsonl", HIER_VAL,
         "--results_json", result_dir / "hierarchy_consistency.json"], cwd=MAIN)
    run([sys.executable, "eval/eval_perplexity_explicit.py", "--checkpoints", *checkpoints,
         "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_test.jsonl",
         "--output", result_dir / "perplexity.json"], cwd=EXT)
    run([sys.executable, "eval/eval_concept_sets.py", "--checkpoints", *checkpoints,
         "--base-model", BASE_MODEL, "--test-jsonl", HIER_VAL,
         "--output", result_dir / "concept_sets.json"], cwd=EXT)
    for label, checkpoint in {**REFERENCE_RUNS, **HIERARCHY_RUNS}.items():
        display_label = label.replace("_", " ")
        run([sys.executable, "eval/eval_mteb.py", "--base-model", BASE_MODEL,
             "--dataset", "c4", "--dataset-type", "embedding", "--tasks", "sts",
             "--run-label", display_label, "--csv-output", result_dir / f"sts_{display_label}.csv",
             "--mteb-output-root", result_dir / "mteb_raw",
             "--adapter-path", checkpoint], cwd=EXT)
    manifest = {label.replace("_", " "): str(path)
                for label, path in {**REFERENCE_RUNS, **HIERARCHY_RUNS}.items()}
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    run([sys.executable, MAIN / "scripts/summarize_concept_experiments.py",
         "--manifest", result_dir / "manifest.json", "--result-dir", result_dir,
         "--output", result_dir / "hierarchy_main_table.csv"], cwd=MAIN)
'''),
        code(r'''
SELECTED_GAMMA = None
if RUN_CONFIRM:
    assert SELECTED_GAMMA in {0.1, 0.25, 0.5}
    for seed in SEEDS:
        REFERENCE_RUNS[f"ntp_seed{seed}"] = train_flat(
            "task16_ntp_reference", seed, 0.0)
        REFERENCE_RUNS[f"augmented_ntp_seed{seed}"] = train_flat(
            "task16_augmented_reference", seed, 0.0, epochs=1,
            train_file=LEAF / "synonyms_train_aug5x.jsonl", data_augmentation=True)
        REFERENCE_RUNS[f"locked_flat_seed{seed}"] = train_flat(
            "task16_flat_reference", seed, FLAT_WEIGHT,
            objective=FLAT_OBJECTIVE, slot_ntp_weight=FLAT_SLOT_NTP_WEIGHT,
            contrast_beta=FLAT_CONTRAST_BETA, train_file=FLAT_SOURCE)
        HIERARCHY_RUNS[f"conditioned_seed{seed}"] = train_hierarchy(
            "conditioned_hierarchy", "conditioned", SELECTED_GAMMA, seed,
            max_samples=TRAIN_SAMPLES)
        HIERARCHY_RUNS[f"independent_seed{seed}"] = train_hierarchy(
            "independent_hypernym", "independent", SELECTED_GAMMA, seed,
            max_samples=TRAIN_SAMPLES)
        HIERARCHY_RUNS[f"shuffled_seed{seed}"] = train_hierarchy(
            "shuffled_hierarchy", "shuffled", SELECTED_GAMMA, seed,
            max_samples=TRAIN_SAMPLES)
    # Drop the gamma-screen keys from the reported table: they are a hyperparameter
    # search, not arms.  Screen and confirm share TRAIN_SAMPLES and HIERARCHY_SURFACES,
    # so at the locked gamma they address the same adapter and resume for free.
    for stale in [k for k in HIERARCHY_RUNS if k.startswith("conditioned_gamma")]:
        HIERARCHY_RUNS.pop(stale, None)
    for label, path in HIERARCHY_RUNS.items(): sync_small_artifacts(path, f"task16_logs/{label}")
    audit_no_drive_weights()
save_runs(HIERARCHY_RUNS, "task16")
save_runs(REFERENCE_RUNS, "task16_references")
'''),
        code(r'''
# Stage 3 is deliberately unreachable until the 1B evidence is locked.
RUN_SCALE_3B = False
ONE_BILLION_GATE_PASSED = False
SCALE_MODEL = "meta-llama/Llama-3.2-3B"
if RUN_SCALE_3B:
    assert ONE_BILLION_GATE_PASSED, "Record the signed-off 1B table before scaling."
    assert SELECTED_GAMMA in {0.1, 0.25, 0.5}
    # Concepts are tokenizer/model specific: regenerate the same 10k C4 sample
    # for 3B, audit the splits, and rebuild hierarchy labels before training.
    run([sys.executable, "data/get_content_words.py", "--model", SCALE_MODEL,
         "--dataset", "c4", "--max_length", "256"], cwd=EXT)
    run([sys.executable, "data/embedding_synonyms.py", "c4",
         "--start", "0", "--end", EXTRACT_SEQUENCES, "--model", SCALE_MODEL,
         *([] if EXTRACT_4BIT else ["--no-4bit"])], cwd=EXT)
    run([sys.executable, "data/merge_synonym_parts.py", "--train-size", SPLIT_TRAIN,
         "--val-size", SPLIT_VAL, "--test-size", SPLIT_TEST, "--expected-count", "2", "--force"], cwd=EXT)
    scale_tag = SCALE_MODEL.split("/")[-1].lower()
    scale_leaf = DATA / "c4" / scale_tag / "embedding"
    run([sys.executable, "data/audit_concept_data.py",
         "--train", scale_leaf / "synonyms_train.jsonl",
         "--validation", scale_leaf / "synonyms_val.jsonl",
         "--test", scale_leaf / "synonyms_test.jsonl",
         "--tokenizer", SCALE_MODEL,
         "--report", DRIVE_RESULTS / "scale3b_data_audit.json"], cwd=EXT)
    run([sys.executable, "data/augment_synonyms.py", "--base-dir", DATA,
         "--num-augmentations", "4", "--seed", "42", "--overwrite"], cwd=EXT)
    scale_flat_source = scale_leaf / "synonyms_train.jsonl"
    if FLAT_CONTRAST_BETA > 0:
        scale_flat_source = scale_leaf / "synonyms_train_conservative_negatives.jsonl"
        run([sys.executable, "data/build_contrastive_negatives.py",
             "--source", scale_leaf / "synonyms_train.jsonl",
             "--topk", DATA / "c4" / scale_tag / "prompting" / "topk_*.jsonl",
             "--output", scale_flat_source,
             "--report", DRIVE_RESULTS / "scale3b_negative_report.json"], cwd=EXT)
    scale_hier_train = scale_leaf / "synonyms_train_hierarchy.jsonl"
    scale_hier_val = scale_leaf / "synonyms_val_hierarchy.jsonl"
    for source, output, split in [
        (scale_flat_source, scale_hier_train, "train"),
        (scale_leaf / "synonyms_val.jsonl", scale_hier_val, "validation")]:
        run([sys.executable, "data/build_hierarchy_dataset.py", "--input", source,
             "--output", output, "--report", DRIVE_RESULTS / f"scale3b_hierarchy_{split}.json"], cwd=EXT)

    THREE_B_DATA_AUDITED = False
    assert THREE_B_DATA_AUDITED, "Compare 1B/3B hashes and coverage reports, then set True."
    SCALE_RUNS = {}
    for seed in SEEDS:
        common = ["--model-name", SCALE_MODEL, "--seed", seed, "--save-strategy", "no",
                  "--report-to", "none", "--per-device-train-batch-size", "2",
                  "--gradient-accumulation-steps", "8"]
        methods = [
            ("ntp", 0.0, "set_marginal", None, 0.0, scale_leaf / "synonyms_train.jsonl", 5),
            ("augmented_ntp", 0.0, "set_marginal", None, 0.0, scale_leaf / "synonyms_train_aug5x.jsonl", 1),
            ("zhang_marginal", 1.0, "set_marginal", None, 0.0, scale_leaf / "synonyms_train.jsonl", 5),
            ("winning_flat", FLAT_WEIGHT, FLAT_OBJECTIVE, FLAT_SLOT_NTP_WEIGHT,
             FLAT_CONTRAST_BETA, scale_flat_source, 5)]
        for label, weight, objective, slot_weight, beta, source, epochs in methods:
            out = RUNS / "scale3b" / label / f"seed_{seed}"
            args = [sys.executable, "train.py", "--dataset", "c4", "--dataset-type", "embedding",
                    "--concept-loss-weight", weight, "--concept-objective", objective,
                    "--contrast-beta", beta, "--train-file", source,
                    "--num-train-epochs", epochs, "--output-dir", out, *common]
            if slot_weight is not None:
                args += ["--slot-ntp-weight", slot_weight]
            if label == "augmented_ntp":
                args += ["--use-data-augmentation"]
            run(args, cwd=EXT)
            SCALE_RUNS[f"{label}_seed{seed}"] = out
        out = RUNS / "scale3b" / "conditioned_hierarchy" / f"seed_{seed}"
        args = [sys.executable, "train_hierarchy.py", "--model-name", SCALE_MODEL,
                "--train-file", scale_hier_train, "--validation-file", scale_hier_val,
                "--output-dir", out, "--hierarchy-mode", "conditioned", "--gamma", SELECTED_GAMMA,
                "--concept-loss-weight", FLAT_WEIGHT, "--concept-objective", FLAT_OBJECTIVE,
                "--contrast-beta", FLAT_CONTRAST_BETA, "--seed", seed,
                "--per-device-batch-size", "2", "--gradient-accumulation-steps", "8",
                "--candidate-microbatch-size", "4", "--max-hierarchy-slots-per-batch", "1"]
        if FLAT_SLOT_NTP_WEIGHT is not None:
            args += ["--slot-ntp-weight", FLAT_SLOT_NTP_WEIGHT]
        run(args, cwd=EXT)
        SCALE_RUNS[f"conditioned_seed{seed}"] = out
'''),
        code(r'''
RUN_SCALE_EVAL = False
if RUN_SCALE_EVAL:
    assert RUN_SCALE_3B and SCALE_RUNS
    result_dir = DRIVE_RESULTS / "task16_scale3b"; result_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [str(path) for path in SCALE_RUNS.values()]
    run([sys.executable, "eval/eval_perplexity_explicit.py", "--checkpoints", *checkpoints,
         "--base-model", SCALE_MODEL, "--test-jsonl", scale_leaf / "synonyms_test.jsonl",
         "--output", result_dir / "perplexity.json"], cwd=EXT)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_hyperlex.py",
         "--checkpoints", *checkpoints, "--tokenizer_path", SCALE_MODEL, "--base_model", SCALE_MODEL,
         "--hyperlex", HYPERLEX_DEV, "--pos", "N", "V",
         "--results_json", result_dir / "hyperlex.json"], cwd=MAIN)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/paired_benchmark_ci.py",
         "--kind", "hyperlex", "--results-json", result_dir / "hyperlex.json",
         "--baseline-index", "0", "--output", result_dir / "hyperlex_paired_ci.json"], cwd=MAIN)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_swords.py",
         "--checkpoints", *checkpoints, "--tokenizer_path", SCALE_MODEL, "--base_model", SCALE_MODEL,
         "--swords_json", MAIN / "data/swords/swords-v1.1_dev.json.gz",
         "--results_json", result_dir / "swords.json", "--modes", "left", "full"], cwd=MAIN)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/paired_benchmark_ci.py",
         "--kind", "swords", "--results-json", result_dir / "swords.json",
         "--baseline-index", "0", "--output", result_dir / "swords_paired_ci.json"], cwd=MAIN)
    for label, checkpoint in SCALE_RUNS.items():
        display_label = label.replace("_", " ")
        run([sys.executable, "eval/eval_mteb.py", "--base-model", SCALE_MODEL,
             "--dataset", "c4", "--dataset-type", "embedding", "--tasks", "sts",
             "--run-label", display_label, "--csv-output", result_dir / f"sts_{display_label}.csv",
             "--mteb-output-root", result_dir / "mteb_raw",
             "--adapter-path", checkpoint], cwd=EXT)
    manifest = {label.replace("_", " "): str(path) for label, path in SCALE_RUNS.items()}
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    run([sys.executable, MAIN / "scripts/summarize_concept_experiments.py",
         "--manifest", result_dir / "manifest.json", "--result-dir", result_dir,
         "--output", result_dir / "scale3b_main_table.csv"], cwd=MAIN)
    for label, path in SCALE_RUNS.items():
        sync_small_artifacts(path, f"task16_scale3b_logs/{label}")
    audit_no_drive_weights()
'''),
        md('''## Publication and scaling gate

The hierarchy claim passes only when conditioned hierarchy improves HyperLex over flat concept training, beats independent and shuffled controls, and retains the flat model’s STS/SWORDS result. Then—and only then—repeat NTP, augmented NTP, Zhang marginal, the winning flat method, and conditioned hierarchy at Llama-3.2-3B with all three seeds. Phrase/idiom work, 8B, OpenWebText, and commonsense reasoning stay deferred.'''),
    ]
    write("research_tasks_16_concept_hierarchies.ipynb", cells)


if __name__ == "__main__":
    NB_DIR.mkdir(parents=True, exist_ok=True)
    task15(); task15b(); task16()
