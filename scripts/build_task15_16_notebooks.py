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
import hashlib, json, os, re, shutil, subprocess, sys, torch

BASE_MODEL = "meta-llama/Llama-3.2-1B"
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
SPLIT_TRAIN = int(EXTRACT_SEQUENCES * 0.8)
SPLIT_VAL = SPLIT_TEST = int(EXTRACT_SEQUENCES * 0.1)

_BAR = re.compile(r"\b(\d+)/(\d+)\s*\[")   # tqdm counter, e.g. "  200/1000 ["
PROGRESS_EVERY = 100                        # print one progress line per this many items

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

ADAPTER_CACHE = DRIVE_PROJECT / "task15_16_adapters"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors",
                 "training_history.jsonl", "run_config.json")

def cache_adapter_to_drive(path):
    """Copy one finished adapter to Drive so a later session can resume."""
    destination = ADAPTER_CACHE / Path(path).relative_to(RUNS)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        source = Path(path) / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    return destination

def restore_adapter_from_drive(path):
    """Return True when a completed adapter for `path` was restored from Drive."""
    source = ADAPTER_CACHE / Path(path).relative_to(RUNS)
    if not (source / "adapter_config.json").is_file():
        return False
    Path(path).mkdir(parents=True, exist_ok=True)
    for name in ADAPTER_FILES:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, Path(path) / name)
    return True

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
if not EXT.exists():
    run(["git", "clone", "https://github.com/christine-zhang1/learning-concepts.git", EXT])
run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=EXT)
patch_file = MAIN / "external" / "learning-concepts.patch"
assert patch_file.exists(), "Commit external/learning-concepts.patch before running Colab."
check = subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=EXT)
if check.returncode == 0:
    run(["git", "apply", str(patch_file)], cwd=EXT)
else:
    reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch_file)], cwd=EXT)
    assert reverse.returncode == 0, "External checkout is neither clean nor exactly patched."

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
import transformers as _tf
print("transformers", _tf.__version__)
run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
run([sys.executable, MAIN / "builddataset/verify_task14_data.py",
     "--repo_root", MAIN, "--download_missing",
     "--report_json", DRIVE_RESULTS / "external_benchmark_integrity.json"], cwd=MAIN)
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
    path = RUNS / method / f"seed_{seed}" / str(value)
    assert_ephemeral(path)
    return path

def finished(path):
    """A run counts as finished when its adapter exists locally or on Drive."""
    if (Path(path) / "adapter_config.json").is_file():
        return True
    return restore_adapter_from_drive(path)

def train_flat(method, seed, concept_weight, *, objective="set_marginal",
               slot_ntp_weight=None, contrast_beta=0.0,
               randomized=False, data_augmentation=False, epochs=5, train_file=None,
               max_samples=None, batch=8, accum=2):
    out = adapter_path(method, seed, f"lambda_{concept_weight}_beta_{contrast_beta}")
    args = [sys.executable, "train.py", "--model-name", BASE_MODEL, "--dataset", "c4",
            "--dataset-type", "embedding", "--concept-loss-weight", concept_weight,
            "--concept-objective", objective, "--contrast-beta", contrast_beta,
            "--seed", seed, "--num-train-epochs", epochs, "--output-dir", out,
            "--save-strategy", "no", "--report-to", "none",
            "--per-device-train-batch-size", batch,
            "--gradient-accumulation-steps", accum]
    if slot_ntp_weight is not None: args += ["--slot-ntp-weight", slot_ntp_weight]
    if randomized: args += ["--randomized-synonyms"]
    if data_augmentation: args += ["--use-data-augmentation"]
    if train_file: args += ["--train-file", train_file]
    if max_samples: args += ["--max-train-samples", max_samples]
    # Released effective batch is 8 x 2 = 16; it is recorded in every config.
    if RESUME_FINISHED_RUNS and finished(out):
        print("resume: already trained, skipping", out)
        return out
    run(args, cwd=EXT)
    cache_adapter_to_drive(out)
    return out
'''


def task15():
    cells = [
        md('''# Task 15 — Reproduce concept training before extending it

This notebook answers one question first: **can we reproduce Zhang, Jurafsky, and Shani on the released C4 setup?** It keeps the released gold-inclusive set-marginal objective unchanged. Our uniform and contrastive losses are not run here.

| Readable method | What changes |
|---|---|
| Pretrained | No post-training |
| NTP fine-tuning | Ordinary next-token training on the same C4 rows |
| Augmented NTP | Synonym substitutions are written into training text |
| Randomized concepts | Same-sized incoherent sets; semantic control |
| Zhang concept marginal | Raises total probability of the observed word plus accepted alternatives |

Primary reproduction evidence is nine-task mean STS, content-word NTP, and global NTP. The untouched model need not lose to NTP on every STS task.'''),
        code(COMMON_SETUP), code(BOOTSTRAP),
        md('''## Rebuild and audit the exact 8k/1k/1k data

The audit hard-fails on split overlap, target misalignment, empty sets, or any concept that is not a complete single token. The observed target is part of every set by construction.'''),
        code(r'''
MODEL_TAG = BASE_MODEL.split("/")[-1].lower()
LEAF = DATA / "c4" / MODEL_TAG / "embedding"
if RUN_DATA:
    # The extraction is the longest stage.  Ten independent 1k shards are
    # cached after completion, so a Colab disconnect loses at most one shard.
    restore_dataset_from_drive(LEAF)
    # get_content_words.py streams into combined.jsonl, so an interrupted pass
    # leaves a SHORT file behind.  Existence is not completion: check the line
    # count, or extraction silently runs on a truncated corpus and only the merge
    # notices, hours later.
    combined = LEAF.parent / "combined.jsonl"
    if combined.is_file():
        rows = sum(1 for _ in combined.open())
        if rows != EXTRACT_SEQUENCES:
            print(f"combined.jsonl has {rows} rows, expected {EXTRACT_SEQUENCES}; regenerating")
            combined.unlink()
    if not combined.is_file():
        run([sys.executable, "data/get_content_words.py", "--model", BASE_MODEL,
             "--dataset", "c4", "--max_length", "256"], cwd=EXT)
        cache_dataset_to_drive(LEAF)
    # An interrupted shard leaves PARTIAL synonyms_/topk_ files behind, so their
    # mere existence does not mean the shard finished.  Record completion in a
    # Drive-side manifest instead; embedding_synonyms.py truncates both files when
    # it restarts a shard, so a re-run is always clean.
    shards_done = load_runs("task15_shards")
    for start in range(0, EXTRACT_SEQUENCES, 1000):
        end = start + 1000
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
        save_runs(shards_done, "task15_shards")
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
         "--report", DRIVE_RESULTS / "data_audit.json"], cwd=EXT)
'''),
        md('''## Unit tests and eight-row GPU smoke run

The smoke run checks the complete QLoRA path before any sweep. It is deleted immediately. The tests cover the released loss, our optional objectives, gradients, and hierarchy sequence scoring.'''),
        code(r'''
if RUN_SMOKE:
    run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=EXT)
    # No map-style preprocessing cache is used. Two independent loads must
    # still produce identical candidate supervision.
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
REPRO_RUNS = load_runs("task15")
if RUN_SCREEN:
    REPRO_RUNS["ntp_seed42"] = train_flat("ntp", 42, 0.0)
    for lam in [0.25, 0.5, 0.75, 1.0]:
        REPRO_RUNS[f"zhang_lambda{lam}_seed42"] = train_flat("zhang_marginal", 42, lam)
    REPRO_RUNS["augmented_ntp_seed42"] = train_flat("augmented_ntp", 42, 0.0, epochs=1,
        train_file=LEAF / "synonyms_train_aug5x.jsonl", data_augmentation=True)
    REPRO_RUNS["randomized_seed42"] = train_flat("randomized", 42, 0.25, randomized=True)

if RUN_CONFIRM:
    for seed in SEEDS:
        REPRO_RUNS[f"ntp_seed{seed}"] = train_flat("ntp", seed, 0.0)
        REPRO_RUNS[f"augmented_ntp_seed{seed}"] = train_flat("augmented_ntp", seed, 0.0,
            epochs=1, train_file=LEAF / "synonyms_train_aug5x.jsonl", data_augmentation=True)
        REPRO_RUNS[f"randomized_seed{seed}"] = train_flat("randomized", seed, 0.25, randomized=True)
        REPRO_RUNS[f"zhang_seed{seed}"] = train_flat("zhang_marginal", seed, 1.0)
    # The lambda sweep already trained zhang_marginal at lambda=1.0 on seed 42, and
    # adapter_path() maps both calls to the same directory.  Two dict keys pointing at
    # one adapter would score it twice and report it as two arms.
    REPRO_RUNS.pop("zhang_lambda1.0_seed42", None)
save_runs(REPRO_RUNS, "task15")

# Use only if the released-batch seed-42 reproduction misses the stated trend.
# This is a named sensitivity run, never a replacement or a cherry-picked row.
RERUN_PAPER_STATED_BATCH = False
if RERUN_PAPER_STATED_BATCH:
    REPRO_RUNS["zhang_seed42_paper_batch"] = train_flat(
        "zhang_marginal_paper_batch", 42, 1.0, batch=2, accum=1)
'''),
        md('''## Evaluation and learning curves

Every checkpoint is scored by the same evaluators. “Global NLL” covers every next-token position; “content-word NLL” covers the semantic slots; “set mass” is total probability assigned to the gold-inclusive valid set. SWORDS and bm-semlex are zero-shot here.'''),
        code(r'''
if RUN_EVAL:
    # A fresh session has the manifest but not the weights; pull them back first.
    REPRO_RUNS = restore_all(REPRO_RUNS)
    checkpoints = [BASE_MODEL, *map(str, REPRO_RUNS.values())]
    result_dir = DRIVE_RESULTS / "task15_reproduction"
    result_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "eval/eval_perplexity_explicit.py", "--checkpoints", *checkpoints,
         "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_test.jsonl",
         "--output", result_dir / "perplexity.json"], cwd=EXT)
    run([sys.executable, "eval/eval_concept_sets.py", "--checkpoints", *checkpoints,
         "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_test.jsonl",
         "--output", result_dir / "concept_sets.json"], cwd=EXT)
    for label, checkpoint in {"pretrained": BASE_MODEL, **REPRO_RUNS}.items():
        display_label = label.replace("_", " ")
        mteb_args = [sys.executable, "eval/eval_mteb.py", "--base-model", BASE_MODEL,
                     "--dataset", "c4", "--dataset-type", "embedding", "--tasks", "sts",
                     "--run-label", display_label,
                     "--csv-output", result_dir / f"sts_{display_label}.csv",
                     "--mteb-output-root", result_dir / "mteb_raw"]
        mteb_args += ["--no-adapter"] if checkpoint == BASE_MODEL else ["--adapter-path", checkpoint]
        run(mteb_args, cwd=EXT)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_swords.py",
         "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
         "--swords_json", MAIN / "data/swords/swords-v1.1_dev.json.gz",
         "--results_json", result_dir / "swords_zero_shot.json", "--modes", "left", "full"], cwd=MAIN)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/paired_benchmark_ci.py",
         "--kind", "swords", "--results-json", result_dir / "swords_zero_shot.json",
         "--baseline-index", "0", "--output", result_dir / "swords_paired_ci_vs_pretrained.json"], cwd=MAIN)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_bm_semlex.py",
         "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
         "--data", MAIN / "data/bm_semlex/curated_200.tsv",
         "--results_json", result_dir / "bm_semlex.json"], cwd=MAIN)
    manifest = {"pretrained": BASE_MODEL, **{label.replace("_", " "): str(path) for label, path in REPRO_RUNS.items()}}
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    run([sys.executable, MAIN / "scripts/summarize_concept_experiments.py",
         "--manifest", result_dir / "manifest.json", "--result-dir", result_dir,
         "--output", result_dir / "flat_main_table.csv"], cwd=MAIN)
    for label, path in REPRO_RUNS.items(): sync_small_artifacts(path, f"task15_logs/{label}")
    audit_no_drive_weights()
'''),
        code(r'''
# Plot only scalar learning curves; adapters remain under /content.
import pandas as pd, seaborn as sns
from matplotlib import pyplot as plt
curves = []
for label, path in REPRO_RUNS.items():
    history = Path(path) / "training_history.jsonl"
    if history.exists():
        frame = pd.read_json(history, lines=True)
        frame["method"] = label
        curves.append(frame)
if curves:
    frame = pd.concat(curves, ignore_index=True)
    value = "loss" if "loss" in frame else "ce_loss"
    sns.lineplot(frame, x="step", y=value, hue="method")
    plt.tight_layout(); plt.savefig(DRIVE_RESULTS / "task15_reproduction" / "training_loss.png", dpi=180)
'''),
        md(r'''## Reproduction gate

Proceed only if Zhang concept marginal beats NTP and augmented NTP on mean STS, improves content-word perplexity over NTP, stays near pretrained global perplexity, and the $\lambda$ curve has the reported direction. If seed 42 fails, rerun that one setting with the paper-stated effective batch and report both configurations—do not silently substitute it.'''),
    ]
    write("research_tasks_15_zhang_reproduction.ipynb", cells)


def task15b():
    cells = [
        md('''# Task 15b — Uniform concept pressure and conservative contrastive separation

This notebook starts **only after Task 15 passes**. It compares two additions to the same Zhang C4 pipeline:

| Method | Question |
|---|---|
| Uniform concept + retention | Does raising every accepted alternative, while retaining ordinary NTP at the observed slot, improve robust concept coverage? |
| Uniform + retention + contrastive | Does explicitly separating accepted from conservative invalid candidates improve SWORDS AUROC? |

The contrastive term is joint token-level training. It is not DPO and not sentence-level SimCSE.'''),
        code(COMMON_SETUP), code(BOOTSTRAP), code(TRAIN_HELPER),
        md('''## Build conservative negatives

Candidates must occur in the model’s top-100 next-token pool, match POS, lie outside the accepted set, share no WordNet synset, not be morphological variants, and fall below the contextual-similarity ceiling. Coverage and every rejection reason are reported.'''),
        code(r'''
MODEL_TAG = BASE_MODEL.split("/")[-1].lower()
LEAF = DATA / "c4" / MODEL_TAG / "embedding"
NEG_TRAIN = LEAF / "synonyms_train_conservative_negatives.jsonl"
if RUN_DATA:
    run([sys.executable, "data/build_contrastive_negatives.py",
         "--source", LEAF / "synonyms_train.jsonl",
         "--topk", DATA / "c4" / MODEL_TAG / "prompting" / "topk_*.jsonl",
         "--output", NEG_TRAIN,
         "--report", DRIVE_RESULTS / "contrastive_negative_report.json",
         "--max-cosine", "0.35", "--max-negatives", "20"], cwd=EXT)
    cache_dataset_to_drive(LEAF)
'''),
        md(r'''## Screen only seed 42

$\alpha\in\{.5,1,2,4\}$ controls uniform pressure. We set the concept-slot NTP weight to 1, so the observed-token term is never removed. Select the largest concept improvement whose global NLL is at most 0.20 above matched NTP. Then hold $\alpha$ fixed and screen $\beta\in\{.25,.5,1\}$.'''),
        code(r'''
OBJECTIVE_RUNS = load_runs("task15b")
if RUN_SCREEN:
    for alpha in [0.5, 1.0, 2.0, 4.0]:
        OBJECTIVE_RUNS[f"uniform_alpha{alpha}_seed42"] = train_flat(
            "uniform_retention", 42, alpha, objective="uniform", slot_ntp_weight=1.0)

# Set this only after the NLL/set-mass screen printed below.
SELECTED_ALPHA = None
if RUN_SCREEN and SELECTED_ALPHA is not None:
    for beta in [0.25, 0.5, 1.0]:
        OBJECTIVE_RUNS[f"contrast_alpha{SELECTED_ALPHA}_beta{beta}_seed42"] = train_flat(
            "uniform_contrastive", 42, SELECTED_ALPHA, objective="uniform",
            slot_ntp_weight=1.0, contrast_beta=beta, train_file=NEG_TRAIN)
'''),
        md('''## Screen evaluation and locked confirmation

Contrastive is promoted only when it improves SWORDS acceptable/rejected AUROC over the identical uniform model while preserving STS and NTP. It is not promoted for a better training loss alone.'''),
        code(r'''
if RUN_EVAL:
    OBJECTIVE_RUNS = restore_all(OBJECTIVE_RUNS)
    result_dir = DRIVE_RESULTS / "task15b_screen"; result_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [str(x) for x in OBJECTIVE_RUNS.values()]
    run([sys.executable, "eval/eval_perplexity_explicit.py", "--checkpoints", *checkpoints,
         "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_test.jsonl",
         "--output", result_dir / "perplexity.json"], cwd=EXT)
    run([sys.executable, "eval/eval_concept_sets.py", "--checkpoints", *checkpoints,
         "--base-model", BASE_MODEL, "--test-jsonl", LEAF / "synonyms_test.jsonl",
         "--output", result_dir / "concept_sets.json"], cwd=EXT)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_swords.py",
         "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
         "--swords_json", MAIN / "data/swords/swords-v1.1_dev.json.gz",
         "--results_json", result_dir / "swords.json", "--modes", "left", "full"], cwd=MAIN)
    # Set this index to the locked uniform run when evaluating the beta screen.
    PAIRED_BASELINE_INDEX = 0
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/paired_benchmark_ci.py",
         "--kind", "swords", "--results-json", result_dir / "swords.json",
         "--baseline-index", PAIRED_BASELINE_INDEX,
         "--output", result_dir / "swords_paired_ci.json"], cwd=MAIN)
    for label, checkpoint in OBJECTIVE_RUNS.items():
        display_label = label.replace("_", " ")
        run([sys.executable, "eval/eval_mteb.py", "--base-model", BASE_MODEL,
             "--dataset", "c4", "--dataset-type", "embedding", "--tasks", "sts",
             "--run-label", display_label, "--csv-output", result_dir / f"sts_{display_label}.csv",
             "--mteb-output-root", result_dir / "mteb_raw",
             "--adapter-path", checkpoint], cwd=EXT)
    run([sys.executable, MAIN / "transformers/examples/pytorch/language-modeling/eval_bm_semlex.py",
         "--checkpoints", *checkpoints, "--tokenizer_path", BASE_MODEL, "--base_model", BASE_MODEL,
         "--data", MAIN / "data/bm_semlex/curated_200.tsv",
         "--results_json", result_dir / "bm_semlex.json"], cwd=MAIN)
    manifest = {label.replace("_", " "): str(path) for label, path in OBJECTIVE_RUNS.items()}
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    run([sys.executable, MAIN / "scripts/summarize_concept_experiments.py",
         "--manifest", result_dir / "manifest.json", "--result-dir", result_dir,
         "--output", result_dir / "flat_extension_table.csv"], cwd=MAIN)
'''),
        code(r'''
# Lock these from the seed-42 screen; do not choose them per seed.
SELECTED_BETA = None
if RUN_CONFIRM:
    assert SELECTED_ALPHA is not None
    for seed in SEEDS:
        OBJECTIVE_RUNS[f"uniform_seed{seed}"] = train_flat(
            "uniform_retention", seed, SELECTED_ALPHA, objective="uniform", slot_ntp_weight=1.0)
        if SELECTED_BETA is not None:
            OBJECTIVE_RUNS[f"contrastive_seed{seed}"] = train_flat(
                "uniform_contrastive", seed, SELECTED_ALPHA, objective="uniform",
                slot_ntp_weight=1.0, contrast_beta=SELECTED_BETA, train_file=NEG_TRAIN)
    for label, path in OBJECTIVE_RUNS.items(): sync_small_artifacts(path, f"task15b_logs/{label}")
    audit_no_drive_weights()
save_runs(OBJECTIVE_RUNS, "task15b")
'''),
        md('''## Decision

Keep at most one of these as a headline extension. If contrastive does not beat the same uniform model on human-labelled SWORDS separation, report it as a negative ablation or omit it. Do not expand to 3B here; the hierarchy experiment is next.'''),
    ]
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
         "--hyperlex", MAIN / "data/hyperlex-data/hyperlex-all.txt",
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
    # for 3B, audit 8k/1k/1k, and rebuild hierarchy labels before training.
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
         "--hyperlex", MAIN / "data/hyperlex-data/hyperlex-all.txt", "--pos", "N", "V",
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
