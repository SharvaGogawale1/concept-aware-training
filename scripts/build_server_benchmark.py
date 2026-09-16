#!/usr/bin/env python3
"""Emit the server extraction benchmark notebook.

It runs the GENUINE Task 15 extraction commands -- upstream's
`data/get_content_words.py` and `data/embedding_synonyms.py`, same flags, same
`--no-4bit` as EXTRACT_4BIT=False -- on a few sequences, so the seconds-per-
sequence it prints is directly comparable to the 5.9 s/seq measured on a Colab L4.
A proxy microbenchmark was the earlier design and was wrong: it timed a guess at
the extractor's shape rather than the extractor.

The GPU shard is run at two sizes and the MARGINAL rate reported, because a single
small shard is dominated by the one-off model load and would flatter the machine.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "server_gpu_benchmark.ipynb"

def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}

CELLS = [
md(r'''
# Server extraction benchmark

Runs the real Task 15 extraction for **Qwen3-1.7B** on a handful of sequences and
reports seconds per sequence, to decide whether Qwen extraction belongs on this
server or on Colab.

The reference below was measured on Llama-3.2-1B, which is 1.24B against Qwen3-1.7B's
1.72B. Scaling by parameters, an L4-equal machine should land near 5.4 s/seq on the
GPU stage and 7.3 end to end.

| Reference, Llama-3.2-1B | s/seq |
|---|---|
| Colab L4, GPU stage (`embedding_synonyms.py`) | 3.9 |
| Colab L4, end to end including the spaCy pass | 5.9 |
| This server, August 2026 | 49 |

Set `GPU_ID` in cell 2, then Run All. Budget 15 to 25 minutes: package install and
the model download dominate, and if the machine is still at August speed the GPU
shard alone takes about 15 of those.

Everything lands beside this notebook.

```
<this directory>/
├── server_gpu_benchmark.ipynb
├── cache/                  model weights and datasets (HF_HOME)
├── outputs/                benchmark JSON to send back
├── data/                   extracted concept data (CONCEPT_DATA_ROOT)
└── learning-concepts/      upstream repo, pinned commit
```
'''),

md("## 1. Which GPUs are free\n\nRead this before setting `GPU_ID`. Pick one with free memory and no other process."),
code(r'''
!nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
'''),

md("## 2. Pick the GPU and lay out the directories\n\n**Set `GPU_ID` here**, before anything initialises CUDA. Getting it wrong means restarting the kernel, not just rerunning the cell."),
code(r'''
GPU_ID = "0"          # <-- SET THIS from the table above

import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU_ID", GPU_ID)

from pathlib import Path
import json, platform, re, subprocess, sys, time

def _notebook_dir():
    override = os.environ.get("CONCEPT_BASE")
    if override:
        return Path(override)
    # Jupyter sets JPY_SESSION_NAME to the notebook's own path.  The working
    # directory equals the notebook's directory only when the kernel happened to
    # start there, so preferring the session path stops a second cache/ tree from
    # appearing wherever the kernel was launched from.
    session = os.environ.get("JPY_SESSION_NAME", "")
    if session.endswith(".ipynb") and Path(session).parent.is_dir():
        return Path(session).parent
    return Path.cwd()

BASE = _notebook_dir().resolve()
CACHE, OUTPUTS = BASE / "cache", BASE / "outputs"
DATA, EXT = BASE / "data", BASE / "learning-concepts"
MAIN = BASE / "concept-aware-training"      # our repo: it carries the upstream patch
for path in (CACHE, OUTPUTS, DATA):
    path.mkdir(parents=True, exist_ok=True)

# Point the HF libraries at cache/ BEFORE importing them.  The older per-library
# variables are set too: the pinned transformers still reads TRANSFORMERS_CACHE.
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(CACHE / "datasets")
# A gated download needs a token.  `huggingface-cli login` writes it under the
# DEFAULT HF_HOME, which we just moved, so carry it forward or every child
# process 401s against a cache that looks fine from here.
_token = Path.home() / ".cache" / "huggingface" / "token"
if not os.environ.get("HF_TOKEN") and _token.is_file():
    os.environ["HF_TOKEN"] = _token.read_text().strip()

# Qwen3-1.7B: ungated, and the model this benchmark exists to budget for.  Set it
# to Qwen/Qwen3-4B to measure that instead -- but the two share a tokenizer, so
# 4B reuses this model's content words and pays only the GPU stage in a real run.
MODEL = "Qwen/Qwen3-1.7B"
UPSTREAM_COMMIT = "b1d414143d11c8ed988b4cccbb06626cc8272bbe"
# Rows for the spaCy/tokenizer pass, and the two GPU shard sizes.  The shard is
# run twice so the fixed model load cancels out of the marginal rate.
BENCH_ROWS, SHARD_SMALL, SHARD_LARGE = 120, 5, 15

def run(argv, cwd=None, quiet=False):
    """Run a child process with the project's environment, returning its output."""
    argv = list(map(str, argv))
    if not quiet:
        print("+", " ".join(argv), flush=True)
    environment = {**os.environ, "CONCEPT_DATA_ROOT": str(DATA)}
    done = subprocess.run(argv, cwd=cwd, env=environment, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if done.returncode:
        print(done.stdout[-3000:])
        raise RuntimeError(f"exit {done.returncode}: {' '.join(argv[:4])}")
    return done.stdout

print("base   ", BASE)
print("GPU_ID ", os.environ["CUDA_VISIBLE_DEVICES"], "| host", platform.node())
'''),

md("## 3. Dependencies\n\nThe transformers pin matters: upstream's extractor reuses a prefix KV cache through an API removed in v5."),
code(r'''
%pip install -q "transformers>=4.51,<4.58" accelerate datasets spacy bitsandbytes scipy
!python -m spacy download en_core_web_sm
import torch, transformers
print("transformers", transformers.__version__, "| torch", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("no GPU visible -- fix GPU_ID in cell 2 and restart the kernel")
props = torch.cuda.get_device_properties(0)
GPU_INFO = {"name": props.name, "total_gb": round(props.total_memory / 1e9, 1),
            "sms": props.multi_processor_count}
print(f"device: {GPU_INFO['name']}  |  {GPU_INFO['total_gb']} GB  |  SMs {GPU_INFO['sms']}")
'''),

md("""## 4. Clone upstream, then apply our patch

Task 15 does not run upstream as released. It pins the commit and applies
`external/learning-concepts.patch` from our repository, which is what adds
`--no-4bit`, `--seed` and the objective flags. Timing the unpatched tree would
measure 4-bit extraction, not the bf16 extraction the real run does."""),
code(r'''
if not (MAIN / ".git").is_dir():
    run(["git", "clone", "https://github.com/SharvaGogawale1/concept-aware-training.git", str(MAIN)])
else:
    run(["git", "-C", str(MAIN), "pull", "--ff-only"])
if not (EXT / ".git").is_dir():
    run(["git", "clone", "https://github.com/christine-zhang1/learning-concepts.git", str(EXT)])

patch_file = MAIN / "external" / "learning-concepts.patch"
assert patch_file.is_file(), f"missing {patch_file}"
# Reset and clean before applying, so re-running this cell is idempotent rather
# than failing on an already-patched tree.  EXT holds upstream code only; the
# corpus lives in DATA, so wiping it is safe.
run(["git", "-C", str(EXT), "checkout", "--detach", UPSTREAM_COMMIT])
run(["git", "-C", str(EXT), "reset", "--hard", UPSTREAM_COMMIT])
run(["git", "-C", str(EXT), "clean", "-fdq"])
run(["git", "apply", str(patch_file)], cwd=EXT)
run([sys.executable, "-m", "pip", "install", "-q", "-e", str(EXT), "--no-deps"])
print("patch applied onto", UPSTREAM_COMMIT[:7])
'''),

md('''## 5. Stage one: content words (CPU, spaCy)

`get_content_words.py` has its row count hardcoded, and the real run processes
10,000 rows. For a benchmark that constant is rewritten to a small number, timed,
and reported per row. The file is restored afterwards so a later real run is
unaffected.

Stage one is CPU-bound spaCy plus tokenisation. In a real run it is paid once per
tokenizer, so Qwen3-4B inherits it from Qwen3-1.7B and skips this entirely.'''),
code(r'''
source = EXT / "data" / "get_content_words.py"
original = source.read_text()

# The row cap is a module constant upstream, not a flag.  Rewrite it for the
# benchmark and assert the rewrite landed: silently failing here would time the
# full 10,000-row pass and look like a catastrophically slow machine.
patched, hits = re.subn(r"^(MAX_SAMPLES\s*=\s*)\d+", rf"\g<1>{BENCH_ROWS}", original, flags=re.M)
if hits != 1:
    print("could not find MAX_SAMPLES; lines mentioning a row cap:")
    for line in original.splitlines():
        if re.search(r"MAX_SAMPLES|max_samples|10000", line):
            print("   ", line.strip())
    raise SystemExit("send those lines back rather than guessing")
source.write_text(patched)

start = time.perf_counter()
run([sys.executable, "data/get_content_words.py", "--model", MODEL,
     "--dataset", "c4", "--max_length", "256"], cwd=EXT)
content_seconds = time.perf_counter() - start
source.write_text(original)          # leave the checkout as we found it

MODEL_TAG = MODEL.split("/")[-1].lower()
LEAF = DATA / "c4" / MODEL_TAG / "embedding"
combined = LEAF.parent / "combined.jsonl"
rows = sum(1 for _ in combined.open()) if combined.is_file() else 0
content_rate = content_seconds / max(rows, 1)
print(f"\nmodel {MODEL}")
print(f"content words: {rows} rows in {content_seconds:.0f}s = {content_rate:.2f} s/row")
'''),

md('''## 6. Stage two: embedding synonyms (GPU)

The stage that dominates the budget, and the one the L4's 3.9 s/seq refers to.
`--no-4bit` matches `EXTRACT_4BIT = False` in the real notebook.

Run at two sizes. A single small shard is dominated by the one-off model load, so
the honest figure is the marginal rate between them.'''),
code(r'''
def shard(end):
    start = time.perf_counter()
    run([sys.executable, "data/embedding_synonyms.py", "c4", "--start", 0, "--end", end,
         "--model", MODEL, "--no-4bit"], cwd=EXT)
    return time.perf_counter() - start

small_seconds = shard(SHARD_SMALL)
print(f"{SHARD_SMALL} sequences: {small_seconds:.0f}s")
large_seconds = shard(SHARD_LARGE)
print(f"{SHARD_LARGE} sequences: {large_seconds:.0f}s")

# (t_large - t_small) / (n_large - n_small) removes the fixed model load, which
# is charged once per invocation and would otherwise inflate the per-sequence
# cost several-fold at these tiny shard sizes.
marginal = (large_seconds - small_seconds) / (SHARD_LARGE - SHARD_SMALL)
load_overhead = small_seconds - marginal * SHARD_SMALL
print(f"\nmarginal GPU rate: {marginal:.2f} s/seq   (model load about {load_overhead:.0f}s per shard)")
print("Colab L4 measured 3.9 s/seq for this stage on Llama-3.2-1B, which is"
      " 1.24B against Qwen3-1.7B's 1.72B, so expect roughly 5.4 on an L4-equal machine.")
'''),

md("## 7. Summary\n\nPaste this block back."),
code(r'''
end_to_end = marginal + content_rate
projected_4000 = end_to_end * 4000 / 3600
# Qwen3-4B is 4.02B against 1.72B and shares this tokenizer, so it pays the GPU
# stage at the parameter ratio and inherits the content words for free.
projected_4b = marginal * (4.02 / 1.72) * 4000 / 3600
summary = {"host": platform.node(), "gpu": GPU_INFO, "model": MODEL,
           "content_words_s_per_row": round(content_rate, 2),
           "embedding_marginal_s_per_seq": round(marginal, 2),
           "embedding_load_overhead_s": round(load_overhead, 1),
           "end_to_end_s_per_seq": round(end_to_end, 2),
           "projected_hours_for_4000_sequences": round(projected_4000, 1),
           "projected_hours_qwen3_4b_gpu_stage_only": round(projected_4b, 1),
           "colab_l4_reference": {"gpu_stage": 3.9, "end_to_end": 5.9, "model": "meta-llama/Llama-3.2-1B"},
           "august_server_measurement_s_per_seq": 49}
path = OUTPUTS / "server_benchmark.json"
path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("\nwritten to", path)
print(f"\n{end_to_end:.1f} s/seq end to end. An L4 would be about 7.3 for this model.")
print(f"Projected {projected_4000:.1f} h for Qwen3-1.7B's 4000 sequences,")
print(f"plus {projected_4b:.1f} h for Qwen3-4B, which reuses these content words.")
print("Under about 8 s/seq the server is worth using; near 49 it is August's machine again.")
'''),
]

NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps({
    "cells": CELLS,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5}, indent=1))
print("wrote", NB, f"({len(CELLS)} cells)")
