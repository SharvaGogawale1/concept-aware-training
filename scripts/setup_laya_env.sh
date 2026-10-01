#!/usr/bin/env bash
# One-time setup for run_iyer_replication.sh: Laya Iyer's released repo at a pinned commit,
# and a `laya` conda env that runs it on her own transformers fork (4.45.0.dev0).  Her
# scripts cannot run on the `concept` env's transformers 4.57: its Trainer passes
# num_items_in_batch to compute_loss, which her CustomTrainer does not accept.
#
# Pins follow her requirements.txt where it names a version (Python 3.12, torch 2.4.0,
# datasets 2.19.1); that file is a CPU-only conda spec, so torch comes from the CUDA 12.1
# wheel index instead.  The rest are pinned to the same period so pip cannot pull a
# release newer than her fork.
#
#   cd ~/fos_retrieval/task15 && bash concept_aware/concept-aware-training/scripts/setup_laya_env.sh
set -euo pipefail
BASE=${CONCEPT_BASE:-$PWD}
LAYA=$BASE/concept_aware/laya-concept-aware-training
COMMIT=a955e7543c9ef79c20a29535b12d7163698c5711          # github.com/layaiyer1/concept-aware-training, 2026-05-05
CONDA=${CONDA:-$HOME/miniconda3/bin/conda}
ENV_PY=$HOME/miniconda3/envs/laya/bin/python

if [ ! -d "$LAYA/.git" ]; then
  git clone -q https://github.com/layaiyer1/concept-aware-training "$LAYA"
fi
git -C "$LAYA" fetch -q origin
git -C "$LAYA" checkout -q "$COMMIT"
echo "Laya's repo at $(git -C "$LAYA" rev-parse HEAD)"

if [ ! -x "$ENV_PY" ]; then
  "$CONDA" create -y -q -n laya -c conda-forge --override-channels python=3.12   # no Anaconda ToS prompt
fi
"$ENV_PY" -m pip install -q --no-cache-dir "numpy<2"
"$ENV_PY" -m pip install -q --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
"$ENV_PY" -m pip install -q --no-cache-dir -e "$LAYA/transformers"
"$ENV_PY" -m pip install -q --no-cache-dir "datasets==2.19.1" "pyarrow==16.1.0" "huggingface-hub==0.24.6" \
  "accelerate==0.34.2" "evaluate==0.4.3" scikit-learn pandas sentencepiece protobuf
"$ENV_PY" - <<'PY'
import torch, transformers, datasets, accelerate
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__, "from", transformers.__file__.split("/src/")[0],
      "| datasets", datasets.__version__, "| accelerate", accelerate.__version__)
assert transformers.__version__ == "4.45.0.dev0", "not Laya's fork"
PY
