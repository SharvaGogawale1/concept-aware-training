#!/usr/bin/env bash
# One family, end to end, unattended: mine the verified data if it is not there,
# gate it, smoke the arms and apply the lambda rule, train the six arms, evaluate,
# print the two-claim gate.  Every stage resumes, so rerunning after a failure
# continues rather than repeats.
#
#   run_verified_pipeline.sh <model-tag> <model-id> <gpu>       (HF_TOKEN in env for gated models)
#   run_verified_pipeline.sh qwen3-1.7b-base Qwen/Qwen3-1.7B-Base 2
#   run_verified_pipeline.sh llama-3.2-1b   meta-llama/Llama-3.2-1B 3
set -euo pipefail
TAG=$1; MODEL=$2; GPU=$3
BASE=${CONCEPT_BASE:-$PWD}
cd "$BASE"
REPO=concept_aware/concept-aware-training
D=concept_aware/data/c4/$TAG
OUT=outputs/$TAG
export HF_HOME=$BASE/concept_aware/hf_cache PYTHONUTF8=1 PYTHONIOENCODING=utf-8 CONCEPT_BASE=$BASE
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }

git -C $REPO pull -q --ff-only || log "git pull failed; continuing with the checkout as is"

# ---- 1. verified data ------------------------------------------------------------
if [ -f "$OUT/verified_report.json" ] && [ -f "$D/embedding/synonyms_train_verified.jsonl" ]; then
  log "verified data present; skipping the miner"
else
  if pgrep -f "build_verified_negatives.py.*$D/embedding/synonyms_train.jsonl" >/dev/null; then
    log "a miner for $TAG is already running; waiting for its report"
    while [ ! -f "$OUT/verified_report.json" ]; do sleep 120; done
  else
    # A partial output with no report is a killed run: replace it.
    [ -f "$D/embedding/synonyms_train_verified.jsonl" ] && rm -f "$D/embedding/synonyms_train_verified.jsonl"
    log "mining verified negatives for $TAG on GPU $GPU (~10 h)"
    CUDA_VISIBLE_DEVICES=$GPU python $REPO/scripts/build_verified_negatives.py \
      --source $D/embedding/synonyms_train.jsonl --topk "$D/prompting/topk_*.jsonl" \
      --model "$MODEL" --ext concept_aware/learning-concepts/data \
      --output $D/embedding/synonyms_train_verified.jsonl \
      --report $OUT/verified_report.json --sample $OUT/verified_sample_50.csv \
      --prune-positives --overwrite 2>&1 | tee -a verify_$TAG.log
  fi
fi

# ---- 2. flags, then the notebook: gate -> smoke + lambda rule -> train -> eval --------
cp $REPO/notebooks/reproducibilty_15b.ipynb reproducibilty_15b_$TAG.ipynb
python3 $REPO/scripts/set_15b_flags.py reproducibilty_15b_$TAG.ipynb --gpu "$GPU" --base "$BASE" \
  --model-tag "$TAG" --stage verified --smoke-steps 40
# The control cell names the model; the flag script keeps whatever is there, so set it.
python3 - "$MODEL" reproducibilty_15b_$TAG.ipynb <<'PY'
import json, re, sys
model, path = sys.argv[1], sys.argv[2]
nb = json.load(open(path))
for c in nb["cells"]:
    src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    if "THE ONLY CELL YOU EDIT" in src:
        src = re.sub(r'^MODEL    = "[^"]*"', f'MODEL    = "{model}"', src, flags=re.M)
        c["source"] = src
json.dump(nb, open(path, "w"), indent=1)
print("MODEL set to", model)
PY
log "running the notebook for $TAG on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU papermill reproducibilty_15b_$TAG.ipynb executed_15b_$TAG.ipynb -k concept --log-output 2>&1 | tee -a verified_$TAG.log
log "done: $OUT/task15b_verified"
