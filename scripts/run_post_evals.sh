#!/usr/bin/env bash
# Every table row for one family, after its training is done: lm-eval harness with
# per-question answers, word similarity (MEN / WS-353 / SimLex-999), and Iyer's
# downstream datasets under Zhang's MTEB probe.  Each step resumes from its own JSON,
# so a rerun after a crash repeats nothing already scored.
#
#   run_post_evals.sh <model-tag> <model-id> <gpu>
#   run_post_evals.sh llama-3.2-1b meta-llama/Llama-3.2-1B 1
#   run_post_evals.sh qwen3-1.7b-base Qwen/Qwen3-1.7B-Base 0
#
# Run from the task15 base directory.  Uses two conda envs by their Python paths
# (`conda run` is not reliable in a non-interactive script): `harness` for lm_eval,
# `concept` for the probes; override with HARNESS_PY / CONCEPT_PY.  HF_HOME is the
# one run_verified_pipeline.sh uses, so gated models authenticate the same way.
# Ends with the paired per-question harness test: ours and mass vs Zhang, seed-matched.
set -euo pipefail
TAG=$1; MODEL=$2; GPU=$3
BASE=${CONCEPT_BASE:-$PWD}
cd "$BASE"
H=concept_aware/concept-aware-training/scripts
OUT=outputs/$TAG
M=$OUT/run_manifests
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$BASE/concept_aware/hf_cache PYTHONUTF8=1 PYTHONIOENCODING=utf-8
HARNESS_PY=${HARNESS_PY:-$HOME/miniconda3/envs/harness/bin/python}
CONCEPT_PY=${CONCEPT_PY:-$HOME/miniconda3/envs/concept/bin/python}
log() { echo "[$(date '+%F %T')] $*"; }

SEEDS="42 123 2024"
ARMS="pretrained"
for family in ntp augmented_ntp randomized zhang; do
  for s in $SEEDS; do ARMS+=" ${family}_seed$s"; done
done
for g in 0.0625 0.125; do
  for s in $SEEDS; do ARMS+=" zhang_plus_uniform_g${g}_seed$s"; done
done
ARMS+=" zhang_plus_mass_g0.0625_seed42 zhang_plus_mass_g0.125_seed42"

# One manifest over both notebooks' runs, for the harness (it reads <dir>/manifest.json).
MERGED=$OUT/post_evals_manifest
mkdir -p "$MERGED"
python3 - "$M/task15.json" "$M/task15b.json" "$MERGED/manifest.json" <<'PY'
import json, sys
merged = {}
for path in sys.argv[1:3]:
    merged.update(json.load(open(path)))
json.dump(merged, open(sys.argv[3], "w"), indent=2)
PY
MISSING=$(python3 - "$MERGED/manifest.json" $ARMS <<'PY'
import json, sys
have = {k.replace(" ", "_") for k in json.load(open(sys.argv[1]))}
print(" ".join(a for a in sys.argv[2:] if a != "pretrained" and a not in have))
PY
)
if [ -n "$MISSING" ]; then
  log "not trained yet, stopping before any scoring: $MISSING"; exit 1
fi

for py in "$HARNESS_PY" "$CONCEPT_PY"; do
  [ -x "$py" ] || { log "missing interpreter: $py (set HARNESS_PY / CONCEPT_PY)"; exit 1; }
done

# The harness always scores the untouched model first; it is not a manifest arm.
log "harness: $(echo $ARMS | wc -w) checkpoints on GPU $GPU"
"$HARNESS_PY" $H/run_harness_eval.py \
  --screen "$MERGED" --base-model "$MODEL" --out $OUT/harness_confirm.json --log-samples --arms ${ARMS#pretrained }

log "word similarity"
"$CONCEPT_PY" $H/eval_word_similarity.py \
  --base-model "$MODEL" --manifests $M/task15.json $M/task15b.json --arms $ARMS \
  --out $OUT/word_similarity.json

log "classification probe (Iyer's datasets, Zhang's protocol)"
"$CONCEPT_PY" $H/eval_classification_probe.py \
  --base-model "$MODEL" --manifests $M/task15.json $M/task15b.json --arms $ARMS \
  --out $OUT/classification_probe.json
log "paired harness test: ours and mass vs Zhang, seed-matched"
PAIRS=""
for g in 0.0625 0.125; do
  for s in $SEEDS; do PAIRS+=" --pair zhang_seed$s:zhang_plus_uniform_g${g}_seed$s"; done
  PAIRS+=" --pair zhang_seed42:zhang_plus_mass_g${g}_seed42"
done
"$CONCEPT_PY" $H/paired_harness_ci.py --harness $OUT/harness_confirm.json $PAIRS
log "done: $TAG"
