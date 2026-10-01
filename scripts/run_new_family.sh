#!/usr/bin/env bash
# A new model family, end to end, on two GPUs, unattended: every table row Llama-1B and
# Qwen-1.7B have, with the same locked settings (shared gammas .0625 and .125, seeds
# 42/123/2024, mass at both gammas, seed 42).  Every phase is a run_verified_pipeline.sh
# stage and every stage resumes, so rerunning this script after a crash skips finished work.
#
#   run_new_family.sh <model-tag> <model-id> <gpu-a> <gpu-b>
#   run_new_family.sh falcon3-1b-base tiiuae/Falcon3-1B-Base 2 3
#
# Phase 1, gpu-a   Task 15 first pass: extract the concept sets (~5.5 h on an A40), smoke,
#                  the seed-42 baselines and randomized lambda=1.  No scoring.
# Phase 2, gpu-a   Task 15 seeds 123/2024 of every baseline, then Task 15's table; then,
#                  once gpu-b's arms exist, run_post_evals.sh (harness, word similarity,
#                  classification probe).
#          gpu-b   the 15b arms: uniform and mass at both gammas (seed 42) and uniform at
#                  seeds 123/2024; then, once gpu-a has TRAINED the baseline seeds, the dev
#                  table (task15b_screen_shared_gamma); then the one locked test pass.
#
# Run from the task15 base directory (holding concept_aware/ and outputs/), in tmux.
# Pull the repo once before launching: nothing here pulls.  Logs: family_<tag>_*.log.
set -euo pipefail
TAG=$1; MODEL=$2; GA=$3; GB=$4
BASE=${CONCEPT_BASE:-$PWD}
cd "$BASE"
export CONCEPT_BASE=$BASE
R=concept_aware/concept-aware-training/scripts
M=outputs/$TAG/run_manifests
log() { echo "[$(date '+%F %T')] $*"; }

SEEDS="42 123 2024"
BASELINES=""
for f in ntp augmented_ntp randomized randomized_lambda1.0 zhang; do
  for s in $SEEDS; do BASELINES+=" ${f}_seed$s"; done
done
ARMS="zhang_plus_mass_g0.0625_seed42 zhang_plus_mass_g0.125_seed42"
for g in 0.0625 0.125; do
  for s in $SEEDS; do ARMS+=" zhang_plus_uniform_g${g}_seed$s"; done
done
GRID=uniform:0.0625,uniform:0.125,mass:0.0625,mass:0.125
LOCKED=uniform:0.0625,uniform:0.125

# Block until every named arm is in the manifest with its adapter on disk.
wait_for() {
  local manifest=$1; shift
  until python3 - "$manifest" "$@" <<'PY'
import json, sys
from pathlib import Path
path, want = Path(sys.argv[1]), sys.argv[2:]
have = {k.replace(" ", "_"): v for k, v in json.loads(path.read_text()).items()} if path.is_file() else {}
missing = [a for a in want if a not in have or not (Path(have[a]) / "adapter_config.json").is_file()]
sys.exit(1 if missing else 0)
PY
  do sleep 300; done
}

log "phase 1: $TAG extraction and seed-42 baselines on GPU $GA"
STAGE=task15 RUN_NAME=first bash $R/run_verified_pipeline.sh "$TAG" "$MODEL" "$GA" --no-eval
wait_for $M/task15.json ntp_seed42 augmented_ntp_seed42 randomized_seed42 randomized_lambda1.0_seed42 zhang_seed42

log "phase 2: baseline seeds on GPU $GA (family_${TAG}_gpuA.log), 15b arms on GPU $GB (family_${TAG}_gpuB.log)"
(
  STAGE=confirm15 RUN_NAME=seeds bash $R/run_verified_pipeline.sh "$TAG" "$MODEL" "$GA" --multiseed
  log "waiting for the 15b arms before the post-training evaluations"
  wait_for $M/task15b.json $ARMS
  bash $R/run_post_evals.sh "$TAG" "$MODEL" "$GA"
) > family_${TAG}_gpuA.log 2>&1 &
A=$!
(
  STAGE=hybrid RUN_NAME=arms bash $R/run_verified_pipeline.sh "$TAG" "$MODEL" "$GB" --no-screen \
    --hybrid-arms $GRID --selected-hybrid $LOCKED --multiseed --no-eval
  log "15b arms trained; waiting for the baseline seeds before the dev table"
  wait_for $M/task15.json $BASELINES
  STAGE=hybrid RUN_NAME=sharedgamma bash $R/run_verified_pipeline.sh "$TAG" "$MODEL" "$GB" --no-screen \
    --hybrid-arms $GRID --selected-hybrid $LOCKED --multiseed --result-tag _shared_gamma \
    --eval-arms "$(echo $ARMS | tr ' ' ',')"
  STAGE=hybrid RUN_NAME=test bash $R/run_verified_pipeline.sh "$TAG" "$MODEL" "$GB" --no-screen \
    --selected-hybrid $LOCKED --result-tag _test --swords-test
) > family_${TAG}_gpuB.log 2>&1 &
B=$!
status=0
wait $A || { status=1; log "gpu-a chain failed: see family_${TAG}_gpuA.log"; }
wait $B || { status=1; log "gpu-b chain failed: see family_${TAG}_gpuB.log"; }
log "done: $TAG (status $status)"
exit $status
