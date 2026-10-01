#!/usr/bin/env bash
# A new model family, end to end, on two GPUs, unattended: every table row Llama-1B and
# Qwen-1.7B have, with the same locked settings (shared gammas .0625 and .125, seeds
# 42/123/2024, mass at both gammas, seed 42).  Every phase is a run_verified_pipeline.sh
# stage and every stage resumes, so rerunning this script after a crash skips finished work.
#
#   run_new_family.sh <model-tag> <model-id> <gpu-a> <gpu-b>
#   run_new_family.sh falcon3-1b-base tiiuae/Falcon3-1B-Base 0 1
#
# Phase 1, gpu-a   Task 15 first pass: extract the concept sets (~5.5 h on an A40), smoke,
#                  the seed-42 baselines and randomized lambda=1.  No scoring.
# Phase 2, gpu-b   the 15b arms: uniform and mass at both gammas (seed 42) and uniform at
#                  seeds 123/2024; then, once gpu-a has TRAINED the baseline seeds, the dev
#                  table (task15b_screen_shared_gamma); then the one locked test pass.
#          gpu-a   (started 15 min after gpu-b) Task 15 seeds 123/2024 of every baseline,
#                  then Task 15's table; then, once gpu-b's arms exist, run_post_evals.sh.
#
# Every notebook pass starts by pulling this repo and re-patching the shared
# learning-concepts checkout, so two passes must never START together (the second
# `git apply` fails on the already-patched tree): hence the 15-minute stagger.  For the
# same reason, push nothing to the repo while this runs.  The shell scripts it calls are
# copied to a snapshot first, so a pull can never rewrite a script that bash is reading.
#
# Run from the task15 base directory (holding concept_aware/ and outputs/), in tmux.
# Logs: this terminal (phase 1), family_<tag>_gpuA.log, family_<tag>_gpuB.log, and each
# pass's papermill log, verified_<tag>_<pass>.log.
set -euo pipefail
TAG=$1; MODEL=$2; GA=$3; GB=$4
if [ -z "${FAMILY_SNAPSHOT:-}" ]; then
  here=$(cd "$(dirname "$0")" && pwd)
  snap=$(mktemp -d "${TMPDIR:-/tmp}/family_${TAG}.XXXXXX")
  cp "$here/run_new_family.sh" "$here/run_verified_pipeline.sh" "$here/run_post_evals.sh" "$snap/"
  FAMILY_SNAPSHOT=$snap exec bash "$snap/run_new_family.sh" "$@"
fi
P=$FAMILY_SNAPSHOT
BASE=${CONCEPT_BASE:-$PWD}
cd "$BASE"
export CONCEPT_BASE=$BASE
M=outputs/$TAG/run_manifests
FAILED=family_${TAG}.failed
rm -f "$FAILED"
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

# Exit 0 when every named arm is in the manifest with its adapter on disk.
have_arms() {
  python3 - "$@" <<'PY'
import json, sys
from pathlib import Path
path, want = Path(sys.argv[1]), sys.argv[2:]
have = {k.replace(" ", "_"): v for k, v in json.loads(path.read_text()).items()} if path.is_file() else {}
missing = [a for a in want if a not in have or not (Path(have[a]) / "adapter_config.json").is_file()]
if missing:
    print(f"  {path.name}: still missing {len(missing)}: {' '.join(missing)}", file=sys.stderr)
sys.exit(1 if missing else 0)
PY
}
# Block until the other GPU's chain has trained them; give up if that chain failed.
wait_for() {
  until have_arms "$@" 2>/dev/null; do
    [ -e "$FAILED" ] && { log "the other GPU's chain failed; not waiting any longer"; return 1; }
    sleep 300
  done
}

log "phase 1: $TAG extraction and seed-42 baselines on GPU $GA"
STAGE=task15 RUN_NAME=first bash $P/run_verified_pipeline.sh "$TAG" "$MODEL" "$GA" --no-eval
have_arms $M/task15.json ntp_seed42 augmented_ntp_seed42 randomized_seed42 randomized_lambda1.0_seed42 zhang_seed42 \
  || { log "phase 1 finished without every seed-42 baseline; stopping"; exit 1; }

# The two chains are written with && on purpose: bash ignores `set -e` inside anything
# on the left of `||`, so a failed stage would otherwise run straight into the next one.
chain_b() {
  STAGE=hybrid RUN_NAME=arms bash $P/run_verified_pipeline.sh "$TAG" "$MODEL" "$GB" --no-screen \
      --hybrid-arms $GRID --selected-hybrid $LOCKED --multiseed --no-eval &&
  have_arms $M/task15b.json $ARMS &&
  log "15b arms trained; waiting for the baseline seeds before the dev table" &&
  wait_for $M/task15.json $BASELINES &&
  STAGE=hybrid RUN_NAME=sharedgamma bash $P/run_verified_pipeline.sh "$TAG" "$MODEL" "$GB" --no-screen \
      --hybrid-arms $GRID --selected-hybrid $LOCKED --multiseed --result-tag _shared_gamma \
      --eval-arms "$(echo $ARMS | tr ' ' ',')" &&
  STAGE=hybrid RUN_NAME=test bash $P/run_verified_pipeline.sh "$TAG" "$MODEL" "$GB" --no-screen \
      --selected-hybrid $LOCKED --result-tag _test --swords-test &&
  log "gpu-b chain done"
}
chain_a() {
  STAGE=confirm15 RUN_NAME=seeds bash $P/run_verified_pipeline.sh "$TAG" "$MODEL" "$GA" --multiseed &&
  have_arms $M/task15.json $BASELINES &&
  log "waiting for the 15b arms before the post-training evaluations" &&
  wait_for $M/task15b.json $ARMS &&
  bash $P/run_post_evals.sh "$TAG" "$MODEL" "$GA" &&
  log "gpu-a chain done"
}

log "phase 2: 15b arms on GPU $GB (family_${TAG}_gpuB.log); baseline seeds on GPU $GA in 15 min (family_${TAG}_gpuA.log)"
{ chain_b || { log "gpu-b chain FAILED"; touch "$FAILED"; }; } > family_${TAG}_gpuB.log 2>&1 &
B=$!
sleep 900
{ chain_a || { log "gpu-a chain FAILED"; touch "$FAILED"; }; } > family_${TAG}_gpuA.log 2>&1 &
A=$!
wait $A $B
if [ -e "$FAILED" ]; then
  log "FAILED: read the ends of family_${TAG}_gpuA.log and family_${TAG}_gpuB.log; rerun to resume"
  exit 1
fi
log "done: $TAG"
