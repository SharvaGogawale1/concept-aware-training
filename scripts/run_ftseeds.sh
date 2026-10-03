#!/usr/bin/env bash
# Fine-tune seed check: the classification-head (head-swap) fine-tune again with another
# fine-tune seed, so every family's table can report mean +- spread over three fine-tunes
# (seed 42 is the run already in seqcls_finetune*.json).  The seed changes which training
# examples are drawn, the head and task-LoRA initialisation and the order; the evaluation
# items stay the same, so differences remain paired item by item.
#
#   run_ftseeds.sh <gpu> <seed> [family ...]          (default: all three, Llama first)
#   run_ftseeds.sh 3 1
#   run_ftseeds.sh 2 2
#
# Systems: the seven of every fine-tune table, plus for Llama the four Iyer replication
# models that carry the findings (her NTP baseline, her released loss -- the dictionary
# one is the same model -- and both fixed losses).  Output, one file per family and seed:
# outputs/<family>/seqcls_finetune_ftseed<seed>.json.  Resumes.
set -euo pipefail
GPU=$1; SEED=$2; shift 2
FAMILIES=${*:-llama-3.2-1b qwen3-1.7b-base falcon3-1b-base}
BASE=${CONCEPT_BASE:-$PWD}
cd "$BASE"
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$BASE/concept_aware/hf_cache PYTHONUTF8=1 PYTHONIOENCODING=utf-8
PY=${CONCEPT_PY:-$HOME/miniconda3/envs/concept/bin/python}
S=concept_aware/concept-aware-training/scripts
[ "$SEED" != 42 ] || { echo "seed 42 is the original run; pick another"; exit 1; }
ARMS7="zhang_seed42 pretrained ntp_seed42 augmented_ntp_seed42 randomized_seed42 zhang_plus_uniform_g0.0625_seed42 zhang_plus_uniform_g0.125_seed42"
log() { echo "[$(date '+%F %T')] $*"; }

for t in $FAMILIES; do
  M=outputs/$t/run_manifests
  manifests="$M/task15.json $M/task15b.json"
  arms=$ARMS7
  case $t in
    llama-3.2-1b)    model=meta-llama/Llama-3.2-1B
                     manifests+=" $M/iyer_replica.json"
                     arms+=" iyer_ntp_syn iyer_nsp_loss_ctx iyer_nsp_loss_ctx_fixed iyer_nsp_loss_dict_fixed" ;;
    qwen3-1.7b-base) model=Qwen/Qwen3-1.7B-Base ;;
    falcon3-1b-base) model=tiiuae/Falcon3-1B-Base ;;
    *) echo "unknown family $t"; exit 1 ;;
  esac
  log "$t: head-swap fine-tune, seed $SEED, on GPU $GPU"
  # shellcheck disable=SC2086
  "$PY" $S/eval_seqcls_finetune.py --base-model "$model" --manifests $manifests --arms $arms \
    --seed "$SEED" --out outputs/$t/seqcls_finetune_ftseed$SEED.json
done
log "done: seed $SEED"
