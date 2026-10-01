#!/usr/bin/env bash
# Iyer et al. (2026) Table 3 at our scale: the synonym (NSP) family on her YouTube data,
# trained with her released code on Llama-3.2-1B, then scored with the same two downstream
# fine-tune protocols as our arms -- so the comparison is 1B against 1B on identical items.
#
# Her released concept loss (custom_trainer.py) is computed under torch.no_grad(): it
# changes the logged loss but adds zero gradient, so the "NSP Loss" models learn plain
# next-token prediction on the concept file.  Both are trained: as released, and fixed.
#
#   label                     script (her repo, pinned)       data (her data/syn/youtube)   Table 3 row
#   iyer_ntp_syn              run_clm.py                      vanilla_train.txt             NTP Synonym Baseline
#   iyer_nsp_aug_ctx          run_clm.py                      context_syn_train.txt         NSP Context-Aware Data Aug.
#   iyer_nsp_aug_dict         run_clm.py                      dict_train.txt                NSP Context-Free Data Aug.
#   iyer_nsp_loss_ctx         run_clm_syn_custom_loss.py      context_loss_train.csv        NSP Loss Context-Aware
#   iyer_nsp_loss_dict        run_clm_syn_custom_loss.py      dict_loss_train.csv           NSP Loss Context-Free
#   iyer_nsp_loss_ctx_fixed   ours: run_clm_differentiable_ncp.py --ncp_reduction mean --ncp_single_token_only
#   iyer_nsp_loss_dict_fixed  the same on dict_loss_train.csv
#
# Every run uses the training command in her README unchanged (block 128, bf16, gradient
# accumulation 4, automatic batch size, her scripts' own 3 epochs / cosine / lr 5e-5), with
# Llama-3.2-1B in place of Llama-3-8B.  The fixed arms keep her objective (the mean of
# log p over the single-token synonyms, weight 1) and her data and arguments, and change
# exactly what the July audit found broken: the concept term now has a gradient, it is read
# at the concept slot rather than a padding position, synonyms are scored as in-context
# continuations (" word", not the bare id), and padding is masked out of the loss.
#
#   run_iyer_replication.sh <gpu> [all|train|eval] [--smoke]
#
# Run from the task15 base directory after setup_laya_env.sh.  Training (~1 h) runs in the
# `laya` env; the downstream fine-tunes (~6 h) in `concept`, into seqcls_finetune_replica.json
# and iyer_finetune_replica.json, kept apart from the files our own runs write.  Both phases
# resume.  --smoke trains three arms for 2 steps and runs one downstream task, in a scratch dir.
set -euo pipefail
GPU=$1; PHASE=${2:-all}; SMOKE=${3:-}
BASE=${CONCEPT_BASE:-$PWD}
cd "$BASE"
MODEL=meta-llama/Llama-3.2-1B
TAG=llama-3.2-1b
MAIN=$BASE/concept_aware/concept-aware-training
LAYA=$BASE/concept_aware/laya-concept-aware-training
COMMIT=a955e7543c9ef79c20a29535b12d7163698c5711
LM=transformers/examples/pytorch/language-modeling
DATA=$LAYA/data/syn/youtube
LAYA_PY=${LAYA_PY:-$HOME/miniconda3/envs/laya/bin/python}
CONCEPT_PY=${CONCEPT_PY:-$HOME/miniconda3/envs/concept/bin/python}
OUT=$BASE/concept_aware/runs/$TAG/iyer_replica
MANIFEST=outputs/$TAG/run_manifests/iyer_replica.json
RESULTS=outputs/$TAG
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$BASE/concept_aware/hf_cache PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export WANDB_DISABLED=true
log() { echo "[$(date '+%F %T')] $*"; }

ARMS="iyer_ntp_syn iyer_nsp_aug_ctx iyer_nsp_aug_dict iyer_nsp_loss_ctx iyer_nsp_loss_dict iyer_nsp_loss_ctx_fixed iyer_nsp_loss_dict_fixed"
EXTRA=()
if [ "$SMOKE" = --smoke ]; then
  ARMS="iyer_ntp_syn iyer_nsp_loss_ctx iyer_nsp_loss_ctx_fixed"
  OUT=$BASE/concept_aware/runs/$TAG/iyer_replica_smoke
  MANIFEST=$OUT/manifest.json
  RESULTS=$OUT
  EXTRA=(--max_steps 2 --max_train_samples 64 --max_eval_samples 16)
fi

for py in "$LAYA_PY" "$CONCEPT_PY"; do
  [ -x "$py" ] || { log "missing interpreter $py: run setup_laya_env.sh first"; exit 1; }
done
[ "$(git -C "$LAYA" rev-parse HEAD)" = "$COMMIT" ] || { log "$LAYA is not at $COMMIT: run setup_laya_env.sh"; exit 1; }

# label -> (where to run, script, train file, validation file, extra arguments)
spec() {
  case $1 in
    iyer_ntp_syn)             echo "$LAYA/$LM run_clm.py vanilla_train.txt vanilla_val.txt" ;;
    iyer_nsp_aug_ctx)         echo "$LAYA/$LM run_clm.py context_syn_train.txt context_syn_val.txt" ;;
    iyer_nsp_aug_dict)        echo "$LAYA/$LM run_clm.py dict_train.txt dict_val.txt" ;;
    iyer_nsp_loss_ctx)        echo "$LAYA/$LM run_clm_syn_custom_loss.py context_loss_train.csv context_loss_val.csv" ;;
    iyer_nsp_loss_dict)       echo "$LAYA/$LM run_clm_syn_custom_loss.py dict_loss_train.csv dict_loss_val.csv" ;;
    iyer_nsp_loss_ctx_fixed)  echo "$MAIN/$LM run_clm_differentiable_ncp.py context_loss_train.csv context_loss_val.csv --ncp_alpha 1.0 --ncp_reduction mean --ncp_single_token_only True" ;;
    iyer_nsp_loss_dict_fixed) echo "$MAIN/$LM run_clm_differentiable_ncp.py dict_loss_train.csv dict_loss_val.csv --ncp_alpha 1.0 --ncp_reduction mean --ncp_single_token_only True" ;;
  esac
}

record() {   # add label -> model dir to the manifest
  python3 - "$MANIFEST" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
path, label, model = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
runs = json.loads(path.read_text()) if path.is_file() else {}
runs[label] = model
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(runs, indent=2))
PY
}

if [ "$PHASE" = all ] || [ "$PHASE" = train ]; then
  free_gb=$(df -BG --output=avail "$BASE" | tail -1 | tr -dc 0-9)
  [ "$free_gb" -ge 25 ] || [ -n "$SMOKE" ] || { log "only ${free_gb} GB free; seven 1B checkpoints need ~18 GB plus headroom"; exit 1; }
  mkdir -p "$OUT"
  for arm in $ARMS; do
    dir=$OUT/$arm
    if [ -e "$dir/.replica_done" ]; then log "trained already: $arm"; record "$arm" "$dir"; continue; fi
    rm -rf "$dir"                                  # a run that died part-way; her scripts refuse a non-empty dir
    read -r where script train val rest <<<"$(spec "$arm")"
    log "training $arm ($script on $train) on GPU $GPU"
    # shellcheck disable=SC2086
    (cd "$where" && "$LAYA_PY" "$script" \
      --model_name_or_path "$MODEL" \
      --train_file "$DATA/$train" --validation_file "$DATA/$val" \
      --save_total_limit 1 --gradient_accumulation_steps 4 \
      --torch_dtype bfloat16 --bf16 True --block_size 128 --auto_find_batch_size True \
      --do_train --do_eval --output_dir "$dir" ${rest:-} ${EXTRA[@]+"${EXTRA[@]}"}) 2>&1 | tee "$dir.log"
    rm -rf "$dir"/checkpoint-*                     # the final model is saved at the top of $dir
    [ -f "$dir/config.json" ] || { log "$arm left no model in $dir"; exit 1; }
    git -C "$where" rev-parse HEAD > "$dir/.replica_done"
    record "$arm" "$dir"
  done
  log "trained: $(python3 -c "import json; print(', '.join(json.load(open('$MANIFEST'))))")"
fi

if [ "$PHASE" = all ] || [ "$PHASE" = eval ]; then
  S=$MAIN/scripts
  # shellcheck disable=SC2086
  set -- $ARMS
  FT=()
  if [ -n "$SMOKE" ]; then FT=(--tasks rte --smoke); fi
  log "classification-head fine-tune on the replication arms"
  "$CONCEPT_PY" $S/eval_seqcls_finetune.py --base-model $MODEL --manifests "$MANIFEST" --arms "$@" \
    --out $RESULTS/seqcls_finetune_replica.json ${FT[@]+"${FT[@]}"}
  log "generated-answer fine-tune (Iyer's protocol) on the replication arms"
  "$CONCEPT_PY" $S/eval_iyer_finetune.py --base-model $MODEL --manifests "$MANIFEST" --arms "$@" \
    --out $RESULTS/iyer_finetune_replica.json ${FT[@]+"${FT[@]}"}
fi
if [ -n "$SMOKE" ]; then
  log "smoke passed; removing $OUT"
  rm -rf "$OUT"
fi
log "done"
