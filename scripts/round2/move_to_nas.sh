#!/usr/bin/env bash
# Put the HF cache and the round-2 checkpoints on the NAS (the root disk is shared and full).
# Safe to run twice; deletes nothing: the local cache is renamed hf_cache.local for you to
# remove once you are satisfied.  The old cache path becomes a link, so every script that
# names concept_aware/hf_cache keeps working.  Refuses while any of our jobs is running.
#
#   bash scripts/round2/move_to_nas.sh
set -euo pipefail
BASE=${CONCEPT_BASE:-$HOME/fos_retrieval/task15}
NAS=${R2_NAS:-/nas/archive/tau-data/sharvag}
REPO=$BASE/concept_aware/concept-aware-training
OLD=$BASE/concept_aware/hf_cache
NEW=$NAS/huggingface
CK=$REPO/results/round2/checkpoints
CKNEW=$NAS/concept_aware/round2_checkpoints

busy=$(ps -u "$USER" -o pid=,args= | grep -E "train\.py|eval_|reproduce_gate|build_sealed_holdout|run_round2|concept_set_stats|papermill|lm_eval|embedding_synonyms" | grep -v grep || true)
if [ -n "$busy" ]; then echo "refusing: jobs that may read the cache are running:"; echo "$busy"; exit 1; fi
mkdir -p "$NEW" "$CKNEW" "$REPO/results/round2"

if [ -L "$OLD" ]; then
  [ "$(readlink -f "$OLD")" = "$(readlink -f "$NEW")" ] || { echo "refusing: $OLD links to $(readlink "$OLD"), not $NEW"; exit 1; }
  echo "hf_cache already links to $NEW"
else
  echo "copying $(du -sh "$OLD" | cut -f1) to $NEW ..."
  rsync -a "$OLD/" "$NEW/"
  diffs=$(rsync -a --checksum --dry-run --itemize-changes "$OLD/" "$NEW/" | wc -l)
  [ "$diffs" -eq 0 ] || { echo "the copy differs in $diffs entries; not switching"; exit 1; }
  mv "$OLD" "$OLD.local"
  ln -s "$NEW" "$OLD"
  echo "hf_cache -> $NEW   (old copy at $OLD.local; rm -rf it to free the space)"
fi
if [ ! -f "$NEW/token" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  cp "$HOME/.cache/huggingface/token" "$NEW/token" && echo "copied the Hugging Face token"
fi
[ -f "$NEW/token" ] && chmod 600 "$NEW/token"

if [ -L "$CK" ]; then
  target=$(readlink "$CK")
  if [ "$target" != "$CKNEW" ]; then
    if [ -z "$(ls -A "$target" 2>/dev/null)" ]; then ln -sfn "$CKNEW" "$CK"; echo "re-pointed checkpoints -> $CKNEW (was $target, empty)"
    else echo "refusing: $CK links to non-empty $target"; exit 1; fi
  else echo "checkpoints already link to $CKNEW"; fi
elif [ -e "$CK" ]; then
  echo "refusing: $CK exists and is not a link"; exit 1
else
  ln -s "$CKNEW" "$CK" && echo "checkpoints -> $CKNEW"
fi
df -h / "$NAS" | sed 1d
