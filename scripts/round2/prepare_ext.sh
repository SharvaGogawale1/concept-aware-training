#!/usr/bin/env bash
# Re-patch the shared learning-concepts checkout with this repo's patch, and record which
# patch it carries (.round2_patch_sha256), which run_round2.py checks before every run.
# Refuses while any job that imports that checkout is running: the notebooks re-patch it
# at their start too, so two writers would leave it in an unknown state.
#
#   cd ~/fos_retrieval/task15 && bash concept_aware/concept-aware-training/scripts/round2/prepare_ext.sh
set -euo pipefail
BASE=${CONCEPT_BASE:-$PWD}
EXT=$BASE/concept_aware/learning-concepts
MAIN=$BASE/concept_aware/concept-aware-training
PIN=b1d414143d11c8ed988b4cccbb06626cc8272bbe
busy=$(ps -u "$USER" -o pid=,args= | grep -E "train\.py|eval_mteb|eval_perplexity_explicit|eval_concept_sets|papermill|embedding_synonyms" | grep -v grep || true)
if [ -n "$busy" ]; then
  echo "refusing: jobs that use $EXT are running:"; echo "$busy"; exit 1
fi
git -C "$EXT" reset --hard "$PIN"
git -C "$EXT" clean -fdq
git -C "$EXT" apply "$MAIN/external/learning-concepts.patch"
sha256sum "$MAIN/external/learning-concepts.patch" | cut -d' ' -f1 > "$EXT/.round2_patch_sha256"
echo "patched $EXT onto ${PIN:0:7} with $(cat "$EXT/.round2_patch_sha256")"
